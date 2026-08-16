# tests/tracing/test_phase4_observability.py
"""Phase 4:可观测性升级的验证。

两件事,对应两个真实问题:

① tracer 去全局单例
   要证明的不是"代码改了",是"改完之后隔离真的成立、且现有调用方
   一行不用改"。这两条必须同时成立,只有前者是破坏性变更,只有后者
   等于没改。

② Span 树查询端
   写入端(parent_trace_id)一直是完备的,读取端此前**完全不存在**——
   代码里没有任何一处查询过 parent_trace_id。所以这里的测试不是
   回归测试,是首次覆盖。最重要的一条是 total_tokens():它回答
   "一次用户请求到底花了多少钱",而这个数字只有把嵌套子 Agent
   (提取/召回/压缩)的开销一起算进来才是对的——那恰恰是没有树视图
   时最容易漏掉的部分。
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from harness.tracing import (
    Tracer,
    SQLiteTraceStorage,
    TraceStorageBase,
    current_tracer,
    default_tracer,
    use_tracer,
)
from harness.tracing import tracer as tracer_module
from harness.tracing.models import LLMCall, ToolEvent, Trace, TraceNode


class _RecordingStorage(TraceStorageBase):
    def __init__(self):
        self.saved: list[Trace] = []

    def save_trace(self, trace: Trace) -> None:
        self.saved.append(trace)

    def get_traces_by_session(self, session_id: str) -> list[dict]:
        return []


# ══ ① tracer 去全局单例 ═══════════════════════════════════════════════

def test_module_level_functions_still_work_unchanged():
    """向后兼容:改造前的调用方式(直接调模块级函数,不碰任何新 API)
    必须原样工作。这条如果挂了,说明这次改造是破坏性的——
    tool_executor.py / openai_client.py 用的就是这种调用方式,
    而它们一行都没改。"""
    storage = _RecordingStorage()
    tracer_module.configure_storage(storage)
    tracer_module.start_trace("t1", "sess", "问题")
    tracer_module.record_tool_event("t1", "search", {"q": "x"}, "结果", 12)
    tracer_module.record_llm_call("t1", 10, 5, "输出", False, 30)
    tracer_module.end_trace("t1", "答案")

    assert len(storage.saved) == 1
    saved = storage.saved[0]
    assert saved.trace_id == "t1"
    assert saved.tool_call_count == 1
    assert saved.llm_call_count == 1
    assert saved.final_reply == "答案"


def test_use_tracer_scopes_to_a_separate_instance():
    outer_storage = _RecordingStorage()
    tracer_module.configure_storage(outer_storage)

    inner = Tracer(storage=_RecordingStorage())
    with use_tracer(inner):
        tracer_module.start_trace("inner-1", "s", "内层")
        tracer_module.end_trace("inner-1", "内层答案")
        assert current_tracer() is inner

    tracer_module.start_trace("outer-1", "s", "外层")
    tracer_module.end_trace("outer-1", "外层答案")

    assert [t.trace_id for t in inner.storage.saved] == ["inner-1"]
    assert [t.trace_id for t in outer_storage.saved] == ["outer-1"]


def test_use_tracer_restores_previous_on_nested_exit():
    """嵌套作用域退出时要精确还原**上一层**,不是简单设回 None——
    用 ContextVar.reset(token) 而不是 set(None) 的原因就在这里。

    注意这里断言的是"回到进入前的那个",而不是"回到 default_tracer"：
    conftest 的 autouse fixture 已经把每个测试包在一层 use_tracer 里了,
    所以最外层本来就不是 default。写成"回到进入前的那个"既准确,
    也不依赖调用方处在哪一层——这正是 reset(token) 相对 set(None)
    的价值所在。
    """
    before = current_tracer()
    a, b = Tracer(), Tracer()
    with use_tracer(a):
        assert current_tracer() is a
        with use_tracer(b):
            assert current_tracer() is b
        assert current_tracer() is a   # 回到 a,不是 None
    assert current_tracer() is before


async def test_concurrent_tasks_get_isolated_tracers():
    """这是"去全局"最核心的一条:asyncio.gather 并发跑两段逻辑,
    各自的 trace 不会跑到对方的存储里。

    改造前做不到 —— 模块级 _storage 全进程只有一个,两个并发任务
    的 trace 必然落进同一个存储,想按租户/环境分流是不可能的。
    """
    sa, sb = _RecordingStorage(), _RecordingStorage()

    async def run(name: str, storage: _RecordingStorage, n: int):
        with use_tracer(Tracer(storage=storage)):
            for i in range(n):
                tracer_module.start_trace(f"{name}-{i}", "s", "任务")
                await asyncio.sleep(0)      # 强制交错调度
                tracer_module.end_trace(f"{name}-{i}", "done")

    await asyncio.gather(run("A", sa, 3), run("B", sb, 3))

    assert sorted(t.trace_id for t in sa.saved) == ["A-0", "A-1", "A-2"]
    assert sorted(t.trace_id for t in sb.saved) == ["B-0", "B-1", "B-2"]


def test_active_count_makes_leaks_visible():
    """改造理由之二:未结束的 trace 会永久留在内存里。改造前这个
    字典是私有的,泄漏不可见;现在能被监控到。"""
    t = Tracer()
    assert t.active_count == 0
    t.start_trace("a", "s", "x")
    t.start_trace("b", "s", "x")
    assert t.active_count == 2
    t.end_trace("a", "done")
    assert t.active_count == 1
    assert t.active_trace_ids() == ["b"]


def test_discard_trace_removes_without_persisting():
    """discard 和 end 的区别是刻意的:end 表示"这次运行结束了,记下来",
    discard 表示"这条不作数,别占内存"。用 end(status='abandoned')
    冒充丢弃会往审计库里写一条并不代表真实运行的记录。"""
    storage = _RecordingStorage()
    t = Tracer(storage=storage)
    t.start_trace("x", "s", "问题")

    assert t.discard_trace("x") is True
    assert t.active_count == 0
    assert storage.saved == []        # 关键:没有落盘
    assert t.discard_trace("x") is False   # 幂等,第二次没东西可丢


def test_events_for_unknown_trace_are_silently_ignored():
    """给一条不存在的 trace 记事件不该抛异常——tracing 是旁路观测,
    它自己出问题不该拖垮主流程(和 memory 子系统的 on_memory_event
    回调包 try/except 是同一条原则)。"""
    t = Tracer()
    t.record_tool_event("nonexistent", "tool", {}, "r", 1)
    t.record_llm_call("nonexistent", 1, 1, "o", False, 1)
    t.end_trace("nonexistent", "reply")   # 也不该抛


# ══ ② Span 树查询端 ═══════════════════════════════════════════════════

def _make_trace(trace_id: str, parent: str | None = None,
                prompt: int = 0, completion: int = 0,
                tool_names: tuple[str, ...] = (),
                offset_sec: int = 0) -> Trace:
    base = datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=offset_sec)
    t = Trace(
        trace_id=trace_id, session_id="sess", user_input=f"输入-{trace_id}",
        final_reply=f"回复-{trace_id}", total_duration_ms=100,
        tool_call_count=len(tool_names), llm_call_count=1 if prompt else 0,
        status="success", parent_trace_id=parent, created_at=base,
    )
    if prompt or completion:
        t.llm_calls.append(LLMCall(
            event_id=f"llm-{trace_id}", trace_id=trace_id,
            prompt_tokens=prompt, completion_tokens=completion,
            token_source="api_usage", output="out", has_tool_calls=bool(tool_names),
            duration_ms=50, timestamp=base,
        ))
    for i, name in enumerate(tool_names):
        t.tool_events.append(ToolEvent(
            event_id=f"tool-{trace_id}-{i}", trace_id=trace_id, tool_name=name,
            args="{}", result="ok", status="success", duration_ms=10, timestamp=base,
        ))
    return t


@pytest.fixture
def storage(tmp_path) -> SQLiteTraceStorage:
    return SQLiteTraceStorage(tmp_path / "trace.db")


def test_indexes_are_created(storage):
    """SQLite 不给 FOREIGN KEY 自动建索引(实测确认过)。没有这些索引,
    树查询的每一步都是全表扫描,代价随整库大小线性增长——trace 攒得
    越多、查审计越慢,和"长期运行的生产库"这个真实场景正好相反。"""
    conn = sqlite3.connect(storage.db_path)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    )}
    conn.close()
    assert {"idx_traces_parent", "idx_traces_session",
            "idx_tool_events_trace", "idx_llm_calls_trace"} <= names


def test_parent_lookup_uses_index_not_full_scan(storage):
    """不满足于"索引建出来了",直接看查询计划确认它真的被用上了。
    索引建了但查询用不上(比如类型不匹配)是常见的假安全感。"""
    conn = sqlite3.connect(storage.db_path)
    plan = " ".join(str(r[-1]) for r in conn.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM traces WHERE parent_trace_id=?", ("x",)
    ))
    conn.close()
    assert "USING INDEX idx_traces_parent" in plan
    assert "SCAN traces" not in plan


def test_get_trace_reconstructs_full_object_with_events(storage):
    storage.save_trace(_make_trace("t1", prompt=100, completion=20,
                                   tool_names=("search", "finish")))

    got = storage.get_trace("t1")

    assert got is not None
    assert got.trace_id == "t1"
    assert got.status == "success"
    assert isinstance(got.created_at, datetime)      # ISO 字符串要解析回来
    assert [e.tool_name for e in got.tool_events] == ["search", "finish"]
    assert got.llm_calls[0].prompt_tokens == 100
    assert got.llm_calls[0].token_source == "api_usage"


def test_get_trace_returns_none_for_missing(storage):
    assert storage.get_trace("does-not-exist") is None


def test_get_children_returns_only_direct_children(storage):
    storage.save_trace(_make_trace("root"))
    storage.save_trace(_make_trace("child-a", parent="root", offset_sec=1))
    storage.save_trace(_make_trace("child-b", parent="root", offset_sec=2))
    storage.save_trace(_make_trace("grandchild", parent="child-a", offset_sec=3))

    children = storage.get_children("root")

    assert [c.trace_id for c in children] == ["child-a", "child-b"]   # 不含孙子


def test_get_trace_tree_builds_correct_structure(storage):
    """典型形状:一次主 run 下面挂着提取子 Agent,提取子 Agent 下面
    还可能有自己的嵌套。"""
    storage.save_trace(_make_trace("main"))
    storage.save_trace(_make_trace("extract", parent="main", offset_sec=1))
    storage.save_trace(_make_trace("recall", parent="main", offset_sec=2))
    storage.save_trace(_make_trace("extract-inner", parent="extract", offset_sec=3))

    tree = storage.get_trace_tree("main")

    assert tree is not None
    assert tree.trace.trace_id == "main"
    assert [c.trace.trace_id for c in tree.children] == ["extract", "recall"]
    assert [c.trace.trace_id for c in tree.children[0].children] == ["extract-inner"]
    assert tree.depth() == 3
    assert {t.trace_id for t in tree.flatten()} == {
        "main", "extract", "recall", "extract-inner"}


def test_get_trace_tree_returns_none_for_missing_root(storage):
    assert storage.get_trace_tree("nope") is None


def test_get_trace_tree_on_leaf_returns_single_node(storage):
    storage.save_trace(_make_trace("solo"))
    tree = storage.get_trace_tree("solo")
    assert tree.children == []
    assert tree.depth() == 1


def test_subtree_query_from_middle_node(storage):
    """从中间节点查,应该只拿到它那一支,不该把整棵树都捞上来
    (root 自己的 parent 指向树外,不能因此被当成孤儿丢掉)。"""
    storage.save_trace(_make_trace("main"))
    storage.save_trace(_make_trace("extract", parent="main", offset_sec=1))
    storage.save_trace(_make_trace("extract-inner", parent="extract", offset_sec=2))
    storage.save_trace(_make_trace("recall", parent="main", offset_sec=3))

    tree = storage.get_trace_tree("extract")

    assert tree.trace.trace_id == "extract"
    assert [c.trace.trace_id for c in tree.children] == ["extract-inner"]
    assert "recall" not in {t.trace_id for t in tree.flatten()}


# ── 审计场景:整棵树的成本聚合(这是树查询存在的主要理由) ─────────────────

def test_total_tokens_aggregates_across_whole_tree(storage):
    """"这次用户请求到底花了多少钱"——必须是整棵树的合计。

    只看顶层 trace 会漏掉提取子 Agent、召回选择器、压缩摘要器这些
    嵌套调用的开销,而那些恰恰是最容易被低估的部分:它们不出现在
    用户可见的对话里,但真金白银地烧掉了 token。
    """
    storage.save_trace(_make_trace("main", prompt=1000, completion=200))
    storage.save_trace(_make_trace("extract", parent="main",
                                   prompt=500, completion=100, offset_sec=1))
    storage.save_trace(_make_trace("recall", parent="main",
                                   prompt=300, completion=50, offset_sec=2))

    tree = storage.get_trace_tree("main")
    prompt, completion = tree.total_tokens()

    assert prompt == 1800        # 1000 + 500 + 300
    assert completion == 350     # 200 + 100 + 50
    # 反向确认:只看顶层会漏掉 800 prompt tokens —— 这正是问题所在
    assert tree.trace.llm_calls[0].prompt_tokens == 1000


def test_all_tool_events_collects_across_tree(storage):
    """审计主力查询:"这次运行到底动了哪些工具"。付款类操作的追溯
    靠的就是这个——权限门控拦住了什么、批准之后执行了什么。"""
    storage.save_trace(_make_trace("main", tool_names=("transfer",)))
    storage.save_trace(_make_trace("extract", parent="main",
                                   tool_names=("memory_write",), offset_sec=1))

    tree = storage.get_trace_tree("main")
    names = [e.tool_name for e in tree.all_tool_events()]

    assert names == ["transfer", "memory_write"]


def test_find_locates_node_anywhere_in_tree(storage):
    storage.save_trace(_make_trace("main"))
    storage.save_trace(_make_trace("deep", parent="main", offset_sec=1))
    storage.save_trace(_make_trace("deeper", parent="deep", offset_sec=2))

    tree = storage.get_trace_tree("main")
    node = tree.find("deeper")

    assert node is not None
    assert node.trace.trace_id == "deeper"
    assert tree.find("nonexistent") is None


# ── 不信任磁盘数据:环路保护 ───────────────────────────────────────────────

def test_cycle_in_parent_links_does_not_hang(storage):
    """parent_trace_id 是从磁盘读回来的,而磁盘上的东西可以被手工
    编辑、被别的进程写坏。一旦形成环,朴素递归会无限循环把进程挂死。

    这和 snapshot/models.py 那条"版本不匹配就显式报错、不半解析"是
    同一条纪律:不信任从磁盘读回来的结构。这里直接手写一个环进 DB,
    确认查询能正常终止。
    """
    storage.save_trace(_make_trace("a"))
    storage.save_trace(_make_trace("b", parent="a", offset_sec=1))
    # 手工制造环:a 的 parent 指回 b
    conn = sqlite3.connect(storage.db_path)
    conn.execute("UPDATE traces SET parent_trace_id='b' WHERE trace_id='a'")
    conn.commit()
    conn.close()

    tree = storage.get_trace_tree("a")   # 不该挂死

    assert tree is not None
    ids = [t.trace_id for t in tree.flatten()]
    assert len(ids) == len(set(ids))     # 每个节点只出现一次


# ── 基类通用实现:只实现两个原语的后端也能拿到树 ─────────────────────────

class _MinimalBackend(TraceStorageBase):
    """只实现 get_trace / get_children 两个最简单的查询,
    get_trace_tree 应该由基类免费提供。"""

    def __init__(self, traces: list[Trace]):
        self._by_id = {t.trace_id: t for t in traces}

    def save_trace(self, trace: Trace) -> None:
        self._by_id[trace.trace_id] = trace

    def get_traces_by_session(self, session_id: str) -> list[dict]:
        return []

    def get_trace(self, trace_id: str) -> Trace | None:
        return self._by_id.get(trace_id)

    def get_children(self, parent_trace_id: str) -> list[Trace]:
        return [t for t in self._by_id.values() if t.parent_trace_id == parent_trace_id]


def test_generic_tree_implementation_works_for_minimal_backend():
    backend = _MinimalBackend([
        _make_trace("root", prompt=10, completion=1),
        _make_trace("kid", parent="root", prompt=20, completion=2),
        _make_trace("grandkid", parent="kid", prompt=30, completion=3),
    ])

    tree = backend.get_trace_tree("root")

    assert tree.depth() == 3
    assert tree.total_tokens() == (60, 6)


def test_generic_tree_implementation_guards_against_cycles():
    a = _make_trace("a", parent="b")
    b = _make_trace("b", parent="a")
    backend = _MinimalBackend([a, b])

    tree = backend.get_trace_tree("a")   # 不该挂死

    ids = [t.trace_id for t in tree.flatten()]
    assert len(ids) == len(set(ids))


def test_backend_without_read_primitives_fails_loudly_not_silently():
    """没实现读取端的后端:构造照常成功、保存照常工作,只有真的去调
    树查询时才报错,而且报错信息要说清楚缺什么、怎么补。
    这是"新能力可选、不配置即无感"在存储层的落地。"""
    backend = _RecordingStorage()
    backend.save_trace(_make_trace("x"))      # 老能力照常

    with pytest.raises(NotImplementedError, match="get_trace"):
        backend.get_trace_tree("x")


# ══ ③ analytics 统计口径修正(树视图照出来的真实 bug) ═══════════════════

def test_avg_tool_calls_not_diluted_by_subagent_traces(storage):
    """真实场景:1 次用户请求,顶层用了 6 个工具,下面挂 3 个各用 1 个
    工具的子 Agent(提取/召回/压缩)。

    修复前:全表扁平平均 = (6+1+1+1)/4 = 2.25 —— 它把子 Agent 的
    trace 当成了独立的用户请求。"平均每次请求用几个工具"的正确答案
    不可能是 2.25。
    """
    from harness.tracing import analytics

    storage.save_trace(_make_trace("main", tool_names=("a", "b", "c", "d", "e", "f")))
    for i, name in enumerate(("extract", "recall", "compact")):
        storage.save_trace(_make_trace(name, parent="main",
                                       tool_names=("x",), offset_sec=i + 1))

    assert analytics.avg_tool_calls(storage) == 6.0
    assert analytics.avg_tool_calls(storage, top_level_only=False) == 2.25  # 旧口径


def test_slowest_traces_excludes_subagent_traces_by_default(storage):
    """子 trace 混进"最慢请求"榜单会让人误以为"某次提取"是一次独立的
    慢请求,而它其实只是某次慢请求的一部分。"""
    from harness.tracing import analytics

    storage.save_trace(_make_trace("main"))
    storage.save_trace(_make_trace("extract", parent="main", offset_sec=1))

    ids = [r["trace_id"] for r in analytics.slowest_traces(storage)]
    assert ids == ["main"]

    ids_all = [r["trace_id"] for r in analytics.slowest_traces(storage, top_level_only=False)]
    assert set(ids_all) == {"main", "extract"}


def test_request_cost_rolls_up_nested_agent_spend(storage):
    """一次请求的**真实**成本必须含嵌套子 Agent —— 只看顶层会系统性
    低估。这是没有树查询之前根本回答不了的问题。"""
    from harness.tracing import analytics

    storage.save_trace(_make_trace("main", prompt=1000, completion=100,
                                   tool_names=("transfer",)))
    storage.save_trace(_make_trace("extract", parent="main",
                                   prompt=500, completion=50,
                                   tool_names=("memory_write",), offset_sec=1))

    cost = analytics.request_cost(storage, "main")

    assert cost["prompt_tokens"] == 1500        # 只看顶层会以为是 1000
    assert cost["completion_tokens"] == 150
    assert cost["total_tokens"] == 1650
    assert cost["trace_count"] == 2
    assert cost["tool_call_count"] == 2          # transfer + memory_write
    assert cost["depth"] == 2


def test_request_cost_returns_empty_for_unknown_trace(storage):
    from harness.tracing import analytics
    assert analytics.request_cost(storage, "nope") == {}


def test_session_cost_counts_requests_not_traces(storage):
    """关键区别:一个会话里跑了 2 次请求、产生了 4 条 trace ——
    request_count 必须是 2,不是 4。直接 COUNT(traces) 就会答成 4,
    这正是"把子 trace 当独立请求"的同一个错误。"""
    from harness.tracing import analytics

    storage.save_trace(_make_trace("req1", prompt=100, completion=10))
    storage.save_trace(_make_trace("req1-sub", parent="req1",
                                   prompt=50, completion=5, offset_sec=1))
    storage.save_trace(_make_trace("req2", prompt=200, completion=20, offset_sec=2))
    storage.save_trace(_make_trace("req2-sub", parent="req2",
                                   prompt=80, completion=8, offset_sec=3))

    cost = analytics.session_cost(storage, "sess")

    assert cost["request_count"] == 2            # 不是 4
    assert cost["prompt_tokens"] == 430          # 100+50+200+80
    assert cost["completion_tokens"] == 43
    assert [r["trace_id"] for r in cost["requests"]] == ["req1", "req2"]
