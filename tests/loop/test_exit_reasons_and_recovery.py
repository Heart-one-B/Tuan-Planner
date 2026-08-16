# tests/agent/test_exit_reasons_and_recovery.py
"""对应待做计划表第三刀任务 3.1-3.12 的核心验证。

这份测试是第三刀里"真正验证行为、不是只验证结构"的部分——第三刀
不是纯重构,是真实的语义修正(bug#1)+ 新机制(静默升档)+ 新钩子
(on_error_snapshot),每一条都需要证明"确实按设计工作",不能只看
代码写得像不像。
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness.agent.agent import Agent
from harness.agent.finish_tool import build_finish_tool
from harness.agent.loop import Budget, wrap_store
from harness.agent.permission import Allow, AllowAllPolicy, Defer
from harness.agent.query import run_to_outcome
from harness.agent.run_context import RunContext
from harness.agent.state import LoopConfig, LoopState, ResumePoint
from harness.llm.base import ContextOverflowError
from harness.snapshot.store import FileSnapshotStore
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor

from tests.agent.fakes import FakeLLMClient, RaisingLLMClient


def _run_ctx() -> RunContext:
    return RunContext.begin("test-agent", "测试任务")


async def _search(query: str) -> str:
    return f"搜索结果:{query}"


def _make_cfg(llm, executor, **overrides) -> LoopConfig:
    defaults = dict(
        llm=llm, tool_executor=executor, budget=Budget(),
        permission_policy=AllowAllPolicy(), tools=executor.schemas,
    )
    defaults.update(overrides)
    return LoopConfig(**defaults)


def _make_state(messages, run_ctx, **overrides) -> LoopState:
    return LoopState(store=wrap_store(messages, run_ctx), **overrides)


class _DummyData(BaseModel):
    answer: str


def _executor_with_search() -> ToolExecutor:
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="search", func=_search, description="", parameters={}))
    return executor


def _executor_with_search_and_finish() -> ToolExecutor:
    executor = _executor_with_search()
    executor.register(build_finish_tool(_DummyData))
    return executor


# ── 任务 3.1/3.2:reason 全路径覆盖 ───────────────────────────────────────

async def test_reason_completed_for_plain_cc_completion():
    llm = FakeLLMClient([{"content": "答案", "tool_calls": []}])
    cfg = _make_cfg(llm, ToolExecutor())
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.reason == "completed"


async def test_reason_terminal_tool_not_called_when_nudges_exhausted():
    llm = FakeLLMClient([
        {"content": "回答1", "tool_calls": []},
        {"content": "回答2(最终)", "tool_calls": []},
    ])
    executor = _executor_with_search_and_finish()
    cfg = _make_cfg(llm, executor, require_terminal_tool="finish", max_terminal_nudges=1)
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.reason == "terminal_tool_not_called"


async def test_reason_completed_when_terminal_tool_succeeds():
    """终止工具成功走的也是 status=completed,reason=completed——
    和纯文本 CC 完成是同一个粗信号,细信号也一致,不需要区分
    '怎么完成的',那是 result is not None 已经能看出来的事。"""
    llm = FakeLLMClient([{
        "content": None,
        "tool_calls": [{"id": "c1", "name": "finish",
                        "arguments": {"status": "ok", "summary": "done", "data": {"answer": "1"}}}],
    }])
    executor = _executor_with_search_and_finish()
    cfg = _make_cfg(llm, executor)
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.reason == "completed"


async def test_reason_max_rounds_vs_max_tool_calls_are_distinguishable():
    """任务 3.3 的核心断言:同样是 status=exhausted,原因必须能分清
    是撞了轮次上限还是工具调用上限——这正是 bug#2 要的能力。"""
    llm_rounds = FakeLLMClient([{"content": None,
                                 "tool_calls": [{"id": "c1", "name": "search", "arguments": {"query": "q"}}]}])
    cfg_rounds = _make_cfg(llm_rounds, _executor_with_search(),
                           budget=Budget(max_rounds=1, exhausted_action="stop"))
    run_ctx1 = _run_ctx()
    state1 = _make_state([{"role": "user", "content": "问题"}], run_ctx1)
    outcome1 = await run_to_outcome(cfg_rounds, state1, run_ctx1)
    assert outcome1.status == "exhausted"
    assert outcome1.reason == "max_rounds"

    llm_tools = FakeLLMClient([{"content": None,
                                "tool_calls": [{"id": "c1", "name": "search", "arguments": {"query": "q"}}]}])
    cfg_tools = _make_cfg(llm_tools, _executor_with_search(),
                          budget=Budget(max_tool_calls=1, exhausted_action="stop"))
    run_ctx2 = _run_ctx()
    state2 = _make_state([{"role": "user", "content": "问题"}], run_ctx2)
    outcome2 = await run_to_outcome(cfg_tools, state2, run_ctx2)
    assert outcome2.status == "exhausted"
    assert outcome2.reason == "max_tool_calls"

    assert outcome1.reason != outcome2.reason


async def test_reason_awaiting_approval():
    class _DeferPolicy:
        async def check(self, tool_name, args, run_ctx):
            return Defer(approval_id="a1")

    llm = FakeLLMClient([{"content": None,
                          "tool_calls": [{"id": "c1", "name": "search", "arguments": {"query": "q"}}]}])
    cfg = _make_cfg(llm, _executor_with_search(), permission_policy=_DeferPolicy())
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "awaiting_approval"
    assert outcome.reason == "awaiting_approval"


# ── 任务 3.4:压缩语义修正,bug#1 的核心验证 ────────────────────────────────

class _AlwaysOverflowStore:
    """每次调 maybe_compact 都"压缩成功"(不检验压缩内容本身,只关心
    压缩恢复的次数控制逻辑),配合总是抛 ContextOverflowError 的 llm,
    制造"反复溢出、反复压缩"的场景。"""
    def __init__(self, messages):
        self._messages = messages
        self.compact_calls = 0

    @property
    def messages(self):
        return self._messages

    def append(self, message):
        self._messages.append(message)

    def note_api_usage(self, prompt_tokens):
        pass

    def offload_tool_result(self, trace_id, tool_call_id, content, max_chars=None):
        return content

    async def maybe_compact(self, llm_client, trace_id, trigger="threshold", focus=None):
        self.compact_calls += 1
        from types import SimpleNamespace
        return SimpleNamespace(
            user_notice=None, degraded=False,
            result=SimpleNamespace(tokens_before=1000, tokens_after=100),
        )


class _AlwaysOverflowLLM(FakeLLMClient):
    """每次 call() 都抛 ContextOverflowError——用来把
    run_query_loop 的溢出恢复计数器逼到上限。"""
    def __init__(self):
        super().__init__([])

    async def call(self, trace_id, stream=False, **kwargs):
        self.calls.append(kwargs)
        raise ContextOverflowError("boom")


async def test_overflow_recovery_capped_by_budget_not_by_single_attempt():
    """修 bug#1 的直接证据:上限是 Budget.max_overflow_recoveries(可以
    大于 1),不是第一/二刀那种"整个 run 只给一次"的硬编码。这里配置
    上限为 3,断言 maybe_compact 被调用了恰好 3 次(第 4 次溢出时才
    真正放弃),而不是第 2 次溢出就死。"""
    run_ctx = _run_ctx()
    llm = _AlwaysOverflowLLM()
    store = _AlwaysOverflowStore([{"role": "user", "content": "问题"}])
    cfg = _make_cfg(llm, ToolExecutor(), budget=Budget(max_overflow_recoveries=3))
    state = LoopState(store=store)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "overflow"
    assert outcome.reason == "context_overflow"
    assert store.compact_calls == 3
    assert outcome.overflow_recovery_count == 3


async def test_overflow_recovery_count_survives_resume_boundary():
    """这是 bug#1 真正被修复的证据:构造一个"已经在恢复前的 run 里
    压缩过 2 次"的现场,通过 resume_point 恢复后再次溢出,应该还能
    再压 1 次(累计到 3、等于上限)才放弃——不会因为跨越了 resume
    边界就把计数器重置回 0。第一/二刀的实现在这里会立刻死掉
    (第二刀虽然搬进了 State,但 Agent.resume_events() 从来没有把
    这个计数器从快照里读回来,等价于每次 resume 都重置)。"""
    run_ctx = _run_ctx()
    llm = _AlwaysOverflowLLM()
    store = _AlwaysOverflowStore([{"role": "user", "content": "问题"}])
    cfg = _make_cfg(llm, ToolExecutor(), budget=Budget(max_overflow_recoveries=3))
    # 模拟"resume 之前已经发生过 2 次紧急压缩"
    state = LoopState(store=store, overflow_recovery_count=2)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "overflow"
    assert store.compact_calls == 1          # 只需要再压 1 次就到上限 3
    assert outcome.overflow_recovery_count == 3


async def test_compacted_this_round_resets_every_round_not_once_per_run():
    """每轮开头重置(CC 的 hasAttemptedReactiveCompact 语义)是防同轮
    死循环用的,不应该影响'能不能压第二次'这件事——能不能压第二次
    完全由 overflow_recovery_count 累计值 vs Budget 上限决定。这条
    测试确认:即便同一个 run 里已经压过(compacted_this_round 曾经
    True 过),只要还没到累计上限,新的一轮仍然可以再压。"""
    run_ctx = _run_ctx()
    llm = FakeLLMClient([
        {"content": "先正常说句话", "tool_calls": []},
    ])
    # 用普通 FakeLLMClient(不是永远溢出的那个),只验证
    # compacted_this_round 在第一轮触发过压缩后,在第二轮开头被清零。
    executor = ToolExecutor()
    cfg = _make_cfg(llm, executor)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)
    state.rounds = 1
    state.compacted_this_round = True   # 模拟"上一轮刚压过"

    # 直接调 run_query_loop 走一轮,验证 compacted_this_round 在轮次
    # 递增时被清零(通过侧面验证:如果没有被清零,不会影响这条纯
    # 文本完成的路径,所以改为直接断言字段本身)。
    from harness.agent.loop import run_query_loop
    async for ev in run_query_loop(cfg, state, run_ctx):
        pass
    assert state.compacted_this_round is False


# ── 任务 3.6/3.7:输出截断——上限、reason、丢弃末位不完整 tool_call ────────

async def test_truncation_exceeds_cap_exits_with_dedicated_reason():
    llm = FakeLLMClient([
        {"content": "第1段", "tool_calls": [], "finish_reason": "length"},
        {"content": "第2段", "tool_calls": [], "finish_reason": "length"},
        {"content": "第3段(最终仍截断)", "tool_calls": [], "finish_reason": "length"},
    ])
    cfg = _make_cfg(llm, ToolExecutor(),
                    budget=Budget(max_output_truncation_recoveries=2))
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "exhausted"
    assert outcome.reason == "max_output_tokens_recovery"
    assert outcome.output_truncation_count == 2
    assert llm.call_count == 3  # 前两次催续写,第三次仍截断就放弃


async def test_truncation_recovers_within_cap():
    llm = FakeLLMClient([
        {"content": "被截断的部分", "tool_calls": [], "finish_reason": "length"},
        {"content": "续写完成", "tool_calls": [], "finish_reason": "stop"},
    ])
    cfg = _make_cfg(llm, ToolExecutor(),
                    budget=Budget(max_output_truncation_recoveries=3))
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.reason == "completed"
    assert outcome.output_truncation_count == 1
    assert outcome.final_text == "续写完成"


async def test_truncation_with_tool_calls_drops_only_last_incomplete_call():
    """任务 3.7 的核心场景:同一响应里有两个 tool_calls,被截断——
    第一个(search)是完整的,应该正常执行;最后一个(finish,必定
    不完整)应该被丢弃、不进消息历史、不留孤儿 tool_use。"""
    llm = FakeLLMClient([
        {
            "content": None,
            "tool_calls": [
                {"id": "c1", "name": "search", "arguments": {"query": "q"}},
                {"id": "c2", "name": "finish", "arguments": '{"status":"ok","sum'},  # 故意不完整
            ],
            "finish_reason": "length",
        },
        {"content": "重新说完整", "tool_calls": []},
    ])
    executor = _executor_with_search_and_finish()
    cfg = _make_cfg(llm, executor)
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.result is None   # finish 没有真正执行(被丢弃了)

    # 消息历史里:search 的完整批处理应该发生过(tool 消息里有搜索结果)
    tool_msgs = [
        m for m in outcome.messages
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "tool"
    ]
    assert any("搜索结果" in (m.get("content") if isinstance(m, dict) else "") for m in tool_msgs)

    # 不该有任何 tool_call_id == "c2" 的 tool 消息(它从未被执行),
    # 也不该有任何 assistant 消息声称调用过 c2(孤儿 tool_use 检查)
    assert not any(
        (m.get("tool_call_id") if isinstance(m, dict) else None) == "c2"
        for m in outcome.messages
    )
    for m in outcome.messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "assistant":
            continue
        tool_calls = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        for tc in (tool_calls or []):
            tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
            assert tc_id != "c2", "被丢弃的不完整 tool_call 不该出现在任何 assistant 消息里"


# ── 任务 3.8:输出截断静默升档 ─────────────────────────────────────────────

async def test_silent_upgrade_retries_with_larger_max_tokens_and_does_not_count():
    """第一次截断且配置了升档值:应该内部静默重试一次,不计入
    output_truncation_count、不产生 nudge 消息。"""
    llm = FakeLLMClient([
        {"content": "被截断", "tool_calls": [], "finish_reason": "length"},
        {"content": "升档后完整了", "tool_calls": [], "finish_reason": "stop"},
    ])
    cfg = _make_cfg(llm, ToolExecutor(),
                    budget=Budget(max_output_tokens=100, max_output_tokens_upgraded=1000))
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.final_text == "升档后完整了"
    assert outcome.output_truncation_count == 0   # 静默升档不计入催续写次数
    assert outcome.output_upgraded is True
    assert llm.call_count == 2
    assert llm.calls[0]["max_tokens"] == 100
    assert llm.calls[1]["max_tokens"] == 1000       # 第二次确实用了升档值


async def test_silent_upgrade_still_truncated_falls_back_to_nudge_path():
    """升档重试之后仍然截断:降级为正常的 Nudge 恢复路径(计入
    output_truncation_count),不会无限重试升档(升档只给一次)。"""
    llm = FakeLLMClient([
        {"content": "被截断", "tool_calls": [], "finish_reason": "length"},   # 第一次(未升档)
        {"content": "升档后仍截断", "tool_calls": [], "finish_reason": "length"},  # 升档重试
        {"content": "第三次续写完成", "tool_calls": [], "finish_reason": "stop"},  # nudge 之后
    ])
    cfg = _make_cfg(llm, ToolExecutor(),
                    budget=Budget(max_output_tokens=100, max_output_tokens_upgraded=1000,
                                 max_output_truncation_recoveries=3))
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.output_truncation_count == 1   # 只有"升档后仍截断"这次计数
    assert outcome.output_upgraded is True
    assert llm.call_count == 3
    assert llm.calls[1]["max_tokens"] == 1000
    assert llm.calls[2]["max_tokens"] == 1000   # 升档后不会退回小预算


async def test_no_upgrade_configured_falls_directly_to_nudge_path():
    """max_output_tokens_upgraded 为 None(默认):不做任何静默重试,
    第一次截断直接进入 Nudge 恢复路径——'不配置就不该有感'。"""
    llm = FakeLLMClient([
        {"content": "被截断", "tool_calls": [], "finish_reason": "length"},
        {"content": "续写完成", "tool_calls": [], "finish_reason": "stop"},
    ])
    cfg = _make_cfg(llm, ToolExecutor())   # 默认 Budget,两个 max_output_tokens 都是 None
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.output_truncation_count == 1
    assert outcome.output_upgraded is False
    assert "max_tokens" not in llm.calls[0]   # 不配置就不传这个参数
    assert llm.call_count == 2


async def test_agent_resume_preserves_overflow_recovery_count_end_to_end(tmp_path):
    """Agent 层的端到端验证:不满足于上面 loop 层的单元测试(直接摆
    一个带初始计数的 LoopState)——这里真的经过"快照落盘 →
    Agent.resume_events() 读回 → LoopState.from_dict() 重建"这条
    完整链路,证明 agent.py 里任务 3.9 的接线是对的,不只是
    state.py/loop.py 本身逻辑对。"""
    from datetime import datetime
    from harness.snapshot.models import RunSnapshot

    llm = _AlwaysOverflowLLM()
    executor = _executor_with_search_and_finish()
    store = FileSnapshotStore(tmp_path / "snap")
    agent = Agent(
        llm_client=llm, tool_executor=executor, system_prompt="test",
        snapshot_store=store, budget=Budget(max_overflow_recoveries=3),
        name="sess-of",
    )
    # 手工落一份"挂起中、resume 之前已经压缩过 2 次"的快照,不需要
    # 真的先跑一轮完整的"溢出→压缩→挂起"序列去凑出这个状态。
    snap = RunSnapshot(
        session_id="sess-of", task="问题", trace_id="t1", status="awaiting_approval",
        rounds=1, tool_calls_used=1,
        messages=[
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "search", "arguments": '{"query":"q"}'}},
            ]},
        ],
        created_at=datetime.now().isoformat(),
        pending_tool_call_id="c1", pending_approval_id="a1",
        overflow_recovery_count=2,
    )
    store.save(snap)

    outcome = await agent.resume("sess-of", Allow())

    assert outcome.status == "overflow"
    # 2(resume 之前已经压过)+ 1(resume 期间又撞了一次就到上限 3)
    assert outcome.overflow_recovery_count == 3


# ── 任务 3.11:run_result() 的 aborted 分支 ────────────────────────────────

async def test_run_result_aborted_branch():
    from harness.agent.loop import LoopOutcome

    class _AbortedAgent(Agent):
        async def run(self, task, history=None, parent_span=None, session_id=None):
            return LoopOutcome(status="aborted", reason="aborted_streaming",
                              rounds=1, tool_calls_used=0, messages=[])

    agent = _AbortedAgent(
        llm_client=FakeLLMClient([]), tool_executor=ToolExecutor(),
        system_prompt="test",
    )

    result = await agent.run_result("问题")

    assert result.status == "error"
    assert result.summary == "运行被中断"


# ── 任务 3.12:on_error_snapshot 钩子 ─────────────────────────────────────

async def test_on_error_snapshot_disabled_by_default_saves_nothing(tmp_path):
    store = FileSnapshotStore(tmp_path / "snap")
    agent = Agent(
        llm_client=RaisingLLMClient(RuntimeError("boom")), tool_executor=ToolExecutor(),
        system_prompt="test", snapshot_store=store, name="sess-err",
        # on_error_snapshot 默认 False
    )

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("问题", session_id="sess-err")

    with pytest.raises(FileNotFoundError):
        store.load_latest("sess-err")


async def test_on_error_snapshot_enabled_saves_status_error_snapshot(tmp_path):
    store = FileSnapshotStore(tmp_path / "snap")
    agent = Agent(
        llm_client=RaisingLLMClient(RuntimeError("boom")), tool_executor=ToolExecutor(),
        system_prompt="test", snapshot_store=store, on_error_snapshot=True,
        name="sess-err2",
    )

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("问题", session_id="sess-err2")

    snap = store.load_latest("sess-err2")
    assert snap.status == "error"
    assert snap.exit_reason == "unhandled_exception:RuntimeError"
    assert any(m.get("role") == "user" for m in snap.messages)  # 已发生的消息现场被保留


# ── empty_response(真实冒烟 Part 4 暴露后补上) ────────────────────────────

async def test_reason_empty_response_when_model_returns_nothing():
    """模型返回 finish_reason="stop" 但 content 为空——status 仍是
    completed(它确实正常结束了,不是耗尽也不是出错),但 reason 要如实
    标注,否则宿主拿到空字符串却被告知"正常完成",没有任何依据决定
    要不要重试。这个场景是真实冒烟实测撞出来的,不是想象的边界。"""
    llm = FakeLLMClient([{"content": "", "tool_calls": [], "finish_reason": "stop"}])
    cfg = _make_cfg(llm, ToolExecutor())
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.reason == "empty_response"
    assert outcome.final_text == ""


async def test_reason_empty_response_for_whitespace_only_content():
    """只有空白字符也算空——模型返回一个换行符不比返回空字符串更有
    价值,不该因为"技术上非空"就报 completed。"""
    llm = FakeLLMClient([{"content": "  \n  ", "tool_calls": [], "finish_reason": "stop"}])
    cfg = _make_cfg(llm, ToolExecutor())
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.reason == "empty_response"


async def test_normal_answer_still_reports_completed_not_empty_response():
    """反向确认:正常有内容的回答不受影响,仍然是 reason=completed。"""
    llm = FakeLLMClient([{"content": "正常答案", "tool_calls": [], "finish_reason": "stop"}])
    cfg = _make_cfg(llm, ToolExecutor())
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.reason == "completed"


async def test_terminal_tool_not_called_takes_precedence_over_empty_response():
    """两个条件同时成立时(要求 finish 工具、模型却返回空文本),
    报 terminal_tool_not_called——它是更具体、对宿主更有行动价值的
    诊断("模型没按要求用工具收口"比"模型没说话"更能指向问题)。"""
    llm = FakeLLMClient([
        {"content": "", "tool_calls": [], "finish_reason": "stop"},
        {"content": "", "tool_calls": [], "finish_reason": "stop"},
    ])
    executor = _executor_with_search_and_finish()
    cfg = _make_cfg(llm, executor, require_terminal_tool="finish", max_terminal_nudges=1)
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.reason == "terminal_tool_not_called"