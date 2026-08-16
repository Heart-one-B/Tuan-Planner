# tests/agent/test_state_and_layers.py
"""对应待做计划表第二刀任务 2.1-2.5 的结构性验证。

这份测试不重复 test_loop_termination.py 已经覆盖的终止判定逻辑,
只验证第二刀这次重构本身声称做到的几件事是不是真的做到了:
  - LoopConfig 真的不可变(frozen)
  - LoopState.to_dict()/from_dict() 真的只导出计数器,store/abort/
    resume_point 真的没进去
  - 第三层(query.py)真的统一做了收尾:异常 → span status=error 且
    re-raise;生成器被提前关闭 → span status=abandoned
  - 第四层(run_query_loop)真的不再自己捕获异常——直接调它、不经过
    query() 包装时,异常应该原样冒出来,span 也不会被它结束
  - _process_tool_calls 真的不再靠 `_tool_batch_done` 事件传递计数
    (决策 E 的直接验证,不只是看文档怎么说)
"""
from __future__ import annotations

import dataclasses

import pytest
from pydantic import BaseModel

from harness.agent.finish_tool import build_finish_tool
from harness.agent.loop import Budget, _process_tool_calls, run_query_loop, wrap_store
from harness.agent.permission import Allow, AllowAllPolicy
from harness.agent.query import query, run_to_outcome
from harness.agent.run_context import RunContext
from harness.agent.state import LoopConfig, LoopState, ResumePoint
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor
from harness.tracing import tracer
from harness.tracing.storage_base import TraceStorageBase

from tests.loop.fakes import FakeLLMClient, RaisingLLMClient


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


class _FakeTraceStorage(TraceStorageBase):
    """捕获 tracer.end_trace() 最终落盘的 Trace,供测试断言 span 的
    收尾状态——tracer.py 是你的原文,不改;这是标准的
    TraceStorageBase 实现方,测试专用。"""
    def __init__(self):
        self.saved: dict[str, object] = {}

    def save_trace(self, trace) -> None:
        self.saved[trace.trace_id] = trace

    def get_traces_by_session(self, session_id: str) -> list[dict]:
        return []


# ── 任务 2.1:LoopConfig 真的不可变 ───────────────────────────────────────

def test_loop_config_is_frozen():
    cfg = _make_cfg(FakeLLMClient([]), ToolExecutor())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.max_terminal_nudges = 99


def test_loop_config_tools_is_a_snapshot_not_recomputed():
    """cfg.tools 在构造时算好、之后每轮复用——不是每次访问都重新
    调用 tool_executor.schemas。构造之后再注册新工具,cfg.tools 不
    应该跟着变。"""
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="a", func=_search, description="", parameters={}))
    cfg = _make_cfg(FakeLLMClient([]), executor)
    assert len(cfg.tools) == 1

    executor.register(ToolDefinition(name="b", func=_search, description="", parameters={}))
    assert len(cfg.tools) == 1          # cfg.tools 是构造时的快照
    assert len(executor.schemas) == 2   # executor 本身确实多了一个


# ── 任务 2.1/2.2:LoopState 序列化边界 ────────────────────────────────────

def test_loop_state_to_dict_only_exports_counters():
    run_ctx = _run_ctx()
    state = LoopState(
        store=wrap_store([], run_ctx), rounds=3, tool_calls_used=5,
        overflow_recovery_count=2, output_truncation_count=1,
        output_upgraded=True, terminal_nudge_count=2,
        resume_point=ResumePoint(tool_call_id="x", decision=Allow()),
        abort=object(), compacted_this_round=True,
    )
    d = state.to_dict()
    assert d == {
        "rounds": 3, "tool_calls_used": 5,
        "overflow_recovery_count": 2, "output_truncation_count": 1,
        "output_upgraded": True, "terminal_nudge_count": 2,
    }
    assert "store" not in d
    assert "abort" not in d
    assert "resume_point" not in d
    assert "compacted_this_round" not in d   # 每轮重置,不该进快照(任务3.4)


def test_loop_state_from_dict_round_trip():
    data = {"rounds": 4, "tool_calls_used": 6,
           "overflow_recovery_count": 2, "output_truncation_count": 1,
           "output_upgraded": True, "terminal_nudge_count": 1}
    run_ctx = _run_ctx()
    store = wrap_store([], run_ctx)

    state = LoopState.from_dict(data, store=store)

    assert state.rounds == 4
    assert state.tool_calls_used == 6
    assert state.overflow_recovery_count == 2
    assert state.output_truncation_count == 1
    assert state.output_upgraded is True
    assert state.terminal_nudge_count == 1
    assert state.resume_point is None   # 不在序列化字段里,默认值
    assert state.compacted_this_round is False  # 同上,每轮重置的字段
    assert state.store is store


def test_loop_state_from_dict_defaults_missing_keys():
    """快照里缺字段(比如 v5 迁移上来、还没有这些计数器)时,from_dict
    应该退回全零默认值,而不是 KeyError——这是给第三刀快照迁移铺路,
    本刀先把这条边界钉死。"""
    run_ctx = _run_ctx()
    store = wrap_store([], run_ctx)
    state = LoopState.from_dict({}, store=store)
    assert state.rounds == 0
    assert state.tool_calls_used == 0
    assert state.overflow_recovery_count == 0
    assert state.output_truncation_count == 0
    assert state.output_upgraded is False
    assert state.terminal_nudge_count == 0


# ── 任务 2.5:第三层统一收尾——异常路径 ────────────────────────────────────

async def test_query_catches_exception_marks_span_error_and_reraises():
    storage = _FakeTraceStorage()
    tracer.configure_storage(storage)
    llm = RaisingLLMClient(RuntimeError("boom"))
    executor = ToolExecutor()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = LoopState(store=wrap_store([{"role": "user", "content": "问题"}], run_ctx))

    with pytest.raises(RuntimeError, match="boom"):
        await run_to_outcome(cfg, state, run_ctx)

    saved = storage.saved[run_ctx.span.trace_id]
    assert saved.status == "error"


# ── 任务 2.5:第三层统一收尾——提前关闭(未产出 outcome)路径 ──────────────

async def test_query_marks_span_abandoned_when_closed_before_outcome():
    storage = _FakeTraceStorage()
    tracer.configure_storage(storage)
    llm = FakeLLMClient([
        {"content": None, "tool_calls": [{
            "id": "c1", "name": "search", "arguments": {"query": "q"},
        }]},
        {"content": "不会用到", "tool_calls": []},
    ])
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="search", func=_search, description="", parameters={}))
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = LoopState(store=wrap_store([{"role": "user", "content": "问题"}], run_ctx))

    gen = query(cfg, state, run_ctx)
    await gen.__anext__()   # 只消费第一个事件,不跑完这次 run
    await gen.aclose()

    saved = storage.saved[run_ctx.span.trace_id]
    assert saved.status == "abandoned"


async def test_query_yields_exactly_one_outcome_event_on_normal_completion():
    """"保证恰好产出一个 outcome"是第三层的明确职责——不是多了就是
    少了,这里直接数。"""
    llm = FakeLLMClient([{"content": "答案", "tool_calls": []}])
    executor = ToolExecutor()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = LoopState(store=wrap_store([{"role": "user", "content": "问题"}], run_ctx))

    outcome_events = [
        ev async for ev in query(cfg, state, run_ctx) if ev["type"] == "outcome"
    ]
    assert len(outcome_events) == 1


# ── 任务 2.3:第四层不自己捕获异常、不自己调 span.end ────────────────────

async def test_run_query_loop_itself_propagates_exceptions_without_sealing_span():
    """直接调 run_query_loop(绕开第三层 query()),异常应该原样冒出来,
    span 不应该被结束——生命周期收尾是第三层的职责,第四层只管转圈。"""
    llm = RaisingLLMClient(RuntimeError("boom"))
    executor = ToolExecutor()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = LoopState(store=wrap_store([{"role": "user", "content": "问题"}], run_ctx))

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in run_query_loop(cfg, state, run_ctx):
            pass

    assert run_ctx.span.trace_id in tracer._active   # 没被 span.end() 弹出
    assert run_ctx.span._ended is False


# ── 决策 E:tool_calls_used 直接在 state 上累加,_tool_batch_done 消失 ────

async def test_process_tool_calls_no_longer_yields_internal_batch_done_event():
    llm = FakeLLMClient([])
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="search", func=_search, description="", parameters={}))
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = LoopState(store=wrap_store([], run_ctx))
    tool_calls = [{"id": "c1", "function": {"name": "search", "arguments": '{"query":"q"}'}}]

    events = [ev async for ev in _process_tool_calls(cfg, state, run_ctx, tool_calls)]

    assert all(ev["type"] != "_tool_batch_done" for ev in events)
    assert state.tool_calls_used == 1   # 直接在 state 上累加,不靠事件传回
    event_types = {ev["type"] for ev in events}
    assert event_types == {"tool_start", "tool_end"}


# ── 任务 2.4:resume_point 恢复后计数器正确延续(不清零) ──────────────────

async def test_resume_point_continues_rounds_and_tool_calls_used_not_reset():
    """构造一个"已经跑过 5 轮、用过 3 次工具调用"的现场,通过
    resume_point 恢复,验证 outcome 里的 rounds/tool_calls_used 是
    在原有基础上累加,不是从 0 重新计——这是快照恢复正确性的基础,
    第三刀的 bug#1 修复也依赖这个前提成立。"""
    llm = FakeLLMClient([])
    executor = _make_executor_with_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)

    messages = [
        {"role": "user", "content": "问题"},
        _fake_assistant_msg_with_tool_call("call_finish", "finish",
                                           {"status": "ok", "summary": "done", "data": {"answer": "1"}}),
    ]
    state = LoopState(
        store=wrap_store(messages, run_ctx),
        rounds=5, tool_calls_used=3,
        resume_point=ResumePoint(tool_call_id="call_finish", decision=Allow()),
    )

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.rounds == 5          # 恢复不凭空推进轮次
    assert outcome.tool_calls_used == 4  # 3(恢复前)+ 1(这次 finish 调用)


class _DummyData(BaseModel):
    answer: str


def _make_executor_with_finish() -> ToolExecutor:
    executor = ToolExecutor()
    executor.register(build_finish_tool(_DummyData))
    return executor


def _fake_assistant_msg_with_tool_call(call_id: str, name: str, arguments: dict) -> dict:
    """构造一条挂起测试要用的、dict 形态的 assistant 消息——模拟快照
    读回后的消息形态(normalize_message 序列化后就是这个形状),
    _find_pending_call_context 需要能在 dict 形态下定位到它。"""
    import json
    return {
        "role": "assistant", "content": None,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }],
    }
