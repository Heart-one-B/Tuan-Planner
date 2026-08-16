# tests/agent/test_abort.py
"""对应待做计划表任务 4.5-4.8 的核心验证。

第一性问题:中断机制存在的意义是"能被外部安全打断,且打断后现场
仍然可用"——不是"打断了就完事"。这份测试盯的是两件事:①检查点
真的在该检查的地方检查;②中断之后,消息历史里不能留下任何会让
provider 拒收(400)的孤儿 tool_use。
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness.agent.abort import AbortSignal
from harness.agent.agent import Agent
from harness.agent.finish_tool import build_finish_tool
from harness.agent.loop import Budget, wrap_store
from harness.agent.permission import AllowAllPolicy
from harness.agent.query import run_to_outcome
from harness.agent.repair import ORPHAN_MARKER
from harness.agent.run_context import RunContext
from harness.agent.state import LoopConfig, LoopState
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor

from tests.loop.fakes import FakeLLMClient


def _role(m) -> str | None:
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)


def _content(m) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    return c or ""


def _tool_call_id(m):
    return m.get("tool_call_id") if isinstance(m, dict) else getattr(m, "tool_call_id", None)


def _run_ctx(abort: AbortSignal | None = None) -> RunContext:
    return RunContext.begin("test-agent", "测试任务", abort=abort)


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


# ── AbortSignal 本身 ──────────────────────────────────────────────────────

def test_abort_signal_starts_unset():
    sig = AbortSignal()
    assert sig.is_set() is False
    assert sig.reason is None


def test_abort_signal_records_reason():
    sig = AbortSignal()
    sig.abort(reason="用户点了取消")
    assert sig.is_set() is True
    assert sig.reason == "用户点了取消"


def test_abort_signal_default_reason():
    sig = AbortSignal()
    sig.abort()
    assert sig.reason == "user_requested"


def test_abort_signal_repeated_call_is_idempotent_uses_latest_reason():
    sig = AbortSignal()
    sig.abort(reason="第一次")
    sig.abort(reason="第二次")
    assert sig.is_set() is True
    assert sig.reason == "第二次"


# ── 检查点①:每轮开头(任务 4.7/4.8)──────────────────────────────────────

async def test_checkpoint_round_start_aborts_before_any_model_call():
    """在第一轮开始之前就已经中断:不该发生任何模型调用,消息历史
    干净(没有任何 assistant/tool 消息需要修复)。"""
    sig = AbortSignal()
    sig.abort(reason="还没开始就取消了")
    llm = FakeLLMClient([{"content": "不该被用到", "tool_calls": []}])
    cfg = _make_cfg(llm, ToolExecutor())
    run_ctx = _run_ctx(abort=sig)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "aborted"
    assert outcome.reason == "aborted_streaming"
    assert llm.call_count == 0   # 一次模型都没调


async def test_checkpoint_round_start_aborts_between_rounds():
    """第一轮正常跑完(工具调用),中断信号在第二轮开始前才被设置——
    验证检查点在每一轮开头都会重新检查,不是只查一次。"""
    sig = AbortSignal()

    async def _search_then_abort(query: str) -> str:
        sig.abort(reason="第一轮做完后取消")
        return f"搜索结果:{query}"

    llm = FakeLLMClient([
        {"content": None, "tool_calls": [{"id": "c1", "name": "search", "arguments": {"query": "q"}}]},
        {"content": "不该被用到", "tool_calls": []},
    ])
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="search", func=_search_then_abort,
                                     description="", parameters={}))
    cfg = _make_cfg(llm, executor)
    run_ctx = _run_ctx(abort=sig)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "aborted"
    assert outcome.reason == "aborted_streaming"
    assert llm.call_count == 1   # 只调了第一轮,第二轮被检查点拦下了
    # 第一轮的工具调用是完整批次(在中断信号被设置之前就已经处理完),
    # 不该有任何孤儿
    assert not any(ORPHAN_MARKER in _content(m)
                   for m in outcome.messages if _role(m) == "tool")


# ── 检查点②:工具批次每个工具之间(任务 4.7/4.8)──────────────────────────

async def test_checkpoint_tool_batch_aborts_mid_batch_and_repairs_orphans():
    """一批 3 个工具调用,第一个执行时触发中断:第一个应该有真实结果,
    第二、三个应该被检查点拦下、随后被修复成占位——不是被静默丢弃。"""
    sig = AbortSignal()

    async def _t1(query: str) -> str:
        sig.abort(reason="执行 t1 期间用户取消")
        return "t1 的真实结果"

    async def _t2(query: str) -> str:
        raise AssertionError("t2 不该被执行——它应该在检查点被拦下")

    llm = FakeLLMClient([{
        "content": None,
        "tool_calls": [
            {"id": "c1", "name": "t1", "arguments": {"query": "a"}},
            {"id": "c2", "name": "t2", "arguments": {"query": "b"}},
            {"id": "c3", "name": "t2", "arguments": {"query": "c"}},
        ],
    }])
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="t1", func=_t1, description="", parameters={}))
    executor.register(ToolDefinition(name="t2", func=_t2, description="", parameters={}))
    cfg = _make_cfg(llm, executor)
    run_ctx = _run_ctx(abort=sig)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "aborted"
    assert outcome.reason == "aborted_tools"

    tool_msgs = {_tool_call_id(m): _content(m) for m in outcome.messages
                if _role(m) == "tool"}
    assert tool_msgs["c1"] == "t1 的真实结果"           # 真实执行过的不受影响
    assert ORPHAN_MARKER in tool_msgs["c2"]              # 被拦下的都补了占位
    assert ORPHAN_MARKER in tool_msgs["c3"]
    assert "用户中断" in tool_msgs["c2"]

    # 不留任何孤儿:assistant 声称调用过的三个 id,每一个都能在
    # tool 消息里找到配对(不管是真实结果还是修复出来的占位)
    assistant_msgs = [m for m in outcome.messages if _role(m) == "assistant"]
    for m in assistant_msgs:
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        for tc in (tcs or []):
            tc_id = tc["id"] if isinstance(tc, dict) else tc.id
            assert tc_id in tool_msgs


async def test_checkpoint_round_start_takes_priority_over_tool_batch_checkpoint():
    """检查点①(每轮开头)和检查点②(工具批次之间)的优先级是确定的:
    如果中断信号在"本轮模型调用发生之前"就已经被设置,检查点①必然
    先截获它,模型根本不会被调用、也就不会有 tool_calls 需要修复——
    "还没调模型、还没往消息历史里追加任何东西"确实是最干净的中断
    时机(设计方案 §5.5 的原话)。检查点②只在"本轮模型已经返回
    tool_calls 之后、处理过程中才触发中断"这种更晚的时机才会成为
    实际命中的那个检查点(见上面的 mid_batch 测试)。"""
    sig = AbortSignal()
    sig.abort(reason="批处理开始前就取消了")

    async def _t(query: str) -> str:
        raise AssertionError("不该有任何工具被执行——检查点①应该先拦下")

    llm = FakeLLMClient([{
        "content": None,
        "tool_calls": [
            {"id": "c1", "name": "t", "arguments": {"query": "a"}},
            {"id": "c2", "name": "t", "arguments": {"query": "b"}},
        ],
    }])
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="t", func=_t, description="", parameters={}))
    cfg = _make_cfg(llm, executor)
    run_ctx = _run_ctx(abort=sig)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "aborted"
    assert outcome.reason == "aborted_streaming"   # 不是 aborted_tools
    assert llm.call_count == 0   # 模型压根没被调用,tool_calls 根本不存在
    assert not any(_role(m) == "tool" for m in outcome.messages)  # 没有任何东西需要修复


# ── 不配置 abort:完全无感(老规矩) ──────────────────────────────────────

async def test_no_abort_signal_configured_never_aborts():
    llm = FakeLLMClient([{"content": "正常完成", "tool_calls": []}])
    cfg = _make_cfg(llm, ToolExecutor())
    run_ctx = _run_ctx(abort=None)   # 默认值,显式写出来强调
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"


# ── 异常路径同样会修复(任务 4.2,和中断共用同一个修复机制) ─────────────────
#
# 注意:ToolExecutor.execute() 本身会捕获工具函数抛出的异常、转成
# 【系统错误】文本正常返回(不会向上抛),所以"工具函数自己炸了"这种
# 场景根本走不到 query() 的异常兜底——那是 ToolExecutor 自己的职责
# 范围。query() 的异常兜底服务的是 ToolExecutor 兜底范围之外的问题,
# 比如下面测的:permission_policy.check() 自己抛异常。

async def test_exception_from_permission_policy_triggers_repair_and_reraise():
    class _BoomPolicy:
        async def check(self, tool_name, args, run_ctx):
            raise RuntimeError("权限策略自己炸了")

    llm = FakeLLMClient([{
        "content": None,
        "tool_calls": [{"id": "c1", "name": "t", "arguments": {"query": "x"}}],
    }])
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="t", func=lambda **kw: "ok",
                                     description="", parameters={}))
    cfg = _make_cfg(llm, executor, permission_policy=_BoomPolicy())
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    with pytest.raises(RuntimeError, match="权限策略自己炸了"):
        await run_to_outcome(cfg, state, run_ctx)

    # state.store.messages 是原地被修复的对象,异常抛出之后仍能读到
    tool_msgs = {_tool_call_id(m): _content(m)
                for m in state.store.messages if _role(m) == "tool"}
    assert "c1" in tool_msgs
    assert ORPHAN_MARKER in tool_msgs["c1"]
    assert "执行异常" in tool_msgs["c1"]


# ── Agent 层:abort 参数穿透 ───────────────────────────────────────────────

def test_agent_run_accepts_abort_kwarg():
    import inspect
    sig = inspect.signature(Agent.run)
    assert "abort" in sig.parameters


def test_agent_events_accepts_abort_kwarg():
    import inspect
    sig = inspect.signature(Agent.events)
    assert "abort" in sig.parameters


def test_agent_resume_accepts_abort_kwarg():
    import inspect
    sig = inspect.signature(Agent.resume)
    assert "abort" in sig.parameters


async def test_agent_run_with_abort_signal_end_to_end():
    """不满足于 loop 层的验证——走一遍 Agent.run(abort=...) 的完整
    公共入口,确认 abort 真的从 Agent 一路传到了 RunContext,循环里
    的检查点能读到。"""
    sig = AbortSignal()
    sig.abort(reason="端到端测试")
    llm = FakeLLMClient([{"content": "不该被用到", "tool_calls": []}])
    agent = Agent(llm_client=llm, tool_executor=ToolExecutor(), system_prompt="test")

    outcome = await agent.run("问题", abort=sig)

    assert outcome.status == "aborted"
    assert outcome.reason == "aborted_streaming"
    assert llm.call_count == 0


def test_run_context_default_abort_is_none():
    """不配置就不该有感——默认行为和之前完全一样。"""
    run_ctx = RunContext.begin("test", "task")
    assert run_ctx.abort is None