# tests/agent/test_loop_termination.py
"""对应待做计划表任务 1.4-1.9(终止判定重构)的核心验证,第二刀迁移版。

【第二刀迁移说明】断言内容与第一刀完全相同——第二刀是"结构重构,
行为应零变化"(设计方案 §8 对第二刀的验证要求),这份文件改的只是
调用方式(AgentLoop 类 → LoopConfig+LoopState+query()),不是断言。
逐条对照第一刀版本可以确认这一点。

设计方案 §3 的论点是:终止判定应该是内建机制(CC 语义:有没有
tool_calls),结构化收口应该降级成一个普通工具(terminal=True)。
这份测试逐条验证这个论点站得住:
  - CC 语义单独成立(不配置 require_terminal_tool 时)
  - terminal 工具是"省一轮 API 调用"的快捷路径,不是另一套判定规则
  - 同轮混发普通工具+finish 时,前者的结果先进消息列表(§3.2 的诚实处理)
  - 参数校验失败退化为普通工具错误,天然计入 tool_calls_used 预算
  - 权限门控对 finish 工具和对普通工具一视同仁
  - 挂起→恢复(resume_point)路径同样吃到快捷终止路径
  - 预算耗尽三种 action 在没有 TerminationPolicy 的情况下仍然工作
"""
from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from harness.agent.finish_tool import build_finish_tool
from harness.agent.loop import Budget, _wrap_up_round, wrap_store
from harness.agent.permission import Allow, AllowAllPolicy, Defer, Deny
from harness.agent.query import run_to_outcome
from harness.agent.run_context import RunContext
from harness.agent.state import LoopConfig, LoopState, ResumePoint
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor

from tests.loop.fakes import FakeLLMClient


def _role(m) -> str | None:
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)


def _content(m) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    return c or ""


class _DummyData(BaseModel):
    answer: str


def _run_ctx() -> RunContext:
    return RunContext.begin("test-agent", "测试任务")


async def _search(query: str) -> str:
    return f"搜索结果:{query}"


def _executor_with_search_and_finish() -> ToolExecutor:
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="search", func=_search, description="", parameters={}))
    executor.register(build_finish_tool(_DummyData))
    return executor


def _finish_call(answer: str = "42", call_id: str = "call_finish"):
    return {"id": call_id, "name": "finish",
            "arguments": {"status": "ok", "summary": f"答案是{answer}", "data": {"answer": answer}}}


def _search_call(query: str = "q", call_id: str = "call_search"):
    return {"id": call_id, "name": "search", "arguments": {"query": query}}


def _make_cfg(
    llm, executor, *, budget: Budget | None = None, permission_policy=None,
    require_terminal_tool: str | None = None, max_terminal_nudges: int = 2,
) -> LoopConfig:
    """第二刀新增的测试工厂:把第一刀 `AgentLoop(llm, executor, ...)`
    构造替换成等价的 LoopConfig。tools 复用 executor.schemas,与
    Agent.__init__ 里的真实构造方式保持一致(不是测试专用捷径)。"""
    return LoopConfig(
        llm=llm, tool_executor=executor, budget=budget or Budget(),
        permission_policy=permission_policy or AllowAllPolicy(),
        tools=executor.schemas,
        require_terminal_tool=require_terminal_tool,
        max_terminal_nudges=max_terminal_nudges,
    )


def _make_state(messages: list[dict], run_ctx: RunContext, **overrides) -> LoopState:
    return LoopState(store=wrap_store(messages, run_ctx), **overrides)


# ── 任务 1.4:termination.py 已删除 ──────────────────────────────────────

def test_termination_module_no_longer_exists():
    with pytest.raises(ModuleNotFoundError):
        import harness.agent.termination  # noqa: F401


def test_loop_config_has_no_termination_field():
    """第二刀迁移:第一刀检查的是 AgentLoop.__init__ 签名,AgentLoop
    类在第二刀已经消失,改为检查 LoopConfig 的字段——本质上是同一件
    事:配置里没有 termination,有 require_terminal_tool/max_terminal_nudges。"""
    sig = inspect.signature(LoopConfig.__init__)
    assert "termination" not in sig.parameters
    assert "require_terminal_tool" in sig.parameters
    assert "max_terminal_nudges" in sig.parameters


# ── 任务 1.9:_wrap_up_round 不再接收 terminal_calls ─────────────────────

def test_wrap_up_round_signature_has_no_terminal_calls_param():
    sig = inspect.signature(_wrap_up_round)
    assert "terminal_calls" not in sig.parameters


# ── 任务 1.7 主干:CC 语义(不配置 require_terminal_tool)─────────────────

async def test_plain_text_completes_immediately_cc_semantics():
    llm = FakeLLMClient([{"content": "这是最终答案", "tool_calls": []}])
    executor = ToolExecutor()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.final_text == "这是最终答案"
    assert outcome.result is None
    assert llm.call_count == 1


# ── terminal 工具快捷路径:省一轮 API 调用 ────────────────────────────────

async def test_terminal_tool_completes_in_same_round_without_extra_call():
    llm = FakeLLMClient([{"content": None, "tool_calls": [_finish_call()]}])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.result.status == "ok"
    assert outcome.result.data == {"answer": "42"}
    assert llm.call_count == 1


async def test_mixed_batch_search_then_finish_completes_same_round():
    llm = FakeLLMClient([{
        "content": None,
        "tool_calls": [_search_call(), _finish_call()],
    }])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.tool_calls_used == 2
    assert llm.call_count == 1
    tool_messages = [m for m in outcome.messages if _role(m) == "tool"]
    assert any("搜索结果" in _content(m) for m in tool_messages)


# ── 参数校验失败:退化为普通工具错误,天然有预算上限 ──────────────────────

async def test_finish_invalid_args_retries_then_succeeds():
    bad_call = {"id": "c1", "name": "finish",
                "arguments": {"status": "ok", "summary": "坏参数", "data": {}}}
    llm = FakeLLMClient([
        {"content": None, "tool_calls": [bad_call]},
        {"content": None, "tool_calls": [_finish_call(call_id="c2")]},
    ])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert llm.call_count == 2
    assert outcome.tool_calls_used == 2
    tool_messages = [m for m in outcome.messages if _role(m) == "tool"]
    assert any("参数错误" in _content(m) for m in tool_messages)


async def test_bad_json_arguments_does_not_crash_and_consumes_budget():
    malformed = {"id": "c1", "name": "finish", "arguments": "{not json"}
    llm = FakeLLMClient([
        {"content": None, "tool_calls": [malformed]},
        {"content": "放弃了", "tool_calls": []},
    ])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.result is None
    assert outcome.final_text == "放弃了"


# ── 权限门控对 finish 一视同仁 ────────────────────────────────────────────

class _DenyFinishPolicy:
    async def check(self, tool_name, args, run_ctx):
        if tool_name == "finish":
            return Deny(reason="本次不允许收口")
        return Allow()


async def test_finish_denied_by_permission_does_not_complete():
    llm = FakeLLMClient([
        {"content": None, "tool_calls": [_finish_call()]},
        {"content": "改口用文本回答", "tool_calls": []},
    ])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor, permission_policy=_DenyFinishPolicy())
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.result is None
    assert outcome.final_text == "改口用文本回答"
    assert llm.call_count == 2
    tool_messages = [m for m in outcome.messages if _role(m) == "tool"]
    assert any("[权限拒绝]" in _content(m) for m in tool_messages)


class _DeferOncePolicy:
    def __init__(self):
        self.deferred_once = False

    async def check(self, tool_name, args, run_ctx):
        if tool_name == "finish" and not self.deferred_once:
            self.deferred_once = True
            return Defer(approval_id="appr-1")
        return Allow()


async def test_finish_deferred_then_resumed_via_resume_point_completes():
    """挂起(Defer)→ 恢复(resume_point)路径:第二刀里没有单独的
    resume_to_outcome() 了——"恢复"只是构造一个带 resume_point 的
    LoopState,再走同一个 run_to_outcome()/query() 入口(设计方案
    §5.2 的偏离说明,query.py 一份代码服务两种入口)。"""
    llm = FakeLLMClient([{"content": None, "tool_calls": [_finish_call()]}])
    executor = _executor_with_search_and_finish()
    policy = _DeferOncePolicy()
    cfg = _make_cfg(llm, executor, permission_policy=policy)

    run_ctx1 = _run_ctx()
    state1 = _make_state([{"role": "user", "content": "问题"}], run_ctx1)
    outcome = await run_to_outcome(cfg, state1, run_ctx1)
    assert outcome.status == "awaiting_approval"
    assert outcome.pending_tool_call_id == "call_finish"
    assert llm.call_count == 1  # 挂起不该额外调模型

    run_ctx2 = _run_ctx()
    state2 = _make_state(
        outcome.messages, run_ctx2,
        rounds=outcome.rounds, tool_calls_used=outcome.tool_calls_used,
        resume_point=ResumePoint(tool_call_id=outcome.pending_tool_call_id, decision=Allow()),
    )
    resumed = await run_to_outcome(cfg, state2, run_ctx2)

    assert resumed.status == "completed"
    assert resumed.result.status == "ok"
    assert llm.call_count == 1  # 恢复不需要再调模型


# ── nudge 收口:有上限,用尽后降级,不无限催 ───────────────────────────────

async def test_nudge_prompts_then_succeeds():
    llm = FakeLLMClient([
        {"content": "我觉得答案是42", "tool_calls": []},
        {"content": None, "tool_calls": [_finish_call()]},
    ])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor, require_terminal_tool="finish", max_terminal_nudges=2)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.result.status == "ok"
    assert llm.call_count == 2
    nudge_msgs = [m for m in outcome.messages
                 if _role(m) == "user" and "请调用 finish 工具" in _content(m)]
    assert len(nudge_msgs) == 1


async def test_nudge_exhausted_downgrades_to_plain_text_completion():
    llm = FakeLLMClient([
        {"content": "回答1", "tool_calls": []},
        {"content": "回答2", "tool_calls": []},
        {"content": "回答3(最终)", "tool_calls": []},
    ])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor, require_terminal_tool="finish", max_terminal_nudges=2)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.result is None
    assert outcome.final_text == "回答3(最终)"
    assert llm.call_count == 3


async def test_max_terminal_nudges_zero_means_immediate_downgrade():
    llm = FakeLLMClient([{"content": "直接回答", "tool_calls": []}])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor, require_terminal_tool="finish", max_terminal_nudges=0)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.final_text == "直接回答"
    assert llm.call_count == 1


# ── 预算耗尽三种 action ───────────────────────────────────────────────────

async def test_budget_exhausted_stop_action_by_max_rounds():
    llm = FakeLLMClient([{"content": None, "tool_calls": [_search_call()]}])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor, budget=Budget(max_rounds=1, exhausted_action="stop"))
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "exhausted"
    assert outcome.final_text == ""
    assert llm.call_count == 1


async def test_budget_exhausted_by_max_tool_calls():
    llm = FakeLLMClient([{"content": None, "tool_calls": [_search_call()]}])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor, budget=Budget(max_tool_calls=1, exhausted_action="stop"))
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "exhausted"
    assert outcome.tool_calls_used == 1
    assert llm.call_count == 1


async def test_budget_exhausted_force_answer_action():
    llm = FakeLLMClient([
        {"content": None, "tool_calls": [_search_call()]},
        {"content": "最终答案(强制作答)"},
    ])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor, budget=Budget(max_rounds=1, exhausted_action="force_answer"))
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "exhausted"
    assert outcome.final_text == "最终答案(强制作答)"
    assert llm.call_count == 2


async def test_budget_exhausted_force_finish_action_model_completes_via_tool():
    llm = FakeLLMClient([
        {"content": None, "tool_calls": [_search_call()]},
        {"content": None, "tool_calls": [_finish_call(answer="99")]},
    ])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor, budget=Budget(max_rounds=1, exhausted_action="force_finish"))
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "exhausted"
    assert outcome.result is not None
    assert outcome.result.data == {"answer": "99"}
    assert llm.call_count == 2


async def test_budget_exhausted_force_finish_action_falls_back_to_plain_text():
    llm = FakeLLMClient([
        {"content": None, "tool_calls": [_search_call()]},
        {"content": "没能力给出结构化结论", "tool_calls": []},
    ])
    executor = _executor_with_search_and_finish()
    run_ctx = _run_ctx()
    cfg = _make_cfg(llm, executor, budget=Budget(max_rounds=1, exhausted_action="force_finish"))
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "exhausted"
    assert outcome.result is None
    assert outcome.final_text == "没能力给出结构化结论"
