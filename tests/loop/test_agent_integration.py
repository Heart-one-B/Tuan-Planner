# tests/agent/test_agent_integration.py
"""对应待做计划表任务 1.6(Agent 构造参数)+ 1.10(extraction.py 迁移)的
集成验证。round1_agent_py_patch_notes.md 里承诺过:补齐 offload.py/
tracer.py 之后就能把这层测试补上——现在补上。

这份测试不重复 test_loop_termination.py 已经覆盖的终止判定逻辑本身,
只验证"接上 Agent 这一层、接上真实 snapshot/memory/context 子系统后,
第一刀的改动有没有露出接缝"。
"""
from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from harness.agent.agent import Agent
from harness.agent.finish_tool import build_finish_tool
from harness.agent.permission import Allow, Defer
from harness.memory.extraction import _build_extraction_agent
from harness.memory.instructions import MemoryConfig
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor
from harness.snapshot.store import FileSnapshotStore

from tests.loop.fakes import FakeLLMClient


class _DummyData(BaseModel):
    answer: str


def _executor_with_finish() -> ToolExecutor:
    executor = ToolExecutor()
    executor.register(build_finish_tool(_DummyData))
    return executor


def _finish_call(answer: str = "42", call_id: str = "call_finish"):
    return {"id": call_id, "name": "finish",
            "arguments": {"status": "ok", "summary": f"答案是{answer}", "data": {"answer": answer}}}


# ── 任务 1.6:构造参数正确转发给 AgentLoop ────────────────────────────────

def test_agent_constructor_has_no_termination_param():
    sig = inspect.signature(Agent.__init__)
    assert "termination" not in sig.parameters
    assert "require_terminal_tool" in sig.parameters
    assert "max_terminal_nudges" in sig.parameters


def test_agent_forwards_terminal_params_to_loop():
    llm = FakeLLMClient([{"content": "答案", "tool_calls": []}])
    executor = _executor_with_finish()
    agent = Agent(
        llm_client=llm, tool_executor=executor, system_prompt="test",
        require_terminal_tool="finish", max_terminal_nudges=1,
    )
    assert agent._loop_config.require_terminal_tool == "finish"
    assert agent._loop_config.max_terminal_nudges == 1


# ── 端到端:Agent.run() 通过 finish 工具收口(最简路径,三个子系统都不配置) ──

async def test_agent_run_completes_via_finish_tool_minimal_config():
    llm = FakeLLMClient([{"content": None, "tool_calls": [_finish_call()]}])
    executor = _executor_with_finish()
    agent = Agent(
        llm_client=llm, tool_executor=executor, system_prompt="你是一个助手",
        require_terminal_tool="finish",
    )

    outcome = await agent.run("问题")

    assert outcome.status == "completed"
    assert outcome.result.status == "ok"
    assert outcome.result.data == {"answer": "42"}
    assert llm.call_count == 1


async def test_agent_run_result_returns_structured_result_directly():
    """run_result() 在 outcome.result 非 None 时直接透传,不走
    final_text 兜底那条路径。"""
    llm = FakeLLMClient([{"content": None, "tool_calls": [_finish_call(answer="7")]}])
    executor = _executor_with_finish()
    agent = Agent(
        llm_client=llm, tool_executor=executor, system_prompt="你是一个助手",
        require_terminal_tool="finish",
    )

    result = await agent.run_result("问题")

    assert result.status == "ok"
    assert result.data == {"answer": "7"}


async def test_agent_run_cc_semantics_without_terminal_tool():
    """不配置 require_terminal_tool 时,Agent 层同样是纯 CC 语义。"""
    llm = FakeLLMClient([{"content": "直接的文本答案", "tool_calls": []}])
    executor = ToolExecutor()
    agent = Agent(llm_client=llm, tool_executor=executor, system_prompt="你是一个助手")

    outcome = await agent.run("问题")

    assert outcome.status == "completed"
    assert outcome.final_text == "直接的文本答案"
    assert outcome.result is None


# ── 真实 snapshot 落盘/读回全链路(含 assistant 消息的 tool_calls 序列化) ──

async def test_agent_with_snapshot_store_persists_completed_run(tmp_path):
    llm = FakeLLMClient([{"content": None, "tool_calls": [_finish_call()]}])
    executor = _executor_with_finish()
    store = FileSnapshotStore(tmp_path / "snapshots")
    agent = Agent(
        llm_client=llm, tool_executor=executor, system_prompt="你是一个助手",
        require_terminal_tool="finish", snapshot_store=store,
        name="sess-1",
    )

    outcome = await agent.run("问题", session_id="sess-1")
    assert outcome.status == "completed"

    snap = store.load_latest("sess-1")
    assert snap.status == "completed"
    assert snap.pending_tool_call_id is None
    # normalize_message 必须能正确序列化 FakeLLMClient 产出的
    # SimpleNamespace 风格 assistant 消息(含 tool_calls)
    assistant_msgs = [m for m in snap.messages if m.get("role") == "assistant"]
    assert any(m.get("tool_calls") for m in assistant_msgs)


# ── 端到端:Defer 挂起 → 落快照 → Agent.resume() 恢复并收口 ──────────────

class _DeferOncePolicy:
    def __init__(self):
        self.deferred_once = False

    async def check(self, tool_name, args, run_ctx):
        if tool_name == "finish" and not self.deferred_once:
            self.deferred_once = True
            return Defer(approval_id="appr-1")
        return Allow()


async def test_agent_full_defer_then_resume_round_trip(tmp_path):
    """这是本轮补测的核心场景:任务 1.6 没有改动 Agent.resume() 的代码,
    但它依赖的 AgentLoop.resume() 在第一刀被大幅改写(去掉了终止工具
    分拣)。这里验证从 Agent.run() 挂起、真实落盘、真实读回、到
    Agent.resume() 收口的完整链路没有断。"""
    llm = FakeLLMClient([{"content": None, "tool_calls": [_finish_call()]}])
    executor = _executor_with_finish()
    store = FileSnapshotStore(tmp_path / "snapshots")
    policy = _DeferOncePolicy()
    agent = Agent(
        llm_client=llm, tool_executor=executor, system_prompt="你是一个助手",
        require_terminal_tool="finish", snapshot_store=store,
        permission_policy=policy, name="sess-defer",
    )

    outcome = await agent.run("问题", session_id="sess-defer")
    assert outcome.status == "awaiting_approval"
    assert llm.call_count == 1

    snap = store.load_latest("sess-defer")
    assert snap.pending_tool_call_id == "call_finish"
    assert snap.pending_approval_id == "appr-1"

    resumed = await agent.resume("sess-defer", Allow())

    assert resumed.status == "completed"
    assert resumed.result.status == "ok"
    assert llm.call_count == 1  # resume 全程没有再调模型

    final_snap = store.load_latest("sess-defer")
    assert final_snap.status == "completed"
    assert final_snap.pending_tool_call_id is None


async def test_agent_finalize_runs_exactly_once_per_run(tmp_path):
    """任务 2.8:_finalize 是四个入口共用的唯一一份收尾。这里不满足于
    '结构上不会重复'这句话本身,直接数 snapshot_store.save() 被调用
    了几次——如果 run()/events() 未来哪次改动不小心又各自调用一次
    _finalize,这条测试会先坏。"""
    llm = FakeLLMClient([{"content": None, "tool_calls": [_finish_call()]}])
    executor = _executor_with_finish()
    store = FileSnapshotStore(tmp_path / "snapshots")
    save_calls = []
    original_save = store.save

    def _counting_save(snap):
        save_calls.append(snap)
        return original_save(snap)

    store.save = _counting_save
    agent = Agent(
        llm_client=llm, tool_executor=executor, system_prompt="你是一个助手",
        require_terminal_tool="finish", snapshot_store=store, name="sess-once",
    )

    outcome = await agent.run("问题", session_id="sess-once")

    assert outcome.status == "completed"
    assert len(save_calls) == 1

def test_extraction_agent_builder_uses_cc_semantics(tmp_path):
    """_build_extraction_agent 不再传 termination=AnswerTermination(),
    验证它现在依赖的是 AgentLoop 默认的 CC 语义(require_terminal_tool
    保持 None),而不是意外遗留了别的收口要求。"""
    llm = FakeLLMClient([])
    config = MemoryConfig(memory_dir=tmp_path / "memory")

    agent = _build_extraction_agent(
        memory_store=None,  # 构造阶段不会用到,build_memory_tools 会先用到,
                            # 但 include_delete=False 时 build_memory_tools
                            # 本身不读 store 的任何数据,只把它闭包进工具函数
        memory_config=config,
        llm_client=llm,
    )

    assert agent._loop_config.require_terminal_tool is None
    assert agent._loop_config.max_terminal_nudges == 2  # 默认值,未被特殊配置改写


async def test_extraction_agent_completes_plain_text_no_memory_written(tmp_path):
    """提取 Agent 真实跑一轮:模型判断"本轮无新增记忆"直接吐文本,
    验证 CC 语义(无 tool_calls 即终止)在提取场景下工作正常。"""
    from harness.memory.store import FileMemoryStore

    llm = FakeLLMClient([{"content": "本轮无新增记忆", "tool_calls": []}])
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    store = FileMemoryStore(tmp_path / "memory")

    agent = _build_extraction_agent(memory_store=store, memory_config=config, llm_client=llm)
    outcome = await agent.run("提取任务", history=[{"role": "user", "content": "之前聊了什么"}])

    assert outcome.status == "completed"
    assert outcome.final_text == "本轮无新增记忆"
    assert llm.call_count == 1
