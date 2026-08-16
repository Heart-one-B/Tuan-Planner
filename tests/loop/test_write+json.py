# tests/agent/test_wire_serialization.py
"""对应真实冒烟 Part 1 暴露的 bug:_call_model 产出的消息(通过
StreamAccumulator.build_message() 构造的 SimpleNamespace)一旦被
直接 append 进 state.store.messages,下一轮把完整历史发给 provider
时,OpenAI SDK 的请求体序列化器无法处理 SimpleNamespace,直接
TypeError。

【为什么之前 119 条 Fake 测试全绿,却测不出这个】
FakeLLMClient 全程在内存里传递 Python 对象,从不会真的调用
`json.dumps` 或任何请求体序列化逻辑——"能跑通"和"能被真实序列化"
是两个不同的断言,前者不能替代后者。这份测试文件的每一条用例都
显式调用 json.dumps(直接验证"能不能被序列化"这件事本身),不满足
于"流程跑完了、没报错"这种弱信号。
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from harness.agent.finish_tool import build_finish_tool
from harness.agent.loop import Budget, _to_wire_message, wrap_store
from harness.agent.permission import AllowAllPolicy
from harness.agent.query import run_to_outcome
from harness.agent.run_context import RunContext
from harness.agent.state import LoopConfig, LoopState
from harness.llm.base import StreamDelta
from harness.llm.streaming import StreamAccumulator
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor

from tests.loop.fakes import StreamingFakeLLMClient


def _run_ctx() -> RunContext:
    return RunContext.begin("test-agent", "测试任务")


def _make_cfg(llm, executor, **overrides) -> LoopConfig:
    defaults = dict(
        llm=llm, tool_executor=executor, budget=Budget(),
        permission_policy=AllowAllPolicy(), tools=executor.schemas,
    )
    defaults.update(overrides)
    return LoopConfig(**defaults)


class _DummyData(BaseModel):
    answer: str


# ── _to_wire_message() 单元测试 ──────────────────────────────────────────

def test_to_wire_message_converts_plain_text_message_to_json_safe_dict():
    acc = StreamAccumulator()
    acc.feed(StreamDelta(content="纯文本回复", finish_reason="stop"))
    msg = acc.build_message()   # SimpleNamespace,不是 dict

    wire = _to_wire_message(msg)

    assert isinstance(wire, dict)
    assert wire["role"] == "assistant"
    assert wire["content"] == "纯文本回复"
    # 这才是真正的断言:不是"看起来像个 dict",是"真的能被 json.dumps"
    json.dumps(wire)


def test_to_wire_message_converts_tool_call_message_to_json_safe_dict():
    acc = StreamAccumulator()
    acc.feed(StreamDelta(tool_call_delta={"index": 0, "id": "c1", "name": "search", "arguments": None}))
    acc.feed(StreamDelta(tool_call_delta={"index": 0, "arguments": '{"q":"x"}'}, finish_reason="tool_calls"))
    msg = acc.build_message()

    wire = _to_wire_message(msg)

    assert wire["tool_calls"] == [{
        "id": "c1", "type": "function",
        "function": {"name": "search", "arguments": '{"q":"x"}'},
    }]
    json.dumps(wire)   # 这一行在修复前会 TypeError(嵌套 SimpleNamespace)


def test_to_wire_message_is_idempotent_on_already_dict_messages():
    """resume 场景下,消息可能已经是 dict 形态(比如从快照读回的历史)——
    _to_wire_message 不该对这类消息做多余的转换或破坏。"""
    already_dict = {"role": "assistant", "content": "已经是dict了", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
    ]}
    assert _to_wire_message(already_dict) is already_dict


# ── 端到端:多轮真实走一遍,历史里的每一条消息都必须能被序列化 ────────────────

async def test_multi_round_history_is_fully_json_serializable_after_each_round():
    """这是真正复现 bug 场景的测试:第一轮模型调用工具(产出一条带
    tool_calls 的 assistant 消息),第二轮模型才给出最终答案——第二轮
    发出去的请求体里,必须包含第一轮那条 assistant 消息,而且必须
    能被完整 json.dumps。这条测试如果不加,回归后要等到真实网络调用
    才会暴露,现在能在 Fake 层面就拦住。"""
    llm = StreamingFakeLLMClient([
        {"stream_deltas": [
            {"tool_call_delta": {"index": 0, "id": "c1", "name": "search", "arguments": '{"q":"x"}'}},
            {"finish_reason": "tool_calls"},
        ]},
        {"stream_deltas": [{"content": "最终答案", "finish_reason": "stop"}]},
    ])
    executor = ToolExecutor()

    async def search(q: str) -> str:
        return f"结果:{q}"
    executor.register(ToolDefinition(name="search", func=search, description="", parameters={}))

    cfg = _make_cfg(llm, executor)
    run_ctx = _run_ctx()
    state = LoopState(store=wrap_store([{"role": "user", "content": "问题"}], run_ctx))

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    assert outcome.final_text == "最终答案"

    # 核心断言:最终历史里的每一条消息,单独拿出来都必须能被
    # json.dumps——这是"能不能发给真实 provider"的最低要求。
    for i, m in enumerate(outcome.messages):
        try:
            json.dumps(m)
        except TypeError as e:
            raise AssertionError(
                f"messages[{i}] 不是 JSON 可序列化的: {m!r} ({e})"
            ) from e

    # 第二轮实际发出去的请求(llm.calls[1])里的 messages 同样要能序列化——
    # 这才是真正复现 bug 的那一刻:第二轮请求体构造时,第一轮产生的
    # assistant 消息必须已经是安全的
    json.dumps(llm.calls[1]["messages"])


async def test_terminal_tool_message_is_json_serializable():
    """finish 工具产生的 assistant 消息(带 tool_calls)同样要能序列化——
    虽然 finish 命中快捷终止路径、不会有"下一轮"把它发出去,但如果
    请求方(比如另一个 Agent 复用同一份历史)把这段历史继续往后传,
    同样不能崩。"""
    llm = StreamingFakeLLMClient([{"stream_deltas": [
        {"tool_call_delta": {"index": 0, "id": "c1", "name": "finish",
                             "arguments": '{"status":"ok","summary":"done","data":{"answer":"1"}}'}},
        {"finish_reason": "tool_calls"},
    ]}])
    executor = ToolExecutor()
    executor.register(build_finish_tool(_DummyData))
    cfg = _make_cfg(llm, executor)
    run_ctx = _run_ctx()
    state = LoopState(store=wrap_store([{"role": "user", "content": "问题"}], run_ctx))

    outcome = await run_to_outcome(cfg, state, run_ctx)

    assert outcome.status == "completed"
    for m in outcome.messages:
        json.dumps(m)