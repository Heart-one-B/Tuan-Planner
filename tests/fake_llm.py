# tests/fake_llm.py
"""
FakeLLMClient —— 不调真实 API 的可脚本化 LLM 客户端。

用法:预先编排好每一轮模型"该说什么",FakeLLMClient 按顺序吐出来。
这样可以在没有 API key、不产生网络调用的情况下,把 AgentLoop 完整跑通,
验证的是"代码的行为",不是"我口头声称的行为"。

设计上刻意模拟 OpenAI SDK 的响应对象形状(resp.choices[0].message.xxx),
这样 AgentLoop / ToolExecutor 不需要知道自己面对的是假客户端还是真客户端。

msg.reasoning / msg.finish_reason / msg.usage 都直接写归一化后的属性
(而不是某 provider 的原始字段名)——FakeLLMClient 扮演的是一个合规的
LLMClientBase 实现,四条契约本该在客户端层履行完毕。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from harness.llm.base import ContextOverflowError, LLMClientBase, NormalizedUsage


def tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


def text_message(content: str, reasoning: str | None = None,
                 finish_reason: str = "stop",
                 usage: tuple[int, int] | None = None) -> SimpleNamespace:
    """usage=(prompt_tokens, completion_tokens) 时附带归一化用量,
    供测试 AgentLoop 的 note_api_usage 通路。"""
    msg = SimpleNamespace(content=content, tool_calls=[], finish_reason=finish_reason)
    if reasoning:
        msg.reasoning = reasoning
    if usage:
        msg.usage = NormalizedUsage(usage[0], usage[1], "api_usage")
    return msg


def tool_call_message(tool_calls: list[SimpleNamespace], reasoning: str | None = None,
                      usage: tuple[int, int] | None = None) -> SimpleNamespace:
    msg = SimpleNamespace(content=None, tool_calls=tool_calls, finish_reason="tool_calls")
    if reasoning:
        msg.reasoning = reasoning
    if usage:
        msg.usage = NormalizedUsage(usage[0], usage[1], "api_usage")
    return msg


class Overflow:
    """脚本项占位符:放进 script 列表表示"这一次调用应抛 ContextOverflowError"。"""
    pass


class FakeLLMClient(LLMClientBase):
    def __init__(self, script: list, stream_script: list[str] | None = None):
        self._script = list(script)
        self._stream_script = list(stream_script or [])
        self.calls: list[dict] = []

    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        self.calls.append({
            "trace_id": trace_id,
            "stream": stream,
            "messages": [dict(m) if isinstance(m, dict) else m for m in kwargs.get("messages", [])],
        })

        if stream:
            if not self._stream_script:
                raise AssertionError("FakeLLMClient: stream_script 已耗尽")
            text = self._stream_script.pop(0)
            return self._make_stream(text)

        if not self._script:
            raise AssertionError(
                "FakeLLMClient: script 已耗尽,但循环又发起了一次非流式调用——"
                "说明被测代码比预期多转了一轮,检查终止/预算逻辑。"
            )
        item = self._script.pop(0)
        if isinstance(item, Overflow):
            raise ContextOverflowError("模拟:prompt 超过模型最大长度")
        return SimpleNamespace(choices=[SimpleNamespace(message=item, finish_reason=item.finish_reason)])

    @staticmethod
    async def _make_stream(text: str):
        for ch in text:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))])