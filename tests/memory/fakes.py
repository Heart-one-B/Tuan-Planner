# tests/fakes.py
from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

from harness.llm.base import LLMClientBase, NormalizedUsage

_id_counter = itertools.count(1)


def _build_tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call_{next(_id_counter)}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


class FakeLLMClient(LLMClientBase):
    """LLMClientBase 的测试替身:响应按脚本顺序消费,每条脚本是一个 dict:
        {"content": str|None, "tool_calls": [(name, args_dict), ...]|None,
         "finish_reason": "stop"|"tool_calls"|"length", "prompt_tokens": int|None}
    不接任何真实网络。calls 属性记录每次调用收到的 messages,供断言
    "提取指令里是否真的带了去重信息/负面清单"这类内容性测试用。
    """

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[list] = []
        self.trace_ids: list[str] = []

    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        self.trace_ids.append(trace_id)
        # 记录浅拷贝而不是引用:AgentLoop 会在这次调用返回后继续原地
        # append 消息到同一个列表对象上(_PlainMessageStore.append),
        # 如果这里存的是引用,测试事后检查 self.calls[i] 时看到的会是
        # "整个 run 结束后"的画面,而不是"这次调用发生那一刻"的画面。
        self.calls.append(list(kwargs.get("messages") or []))
        if not self._responses:
            raise RuntimeError(
                f"FakeLLMClient: 脚本已耗尽(第 {len(self.calls)} 次调用时无更多可消费的响应)"
            )
        spec = self._responses.pop(0)

        content = spec.get("content")
        tool_calls_spec = spec.get("tool_calls")
        tool_calls = (
            [_build_tool_call(name, args) for name, args in tool_calls_spec]
            if tool_calls_spec else []
        )
        msg = SimpleNamespace(content=content, tool_calls=tool_calls)
        msg.finish_reason = spec.get("finish_reason", "tool_calls" if tool_calls else "stop")

        prompt_tokens = spec.get("prompt_tokens")
        if prompt_tokens is not None:
            msg.usage = NormalizedUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=spec.get("completion_tokens", 0),
                source="api_usage",
            )

        choice = SimpleNamespace(message=msg, finish_reason=msg.finish_reason)
        return SimpleNamespace(choices=[choice], usage=None)


class RaisingLLMClient(LLMClientBase):
    """恒定抛异常的测试替身,用于测熔断/失败路径。"""

    def __init__(self, exc: Exception | None = None):
        self.exc = exc or RuntimeError("模拟 LLM 调用失败")
        self.call_count = 0

    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        self.call_count += 1
        raise self.exc


def text_finish(text: str) -> dict:
    """快捷构造"模型直接以纯文本结束"的一条脚本响应。"""
    return {"content": text, "tool_calls": None, "finish_reason": "stop"}


def tool_call_then(name: str, args: dict) -> dict:
    """快捷构造"模型调用一个工具"的一条脚本响应。"""
    return {"content": None, "tool_calls": [(name, args)], "finish_reason": "tool_calls"}