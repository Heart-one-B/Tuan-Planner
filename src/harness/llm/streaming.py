# harness/llm/streaming.py
from __future__ import annotations

from types import SimpleNamespace

from harness.llm.base import NormalizedUsage, StreamDelta


class StreamAccumulator:
    """消费 StreamDelta 流,攒出一个满足 LLMClientBase 五条归一化
    契约(reasoning/finish_reason/usage/tool_calls)的完整 message
    对象——循环拿到的东西和非流式路径完全一样,下游(权限门控/
    finish_tool/offload/normalize_message)不需要知道背后是不是
    流式。这是"归一化责任留在 llm 层,不漏给 loop"这条设计原则
    (设计方案 §5.4)的直接落地。

    usage 字段刻意不在"从未收到过"时伪造一个全零的 NormalizedUsage——
    build_message() 让它保持 None,调用方(_call_model)据此判断
    "这次没有真实用量可记",不会拿一个假的 0 去污染 TokenCounter
    (那样比"没有读数、继续估算"更糟,是伪造的确定性)。
    """

    def __init__(self):
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._tool_calls: dict[int, dict] = {}   # index -> {"id","name","arguments"}
        self._finish_reason: str | None = None
        self._usage: NormalizedUsage | None = None

    def feed(self, delta: StreamDelta) -> None:
        if delta.content:
            self._content_parts.append(delta.content)
        if delta.reasoning:
            self._reasoning_parts.append(delta.reasoning)
        if delta.finish_reason is not None:
            self._finish_reason = delta.finish_reason
        if delta.usage is not None:
            self._usage = delta.usage
        if delta.tool_call_delta is not None:
            td = delta.tool_call_delta
            idx = td["index"]
            entry = self._tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
            if td.get("id"):
                entry["id"] = td["id"]
            if td.get("name"):
                entry["name"] = (entry["name"] or "") + td["name"]
            if td.get("arguments"):
                entry["arguments"] = (entry["arguments"] or "") + td["arguments"]

    def build_message(self):
        content = "".join(self._content_parts) or None
        reasoning = "".join(self._reasoning_parts) or None
        tool_calls = None
        if self._tool_calls:
            tool_calls = [
                SimpleNamespace(
                    id=self._tool_calls[i]["id"],
                    function=SimpleNamespace(
                        name=self._tool_calls[i]["name"],
                        arguments=self._tool_calls[i]["arguments"],
                    ),
                )
                for i in sorted(self._tool_calls)
            ]
        return SimpleNamespace(
            role="assistant",
            content=content, reasoning=reasoning, tool_calls=tool_calls,
            finish_reason=self._finish_reason, usage=self._usage,
        )