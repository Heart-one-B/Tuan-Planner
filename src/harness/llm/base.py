# harness/llm/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


class ContextOverflowError(Exception):
    pass


@dataclass
class NormalizedUsage:
    prompt_tokens: int
    completion_tokens: int
    source: str


@dataclass
class StreamDelta:
    """一次流式调用里,一个 chunk 归一化后的增量(第五刀,任务 5.1)。

    content/reasoning: 追加到累积文本的增量片段,None 表示这个 chunk
    没有携带这部分。

    tool_call_delta: 单个 tool_call 的增量分片,形如
      {"index": int, "id": str|None, "name": str|None, "arguments": str|None}
    index 是 provider 在这次响应里给这个 tool_call 分配的位置序号
    (不是全局唯一 id)。id 通常只在该 index 第一次出现的分片里非
    None,后续同一 index 的分片只补 name/arguments 的片段,消费方
    (StreamAccumulator)按 index 累加。

    finish_reason: 只在最后一个 chunk(或专门的收尾 chunk)上非
    None,值域和非流式路径一致("stop"/"length"/"tool_calls"/...)。

    usage: 只在支持 usage 上报的 provider 的收尾 chunk 上非 None
    (常见形态是一个 choices 为空、只带 usage 的收尾 chunk)。
    """
    content: str | None = None
    reasoning: str | None = None
    tool_call_delta: dict | None = None
    finish_reason: str | None = None
    usage: NormalizedUsage | None = None


class LLMClientBase(ABC):
    """Abstract interface for LLM clients.

    实现方必须履行的六条归一化契约(前五条见原有说明,第六条是
    第五刀新增):

      1-5. (不变,略——msg.reasoning/finish_reason/usage、
           ContextOverflowError、剥离 _hid 字段)

      6. stream() 的实现方(如果重写了默认实现)必须:
         - tool_call_delta 按 index 正确分片,不能假设同一个
           tool_call 的信息在单个 chunk 里到齐
         - finish_reason 可能出现在最后一条内容 chunk 上,也可能
           出现在单独一条 content/tool_call_delta 都为 None、专门
           收尾的 chunk 上,两种形态消费方(StreamAccumulator)都要
           扛得住
         - usage(如果 provider 支持)放在收尾 chunk 上,不得让
           估算值冒充真实值(source 如实标注)

    未履行契约同样不会报错,只会让能力静默失效——和前五条契约是
    同一条纪律。
    """

    @abstractmethod
    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        ...

    async def stream(self, trace_id: str, **kwargs) -> AsyncIterator[StreamDelta]:
        """默认实现:退化为一次非流式调用(内部调用 self.call()),
        把完整响应包成若干个 StreamDelta 一次性吐出——这是"新能力
        可选、不配置即无感"在客户端接口层面的落地。

        关键:任何现有 LLMClientBase 实现方(包括测试用的
        FakeLLMClient)只要不重写这个方法,循环改走 stream() 之后,
        行为和以前完全一样,只是没有"边收边吐"的体感,不需要跟着
        改一行代码。真正做流式的实现方(如 OpenAIClient)应该重写
        这个方法,履行上面的第六条契约。
        """
        resp = await self.call(trace_id=trace_id, stream=False, **kwargs)
        msg = resp.choices[0].message
        yield StreamDelta(
            content=msg.content,
            reasoning=getattr(msg, "reasoning", None),
            finish_reason=getattr(msg, "finish_reason", None),
            usage=getattr(msg, "usage", None),
        )
        for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            yield StreamDelta(tool_call_delta={
                "index": i, "id": tc.id,
                "name": tc.function.name, "arguments": tc.function.arguments,
            })