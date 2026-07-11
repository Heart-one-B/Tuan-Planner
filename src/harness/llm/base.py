from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ContextOverflowError(Exception):
    """provider 报告"这次请求的 prompt 超过了模型能接受的最大长度"时,
    实现方必须捕获底层 provider 特定的错误信号,归一化抛出这个类型。
    AgentLoop 只认这一个异常类型来触发紧急压缩,不认任何 provider 的
    原始错误码/消息文本——这和 reasoning/usage 归一化是同一条原则。"""


@dataclass
class NormalizedUsage:
    """一次调用的 token 用量,归一化后的三元组。"""
    prompt_tokens: int
    completion_tokens: int
    source: str   # "api_usage" | "estimated" —— 与 tracing 的约定一致


class LLMClientBase(ABC):
    """Abstract interface for LLM clients.

    Implement this to swap providers (OpenAI, Anthropic, local, mock).
    The harness only depends on this interface — never on a concrete client.

    实现方必须履行的四条归一化契约(AgentLoop 依赖它们,且只认这四个
    约定名字,不知道、也不该知道背后是哪家 provider):

      1. msg.reasoning: str | None
         各 provider 的推理字段名不同(如 DeepSeek/Qwen 的 reasoning_content),
         调用方在返回前统一写到这个属性上。没有独立推理通道则设为 None。

      2. msg.finish_reason: str | None
         用 OpenAI 的词表("stop" | "length" | "tool_calls" |
         "content_filter" | 其他)。AgentLoop 用它判断一段纯文本回复
         是模型主动结束,还是被 max_tokens 腰斩(finish_reason=="length"
         时绝不能当作正常完成处理)。

      3. msg.usage: NormalizedUsage
         优先用 provider 返回的真实用量;拿不到时退回估算,
         source 如实标注 "estimated",不得让估算值冒充真实值。

      4. ContextOverflowError
         provider 报告"prompt 超过模型最大长度"时(通常是一种特定的
         400 类错误,不同 provider 错误码不同),必须捕获后归一化
         抛出本模块的 ContextOverflowError,不能让原始异常类型泄漏
         给 AgentLoop——AgentLoop 靠这个类型触发紧急压缩重试。

    未履行契约的实现方不会报错,只会让对应能力静默失效(如 thinking
    事件不出现、溢出不触发紧急压缩)——这是 Phase 1 测试 FakeLLMClient
    时实际踩过的坑,写在这里防止第二次踩。
    """

    @abstractmethod
    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        """Make a chat completion call.

        Args:
            trace_id: Correlates this call in the tracing layer.
            stream:   If True, return an async iterable stream instead of
                      a full response object.
            **kwargs: Passed through to the underlying API
                      (messages, tools, temperature, etc.).

        Returns:
            OpenAI-SDK-compatible response object with a .choices attribute;
            .choices[0].message 满足上述四条归一化契约。
        """
        ...