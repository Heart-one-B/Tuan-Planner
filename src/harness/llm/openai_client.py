# harness/llm/openai_client.py
import logging
import time

from openai import AsyncOpenAI

from harness.llm.base import LLMClientBase

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClientBase):
    """OpenAI-compatible LLM client.

    Works with any OpenAI-compatible endpoint (OpenAI, DashScope, etc.).
    Handles model injection, extra_body, tracing, and reasoning-channel
    normalization on every non-stream call.

    推理通道归一化:
      不同提供商把"思考过程"放在不同字段——DeepSeek/Qwen 用
      reasoning_content。这里统一抹平成 msg.reasoning 一个约定属性,
      AgentLoop 只认这一个名字,不需要知道背后是哪家提供商。
      回传规则同样在这里处理:DeepSeek 明确要求 reasoning_content
      不能被回传进下一轮 messages(会报错),_strip_reasoning_for_replay
      在构造下一轮请求前把它剥掉——这个坑不能让 AgentLoop 去踩。

    Examples:
        # OpenAI
        client = OpenAIClient(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
        )

        # DashScope / Qwen(开启 thinking 模式)
        client = OpenAIClient(
            api_key="...",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen-max",
            extra_body={"enable_thinking": True},
        )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        extra_body: dict = None,
    ):
        self.model_name = model_name
        self.extra_body = extra_body or {}
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        start = time.time()

        # 回传前剥掉上一轮可能残留的推理字段(DeepSeek 等提供商要求如此;
        # 对不需要剥的提供商这是无害的空操作)
        if "messages" in kwargs:
            kwargs = {**kwargs, "messages": self._strip_reasoning_for_replay(kwargs["messages"])}

        response = await self._client.chat.completions.create(
            model=self.model_name,
            extra_body=self.extra_body,
            stream=stream,
            **kwargs,
        )

        duration = int((time.time() - start) * 1000)

        if not stream:
            msg = response.choices[0].message
            reasoning = self._extract_reasoning(msg)
            # 归一化属性:AgentLoop 只读 msg.reasoning,不关心来源字段名
            msg.reasoning = reasoning

            logger.debug(
                f"[OpenAIClient] done  duration={duration}ms  "
                f"has_tool_calls={bool(msg.tool_calls)}  "
                f"has_reasoning={bool(reasoning)}"
            )
            from harness.tracing import tracer
            tracer.record_llm_call(
                trace_id=trace_id,
                input_messages=kwargs.get("messages", []),
                output=msg.content or "",
                has_tool_calls=bool(msg.tool_calls),
                duration_ms=duration,
                reasoning=reasoning,
            )

        return response

    # ── 推理通道归一化 ──────────────────────────────────────────────────

    @staticmethod
    def _extract_reasoning(msg) -> str | None:
        """各提供商推理字段名不一致,这里统一探测。
        目前覆盖:DeepSeek / Qwen(reasoning_content)。
        新增提供商在此扩展,不改调用方。"""
        for attr in ("reasoning_content", "reasoning"):
            val = getattr(msg, attr, None)
            if val:
                return val
        return None

    @staticmethod
    def _strip_reasoning_for_replay(messages: list) -> list:
        """构造下一轮请求前,剥掉历史消息里的推理字段。
        DeepSeek 明确要求 reasoning_content 不能随 assistant 消息回传,
        否则报错;这个提供商特定的坑收敛在客户端层,不泄漏给 AgentLoop。"""
        cleaned = []
        for m in messages:
            if isinstance(m, dict) and "reasoning_content" in m:
                m = {k: v for k, v in m.items() if k != "reasoning_content"}
            cleaned.append(m)
        return cleaned