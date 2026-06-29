import logging
import time

from openai import AsyncOpenAI

from harness.llm.base import LLMClientBase

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClientBase):
    """OpenAI-compatible LLM client.

    Works with any OpenAI-compatible endpoint (OpenAI, DashScope, etc.).
    Handles model injection, extra_body, and tracing on every non-stream call.

    Examples:
        # OpenAI
        client = OpenAIClient(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
        )

        # DashScope / Qwen
        client = OpenAIClient(
            api_key="...",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen-max",
            extra_body={"enable_thinking": False},
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

        response = await self._client.chat.completions.create(
            model=self.model_name,
            extra_body=self.extra_body,
            stream=stream,
            **kwargs,
        )

        duration = int((time.time() - start) * 1000)

        if not stream:
            msg = response.choices[0].message
            logger.debug(
                f"[OpenAIClient] done  duration={duration}ms  "
                f"has_tool_calls={bool(msg.tool_calls)}"
            )
            from harness.tracing import tracer
            tracer.record_llm_call(
                trace_id=trace_id,
                input_messages=kwargs.get("messages", []),
                output=msg.content or "",
                has_tool_calls=bool(msg.tool_calls),
                duration_ms=duration,
            )

        return response