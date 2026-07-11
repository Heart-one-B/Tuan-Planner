# harness/llm/openai_client.py
import logging
import time

import openai
from openai import AsyncOpenAI

from harness.llm.base import ContextOverflowError, LLMClientBase, NormalizedUsage
from harness.llm.retry import retry_with_backoff

logger = logging.getLogger(__name__)


_RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504, 529}

# 上下文超长的启发式判断:openai SDK 没有为此单独的异常类,
# 表现为 BadRequestError 带特定 code 或消息文本包含关键词。
# 这是最佳努力检测,不同 provider/未来 SDK 版本可能变化——
# 已知局限,不保证覆盖所有 provider 的溢出错误形态。
_OVERFLOW_ERROR_CODES = {"context_length_exceeded"}
_OVERFLOW_MESSAGE_HINTS = ("maximum context length", "context_length_exceeded")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in _RETRYABLE_STATUS_CODES


def _is_overflow(exc: Exception) -> bool:
    if not isinstance(exc, openai.BadRequestError):
        return False
    code = getattr(exc, "code", None)
    if code in _OVERFLOW_ERROR_CODES:
        return True
    msg = str(exc).lower()
    return any(hint in msg for hint in _OVERFLOW_MESSAGE_HINTS)


class OpenAIClient(LLMClientBase):
    def __init__(self, api_key: str, base_url: str, model_name: str,
                extra_body: dict = None, max_retries: int = 5,
                base_delay: float = 1.0, max_delay: float = 30.0):
        self.model_name = model_name
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        start = time.time()
        if "messages" in kwargs:
            kwargs = {**kwargs, "messages": self._strip_reasoning_for_replay(kwargs["messages"])}

        try:
            response = await retry_with_backoff(
                fn=lambda: self._client.chat.completions.create(
                    model=self.model_name, extra_body=self.extra_body, stream=stream, **kwargs,
                ),
                is_retryable=_is_retryable,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                max_delay=self.max_delay,
            )
        except Exception as e:
            if _is_overflow(e):
                raise ContextOverflowError(str(e)) from e
            raise

        duration = int((time.time() - start) * 1000)

        if not stream:
            msg = response.choices[0].message
            reasoning = self._extract_reasoning(msg)
            msg.reasoning = reasoning
            msg.finish_reason = getattr(response.choices[0], "finish_reason", None)

            prompt_tokens, completion_tokens, token_source = self._extract_usage(
                response, kwargs.get("messages", []), msg.content or "",
            )
            msg.usage = NormalizedUsage(prompt_tokens, completion_tokens, token_source)

            logger.debug(
                f"[OpenAIClient] done  duration={duration}ms  "
                f"has_tool_calls={bool(msg.tool_calls)}  "
                f"tokens={prompt_tokens}+{completion_tokens}({token_source})  "
                f"finish_reason={msg.finish_reason}"
            )
            from harness.tracing import tracer
            tracer.record_llm_call(
                trace_id=trace_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                token_source=token_source,
                output=msg.content or "",
                has_tool_calls=bool(msg.tool_calls),
                duration_ms=duration,
                reasoning=reasoning,
            )

        return response

    @staticmethod
    def _extract_usage(response, input_messages: list, output_text: str) -> tuple[int, int, str]:
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            if prompt is not None and completion is not None:
                return prompt, completion, "api_usage"
        prompt_chars = sum(
            len(m["content"]) if isinstance(m, dict) and isinstance(m.get("content"), str)
            else len(m.content or "") if hasattr(m, "content") and isinstance(m.content, str)
            else 0
            for m in input_messages
        )
        return prompt_chars, len(output_text), "estimated"

    @staticmethod
    def _extract_reasoning(msg) -> str | None:
        for attr in ("reasoning_content", "reasoning"):
            val = getattr(msg, attr, None)
            if val:
                return val
        return None

    @staticmethod
    def _strip_reasoning_for_replay(messages: list) -> list:
        cleaned = []
        for m in messages:
            if isinstance(m, dict) and "reasoning_content" in m:
                m = {k: v for k, v in m.items() if k != "reasoning_content"}
            cleaned.append(m)
        return cleaned