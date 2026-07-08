# harness/llm/openai_client.py
import logging
import time

import openai
from openai import AsyncOpenAI

from harness.llm.base import LLMClientBase
from harness.llm.retry import retry_with_backoff

logger = logging.getLogger(__name__)


# ── 错误分类:瞬态(值得重试) vs 永久(重试无用,快速失败) ────────────────
# 与 tools/exceptions.py 的 RetryableError/ParamError 分类哲学同构。
_RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,   # 网络层面连不上
    openai.APITimeoutError,      # 请求超时
    openai.RateLimitError,       # 429——注:配额耗尽型 429 重试也无用,
                                  # 但暂不做精细区分,重试预算上限兜底,已知简化
    openai.InternalServerError,  # 5xx
)
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504, 529}   # 529 常见于部分兼容端点


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in _RETRYABLE_STATUS_CODES


class OpenAIClient(LLMClientBase):
    """OpenAI-compatible LLM client.

    重试(Phase 0 修复):
      SDK 自带重试被显式关闭(max_retries=0),统一由 retry_with_backoff
      负责——避免两层重试各自倒计时导致总延迟不可预测,且只有这一层的
      重试事件会被日志记录、可观测。只重试瞬态错误,永久性错误(400/401/
      403/404)直接快速失败,不浪费重试预算。

    Token 计数:优先读 response.usage,缺失时估算并标注来源。
    推理通道归一化:不同提供商的推理字段名统一抹平成 msg.reasoning。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        extra_body: dict = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
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

        response = await retry_with_backoff(
            fn=lambda: self._client.chat.completions.create(
                model=self.model_name, extra_body=self.extra_body, stream=stream, **kwargs,
            ),
            is_retryable=_is_retryable,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
        )

        duration = int((time.time() - start) * 1000)

        if not stream:
            msg = response.choices[0].message
            reasoning = self._extract_reasoning(msg)
            msg.reasoning = reasoning

            prompt_tokens, completion_tokens, token_source = self._extract_usage(
                response, kwargs.get("messages", []), msg.content or "",
            )

            logger.debug(
                f"[OpenAIClient] done  duration={duration}ms  "
                f"has_tool_calls={bool(msg.tool_calls)}  "
                f"tokens={prompt_tokens}+{completion_tokens}({token_source})"
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