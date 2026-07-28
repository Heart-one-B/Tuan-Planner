# harness/llm/openai_client.py
import logging
import time

import openai
from openai import AsyncOpenAI

from harness.llm.base import ContextOverflowError, LLMClientBase, NormalizedUsage, StreamDelta
from harness.llm.retry import retry_with_backoff
from harness.message_id import strip_hid

logger = logging.getLogger(__name__)


_RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504, 529}

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
        """【第五刀:不改动】这个方法及其 stream=True 分支保持原样——
        stream=True 时依然是不归一化的裸 chunk 流(见类前的 docstring
        补充说明)。harness 自己(loop.py)第五刀之后不再调用
        `.call(stream=True)`,统一走下面新增的 `.stream()`;这个方法
        的 stream=True 分支作为既有公开契约的一部分保留,不做破坏性
        删除,只是标注为"harness 内部不再走这条路的遗留分支"。
        """
        start = time.time()
        if "messages" in kwargs:
            kwargs = {**kwargs, "messages": self._sanitize_for_wire(kwargs["messages"])}

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

    async def stream(self, trace_id: str, **kwargs):
        """第五刀新增(任务 5.4 修订版)。真正的流式实现,履行
        LLMClientBase 归一化契约第六条。

        【如实记账】这个方法的逻辑基于 OpenAI SDK 文档描述的流式
        chunk 形状(chunk.choices[0].delta.content/tool_calls,
        chunk.usage 出现在 stream_options={"include_usage":True}
        时的收尾 chunk 上)写就,在本次交付环境里只做了 import/
        结构层面的验证——没有条件对接真实 OpenAI 兼容 API 做实际
        调用(网络策略不允许访问 api.openai.com)。落地前必须在你
        自己能访问真实 API 的环境里跑一次真实冒烟,这是设计方案
        "分五刀"里唯一明确要求、也是唯一没有被这次交付验证过的一步。
        """
        if "messages" in kwargs:
            kwargs = {**kwargs, "messages": self._sanitize_for_wire(kwargs["messages"])}

        try:
            response = await retry_with_backoff(
                fn=lambda: self._client.chat.completions.create(
                    model=self.model_name, extra_body=self.extra_body,
                    stream=True, stream_options={"include_usage": True}, **kwargs,
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

        start = time.time()
        content_acc = ""
        reasoning_acc = ""
        has_tool_calls = False
        captured_usage: NormalizedUsage | None = None

        async for chunk in response:
            if not chunk.choices:
                # 常见边界:最后一个 chunk 只带 usage、choices 为空
                # (Phase 1 真实踩过的坑,第五刀在流式路径上同样要扛住)
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    captured_usage = NormalizedUsage(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        source="api_usage",
                    )
                    yield StreamDelta(usage=captured_usage)
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = getattr(choice, "finish_reason", None)

            content = getattr(delta, "content", None)
            reasoning = None
            for attr in ("reasoning_content", "reasoning"):
                val = getattr(delta, attr, None)
                if val:
                    reasoning = val
                    break

            if content:
                content_acc += content
            if reasoning:
                reasoning_acc += reasoning

            tc_fragments = []
            tcs = getattr(delta, "tool_calls", None)
            if tcs:
                has_tool_calls = True
                # 【修复】原实现是 `tc = tcs[0]`,注释写着"每个 chunk 通常
                # 只带一个 tool_call 的分片"——"通常"两个字就是问题所在:
                # 并行工具调用(parallel tool calling)时,provider 完全
                # 可能在同一个 chunk 里塞多个 tool_call 的分片,取 [0]
                # 会把其余的**静默丢掉**,拼出来的 tool_calls 缺项或
                # 参数残缺,是数据损坏级的 bug,而且不报错。
                # Fake 测试测不出这个(Fake 每个 chunk 只发一个分片),
                # 只有真实 API 的并行调用才会暴露——与其等冒烟撞上,
                # 不如直接按"一个 chunk 可能有多个分片"来写。
                for tc in tcs:
                    fn = getattr(tc, "function", None)
                    tc_fragments.append({
                        "index": tc.index,
                        "id": getattr(tc, "id", None),
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", None) if fn else None,
                    })

            if not tc_fragments:
                yield StreamDelta(
                    content=content, reasoning=reasoning, finish_reason=finish_reason,
                )
            else:
                # content/reasoning 挂在第一个分片上、finish_reason 挂在
                # 最后一个分片上——保证这个 chunk 携带的每一样信息都
                # 恰好被下游 StreamAccumulator 消费一次,不重不漏。
                last = len(tc_fragments) - 1
                for i, frag in enumerate(tc_fragments):
                    yield StreamDelta(
                        content=content if i == 0 else None,
                        reasoning=reasoning if i == 0 else None,
                        tool_call_delta=frag,
                        finish_reason=finish_reason if i == last else None,
                    )

        duration = int((time.time() - start) * 1000)
        logger.debug(
            f"[OpenAIClient] stream done  duration={duration}ms  "
            f"has_tool_calls={has_tool_calls}  "
            f"usage={'api_usage' if captured_usage else 'estimated'}"
        )
        from harness.tracing import tracer
        tracer.record_llm_call(
            trace_id=trace_id,
            prompt_tokens=captured_usage.prompt_tokens if captured_usage else 0,
            completion_tokens=captured_usage.completion_tokens if captured_usage else 0,
            token_source=captured_usage.source if captured_usage else "estimated",
            output=content_acc, has_tool_calls=has_tool_calls, duration_ms=duration,
            reasoning=reasoning_acc or None,
        )

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
    def _sanitize_for_wire(messages: list) -> list:
        cleaned = []
        for m in messages:
            if isinstance(m, dict):
                if "reasoning_content" in m:
                    m = {k: v for k, v in m.items() if k != "reasoning_content"}
                m = strip_hid(m)
            cleaned.append(m)
        return cleaned