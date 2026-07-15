# harness/context/token_counter.py
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 估算系数:每 token 折算多少字符。取 2(偏保守)而不是中英混合更常见的 2.5-3——
# 保守 = 高估占用 = 更早触发压缩。两种误差的代价不对称:
# 高估的代价是压缩略微提前(多花一点压缩成本);低估的代价是撞墙报错、
# 循环炸穿。宁可付前者。
_CHARS_PER_TOKEN = 2


def estimate_tokens(text: str) -> int:
    """字符数粗估 token 数,刻意保守(高估)。仅作兜底,任何有 API usage
    可用的场合都不该用它。"""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list) -> int:
    """整个消息列表的粗估。覆盖 dict 消息和 SDK 消息对象两种形态
    (与 tracer 旧实现的取值逻辑一致),tool_calls 的 JSON 参数也计入——
    它们同样占窗口,漏掉会系统性低估。"""
    total = 0
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content")
            if isinstance(content, str):
                total += estimate_tokens(content)
            for tc in (m.get("tool_calls") or []):  # 审查修复:dict 形态的
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}  # tool_calls 此前漏计,
                total += estimate_tokens(str(fn.get("arguments", "")))  # 快照恢复统一 dict 后
        else:  # 这将是主路径
            content = getattr(m, "content", None)
            if isinstance(content, str):
                total += estimate_tokens(content)
            for tc in (getattr(m, "tool_calls", None) or []):
                args = getattr(getattr(tc, "function", None), "arguments", "") or ""
                total += estimate_tokens(str(args))
    return total


class TokenCounter:
    """当前窗口占用的追踪器。

    主数据源是每轮 LLM 调用返回的 API usage(Phase 0① 铺的路):
    上一轮的 prompt_tokens 就是"发出去的完整历史"的精确占用读数,
    不需要自己反复全量估算。估算只在两个场合兜底:
      1. 冷启动——还没发出过任何调用,只能估
      2. 增量——上次 API 读数之后又 append 的消息,在下次读数刷新前先估上

    source 属性如实标注当前数字的可信度,和 tracing 的 token_source
    是同一套约定(api_usage / estimated)。
    """

    def __init__(self):
        self._last_api_prompt_tokens: int | None = None
        self._pending_estimate: int = 0     # 上次 API 读数之后新增消息的估算量

    def note_api_usage(self, prompt_tokens: int) -> None:
        """每轮 LLM 调用返回后调用:用真实读数刷新,清空增量估算。
        prompt_tokens 覆盖的是这次调用发出的完整历史,所以旧的增量
        已经被包含在内,归零。"""
        self._last_api_prompt_tokens = prompt_tokens
        self._pending_estimate = 0

    def note_appended(self, text: str) -> None:
        """API 读数之后又有消息进入历史(工具结果回填等),先估上。"""
        self._pending_estimate += estimate_tokens(text)

    def note_appended_tokens(self, tokens: int) -> None:
        """按已估算好的 token 数记增量。与 note_appended(text) 并存:
        调用方能拿到完整消息对象时用这个(口径与 estimate_messages_tokens
        一致),只有纯文本时用旧的。"""
        self._pending_estimate += tokens

    def reset(self, messages: list | None = None) -> None:
        """历史被整体替换(压缩后)时调用:旧读数全部作废,按新历史重新估。"""
        self._last_api_prompt_tokens = None
        self._pending_estimate = estimate_messages_tokens(messages or [])

    @property
    def current_tokens(self) -> int:
        if self._last_api_prompt_tokens is not None:
            return self._last_api_prompt_tokens + self._pending_estimate
        return self._pending_estimate

    @property
    def source(self) -> str:
        return "api_usage" if self._last_api_prompt_tokens is not None else "estimated"