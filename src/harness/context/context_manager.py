# harness/context/context_manager.py
from __future__ import annotations

import logging
from dataclasses import dataclass

from harness.context.budget import ContextBudget
from harness.context.compactor import CompactionOutcome, Compactor
from harness.context.models import ContextSnapshot
from harness.context.offload import OffloadStore, clear_stale_tool_results, offload_if_oversized
from harness.context.token_counter import TokenCounter, estimate_tokens
from harness.llm.base import LLMClientBase

logger = logging.getLogger(__name__)


@dataclass
class ContextManagerConfig:
    """组装 ContextManager 所需的三样东西,打包成一个可传递的配置对象。
    Agent 拿它作为可选构造参数,每次 run() 据此现造一个新的
    ContextManager 实例(不是复用同一个)——原因见 ContextManager 的
    docstring:它持有可变会话状态,必须像 RunContext 一样"每次运行现造",
    否则并发调用会重犯 Phase 1 已经修过的状态串台问题。
    """
    budget: ContextBudget
    offload_store: OffloadStore
    compactor: Compactor

    def build(self) -> "ContextManager":
        return ContextManager(self.budget, self.offload_store, self.compactor)


def _message_text(message) -> str:
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


class ContextManager:
    """上下文管理的编排入口:预算记账 → 卸载 → 压缩,三层按便宜优先的
    顺序串起来。

    通用性边界(刻意维护):本类不 import 任何 harness.agent 下的东西,
    不知道 AgentLoop、RunContext、TerminationPolicy 的存在。它只认
    "messages 列表 + trace_id + llm_client"这几个最朴素的概念——
    任何循环实现(不只 AgentLoop,未来 DagScheduler 如果解冻也能直接用)
    拿它管自己的消息历史,零改动。

    对外方法名刻意与"什么都不做的哑实现"保持一致(见 loop.py 里的
    _PlainMessageStore),这样 AgentLoop 可以对着同一个接口编程,
    完全不需要"有没有配置 ContextManager"的分支判断。

    每次运行必须现造一个新实例(通过 ContextManagerConfig.build()),
    不可跨 run() 复用——它持有的 _messages/counter 是这次运行独有的
    可变状态。
    """

    def __init__(self, budget: ContextBudget, offload_store: OffloadStore, compactor: Compactor):
        self.budget = budget
        self.offload_store = offload_store
        self.compactor = compactor
        self.counter = TokenCounter()
        self._messages: list = []
        self._prefix_len: int = 0
        self.compaction_history: list = []
        self.offload_records: list = []

    # ── 初始化 ──────────────────────────────────────────────────────────

    def init(self, messages: list, prefix_len: int) -> None:
        """开始一次运行时调用一次。prefix_len 由调用方显式声明——
        种子消息(system+history+task)是连续的前缀,调用方(如 Agent)
        构造种子列表时就知道长度,不需要本层去猜。"""
        self._messages = list(messages)
        self._prefix_len = prefix_len
        self.counter.reset(self._messages)

    @property
    def messages(self) -> list:
        return self._messages

    # ── 追加消息(第③④⑤阶段用) ───────────────────────────────────────

    def append(self, message) -> None:
        self._messages.append(message)
        self.counter.note_appended(_message_text(message))

    def note_api_usage(self, prompt_tokens: int) -> None:
        self.counter.note_api_usage(prompt_tokens)

    # ── 第1层:卸载(管"太大") ────────────────────────────────────────

    def offload_tool_result(
        self, trace_id: str, tool_call_id: str, content: str,
        max_chars: int | None = None,
    ) -> str:
        effective_max = max_chars if max_chars is not None else self.budget.default_tool_result_max_chars
        text, record = offload_if_oversized(
            content, self.offload_store, trace_id, tool_call_id,
            effective_max, self.budget.offload_preview_chars,
        )
        if record:
            self.offload_records.append(record)
        return text

    # ── 第0+1层的便宜清理(管"太旧") ──────────────────────────────────

    def _clear_stale(self, trace_id: str) -> int:
        cleared = clear_stale_tool_results(
            self._messages, self.budget.tool_result_clear_after_rounds,
            store=self.offload_store, trace_id=trace_id,   # 换页而非销毁
        )
        if cleared:
            # 消息内容被就地修改,旧的 API usage 读数已不能反映当前
            # 消息列表的真实占用,必须失效并重新估算——这不是 bug,
            # 是"任何改动 messages 内容的操作之后都要 reset"这条
            # 规则的直接体现,to_snapshot/compact 之后同理。
            self.counter.reset(self._messages)
        return cleared

    # ── 第2层:LLM 压缩 ─────────────────────────────────────────────────

    async def maybe_compact(
        self,
        llm_client: LLMClientBase,
        trace_id: str,
        trigger: str = "threshold",
        focus: str | None = None,
    ) -> CompactionOutcome | None:
        """threshold:先查是否需要,不需要则不动;需要则先做免费的
        沉底清理,清完重新判断,还不够才上 LLM 压缩(便宜优先)。
        overflow:已经撞墙的紧急调用,跳过判断直接压(此刻判断"要不要"
        已经没有意义——不压就是死)。
        manual:用户显式要求,同样直接压。
        """
        if trigger == "threshold":
            if not self.budget.should_compact(self.counter.current_tokens):
                return None
            self._clear_stale(trace_id)
            if not self.budget.should_compact(self.counter.current_tokens):
                logger.info("[ContextManager] 沉底清理已经足够,跳过 LLM 压缩")
                return None

        outcome = await self.compactor.compact(
            self._messages, fallback_client=llm_client, trace_id=trace_id,
            trigger=trigger, focus=focus,
            keep_recent_rounds=self.budget.keep_recent_rounds,
            prefix_len=self._prefix_len,
            offload_store=self.offload_store,             # 可恢复压缩
            fit_within=self.budget.effective_window,      # overflow一档瘦身后的容纳判据
        )
        self._messages = outcome.new_messages
        self.compaction_history.append(outcome.result)
        self.counter.reset(self._messages)   # 历史被整体替换,旧读数作废
        return outcome

    # ── 快照(Phase 3 挂载点) ────────────────────────────────────────────

    def to_snapshot(self) -> ContextSnapshot:
        return ContextSnapshot(
            messages=list(self._messages),
            compaction_history=list(self.compaction_history),
            offload_records=list(self.offload_records),
            total_tokens=self.counter.current_tokens,
            token_source=self.counter.source,
        )

    @classmethod
    def from_snapshot(
        cls, snapshot: ContextSnapshot, budget: ContextBudget,
        offload_store: OffloadStore, compactor: Compactor, prefix_len: int,
    ) -> "ContextManager":
        """从快照恢复。注意:跨进程恢复后 API usage 的连续性无法保证
        (上次的真实读数属于上个进程的一次调用),counter 统一重新估算,
        source 会是 estimated 直到下一次真实 API 调用刷新——这是
        "任何时候不确定就保守估算"原则的自然结果,不是缺陷。"""
        cm = cls(budget, offload_store, compactor)
        cm._messages = list(snapshot.messages)
        cm._prefix_len = prefix_len
        cm.compaction_history = list(snapshot.compaction_history)
        cm.offload_records = list(snapshot.offload_records)
        cm.counter.reset(cm._messages)
        return cm