# harness/agent/loop.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Literal, Protocol

from harness.agent.run_context import RunContext
from harness.agent.termination import Finish, Nudge, Reject, TerminationPolicy
from harness.llm.base import ContextOverflowError, LLMClientBase
from harness.tools.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


# ── 消息存储接口:ContextManager 和"什么都不做的哑实现"都满足它 ──────────
# AgentLoop 永远只对着这个接口编程,不关心背后是否配置了上下文管理——
# 消灭了六阶段骨架里到处出现 "if context_manager: ... else: ..." 分支的可能。
class MessageStore(Protocol):
    @property
    def messages(self) -> list: ...
    def append(self, message) -> None: ...
    def note_api_usage(self, prompt_tokens: int) -> None: ...
    def offload_tool_result(self, trace_id: str, tool_call_id: str,
                            content: str, max_chars: int | None = None) -> str: ...
    async def maybe_compact(self, llm_client, trace_id: str,
                            trigger: str = "threshold", focus: str | None = None): ...


class _PlainMessageStore:
    """未配置 ContextManager 时的哑实现:裸列表,上下文管理相关方法全部
    是空操作。这就是"新能力必须可选、不配置即无感"的具体落实——
    现有不使用 Phase 2 能力的调用方,行为与 Phase 1 完全一致。"""

    def __init__(self, messages: list):
        self._messages = messages

    @property
    def messages(self) -> list:
        return self._messages

    def append(self, message) -> None:
        self._messages.append(message)

    def note_api_usage(self, prompt_tokens: int) -> None:
        pass

    def offload_tool_result(self, trace_id, tool_call_id, content, max_chars=None) -> str:
        return content

    async def maybe_compact(self, llm_client, trace_id, trigger="threshold", focus=None):
        return None


def _wrap_store(messages: list, run_ctx: RunContext) -> MessageStore:
    if run_ctx.context_manager is not None:
        run_ctx.context_manager.init(messages, prefix_len=len(messages))
        return run_ctx.context_manager
    return _PlainMessageStore(messages)


# ── 预算策略 ────────────────────────────────────────────────────────────
@dataclass
class Budget:
    max_tool_calls: int = 10
    max_rounds: int = 24
    exhausted_action: Literal["stop", "force_answer", "force_finish"] = "force_answer"
    exhausted_prompt: str = (
        "你已用完工具调用次数,请根据已有信息尽力完成任务;"
        "信息不足请如实说明缺少什么,不要编造。"
    )


# ── 循环产出 ────────────────────────────────────────────────────────────
@dataclass
class LoopOutcome:
    """
    status:
      completed  终止策略正常判定结束
      exhausted  预算耗尽后收口
      overflow   上下文窗口溢出且紧急压缩后仍无法完成(第三值,Phase 2 新增)——
                 与 exhausted 性质不同:exhausted 是"模型没在预算内做完",
                 overflow 是"任务规模超出单次窗口的物理承载能力",
                 上层的正确应对不同(前者可调预算重跑,后者需要拆分任务),
                 必须如实区分,不能合并成同一个值糊弄过去。
    """
    status: Literal["completed", "exhausted", "overflow"]
    final_text: str = ""
    result: Any = None
    rounds: int = 0
    tool_calls_used: int = 0
    messages: list = field(default_factory=list)


class AgentLoop:
    """Agent 循环内核。

    六阶段骨架(Phase 2 后,插槽全部兑现):
      ① 组装上下文      —— messages = store.messages(store 可能是 ContextManager)
      ② 调模型          —— 溢出反应:捕获 ContextOverflowError,紧急压缩重试一次
      ③ 解读响应        —— thinking 事件;finish_reason=="length" 时不当作完成
      ④ 分发工具        —— [权限门控插槽:Phase 6]
      ⑤ 回填结果        —— 过 store.offload_tool_result 精简超大结果
      ⑥ 预算/溢出检查   —— 每轮调用后 note_api_usage + maybe_compact(threshold)
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        tool_executor: ToolExecutor,
        termination: TerminationPolicy,
        budget: Budget | None = None,
    ):
        self.llm = llm_client
        self.tool_executor = tool_executor
        self.termination = termination
        self.budget = budget or Budget()

    async def run(
        self, messages: list[dict], run_ctx: RunContext
    ) -> AsyncGenerator[dict, None]:
        span = run_ctx.span
        store = _wrap_store(messages, run_ctx)
        tools = self.tool_executor.schemas + self.termination.extra_tools()
        rounds = 0
        tool_calls_used = 0
        overflow_recovered = False   # 溢出重试只给一次机会,不无限烧压缩预算

        try:
            while True:
                rounds += 1

                if rounds > self.budget.max_rounds:
                    async for ev in self._exhaust(store, run_ctx, rounds, tool_calls_used):
                        yield ev
                    return

                # ① 组装上下文
                # ② 调模型(溢出反应路径)
                try:
                    resp = await self.llm.call(
                        trace_id=span.trace_id, messages=store.messages, tools=tools,
                    )
                except ContextOverflowError as e:
                    if overflow_recovered:
                        logger.error(f"[AgentLoop] 紧急压缩后仍溢出,终止: {e}")
                        yield self._overflow_outcome(store, rounds, tool_calls_used, span)
                        return
                    logger.warning(f"[AgentLoop] 上下文溢出,触发紧急压缩: {e}")
                    outcome = await store.maybe_compact(self.llm, span.trace_id, trigger="overflow")
                    overflow_recovered = True
                    if outcome and outcome.user_notice:
                        yield {"type": "notice", "content": outcome.user_notice}
                    if outcome:
                        yield {"type": "context_compacted", "trigger": "overflow",
                               "tokens_before": outcome.result.tokens_before,
                               "tokens_after": outcome.result.tokens_after,
                               "degraded": outcome.degraded}
                    continue   # 用压缩后的上下文重试这一轮

                msg = resp.choices[0].message
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    store.note_api_usage(usage.prompt_tokens)

                # ③ 解读响应
                thinking = getattr(msg, "reasoning", None) or (
                    msg.content if msg.tool_calls else None
                )
                if thinking:
                    yield {"type": "thinking", "content": thinking}

                finish_reason = getattr(msg, "finish_reason", None)
                if not msg.tool_calls:
                    if finish_reason == "length":
                        # 被 max_tokens 腰斩,不是模型主动结束——保留已生成的
                        # 部分内容,催促继续,不当作 Finish 处理
                        logger.warning("[AgentLoop] 响应被截断(finish_reason=length),催促续写")
                        store.append(msg)
                        store.append({
                            "role": "user",
                            "content": "你上一条回复因达到长度限制被截断,请继续完成刚才未说完的内容。",
                        })
                        continue

                    decision = self.termination.on_plain_message(msg)
                    if isinstance(decision, Finish):
                        # 审查修复(真实LLM冒烟发现):此前这条最终文本消息
                        # 从未 append 进 store 就直接返回了——outcome.messages
                        # /快照里因此永远缺失"模型自己说出的结论",只有走
                        # FinishToolTermination(调用finish工具收尾)才会
                        # 因为工具调用阶段已 append 而侥幸完整。续跑场景下
                        # 这正是最需要保留的一句话:模型刚得出的结论。
                        store.append(msg)
                        for ev in self._finish_events(decision):
                            yield ev
                        yield self._outcome(
                            "completed", decision, rounds, tool_calls_used, store.messages, span,
                        )
                        return
                    store.append({"role": "user", "content": decision.message})
                    continue

                # ④⑤ 工具调用
                store.append(msg)
                normal_calls = [tc for tc in msg.tool_calls if not self.termination.owns(tc)]
                terminal_calls = [tc for tc in msg.tool_calls if self.termination.owns(tc)]

                for tc in normal_calls:
                    tool_calls_used += 1
                    yield {"type": "tool_start", "tool": tc.function.name,
                           "args": tc.function.arguments}
                    result = await self.tool_executor.execute(
                        tc, trace_id=span.trace_id, run_ctx=run_ctx,
                    )
                    yield {"type": "tool_end", "tool": tc.function.name, "result": result}
                    # ⑤ 回填前过卸载(单工具阈值优先,None落回全局默认)
                    text = store.offload_tool_result(
                        span.trace_id, tc.id, str(result),
                        max_chars=self.tool_executor.max_result_chars_for(tc.function.name),
                    )
                    store.append({"role": "tool", "tool_call_id": tc.id, "content": text})

                finished: Finish | None = None
                for tc in terminal_calls:
                    decision = self.termination.intercept(tc, forced=False)
                    if isinstance(decision, Finish):
                        finished = decision
                        break
                    store.append({
                        "role": "tool", "tool_call_id": decision.tool_call_id,
                        "content": decision.message,
                    })

                if finished is not None:
                    for ev in self._finish_events(finished):
                        yield ev
                    yield self._outcome(
                        "completed", finished, rounds, tool_calls_used, store.messages, span,
                    )
                    return

                # ⑥ 预算检查 + 主动压缩检查(便宜优先,内部自行判断要不要动)
                if tool_calls_used >= self.budget.max_tool_calls:
                    async for ev in self._exhaust(store, run_ctx, rounds, tool_calls_used):
                        yield ev
                    return

                outcome = await store.maybe_compact(self.llm, span.trace_id, trigger="threshold")
                if outcome:
                    if outcome.user_notice:
                        yield {"type": "notice", "content": outcome.user_notice}
                    yield {"type": "context_compacted", "trigger": "threshold",
                           "tokens_before": outcome.result.tokens_before,
                           "tokens_after": outcome.result.tokens_after,
                           "degraded": outcome.degraded}

        except Exception as e:
            logger.error(f"[AgentLoop] error: {e}", exc_info=True)
            span.end("", status="error")
            raise

    # ── 预算耗尽收口 ────────────────────────────────────────────────────
    async def _exhaust(
        self, store: MessageStore, run_ctx: RunContext, rounds: int, used: int
    ) -> AsyncGenerator[dict, None]:
        span = run_ctx.span
        action = self.budget.exhausted_action
        logger.warning(f"[AgentLoop] budget exhausted (action={action}, "
                       f"rounds={rounds}, tool_calls={used})")

        if action == "stop":
            yield self._outcome("exhausted", Finish(), rounds, used, store.messages, span,
                                trace_status="timeout")
            return

        store.append({"role": "user", "content": self.budget.exhausted_prompt})

        if action == "force_answer":
            final_text = ""
            stream = await self.llm.call(
                trace_id=span.trace_id, messages=store.messages, stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    final_text += content
                    yield {"type": "token", "content": content}
            # 同一类修复:流式拼出的最终答案同样要入账,否则预算耗尽路径下
            # 快照/续跑一样会丢失模型的最终结论。
            if final_text:
                store.append({"role": "assistant", "content": final_text})
            yield self._outcome("exhausted", Finish(final_text=final_text),
                                rounds, used, store.messages, span, trace_status="timeout")
            return

        tools = self.tool_executor.schemas + self.termination.extra_tools()
        resp = await self.llm.call(trace_id=span.trace_id, messages=store.messages, tools=tools)
        msg = resp.choices[0].message
        finished: Finish | None = None
        for tc in (msg.tool_calls or []):
            if self.termination.owns(tc):
                decision = self.termination.intercept(tc, forced=True)
                if isinstance(decision, Finish):
                    finished = decision
                break
        yield self._outcome("exhausted", finished or Finish(),
                            rounds, used, store.messages, span, trace_status="timeout")

    # ── 内部工具 ────────────────────────────────────────────────────────
    @staticmethod
    def _finish_events(decision: Finish):
        for char in decision.final_text or "":
            yield {"type": "token", "content": char}

    @staticmethod
    def _outcome(status, decision: Finish, rounds, used, messages, span,
                 trace_status: str = "success") -> dict:
        span.end(decision.final_text, status=trace_status)
        return {"type": "outcome", "outcome": LoopOutcome(
            status=status, final_text=decision.final_text, result=decision.result,
            rounds=rounds, tool_calls_used=used, messages=messages,
        )}

    @staticmethod
    def _overflow_outcome(store: MessageStore, rounds, used, span) -> dict:
        span.end("", status="overflow")
        return {"type": "outcome", "outcome": LoopOutcome(
            status="overflow", rounds=rounds, tool_calls_used=used, messages=store.messages,
        )}


async def run_to_outcome(
    loop: AgentLoop, messages: list[dict], run_ctx: RunContext
) -> LoopOutcome:
    outcome: LoopOutcome | None = None
    async for ev in loop.run(messages, run_ctx):
        if ev["type"] == "outcome":
            outcome = ev["outcome"]
    assert outcome is not None, "AgentLoop 未产出 outcome(不应发生)"
    return outcome