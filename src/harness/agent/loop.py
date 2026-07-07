# harness/agent/loop.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Literal

from harness.agent.run_context import RunContext
from harness.agent.termination import Finish, Nudge, Reject, TerminationPolicy
from harness.llm.base import LLMClientBase
from harness.tools.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


# ── 预算策略 ────────────────────────────────────────────────────────────
@dataclass
class Budget:
    """轮次/工具调用预算,以及耗尽时的收口动作。

    exhausted_action:
      force_answer  追加提示后让模型基于已有信息流式作答(原 ReactAgent 语义)
      force_finish  追加提示后再给一轮机会调终止工具收口(原 StructuredAgent 语义)
      stop          直接结束,outcome.status = exhausted
    """
    max_tool_calls: int = 10
    max_rounds: int = 24          # 护栏:防止 Nudge 拉锯导致的无工具死循环
    exhausted_action: Literal["stop", "force_answer", "force_finish"] = "force_answer"
    exhausted_prompt: str = (
        "你已用完工具调用次数,请根据已有信息尽力完成任务;"
        "信息不足请如实说明缺少什么,不要编造。"
    )


# ── 循环产出 ────────────────────────────────────────────────────────────
@dataclass
class LoopOutcome:
    """一次循环运行的结果信封。

    status:
      completed  终止策略正常判定结束
      exhausted  预算耗尽后收口(可能带着 force_answer/force_finish 的产物)
    result 是终止策略给的结构化载荷(如 AgentResult),内核不解读。
    预算耗尽不再可能被静默包装成"成功"——status 会如实暴露给上层。
    """
    status: Literal["completed", "exhausted"]
    final_text: str = ""
    result: Any = None
    rounds: int = 0
    tool_calls_used: int = 0
    messages: list = field(default_factory=list)   # 最终消息历史(调试/快照用)


# ── 循环内核 ────────────────────────────────────────────────────────────
class AgentLoop:
    """Agent 循环内核:整个 harness 里唯一的一份 while。

    六阶段骨架(对应 Claude Code 的 turn 流水线,缺的阶段是显式预留的插槽):
      ① 组装上下文      —— 现在=裸 messages;Phase 2 换 ContextManager,只改这里
      ② 调模型
      ③ 解读响应        —— 推理内容浮出事件流;纯文本 → 终止策略判定
      ④ 分发工具        —— [权限门控插槽:Phase 6 插在 execute 之前]
                           终止调用与普通调用分拣:普通的先全部执行,
                           终止的最后处理(结构性修复混发丢结果 bug)
      ⑤ 回填结果        —— 每个 tool_call_id 必有对应 tool 消息(API 硬约束)
                           [结果精简插槽:Phase 2 的 tool_result_policy 插这里]
      ⑥ 预算/溢出检查   —— 耗尽走 Budget.exhausted_action 收口
                           [已知缺口:上下文溢出(prompt_too_long)尚未作为
                            独立终止条件处理,Phase 2 接入时补]

    产出事件流,最后一个事件固定为:
        {"type": "outcome", "outcome": LoopOutcome}
    不需要流式的调用方用 run_to_outcome() 直取结果。

    span 生命周期由内核负责关闭(含异常路径),业务 Agent 不再写 trace 八股。
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
        tools = self.tool_executor.schemas + self.termination.extra_tools()
        rounds = 0
        tool_calls_used = 0

        try:
            while True:
                rounds += 1

                # ⑥ 轮次护栏(前置检查:防 Nudge 无限拉锯)
                if rounds > self.budget.max_rounds:
                    async for ev in self._exhaust(messages, run_ctx, rounds, tool_calls_used):
                        yield ev
                    return

                # ① 组装上下文(Phase 2: messages = run_ctx.context_manager.messages)
                # ② 调模型
                resp = await self.llm.call(
                    trace_id=span.trace_id, messages=messages, tools=tools,
                )
                msg = resp.choices[0].message

                # ③ 解读响应:推理内容浮出事件流。
                # msg.reasoning 是 LLMClientBase 归一化后的约定属性(各提供商的
                # reasoning_content / thinking 字段在客户端层已抹平);
                # 没有独立推理通道的模型,工具调用附带的 content 本身就是推理产物。
                thinking = getattr(msg, "reasoning", None) or (
                    msg.content if msg.tool_calls else None
                )
                if thinking:
                    yield {"type": "thinking", "content": thinking}

                # ③ 纯文本响应 → 终止策略判定
                if not msg.tool_calls:
                    decision = self.termination.on_plain_message(msg)
                    if isinstance(decision, Finish):
                        for ev in self._finish_events(decision):
                            yield ev
                        yield self._outcome(
                            "completed", decision, rounds, tool_calls_used, messages, span,
                        )
                        return
                    # Nudge:催一句,继续下一轮
                    messages.append({"role": "user", "content": decision.message})
                    continue

                # ④⑤ 工具调用:先分拣,普通调用全部执行完,终止调用最后处理
                messages.append(msg)
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
                    # ⑤ 回填(Phase 2 在此插 tool_result_policy 精简)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id, "content": str(result),
                    })

                finished: Finish | None = None
                for tc in terminal_calls:
                    decision = self.termination.intercept(tc, forced=False)
                    if isinstance(decision, Finish):
                        finished = decision
                        break
                    # Reject:回填错误让模型重试(保证该 tool_call_id 有响应)
                    messages.append({
                        "role": "tool", "tool_call_id": decision.tool_call_id,
                        "content": decision.message,
                    })

                if finished is not None:
                    for ev in self._finish_events(finished):
                        yield ev
                    yield self._outcome(
                        "completed", finished, rounds, tool_calls_used, messages, span,
                    )
                    return

                # ⑥ 工具预算检查
                if tool_calls_used >= self.budget.max_tool_calls:
                    async for ev in self._exhaust(messages, run_ctx, rounds, tool_calls_used):
                        yield ev
                    return

        except Exception as e:
            # span 在异常路径也必须闭合,否则 trace 永远悬挂
            logger.error(f"[AgentLoop] error: {e}", exc_info=True)
            span.end("", status="error")
            raise

    # ── 预算耗尽收口 ────────────────────────────────────────────────────
    async def _exhaust(
        self, messages: list[dict], run_ctx: RunContext, rounds: int, used: int
    ) -> AsyncGenerator[dict, None]:
        span = run_ctx.span
        action = self.budget.exhausted_action
        logger.warning(f"[AgentLoop] budget exhausted (action={action}, "
                       f"rounds={rounds}, tool_calls={used})")

        if action == "stop":
            yield self._outcome("exhausted", Finish(), rounds, used, messages, span,
                                trace_status="timeout")
            return

        messages.append({"role": "user", "content": self.budget.exhausted_prompt})

        if action == "force_answer":
            # 流式让模型基于已有信息作答
            final_text = ""
            stream = await self.llm.call(
                trace_id=span.trace_id, messages=messages, stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    final_text += content
                    yield {"type": "token", "content": content}
            yield self._outcome("exhausted", Finish(final_text=final_text),
                                rounds, used, messages, span, trace_status="timeout")
            return

        # force_finish:再给一轮机会调终止工具
        tools = self.tool_executor.schemas + self.termination.extra_tools()
        resp = await self.llm.call(trace_id=span.trace_id, messages=messages, tools=tools)
        msg = resp.choices[0].message
        finished: Finish | None = None
        for tc in (msg.tool_calls or []):
            if self.termination.owns(tc):
                decision = self.termination.intercept(tc, forced=True)
                if isinstance(decision, Finish):
                    finished = decision
                break
        yield self._outcome("exhausted", finished or Finish(),
                            rounds, used, messages, span, trace_status="timeout")

    # ── 内部工具 ────────────────────────────────────────────────────────
    @staticmethod
    def _finish_events(decision: Finish):
        """最终文本按 token 事件发出(与原 ReactAgent 行为对齐)。"""
        for char in decision.final_text or "":
            yield {"type": "token", "content": char}

    @staticmethod
    def _outcome(status, decision: Finish, rounds, used, messages, span,
                 trace_status: str = "success") -> dict:
        span.end(decision.final_text, status=trace_status)
        return {"type": "outcome", "outcome": LoopOutcome(
            status=status,
            final_text=decision.final_text,
            result=decision.result,
            rounds=rounds,
            tool_calls_used=used,
            messages=messages,
        )}


async def run_to_outcome(
    loop: AgentLoop, messages: list[dict], run_ctx: RunContext
) -> LoopOutcome:
    """不关心事件流的调用方直取最终结果。"""
    outcome: LoopOutcome | None = None
    async for ev in loop.run(messages, run_ctx):
        if ev["type"] == "outcome":
            outcome = ev["outcome"]
    assert outcome is not None, "AgentLoop 未产出 outcome(不应发生)"
    return outcome