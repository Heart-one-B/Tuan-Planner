# harness/agent/agent.py
from __future__ import annotations

from typing import AsyncGenerator, Type

from pydantic import BaseModel

from harness.agent.loop import AgentLoop, Budget, LoopOutcome, run_to_outcome
from harness.agent.result import AgentResult
from harness.agent.run_context import RunContext
from harness.agent.termination import (
    AnswerTermination,
    FinishToolTermination,
    TerminationPolicy,
)
from harness.llm.base import LLMClientBase
from harness.tools.tool_executor import ToolExecutor
from harness.tracing.span import Span


class Agent:
    """harness 中唯一的 Agent 类型:ReAct 循环 + 一套配置。

    第一性原理:范式只有一个(模型决定下一步→执行→观测→重复),
    终止策略、预算、输出契约都是配置维度,不构成新类型——
    "structured agent"不是另一种 Agent,是本类 + FinishToolTermination
    这份配置(见下方 structured_agent() 工厂)。

    与 Claude Code 同构:单一循环,sub-agent = 同一循环换一套配置
    (不同 prompt / 工具子集 / 隔离上下文),而非另一个类。

    三个访问接口,按调用方需要选用:
      events()      事件流(UI/流式场景)
      run()         直取 LoopOutcome(编排场景;基础设施异常会抛出)
      run_result()  收敛为 AgentResult 且永不抛异常
                    (子 Agent 被当作工具嵌套调用时的组合契约)
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        tool_executor: ToolExecutor,
        system_prompt: str,
        termination: TerminationPolicy | None = None,
        budget: Budget | None = None,
        name: str = "agent",
    ):
        self.system_prompt = system_prompt
        self.name = name
        self._loop = AgentLoop(
            llm_client=llm_client,
            tool_executor=tool_executor,
            termination=termination or AnswerTermination(),
            budget=budget or Budget(),
        )

    # ── 接口一:事件流 ──────────────────────────────────────────────────
    async def events(
        self,
        task: str,
        history: list[dict] | None = None,
        parent_span: Span | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """流式事件:thinking / tool_start / tool_end / token / outcome。"""
        run_ctx = RunContext.begin(session_id or self.name, task, parent=parent_span)
        async for ev in self._loop.run(self._messages(task, history), run_ctx):
            yield ev

    # ── 接口二:直取结果 ────────────────────────────────────────────────
    async def run(
        self,
        task: str,
        history: list[dict] | None = None,
        parent_span: Span | None = None,
        session_id: str | None = None,
    ) -> LoopOutcome:
        """跑到底,返回 LoopOutcome。基础设施异常(网络/LLM)原样抛出,
        由调用方决定重试或上报——不替调用方吞错误。"""
        run_ctx = RunContext.begin(session_id or self.name, task, parent=parent_span)
        return await run_to_outcome(self._loop, self._messages(task, history), run_ctx)

    # ── 接口三:子 Agent 组合契约 ───────────────────────────────────────
    async def run_result(
        self,
        task: str,
        parent_span: Span | None = None,
    ) -> AgentResult:
        """一切收敛为 AgentResult,永不抛异常。
        本 Agent 被上层当作工具/子 Agent 嵌套调用时用这个接口:
        上层拿到的永远是统一信封,按 status 决策,不需要 try/except。"""
        try:
            outcome = await self.run(task, parent_span=parent_span)
        except Exception as e:
            return AgentResult(status="error", summary=f"运行异常: {e}", data={})

        if outcome.result is not None:
            return outcome.result
        if outcome.status == "completed":
            # 终止策略没给结构化载荷(如 AnswerTermination):文本装进信封
            return AgentResult(status="ok", summary=outcome.final_text[:200],
                               data={"text": outcome.final_text})
        return AgentResult(status="error", summary="未能在限定轮次内收口", data={})

    # ── 内部 ──
    def _messages(self, task: str, history: list[dict] | None) -> list[dict]:
        return (
            [{"role": "system", "content": self.system_prompt}]
            + (history or [])
            + [{"role": "user", "content": task}]
        )

