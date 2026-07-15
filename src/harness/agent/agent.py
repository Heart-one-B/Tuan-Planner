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
from harness.context.context_manager import ContextManagerConfig
from harness.llm.base import LLMClientBase
from harness.snapshot import SnapshotStore, build_snapshot
from harness.tools.tool_executor import ToolExecutor
from harness.tracing.span import Span


class Agent:
    """harness 中唯一的 Agent 类型:ReAct 循环 + 一套配置。

    context_config: 可选。传入即启用 Phase 2 的三层上下文管理
    (预算记账/卸载/压缩);不传则每次 run() 走裸消息列表,行为与
    Phase 1 完全一致——新能力必须是可选的,不能强迫现有调用方升级认知。
    每次 run() 都会用 context_config.build() 现造一个新的 ContextManager
    (不复用),原因和 RunContext 每次现造一样:避免并发状态串台。

    snapshot_store: 可选。传入即在每次 run 结束后自动落一份 RunSnapshot
    (Phase 3 L3)。同样是"不传即无感"——快照能力不强迫现有调用方感知
    持久化的存在。落在 harness.snapshot 而不是本文件所在的 agent 包下:
    快照是"运行相关的持久化"这一类关注点,和 tracing 同构,在这个仓库里
    从来不是塞进 agent/ 的,理由见 harness/snapshot/models.py 的 docstring。
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        tool_executor: ToolExecutor,
        system_prompt: str,
        termination: TerminationPolicy | None = None,
        budget: Budget | None = None,
        context_config: ContextManagerConfig | None = None,
        snapshot_store: SnapshotStore | None = None,
        name: str = "agent",
    ):
        self.system_prompt = system_prompt
        self.name = name
        self.context_config = context_config
        self.snapshot_store = snapshot_store
        if context_config is not None:
            # 审查修复(致命项):取回工具此前从未被真实装配链路注册——
            # 占位符/预览/摘要到处告诉模型"可调用 read_offloaded_result
            # 取回",但工具不存在,可恢复压缩的承诺是空头支票(测试全绿
            # 是因为测试手动注册了)。不变量:上下文里的每个承诺必须有
            # 真实实现兑现。幂等:重名注册是覆盖语义,多Agent共享executor
            # 且同一offload根目录时安全。
            from harness.context.offload import RETRIEVAL_TOOL_NAME, build_retrieval_tool
            tool_executor.register(build_retrieval_tool(context_config.offload_store))
        self._loop = AgentLoop(
            llm_client=llm_client,
            tool_executor=tool_executor,
            termination=termination or AnswerTermination(),
            budget=budget or Budget(),
        )

    async def events(
        self,
        task: str,
        history: list[dict] | None = None,
        parent_span: Span | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        run_ctx = self._begin_run(task, history, parent_span, session_id)
        final_outcome: LoopOutcome | None = None
        async for ev in self._loop.run(self._messages(task, history), run_ctx):
            if ev["type"] == "outcome":
                final_outcome = ev["outcome"]
            yield ev
        if final_outcome is not None:
            self._maybe_snapshot(final_outcome, run_ctx, session_id, task)

    async def run(
        self,
        task: str,
        history: list[dict] | None = None,
        parent_span: Span | None = None,
        session_id: str | None = None,
    ) -> LoopOutcome:
        run_ctx = self._begin_run(task, history, parent_span, session_id)
        outcome = await run_to_outcome(self._loop, self._messages(task, history), run_ctx)
        self._maybe_snapshot(outcome, run_ctx, session_id, task)
        return outcome

    async def run_result(
        self,
        task: str,
        parent_span: Span | None = None,
    ) -> AgentResult:
        try:
            outcome = await self.run(task, parent_span=parent_span)
        except Exception as e:
            return AgentResult(status="error", summary=f"运行异常: {e}", data={})

        if outcome.result is not None:
            return outcome.result
        if outcome.status == "completed":
            return AgentResult(status="ok", summary=outcome.final_text[:200],
                               data={"text": outcome.final_text})
        if outcome.status == "overflow":
            return AgentResult(status="error", summary="任务规模超出上下文窗口承载能力,建议拆分", data={})
        return AgentResult(status="error", summary="未能在限定轮次内收口", data={})

    def _begin_run(self, task, history, parent_span, session_id) -> RunContext:
        run_ctx = RunContext.begin(session_id or self.name, task, parent=parent_span)
        if self.context_config is not None:
            run_ctx.context_manager = self.context_config.build()
        return run_ctx

    def _messages(self, task: str, history: list[dict] | None) -> list[dict]:
        return (
            [{"role": "system", "content": self.system_prompt}]
            + (history or [])
            + [{"role": "user", "content": task}]
        )

    def _maybe_snapshot(self, outcome: LoopOutcome, run_ctx: RunContext,
                        session_id: str | None, task: str) -> None:
        """run 结束自动拍一份快照。仅在配置了 snapshot_store 时生效——
        "新能力必须可选、不配置即无感"的老规矩。"""
        if self.snapshot_store is None:
            return
        snap = build_snapshot(outcome, run_ctx, session_id or self.name, task)
        self.snapshot_store.save(snap)