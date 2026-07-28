# harness/agent/agent.py
from __future__ import annotations

import logging
from typing import AsyncGenerator

from harness.agent.abort import AbortSignal
from harness.agent.loop import Budget, LoopOutcome, wrap_store
from harness.agent.permission import Allow, AllowAllPolicy, Deny, PermissionPolicy
from harness.agent.query import query, run_to_outcome
from harness.agent.result import AgentResult
from harness.agent.run_context import RunContext
from harness.agent.state import LoopConfig, LoopState, ResumePoint
from harness.context.context_manager import ContextManagerConfig
from harness.llm.base import LLMClientBase
from harness.memory.extraction import ExtractionBookkeeping, maybe_extract
from harness.memory.inject import build_index_message, strip_old_index_messages
from harness.memory.instructions import MEMORY_RULES_TEMPLATE, MemoryConfig, load_static_instructions
from harness.memory.recall import RecallOutcome, SurfacedMemory, maybe_recall
from harness.memory.store import FileMemoryStore
from harness.memory.tools import build_memory_tools
from harness.message_id import get_hid, tag_message
from harness.snapshot import SnapshotStore, build_snapshot
from harness.tools.tool_executor import ToolExecutor
from harness.tracing.span import Span

logger = logging.getLogger(__name__)


def _build_error_snapshot(state: LoopState, run_ctx: RunContext, session_id: str,
                          task: str, bookkeeping: ExtractionBookkeeping,
                          error: Exception):
    """任务 3.12(决策 D)的落点。不走 build_snapshot(outcome, ...)——
    那个函数要求一个真实的 LoopOutcome,而这里恰恰是"query() 都没能
    产出 outcome 就崩了"的场景,没有 outcome 可用。直接构造
    RunSnapshot:messages 取 state.store.messages(哪怕这次 run 没
    走完,已经发生的对话历史仍然是有价值的现场),status 用自由字符串
    "error"(RunSnapshot.status 本来就没有类型层面的枚举限制,不需要
    像 LoopOutcome.status 那样费心塞进 Literal),exit_reason 记录
    异常类型,方便宿主一眼看出崩在哪类问题上。"""
    from datetime import datetime
    from harness.snapshot.models import RunSnapshot, normalize_message

    return RunSnapshot(
        session_id=session_id, task=task, trace_id=run_ctx.span.trace_id,
        status="error", rounds=state.rounds, tool_calls_used=state.tool_calls_used,
        messages=[normalize_message(m) for m in state.store.messages],
        created_at=datetime.now().isoformat(),
        extraction_boundary_hid=bookkeeping.boundary_hid,
        runs_since_extraction=bookkeeping.runs_since,
        overflow_recovery_count=state.overflow_recovery_count,
        output_truncation_count=state.output_truncation_count,
        output_upgraded=state.output_upgraded,
        terminal_nudge_count=state.terminal_nudge_count,
        exit_reason=f"unhandled_exception:{type(error).__name__}",
    )


class Agent:
    """第一层(ask)+ 第二层(QueryEngine)。

    【第二刀改动】不再持有一个 AgentLoop 实例——第一刀里 AgentLoop
    本质上只是"不变配置 + 一堆方法",第二刀把配置部分收进
    LoopConfig(frozen,构造一次、跨 run 复用),方法部分变成 loop.py
    里的模块级函数,类本身没有继续存在的理由。self._loop_config 是
    这次改动在 Agent 内部留下的唯一可见痕迹。
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        tool_executor: ToolExecutor,
        system_prompt: str,
        budget: Budget | None = None,
        context_config: ContextManagerConfig | None = None,
        snapshot_store: SnapshotStore | None = None,
        memory_config: MemoryConfig | None = None,
        permission_policy: PermissionPolicy | None = None,
        require_terminal_tool: str | None = None,
        max_terminal_nudges: int = 2,
        on_error_snapshot: bool = False,
        name: str = "agent",
    ):
        self.system_prompt = system_prompt
        self.name = name
        self.context_config = context_config
        self.snapshot_store = snapshot_store
        self.memory_config = memory_config
        self.on_error_snapshot = on_error_snapshot
        # 任务 3.12(决策 D):异常路径默认不落快照,维持现状——这是
        # "宿主想不想要一个可以从异常点 resume 的快照"的选择权,不是
        # harness 替宿主决定。开启后,events()/resume_events() 在
        # 未捕获异常冒泡之前,会尽力落一份 status="error" 的快照,
        # 见 _build_error_snapshot() 的说明。
        self._memory_store = FileMemoryStore(memory_config.memory_dir) if memory_config else None
        self._extraction_bookkeeping: dict[str, ExtractionBookkeeping] = {}
        self._surfaced_memories: dict[str, list[SurfacedMemory]] = {}

        if context_config is not None:
            from harness.context.offload import RETRIEVAL_TOOL_NAME, build_retrieval_tool
            tool_executor.register(build_retrieval_tool(context_config.offload_store))

        if memory_config is not None:
            for tool in build_memory_tools(self._memory_store, memory_config):
                tool_executor.register(tool)

        self._llm = llm_client  # 提取子 Agent 复用主 Agent 的 llm 作兜底,见 _run_extraction
        self._loop_config = LoopConfig(
            llm=llm_client,
            tool_executor=tool_executor,
            budget=budget or Budget(),
            permission_policy=permission_policy or AllowAllPolicy(),
            tools=tool_executor.schemas,
            require_terminal_tool=require_terminal_tool,
            max_terminal_nudges=max_terminal_nudges,
        )

    # ── 第二层(QueryEngine):组装现场,驱动 query(),两个入口 ─────────────

    async def events(
        self,
        task: str,
        history: list[dict] | None = None,
        parent_span: Span | None = None,
        session_id: str | None = None,
        abort: AbortSignal | None = None,
    ) -> AsyncGenerator[dict, None]:
        run_ctx = self._begin_run(task, history, parent_span, session_id, abort)
        sid = session_id or self.name
        recall = await self._maybe_recall(task, history, session_id, run_ctx)
        messages, task_hid = self._messages(task, history, recall.injection_message)
        extraction_bk = self._anchor_extraction_boundary(sid, task_hid)

        store = wrap_store(messages, run_ctx)
        state = LoopState(store=store)

        outcome: LoopOutcome | None = None
        try:
            async for ev in query(self._loop_config, state, run_ctx):
                if ev["type"] == "outcome":
                    outcome = ev["outcome"]
                yield ev
        except Exception as e:
            if self.on_error_snapshot and self.snapshot_store is not None:
                snap = _build_error_snapshot(state, run_ctx, sid, task, extraction_bk, e)
                self.snapshot_store.save(snap)
            raise
        if outcome is not None:
            await self._finalize(outcome, run_ctx, session_id, task, extraction_bk)

    async def resume_events(
        self, session_id: str, decision: Allow | Deny,
        abort: AbortSignal | None = None,
    ) -> AsyncGenerator[dict, None]:
        """【第二刀新增】events() 的恢复版本——组装现场的方式不同
        (从快照读,不是从 task/history 组装),但驱动 query() 的方式
        和收尾方式(_finalize)与 events() 完全一致,只此一份。

        对应第一刀里 Agent.resume() 的职责,现在拆成"组装现场"
        (这个方法)+"drain 成单个结果"(下面的 resume())两层,
        和 run()/events() 的关系完全对称。
        """
        if self.snapshot_store is None:
            raise ValueError("resume_events() 需要配置 snapshot_store 才能找回挂起的运行现场")
        snap = self.snapshot_store.load_latest(session_id)
        if snap.pending_tool_call_id is None:
            raise ValueError(f"session={session_id} 当前没有待处理的审批,resume 无事可做")

        run_ctx = RunContext.begin(session_id, snap.task, parent=None, abort=abort)
        store = wrap_store(snap.messages, run_ctx)
        # 任务 3.9 的接线:从 v6 快照读回全部计数器重建 LoopState,
        # 而不是像第二刀那样只读 rounds/tool_calls_used——这是
        # bug#1("overflow_recovered 跨 resume 边界重置")真正被修复
        # 的地方。用 LoopState.from_dict() 而不是手写四个关键字参数,
        # 是为了让"resume 时要从快照恢复哪些字段"这件事只在一处
        # (state.py 的 from_dict)描述,这里不重复枚举、不会漏项。
        state = LoopState.from_dict(
            {
                "rounds": snap.rounds,
                "tool_calls_used": snap.tool_calls_used,
                "overflow_recovery_count": snap.overflow_recovery_count,
                "output_truncation_count": snap.output_truncation_count,
                "output_upgraded": snap.output_upgraded,
                "terminal_nudge_count": snap.terminal_nudge_count,
            },
            store=store,
        )
        state.resume_point = ResumePoint(
            tool_call_id=snap.pending_tool_call_id, decision=decision,
        )

        bookkeeping = self._load_extraction_bookkeeping(session_id)
        outcome: LoopOutcome | None = None
        try:
            async for ev in query(self._loop_config, state, run_ctx):
                if ev["type"] == "outcome":
                    outcome = ev["outcome"]
                yield ev
        except Exception as e:
            if self.on_error_snapshot and self.snapshot_store is not None:
                snap_err = _build_error_snapshot(
                    state, run_ctx, session_id, snap.task, bookkeeping, e,
                )
                self.snapshot_store.save(snap_err)
            raise
        if outcome is not None:
            await self._finalize(outcome, run_ctx, session_id, snap.task, bookkeeping)

    # ── 第一层(ask):一次性入口,drain 对应的 events 生成器 ───────────────

    async def run(
        self,
        task: str,
        history: list[dict] | None = None,
        parent_span: Span | None = None,
        session_id: str | None = None,
        abort: AbortSignal | None = None,
    ) -> LoopOutcome:
        outcome: LoopOutcome | None = None
        async for ev in self.events(task, history, parent_span, session_id, abort):
            if ev["type"] == "outcome":
                outcome = ev["outcome"]
        assert outcome is not None, "events() 未产出 outcome(不应发生)"
        return outcome

    async def resume(self, session_id: str, decision: Allow | Deny,
                     abort: AbortSignal | None = None) -> LoopOutcome:
        outcome: LoopOutcome | None = None
        async for ev in self.resume_events(session_id, decision, abort):
            if ev["type"] == "outcome":
                outcome = ev["outcome"]
        assert outcome is not None, "resume_events() 未产出 outcome(不应发生)"
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
        if outcome.status == "aborted":
            # 任务 3.11:中断是用户意志,不是资源耗尽或系统故障——
            # summary 如实说明这一点,不与"未能在限定轮次内收口"
            # (真正的耗尽)混为一谈,宿主日后按 summary 排查时不会
            # 误判成任务太难。第四刀落地 AbortSignal 之前,这个分支
            # 不会被真实触发,但类型和落点先留好。
            return AgentResult(status="error", summary="运行被中断", data={})
        return AgentResult(status="error", summary="未能在限定轮次内收口", data={})

    async def extract_now(self, session_id: str, outcome: LoopOutcome) -> ExtractionBookkeeping:
        if self.memory_config is None:
            return ExtractionBookkeeping()
        sid = session_id or self.name
        bookkeeping = self._load_extraction_bookkeeping(sid)
        new_bookkeeping = await self._run_extraction(outcome, sid, None, bookkeeping)
        self._write_back_extraction_snapshot(sid, new_bookkeeping)
        return new_bookkeeping

    # ── 唯一一份会话收尾(任务 2.8) ────────────────────────────────────
    async def _finalize(
        self, outcome: LoopOutcome, run_ctx: RunContext,
        session_id: str | None, task: str, bookkeeping: ExtractionBookkeeping,
    ) -> None:
        """第一刀里 run()/events()/resume() 各有一份几乎相同的收尾
        (判是否挂起 → 可能提取 → 落快照);第二刀收口成这一份,
        events()/resume_events() 两个入口共用——再加第一层的
        run()/resume() 是 drain 这两者,不需要各自再实现一次收尾。
        新增第五个入口时不需要同步任何东西,这是设计方案要根治的
        bug #3(收尾逻辑重复三份)从结构上不可能再发生的直接体现。

        故意放在生成器体内、`async for` 循环结束之后,不放 `finally`:
        调用方提前 break 时这次 run 没有终态,本就不该提取、不该落
        快照;放 finally 会在 `aclose()` 期间执行 await,事件循环状态
        不确定,是给自己埋雷(设计方案 §5.3 的原话)。

        取 surfaced_memories 统一用 _load_surfaced_memories(sid)
        (优先读快照、否则退回进程内缓存),而不是像第一刀 run()/
        events() 那样直接读 self._surfaced_memories.get(sid)——两者
        在 _maybe_recall 刚执行完的这一刻是等价的(缓存总是最新),
        统一成一种写法只是消灭一个本可以避免的不一致,不是行为变更。
        """
        sid = session_id or self.name
        if outcome.status != "awaiting_approval":
            bookkeeping = await self._maybe_extract_auto(
                outcome, session_id, run_ctx.span, bookkeeping,
            )
        self._maybe_snapshot(outcome, run_ctx, session_id, task,
                             bookkeeping, self._load_surfaced_memories(sid))

    # ── 提取(第二期,不变) ──────────────────────────────────────────

    def _anchor_extraction_boundary(
        self, session_id: str, task_hid: str,
    ) -> ExtractionBookkeeping:
        if self.memory_config is None:
            return ExtractionBookkeeping()
        bookkeeping = self._load_extraction_bookkeeping(session_id)
        if bookkeeping.boundary_hid is not None:
            return bookkeeping
        anchored = ExtractionBookkeeping(boundary_hid=task_hid, runs_since=bookkeeping.runs_since)
        self._extraction_bookkeeping[session_id] = anchored
        return anchored

    async def _maybe_extract_auto(
        self, outcome: LoopOutcome, session_id: str | None,
        parent_span: Span | None, bookkeeping: ExtractionBookkeeping,
    ) -> ExtractionBookkeeping:
        if self.memory_config is None or self.memory_config.extraction_mode != "auto":
            return bookkeeping
        return await self._run_extraction(outcome, session_id, parent_span, bookkeeping)

    async def _run_extraction(
        self, outcome: LoopOutcome, session_id: str | None,
        parent_span: Span | None, bookkeeping: ExtractionBookkeeping,
    ) -> ExtractionBookkeeping:
        sid = session_id or self.name
        new_bookkeeping = await maybe_extract(
            session_id=sid,
            all_messages=outcome.messages,
            bookkeeping=bookkeeping,
            memory_store=self._memory_store,
            memory_config=self.memory_config,
            fallback_llm_client=self._llm,
            parent_span=parent_span,
        )
        self._extraction_bookkeeping[sid] = new_bookkeeping
        return new_bookkeeping

    def _load_extraction_bookkeeping(self, session_id: str) -> ExtractionBookkeeping:
        if self.snapshot_store is not None:
            try:
                snap = self.snapshot_store.load_latest(session_id)
                return ExtractionBookkeeping(
                    boundary_hid=snap.extraction_boundary_hid,
                    runs_since=snap.runs_since_extraction,
                )
            except FileNotFoundError:
                pass
        return self._extraction_bookkeeping.get(session_id, ExtractionBookkeeping())

    def _write_back_extraction_snapshot(
        self, session_id: str, bookkeeping: ExtractionBookkeeping,
    ) -> None:
        if self.snapshot_store is None:
            return
        try:
            snap = self.snapshot_store.load_latest(session_id)
        except FileNotFoundError:
            logger.warning(
                f"[Agent] extract_now: session={session_id} 尚无可回写的快照,"
                f"提取簿记仅留在进程内存"
            )
            return
        snap.extraction_boundary_hid = bookkeeping.boundary_hid
        snap.runs_since_extraction = bookkeeping.runs_since
        self.snapshot_store.save(snap)

    # ── 召回(第三期,不变) ──────────────────────────────────────────

    async def _maybe_recall(
        self, task: str, history: list[dict] | None,
        session_id: str | None, run_ctx: RunContext,
    ) -> RecallOutcome:
        if self.memory_config is None:
            return RecallOutcome()
        sid = session_id or self.name
        already = self._load_surfaced_memories(sid)
        outcome = await maybe_recall(
            session_id=sid,
            task=task,
            trace_id=run_ctx.span.trace_id,
            history=list(history or []),
            already_surfaced=already,
            memory_store=self._memory_store,
            memory_config=self.memory_config,
            fallback_llm_client=self._llm,
        )
        by_name = {m.name: m for m in already}
        for m in outcome.newly_surfaced:
            by_name[m.name] = m
        self._surfaced_memories[sid] = list(by_name.values())
        return outcome

    def _load_surfaced_memories(self, session_id: str) -> list[SurfacedMemory]:
        if self.snapshot_store is not None:
            try:
                snap = self.snapshot_store.load_latest(session_id)
                return [SurfacedMemory.from_dict(d) for d in snap.surfaced_memories]
            except FileNotFoundError:
                pass
        return list(self._surfaced_memories.get(session_id, []))

    # ── 通用 ────────────────────────────────────────────────────────

    def _begin_run(self, task, history, parent_span, session_id, abort=None) -> RunContext:
        run_ctx = RunContext.begin(session_id or self.name, task, parent=parent_span, abort=abort)
        if self.context_config is not None:
            run_ctx.context_manager = self.context_config.build()
        return run_ctx

    def _messages(
        self, task: str, history: list[dict] | None,
        recall_injection: dict | None = None,
    ) -> tuple[list[dict], str]:
        history = list(history or [])

        system_content = self.system_prompt
        index_message = None
        if self.memory_config is not None:
            history = strip_old_index_messages(history)

            static = load_static_instructions(self.memory_config)
            if static:
                system_content = f"{system_content}\n\n{static}"
            system_content = f"{system_content}\n{MEMORY_RULES_TEMPLATE}"

            index_message = build_index_message(self._memory_store, self.memory_config)

        messages = [{"role": "system", "content": system_content}]
        if index_message is not None:
            messages.append(index_message)
        messages += history
        if recall_injection is not None:
            messages.append(recall_injection)
        task_message = tag_message({"role": "user", "content": task})
        messages.append(task_message)
        return messages, get_hid(task_message)

    def _maybe_snapshot(self, outcome: LoopOutcome, run_ctx: RunContext,
                        session_id: str | None, task: str,
                        bookkeeping: ExtractionBookkeeping,
                        surfaced_memories: list | None = None) -> None:
        if self.snapshot_store is None:
            return
        sid = session_id or self.name
        snap = build_snapshot(outcome, run_ctx, sid, task,
                              extraction_boundary_hid=bookkeeping.boundary_hid,
                              runs_since_extraction=bookkeeping.runs_since,
                              surfaced_memories=surfaced_memories or [],
                              pending_approval_id=outcome.pending_approval_id,
                              pending_tool_call_id=outcome.pending_tool_call_id)
        self.snapshot_store.save(snap)