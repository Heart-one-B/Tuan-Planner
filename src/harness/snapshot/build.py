# harness/snapshot/build.py
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from harness.snapshot.models import RunSnapshot, normalize_message, record_to_dict, result_to_dict

if TYPE_CHECKING:
    from harness.agent.loop import LoopOutcome
    from harness.agent.run_context import RunContext


def build_snapshot(outcome: "LoopOutcome", run_ctx: "RunContext",
                   session_id: str, task: str,
                   extraction_boundary_hid: str | None = None,
                   runs_since_extraction: int = 0,
                   surfaced_memories: list | None = None,
                   pending_approval_id: str | None = None,
                   pending_tool_call_id: str | None = None) -> RunSnapshot:
    cm = run_ctx.context_manager
    ctx = cm.to_snapshot() if cm is not None else None
    return RunSnapshot(
        session_id=session_id,
        task=task,
        trace_id=run_ctx.span.trace_id,
        status=outcome.status,
        rounds=outcome.rounds,
        tool_calls_used=outcome.tool_calls_used,
        messages=[normalize_message(m) for m in outcome.messages],
        compaction_history=[result_to_dict(c) for c in (ctx.compaction_history if ctx else [])],
        offload_records=[record_to_dict(r) for r in (ctx.offload_records if ctx else [])],
        total_tokens=ctx.total_tokens if ctx else 0,
        token_source=ctx.token_source if ctx else "estimated",
        created_at=datetime.now().isoformat(),
        extraction_boundary_hid=extraction_boundary_hid,
        runs_since_extraction=runs_since_extraction,
        surfaced_memories=[m.to_dict() for m in surfaced_memories] if surfaced_memories else [],
        pending_approval_id=pending_approval_id,
        pending_tool_call_id=pending_tool_call_id,
        # 任务 3.10:不需要 build_snapshot() 新增参数——outcome(LoopOutcome)
        # 在第三刀已经携带这四个字段(见 loop.py 的每个构造点),直接读。
        exit_reason=outcome.reason,
        overflow_recovery_count=outcome.overflow_recovery_count,
        output_truncation_count=outcome.output_truncation_count,
        output_upgraded=outcome.output_upgraded,
        terminal_nudge_count=outcome.terminal_nudge_count,
    )