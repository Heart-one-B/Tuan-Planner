# harness/snapshot/build.py
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from harness.snapshot.models import RunSnapshot, normalize_message, record_to_dict, result_to_dict

if TYPE_CHECKING:
    # 仅类型标注用,不产生运行期依赖——models.py 的零依赖纪律在这里
    # 也尽量维持,只是本文件的职责本来就是"跨到 agent 层去取数据",
    # 这层耦合无法避免,但局部化到这一个文件里,不扩散。
    from harness.agent.loop import LoopOutcome
    from harness.agent.run_context import RunContext


def build_snapshot(outcome: "LoopOutcome", run_ctx: "RunContext",
                   session_id: str, task: str) -> RunSnapshot:
    """从 LoopOutcome + RunContext 构造快照。

    未配置 ContextManager 时同样可用(messages-only,ctx 相关字段留空
    默认值)——快照能力不强迫用户先启用上下文管理,"可选、不配置即
    无感"的老规矩在这里继续成立。
    """
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
    )