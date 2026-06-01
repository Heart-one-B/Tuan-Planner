from __future__ import annotations

from src.graph.state import AgentState
from src.utils.state_utils import _append_error, _resolve_replan_reason, _normalize_replan_reason_type


def repair_loop_node(state: AgentState) -> AgentState:
    """
    Repair Loop Node：分析校验失败原因，生成重规划指令，回到 candidate_planning_node。

    violation 类型及对应指令：
      transition_too_rushed → 要求扩大相邻 step 间距，或换距离更近的 POI
      spare_gap / spare_tail → 明确告知空档时段，要求补充活动
    """
    print("[Repair Loop Node] 分析失败原因，生成重规划指令...")

    rule_validation_result = state.get("rule_validation_result") or {}
    invalid_plans: list[dict] = rule_validation_result.get("invalid_plans") or []
    replan_count: int = state.get("replan_count") or 0

    # 汇总所有 invalid 方案的 violations，生成重规划提示
    hints: list[str] = []
    for plan in invalid_plans:
        for v in (plan.get("violations") or []):
            if isinstance(v, str) and v not in hints:
                hints.append(v)

    # 兜底：读 replan_reason
    if not hints:
        reason = _resolve_replan_reason(state)
        if reason:
            hints.append(reason)

    replan_reason = "\n".join(hints) if hints else "方案校验未通过，请重新生成候选方案"
    replan_reason_type = _normalize_replan_reason_type(state)

    print(f"[Repair Loop Node] 重规划原因（第 {replan_count + 1} 次）：\n{replan_reason}")

    return {
        "replan_count":        replan_count + 1,
        "replan_reason":       replan_reason,
        "replan_reason_type":  replan_reason_type,
        # 清空上一轮的候选，让 candidate_planning_node 重新生成
        "candidate_plans":     {},
        "rule_validation_result": {},
        "scoring_result":      {},
    }