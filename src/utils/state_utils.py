# src/utils/state_utils.py
from __future__ import annotations

from src.graph.state import AgentState


def _append_error(state: AgentState, error: str) -> dict:
    errors = list(state.get("errors") or [])
    errors.append(error)
    return {"errors": errors}


def _candidate_activities(candidate: dict) -> list[dict]:
    """从单个候选方案中提取所有活动，去重。"""
    if not isinstance(candidate, dict):
        return []

    out: list[dict] = []
    seen: set[str] = set()

    def _add(value: object) -> None:
        if not isinstance(value, dict) or not value:
            return
        key = value.get("id") or value.get("name")
        if isinstance(key, str) and key:
            if key in seen:
                return
            seen.add(key)
        out.append(value)

    activities = candidate.get("activities")
    if isinstance(activities, list):
        for item in activities:
            _add(item)
    _add(candidate.get("activity"))
    _add(candidate.get("secondary_activity"))
    return out


def _resolve_replan_reason(state: AgentState) -> str:
    """读取重规划原因；没有则从校验违约项拼合。"""
    reason = state.get("replan_reason")
    if isinstance(reason, str) and reason.strip():
        return reason
    result = state.get("rule_validation_result")
    if isinstance(result, dict):
        violations = result.get("violations")
        if isinstance(violations, list):
            parts = [v for v in violations if isinstance(v, str) and v.strip()]
            if parts:
                return "; ".join(parts)
    return ""


def _normalize_replan_reason_type(state: AgentState) -> str:
    """推断重规划类型标签。"""
    raw = state.get("replan_reason_type")
    if raw in {"validation_failure", "user_feedback", "execution_core_change"}:
        return raw
    if isinstance(state.get("execution_result"), dict) and state["execution_result"].get("core_plan_changed"):
        return "execution_core_change"
    if state.get("user_confirmed") is False:
        return "user_feedback"
    return "validation_failure"


# ── 通用小工具 ────────────────────────────────────────────────────────────────

def _is_pos_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def _append_unique_hint(hints: list[str], hint: str) -> None:
    if isinstance(hint, str) and hint and hint not in hints:
        hints.append(hint)