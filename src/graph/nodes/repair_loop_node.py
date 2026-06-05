from __future__ import annotations

from src.graph.state import AgentState
from src.utils.state_utils import _append_error, _resolve_replan_reason, _normalize_replan_reason_type


def _hhmm_to_minutes(t: str) -> int | None:
    if not isinstance(t, str) or ":" not in t:
        return None
    try:
        h, m = map(int, t.split(":", 1))
        return h * 60 + m
    except ValueError:
        return None


def _minutes_to_hhmm(total: int) -> str:
    total = max(0, min(total, 23 * 60 + 59))
    return f"{total // 60:02d}:{total % 60:02d}"


def _build_precise_hint(violation: str, steps: list[dict]) -> str:
    """
    将 violation 原始字符串转换为带精确时段的修复指令。

    spare_gap → 明确告知空档起止时间和建议时长
    dinner_too_late → 告知超出多少分钟，需压缩哪个下午活动
    lunch_too_late → 告知超出多少分钟，需压缩哪个上午活动
    其他 → 原样返回
    """
    if violation.startswith("spare_gap:"):
        # 从 violation 里提取已有的时段信息（rule_validation 已经带了）
        # 格式：spare_gap: 「A」(HH:MM) 到「B」(HH:MM) 有 N 分钟空档，请在此时段（HH:MM ~ HH:MM）补充一个活动
        return (
            f"{violation}\n"
            f"  → 修复建议：在括号内标注的时段中，从候选池中选一个尚未使用的 POI 插入，"
            f"duration_tier=short 或 medium 均可，phase 填 afternoon 或对应时段。"
        )

    elif violation.startswith("dinner_too_late:"):
        # 找最后一个下午活动
        afternoon_steps = [s for s in steps if s.get("phase") in ("morning", "afternoon")]
        if afternoon_steps:
            last_af = afternoon_steps[-1]
            label   = last_af.get("label") or "下午最后一个活动"
            dur     = last_af.get("duration_minutes") or 90
            end_t   = last_af.get("end_time") or "?"
            return (
                f"{violation}\n"
                f"  → 修复建议：压缩「{label}」的时长（当前约 {dur} 分钟），"
                f"使晚饭能在 19:00 前开始。"
                f"压缩后若晚饭与前序活动间隔超过 60 分钟，系统会自动整体前移晚饭及后续节点。"
            )
        return violation

    elif violation.startswith("lunch_too_late:"):
        morning_steps = [s for s in steps if s.get("phase") == "morning"]
        if morning_steps:
            last_mo = morning_steps[-1]
            label   = last_mo.get("label") or "上午最后一个活动"
            dur     = last_mo.get("duration_minutes") or 90
            return (
                f"{violation}\n"
                f"  → 修复建议：压缩「{label}」的时长（当前约 {dur} 分钟），"
                f"使午饭能在 13:00 前开始。"
            )
        return violation

    else:
        return violation


def repair_loop_node(state: AgentState) -> AgentState:
    """
    Repair Loop Node：分析校验失败原因，生成带精确时段信息的重规划指令。

    核心改动：
    - spare_gap 类 violation 在原有时段信息基础上补充修复建议
    - dinner_too_late / lunch_too_late 告知具体需要压缩哪个活动
    - 所有 hint 都带明确的时间区间，让模型有方向地修正而非重新自由发挥
    """
    print("[Repair Loop Node] 分析失败原因，生成重规划指令...")

    rule_validation_result = state.get("rule_validation_result") or {}
    invalid_plans: list[dict] = rule_validation_result.get("invalid_plans") or []
    replan_count: int = state.get("replan_count") or 0

    hints: list[str] = []
    seen: set[str] = set()

    for plan in invalid_plans:
        steps = plan.get("steps") or []
        for v in (plan.get("violations") or []):
            if not isinstance(v, str):
                continue
            precise = _build_precise_hint(v, steps)
            if precise not in seen:
                hints.append(precise)
                seen.add(precise)

    # 兜底
    if not hints:
        reason = _resolve_replan_reason(state)
        if reason:
            hints.append(reason)

    replan_reason = "\n".join(hints) if hints else "方案校验未通过，请重新生成候选方案"
    replan_reason_type = _normalize_replan_reason_type(state)

    print(f"[Repair Loop Node] 重规划原因（第 {replan_count + 1} 次）：\n{replan_reason}")
    prev_valid = (state.get("rule_validation_result") or {}).get("valid_plans") or []
    return {
        "replan_count":           replan_count + 1,
        "replan_reason":          replan_reason,
        "replan_reason_type":     replan_reason_type,
        "candidate_plans":        {},
        "rule_validation_result": {
            "valid_plans": prev_valid,  # 保留上一轮已通过的方案
            "invalid_plans": [],
        },
        "scoring_result":         {},
    }