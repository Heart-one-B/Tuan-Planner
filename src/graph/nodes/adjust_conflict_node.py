from __future__ import annotations

from src.graph.state import AgentState

# 校验码 → 人话模板
_VIOLATION_HINTS = {
    "time_overlap":      "时间安排上有重叠",
    "overflow_end_time": "整体时间超出了你设定的结束时间",
    "lunch_too_late":    "午饭时间被推得太晚（超过 13:00）",
    "dinner_too_late":   "晚饭时间被推得太晚（超过 19:00）",
    "missing_lunch":     "缺少午饭安排",
    "missing_dinner":    "缺少晚饭安排",
}


def _humanize_violations(invalid_plans: list[dict]) -> list[str]:
    """把所有方案的 violations 去重、转成人话。"""
    seen: set[str] = set()
    hints: list[str] = []
    for plan in invalid_plans:
        for v in (plan.get("violations") or []):
            if not isinstance(v, str):
                continue
            code = v.split(":", 1)[0].strip()
            hint = _VIOLATION_HINTS.get(code)
            if hint and hint not in seen:
                seen.add(hint)
                hints.append(hint)
            elif not hint:
                # 没匹配到模板，直接用原始描述（去掉 code 前缀）
                desc = v.split(":", 1)[1].strip() if ":" in v else v
                if desc and desc not in seen:
                    seen.add(desc)
                    hints.append(desc)
    return hints


def adjust_conflict_node(state: AgentState) -> AgentState:
    """
    Adjust Conflict Node：用户对方案的修改导致时间冲突时，告知用户，让用户自己衡量。

    只在 adjust 模式校验失败时触发。不自动修复，把冲突讲清楚，交还决定权给用户。
    """
    print("[Adjust Conflict Node] 用户修改导致冲突，生成提示...")

    feedback_summary = (state.get("feedback_summary") or "你的修改").strip()
    result = state.get("rule_validation_result") or {}
    invalid_plans = result.get("invalid_plans") or []

    hints = _humanize_violations(invalid_plans)

    if hints:
        conflict_desc = "；".join(hints)
        message = (
            f"按你说的「{feedback_summary}」调整后，发现一个问题：{conflict_desc}。\n\n"
            f"这样改会让行程时间排不开。你可以考虑：\n"
            f"- 换一个时长更短的选择\n"
            f"- 去掉行程里的某个活动腾出时间\n"
            f"- 或者保持原方案不改\n\n"
            f"你想怎么调整？告诉我就行。"
        )
    else:
        message = (
            f"按你说的「{feedback_summary}」调整后，行程时间有点排不开。\n"
            f"你可以换个时长更短的选择，或者去掉某个活动腾时间。你想怎么调整？"
        )

    print(f"[Adjust Conflict Node] 冲突提示：{hints}")

    return {
        "final_message": message,
    }