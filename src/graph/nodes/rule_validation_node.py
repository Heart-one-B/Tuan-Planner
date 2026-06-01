from __future__ import annotations

from src.graph.state import AgentState
from src.utils.state_utils import _append_error

_MIN_TRANSITION_MINUTES = 10   # 低于此间隔视为仓促
_SPARE_GAP_MINUTES      = 90   # 高于此间隔视为有空档可加活动


def _hhmm_to_minutes(t: str) -> int | None:
    if not isinstance(t, str) or ":" not in t:
        return None
    try:
        h, m = map(int, t.split(":", 1))
        return h * 60 + m
    except ValueError:
        return None



def rule_validation_node(state: AgentState) -> AgentState:
    """
    Rule Validation Node：对候选方案进行规则校验。

    规则：时间合理性
      - 相邻 step 间隔 < 10 分钟 → violation（仓促，物理不可行）
      - 相邻 step 间隔 or 末尾剩余 >= 90 分钟 → violation（有空档，打回重规划）

    天气与室内外的判断完全交给 candidate_planning_node 的 LLM 处理。
    有 violation → invalid_plans；无问题 → valid_plans。
    """
    print("[Rule Validation Node] 校验候选计划...")
    try:
        candidate_plans     = state.get("candidate_plans") or {}
        fact_gathering_result = state.get("fact_gathering_result") or {}
        plan_context        = state.get("plan_context") or {}

        candidates: list[dict] = candidate_plans.get("candidates") or []
        end_time: str = plan_context.get("end_time") or ""

        valid_plans:   list[dict] = []
        invalid_plans: list[dict] = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            cid      = candidate.get("id") or "unknown"
            steps    = [s for s in (candidate.get("steps") or []) if isinstance(s, dict)]
            violations: list[str] = []
            warnings:   list[str] = []

            # ── 规则1：时间合理性 ─────────────────────────────────────────────
            for i in range(len(steps) - 1):
                curr = steps[i]
                nxt  = steps[i + 1]
                curr_end   = _hhmm_to_minutes(curr.get("end_time") or "")
                next_start = _hhmm_to_minutes(nxt.get("start_time") or "")

                if curr_end is None or next_start is None:
                    continue

                gap = next_start - curr_end
                curr_label = curr.get("label") or curr.get("poi_id") or f"step_{i+1}"
                next_label = nxt.get("label")  or nxt.get("poi_id")  or f"step_{i+2}"

                if gap < _MIN_TRANSITION_MINUTES:
                    violations.append(
                        f"transition_too_rushed: 「{curr_label}」结束到「{next_label}」开始仅 {gap} 分钟，时间仓促"
                    )
                elif gap >= _SPARE_GAP_MINUTES:
                    curr_phase = steps[i].get("phase") or ""
                    next_phase = steps[i + 1].get("phase") or ""
                    # 以下属于合理过渡，不触发 violation：
                    # - 下一个是 dinner（晚饭前休息）
                    # - 下一个是 evening 或当前是 dinner（dinner→evening 的消化时间）
                    if next_phase in {"dinner", "evening"} or curr_phase == "dinner":
                        pass
                    else:
                        curr_end_str   = steps[i].get("end_time") or ""
                        next_start_str = steps[i + 1].get("start_time") or ""
                        violations.append(
                            f"spare_gap: 「{curr_label}」({curr_end_str}) 到「{next_label}」({next_start_str}) 有 {gap} 分钟空档，"
                            f"请在此时段补充一个活动"
                        )

            # 检查最后一个 step 到计划结束的剩余时间
            if steps and end_time:
                last_end  = _hhmm_to_minutes(steps[-1].get("end_time") or "")
                plan_end  = _hhmm_to_minutes(end_time)
                if last_end is not None and plan_end is not None:
                    remaining = plan_end - last_end
                    last_phase = steps[-1].get("phase") or ""
                    # 晚饭/夜间活动之后的剩余时间是自由时间，不强制补活动
                    if remaining >= _SPARE_GAP_MINUTES and last_phase not in {"dinner", "evening"}:
                        last_label = steps[-1].get("label") or f"step_{len(steps)}"
                        violations.append(
                            f"spare_tail: 「{last_label}」结束后({steps[-1].get('end_time')})距计划结束({end_time})"
                            f"还有 {remaining} 分钟，请补充活动"
                        )

            print(
                f"[Rule Validation Node] {cid}: "
                f"violations={violations}, warnings={warnings}"
            )

            if violations:
                invalid_plans.append({
                    "candidate_id": cid,
                    "violations":   violations,
                    "warnings":     warnings,
                })
            else:
                valid_plans.append({
                    **candidate,
                    "warnings": warnings,
                })

        return {
            "rule_validation_result": {
                "plan_mode":    candidate_plans.get("plan_mode", "activity_plus_meal"),
                "valid_plans":  valid_plans,
                "invalid_plans": invalid_plans,
            }
        }

    except Exception as exc:
        print(f"[Rule Validation Node][ERROR] {exc}")
        update = _append_error(state, f"Rule Validation failed: {exc}")
        update["rule_validation_result"] = {
            "plan_mode":     "activity_plus_meal",
            "valid_plans":   [],
            "invalid_plans": [],
        }
        return update