from __future__ import annotations

from src.graph.state import AgentState
from src.utils.state_utils import _append_error

_MIN_TRANSITION_MINUTES = 10   # 低于此间隔视为仓促
_SPARE_GAP_MINUTES      = 90   # 高于此间隔视为有空档可加活动

# 餐饮时间窗口上限（超出才报 violation）
_LUNCH_WINDOW_END  = 13 * 60        # 午饭最晚 13:00 开始
_DINNER_WINDOW_END = 19 * 60        # 晚饭最晚 19:00 开始


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

    物理可行性（必须拦截）：
      - 相邻 step 时间重叠（gap < 0）→ time_overlap
      - 末段结束超过用户给定 end_time → overflow_end_time

    餐饮时间窗口（自适应跟随活动，超出上限才报错）：
      - lunch 开始 > 13:00 → lunch_too_late（需压缩上午活动）
      - dinner 开始 > 19:00 → dinner_too_late（需压缩下午活动）

    密度合理性（补活动）：
      - 相邻非餐饮时段间隔 >= 90 分钟 → spare_gap
      - 末段非夜间/晚餐后剩余 >= 90 分钟 → spare_tail

    注意：dinner → evening 之间不强制消化间隔，只加通勤时间即可。
    """
    print("[Rule Validation Node] 校验候选计划...")
    try:
        candidate_plans = state.get("candidate_plans") or {}
        plan_context    = state.get("plan_context") or {}
        eta: dict       = (state.get("fact_gathering_result") or {}).get("eta") or {}

        candidates: list[dict] = candidate_plans.get("candidates") or []
        end_time: str = plan_context.get("end_time") or ""

        valid_plans:   list[dict] = []
        invalid_plans: list[dict] = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            cid   = candidate.get("id") or "unknown"
            steps = [s for s in (candidate.get("steps") or []) if isinstance(s, dict)]
            violations: list[str] = []
            warnings:   list[str] = []

            # ── 规则1：相邻 step 时间合理性 ──────────────────────────────────
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
                curr_phase = curr.get("phase") or ""
                next_phase = nxt.get("phase")  or ""

                if gap < 0:
                    violations.append(
                        f"time_overlap: 「{curr_label}」结束({curr.get('end_time')})晚于"
                        f"「{next_label}」开始({nxt.get('start_time')})，时间重叠，物理不可行"
                    )
                elif gap >= _SPARE_GAP_MINUTES:
                    # dinner/evening 前的等待：dinner 是自适应的，不应出现大空档
                    # 只有非餐饮衔接才允许短暂等待
                    is_dinner_wait = next_phase == "dinner"
                    is_evening_after_dinner = (curr_phase == "dinner" and next_phase == "evening")

                    if is_evening_after_dinner:
                        # dinner → evening 不校验空档（只加通勤，不强制间隔）
                        pass
                    elif is_dinner_wait and gap <= _SPARE_GAP_MINUTES:
                        pass  # 合理等待，放行
                    else:
                        violations.append(
                            f"spare_gap: 「{curr_label}」({curr.get('end_time')}) 到"
                            f"「{next_label}」({nxt.get('start_time')}) 有 {gap} 分钟空档，"
                            f"请在此时段（{curr.get('end_time')} ~ {nxt.get('start_time')}）补充一个活动"
                        )

            # ── 规则2：餐饮时间窗口校验 ──────────────────────────────────────
            for step in steps:
                phase      = step.get("phase") or ""
                start_m    = _hhmm_to_minutes(step.get("start_time") or "")
                step_label = step.get("label") or step.get("poi_id") or phase

                if phase == "lunch" and start_m is not None and start_m > _LUNCH_WINDOW_END:
                    violations.append(
                        f"lunch_too_late: 「{step_label}」开始于 {step.get('start_time')}，"
                        f"超出午饭合理窗口上限 13:00，请压缩上午活动时长"
                    )
                elif phase == "dinner" and start_m is not None and start_m > _DINNER_WINDOW_END:
                    violations.append(
                        f"dinner_too_late: 「{step_label}」开始于 {step.get('start_time')}，"
                        f"超出晚饭合理窗口上限 19:00，请压缩下午活动时长"
                    )

            # ── 规则3：末段时间合理性 ────────────────────────────────────────
            if steps and end_time:
                last       = steps[-1]
                last_end   = _hhmm_to_minutes(last.get("end_time") or "")
                plan_end   = _hhmm_to_minutes(end_time)
                last_label = last.get("label") or f"step_{len(steps)}"
                last_phase = last.get("phase") or ""

                if last_end is not None and plan_end is not None:
                    _OVERFLOW_TOLERANCE = 15
                    if last_end > plan_end + _OVERFLOW_TOLERANCE:
                        overflow = last_end - plan_end
                        violations.append(
                            f"overflow_end_time: 「{last_label}」结束于 {last.get('end_time')}，"
                            f"超出计划结束时间 {end_time} 共 {overflow} 分钟，超出过多"
                        )
                    else:
                        remaining = plan_end - last_end
                        if remaining >= _SPARE_GAP_MINUTES and last_phase not in {"dinner", "evening"}:
                            violations.append(
                                f"spare_tail: 「{last_label}」结束后({last.get('end_time')})"
                                f"距计划结束({end_time})还有 {remaining} 分钟，请补充活动"
                            )

            print(f"[Rule Validation Node] {cid}: violations={violations}, warnings={warnings}")

            if violations:
                invalid_plans.append({
                    "candidate_id": cid,
                    "violations":   violations,
                    "warnings":     warnings,
                    # 把完整 steps 也带过去，供 repair_loop 计算时段
                    "steps":        steps,
                })
            else:
                valid_plans.append({**candidate, "warnings": warnings})

        return {
            "rule_validation_result": {
                "plan_mode":     candidate_plans.get("plan_mode", "activity_plus_meal"),
                "valid_plans":   valid_plans,
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