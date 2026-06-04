from __future__ import annotations

from src.graph.state import AgentState
from src.utils.state_utils import _append_error

_OVERFLOW_TOLERANCE = 15        # 末段超出 end_time 的容忍（用户可自行提前结束）

# 餐饮合法窗口上限（超出才报错）
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
    Rule Validation Node（v3）：配合"模型估真实时长"的求解器。

    物理可行性（全部规则）：
      - time_overlap        : 相邻 step 时间重叠（gap < 0），物理不可行
      - overflow_end_time   : 末段结束超出用户 end_time 超过容忍值
      - lunch_too_late      : 午饭开始 > 13:00（上午活动太多顶飞午饭）
      - dinner_too_late     : 晚饭开始 > 19:00（下午活动太多顶飞晚饭）

    已删除：spare_gap
      模型现在给真实时长估计，活动间的自然空档（等饭点、通勤缓冲、轻松留白）
      不是物理错误，不应触发 repair_loop。模型自检（System Prompt 时段覆盖自检）
      负责主动预防"下午排太少"的语义问题，这里只守物理边界。
    """
    print("[Rule Validation Node] 校验候选计划...")
    try:
        candidate_plans = state.get("candidate_plans") or {}
        plan_context    = state.get("plan_context") or {}

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

            # ── 规则1：相邻 step 时间合理性（重叠 / dinner 前大空档）──────────
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
                next_phase = nxt.get("phase")  or ""
                curr_phase = curr.get("phase") or ""

                if gap < 0:
                    violations.append(
                        f"time_overlap: 「{curr_label}」结束({curr.get('end_time')})晚于"
                        f"「{next_label}」开始({nxt.get('start_time')})，时间重叠，物理不可行"
                    )
                # gap >= 0 的情况（含自然等待、留白）一律放行，不报 spare_gap

            # ── 规则2：餐饮时间窗口 + 必须项检查 ──────────────────────────────
            plan_start_m = _hhmm_to_minutes(plan_context.get("start_time") or "09:00")
            plan_end_m   = _hhmm_to_minutes(end_time) if end_time else 21 * 60
            has_lunch  = any(s.get("phase") == "lunch"  for s in steps)
            has_dinner = any(s.get("phase") == "dinner" for s in steps)

            # 时间窗跨越午饭时段但没有 lunch step
            if not has_lunch and plan_start_m < 13 * 60 and plan_end_m > 12 * 60:
                violations.append(
                    "missing_lunch: 时间窗覆盖午饭时段但未安排午饭，请补充一个 lunch 餐厅"
                )
            # 时间窗跨越晚饭时段但没有 dinner step
            if not has_dinner and plan_start_m < 20 * 60 and plan_end_m >= 18 * 60 + 30:
                violations.append(
                    "missing_dinner: 时间窗覆盖晚饭时段但未安排晚饭，请补充一个 dinner 餐厅"
                )

            for step in steps:
                phase   = step.get("phase") or ""
                start_m = _hhmm_to_minutes(step.get("start_time") or "")
                label   = step.get("label") or step.get("poi_id") or phase
                if phase == "lunch" and start_m is not None and start_m > _LUNCH_WINDOW_END:
                    violations.append(
                        f"lunch_too_late: 「{label}」开始于 {step.get('start_time')}，"
                        f"超出午饭窗口上限 13:00，请压缩或减少上午活动"
                    )
                elif phase == "dinner" and start_m is not None and start_m > _DINNER_WINDOW_END:
                    violations.append(
                        f"dinner_too_late: 「{label}」开始于 {step.get('start_time')}，"
                        f"超出晚饭窗口上限 19:00，请压缩或减少下午活动"
                    )

            # ── 规则3：末段超出 end_time ────────────────────────────────────
            if steps and end_time:
                last     = steps[-1]
                last_end = _hhmm_to_minutes(last.get("end_time") or "")
                plan_end = _hhmm_to_minutes(end_time)
                last_label = last.get("label") or f"step_{len(steps)}"
                if last_end is not None and plan_end is not None:
                    if last_end > plan_end + _OVERFLOW_TOLERANCE:
                        overflow = last_end - plan_end
                        violations.append(
                            f"overflow_end_time: 「{last_label}」结束于 {last.get('end_time')}，"
                            f"超出计划结束时间 {end_time} 共 {overflow} 分钟"
                        )

            print(f"[Rule Validation Node] {cid}: violations={violations}, warnings={warnings}")

            if violations:
                invalid_plans.append({
                    "candidate_id": cid,
                    "violations":   violations,
                    "warnings":     warnings,
                    "steps":        steps,   # 带给 repair_loop 计算精确时段
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