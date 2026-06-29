from __future__ import annotations


_OVERFLOW_TOLERANCE = 15
_LUNCH_WINDOW_END  = 13 * 60
_DINNER_WINDOW_END = 19 * 60


def _hhmm_to_minutes(t: str) -> int | None:
    if not isinstance(t, str) or ":" not in t:
        return None
    try:
        h, m = map(int, t.split(":", 1))
        return h * 60 + m
    except ValueError:
        return None


def validate_candidate(candidate: dict, plan_context: dict) -> list[str]:
    """校验单个候选方案的物理可行性，返回 violations 列表，空列表表示通过。"""
    steps = [s for s in (candidate.get("steps") or []) if isinstance(s, dict)]
    violations: list[str] = []

    start_m    = _hhmm_to_minutes(plan_context.get("start_time") or "09:00") or 0
    end_time   = plan_context.get("end_time") or ""
    plan_end_m = _hhmm_to_minutes(end_time) if end_time else 21 * 60

    has_lunch  = any(s.get("phase") == "lunch"  for s in steps)
    has_dinner = any(s.get("phase") == "dinner" for s in steps)

    if not has_lunch and start_m < 13 * 60 and plan_end_m > 12 * 60:
        violations.append("missing_lunch: 时间窗覆盖午饭时段但未安排午饭")

    if not has_dinner and start_m < 20 * 60 and plan_end_m >= 18 * 60 + 30:
        violations.append("missing_dinner: 时间窗覆盖晚饭时段但未安排晚饭")

    for step in steps:
        phase = step.get("phase") or ""
        start = _hhmm_to_minutes(step.get("start_time") or "")
        label = step.get("label") or phase
        if phase == "lunch" and start is not None and start > _LUNCH_WINDOW_END:
            violations.append(
                f"lunch_too_late: 「{label}」开始于 {step.get('start_time')}，"
                f"超出午饭窗口上限 13:00"
            )
        elif phase == "dinner" and start is not None and start > _DINNER_WINDOW_END:
            violations.append(
                f"dinner_too_late: 「{label}」开始于 {step.get('start_time')}，"
                f"超出晚饭窗口上限 19:00"
            )

    if steps and end_time:
        last       = steps[-1]
        last_end   = _hhmm_to_minutes(last.get("end_time") or "")
        last_label = last.get("label") or "最后一步"
        if last_end is not None and plan_end_m is not None:
            if last_end > plan_end_m + _OVERFLOW_TOLERANCE:
                overflow = last_end - plan_end_m
                violations.append(
                    f"overflow: 「{last_label}」结束于 {last.get('end_time')}，"
                    f"超出计划结束时间 {end_time} 共 {overflow} 分钟"
                )

    return violations


def build_rhythm_info(steps: list[dict], eta_dict: dict) -> list[str]:
    """
    把相邻步骤之间的时间间隔转成自然语言描述。
    区分通勤时间和实际等待时间，供模型在整体体验维度参考。
    """
    infos = []
    for i in range(len(steps) - 1):
        curr = steps[i]
        nxt  = steps[i + 1]
        curr_end   = _hhmm_to_minutes(curr.get("end_time") or "")
        next_start = _hhmm_to_minutes(nxt.get("start_time") or "")
        if curr_end is None or next_start is None:
            continue

        gap = next_start - curr_end
        eta = eta_dict.get(nxt.get("poi_id") or "", 0)
        if isinstance(eta, dict):
            eta = eta.get("eta_minutes") or 0
        idle = max(0, gap - eta)

        infos.append(
            f"「{curr.get('label')}」→「{nxt.get('label')}」："
            f"间隔 {gap} 分钟（含通勤 {eta} 分钟，实际等待 {idle} 分钟）"
        )
    return infos


def build_candidate_summary(
    candidate: dict,
    eta_dict: dict,
) -> dict:
    """
    把候选方案压缩成评分需要的最小信息集。
    包含时间轴摘要、平均通勤、平均评分、节奏信息。
    """
    steps       = candidate.get("steps") or []
    activities  = candidate.get("activities") or []
    restaurants = candidate.get("restaurants") or []

    # 平均通勤
    etas = [
        a.get("eta_minutes") for a in activities + restaurants
        if isinstance(a.get("eta_minutes"), (int, float))
    ]
    avg_eta = round(sum(etas) / len(etas), 1) if etas else None

    # 平均评分
    ratings = []
    for poi in activities + restaurants:
        try:
            r = float(poi.get("rating") or 0)
            if r > 0:
                ratings.append(r)
        except (ValueError, TypeError):
            pass
    avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else None

    # 时间轴摘要
    timeline_summary = [
        {
            "time":             s.get("start_time", ""),
            "end_time":         s.get("end_time", ""),
            "label":            s.get("label", ""),
            "phase":            s.get("phase", ""),
            "duration_minutes": s.get("duration_minutes"),
        }
        for s in steps
    ]

    return {
        "candidate_id":  candidate.get("id", ""),
        "title":         candidate.get("title", ""),
        "timeline":      timeline_summary,
        "avg_eta_minutes": avg_eta,
        "avg_rating":    avg_rating,
        "reasoning":     candidate.get("reasoning", []),
        "rhythm_info":   build_rhythm_info(steps, eta_dict),
    }