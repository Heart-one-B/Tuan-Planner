from __future__ import annotations
from typing import Any

# ── 常量 ──────────────────────────────────────────────────────────────────────
_DEFAULT_TRAVEL = 12
_MEAL_MIN, _MEAL_MAX, _MEAL_DEFAULT = 20, 150, 60
_ACTIVITY_MIN, _ACTIVITY_MAX = 20, 240
_LUNCH_WINDOW  = (11 * 60 + 30, 13 * 60)
_DINNER_WINDOW = (17 * 60 + 30, 19 * 60)
_EVENING_FLOOR = 19 * 60
_OVERFLOW_TOLERANCE = 15
_LUNCH_END_HARD  = 13 * 60
_DINNER_END_HARD = 19 * 60


def hhmm_to_minutes(t: str) -> int:
    try:
        h, m = map(int, t.split(":"))
        return h * 60 + m
    except Exception:
        return 10 * 60


def minutes_to_hhmm(total: int) -> str:
    total = max(0, min(total, 23 * 60 + 59))
    return f"{total // 60:02d}:{total % 60:02d}"


def calc_slot_budgets(start_time: str, end_time: str) -> dict:
    start = hhmm_to_minutes(start_time)
    end   = hhmm_to_minutes(end_time) if end_time else 21 * 60
    budgets: dict = {}
    if start < 11 * 60 + 30:
        budgets["morning"] = 11 * 60 + 30 - start
    if start < 17 * 60 + 30 and end > 13 * 60:
        budgets["afternoon"] = 17 * 60 + 30 - 13 * 60
    if end > 19 * 60:
        budgets["evening"] = max(0, end - 19 * 60)
    return budgets


def _meal_duration(step: dict) -> int:
    d = step.get("duration_minutes")
    if isinstance(d, (int, float)):
        return max(_MEAL_MIN, min(int(d), _MEAL_MAX))
    return _MEAL_DEFAULT


def _activity_duration(step: dict) -> int:
    d = step.get("duration_minutes")
    if not isinstance(d, (int, float)):
        d = 90
    return max(_ACTIVITY_MIN, min(int(d), _ACTIVITY_MAX))


def _activity_floor(step: dict, current: int) -> int:
    flex = step.get("duration_flex")
    if isinstance(flex, list) and len(flex) == 2 and isinstance(flex[0], (int, float)):
        return max(_ACTIVITY_MIN, min(int(flex[0]), current))
    return current


def _travel_to(step: dict, eta: dict) -> int:
    rec = eta.get(step.get("poi_id") or "")
    if isinstance(rec, dict) and rec.get("eta_minutes") is not None:
        return int(rec["eta_minutes"])
    return _DEFAULT_TRAVEL


def assign_step_times(
    steps: list[dict],
    start_time: str,
    end_time: str = "",
    eta: dict | None = None,
) -> list[dict]:
    """时刻求解器:把模型给的步骤结构 + 时长,计算出每步的具体开始/结束时刻。"""
    eta = eta or {}
    plan_start = hhmm_to_minutes(start_time)
    plan_end   = hhmm_to_minutes(end_time) if end_time else None
    n = len(steps)
    if n == 0:
        return []

    durations: list[int] = []
    travels:   list[int] = []
    for i, step in enumerate(steps):
        if step.get("poi_type") == "restaurant":
            durations.append(_meal_duration(step))
        else:
            durations.append(_activity_duration(step))
        travels.append(0 if i == 0 else _travel_to(step, eta))

    def _layout(durs: list[int]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = plan_start
        for i, step in enumerate(steps):
            phase = step.get("phase") or "afternoon"
            cursor += travels[i]
            if phase == "lunch":
                cursor = max(cursor, _LUNCH_WINDOW[0])
            elif phase == "dinner":
                cursor = max(cursor, _DINNER_WINDOW[0])
            elif phase == "evening":
                cursor = max(cursor, _EVENING_FLOOR)
            spans.append((cursor, cursor + durs[i]))
            cursor += durs[i]
        return spans

    spans = _layout(durations)

    # 饭点保护
    changed = True
    while changed:
        changed = False
        for i in range(len(steps) - 1):
            if steps[i].get("poi_type") == "restaurant":
                continue
            next_phase = steps[i + 1].get("phase") or ""
            if next_phase not in ("lunch", "dinner"):
                continue
            hard = _LUNCH_END_HARD if next_phase == "lunch" else _DINNER_END_HARD
            if spans[i][1] + travels[i + 1] <= hard:
                continue
            target = hard - travels[i + 1]
            new_dur = max(_activity_floor(steps[i], durations[i]), target - spans[i][0])
            if new_dur < durations[i]:
                durations[i] = new_dur
                spans = _layout(durations)
                changed = True
                break

    # 超时收口
    if plan_end is not None and spans[-1][1] > plan_end + _OVERFLOW_TOLERANCE:
        overshoot = spans[-1][1] - plan_end
        for i in range(n - 1, -1, -1):
            if overshoot <= 0:
                break
            if steps[i].get("poi_type") == "restaurant":
                continue
            floor = _activity_floor(steps[i], durations[i])
            cut = min(durations[i] - floor, overshoot)
            if cut > 0:
                durations[i] -= cut
                overshoot    -= cut
        spans = _layout(durations)
        if spans[-1][1] > plan_end:
            last_start = spans[-1][0]
            durations[-1] = max(_ACTIVITY_MIN, plan_end - last_start)
            spans = _layout(durations)

    result = []
    for i, step in enumerate(steps):
        step = dict(step)
        s, e = spans[i]
        step["start_time"]       = minutes_to_hhmm(s)
        step["end_time"]         = minutes_to_hhmm(e)
        step["duration_minutes"] = durations[i]
        result.append(step)
    return result


def build_step_index(
    steps: list[dict],
    activities_by_id: dict[str, dict],
    restaurants_by_id: dict[str, dict],
    waypoints_by_id: dict[str, dict],
) -> dict[str, Any]:
    """从 steps 构建活动/餐厅/途径点索引和时间轴。"""
    activities_out: list[dict] = []
    restaurants_out: list[dict] = []
    waypoints_out: list[dict] = []
    timeline: list[dict] = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        poi_type = step.get("poi_type") or ""
        poi_id   = step.get("poi_id") or ""
        label    = step.get("label") or (
            "活动" if poi_type == "activity"
            else "用餐" if poi_type == "restaurant"
            else "途径点"
        )

        if poi_type == "activity":
            poi = activities_by_id.get(poi_id, {})
            if poi:
                activities_out.append(poi)
        elif poi_type == "restaurant":
            poi = restaurants_by_id.get(poi_id, {})
            if poi:
                restaurants_out.append(poi)
        elif poi_type == "waypoint":
            poi = waypoints_by_id.get(poi_id, {})
            if poi:
                waypoints_out.append(poi)
        else:
            poi = {}

        timeline.append({
            "time":     step.get("start_time") or "",
            "end_time": step.get("end_time") or "",
            "item":     poi.get("name") or "待确认",
            "label":    label,
            "type":     f"{poi_type}_{step.get('phase','')}" if poi_type else step.get("phase",""),
            "ref_type": poi_type,
            "ref_id":   poi_id,
            "duration": step.get("duration_minutes"),
        })

    dinner_steps = [s for s in steps if s.get("phase") == "dinner"]
    dinner_ids   = [s.get("poi_id") for s in dinner_steps if s.get("poi_id")]
    primary_restaurant = (
        restaurants_by_id.get(dinner_ids[0])
        if dinner_ids else
        (restaurants_out[0] if restaurants_out else {})
    )

    return {
        "activities":         activities_out,
        "activity":           activities_out[0] if activities_out else {},
        "secondary_activity": activities_out[1] if len(activities_out) > 1 else {},
        "restaurants":        restaurants_out,
        "restaurant":         primary_restaurant or {},
        "waypoints":          waypoints_out,
        "timeline":           timeline,
    }

def validate_meal_phase_consistency(steps: list[dict]) -> list[dict]:
    """校验 category 和 phase/poi_type 的一致性。

    不判断任何POI的语义（那是模型的职责，通过category字段体现），
    只检查模型自己填的 category 和 phase/poi_type 是否自相矛盾——
    category != "meal" 的 step 不该占用 lunch/dinner 这两个phase。
    发现矛盾时以 category 为准修正 phase/poi_type。
    """
    fixed = []
    for step in steps:
        s = dict(step)
        category = s.get("category")
        if category and category != "meal" and s.get("phase") in ("lunch", "dinner"):
            s["phase"] = "evening" if s.get("phase") == "dinner" else "afternoon"
            s["poi_type"] = "waypoint" if category == "snack_drink" else s.get("poi_type", "activity")
        fixed.append(s)
    return fixed


def validate_and_normalize_steps(
    raw_steps: list[dict],
    activities_by_id: dict,
    restaurants_by_id: dict,
    waypoints_by_id: dict,
) -> list[dict]:
    """校验幻觉 poi_id,修正 poi_type,返回合法 steps。"""
    valid = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        poi_id   = step.get("poi_id") or ""
        poi_type = step.get("poi_type") or ""
        in_act   = poi_id in activities_by_id
        in_rest  = poi_id in restaurants_by_id
        in_wp    = poi_id in waypoints_by_id

        if not in_act and not in_rest and not in_wp:
            continue  # 幻觉,丢弃

        if poi_type == "restaurant" and not in_rest and in_act:
            step = dict(step); step["poi_type"] = "activity"
        elif poi_type == "activity" and not in_act and in_rest:
            step = dict(step); step["poi_type"] = "restaurant"

        valid.append(step)
    return valid