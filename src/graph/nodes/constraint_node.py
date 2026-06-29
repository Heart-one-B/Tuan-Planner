from __future__ import annotations

from typing import Any

from src.graph.state import AgentState


# ── 工具函数(从旧 constraint_build_node 直接搬,无修改)────────────────────────

def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _coerce_positive_int(v: Any) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return v if v > 0 else None


def _text_list(v: Any) -> list[str]:
    if isinstance(v, str):
        s = v.strip()
        return [s] if s and s not in {"无", "none", "None"} else []
    if isinstance(v, list):
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                s = item.strip()
                if s and s not in {"无", "none", "None"} and s not in out:
                    out.append(s)
        return out
    return []


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if x and not (x in seen or seen.add(x))]


def _build_plan_context(intent: dict[str, Any]) -> dict[str, Any]:
    _PARTY = {
        "family": "family_with_kids",
        "friends": "friends_group",
        "couple": "couple",
        "team": "team",
    }
    _DEFAULTS = {
        "max_traffic_minutes": 40,
        "max_queue_minutes": 30,
        "max_total_travel_minutes": 60,
    }

    # 场景与人员
    scenario = intent.get("scenario") or "unknown"
    party = _PARTY.get(scenario, "default")

    participants = _safe_dict(intent.get("participants"))
    people_count = _coerce_positive_int(participants.get("people_count"))
    has_child = bool(participants.get("has_child"))
    child_age = _coerce_positive_int(participants.get("child_age"))

    # 时间
    time_info = _safe_dict(intent.get("time"))
    date_label = time_info.get("date_label") or ""
    start_time = time_info.get("start_time") or ""
    end_time = time_info.get("end_time") or ""
    time_phrase = time_info.get("time_phrase") or ""

    # 位置
    location = _safe_dict(intent.get("location"))
    origin_area = location.get("origin_area_hint") or ""

    # 偏好
    preferences = _safe_dict(intent.get("preferences"))
    diet_preference = _text_list(preferences.get("diet_preference"))
    activity_style = _text_list(preferences.get("activity_style"))
    must_avoid = _text_list(preferences.get("must_avoid"))

    distance_raw = (preferences.get("distance_preference") or "").strip()
    distance_preference = (
        "nearby" if distance_raw in {"别太远", "附近", "近一点"} else "balanced"
    )
    pace = (
        "relaxed" if "轻松" in activity_style
        else "compact" if "紧凑" in activity_style
        else "balanced"
    )

    # 行程类型
    raw_query = intent.get("raw_query") or ""
    need_activity, need_restaurant = True, True
    if any(t in raw_query for t in ("只吃饭", "只吃个饭", "找餐厅", "约饭")):
        need_activity = False
    elif any(t in raw_query for t in ("不吃饭", "只玩", "只安排活动")):
        need_restaurant = False

    if scenario == "family" and has_child:
        plan_mode = "family_with_kids"
    elif not need_activity:
        plan_mode = "meal_only"
    elif not need_restaurant:
        plan_mode = "activity_only"
    else:
        plan_mode = "activity_plus_meal"

    # 关键词
    restaurant_keywords = _dedupe(
        _text_list(intent.get("restaurant_keywords"))
        or diet_preference
        or ["简餐", "聚餐", "特色餐厅"]
    )
    if len(restaurant_keywords) < 3:
        restaurant_keywords = _dedupe(
            restaurant_keywords + ["简餐", "聚餐", "特色餐厅"]
        )

    activity_keywords = _dedupe(
        _text_list(intent.get("activity_keywords"))
        or (
            ["儿童乐园", "科技馆", "公园", "博物馆", "商场"]
            if has_child
            else ["公园", "美术馆", "博物馆", "商场", "购物中心"]
        )
    )

    # 限制阈值
    max_traffic = _DEFAULTS["max_traffic_minutes"]
    if has_child or distance_preference == "nearby":
        max_traffic = min(max_traffic, 30)
    indoor_preferred = any(t in must_avoid for t in ("室外", "暴晒", "下雨"))

    return {
        # 基础
        "scenario": scenario,
        "plan_mode": plan_mode,
        "raw_query": raw_query,

        # 时间
        "date_label": date_label,
        "start_time": start_time,
        "end_time": end_time,
        "time_phrase": time_phrase,

        # 人员
        "people_count": people_count,
        "child_friendly": has_child,
        "child_age": child_age,

        # 位置
        "origin_area": origin_area,

        # 搜索关键词
        "activity_keywords": activity_keywords,
        "restaurant_keywords": restaurant_keywords,
        "activity_explicit_types": _text_list(intent.get("activity_explicit_types")),
        "restaurant_explicit_types": _text_list(intent.get("restaurant_explicit_types")),
        "waypoint_requests": intent.get("waypoint_requests") or [],

        # 限制
        "max_traffic_minutes": max_traffic,

        # 偏好
        "preferences": {
            "diet": diet_preference,
            "activity": activity_style,
            "avoid": must_avoid,
            "pace": pace,
            "distance": distance_preference,
        },
    }


# ── 节点 ──────────────────────────────────────────────────────────────────────

async def constraint_node(state: AgentState) -> dict:
    task_log = list(state.get("task_log") or [])
    errors = list(state.get("errors") or [])

    try:
        plan_context = _build_plan_context(intent=state.get("intent") or {})
        task_log.append(
            f"constraint: scenario={plan_context.get('scenario')} "
            f"plan_mode={plan_context.get('plan_mode')} "
            f"origin={plan_context.get('origin_area')} "
            f"start={plan_context.get('start_time')} "
            f"end={plan_context.get('end_time')}"
        )
        return {
            "plan_context": plan_context,
            "task_log": task_log,
            "errors": errors,
        }
    except Exception as e:
        errors.append({
            "node": "constraint",
            "error": str(e),
            "recoverable": False,
        })
        task_log.append(f"constraint: failed {e}")
        return {"task_log": task_log, "errors": errors}