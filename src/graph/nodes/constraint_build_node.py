from __future__ import annotations

from typing import Any

from src.graph.state import AgentState
from src.utils.state_utils import _append_error


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
    return [x for x in items if x and not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def _build_plan_context(
    intent: dict[str, Any],
    replan_reason: str = "",
    replan_reason_type: str = "",
    runtime_origin_area: str = "",
    runtime_origin_coordinates: str = "",
) -> dict[str, Any]:
    party_map = {
        "family": "family_with_kids",
        "friends": "friends_group",
        "couple": "couple",
        "team": "team",
    }
    defaults = {
        "max_traffic_minutes": 40,
        "max_queue_minutes": 30,
        "max_total_travel_minutes": 60,
        "people_count": 2,
    }

    scenario = intent.get("scenario") or "family"
    party = party_map.get(scenario, "default")

    participants = _safe_dict(intent.get("participants"))
    people_count = _coerce_positive_int(participants.get("people_count")) or defaults["people_count"]
    has_child = bool(participants.get("has_child"))
    child_age = _coerce_positive_int(participants.get("child_age"))

    time_info = _safe_dict(intent.get("time"))
    date_label = time_info.get("date_label") or ""
    start_time = time_info.get("start_time") or ""
    end_time = time_info.get("end_time") or ""
    time_phrase = time_info.get("time_phrase") or ""

    location = _safe_dict(intent.get("location"))
    origin_area = location.get("origin_area_hint") or runtime_origin_area or ""
    origin_coordinates = runtime_origin_coordinates.strip() if isinstance(runtime_origin_coordinates, str) else ""

    preferences = _safe_dict(intent.get("preferences"))
    diet_preference = _text_list(preferences.get("diet_preference"))
    activity_style = _text_list(preferences.get("activity_style"))
    must_avoid = _text_list(preferences.get("must_avoid"))

    distance_raw = (preferences.get("distance_preference") or "").strip()
    distance_preference = "nearby" if distance_raw in {"别太远", "附近", "近一点"} else "balanced"
    pace = "relaxed" if "轻松" in activity_style else ("compact" if "紧凑" in activity_style else "balanced")

    raw_query = intent.get("raw_query") or ""
    need_activity, need_restaurant = True, True
    if any(t in raw_query for t in ("只吃饭", "只吃个饭", "找餐厅", "约饭", "聚餐")):
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

    restaurant_keywords = _dedupe(
        _text_list(intent.get("restaurant_keywords")) or diet_preference or ["简餐", "聚餐", "特色餐厅"]
    )
    if len(restaurant_keywords) < 3:
        restaurant_keywords = _dedupe(restaurant_keywords + ["简餐", "聚餐", "特色餐厅"])

    activity_explicit_types = _text_list(intent.get("activity_explicit_types"))
    base_activity_keywords = (
        _text_list(intent.get("activity_keywords"))
        or (["儿童乐园", "科技馆", "公园", "博物馆", "商场"] if has_child
            else ["公园", "美术馆", "博物馆", "商场", "购物中心"])
    )
    activity_keywords = _dedupe(activity_explicit_types + base_activity_keywords)

    max_traffic = defaults["max_traffic_minutes"]
    if has_child or distance_preference == "nearby":
        max_traffic = min(max_traffic, 30)
    indoor_preferred = any(t in must_avoid for t in ("室外", "暴晒", "下雨"))

    return {
        "scenario": scenario,
        "party": party,
        "plan_mode": plan_mode,
        "raw_query": raw_query,
        "date_label": date_label,
        "start_time": start_time,
        "end_time": end_time,
        "time_phrase": time_phrase,
        "people_count": people_count,
        "child_friendly": has_child,
        "child_age": child_age,
        "origin_area": origin_area,
        "origin_coordinates": origin_coordinates,
        "restaurant_keywords": restaurant_keywords,
        "activity_keywords": activity_keywords,
        "activity_explicit_types": activity_explicit_types,
        "exclude_restaurant": _dedupe([t for t in must_avoid if t in {"火锅", "烧烤", "烤肉", "自助"}]),
        "need_activity": need_activity,
        "need_restaurant": need_restaurant,
        "search_radius": "near" if max_traffic <= 30 else "medium",
        "max_traffic_minutes": max_traffic,
        "max_queue_minutes": defaults["max_queue_minutes"],
        "max_total_travel_minutes": defaults["max_total_travel_minutes"],
        "indoor_preferred": indoor_preferred,
        "diet_preference": diet_preference,
        "activity_style": activity_style,
        "must_avoid": must_avoid,
        "distance_preference": distance_preference,
        "pace_preference": pace,
        "replan_hints": [replan_reason] if replan_reason and replan_reason.strip() else [],
        "replan_reason_type": replan_reason_type or "",
    }


def constraint_build_node(state: AgentState) -> AgentState:
    print("[Constraint Build Node] 汇总规划约束...")
    try:
        plan_context = _build_plan_context(
            intent=state.get("intent") or {},
            replan_reason=state.get("replan_reason", ""),
            replan_reason_type=state.get("replan_reason_type", ""),
            runtime_origin_area=state.get("runtime_origin_area", ""),
            runtime_origin_coordinates=state.get("runtime_origin_coordinates", ""),
        )
        return {"plan_context": plan_context}
    except Exception as exc:
        print(f"[Constraint Build Node][ERROR] {exc}")
        return _append_error(state, f"Constraint Build failed: {exc}")
