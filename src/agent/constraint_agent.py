"""Constraint Agent: merge intent, retrieval, defaults, and replan feedback."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_POLICY: dict[str, Any] = {
    "max_traffic_minutes": 40,
    "max_queue_minutes": 30,
    "max_total_travel_minutes": 60,
    "indoor_preferred": False,
    "party": "default",
    "origin_area": "area_central",
    "origin_type": "default",
    "location_source": "default",
    "people_count": 2,
    "budget_level": "default",
    "weather_sensitivity": "normal",
    "pace_preference": "balanced",
}

_PARTY_BY_SCENARIO = {
    "family": "family_with_kids",
    "friends": "friends_group",
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _take_top_ids(container: dict[str, Any], key: str, limit: int = 3) -> list[str]:
    items = container.get(key)
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if len(out) >= limit:
            break
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id is None:
            continue
        out.append(item_id)
    return out


def _coerce_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value and value not in {"无", "none", "None"} else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if item and item not in {"无", "none", "None"} and item not in out:
                    out.append(item)
        return out
    return []


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out


def _derive_time_window(time_phrase: str) -> str:
    phrase = time_phrase or ""
    is_weekend = any(token in phrase for token in ("周末", "周六", "周日", "周天"))
    if "晚上" in phrase:
        return "weekend_evening" if is_weekend else "today_evening"
    if "下午" in phrase or "中午" in phrase or "白天" in phrase:
        return "weekend_afternoon" if is_weekend else "today_afternoon"
    return ""


class ConstraintAgent:
    def collect(
        self,
        intent: dict[str, Any] | None,
        retrieval_context: dict[str, Any] | None,
        replan_reason: str = "",
        replan_reason_type: str = "",
        runtime_origin_area: str = "",
        runtime_origin_coordinates: str = "",
    ) -> dict[str, Any]:
        intent = _safe_dict(intent)
        retrieval_context = _safe_dict(retrieval_context)

        scenario_raw = intent.get("scenario")
        scenario = scenario_raw if isinstance(scenario_raw, str) and scenario_raw else "family"
        party = _PARTY_BY_SCENARIO.get(scenario, "default")
        preferences = _safe_dict(intent.get("preferences"))

        time_info = _safe_dict(intent.get("time"))
        date_label = time_info.get("date_label") if isinstance(time_info.get("date_label"), str) else ""
        daypart = time_info.get("daypart") if isinstance(time_info.get("daypart"), str) else ""
        time_phrase = time_info.get("time_phrase") if isinstance(time_info.get("time_phrase"), str) else ""
        explicit_time_window = intent.get("time_window")
        if isinstance(explicit_time_window, str) and explicit_time_window:
            time_window = explicit_time_window
        else:
            time_window = _derive_time_window(time_phrase)

        start_time_hint = time_info.get("start_time_hint")
        start_time = start_time_hint.strip() if isinstance(start_time_hint, str) and start_time_hint.strip() else ""
        duration_hours = _coerce_positive_int(time_info.get("duration_hours_hint")) or _coerce_positive_int(intent.get("duration_hours"))

        participants = _safe_dict(intent.get("participants"))
        people_count = (
            _coerce_positive_int(participants.get("people_count"))
            or _coerce_positive_int(intent.get("people_count"))
            or DEFAULT_POLICY["people_count"]
        )
        has_child = bool(participants.get("has_child")) or intent.get("child_friendly") is True
        child_age = _coerce_positive_int(participants.get("child_age"))
        child_friendly_required = has_child or intent.get("child_friendly") is True

        diet_preference = _coerce_text_list(intent.get("diet_preference"))
        raw_query = intent.get("raw_query")
        raw_diet_keywords: list[str] = []
        if isinstance(raw_query, str):
            for token in ("火锅", "烧烤", "烤肉", "西餐", "日料", "韩餐", "川菜", "粤菜", "湘菜", "轻食", "自助"):
                if token in raw_query and token not in raw_diet_keywords:
                    raw_diet_keywords.append(token)
        merged_diet_preference = _dedupe_keep_order(diet_preference + raw_diet_keywords)

        distance_preference_raw = preferences.get("distance_preference")
        if not isinstance(distance_preference_raw, str):
            distance_preference_raw = ""
        distance_preference_raw = distance_preference_raw.strip()
        if distance_preference_raw in {"别太远", "附近", "近一点"}:
            distance_preference = "nearby"
        elif distance_preference_raw:
            distance_preference = "balanced"
        else:
            distance_preference = "balanced"

        activity_style = _coerce_text_list(preferences.get("activity_style"))
        must_avoid = _coerce_text_list(preferences.get("must_avoid"))

        pace_preference = DEFAULT_POLICY["pace_preference"]
        if "轻松" in activity_style:
            pace_preference = "relaxed"
        elif "紧凑" in activity_style:
            pace_preference = "compact"

        location_info = _safe_dict(intent.get("location"))
        origin_area_hint = location_info.get("origin_area_hint")
        if isinstance(origin_area_hint, str) and origin_area_hint:
            origin_area = origin_area_hint
            origin_type = "home"
            location_source = "user_provided"
        elif isinstance(runtime_origin_area, str) and runtime_origin_area.strip():
            origin_area = runtime_origin_area.strip()
            origin_type = "current"
            location_source = "runtime_location"
        else:
            origin_area = ""
            origin_type = DEFAULT_POLICY["origin_type"]
            location_source = DEFAULT_POLICY["location_source"]

        max_traffic = DEFAULT_POLICY["max_traffic_minutes"]
        intent_traffic = _coerce_positive_int(intent.get("max_traffic_minutes"))
        if intent_traffic is not None:
            max_traffic = intent_traffic
        if child_friendly_required:
            max_traffic = min(max_traffic, 30)

        max_queue = DEFAULT_POLICY["max_queue_minutes"]
        intent_queue = _coerce_positive_int(intent.get("max_queue_minutes"))
        if intent_queue is not None:
            max_queue = intent_queue

        indoor_preferred = bool(DEFAULT_POLICY["indoor_preferred"])
        if any(token in must_avoid for token in ("室外", "暴晒", "下雨")):
            indoor_preferred = True

        replan_hints: list[str] = []
        if isinstance(replan_reason, str) and replan_reason.strip():
            replan_hints.append(replan_reason)

        retrieval_pois = _take_top_ids(retrieval_context, "pois", limit=3)
        retrieval_notes = _take_top_ids(retrieval_context, "notes", limit=3)
        next_constraint_build = _safe_dict(retrieval_context.get("next_constraint_build"))

        raw_query_text = raw_query if isinstance(raw_query, str) else ""
        query_text = raw_query_text.strip()
        need_activity = True
        need_restaurant = True
        if any(token in query_text for token in ("只吃饭", "只吃个饭", "找餐厅", "约饭", "聚餐")):
            need_activity = False
            need_restaurant = True
        elif any(token in query_text for token in ("不吃饭", "不用吃饭", "只玩", "只安排活动")):
            need_activity = True
            need_restaurant = False

        request_type = "generic_local_plan"
        if scenario == "family" and child_friendly_required:
            request_type = "family_with_kids"
        elif scenario == "friends":
            request_type = "friends_social"
        elif not need_activity and need_restaurant:
            request_type = "meal_only"
        elif need_activity and not need_restaurant:
            request_type = "activity_first"

        plan_mode = "activity_plus_meal"
        if not need_activity and need_restaurant:
            plan_mode = "meal_only"
        elif need_activity and not need_restaurant:
            plan_mode = "activity_only"
        elif not merged_diet_preference and scenario == "friends":
            plan_mode = "light_social"

        search_radius_level = "near" if max_traffic <= 30 or distance_preference == "nearby" else "medium"
        queue_sensitivity = "high" if child_friendly_required or max_queue <= 20 else "medium"

        restaurant_keywords = _coerce_text_list(intent.get("restaurant_keywords"))
        activity_keywords = _coerce_text_list(intent.get("activity_keywords"))
        activity_search_keywords = _coerce_text_list(intent.get("activity_search_keywords"))
        restaurant_explicit_types = _coerce_text_list(intent.get("restaurant_explicit_types"))
        activity_explicit_types = _coerce_text_list(intent.get("activity_explicit_types"))
        if not restaurant_keywords:
            restaurant_keywords = _coerce_text_list(preferences.get("restaurant_keywords"))
        if not activity_keywords:
            activity_keywords = _coerce_text_list(preferences.get("activity_keywords"))
        if not activity_search_keywords:
            activity_search_keywords = _coerce_text_list(preferences.get("activity_search_keywords"))
        if not restaurant_explicit_types:
            restaurant_explicit_types = _coerce_text_list(preferences.get("restaurant_explicit_types"))
        if not activity_explicit_types:
            activity_explicit_types = _coerce_text_list(preferences.get("activity_explicit_types"))

        restaurant_keywords = _dedupe_keep_order(restaurant_keywords)
        activity_keywords = _dedupe_keep_order(activity_keywords)
        activity_search_keywords = _dedupe_keep_order(activity_search_keywords)
        restaurant_explicit_types = _dedupe_keep_order(restaurant_explicit_types)
        activity_explicit_types = _dedupe_keep_order(activity_explicit_types)

        if not restaurant_keywords:
            restaurant_keywords = _dedupe_keep_order(merged_diet_preference[:])
        if len(restaurant_keywords) < 3:
            restaurant_keywords.extend(["简餐", "聚餐", "特色餐厅"])
        if not activity_keywords:
            activity_keywords = ["科技馆", "博物馆", "儿童乐园", "公园", "商场"]
        if not activity_search_keywords:
            activity_search_keywords = activity_keywords[:]

        exclude_keywords_restaurant: list[str] = []
        if any(token in must_avoid for token in ("火锅", "烧烤", "烤肉", "自助")):
            exclude_keywords_restaurant.extend([token for token in ("火锅", "烧烤", "烤肉", "自助") if token in must_avoid])
        if any(token in raw_query_text for token in ("不吃火锅", "不吃烧烤")):
            exclude_keywords_restaurant.extend([token for token in ("火锅", "烧烤") if token not in exclude_keywords_restaurant])

        preferred_activity_tags = _dedupe_keep_order(activity_style + activity_explicit_types)
        mapped_restaurant_keywords = _dedupe_keep_order(restaurant_keywords + restaurant_explicit_types)
        mapped_activity_keywords = _dedupe_keep_order(activity_keywords + activity_explicit_types)

        constraints: dict[str, Any] = {
            "scenario": scenario,
            "party": party,
            "time_window": time_window,
            "date_label": date_label,
            "daypart": daypart,
            "time_phrase": time_phrase,
            "start_time": start_time,
            "duration_hours": duration_hours,
            "people_count": people_count,
            "child_friendly_required": child_friendly_required,
            "child_age": child_age,
            "origin_area": origin_area,
            "origin_coordinates": runtime_origin_coordinates.strip() if isinstance(runtime_origin_coordinates, str) else "",
            "origin_type": origin_type,
            "location_source": location_source,
            "diet_preference": merged_diet_preference,
            "restaurant_keywords": restaurant_keywords,
            "activity_keywords": activity_keywords,
            "activity_search_keywords": activity_search_keywords,
            "restaurant_explicit_types": restaurant_explicit_types,
            "activity_explicit_types": activity_explicit_types,
            "max_traffic_minutes": max_traffic,
            "max_queue_minutes": max_queue,
            "max_total_travel_minutes": DEFAULT_POLICY["max_total_travel_minutes"],
            "indoor_preferred": indoor_preferred,
            "replan_hints": replan_hints,
            "retrieval_pois": retrieval_pois,
            "retrieval_notes": retrieval_notes,
            "replan_reason_type": replan_reason_type if isinstance(replan_reason_type, str) else "",
            "need_activity": need_activity,
            "need_restaurant": need_restaurant,
        }

        hard_constraints: dict[str, Any] = {
            "time_window": time_window,
            "date_label": date_label,
            "daypart": daypart,
            "start_time": start_time,
            "duration_hours": duration_hours,
            "latest_end_time": "",
            "origin_area": origin_area,
            "origin_coordinates": runtime_origin_coordinates.strip() if isinstance(runtime_origin_coordinates, str) else "",
            "origin_type": origin_type,
            "party_size": people_count,
            "child_friendly_required": child_friendly_required,
            "child_age": child_age,
            "activity_required": need_activity,
            "restaurant_required": need_restaurant,
            "must_open_at_arrival": True,
            "must_open_during_stay": True,
            "max_traffic_minutes": max_traffic,
            "max_total_travel_minutes": DEFAULT_POLICY["max_total_travel_minutes"],
            "max_queue_minutes": max_queue,
            "max_budget_total": None,
            "max_budget_per_person": None,
            "indoor_only": bool(indoor_preferred),
            "avoid_high_risk_weather": True,
            "dietary_must_match": False,
            "must_avoid_poi_ids": [],
        }

        soft_preferences: dict[str, Any] = {
            "diet_preference": merged_diet_preference,
            "activity_style": activity_style,
            "atmosphere_preference": ["适合家庭"] if scenario == "family" else (["适合社交"] if scenario == "friends" else []),
            "must_avoid": must_avoid,
            "budget_level": DEFAULT_POLICY["budget_level"],
            "weather_sensitivity": DEFAULT_POLICY["weather_sensitivity"],
            "queue_sensitivity": queue_sensitivity,
            "distance_preference": distance_preference,
            "photo_friendly_preferred": False,
            "indoor_preferred": indoor_preferred,
            "restaurant_cuisine_preference": mapped_restaurant_keywords,
            "activity_tag_preference": preferred_activity_tags,
            "pace_preference": pace_preference,
            "review_score_preference": "high",
        }

        query_constraints: dict[str, Any] = {
            "origin_area": origin_area,
            "origin_coordinates": runtime_origin_coordinates.strip() if isinstance(runtime_origin_coordinates, str) else "",
            "time_window": time_window,
            "party_size": people_count,
            "need_activity": need_activity,
            "need_restaurant": need_restaurant,
            "search_radius_level": search_radius_level,
            "indoor_preferred": bool(indoor_preferred),
            "max_candidate_eta_minutes": max_traffic,
            "max_candidate_queue_minutes": max_queue,
            "child_friendly_preferred": child_friendly_required,
            "budget_level": DEFAULT_POLICY["budget_level"],
            "keywords_activity": mapped_activity_keywords,
            "activity_search_keywords": activity_search_keywords,
            "keywords_restaurant": mapped_restaurant_keywords or merged_diet_preference,
            "preferred_cuisines": mapped_restaurant_keywords,
            "preferred_activity_tags": preferred_activity_tags,
            "exclude_poi_ids": [],
            "exclude_keywords_restaurant": exclude_keywords_restaurant,
            "weather_guard": "indoor_preferred" if indoor_preferred else "normal",
        }

        validation_profile: dict[str, Any] = {
            "check_time_feasibility": True,
            "check_weather_compatibility": True,
            "check_opening_hours": True,
            "check_eta_threshold": True,
            "check_total_travel_minutes": True,
            "check_queue_threshold": True,
            "check_budget_limit": False,
            "check_party_fit": True,
            "check_indoor_outdoor_conflict": True,
        }

        scoring_profile: dict[str, Any] = {
            "weights": {
                "quality_score": 0.22,
                "convenience_score": 0.20,
                "comfort_score": 0.20,
                "preference_rule_score": 0.18,
                "semantic_match_score": 0.10,
                "risk_penalty": 0.10,
            },
            "prefer_short_distance": distance_preference == "nearby",
            "prefer_low_queue": max_queue <= 20,
            "prefer_indoor_when_bad_weather": True,
            "semantic_match_enabled": True,
        }

        context_memory: dict[str, Any] = {
            "raw_query": raw_query_text,
            "retrieval_pois": retrieval_pois,
            "retrieval_notes": retrieval_notes,
            "replan_hints": replan_hints,
            "replan_reason_type": replan_reason_type if isinstance(replan_reason_type, str) else "",
            "defaults_applied": [],
            "location_source": location_source,
            "origin_coordinates": runtime_origin_coordinates.strip() if isinstance(runtime_origin_coordinates, str) else "",
        }

        if not start_time:
            context_memory["defaults_applied"].append("start_time:not_provided")
        if duration_hours is None:
            context_memory["defaults_applied"].append("duration_hours:not_provided")
        if not origin_area:
            context_memory["defaults_applied"].append("origin_area:empty")

        if next_constraint_build:
            hard_constraints.update(_safe_dict(next_constraint_build.get("hard_constraints")))
            soft_preferences.update(_safe_dict(next_constraint_build.get("soft_preferences")))
            query_constraints.update(_safe_dict(next_constraint_build.get("query_constraints")))
            validation_profile.update(_safe_dict(next_constraint_build.get("validation_profile")))
            scoring_profile.update(_safe_dict(next_constraint_build.get("scoring_profile")))
            context_memory.update(_safe_dict(next_constraint_build.get("context_memory")))
            request_type = next_constraint_build.get("request_type", request_type)
            plan_mode = next_constraint_build.get("plan_mode", plan_mode)

        return {
            "constraints": deepcopy(constraints),
            "constraint_build": {
                "request_type": request_type,
                "plan_mode": plan_mode,
                "hard_constraints": deepcopy(hard_constraints),
                "soft_preferences": deepcopy(soft_preferences),
                "query_constraints": deepcopy(query_constraints),
                "validation_profile": deepcopy(validation_profile),
                "scoring_profile": deepcopy(scoring_profile),
                "context_memory": deepcopy(context_memory),
            },
        }
