"""Constraint Agent: merge intent, retrieval, defaults, and replan feedback."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_POLICY: dict[str, Any] = {
    "max_traffic_minutes": 40,
    "max_queue_minutes": 30,
    "indoor_preferred": False,
    "party": "default",
    "origin_area": "area_central",
    "origin_type": "default",
    "location_source": "default",
    "people_count": 2,
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


def _derive_time_window(time_phrase: str) -> str:
    phrase = time_phrase or ""
    is_weekend = any(token in phrase for token in ("周末", "周六", "周日", "周天"))
    if "晚上" in phrase or "今晚" in phrase:
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

        time_info = _safe_dict(intent.get("time"))
        date_label = time_info.get("date_label")
        if not isinstance(date_label, str):
            date_label = ""
        daypart = time_info.get("daypart")
        if not isinstance(daypart, str):
            daypart = ""
        time_phrase = time_info.get("time_phrase")
        if not isinstance(time_phrase, str):
            time_phrase = ""
        explicit_time_window = intent.get("time_window")
        if isinstance(explicit_time_window, str) and explicit_time_window:
            time_window = explicit_time_window
        else:
            time_window = _derive_time_window(time_phrase)

        start_time_hint = time_info.get("start_time_hint")
        start_time = start_time_hint.strip() if isinstance(start_time_hint, str) and start_time_hint.strip() else ""
        duration_hours = (
            _coerce_positive_int(time_info.get("duration_hours_hint"))
            or _coerce_positive_int(intent.get("duration_hours"))
        )

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
            for token in ("烤肉", "烧烤", "火锅", "西餐", "日料", "韩餐", "川菜", "粤菜", "湘菜", "轻食", "自助"):
                if token in raw_query and token not in raw_diet_keywords:
                    raw_diet_keywords.append(token)
        merged_diet_preference = diet_preference[:]
        for token in raw_diet_keywords:
            if token not in merged_diet_preference:
                merged_diet_preference.append(token)

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

        replan_hints: list[str] = []
        if isinstance(replan_reason, str) and replan_reason.strip():
            replan_hints.append(replan_reason)

        retrieval_pois = _take_top_ids(retrieval_context, "pois", limit=3)
        retrieval_notes = _take_top_ids(retrieval_context, "notes", limit=3)

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
            "max_traffic_minutes": max_traffic,
            "max_queue_minutes": max_queue,
            "indoor_preferred": indoor_preferred,
            "replan_hints": replan_hints,
            "retrieval_pois": retrieval_pois,
            "retrieval_notes": retrieval_notes,
            "replan_reason_type": replan_reason_type if isinstance(replan_reason_type, str) else "",
        }

        request_type = "generic_local_plan"
        if scenario == "family" and child_friendly_required:
            request_type = "family_with_kids"
        elif scenario == "friends":
            request_type = "friends_social"

        plan_mode = "activity_plus_meal"
        if not merged_diet_preference and scenario == "friends":
            plan_mode = "light_social"

        hard_constraints: dict[str, Any] = {
            "time_window": time_window,
            "date_label": date_label,
            "daypart": daypart,
            "start_time": start_time,
            "duration_hours": duration_hours,
            "origin_area": origin_area,
            "origin_coordinates": runtime_origin_coordinates.strip() if isinstance(runtime_origin_coordinates, str) else "",
            "origin_type": origin_type,
            "party_size": people_count,
            "child_friendly_required": child_friendly_required,
            "child_age": child_age,
            "max_traffic_minutes": max_traffic,
            "max_queue_minutes": max_queue,
            "indoor_only": bool(indoor_preferred),
            "dietary_must_match": False,
        }

        soft_preferences: dict[str, Any] = {
            "diet_preference": merged_diet_preference,
            "activity_style": [],
            "must_avoid": [],
            "budget_level": "default",
            "weather_sensitivity": "normal",
            "queue_sensitivity": "high" if child_friendly_required else "medium",
            "distance_preference": "nearby" if max_traffic <= 30 else "default",
            "photo_friendly_preferred": False,
            "atmosphere_preferred": scenario == "friends",
        }

        query_constraints: dict[str, Any] = {
            "origin_area": origin_area,
            "origin_coordinates": runtime_origin_coordinates.strip() if isinstance(runtime_origin_coordinates, str) else "",
            "time_window": time_window,
            "party_size": people_count,
            "need_activity": True,
            "need_restaurant": True,
            "search_radius_level": "near" if max_traffic <= 30 else "default",
            "indoor_preferred": bool(indoor_preferred),
            "keywords_activity": [],
            "keywords_restaurant": merged_diet_preference,
        }

        validation_profile: dict[str, Any] = {
            "check_time_feasibility": True,
            "check_weather_compatibility": True,
            "check_opening_hours": True,
            "check_eta_threshold": True,
            "check_queue_threshold": True,
            "check_party_fit": True,
            "check_indoor_outdoor_conflict": True,
        }

        scoring_profile: dict[str, Any] = {
            "weights": {
                "semantic_match": 0.30,
                "time_relaxation": 0.20,
                "weather_fit": 0.15,
                "distance_fit": 0.15,
                "queue_fit": 0.10,
                "review_quality": 0.10,
            },
            "prefer_short_distance": max_traffic <= 30,
            "prefer_low_queue": max_queue <= 20,
            "prefer_indoor_when_bad_weather": True,
        }

        context_memory: dict[str, Any] = {
            "raw_query": intent.get("raw_query", "") if isinstance(intent.get("raw_query"), str) else "",
            "retrieval_pois": retrieval_pois,
            "retrieval_notes": retrieval_notes,
            "replan_hints": replan_hints,
            "replan_reason_type": replan_reason_type if isinstance(replan_reason_type, str) else "",
            "defaults_applied": [],
            "location_source": location_source,
            "origin_coordinates": runtime_origin_coordinates.strip() if isinstance(runtime_origin_coordinates, str) else "",
        }

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
