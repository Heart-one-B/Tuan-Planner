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
            origin_area = DEFAULT_POLICY["origin_area"]
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
            "origin_type": origin_type,
            "location_source": location_source,
            "diet_preference": diet_preference,
            "max_traffic_minutes": max_traffic,
            "max_queue_minutes": max_queue,
            "indoor_preferred": indoor_preferred,
            "replan_hints": replan_hints,
            "retrieval_pois": retrieval_pois,
            "retrieval_notes": retrieval_notes,
            "replan_reason_type": replan_reason_type if isinstance(replan_reason_type, str) else "",
        }
        return deepcopy(constraints)
