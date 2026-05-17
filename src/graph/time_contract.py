"""Code-side time field contract for tools and downstream nodes.

All tool nodes should read time semantics only from ``constraints``.
This module centralizes the minimum required time fields for each node,
so time-related checks do not drift across the codebase.
"""

from __future__ import annotations

from typing import Mapping


TIME_SEMANTIC_FIELDS: tuple[str, ...] = (
    "date_label",
    "daypart",
    "time_phrase",
    "time_window",
    "start_time",
    "duration_hours",
)


MIN_TIME_FIELDS_BY_NODE: dict[str, tuple[str, ...]] = {
    "weather_check": ("date_label", "daypart"),
    "activity_search": ("daypart",),
    "restaurant_search": ("daypart",),
    "traffic_eta": ("date_label", "daypart"),
    "queue_check": ("time_window",),
    "crowd_risk": ("time_window",),
    "validate_plan": ("date_label", "daypart", "time_window"),
}


def missing_required_time_fields(
    constraints: Mapping[str, object] | None,
    node_name: str,
) -> list[str]:
    required_fields = MIN_TIME_FIELDS_BY_NODE.get(node_name, ())
    if not required_fields:
        return []

    if not isinstance(constraints, Mapping):
        return list(required_fields)

    missing: list[str] = []
    for field in required_fields:
        value = constraints.get(field)
        if isinstance(value, str):
            if not value.strip():
                missing.append(field)
            continue
        if value is None:
            missing.append(field)
    return missing
