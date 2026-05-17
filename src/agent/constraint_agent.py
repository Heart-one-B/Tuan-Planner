"""Constraint Agent：把 Intent + Retrieval + 默认策略 + Replan 反馈汇总为统一约束。

设计说明
--------
* **不调用 LLM**：纯函数实现。相同输入恒等输出。
* 输出的 ``constraints`` dict 是后续 6 个并行工具节点（weather / activities /
  restaurants / traffic / queue / crowd）的统一输入契约。
* 默认策略以 module-level ``DEFAULT_POLICY`` 暴露，便于 demo 期单点调参。
* 容错：``intent`` / ``retrieval_context`` 允许 None / 非 dict，按缺省路径产出，
  不抛异常；具体异常兜底由 ``constraint_collect_node`` 负责。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# 默认策略（demo 单点调参入口；不可在运行时被修改）。
DEFAULT_POLICY: dict[str, Any] = {
    "max_traffic_minutes": 40,
    "max_queue_minutes": 30,
    "indoor_preferred": False,
    "time_window": "today_afternoon",
    "party": "default",
    "origin_area": "area_central",
    "origin_type": "default",
    "location_source": "default",
    "start_time": "14:00",
    "duration_hours": 5,
    "people_count": 2,
}


_PARTY_BY_SCENARIO = {
    "family": "family_with_kids",
    "friends": "friends_group",
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_positive_int(value: Any) -> int | None:
    """仅接受非布尔的 int 且 > 0；其余返回 None（不覆盖默认）。"""
    if isinstance(value, bool):  # bool 是 int 子类，必须前置过滤
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _take_top_ids(container: dict[str, Any], key: str, limit: int = 3) -> list[str]:
    """从 container[key] 取前 ``limit`` 个 dict 的 id 字段。容错任意非法结构。"""
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


class ConstraintAgent:
    """Constraint Collect 纯函数 Agent：合并 intent / retrieval / 默认策略 / replan 反馈。"""

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

        # --- scenario / party ---
        scenario_raw = intent.get("scenario")
        scenario = scenario_raw if isinstance(scenario_raw, str) and scenario_raw else "family"
        party = _PARTY_BY_SCENARIO.get(scenario, "default")

        # --- time_window ---
        time_window_raw = intent.get("time_window")
        time_info = _safe_dict(intent.get("time"))
        time_window = (
            time_window_raw
            if isinstance(time_window_raw, str) and time_window_raw
            else (
                "today_afternoon"
                if isinstance(time_info.get("time_phrase"), str)
                and "下午" in time_info.get("time_phrase", "")
                else DEFAULT_POLICY["time_window"]
            )
        )
        start_time = (
            time_info.get("start_time_hint")
            if isinstance(time_info.get("start_time_hint"), str) and time_info.get("start_time_hint")
            else DEFAULT_POLICY["start_time"]
        )
        duration_hours = (
            _coerce_positive_int(time_info.get("duration_hours_hint"))
            or _coerce_positive_int(intent.get("duration_hours"))
            or DEFAULT_POLICY["duration_hours"]
        )

        # --- participants / people_count / child flags ---
        participants = _safe_dict(intent.get("participants"))
        people_count = (
            _coerce_positive_int(participants.get("people_count"))
            or _coerce_positive_int(intent.get("people_count"))
            or DEFAULT_POLICY["people_count"]
        )
        has_child = bool(participants.get("has_child")) or intent.get("child_friendly") is True
        child_age = _coerce_positive_int(participants.get("child_age"))
        child_friendly_required = has_child or intent.get("child_friendly") is True

        # --- diet_preference ---
        diet_preference = _coerce_text_list(intent.get("diet_preference"))

        # --- location / origin ---
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

        # --- max_traffic_minutes ---
        max_traffic = DEFAULT_POLICY["max_traffic_minutes"]
        intent_traffic = _coerce_positive_int(intent.get("max_traffic_minutes"))
        if intent_traffic is not None:
            max_traffic = intent_traffic
        # child_friendly 收紧：与 intent 覆盖共存时取较小值（孩子优先）
        if child_friendly_required:
            max_traffic = min(max_traffic, 30)

        # --- max_queue_minutes ---
        max_queue = DEFAULT_POLICY["max_queue_minutes"]
        intent_queue = _coerce_positive_int(intent.get("max_queue_minutes"))
        if intent_queue is not None:
            max_queue = intent_queue

        # --- indoor_preferred ---
        # 任务约定：child_friendly 不直接抬升此项，交由天气节点决定；这里固定取策略默认。
        indoor_preferred = bool(DEFAULT_POLICY["indoor_preferred"])

        # --- replan_hints ---
        replan_hints: list[str] = []
        if isinstance(replan_reason, str) and replan_reason.strip():
            replan_hints.append(replan_reason)

        # --- retrieval id 列表 ---
        retrieval_pois = _take_top_ids(retrieval_context, "pois", limit=3)
        retrieval_notes = _take_top_ids(retrieval_context, "notes", limit=3)

        constraints: dict[str, Any] = {
            "scenario": scenario,
            "party": party,
            "time_window": time_window,
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
        # deepcopy 防御：避免外部对返回 dict 的修改污染下次调用的局部变量（虽然本实现不复用，
        # 但保持纯函数语义更稳）。
        return deepcopy(constraints)
