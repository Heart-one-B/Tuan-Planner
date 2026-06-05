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


def _merge_keywords(existing: list[str], new: list[str]) -> list[str]:
    """去重追加新关键词到已有列表。"""
    seen = set(existing)
    return existing + [k for k in new if k and k not in seen]


def _build_plan_context(
    intent: dict[str, Any],
    replan_reason: str = "",
    replan_reason_type: str = "",
    runtime_origin_area: str = "",
    runtime_origin_coordinates: str = "",
) -> dict[str, Any]:

    _PARTY = {"family": "family_with_kids", "friends": "friends_group",
               "couple": "couple", "team": "team"}
    _DEFAULTS = {"max_traffic_minutes": 40, "max_queue_minutes": 30,
                 "max_total_travel_minutes": 60}

    # 场景与人员
    scenario = intent.get("scenario") or "family"
    party = _PARTY.get(scenario, "default")

    participants = _safe_dict(intent.get("participants"))
    people_count = _coerce_positive_int(participants.get("people_count")) or _DEFAULTS["people_count"]
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
    origin_area = location.get("origin_area_hint") or runtime_origin_area or ""
    origin_coordinates = runtime_origin_coordinates.strip() if isinstance(runtime_origin_coordinates, str) else ""

    # 偏好
    preferences = _safe_dict(intent.get("preferences"))
    diet_preference = _text_list(preferences.get("diet_preference"))
    activity_style = _text_list(preferences.get("activity_style"))
    must_avoid = _text_list(preferences.get("must_avoid"))

    distance_raw = (preferences.get("distance_preference") or "").strip()
    distance_preference = "nearby" if distance_raw in {"别太远", "附近", "近一点"} else "balanced"

    pace = "relaxed" if "轻松" in activity_style else ("compact" if "紧凑" in activity_style else "balanced")

    # 行程类型
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

    # 关键词
    restaurant_keywords = _dedupe(
        _text_list(intent.get("restaurant_keywords")) or diet_preference or ["简餐", "聚餐", "特色餐厅"]
    )
    if len(restaurant_keywords) < 3:
        restaurant_keywords = _dedupe(restaurant_keywords + ["简餐", "聚餐", "特色餐厅"])

    activity_keywords = _dedupe(
        _text_list(intent.get("activity_keywords"))
        or (["儿童乐园", "科技馆", "公园", "博物馆", "商场"] if has_child
            else ["公园", "美术馆", "博物馆", "商场", "购物中心"])
    )

    # 限制阈值
    max_traffic = _DEFAULTS["max_traffic_minutes"]
    if has_child or distance_preference == "nearby":
        max_traffic = min(max_traffic, 30)
    indoor_preferred = any(t in must_avoid for t in ("室外", "暴晒", "下雨"))

    return {
        # 基础
        "scenario": scenario,
        "party": party,
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
        "origin_coordinates": origin_coordinates,
        # 搜索
        "restaurant_keywords": restaurant_keywords,
        "activity_keywords": activity_keywords,
        "exclude_restaurant": _dedupe([t for t in must_avoid if t in {"火锅", "烧烤", "烤肉", "自助"}]),
        "need_activity": need_activity,
        "need_restaurant": need_restaurant,
        "search_radius": "near" if max_traffic <= 30 else "medium",
        # 限制
        "max_traffic_minutes": max_traffic,
        "max_queue_minutes": _DEFAULTS["max_queue_minutes"],
        "max_total_travel_minutes": _DEFAULTS["max_total_travel_minutes"],
        "indoor_preferred": indoor_preferred,
        # 偏好
        "diet_preference": diet_preference,
        "activity_style": activity_style,
        "must_avoid": must_avoid,
        "distance_preference": distance_preference,
        "pace_preference": pace,
        # 重规划
        "replan_hints": [replan_reason] if replan_reason and replan_reason.strip() else [],
        "replan_reason_type": replan_reason_type or "",
    }


def constraint_build_node(state: AgentState) -> AgentState:
    """Constraint Build Node：将 intent 汇总为 plan_context，供下游节点使用。

    adjust 路径（多轮反馈）：在已有 plan_context 基础上追加增量关键词和用户反馈。
    其他路径：从 intent 重新构建完整 plan_context。
    """
    print("[Constraint Build Node] 汇总规划约束...")
    try:
        feedback_route = state.get("feedback_route") or ""

        if feedback_route == "adjust" and state.get("plan_context"):
            # ── adjust 路径：增量合并 ──────────────────────────────────────
            plan_context = dict(state["plan_context"])

            plan_context["activity_keywords"] = _merge_keywords(
                plan_context.get("activity_keywords") or [],
                state.get("incremental_activity_keywords") or [],
            )
            plan_context["restaurant_keywords"] = _merge_keywords(
                plan_context.get("restaurant_keywords") or [],
                state.get("incremental_restaurant_keywords") or [],
            )

            # 把用户反馈追加进 raw_query，供 candidate_planning 参考
            feedback_summary = (state.get("feedback_summary") or "").strip()
            if feedback_summary:
                prev_query = plan_context.get("raw_query") or ""
                plan_context["raw_query"] = f"{prev_query}\n[用户反馈]{feedback_summary}".strip()

            print(
                f"[Constraint Build Node] adjust 模式 | "
                f"活动关键词={plan_context['activity_keywords']} | "
                f"餐厅关键词={plan_context['restaurant_keywords']}"
            )
            return {
                "plan_context":  plan_context,
                "replan_reason": feedback_summary,
                "replan_count":  0,   # 重置重规划计数
            }

        # ── 正常路径：从 intent 重新构建 ────────────────────────────────────
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