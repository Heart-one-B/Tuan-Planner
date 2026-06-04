from __future__ import annotations

from typing import Any

from src.graph.state import AgentState
from src.tools.cached_amap_client import CachedAmapClient
from src.utils.plan_poi_details_store import load_plan_poi_details, save_plan_poi_details
from src.utils.poi_enrichment_rules import build_enriched_poi_detail
from src.utils.state_utils import _append_error


def _fact_poi_maps(state: AgentState) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    facts = state.get("fact_gathering_result") or {}
    activities = facts.get("activities") or state.get("activities") or []
    restaurants = facts.get("restaurants") or state.get("restaurants") or []
    activity_by_id = {
        item["id"]: item
        for item in activities
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id")
    }
    restaurant_by_id = {
        item["id"]: item
        for item in restaurants
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id")
    }
    return activity_by_id, restaurant_by_id


def _collect_candidate_pois(candidate_plans: dict[str, Any]) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_ref(poi_id: Any, poi_role: Any) -> None:
        pid = str(poi_id or "").strip()
        role = str(poi_role or "").strip()
        if not pid:
            return
        if role not in {"activity", "restaurant"}:
            role = "activity"
        key = (pid, role)
        if key not in seen:
            refs.append(key)
            seen.add(key)

    candidates = candidate_plans.get("candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for step in candidate.get("steps") or []:
            if isinstance(step, dict):
                add_ref(step.get("poi_id"), step.get("poi_type"))
        for activity in candidate.get("activities") or []:
            if isinstance(activity, dict):
                add_ref(activity.get("id"), "activity")
        for restaurant in candidate.get("restaurants") or []:
            if isinstance(restaurant, dict):
                add_ref(restaurant.get("id"), "restaurant")
        activity = candidate.get("activity")
        if isinstance(activity, dict):
            add_ref(activity.get("id"), "activity")
        restaurant = candidate.get("restaurant")
        if isinstance(restaurant, dict):
            add_ref(restaurant.get("id"), "restaurant")

    return refs


def _first_detail_payload(raw_detail: Any) -> dict[str, Any]:
    if not isinstance(raw_detail, dict):
        return {}
    data: Any = raw_detail.get("return", raw_detail)
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}
    if isinstance(data, dict):
        for key in ("poi", "detail"):
            nested = data.get(key)
            if isinstance(nested, dict):
                data = nested
                break
        for key in ("pois", "results"):
            nested_list = data.get(key)
            if isinstance(nested_list, list) and nested_list and isinstance(nested_list[0], dict):
                data = nested_list[0]
                break
    return data if isinstance(data, dict) else {}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "、".join(str(item) for item in value if item)
    return ""


def _detail_business_area(detail: dict[str, Any]) -> str:
    return _text(detail.get("business_area") or detail.get("businessarea"))


_MISSING_TEXT_VALUES = {"暂无", "待确认", "无", "未知", "N/A", "NA", "n/a", "-", "--"}


def _has_value(value: Any) -> bool:
    if value in (None, [], {}):
        return False
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and text not in _MISSING_TEXT_VALUES
    return True


def _detail_rating(detail: dict[str, Any]) -> Any:
    if _has_value(detail.get("rating")):
        return detail.get("rating")
    biz_ext = detail.get("biz_ext")
    if isinstance(biz_ext, dict) and _has_value(biz_ext.get("rating")):
        return biz_ext.get("rating")
    return ""


def _rank_label(category: str, business_area: str) -> str:
    if business_area and category and category != "通用":
        return f"{business_area}{category}推荐"
    if business_area:
        return f"{business_area}人气推荐"
    if category and category != "通用":
        return f"{category}推荐"
    return "本地生活推荐"


def _merge_detail_into_base(base_poi: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_poi)
    field_pairs = {
        "name": ("name",),
        "address": ("address",),
        "location": ("location",),
        "type": ("type", "typecode"),
        "tel": ("tel",),
    }
    for target, candidates in field_pairs.items():
        if merged.get(target):
            continue
        for key in candidates:
            if detail.get(key):
                merged[target] = detail[key]
                break

    business_area = _detail_business_area(detail)
    if business_area:
        merged["business_area"] = business_area
    rating = _detail_rating(detail)
    if rating and not _has_value(merged.get("rating")):
        merged["rating"] = rating
    return merged


def _load_mcp_detail(api: CachedAmapClient, poi_id: str) -> dict[str, Any]:
    try:
        return _first_detail_payload(api.poi_detail(poi_id))
    except Exception as exc:
        print(f"[Plan POI Detail Sync Node][WARN] 详情搜索失败 poi_id={poi_id}: {exc}")
        return {}


def plan_poi_detail_sync_node(state: AgentState) -> AgentState:
    print("[Plan POI Detail Sync Node] 同步候选计划 POI 详情...")
    try:
        candidate_plans = state.get("candidate_plans") or {}
        if not isinstance(candidate_plans, dict):
            return {"plan_poi_details": {}}

        activity_by_id, restaurant_by_id = _fact_poi_maps(state)
        stored_details = load_plan_poi_details()
        changed = False
        details_for_plan: dict[str, dict[str, Any]] = {}
        api: CachedAmapClient | None = None

        for poi_id, poi_role in _collect_candidate_pois(candidate_plans):
            detail = stored_details.get(poi_id)
            if not isinstance(detail, dict):
                if poi_role == "restaurant":
                    base_poi = restaurant_by_id.get(poi_id, {})
                else:
                    base_poi = activity_by_id.get(poi_id, {})
                if not _text(base_poi.get("business_area")) or not _has_value(base_poi.get("rating")):
                    api = api or CachedAmapClient()
                    mcp_detail = _load_mcp_detail(api, poi_id)
                    base_poi = _merge_detail_into_base(base_poi, mcp_detail)
                detail = build_enriched_poi_detail(
                    poi_id=poi_id,
                    poi_role=poi_role,
                    base_poi=base_poi,
                )
                stored_details[poi_id] = detail
                changed = True
            elif not _text(detail.get("business_area")) or not _has_value(detail.get("rating")):
                api = api or CachedAmapClient()
                mcp_detail = _load_mcp_detail(api, poi_id)
                business_area = _detail_business_area(mcp_detail)
                rating = _detail_rating(mcp_detail)
                if business_area or rating:
                    detail = dict(detail)
                    if business_area and not _text(detail.get("business_area")):
                        detail["business_area"] = business_area
                        detail["rank_label"] = _rank_label(detail.get("category") or "通用", business_area)
                    if rating and not _has_value(detail.get("rating")):
                        detail["rating"] = rating
                    stored_details[poi_id] = detail
                    changed = True
            details_for_plan[poi_id] = detail

        if changed:
            save_plan_poi_details(stored_details)
            print(f"[Plan POI Detail Sync Node] 已写入 {len(details_for_plan)} 条本轮 POI 详情。")
        else:
            print(f"[Plan POI Detail Sync Node] 已复用 {len(details_for_plan)} 条本轮 POI 详情。")

        return {"plan_poi_details": details_for_plan}
    except Exception as exc:
        print(f"[Plan POI Detail Sync Node][ERROR] {exc}")
        update = _append_error(state, f"Plan POI detail sync failed: {exc}")
        update["plan_poi_details"] = {}
        return update
