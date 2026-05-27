# src/graph/nodes/fact_gathering_node.py
from src.graph.state import AgentState
from src.tools.mock_api import MockToolAPI

from src.utils.state_utils import (
    _fact_query_constraints, _print_time_context, _has_required_time_fields, _append_error,
    _derive_traffic_depart_context
)
from src.utils.weather_utils import (
    _derive_weather_scenario_key, _weather_requires_indoor, _prune_weather_risky_activity_keywords
)
from src.utils.poi_utils import (
    _normalize_text_list, _merge_text_lists, _activity_environment,
    _matches_keywords, _dedupe_activities_by_identity, _shortlist_with_detail_gate,
    _activity_bucket_key_from_name, _contains_excluded_keywords, _dedupe_restaurants_by_identity,
    _restaurant_bucket_key_from_name
)
from src.utils.time_utils import (
    _activity_matches_daypart, _restaurant_matches_daypart, _infer_queue_time_slot, _infer_crowd_time_slot
)


def fact_gathering_node(state: AgentState) -> AgentState:
    """Fact Gathering Node: 聚合采集外部客观事实。
    合并了原：
      - weather_check_node (天气查询)
      - activity_search_node (活动搜索)
      - restaurant_search_node (餐厅搜索)
      - traffic_eta_node (通勤ETA计算)
      - queue_check_node (排队情况评估)
      - crowd_risk_node (人流饱和风险评估)
      - fact_gathering_node (事实数据拼装)
    """
    print("[Fact Gathering Node] 并行采集环境/交通/排队/POI候选事实并组装...")
    api = MockToolAPI()
    constraints = _fact_query_constraints(state)
    _print_time_context("Fact Gathering", constraints)

    weather = {}
    activities = []
    restaurants = []
    traffic = {}
    queue = {}
    crowd = {}
    errors = list(state.get("errors", []))
    daypart = constraints.get("daypart") or ""

    # 1. 天气采集
    try:
        if _has_required_time_fields(constraints, "weather_check"):
            scenario_key = _derive_weather_scenario_key(constraints)
            weather = api.get_weather(scenario_key, origin_area=constraints.get("origin_area") or "",
                                      runtime_origin_area=state.get("runtime_origin_area", "") or "") or {}
            weather["scenario_key_used"] = scenario_key
            weather["date_label_used"] = constraints.get("date_label")
            weather["daypart_used"] = constraints.get("daypart")
        else:
            weather = {"target_id": "weather", "status": "unknown", "weather": "", "risk_level": "unknown",
                       "advice": ""}
    except Exception as exc:
        errors.append(f"Weather check failed: {exc}")

    # 2. 活动检索
    try:
        if _has_required_time_fields(constraints, "activity_search") and constraints.get("need_activity") is not False:
            scenario = constraints.get("scenario") or "family"
            activity_explicit_types = _normalize_text_list(constraints.get("activity_explicit_types"))
            keywords_activity = _merge_text_lists(constraints.get("keywords_activity"), activity_explicit_types)
            activity_search_keywords = _merge_text_lists(constraints.get("activity_search_keywords"),
                                                         activity_explicit_types)
            preferred_activity_tags = _normalize_text_list(constraints.get("preferred_activity_tags"))

            weather_risk = weather.get("risk_level") or weather.get("risk") or ""
            indoor_required = (
                    constraints.get("indoor_preferred") is True or constraints.get("indoor_only") is True
                    or constraints.get("weather_guard") == "indoor_preferred" or _weather_requires_indoor(weather_risk)
            )
            activity_search_keywords, _ = _prune_weather_risky_activity_keywords(activity_search_keywords,
                                                                                 indoor_required=indoor_required)
            keywords_activity, _ = _prune_weather_risky_activity_keywords(keywords_activity,
                                                                          indoor_required=indoor_required)

            raw_activities = api.search_activities(
                scenario, activity_keywords=activity_search_keywords or keywords_activity,
                origin_area=constraints.get("origin_area") or "",
                runtime_origin_area=state.get("runtime_origin_area", "") or "",
                runtime_origin_coordinates=state.get("runtime_origin_coordinates", "") or "", enrich_details=False,
            ) or []

            normalized_activities = [dict(item) for item in raw_activities if isinstance(item, dict)]
            if constraints.get("child_friendly_required") or constraints.get("child_friendly_preferred"):
                child_friendly_matches = [item for item in normalized_activities if item.get("child_friendly") is True]
                if child_friendly_matches:
                    normalized_activities = child_friendly_matches

            if indoor_required:
                normalized_activities = [item for item in normalized_activities if
                                         _activity_environment(item) == "indoor"]

            has_keyword_sourced_results = any(
                item.get("search_keyword_sources") or item.get("keyword_source") for item in normalized_activities)
            if keywords_activity and not (activity_search_keywords and has_keyword_sourced_results):
                keyword_matched = [item for item in normalized_activities if _matches_keywords(item, keywords_activity)]
                normalized_activities = keyword_matched or [item for item in normalized_activities if
                                                            _matches_keywords(item, activity_search_keywords)] or []

            if preferred_activity_tags:
                tag_matched = [item for item in normalized_activities if
                               _matches_keywords(item, preferred_activity_tags)]
                if tag_matched:
                    normalized_activities = tag_matched

            def _activity_search_keyword_bucket(item: dict) -> str:
                sources = item.get("search_keyword_sources")
                if isinstance(sources, list):
                    for s in sources:
                        if s in activity_search_keywords:
                            return s
                return item.get("keyword_source") or _activity_bucket_key_from_name(item.get("name") or "")

            activities = _shortlist_with_detail_gate(
                _dedupe_activities_by_identity(normalized_activities), bucket_key_fn=_activity_bucket_key_from_name,
                api=api, daypart=daypart, daypart_match_fn=_activity_matches_daypart, per_bucket_limit=2,
                max_scan_per_bucket=10,
                bucket_order=activity_search_keywords,
                bucket_key_from_item_fn=_activity_search_keyword_bucket if activity_search_keywords else None,
                allow_unverified_fallback=bool(activity_search_keywords),
                total_limit=10 if activity_search_keywords else None,
            )
            for item in activities:
                item["daypart_used"] = daypart
    except Exception as exc:
        errors.append(f"Activity search failed: {exc}")

    # 3. 餐厅检索
    try:
        if _has_required_time_fields(constraints, "restaurant_search") and constraints.get(
                "need_restaurant") is not False:
            query_keywords = _merge_text_lists(constraints.get("keywords_restaurant"),
                                               constraints.get("restaurant_explicit_types"))
            exclude_keywords = _normalize_text_list(constraints.get("exclude_keywords_restaurant"))
            diet_preference = query_keywords or constraints.get("diet_preference") or ""
            scenario = constraints.get("scenario") or ""

            if query_keywords:
                raw_restaurants = []
                seen_ids = set()
                for keyword in query_keywords:
                    batch = api.search_restaurants(
                        [keyword], origin_area=constraints.get("origin_area") or "",
                        runtime_origin_area=state.get("runtime_origin_area", "") or "",
                        runtime_origin_coordinates=state.get("runtime_origin_coordinates", "") or "",
                        enrich_details=False,
                    ) or []
                    for item in batch:
                        if not isinstance(item, dict):
                            continue
                        item_copy = dict(item)
                        item_copy["keyword_source"] = keyword
                        sources = item_copy.get("search_keyword_sources") or []
                        if keyword not in sources:
                            sources.append(keyword)
                        item_copy["search_keyword_sources"] = sources
                        item_id = item.get("id")
                        if item_id and item_id in seen_ids:
                            continue
                        if item_id:
                            seen_ids.add(item_id)
                        raw_restaurants.append(item_copy)
            else:
                raw_restaurants = api.search_restaurants(
                    diet_preference, origin_area=constraints.get("origin_area") or "",
                    runtime_origin_area=state.get("runtime_origin_area", "") or "",
                    runtime_origin_coordinates=state.get("runtime_origin_coordinates", "") or "", enrich_details=False,
                ) or []

            normalized_restaurants = [dict(item) for item in raw_restaurants if isinstance(item, dict)]
            if exclude_keywords:
                normalized_restaurants = [item for item in normalized_restaurants if
                                          not _contains_excluded_keywords(item, exclude_keywords)]

            def _restaurant_search_keyword_bucket(item: dict) -> str:
                sources = item.get("search_keyword_sources")
                if isinstance(sources, list):
                    for s in sources:
                        if s in query_keywords:
                            return s
                return item.get("keyword_source") or _restaurant_bucket_key_from_name(item.get("name") or "")

            restaurants = _shortlist_with_detail_gate(
                _dedupe_restaurants_by_identity(normalized_restaurants), bucket_key_fn=_restaurant_bucket_key_from_name,
                api=api, daypart=daypart, daypart_match_fn=_restaurant_matches_daypart, per_bucket_limit=2,
                max_scan_per_bucket=10,
                bucket_order=query_keywords,
                bucket_key_from_item_fn=_restaurant_search_keyword_bucket if query_keywords else None,
                allow_unverified_fallback=bool(query_keywords), total_limit=10 if query_keywords else None,
                debug_label="Restaurant Search Node",
            )

            def _scenario_rank(item: dict) -> int:
                tags = item.get("tags") or []
                tags_semantic = item.get("tags_semantic") or []
                text = " ".join(tags + tags_semantic)
                if scenario == "friends" and any(t in text for t in ("聚会", "音乐", "氛围", "晚餐", "酒馆")):
                    return 0
                if scenario == "family" and any(t in text for t in ("健康", "轻食", "有机", "简餐")):
                    return 0
                return 1

            restaurants.sort(key=_scenario_rank)
            for item in restaurants:
                item["daypart_used"] = daypart
    except Exception as exc:
        errors.append(f"Restaurant search failed: {exc}")

    # 4. 评估通勤 ETA 事实
    try:
        date_label = constraints.get("date_label")
        if isinstance(date_label, str) and date_label.strip() in {"今天", "today"}:
            if _has_required_time_fields(constraints, "traffic_eta"):
                origin_area = constraints.get("origin_area") or "area_central"
                origin_coordinates = state.get("runtime_origin_coordinates") or ""
                traffic_origin = origin_coordinates if origin_coordinates.strip() else origin_area
                depart_context = _derive_traffic_depart_context(constraints)

                activity_ids = []
                activities_by_scenario = api.db.get("activities", {}) or {}
                for scenario_key in ("family", "friends"):
                    for item in activities_by_scenario.get(scenario_key, []) or []:
                        if isinstance(item, dict) and item.get("id"):
                            activity_ids.append(item["id"])
                restaurant_ids = [item["id"] for item in (api.db.get("restaurants", []) or []) if
                                  isinstance(item, dict) and item.get("id")]

                eta_by_target = {}
                for target_id in (activity_ids + restaurant_ids):
                    record = api.get_traffic_eta(traffic_origin, target_id, depart_context) or {}
                    eta_by_target[target_id] = {
                        "eta_minutes": record.get("eta_minutes"),
                        "congestion": record.get("congestion"),
                        "fallback_hint": record.get("fallback_hint"),
                        "depart_context_used": depart_context,
                    }
                traffic = {
                    "origin_area_used": origin_area, "origin_coordinates_used": origin_coordinates,
                    "traffic_origin_used": traffic_origin, "depart_context_used": depart_context,
                    "eta_by_target": eta_by_target, "enabled_for_today_only": True,
                }
            else:
                traffic = {"origin_area_used": constraints.get("origin_area") or "", "depart_context_used": "",
                           "eta_by_target": {}}
        else:
            traffic = {
                "origin_area_used": constraints.get("origin_area") or "",
                "origin_coordinates_used": state.get("runtime_origin_coordinates", "") or "",
                "traffic_origin_used": "", "depart_context_used": "", "eta_by_target": {},
                "enabled_for_today_only": True,
            }
    except Exception as exc:
        errors.append(f"Traffic ETA failed: {exc}")

    # 5. 排队事实估算
    try:
        time_window = constraints.get("time_window") or ""
        if _has_required_time_fields(constraints, "queue_check"):
            time_slot = _infer_queue_time_slot(time_window)
            people_count = constraints.get("people_count") or 2
            if isinstance(people_count, bool) or not isinstance(people_count, int) or people_count <= 0:
                people_count = 2

            restaurant_ids = [item["id"] for item in (api.db.get("restaurants", []) or []) if
                              isinstance(item, dict) and item.get("id")]
            wait_by_restaurant = {}
            for rid in restaurant_ids:
                record = api.estimate_restaurant_queue(rid, time_slot, people_count) or {}
                wait_by_restaurant[rid] = {
                    "wait_minutes": record.get("wait_minutes"), "party_acceptable": record.get("party_acceptable"),
                    "fallback_hint": record.get("fallback_hint"),
                }
            queue = {"time_slot_used": time_slot, "wait_by_restaurant": wait_by_restaurant}
        else:
            queue = {"time_slot_used": "", "wait_by_restaurant": {}}
    except Exception as exc:
        errors.append(f"Queue check failed: {exc}")

    # 6. 人流饱和检测
    try:
        time_window = constraints.get("time_window") or ""
        if _has_required_time_fields(constraints, "crowd_risk"):
            time_slot = _infer_crowd_time_slot(time_window)
            activity_ids = []
            activities_by_scenario = api.db.get("activities", {}) or {}
            for scenario_key in ("family", "friends"):
                for item in activities_by_scenario.get(scenario_key, []) or []:
                    if isinstance(item, dict) and item.get("id"):
                        activity_ids.append(item["id"])

            crowd_by_activity = {}
            for aid in activity_ids:
                record = api.evaluate_crowd_risk(aid, time_slot) or {}
                crowd_by_activity[aid] = {
                    "risk_level": record.get("risk_level"), "fallback_hint": record.get("fallback_hint"),
                }
            crowd = {"crowd_by_activity": crowd_by_activity}
        else:
            crowd = {"crowd_by_activity": {}}
    except Exception as exc:
        errors.append(f"Crowd risk evaluation failed: {exc}")

    # 统一拼装最终大事实库
    query_constraints = {}
    constraint_build = state.get("constraint_build")
    if isinstance(constraint_build, dict):
        query_constraints = constraint_build.get("query_constraints") or {}

    fact_gathering_result = {
        "query_constraints": query_constraints, "weather": weather, "activities": activities,
        "restaurants": restaurants, "traffic": traffic, "queue": queue, "crowd": crowd,
    }

    return {
        "fact_gathering_result": fact_gathering_result, "weather": weather, "activities": activities,
        "restaurants": restaurants, "traffic": traffic, "queue": queue, "crowd": crowd, "errors": errors,
    }