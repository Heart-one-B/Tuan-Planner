# src/graph/nodes/final_plan_node.py
from datetime import datetime
from src.graph.state import AgentState
from src.tools.mock_api import MockToolAPI
from src.utils.state_utils import _append_error
from src.utils.route_utils import _resolve_poi_location


def final_plan_node(state: AgentState) -> AgentState:
    """Final Plan Node: 选择最高评分方案，并为其注入精密计算的时间轴时刻。
    合并了原：
      - final_plan_node (最优推荐精选提取)
      - schedule_timing_node (动态时间排期表解算)
    """
    print("[Final Plan Node] 精选高评分方案，并进行细粒度时刻排期算绘...")
    try:
        scoring_result = state.get("scoring_result") or {}
        rule_validation_result = state.get("rule_validation_result") or {}
        scored_candidates = scoring_result.get("scored_candidates") or []
        valid_plans = rule_validation_result.get("valid_plans") or []

        # 1. 提取最高评分项
        best_candidate_score = None
        best_score = None
        for item in scored_candidates:
            if not isinstance(item, dict):
                continue
            score = item.get("final_score")
            if not isinstance(score, (int, float)):
                continue
            if best_score is None or score > best_score:
                best_candidate_score = item
                best_score = score

        best_candidate = {}
        best_candidate_id = ""
        if isinstance(best_candidate_score, dict):
            best_candidate_id = best_candidate_score.get("candidate_id") or ""
            for item in valid_plans:
                if not isinstance(item, dict):
                    continue
                if item.get("id") == best_candidate_id:
                    best_candidate = dict(item)
                    break

        if not best_candidate and isinstance(best_candidate_score, dict):
            best_candidate = dict(best_candidate_score)

        # 2. 动态时刻解算
        segment_eta = {}
        timeline = []
        normalized_date_label = ""
        if best_candidate:
            activity = best_candidate.get("activity") or {}
            restaurant = best_candidate.get("restaurant") or {}
            restaurants = best_candidate.get("restaurants") or []
            raw_timeline = best_candidate.get("timeline") or []

            constraints = state.get("constraints") or {}
            normalized_time = state.get("normalized_time") or {}
            date_label = constraints.get("date_label") or ""
            daypart = constraints.get("daypart") or ""
            time_phrase = constraints.get("time_phrase") or ""
            raw_query = state.get("intent", {}).get("raw_query") if isinstance(state.get("intent"), dict) else ""
            origin_coordinates = state.get("runtime_origin_coordinates") or ""
            normalized_date_label = normalized_time.get("normalized_date_label") or date_label

            now = datetime.now()
            now_minutes = now.hour * 60 + now.minute
            cutoff_minutes = 18 * 60 if daypart in {"上午", "下午", "全天"} else 21 * 60
            if normalized_date_label == "今天" and now_minutes > cutoff_minutes:
                normalized_date_label = "明天"
                if isinstance(time_phrase, str) and time_phrase:
                    time_phrase = time_phrase.replace("今天", "明天", 1)

            api = MockToolAPI()
            activity_by_id = {activity["id"]: activity} if activity.get("id") else {}
            restaurants_by_id = {item["id"]: item for item in restaurants if isinstance(item, dict) and item.get("id")}
            if restaurant.get("id") and restaurant["id"] not in restaurants_by_id:
                restaurants_by_id[restaurant["id"]] = restaurant

            amap = api._get_amap()

            def _distance_minutes(origin, destination) -> int | None:
                if not amap or not origin or not destination:
                    return None
                try:
                    result = amap.maps_distance(origin, destination, "1") or {}
                    results = result.get("results") or []
                    if results:
                        duration = results[0].get("duration")
                        if isinstance(duration, str) and duration.isdigit():
                            return max(1, int(duration) // 60)
                    return None
                except Exception:
                    return None

            def _minutes_to_hhmm(total_minutes: int) -> str:
                total_minutes = max(0, int(total_minutes))
                return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"

            def _item_floor_minutes(item_type: str) -> int:
                meal_floor = 18 * 60 if "晚餐" in str(raw_query) else (12 * 60 if "午餐" in str(raw_query) else 0)
                mapping = {"activity_morning": 9 * 60 + 30, "activity_afternoon": 14 * 60,
                           "lunch": max(12 * 60, meal_floor), "dinner": max(18 * 60, meal_floor),
                           "restaurant": meal_floor, "meal": meal_floor}
                return mapping.get(item_type, 14 * 60 if daypart == "下午" else (
                    9 * 60 + 30 if daypart in {"上午", "全天"} else 18 * 60))

            def _item_duration_minutes(item_type: str) -> int:
                if item_type in {"activity", "activity_morning", "activity_afternoon", "indoor", "outdoor"}:
                    return 90
                if item_type in {"restaurant", "meal", "lunch", "dinner"}:
                    return 75
                return 60

            def _resolve_timeline_target(item: dict) -> dict:
                ref_type = item.get("ref_type")
                ref_id = item.get("ref_id")
                if ref_type == "activity" and ref_id:
                    return activity_by_id.get(ref_id, activity)
                if ref_type == "restaurant" and ref_id:
                    return restaurants_by_id.get(ref_id, restaurant)
                return activity if str(item.get("type")).startswith("activity") else restaurant

            if not raw_timeline:
                fallback_timeline = []
                if activity:
                    fallback_timeline.append(
                        {"time": time_phrase or "活动", "item": activity.get("name") or "待确认活动",
                         "type": "activity", "ref_type": "activity", "ref_id": activity.get("id", "")})
                if restaurant:
                    fallback_timeline.append(
                        {"time": "用餐", "item": restaurant.get("name") or "待确认餐厅", "type": "restaurant",
                         "ref_type": "restaurant", "ref_id": restaurant.get("id", "")})
                raw_timeline = fallback_timeline

            first_location = _resolve_poi_location(_resolve_timeline_target(raw_timeline[0]),
                                                   api) if raw_timeline else ""
            first_eta_minutes = _distance_minutes(origin_coordinates, first_location) if first_location else None
            if first_eta_minutes is not None:
                segment_eta["origin_to_first_stop_minutes"] = first_eta_minutes

            base_start_minutes = _item_floor_minutes(raw_timeline[0].get("type", "")) if raw_timeline else 14 * 60
            if normalized_date_label == "今天":
                earliest_start_minutes = now_minutes + 15 + (first_eta_minutes or 0)
                base_start_minutes = max(base_start_minutes, earliest_start_minutes)
                if daypart in {"下午", "晚上"} and base_start_minutes > cutoff_minutes:
                    normalized_date_label = "明天"
                    if isinstance(time_phrase, str) and time_phrase:
                        time_phrase = time_phrase.replace("今天", "明天", 1)
                    base_start_minutes = _item_floor_minutes(
                        raw_timeline[0].get("type", "")) if raw_timeline else 14 * 60

            cursor_minutes = base_start_minutes
            previous_location = ""
            previous_type = ""

            for index, raw_item in enumerate(raw_timeline):
                if not isinstance(raw_item, dict):
                    continue

                item_copy = dict(raw_item)
                item_type = item_copy.get("type") or ""
                item_location = _resolve_poi_location(_resolve_timeline_target(item_copy), api)
                if index == 0:
                    item_start_minutes = cursor_minutes
                else:
                    travel_minutes = _distance_minutes(previous_location,
                                                       item_location) if previous_location and item_location else None
                    if travel_minutes is not None:
                        segment_eta[f"leg_{index}_minutes"] = travel_minutes
                    item_start_minutes = cursor_minutes + (travel_minutes or 0) + 15
                    item_start_minutes = max(item_start_minutes, _item_floor_minutes(item_type))
                    if previous_type in {"lunch", "dinner", "restaurant", "meal"} and item_type == "activity_afternoon":
                        item_start_minutes = max(item_start_minutes, 14 * 60)

                item_copy["time"] = _minutes_to_hhmm(item_start_minutes)
                timeline.append(item_copy)

                cursor_minutes = item_start_minutes + _item_duration_minutes(item_type)
                previous_location = item_location
                previous_type = item_type

            best_candidate["timeline"] = timeline

        return {
            "final_plan_result": {
                "selected_candidate": best_candidate or {}, "selected_candidate_id": best_candidate_id,
                "final_score": best_score if best_score is not None else 0,
                "all_scored_candidates": scored_candidates,
            },
            "schedule_timing_result": {
                "segment_eta": segment_eta, "timeline": timeline, "normalized_date_label": normalized_date_label,
            },
        }
    except Exception as exc:
        print(f"[Final Plan Node][WARN] 排期计算异常: {exc}")
        update = _append_error(state, f"Final Plan calculation failed: {exc}")
        update["final_plan_result"] = {"selected_candidate": {}, "selected_candidate_id": "", "final_score": 0,
                                       "all_scored_candidates": []}
        return update