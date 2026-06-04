# src/utils/route_utils.py

def _distance_result_minutes(result) -> int | None:
    """高德路径测距接口结果提取分钟数"""
    if not isinstance(result, dict):
        return None
    results = result.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None
    duration = results[0].get("duration")
    if isinstance(duration, str) and duration.isdigit():
        return max(1, int(duration) // 60)
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        return max(1, int(duration) // 60)
    return None


def _candidate_poi_maps(candidate: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    activity_by_id = {}
    restaurant_by_id = {}

    def add(target: dict, bucket: dict[str, dict]) -> None:
        if not isinstance(target, dict):
            return
        target_id = target.get("id")
        if isinstance(target_id, str) and target_id:
            bucket[target_id] = target

    add(candidate.get("activity"), activity_by_id)
    add(candidate.get("secondary_activity"), activity_by_id)
    for item in candidate.get("activities") if isinstance(candidate.get("activities"), list) else []:
        add(item, activity_by_id)

    add(candidate.get("restaurant"), restaurant_by_id)
    for item in candidate.get("restaurants") if isinstance(candidate.get("restaurants"), list) else []:
        add(item, restaurant_by_id)

    return activity_by_id, restaurant_by_id


def _resolve_poi_location(item: dict, api) -> str:
    """解算 POI 地理物理坐标或地址描述 (高内聚门禁，如果详情未加载则自动查询高德详情)"""
    coordinates = item.get("coordinates") if isinstance(item.get("coordinates"), str) else ""
    if coordinates.strip():
        return coordinates.strip()
    location = item.get("location") if isinstance(item.get("location"), str) else ""
    if location.strip():
        return location.strip()

    item_id = item.get("id")
    source = item.get("source")
    if not isinstance(item_id, str) or source != "mcp" or item.get("detail_loaded") is True:
        return ""
    try:
        detail_item = api.enrich_poi_details([item])
        if isinstance(detail_item, list) and detail_item and isinstance(detail_item[0], dict):
            detail = detail_item[0]
            detail_coordinates = detail.get("coordinates") if isinstance(detail.get("coordinates"), str) else ""
            if detail_coordinates.strip():
                return detail_coordinates.strip()
            detail_location = detail.get("location") if isinstance(detail.get("location"), str) else ""
            if detail_location.strip():
                return detail_location.strip()
    except Exception:
        return ""
    return ""


def _candidate_route_stops(candidate: dict) -> list[dict]:
    """提取计划候选集链路中所含的所有关键物理停靠点 (按顺序)"""
    activity_by_id, restaurant_by_id = _candidate_poi_maps(candidate)
    stops = []

    def append_stop(poi_type: str, poi_id: str) -> None:
        if poi_type == "activity":
            target = activity_by_id.get(poi_id)
        elif poi_type == "restaurant":
            target = restaurant_by_id.get(poi_id)
        else:
            target = None
        if isinstance(target, dict) and target:
            stops.append({"poi_type": poi_type, "poi_id": poi_id, "target": target})

    steps = candidate.get("steps")
    if isinstance(steps, list) and steps:
        for step in steps:
            if not isinstance(step, dict):
                continue
            poi_type = step.get("poi_type")
            poi_id = step.get("poi_id")
            if isinstance(poi_type, str) and isinstance(poi_id, str) and poi_id:
                append_stop(poi_type, poi_id)
        if stops:
            return stops

    timeline = candidate.get("timeline")
    if isinstance(timeline, list) and timeline:
        for item in timeline:
            if not isinstance(item, dict):
                continue
            ref_type = item.get("ref_type")
            ref_id = item.get("ref_id")
            if isinstance(ref_type, str) and isinstance(ref_id, str) and ref_id:
                append_stop(ref_type, ref_id)
        if stops:
            return stops

    activity = candidate.get("activity")
    if isinstance(activity, dict) and isinstance(activity.get("id"), str):
        stops.append({"poi_type": "activity", "poi_id": activity["id"], "target": activity})
    restaurant = candidate.get("restaurant")
    if isinstance(restaurant, dict) and isinstance(restaurant.get("id"), str):
        stops.append({"poi_type": "restaurant", "poi_id": restaurant["id"], "target": restaurant})
    return stops


def _estimate_candidate_route_minutes(*, candidate: dict, origin_coordinates: str, api) -> dict:
    """测算并统计单条候选计划的全链路总耗时与各物理路段开销"""
    stops = _candidate_route_stops(candidate if isinstance(candidate, dict) else {})
    amap = api._get_amap()
    if not amap or not isinstance(origin_coordinates, str) or not origin_coordinates.strip():
        return {
            "total_route_minutes": None, "segments": [], "missing_segments": [], "route_status": "unavailable",
        }

    segments = []
    missing_segments = []
    total_route_minutes = 0
    previous_location = origin_coordinates.strip()
    previous_label = "origin"

    for stop in stops:
        target = stop.get("target") or {}
        poi_id = stop.get("poi_id") or ""
        destination = _resolve_poi_location(target, api)
        segment = {
            "from": previous_label, "to": poi_id, "from_location": previous_location, "to_location": destination,
        }
        if not previous_location or not destination:
            missing_segments.append({**segment, "status": "missing_location"})
            previous_location = destination or previous_location
            previous_label = poi_id or previous_label
            continue
        try:
            minutes = _distance_result_minutes(amap.maps_distance(previous_location, destination, "1"))
        except Exception:
            minutes = None
        if minutes is None:
            missing_segments.append({**segment, "status": "missing_distance"})
        else:
            total_route_minutes += minutes
            segments.append({**segment, "minutes": minutes, "status": "ok"})
        previous_location = destination
        previous_label = poi_id or previous_label

    if not segments and missing_segments:
        route_status = "unavailable"
        total_value = None
    elif missing_segments:
        route_status = "partial"
        total_value = total_route_minutes
    else:
        route_status = "ok"
        total_value = total_route_minutes

    return {
        "total_route_minutes": total_value, "segments": segments, "missing_segments": missing_segments, "route_status": route_status,
    }