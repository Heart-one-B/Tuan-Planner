from __future__ import annotations

import json
from typing import Any

from src.graph.state import AgentState
from src.tools.cached_amap_client import CachedAmapClient
from src.utils.state_utils import _append_error

_CHILD_KEYWORDS = ("儿童", "亲子", "乐园", "科技馆", "博物馆", "动物园", "水族")
_RADIUS = {"near": "3000", "medium": "5000"}

# 最小化验证版核心改动：池子构建从"首词吃满 [:10]"改为"逐词限额轮询"。
# 每个关键词最多贡献 _PER_KEYWORD_LIMIT 条，保证每个关键词都能进池子，
# 池子总上限仍是 _POOL_LIMIT。这样下午聚会的"桌游/台球"不会被"酒吧"挤掉。
_PER_KEYWORD_LIMIT = 3
_EXPLICIT_KEYWORD_LIMIT = 3
_POOL_LIMIT = 12


def _safe_list(v: Any) -> list:
    return v if isinstance(v, list) else []


def _collect_pois_round_robin(
    api: CachedAmapClient,
    keywords: list[str],
    *,
    location: str,
    city: str,
    radius: str,
    poi_type: str = "",
    exclude: list[str] | None = None,
    post_process=None,
) -> list[dict]:
    """
    逐关键词限额收集 POI，轮询合并，去重。
    - 每个关键词先各取最多 _PER_KEYWORD_LIMIT 条（保证多样性）
    - 若总量不足 _POOL_LIMIT，再从各关键词剩余结果补齐
    - post_process: 可选，对单个 poi 做加工（如 infer_environment），返回加工后的 poi
    """
    exclude = exclude or []
    # 先把每个关键词的搜索结果各自存好（不立即截断）
    per_kw_results: list[list[dict]] = []
    for kw in keywords:
        try:
            pois = api.search_pois(kw, location=location, city=city, radius=radius, poi_type=poi_type)
        except Exception:
            pois = []
        cleaned: list[dict] = []
        for poi in pois:
            if not isinstance(poi, dict):
                continue
            name = poi.get("name") or ""
            if exclude and any(ex in name for ex in exclude):
                continue
            if post_process:
                poi = post_process(poi)
            cleaned.append(poi)
        per_kw_results.append(cleaned)

    seen_ids: set[str] = set()
    pool: list[dict] = []

    def _try_add(poi: dict) -> bool:
        pid = poi.get("id") or ""
        if pid and pid in seen_ids:
            return False
        if pid:
            seen_ids.add(pid)
        pool.append(poi)
        return True

    # 第一轮：每个关键词各取前 _PER_KEYWORD_LIMIT 条
    for results in per_kw_results:
        added = 0
        for poi in results:
            if added >= _PER_KEYWORD_LIMIT:
                break
            if _try_add(poi):
                added += 1
        if len(pool) >= _POOL_LIMIT:
            return pool[:_POOL_LIMIT]

    # 第二轮：池子还没满，从各关键词剩余结果补齐
    idx = _PER_KEYWORD_LIMIT
    while len(pool) < _POOL_LIMIT:
        progressed = False
        for results in per_kw_results:
            if idx < len(results):
                if _try_add(results[idx]):
                    progressed = True
                if len(pool) >= _POOL_LIMIT:
                    break
        idx += 1
        if not progressed:
            break

    return pool[:_POOL_LIMIT]


def _clean_keyword_results(
    pois: list[dict],
    *,
    exclude: list[str] | None = None,
    post_process=None,
    explicit_keyword: str = "",
) -> list[dict]:
    exclude = exclude or []
    cleaned: list[dict] = []
    for poi in pois:
        if not isinstance(poi, dict):
            continue
        name = poi.get("name") or ""
        if exclude and any(ex in name for ex in exclude):
            continue
        item = post_process(poi) if post_process else poi
        if explicit_keyword:
            item = {**item, "explicit_activity_type": explicit_keyword}
        cleaned.append(item)
    return cleaned


def _search_keyword_with_city_fallback(
    api: CachedAmapClient,
    keyword: str,
    *,
    location: str,
    city: str,
    radius: str,
    poi_type: str = "",
) -> list[dict]:
    try:
        pois = api.search_pois(keyword, location=location, city=city, radius=radius, poi_type=poi_type)
    except Exception:
        pois = []
    if pois or not city or not location:
        return pois
    try:
        return api.search_pois(keyword, location="", city=city, radius=radius, poi_type=poi_type)
    except Exception:
        return []


def _collect_explicit_activity_pois(
    api: CachedAmapClient,
    keywords: list[str],
    *,
    location: str,
    city: str,
    radius: str,
    post_process=None,
) -> tuple[list[dict], dict[str, list[str]]]:
    pool: list[dict] = []
    matched: list[str] = []
    missing: list[str] = []

    for keyword in keywords:
        raw_results = _search_keyword_with_city_fallback(
            api,
            keyword,
            location=location,
            city=city,
            radius=radius,
        )
        cleaned = _clean_keyword_results(
            raw_results,
            post_process=post_process,
            explicit_keyword=keyword,
        )
        if cleaned:
            matched.append(keyword)
            pool.extend(cleaned[:_EXPLICIT_KEYWORD_LIMIT])
        else:
            missing.append(keyword)

    return pool, {"keywords": keywords, "matched": matched, "missing": missing}


def _merge_poi_pools(*pools: list[dict], limit: int = _POOL_LIMIT) -> list[dict]:
    seen_ids: set[str] = set()
    merged: list[dict] = []
    for pool in pools:
        for poi in pool:
            if not isinstance(poi, dict):
                continue
            pid = poi.get("id") or ""
            if pid and pid in seen_ids:
                continue
            if pid:
                seen_ids.add(pid)
            merged.append(poi)
            if len(merged) >= limit:
                return merged
    return merged


def _is_explicit_activity(poi: dict, explicit_ids: set[str]) -> bool:
    pid = poi.get("id") or ""
    return bool(poi.get("explicit_activity_type") or (pid and pid in explicit_ids))


def fact_gathering_node(state: AgentState) -> AgentState:
    """
    Fact Gathering Node（最小化验证版）：检索循环改轮询，保证池子多样。
    """
    print("[Fact Gathering Node] 采集事实数据...")

    plan   = state.get("plan_context") or {}
    errors = list(state.get("errors") or [])
    api    = CachedAmapClient()

    # ── 1. Geocode ────────────────────────────────────────────────────────────
    origin_area:        str = plan.get("origin_area") or ""
    origin_coordinates: str = plan.get("origin_coordinates") or ""

    geo = {"city": "", "coordinates": "", "district": ""}
    try:
        geo = api.geocode(origin_area)
        if origin_coordinates:
            geo["coordinates"] = origin_coordinates
    except Exception as exc:
        errors.append(f"Geocode failed: {exc}")

    city:        str = geo.get("city") or ""
    coordinates: str = geo.get("coordinates") or ""
    print(f"[Fact Gathering] origin={origin_area!r} → city={city!r}, coord={coordinates!r}")

    # ── 2. 天气 ───────────────────────────────────────────────────────────────
    weather: dict = {}
    try:
        weather = api.weather(city) if city else {}
    except Exception as exc:
        errors.append(f"Weather failed: {exc}")

    # ── 3. 活动搜索（轮询）────────────────────────────────────────────────────
    activities:  list[dict] = []
    activity_explicit_search: dict[str, list[str]] = {"keywords": [], "matched": [], "missing": []}
    child_friendly: bool    = bool(plan.get("child_friendly"))
    weather_high:   bool    = (weather.get("risk_level") or "low") == "high"
    radius:         str     = _RADIUS.get(plan.get("search_radius") or "medium", "5000")

    if plan.get("need_activity") is not False:
        try:
            explicit_keywords = [
                item.strip()
                for item in _safe_list(plan.get("activity_explicit_types"))
                if isinstance(item, str) and item.strip()
            ]
            activity_keywords = [
                item.strip()
                for item in _safe_list(plan.get("activity_keywords"))
                if isinstance(item, str) and item.strip()
            ]
            general_keywords = [kw for kw in activity_keywords if kw not in explicit_keywords]
            post_process_activity = lambda p: {**p, "environment": CachedAmapClient.infer_environment(p)}

            explicit_activities, activity_explicit_search = _collect_explicit_activity_pois(
                api,
                explicit_keywords,
                location=coordinates,
                city=city,
                radius=radius,
                post_process=post_process_activity,
            )
            general_activities = _collect_pois_round_robin(
                api,
                general_keywords,
                location=coordinates,
                city=city,
                radius=radius,
                post_process=post_process_activity,
            )
            activities = _merge_poi_pools(explicit_activities, general_activities, limit=_POOL_LIMIT)
            explicit_ids = {
                item.get("id") or ""
                for item in explicit_activities
                if isinstance(item, dict) and item.get("id")
            }

            if child_friendly:
                cf = [a for a in activities
                      if _is_explicit_activity(a, explicit_ids)
                      or any(t in (a.get("name") or "") for t in _CHILD_KEYWORDS)]
                activities = cf or activities

            if weather_high:
                indoor = [
                    a for a in activities
                    if _is_explicit_activity(a, explicit_ids) or a.get("environment") == "indoor"
                ]
                if indoor:
                    activities = indoor
        except Exception as exc:
            errors.append(f"Activity search failed: {exc}")

    # ── 4. 餐厅搜索（轮询）────────────────────────────────────────────────────
    restaurants: list[dict] = []

    if plan.get("need_restaurant") is not False:
        try:
            restaurants = _collect_pois_round_robin(
                api,
                _safe_list(plan.get("restaurant_keywords")),
                location=coordinates,
                city=city,
                radius=radius,
                poi_type="050000",
                exclude=_safe_list(plan.get("exclude_restaurant")),
            )
        except Exception as exc:
            errors.append(f"Restaurant search failed: {exc}")

    # ── 5. 补全坐标 ───────────────────────────────────────────────────────────
    if coordinates:
        for poi in activities + restaurants:
            if poi.get("location"):
                continue
            pid = poi.get("id") or ""
            if not pid:
                continue
            try:
                detail = api.poi_detail(pid)
                if isinstance(detail, dict):
                    loc = detail.get("location") or ""
                    if loc:
                        poi["location"] = loc
            except Exception:
                continue

    # ── 6. ETA ────────────────────────────────────────────────────────────────
    eta: dict[str, Any] = {}
    if coordinates:
        for poi in activities + restaurants:
            pid  = poi.get("id") or ""
            dest = poi.get("location") or ""
            if not pid or not dest:
                continue
            try:
                result = api.distance(coordinates, dest)
                if result:
                    eta[pid] = result
            except Exception:
                continue

    # ── 7. Waypoint 搜索 ─────────────────────────────────────────────────────
    # 用户点名的途径需求（DQ/星巴克/奶茶等），单独搜索，结果存入 waypoints 池
    waypoints: list[dict] = []
    intent: dict = state.get("intent") or {}
    waypoint_requests: list[dict] = intent.get("waypoint_requests") or []

    if waypoint_requests and coordinates:
        seen_waypoint_ids: set[str] = set()
        for req in waypoint_requests:
            kw = req.get("keyword") or req.get("raw_text") or ""
            if not kw:
                continue
            try:
                pois = api.search_pois(kw, location=coordinates, city=city, radius=radius)
                found = []
                for poi in pois[:3]:
                    pid = poi.get("id") or ""
                    if pid and pid not in seen_waypoint_ids:
                        seen_waypoint_ids.add(pid)
                        poi["waypoint_keyword"] = kw
                        poi["waypoint_raw"] = req.get("raw_text") or kw
                        poi["waypoint_time_hint"] = req.get("time_hint")
                        found.append(poi)
                if found:
                    waypoints.extend(found)
                    print(f"[Fact Gathering] waypoint '{kw}' 找到 {len(found)} 条")
                else:
                    # 搜不到时记录，告知用户
                    waypoints.append({
                        "id": "",
                        "name": f"未找到：{kw}",
                        "waypoint_keyword": kw,
                        "waypoint_raw": req.get("raw_text") or kw,
                        "waypoint_time_hint": req.get("time_hint"),
                        "not_found": True,
                    })
                    print(f"[Fact Gathering] waypoint '{kw}' 附近无结果")
            except Exception as exc:
                errors.append(f"Waypoint search failed for '{kw}': {exc}")

    # 补全 waypoint 坐标和 ETA
    if coordinates:
        for poi in waypoints:
            if poi.get("not_found"):
                continue
            if not poi.get("location"):
                pid = poi.get("id") or ""
                if pid:
                    try:
                        detail = api.poi_detail(pid)
                        if isinstance(detail, dict) and detail.get("location"):
                            poi["location"] = detail["location"]
                    except Exception:
                        pass
            pid = poi.get("id") or ""
            dest = poi.get("location") or ""
            if pid and dest:
                try:
                    result = api.distance(coordinates, dest)
                    if result:
                        eta[pid] = result
                except Exception:
                    pass

    fact_gathering_result = {
        "weather":     weather,
        "activities":  activities,
        "activity_explicit_search": activity_explicit_search,
        "restaurants": restaurants,
        "waypoints":   waypoints,
        "eta":         eta,
    }

    print("\n" + "=" * 40 + " [FACT GATHERING RESULT] " + "=" * 40)
    print(json.dumps(fact_gathering_result, ensure_ascii=False, indent=2))
    print("=" * 105 + "\n")

    return {
        "fact_gathering_result": fact_gathering_result,
        "weather":               weather,
        "activities":            activities,
        "activity_explicit_search": activity_explicit_search,
        "restaurants":           restaurants,
        "waypoints":             waypoints,
        "eta":                   eta,
        "errors":                errors,
    }
