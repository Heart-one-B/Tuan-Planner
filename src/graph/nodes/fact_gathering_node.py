from __future__ import annotations

import json
from typing import Any

from src.graph.state import AgentState
from src.tools.cached_amap_client import CachedAmapClient
from src.utils.state_utils import _append_error

_CHILD_KEYWORDS = ("儿童", "亲子", "乐园", "科技馆", "博物馆", "动物园", "水族")
_RADIUS = {"near": "3000", "medium": "5000"}


def _safe_list(v: Any) -> list:
    return v if isinstance(v, list) else []


def fact_gathering_node(state: AgentState) -> AgentState:
    """
    Fact Gathering Node：将 plan_context 中的文字信息转化为真实外部数据。

    核心流程：
      1. geocode(origin_area)          → city + coordinates
      2. weather(city)                 → 天气 + 风险等级
      3. search_pois(activity_keywords) → 活动候选列表
      4. search_pois(restaurant_keywords) → 餐厅候选列表
      5. distance(origin → each POI)   → ETA（有坐标时）

    所有调用经 CachedAmapClient，命中缓存直接返回，不消耗高德额度。
    """
    print("[Fact Gathering Node] 采集事实数据...")

    plan   = state.get("plan_context") or {}
    errors = list(state.get("errors") or [])
    api    = CachedAmapClient()

    # ── 1. Geocode：出发地文字 → 城市 + 坐标 ─────────────────────────────────
    origin_area:        str = plan.get("origin_area") or ""
    origin_coordinates: str = plan.get("origin_coordinates") or ""

    geo = {"city": "", "coordinates": "", "district": ""}
    try:
        geo = api.geocode(origin_area)
        # plan_context 已有精确坐标时优先使用（用户授权定位时写入）
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

    # ── 3. 活动搜索 ───────────────────────────────────────────────────────────
    activities:  list[dict] = []
    child_friendly: bool    = bool(plan.get("child_friendly"))
    weather_high:   bool    = (weather.get("risk_level") or "low") == "high"
    radius:         str     = _RADIUS.get(plan.get("search_radius") or "medium", "5000")

    if plan.get("need_activity") is not False:
        try:
            seen_ids: set[str] = set()
            for kw in _safe_list(plan.get("activity_keywords")):
                pois = api.search_pois(
                    kw,
                    location=coordinates,
                    city=city,
                    radius=radius,
                )
                for poi in pois:
                    pid = poi.get("id") or ""
                    if pid and pid in seen_ids:
                        continue
                    if pid:
                        seen_ids.add(pid)
                    poi["environment"] = CachedAmapClient.infer_environment(poi)
                    activities.append(poi)

            # 亲子过滤：优先保留名称含亲子关键词的场所
            if child_friendly:
                cf = [a for a in activities
                      if any(t in (a.get("name") or "") for t in _CHILD_KEYWORDS)]
                activities = cf or activities

            # 天气过滤：高风险天气优先室内
            if weather_high:
                indoor = [a for a in activities if a.get("environment") == "indoor"]
                if indoor:
                    activities = indoor

            activities = activities[:10]
        except Exception as exc:
            errors.append(f"Activity search failed: {exc}")

    # ── 4. 餐厅搜索 ───────────────────────────────────────────────────────────
    restaurants: list[dict] = []

    if plan.get("need_restaurant") is not False:
        try:
            exclude: list[str] = _safe_list(plan.get("exclude_restaurant"))
            seen_ids = set()
            for kw in _safe_list(plan.get("restaurant_keywords")):
                pois = api.search_pois(
                    kw,
                    location=coordinates,
                    city=city,
                    radius=radius,
                    poi_type="050000",
                )
                for poi in pois:
                    pid = poi.get("id") or ""
                    if pid and pid in seen_ids:
                        continue
                    if pid:
                        seen_ids.add(pid)
                    if exclude and any(ex in (poi.get("name") or "") for ex in exclude):
                        continue
                    restaurants.append(poi)

            restaurants = restaurants[:10]
        except Exception as exc:
            errors.append(f"Restaurant search failed: {exc}")

    # ── 5. ETA（仅有坐标时计算）──────────────────────────────────────────────
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

    # ── 输出 ─────────────────────────────────────────────────────────────────
    fact_gathering_result = {
        "weather":     weather,
        "activities":  activities,
        "restaurants": restaurants,
        "eta":         eta,
    }

    print("\n" + "=" * 40 + " [FACT GATHERING RESULT] " + "=" * 40)
    print(json.dumps(fact_gathering_result, ensure_ascii=False, indent=2))
    print("=" * 105 + "\n")

    return {
        "fact_gathering_result": fact_gathering_result,
        "weather":               weather,
        "activities":            activities,
        "restaurants":           restaurants,
        "eta":                   eta,
        "errors":                errors,
    }