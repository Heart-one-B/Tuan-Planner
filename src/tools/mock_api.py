import json
import random
import time
from datetime import datetime, timedelta

from src.utils.path_tool import get_abs_path
from src.tools.amap_mcp_client import AmapMCPClient
from src.utils.config_handler import tools_conf


_POI_DETAIL_CACHE_FILE = get_abs_path("data/poi_detail_cache.json")
_POI_DETAIL_CACHE_TTL_DAYS = 7
_MEMORY_POI_DETAIL_CACHE: dict[str, dict | None] = {}
_FILE_POI_DETAIL_CACHE: dict[str, dict] | None = None

_ACTIVITY_ENV_VALUES = {"indoor", "outdoor", "mixed", "unknown"}
_STRONG_INDOOR_TOKENS = (
    "科技馆",
    "博物馆",
    "美术馆",
    "艺术馆",
    "商场",
    "商城",
    "百货",
    "商业场",
    "购物中心",
    "购物广场",
    "商业综合体",
    "ifs",
    "水族馆",
    "海洋馆",
    "影像馆",
    "展览馆",
    "文化馆",
    "图书馆",
    "纪念馆",
)
_OUTDOOR_TOKENS = (
    "公园",
    "森林",
    "湿地",
    "绿道",
    "步道",
    "广场",
    "露营",
    "营地",
    "江滩",
    "河滨",
    "湖",
    "山",
    "农场",
    "草坪",
    "户外",
    "室外",
)
_INDOOR_TOKENS = (
    "商场",
    "商城",
    "百货",
    "商业场",
    "购物中心",
    "购物广场",
    "商业综合体",
    "mall",
    "博物馆",
    "科技馆",
    "美术馆",
    "艺术馆",
    "水族馆",
    "海洋馆",
    "展览",
    "影院",
    "电影院",
    "剧场",
    "剧院",
    "儿童乐园",
    "亲子乐园",
    "游乐中心",
    "室内",
    "馆",
    "中心",
)
_MIXED_TOKENS = ("动物园", "植物园", "游乐园", "景区", "度假区")


class MockToolAPI:
    def __init__(self):
        db_path = get_abs_path("data/mock_db.json")
        with open(db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)
        self._amap = None
        self._poi_detail_cache = _MEMORY_POI_DETAIL_CACHE

    def _get_amap(self):
        if self._amap is None:
            try:
                self._amap = AmapMCPClient()
            except Exception:
                self._amap = False
        return self._amap if self._amap is not False else None

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat()

    @staticmethod
    def _is_cache_fresh(updated_at: str) -> bool:
        if not isinstance(updated_at, str) or not updated_at.strip():
            return False
        try:
            updated = datetime.fromisoformat(updated_at.strip())
        except ValueError:
            return False
        return datetime.utcnow() - updated <= timedelta(days=_POI_DETAIL_CACHE_TTL_DAYS)

    @staticmethod
    def _synthetic_rating_for_poi(poi_id: str) -> float:
        seed = sum(ord(ch) for ch in poi_id)
        rng = random.Random(seed)
        return round(rng.uniform(3.8, 4.9), 1)

    def _load_file_poi_detail_cache(self) -> dict[str, dict]:
        global _FILE_POI_DETAIL_CACHE
        if isinstance(_FILE_POI_DETAIL_CACHE, dict):
            return _FILE_POI_DETAIL_CACHE
        try:
            with open(_POI_DETAIL_CACHE_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                _FILE_POI_DETAIL_CACHE = payload
            else:
                _FILE_POI_DETAIL_CACHE = {}
        except Exception:
            _FILE_POI_DETAIL_CACHE = {}
        return _FILE_POI_DETAIL_CACHE

    def _save_file_poi_detail_cache(self) -> None:
        cache = self._load_file_poi_detail_cache()
        with open(_POI_DETAIL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def _build_cached_detail_payload(self, poi_id: str, detail: dict | None, *, rating: float | None = None, rating_source: str = "") -> dict:
        payload: dict = {
            "updated_at": self._now_iso(),
            "detail": detail if isinstance(detail, dict) else None,
        }
        if isinstance(rating, (int, float)) and not isinstance(rating, bool):
            payload["rating"] = float(rating)
        if isinstance(rating_source, str) and rating_source:
            payload["rating_source"] = rating_source
        return payload

    def _resolve_cached_detail_payload(self, poi_id: str) -> dict | None:
        memory_payload = self._poi_detail_cache.get(poi_id)
        if isinstance(memory_payload, dict):
            return memory_payload
        file_cache = self._load_file_poi_detail_cache()
        file_payload = file_cache.get(poi_id)
        if isinstance(file_payload, dict):
            self._poi_detail_cache[poi_id] = dict(file_payload)
            return file_payload
        return None

    def _assign_cached_rating(self, item: dict, poi_id: str) -> None:
        cached_payload = self._resolve_cached_detail_payload(poi_id)
        if isinstance(cached_payload, dict):
            rating = cached_payload.get("rating")
            if isinstance(rating, (int, float)) and not isinstance(rating, bool):
                item["rating"] = float(rating)
                item["rating_source"] = cached_payload.get("rating_source") or "cache"
                return

        synthetic_rating = self._synthetic_rating_for_poi(poi_id)
        item["rating"] = synthetic_rating
        item["rating_source"] = "synthetic"
        payload = self._build_cached_detail_payload(
            poi_id,
            cached_payload.get("detail") if isinstance(cached_payload, dict) else None,
            rating=synthetic_rating,
            rating_source="synthetic",
        )
        self._poi_detail_cache[poi_id] = payload
        file_cache = self._load_file_poi_detail_cache()
        file_cache[poi_id] = payload
        self._save_file_poi_detail_cache()

    @staticmethod
    def _append_text_parts(parts: list[str], value) -> None:
        if isinstance(value, str):
            if value.strip():
                parts.append(value.strip())
            return
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())

    @classmethod
    def _infer_activity_environment(cls, item: dict, *, respect_existing: bool = True) -> str:
        if not isinstance(item, dict):
            return "unknown"

        explicit = item.get("activity_environment")
        if respect_existing and isinstance(explicit, str) and explicit in _ACTIVITY_ENV_VALUES:
            return explicit

        legacy_type = item.get("type")
        if isinstance(legacy_type, str) and legacy_type in {"indoor", "outdoor", "mixed"}:
            return legacy_type

        parts: list[str] = []
        for key in ("name", "type", "address", "location", "description", "alias"):
            cls._append_text_parts(parts, item.get(key))
        cls._append_text_parts(parts, item.get("tags"))
        cls._append_text_parts(parts, item.get("tags_semantic"))
        cls._append_text_parts(parts, item.get("keyword_source"))
        cls._append_text_parts(parts, item.get("search_keyword_sources"))
        raw_poi = item.get("raw_poi")
        if isinstance(raw_poi, dict):
            for key in ("name", "type", "address"):
                cls._append_text_parts(parts, raw_poi.get(key))

        text = " ".join(parts).lower()
        if any(token.lower() in text for token in _STRONG_INDOOR_TOKENS):
            return "indoor"
        if any(token.lower() in text for token in _MIXED_TOKENS):
            return "mixed"
        if any(token.lower() in text for token in _OUTDOOR_TOKENS):
            return "outdoor"
        if any(token.lower() in text for token in _INDOOR_TOKENS):
            return "indoor"
        return "unknown"

    @classmethod
    def _attach_activity_environment(cls, item: dict, *, refresh: bool = False) -> None:
        if isinstance(item, dict):
            item["activity_environment"] = cls._infer_activity_environment(item, respect_existing=not refresh)

    @staticmethod
    def _resolve_weather_city(origin_area=None, runtime_origin_area=None):
        candidates = [origin_area, runtime_origin_area]
        area_to_city = {
            "国贸": "北京",
            "望京": "北京",
            "朝阳": "北京",
            "海淀": "北京",
            "北京": "北京",
            "上海": "上海",
            "杭州": "杭州",
            "广州": "广州",
            "深圳": "深圳",
            "成都": "成都",
            "春熙路": "成都",
            "天府广场": "成都",
            "高新区": "成都",
            "锦江": "成都",
            "武侯": "成都"
        }

        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            text = candidate.strip()
            if not text:
                continue
            for key, city in area_to_city.items():
                if key in text:
                    return city

        default_city = tools_conf.get("default_weather_city")
        if isinstance(default_city, str) and default_city.strip():
            return default_city.strip()
        return "北京"

    @staticmethod
    def _normalize_amap_weather(payload):
        if not isinstance(payload, dict):
            return None

        forecasts = payload.get("forecasts")
        if not isinstance(forecasts, list) or not forecasts:
            return None

        today = forecasts[0] if isinstance(forecasts[0], dict) else {}
        dayweather = str(today.get("dayweather") or "").strip()
        nightweather = str(today.get("nightweather") or "").strip()
        daytemp = str(today.get("daytemp") or "").strip()
        nighttemp = str(today.get("nighttemp") or "").strip()
        city = payload.get("city")
        if not isinstance(city, str) or not city.strip():
            city = None

        weather_text = dayweather or nightweather or "未知"
        if daytemp and nighttemp:
            weather_text = f"{weather_text} {nighttemp}~{daytemp}℃"
        elif daytemp:
            weather_text = f"{weather_text} {daytemp}℃"

        high_risk_tokens = ("暴雨", "大暴雨", "雷阵雨", "雷雨", "雨", "雪", "冰雹", "大风", "台风")
        medium_risk_tokens = ("阴", "多云")

        risk_level = "Low"
        if any(token in weather_text for token in high_risk_tokens):
            risk_level = "High"
        elif any(token in weather_text for token in medium_risk_tokens):
            risk_level = "Medium"

        advice = "天气良好，可正常安排活动。"
        fallback_hint = ""
        if risk_level == "High":
            advice = "天气风险较高，建议优先安排室内活动。"
            fallback_hint = "天气不稳定，优先选择室内场馆，并预留路线调整空间。"
        elif risk_level == "Medium":
            advice = "天气一般，建议准备室内备选方案。"

        return {
            "target_id": "weather",
            "status": "ok",
            "weather": weather_text,
            "risk": risk_level,
            "risk_level": risk_level,
            "advice": advice,
            "fallback_hint": fallback_hint,
            "source": "mcp",
            "provider": "amap_mcp",
            "resolved_city": city,
            "raw_weather": payload,
        }

    @staticmethod
    def _normalize_amap_pois(payload, *, kind: str):
        if not isinstance(payload, dict):
            return []

        pois = payload.get("pois")
        if not isinstance(pois, list):
            return []

        normalized = []
        for idx, poi in enumerate(pois, start=1):
            if not isinstance(poi, dict):
                continue
            poi_id = poi.get("id") or f"{kind.upper()}_{idx}"
            name = poi.get("name") or f"{kind}_{idx}"
            address = poi.get("address") or poi.get("pname") or ""
            location = poi.get("location") or ""
            type_name = poi.get("type") or ""

            tags = []
            if isinstance(type_name, str) and type_name.strip():
                tags.extend([part.strip() for part in type_name.split(";") if part.strip()])

            item = {
                "id": poi_id,
                "name": name,
                "type": type_name,
                "location": address,
                "coordinates": location,
                "address": address,
                "source": "mcp",
                "provider": "amap_mcp",
                "raw_poi": poi,
                "tags": tags,
                "tags_semantic": tags,
            }

            if kind == "activity":
                item["available_dayparts"] = ["下午", "晚上"]
                item["peak_hours"] = ["13:00-17:00", "18:00-21:00"]
                item["child_friendly"] = True
                item["description"] = address or name
                MockToolAPI._attach_activity_environment(item)
            else:
                item["available_dayparts"] = ["下午", "晚上"]
                item["peak_hours"] = ["11:00-14:00", "17:00-21:00"]
                item["description"] = address or name

            normalized.append(item)

        return normalized

    def _enrich_poi_details(self, items: list[dict]) -> list[dict]:
        amap = self._get_amap()
        if not isinstance(items, list):
            return items if isinstance(items, list) else []
        enriched = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_copy = dict(item)
            poi_id = item_copy.get("id")
            if not isinstance(poi_id, str) or not poi_id:
                enriched.append(item_copy)
                continue
            if item_copy.get("detail_loaded") is True:
                if not item_copy.get("rating"):
                    self._assign_cached_rating(item_copy, poi_id)
                enriched.append(item_copy)
                continue
            cached_payload = self._resolve_cached_detail_payload(poi_id)
            cached_detail = cached_payload.get("detail") if isinstance(cached_payload, dict) else None
            if isinstance(cached_detail, dict) and self._is_cache_fresh(cached_payload.get("updated_at", "")):
                if item_copy.get("coordinates") and not item_copy.get("source_coordinates"):
                    item_copy["source_coordinates"] = item_copy.get("coordinates")
                for key in ("open_time", "opentime2", "address", "city", "business_area", "type", "alias"):
                    if cached_detail.get(key) and not item_copy.get(key):
                        item_copy[key] = cached_detail.get(key)
                detail_coordinates = cached_detail.get("coordinates")
                if isinstance(detail_coordinates, str) and detail_coordinates.strip():
                    item_copy["coordinates"] = detail_coordinates
                elif not item_copy.get("coordinates") and isinstance(cached_detail.get("location"), str):
                    item_copy["coordinates"] = cached_detail.get("location")
                elif item_copy.get("source_coordinates"):
                    item_copy["coordinates"] = item_copy["source_coordinates"]
                item_copy["detail_loaded"] = True
                if item_copy.get("child_friendly") is True:
                    self._attach_activity_environment(item_copy, refresh=True)
                self._assign_cached_rating(item_copy, poi_id)
                enriched.append(item_copy)
                continue
            detail = None
            if amap is not None:
                try:
                    detail = amap.maps_search_detail(poi_id)
                except Exception:
                    detail = None
            if isinstance(detail, dict):
                if item_copy.get("coordinates") and not item_copy.get("source_coordinates"):
                    item_copy["source_coordinates"] = item_copy.get("coordinates")
                for key in ("open_time", "opentime2", "address", "city", "business_area", "type", "alias"):
                    if detail.get(key) and not item_copy.get(key):
                        item_copy[key] = detail.get(key)
                detail_coordinates = detail.get("coordinates")
                if isinstance(detail_coordinates, str) and detail_coordinates.strip():
                    item_copy["coordinates"] = detail_coordinates
                elif not item_copy.get("coordinates") and isinstance(detail.get("location"), str):
                    item_copy["coordinates"] = detail.get("location")
                elif item_copy.get("source_coordinates"):
                    item_copy["coordinates"] = item_copy["source_coordinates"]
                item_copy["detail_loaded"] = True
            if item_copy.get("child_friendly") is True:
                self._attach_activity_environment(item_copy, refresh=True)

            cached_rating = None
            cached_rating_source = ""
            if isinstance(cached_payload, dict):
                rating_value = cached_payload.get("rating")
                if isinstance(rating_value, (int, float)) and not isinstance(rating_value, bool):
                    cached_rating = float(rating_value)
                    cached_rating_source = cached_payload.get("rating_source") or "cache"
            payload = self._build_cached_detail_payload(
                poi_id,
                detail if isinstance(detail, dict) else cached_detail,
                rating=cached_rating,
                rating_source=cached_rating_source,
            )
            self._poi_detail_cache[poi_id] = payload
            file_cache = self._load_file_poi_detail_cache()
            file_cache[poi_id] = payload
            self._assign_cached_rating(item_copy, poi_id)
            file_cache[poi_id] = self._poi_detail_cache.get(poi_id) or payload
            self._save_file_poi_detail_cache()
            enriched.append(item_copy)
        return enriched

    @staticmethod
    def _build_activity_keywords(scenario: str) -> str:
        if scenario == "family":
            return "亲子 乐园 儿童 活动"
        if scenario == "friends":
            return "聚会 玩乐 展览 桌游"
        return "休闲 娱乐 活动"

    @staticmethod
    def _normalize_activity_keyword_list(activity_keywords, scenario: str) -> list[str]:
        if isinstance(activity_keywords, list):
            candidates = [str(item).strip() for item in activity_keywords if str(item).strip()]
        elif isinstance(activity_keywords, str):
            candidates = [item.strip() for item in activity_keywords.split() if item.strip()]
        else:
            candidates = []
        if not candidates:
            candidates = [item.strip() for item in MockToolAPI._build_activity_keywords(scenario).split() if item.strip()]

        deduped: list[str] = []
        for keyword in candidates:
            if keyword and keyword not in deduped:
                deduped.append(keyword)
        return deduped

    @staticmethod
    def _merge_keyword_activity_results(existing_items: list[dict], new_items: list[dict], keyword: str, per_keyword_limit: int) -> None:
        for item in new_items[:per_keyword_limit]:
            if not isinstance(item, dict):
                continue
            item_copy = dict(item)
            item_copy["keyword_source"] = keyword
            sources = item_copy.get("search_keyword_sources")
            if not isinstance(sources, list):
                sources = []
            if keyword not in sources:
                sources.append(keyword)
            item_copy["search_keyword_sources"] = sources
            item_copy["activity_environment"] = MockToolAPI._infer_activity_environment(
                item_copy,
                respect_existing=False,
            )

            item_id = item_copy.get("id") if isinstance(item_copy.get("id"), str) else ""
            item_name = item_copy.get("name") if isinstance(item_copy.get("name"), str) else ""
            for existing in existing_items:
                existing_id = existing.get("id") if isinstance(existing.get("id"), str) else ""
                existing_name = existing.get("name") if isinstance(existing.get("name"), str) else ""
                if (item_id and existing_id == item_id) or (item_name and existing_name == item_name):
                    existing_sources = existing.get("search_keyword_sources")
                    if not isinstance(existing_sources, list):
                        existing_sources = []
                    if keyword not in existing_sources:
                        existing_sources.append(keyword)
                    existing["search_keyword_sources"] = existing_sources
                    if not existing.get("keyword_source"):
                        existing["keyword_source"] = keyword
                    existing["activity_environment"] = MockToolAPI._infer_activity_environment(
                        existing,
                        respect_existing=False,
                    )
                    break
            else:
                existing_items.append(item_copy)

    @staticmethod
    def _build_restaurant_keywords(diet_preference) -> str:
        if isinstance(diet_preference, list):
            diet_text = " ".join(str(item).strip() for item in diet_preference if str(item).strip())
        else:
            diet_text = str(diet_preference or "").strip()

        if any(token in diet_text for token in ("烤肉", "烧烤")):
            return "烤肉 烧烤 烤串 自助烤肉 韩式烤肉"
        if "火锅" in diet_text:
            return "火锅 麻辣火锅 鸳鸯锅"
        if "轻食" in diet_text:
            return "轻食 沙拉 健康餐"
        if "减脂" in diet_text or "减肥" in diet_text:
            return "轻食 沙拉 健康餐"
        if diet_text:
            return diet_text
        return "餐厅 美食"

    # ------------------------------------------------------------------
    # Legacy-compatible methods
    # ------------------------------------------------------------------

    def search_activities(
        self,
        scenario: str,
        activity_keywords=None,
        origin_area: str = "",
        runtime_origin_area: str = "",
        runtime_origin_coordinates: str = "",
        enrich_details: bool = True,
    ):
        requested_city = self._resolve_weather_city(origin_area, runtime_origin_area)
        amap = self._get_amap()
        if amap is not None:
            try:
                keyword_list = self._normalize_activity_keyword_list(activity_keywords, scenario)
                normalized = []
                search_modes: list[str] = []
                per_keyword_limit = 5
                for keyword in keyword_list:
                    keyword_results = []
                    keyword_search_mode = ""
                    if isinstance(runtime_origin_coordinates, str) and runtime_origin_coordinates.strip():
                        payload = amap.maps_around_search(
                            keyword,
                            location=runtime_origin_coordinates.strip(),
                            radius="10000",
                        )
                        keyword_results = self._normalize_amap_pois(payload, kind="activity")
                        if keyword_results:
                            keyword_search_mode = "around"
                    if not keyword_results:
                        payload = amap.maps_text_search(keyword, city=requested_city)
                        keyword_results = self._normalize_amap_pois(payload, kind="activity")
                        if keyword_results:
                            keyword_search_mode = "text"
                    if keyword_results:
                        if keyword_search_mode and keyword_search_mode not in search_modes:
                            search_modes.append(keyword_search_mode)
                        for item in keyword_results:
                            item["search_keyword"] = keyword
                            item["requested_city"] = requested_city
                            item["search_mode"] = keyword_search_mode
                        self._merge_keyword_activity_results(
                            normalized,
                            keyword_results,
                            keyword,
                            per_keyword_limit,
                        )
                if normalized:
                    for item in normalized:
                        item["requested_city"] = requested_city
                        item["search_mode"] = "+".join(search_modes) if search_modes else item.get("search_mode", "")
                        poi_id = item.get("id")
                        if isinstance(poi_id, str) and poi_id:
                            self._assign_cached_rating(item, poi_id)
                    return self._enrich_poi_details(normalized) if enrich_details else normalized
            except Exception:
                pass
        fallback_items = [
            dict(item)
            for item in self.db["activities"].get(scenario, self.db["activities"]["family"])
            if isinstance(item, dict)
        ]
        for item in fallback_items:
            poi_id = item.get("id")
            self._attach_activity_environment(item)
            if isinstance(poi_id, str) and poi_id:
                self._assign_cached_rating(item, poi_id)
        return fallback_items

    def search_restaurants(
        self,
        diet_preference,
        origin_area: str = "",
        runtime_origin_area: str = "",
        runtime_origin_coordinates: str = "",
        enrich_details: bool = True,
    ):
        requested_city = self._resolve_weather_city(origin_area, runtime_origin_area)
        amap = self._get_amap()
        if amap is not None:
            try:
                keywords = self._build_restaurant_keywords(diet_preference)
                normalized = []
                search_mode = ""
                if isinstance(runtime_origin_coordinates, str) and runtime_origin_coordinates.strip():
                    payload = amap.maps_around_search(
                        keywords,
                        location=runtime_origin_coordinates.strip(),
                        radius="10000",
                    )
                    normalized = self._normalize_amap_pois(payload, kind="restaurant")
                    if normalized:
                        search_mode = "around"
                if not normalized:
                    payload = amap.maps_text_search(keywords, city=requested_city)
                    normalized = self._normalize_amap_pois(payload, kind="restaurant")
                    if normalized:
                        search_mode = "text"
                if normalized:
                    for item in normalized:
                        item["requested_city"] = requested_city
                        item["search_mode"] = search_mode
                        poi_id = item.get("id")
                        if isinstance(poi_id, str) and poi_id:
                            self._assign_cached_rating(item, poi_id)
                    return self._enrich_poi_details(normalized) if enrich_details else normalized
            except Exception:
                pass
        if isinstance(diet_preference, list):
            diet_tokens = [str(item) for item in diet_preference]
            diet_text = " ".join(diet_tokens)
        else:
            diet_text = str(diet_preference or "")
        if "??" in diet_text or "??" in diet_text:
            fallback_items = [dict(r) for r in self.db["restaurants"] if any("?" in str(tag) for tag in r.get("tags", []))]
        else:
            fallback_items = [dict(r) for r in self.db["restaurants"] if not any("?" in str(tag) for tag in r.get("tags", []))]
        for item in fallback_items:
            poi_id = item.get("id")
            if isinstance(poi_id, str) and poi_id:
                self._assign_cached_rating(item, poi_id)
        return fallback_items
    def enrich_poi_details(self, items: list[dict]) -> list[dict]:
        return self._enrich_poi_details(items)

    def check_availability(self, venue_id: str, time_slot: str):
        if venue_id == "R1":
            return False, "当前时段已满座，需要排队60分钟"
        return True, "余位充足"

    def reserve_venue(self, venue_id: str, user_info: str):
        time.sleep(1)
        return {"status": "success", "order_id": f"MT{int(time.time())}"}

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def get_weather(self, scenario_key: str = "default", origin_area: str = "", runtime_origin_area: str = ""):
        requested_city = self._resolve_weather_city(origin_area, runtime_origin_area)
        amap = self._get_amap()
        if amap is not None:
            try:
                result = amap.maps_weather(requested_city)
                normalized = self._normalize_amap_weather(result)
                if isinstance(normalized, dict) and normalized:
                    normalized["requested_city"] = requested_city
                    return normalized
            except Exception:
                pass
        scenarios = self.db.get("weather_scenarios", {})
        default_payload = {
            "weather": "35℃ 阵雨",
            "risk": "High",
            "advice": "建议室内活动",
        }
        payload = scenarios.get(scenario_key) or scenarios.get("default") or default_payload

        result = {
            "target_id": "weather",
            "status": "ok",
            "weather": payload.get("weather", default_payload["weather"]),
            "risk": payload.get("risk", default_payload["risk"]),
            "advice": payload.get("advice", default_payload["advice"]),
            "source": "mock",
            "provider": "local_mock_db",
            "resolved_city": None,
            "requested_city": requested_city,
        }
        result["risk_level"] = result["risk"]
        return result

    # ------------------------------------------------------------------
    # Time-aware tools
    # ------------------------------------------------------------------

    def get_traffic_eta(self, origin: str, destination: str, depart_time=None):
        target_id = f"{origin}->{destination}"
        traffic = self.db.get("traffic", {})
        record = traffic.get(target_id)
        if record is None:
            return {
                "status": "unknown",
                "target_id": target_id,
                "eta_minutes": None,
                "reason": "无路线数据",
            }

        congestion = record.get("congestion", "unknown")
        eta_minutes = record.get("eta_minutes")
        depart_context = depart_time if isinstance(depart_time, str) else ""
        if eta_minutes is not None:
            if "周五:晚上" in depart_context:
                eta_minutes += 15
                congestion = "high"
            elif "周末:下午" in depart_context:
                eta_minutes += 5
                if congestion == "low":
                    congestion = "medium"

        result = {
            "status": "ok",
            "target_id": target_id,
            "eta_minutes": eta_minutes,
            "congestion": congestion,
            "reason": f"路线 {target_id} 预计 {eta_minutes} 分钟，拥堵程度 {congestion}",
            "depart_context_used": depart_context,
        }
        if congestion == "high":
            result["fallback_hint"] = "通勤拥堵明显，建议替换为更近的备选地点或调整出发时间"
        return result

    def estimate_restaurant_queue(self, restaurant_id: str, arrival_time: str, party_size: int = 2):
        key = f"{restaurant_id}@{arrival_time}"
        queues = self.db.get("queues", {})
        record = queues.get(key)
        if record is None:
            return {
                "status": "unknown",
                "target_id": restaurant_id,
                "wait_minutes": None,
                "reason": "无排队数据",
            }

        wait_minutes = record.get("wait_minutes")
        party_acceptable = record.get("party_acceptable", True)
        result = {
            "status": "ok",
            "target_id": restaurant_id,
            "wait_minutes": wait_minutes,
            "party_acceptable": party_acceptable,
            "reason": f"{arrival_time} 时段预计排队 {wait_minutes} 分钟",
        }
        if isinstance(wait_minutes, (int, float)) and wait_minutes > 30:
            result["fallback_hint"] = "排队时间过长，建议改去近距离的备选餐厅或错峰前往"
        return result

    def evaluate_crowd_risk(self, target_id: str, time_slot: str):
        key = f"{target_id}@{time_slot}"
        crowd = self.db.get("crowd", {})
        record = crowd.get(key)
        if record is None:
            return {
                "status": "unknown",
                "target_id": target_id,
                "risk_level": "unknown",
                "reason": "无人流数据",
            }

        crowd_level = record.get("crowd_level", "unknown")
        advice = record.get("advice", "") or ""
        result = {
            "status": "ok",
            "target_id": target_id,
            "risk_level": crowd_level,
            "reason": advice,
        }
        if crowd_level == "high":
            result["fallback_hint"] = advice or "人流密集，建议改去同类型的低拥挤场地或错峰前往"
        return result

    def semantic_search(self, keywords, kind: str = "activity"):
        if not keywords:
            return []

        if kind == "activity":
            activities = self.db.get("activities", {})
            items = list(activities.get("family", [])) + list(activities.get("friends", []))
        elif kind == "restaurant":
            items = list(self.db.get("restaurants", []))
        else:
            return []

        normalized_keywords = [str(k).strip() for k in keywords if str(k).strip()]
        if not normalized_keywords:
            return []
        lowered_keywords = [k.lower() for k in normalized_keywords]

        scored = []
        for item in items:
            tags = item.get("tags_semantic", []) or []
            description = item.get("description", "") or ""
            text = (" ".join(tags) + " " + description).lower()

            score = 0
            matched = []
            for original_kw, kw_lower in zip(normalized_keywords, lowered_keywords):
                if not kw_lower:
                    continue
                count = text.count(kw_lower)
                if count > 0:
                    score += count
                    if original_kw not in matched:
                        matched.append(original_kw)

            if score > 0:
                scored.append(
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "score": score,
                        "matched_keywords": matched,
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:5]
