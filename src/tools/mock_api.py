import json
import time

from src.utils.path_tool import get_abs_path
from src.tools.amap_mcp_client import AmapMCPClient
from src.utils.config_handler import tools_conf


class MockToolAPI:
    def __init__(self):
        db_path = get_abs_path("data/mock_db.json")
        with open(db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)
        self._amap = None

    def _get_amap(self):
        if self._amap is None:
            try:
                self._amap = AmapMCPClient()
            except Exception:
                self._amap = False
        return self._amap if self._amap is not False else None

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
                "type": "indoor" if kind == "activity" else "restaurant",
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
            else:
                item["available_dayparts"] = ["下午", "晚上"]
                item["peak_hours"] = ["11:00-14:00", "17:00-21:00"]
                item["description"] = address or name

            normalized.append(item)

        return normalized

    @staticmethod
    def _build_activity_keywords(scenario: str) -> str:
        if scenario == "family":
            return "亲子 乐园 儿童 活动"
        if scenario == "friends":
            return "聚会 玩乐 展览 桌游"
        return "休闲 娱乐 活动"

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
        origin_area: str = "",
        runtime_origin_area: str = "",
        runtime_origin_coordinates: str = "",
    ):
        requested_city = self._resolve_weather_city(origin_area, runtime_origin_area)
        amap = self._get_amap()
        if amap is not None:
            try:
                keywords = self._build_activity_keywords(scenario)
                normalized = []
                search_mode = ""
                if isinstance(runtime_origin_coordinates, str) and runtime_origin_coordinates.strip():
                    payload = amap.maps_around_search(
                        keywords,
                        location=runtime_origin_coordinates.strip(),
                        radius="10000",
                    )
                    normalized = self._normalize_amap_pois(payload, kind="activity")
                    if normalized:
                        search_mode = "around"
                if not normalized:
                    payload = amap.maps_text_search(keywords, city=requested_city)
                    normalized = self._normalize_amap_pois(payload, kind="activity")
                    if normalized:
                        search_mode = "text"
                if normalized:
                    for item in normalized:
                        item["requested_city"] = requested_city
                        item["search_mode"] = search_mode
                    return normalized
            except Exception:
                pass
        return self.db["activities"].get(scenario, self.db["activities"]["family"])

    def search_restaurants(
        self,
        diet_preference,
        origin_area: str = "",
        runtime_origin_area: str = "",
        runtime_origin_coordinates: str = "",
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
                    return normalized
            except Exception:
                pass
        if isinstance(diet_preference, list):
            diet_tokens = [str(item) for item in diet_preference]
            diet_text = " ".join(diet_tokens)
        else:
            diet_text = str(diet_preference or "")
        if "减脂" in diet_text or "减肥" in diet_text:
            return [r for r in self.db["restaurants"] if "减脂" in r["tags"]]
        return [r for r in self.db["restaurants"] if "减脂" not in r["tags"]]

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
