import json
import time

from src.utils.path_tool import get_abs_path


class MockToolAPI:
    def __init__(self):
        db_path = get_abs_path("data/mock_db.json")
        with open(db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)

    # ------------------------------------------------------------------
    # Legacy-compatible methods
    # ------------------------------------------------------------------

    def search_activities(self, scenario: str):
        return self.db["activities"].get(scenario, self.db["activities"]["family"])

    def search_restaurants(self, diet_preference):
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

    def get_weather(self, scenario_key: str = "default"):
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
