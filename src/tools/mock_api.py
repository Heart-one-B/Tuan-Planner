import json
import time
from src.utils.path_tool import get_abs_path


class MockToolAPI:
    def __init__(self):
        # 使用 path_tool 获取绝对路径加载数据
        db_path = get_abs_path("data/mock_db.json")
        with open(db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)

    # ------------------------------------------------------------------
    # 旧方法（保持兼容，行为不变）
    # ------------------------------------------------------------------

    def search_activities(self, scenario: str):
        """根据场景搜索活动（兼容旧行为）。"""
        return self.db["activities"].get(scenario, self.db["activities"]["family"])

    def search_restaurants(self, diet_preference):
        """根据饮食偏好搜索餐厅（兼容旧行为）。"""
        if isinstance(diet_preference, list):
            diet_tokens = [str(item) for item in diet_preference]
            diet_text = " ".join(diet_tokens)
        else:
            diet_text = str(diet_preference or "")
        if "减脂" in diet_text or "减肥" in diet_text:
            return [r for r in self.db["restaurants"] if "减脂" in r["tags"]]
        return [r for r in self.db["restaurants"] if "减脂" not in r["tags"]]

    def check_availability(self, venue_id: str, time_slot: str):
        """模拟余位检查：故意让 R1 满座（兼容旧行为）。"""
        if venue_id == "R1":
            return False, "当前时段已满座，需排队60分钟"
        return True, "余位充足"

    def reserve_venue(self, venue_id: str, user_info: str):
        """模拟预订（兼容旧行为，仍包含 sleep 与 time 调用）。"""
        time.sleep(1)  # 模拟网络延迟
        return {"status": "success", "order_id": f"MT{int(time.time())}"}

    # ------------------------------------------------------------------
    # 改造方法
    # ------------------------------------------------------------------

    def get_weather(self, scenario_key: str = "default"):
        """模拟获取天气：默认返回 35℃ 阵雨 High，可通过 scenario_key 切换场景。

        - 未提供 scenario_key 或命中 default 时，返回与历史一致的高温阵雨数据，
          以兼容现有 PlanningAgent 行为。
        - 未知 scenario_key 自动回退到 default。
        - 在原有 weather/risk/advice 基础上额外暴露 target_id 与 risk_level，
          以统一 6 个并行节点的结构化返回风格。
        """
        scenarios = self.db.get("weather_scenarios", {})
        default_payload = {
            "weather": "35℃ 阵雨",
            "risk": "High",
            "advice": "建议室内活动",
        }
        payload = scenarios.get(scenario_key) or scenarios.get("default") or default_payload

        result = {
            "target_id": "weather",
            "weather": payload.get("weather", default_payload["weather"]),
            "risk": payload.get("risk", default_payload["risk"]),
            "advice": payload.get("advice", default_payload["advice"]),
        }
        # risk_level 与 risk 同义，统一命名风格供新节点消费
        result["risk_level"] = result["risk"]
        return result

    # ------------------------------------------------------------------
    # 新增工具：6 个并行节点 + Retrieval Node
    # ------------------------------------------------------------------

    def get_traffic_eta(self, origin: str, destination: str, depart_time=None):
        """查询 origin->destination 的预计通勤数据。

        depart_time 仅占位，未参与查询。
        """
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
        result = {
            "status": "ok",
            "target_id": target_id,
            "eta_minutes": eta_minutes,
            "congestion": congestion,
            "reason": f"路线 {target_id} 预计 {eta_minutes} 分钟，拥堵程度 {congestion}",
        }
        if congestion == "high":
            result["fallback_hint"] = "通勤拥堵明显，建议替换为更近的备选地点或调整出发时间"
        return result

    def estimate_restaurant_queue(self, restaurant_id: str, arrival_time: str, party_size: int = 2):
        """估算指定餐厅在 arrival_time 时段的排队情况。

        arrival_time 期望为 'lunch'/'dinner' 等 slot key。
        party_size 仅占位，不参与查询。
        """
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
        """评估指定地点在指定时段的人流拥挤程度。"""
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
        """基于 tags_semantic + description 的语义子串匹配。

        - keywords: list[str]，空 list 直接返回 []。
        - kind: 'activity' 合并 family+friends；'restaurant' 检索餐厅；其他值返回 []。
        - 评分：每个关键词在合并文本中每出现一次计 1 分（大小写不敏感子串匹配）。
        - matched_keywords 去重列出至少命中一次的关键词。
        - 按 score 降序返回 score>0 的条目，最多 5 条。
        """
        if not keywords:
            return []

        if kind == "activity":
            activities = self.db.get("activities", {})
            items = list(activities.get("family", [])) + list(activities.get("friends", []))
        elif kind == "restaurant":
            items = list(self.db.get("restaurants", []))
        else:
            return []

        # 规范化关键词
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
                scored.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "score": score,
                    "matched_keywords": matched,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:5]
