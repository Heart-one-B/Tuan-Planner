"""
mock_api.py — 本地 Mock 工具，仅用于尚未接入真实 API 的节点。

当前保留：
  - MockBookingAPI：预约/订座的 mock 实现，供 execution_node 开发阶段使用

已移除（功能已由 CachedAmapClient 接管）：
  - search_activities / search_restaurants
  - get_weather
  - get_traffic_eta / estimate_restaurant_queue / evaluate_crowd_risk
  - _enrich_poi_details / _normalize_amap_* / semantic_search 等
"""
from __future__ import annotations

import time


class MockBookingAPI:
    """预约/订座 Mock，execution_node 开发阶段使用。"""

    def check_availability(self, venue_id: str, time_slot: str) -> tuple[bool, str]:
        """检查场馆/餐厅是否有位。"""
        if venue_id == "R1":
            return False, "当前时段已满座，需要排队 60 分钟"
        return True, "余位充足"

    def reserve_venue(self, venue_id: str, user_info: str) -> dict:
        """预约场馆/餐厅，返回订单信息。"""
        time.sleep(0.5)
        return {
            "status":   "success",
            "order_id": f"MT{int(time.time())}",
            "venue_id": venue_id,
        }