import time
import random
import string
from src.tools.mock_api import MockBookingAPI


def _fake_order_id(prefix: str = "MT") -> str:
    """生成模拟订单号"""
    suffix = ''.join(random.choices(string.digits, k=8))
    return f"{prefix}{suffix}"


class ExecutionAgent:
    def __init__(self):
        self.tools = MockBookingAPI()

    def execute(self, plan: dict):
        print("\n[Execution Agent] 用户已确认，正在执行模拟预订请求...")
        activities = plan.get("activities") or []
        restaurant = plan.get("restaurant") or {}

        orders = []

        # ── 活动预订（容错：id 缺失时使用占位符） ──────────────────────
        if isinstance(activities, list) and activities and isinstance(activities[0], dict):
            act = activities[0]
            act_id = act.get("id") or act.get("name") or "ACT_MOCK"
            act_name = act.get("name") or "活动"
            order_id = _fake_order_id("ACT")
            print(f"[Mock Order] 活动预订成功 | {act_name} | 订单号: {order_id}")
            orders.append({
                "type": "activity",
                "status": "success",
                "order_id": order_id,
                "venue_id": act_id,
                "name": act_name,
                "note": "模拟预订成功，实际接入 API 后将完成真实下单。",
            })
        else:
            order_id = _fake_order_id("ACT")
            print(f"[Mock Order] 活动信息不完整，生成占位订单: {order_id}")
            orders.append({
                "type": "activity",
                "status": "simulated",
                "order_id": order_id,
                "venue_id": "UNKNOWN",
                "name": "待确认活动",
                "note": "活动信息不完整，已生成模拟订单。",
            })

        # ── 餐厅预订（容错：id 缺失时使用占位符） ──────────────────────
        if isinstance(restaurant, dict) and restaurant:
            rest_id = restaurant.get("id") or restaurant.get("name") or "REST_MOCK"
            rest_name = restaurant.get("name") or "餐厅"
            order_id2 = _fake_order_id("RST")
            print(f"[Mock Reservation] 餐厅预订成功 | {rest_name} | 订单号: {order_id2}")
            orders.append({
                "type": "restaurant",
                "status": "success",
                "order_id": order_id2,
                "venue_id": rest_id,
                "name": rest_name,
                "note": "模拟预订成功，实际接入 API 后将完成真实订座。",
            })
        else:
            order_id2 = _fake_order_id("RST")
            print(f"[Mock Reservation] 餐厅信息不完整，生成占位订单: {order_id2}")
            orders.append({
                "type": "restaurant",
                "status": "simulated",
                "order_id": order_id2,
                "venue_id": "UNKNOWN",
                "name": "待确认餐厅",
                "note": "餐厅信息不完整，已生成模拟订单。",
            })

        print("[OK] 所有模拟行程凭证已生成。")
        return {
            "status":           "success",
            "core_plan_changed": False,
            "non_core_failures": [],
            "message": "所有关键预约已模拟完成（当前为 Mock 模式，未进行真实下单）。",
            "orders": orders,
        }