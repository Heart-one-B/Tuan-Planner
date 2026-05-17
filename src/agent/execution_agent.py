from src.tools.mock_api import MockToolAPI


class ExecutionAgent:
    def __init__(self):
        self.tools = MockToolAPI()

    def execute(self, plan: dict):
        print("\n[Execution Agent] 用户已确认，正在静默执行并发现预订请求...")
        activities = plan.get("activities") or []
        restaurant = plan.get("restaurant") or {}
        if not isinstance(activities, list) or not activities or not isinstance(activities[0], dict):
            return {
                "status": "error",
                "message": "主活动信息缺失，无法执行预约。",
                "core_plan_changed": True,
                "non_core_failures": [],
                "orders": [],
            }
        if not isinstance(restaurant, dict) or not restaurant.get("id"):
            return {
                "status": "error",
                "message": "主餐厅信息缺失，无法执行订座。",
                "core_plan_changed": True,
                "non_core_failures": [],
                "orders": [],
            }

        act_id = activities[0]["id"]
        res = self.tools.reserve_venue(act_id, "User_XiaoMing")
        print(f"[Mock Order] 娱乐门票预订成功！订单号: {res['order_id']}")

        rest_id = restaurant["id"]
        res2 = self.tools.reserve_venue(rest_id, "User_XiaoMing")
        print(f"[Mock Reservation] 餐厅预订成功！订单号: {res2['order_id']}")

        print("[OK] 所有行程凭证已生成。")
        return {
            "status": "success",
            "core_plan_changed": False,
            "non_core_failures": [],
            "message": "所有关键预约已处理完成。",
            "orders": [
                {"type": "activity", **res},
                {"type": "restaurant", **res2},
            ],
        }
