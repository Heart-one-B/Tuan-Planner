from src.tools.mock_api import MockToolAPI


class ExecutionAgent:
    def __init__(self):
        self.tools = MockToolAPI()

    def execute(self, plan: dict):
        print("\n[Execution Agent] 用户已确认，正在静默执行并发现预订请求...")

        act_id = plan["activities"][0]["id"]
        res = self.tools.reserve_venue(act_id, "User_XiaoMing")
        print(f"[Mock Order] 娱乐门票预订成功！订单号: {res['order_id']}")

        rest_id = plan["restaurant"]["id"]
        res2 = self.tools.reserve_venue(rest_id, "User_XiaoMing")
        print(f"[Mock Reservation] 餐厅预订成功！订单号: {res2['order_id']}")

        print("[OK] 所有行程凭证已生成。")
        return {
            "status": "success",
            "orders": [
                {"type": "activity", **res},
                {"type": "restaurant", **res2},
            ],
        }
