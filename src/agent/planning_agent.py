from src.tools.mock_api import MockToolAPI


class PlanningAgent:
    def __init__(self):
        self.tools = MockToolAPI()

    def plan(self, intent: dict) -> dict:
        print("\n[Planning Agent] 开始生成行程并动态调用工具...")
        plan = {"activities": [], "restaurant": None, "exceptions_handled": []}

        # 1. 查天气，判断风险
        weather = self.tools.get_weather()
        print(f"[Tool Call] 查询天气: {weather['weather']}")
        require_indoor = False
        if weather["risk"] == "High":
            print("[Fallback] 检测到高温阵雨，自动将活动约束为【室内】！")
            plan["exceptions_handled"].append("天气炎热/有雨，已自动将原定的室外活动切换为室内。")
            require_indoor = True

        # 2. 匹配活动
        activities = self.tools.search_activities(intent.get("scenario", "family"))
        for act in activities:
            if require_indoor and act["type"] != "indoor":
                continue
            plan["activities"].append(act)
            print(f"[Tool Call] 锁定活动: {act['name']}")
            break

        # 3. 匹配餐厅 (带排队异常救场)
        restaurants = self.tools.search_restaurants(intent.get("diet_preference", ""))
        primary_res = restaurants[0]

        print(f"[Tool Call] 检查首选餐厅 ({primary_res['name']}) 余位...")
        is_avail, msg = self.tools.check_availability(primary_res["id"], "17:30")

        if not is_avail:
            print(f"[Fallback] 餐厅满座: {msg}。启动备选方案...")
            backup_res = restaurants[1]
            print(f"[Tool Call] 检查备选餐厅 ({backup_res['name']}) 余位...")
            is_backup_avail, _ = self.tools.check_availability(backup_res["id"], "17:30")
            if is_backup_avail:
                plan["restaurant"] = backup_res
                plan["exceptions_handled"].append(
                    f"原定 {primary_res['name']} 已满座，已为您更换为同等健康标准的 {backup_res['name']}。")
                print("[OK] 备选餐厅锁定成功！")
        else:
            plan["restaurant"] = primary_res

        return plan
