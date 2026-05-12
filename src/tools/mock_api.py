import json
import time
from src.utils.path_tool import get_abs_path

class MockToolAPI:
    def __init__(self):
        # 使用你写的 path_tool 获取绝对路径加载数据
        db_path = get_abs_path("data/mock_db.json")
        with open(db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)

    def get_weather(self):
        """模拟获取天气：故意返回高温阵雨，用于触发后续的“救场”逻辑"""
        return {"weather": "35℃ 阵雨", "risk": "High", "advice": "建议室内活动"}

    def search_activities(self, scenario: str):
        """根据场景搜索活动"""
        return self.db["activities"].get(scenario, self.db["activities"]["family"])

    def search_restaurants(self, diet_preference: str):
        """根据饮食偏好搜索餐厅"""
        if "减脂" in diet_preference or "减肥" in diet_preference:
            return [r for r in self.db["restaurants"] if "减脂" in r["tags"]]
        return [r for r in self.db["restaurants"] if "减脂" not in r["tags"]]

    def check_availability(self, venue_id: str, time_slot: str):
        """模拟余位检查：故意让 R1 满座"""
        if venue_id == "R1":
            return False, "当前时段已满座，需排队60分钟"
        return True, "余位充足"

    def reserve_venue(self, venue_id: str, user_info: str):
        """模拟预订"""
        time.sleep(1) # 模拟网络延迟
        return {"status": "success", "order_id": f"MT{int(time.time())}"}