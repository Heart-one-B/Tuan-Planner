import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from src.model.factory import chat_model


class IntentAgent:
    def parse(self, user_input: str) -> dict:
        print("🤖 [Intent Agent] 正在解析用户自然语言意图...")
        system_prompt = """
        你是一个意图解析助手。请提取用户的本地生活需求，并严格返回JSON格式，不要包含任何其他文字。
        必须包含以下字段:
        - scenario: "family" 或 "friends"
        - child_friendly: bool
        - diet_preference: 字符串 (如 "减脂", "无")
        """

        try:
            response = chat_model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ])
            # 正则清理可能带有的 markdown json 标签
            content = response.content
            json_str = re.search(r'\{.*\}', content, re.DOTALL)
            intent = json.loads(json_str.group() if json_str else content)
            print(f"✅ 解析结果: {intent}")
            return intent
        except Exception as e:
            print(f"❌ 解析失败，使用默认家庭意图: {e}")
            return {"scenario": "family", "child_friendly": True, "diet_preference": "减脂"}