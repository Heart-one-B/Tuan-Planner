import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from src.model.factory import chat_model


# 启发式关键词：用于 LLM 解析失败时兜底判断 is_leisure_planning
_LEISURE_KEYWORDS = (
    "安排", "规划", "玩", "吃", "聚会", "带孩子", "逛", "约", "计划", "出门",
)


def _heuristic_is_leisure(user_input: str) -> bool:
    """关键词启发式：用户输入中出现任一休闲规划关键词即视为规划任务。"""
    text = user_input or ""
    return any(kw in text for kw in _LEISURE_KEYWORDS)


class IntentAgent:
    def parse(self, user_input: str) -> dict:
        print("[Intent Agent] 正在解析用户自然语言意图...")
        system_prompt = """
你是一个本地生活意图解析助手。请严格分析用户输入，并仅返回 JSON（不要包含任何其他文字、不要包裹 markdown 代码块）。

必须包含以下 6 个字段：
- scenario: 字符串，"family"（家庭场景）/ "friends"（朋友场景）/ "none"（与本地生活无关）。
- child_friendly: bool，是否需要儿童友好的安排。
- diet_preference: 字符串，例如 "减脂" / "无辣" / "素食" / "无"。
- is_leisure_planning: bool，用户是否在做"半日 / 一日休闲活动规划"，如吃喝玩乐、家庭出游、朋友聚餐、约会安排等。
    判定指南：
    * "帮我安排周日全家出游" / "想找个适合带孩子的餐厅" / "下午想和朋友逛街吃饭" → true。
    * "今天天气怎样" / "帮我写一段 Python 代码" / "翻译这段英文" / "科普一下什么是 RAG" / "Python 字典怎么排序" → false。
- need_retrieval: bool，是否需要补充语义检索信息。当用户提到具体偏好（特定菜系 / 口味 / 活动类型 / 小众需求 / 特定地点或活动类别）时设为 true，否则 false。
- raw_query: 字符串，原样回写用户的输入。

请基于用户输入推断字段，不要编造与用户输入无关的偏好。仅输出 JSON。
"""

        try:
            response = chat_model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ])
            content = response.content
            # 正则清理可能带有的 markdown json 标签
            json_str = re.search(r'\{.*\}', content, re.DOTALL)
            intent = json.loads(json_str.group() if json_str else content)
            # 字段补全：防止 LLM 漏字段或返回非预期结构
            intent.setdefault("scenario", "family")
            intent.setdefault("child_friendly", False)
            intent.setdefault("diet_preference", "无")
            intent.setdefault(
                "is_leisure_planning", _heuristic_is_leisure(user_input)
            )
            intent.setdefault("need_retrieval", False)
            intent.setdefault("raw_query", user_input)
            print(f"[OK] 解析结果: {intent}")
            return intent
        except Exception as e:
            print(f"[WARN] 解析失败，使用默认家庭意图 + 关键词启发式: {e}")
            return {
                "scenario": "family",
                "child_friendly": True,
                "diet_preference": "减脂",
                "is_leisure_planning": _heuristic_is_leisure(user_input),
                "need_retrieval": False,
                "raw_query": user_input,
            }
