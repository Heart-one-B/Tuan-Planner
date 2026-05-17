import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from src.model.factory import chat_model


# 启发式关键词：用于 LLM 解析失败时兜底判断 is_leisure_planning
_LEISURE_KEYWORDS = (
    "安排", "规划", "玩", "吃", "聚会", "带孩子", "逛", "约", "计划", "出门",
)
_TIME_HINT_KEYWORDS = (
    "今天", "下午", "晚上", "周末", "明天", "后天", "今晚", "周六", "周日",
)
_LOCATION_SENSITIVE_KEYWORDS = (
    "离家近", "附近", "别跑太远", "别离家太远", "就在这边", "近一点",
)
_FAMILY_KEYWORDS = ("老婆", "孩子", "儿子", "女儿", "全家", "老公")
_FRIENDS_KEYWORDS = ("朋友", "同学", "同事", "聚会")
_LOCATION_HINTS = ("国贸", "望京", "朝阳", "海淀", "家附近", "公司附近")


def _heuristic_is_leisure(user_input: str) -> bool:
    """关键词启发式：用户输入中出现任一休闲规划关键词即视为规划任务。"""
    text = user_input or ""
    return any(kw in text for kw in _LEISURE_KEYWORDS)


def _infer_missing_slots(
    user_input: str,
    intent: dict,
    runtime_origin_area: str = "",
) -> tuple[bool, dict[str, list[str]], str]:
    """根据用户输入与初步 intent 输出澄清结果。"""
    text = user_input or ""
    missing: dict[str, list[str]] = {"global": []}

    if intent.get("is_leisure_planning") is not True:
        return False, {}, ""

    scenario = intent.get("scenario")
    if scenario not in {"family", "friends"}:
        if any(kw in text for kw in _FAMILY_KEYWORDS):
            scenario = "family"
            intent["scenario"] = scenario
        elif any(kw in text for kw in _FRIENDS_KEYWORDS):
            scenario = "friends"
            intent["scenario"] = scenario
        else:
            missing["global"].append("scenario")

    if not any(kw in text for kw in _TIME_HINT_KEYWORDS):
        missing["global"].append("time_window")

    location_sensitive = any(kw in text for kw in _LOCATION_SENSITIVE_KEYWORDS)
    has_location_hint = any(kw in text for kw in _LOCATION_HINTS)
    has_runtime_origin = isinstance(runtime_origin_area, str) and bool(runtime_origin_area.strip())
    if location_sensitive and not has_location_hint and not has_runtime_origin:
        missing["global"].append("origin_area")

    if not missing["global"]:
        return False, {}, ""

    if "scenario" in missing["global"]:
        return True, missing, "这次是想和家人出门，还是和朋友一起安排？"
    if "time_window" in missing["global"]:
        return True, missing, "你大概想安排在今天下午、晚上，还是这个周末？"
    if "origin_area" in missing["global"]:
        return True, missing, "你现在大概想从哪个区域出发？比如家附近、国贸、望京这类位置，我可以尽量帮你安排得更近一些。"
    return True, missing, "你是想让我帮你安排一个本地半日活动吗？如果是，可以告诉我是和谁一起、什么时候出门。"


def _extract_location_info(user_input: str) -> dict:
    text = user_input or ""
    for hint in _LOCATION_HINTS:
        if hint in text:
            return {"origin_area_hint": hint, "location_text": hint}
    return {"origin_area_hint": None, "location_text": None}


def _normalize_intent(user_input: str, intent: dict) -> dict:
    """把 LLM/兜底结果归一到后续节点可稳定消费的结构。"""
    normalized = dict(intent) if isinstance(intent, dict) else {}
    scenario = normalized.get("scenario")
    if scenario not in {"family", "friends", "unknown", "none"}:
        scenario = "unknown" if _heuristic_is_leisure(user_input) else "none"
    normalized["scenario"] = scenario

    child_friendly = bool(normalized.get("child_friendly"))
    diet_preference = normalized.get("diet_preference", "无")
    if isinstance(diet_preference, str):
        diet_preferences = [] if diet_preference in {"", "无", "none", "None"} else [diet_preference]
    elif isinstance(diet_preference, list):
        diet_preferences = [str(item).strip() for item in diet_preference if str(item).strip() and str(item).strip() not in {"无", "none", "None"}]
    else:
        diet_preferences = []

    participants = normalized.get("participants")
    if not isinstance(participants, dict):
        participants = {}
    participants.setdefault("people_count", None)
    participants.setdefault("has_child", child_friendly)
    participants.setdefault("child_age", None)
    normalized["participants"] = participants

    time_info = normalized.get("time")
    if not isinstance(time_info, dict):
        time_info = {}
    if not isinstance(time_info.get("time_phrase"), str) or not time_info.get("time_phrase"):
        if "下午" in user_input:
            time_info["time_phrase"] = "今天下午"
        elif "晚上" in user_input or "今晚" in user_input:
            time_info["time_phrase"] = "今天晚上"
        elif "周末" in user_input:
            time_info["time_phrase"] = "周末"
        else:
            time_info["time_phrase"] = None
    time_info.setdefault("start_time_hint", None)
    time_info.setdefault("duration_hours_hint", None)
    normalized["time"] = time_info

    location_info = normalized.get("location")
    if not isinstance(location_info, dict):
        location_info = {}
    extracted_location = _extract_location_info(user_input)
    location_info.setdefault("origin_area_hint", extracted_location["origin_area_hint"])
    location_info.setdefault("location_text", extracted_location["location_text"])
    normalized["location"] = location_info

    preferences = normalized.get("preferences")
    if not isinstance(preferences, dict):
        preferences = {}
    if not isinstance(preferences.get("distance_preference"), str) or not preferences.get("distance_preference"):
        preferences["distance_preference"] = "别太远" if ("远" in user_input or "附近" in user_input) else ""
    preferences["diet_preference"] = diet_preferences
    activity_style = preferences.get("activity_style")
    preferences["activity_style"] = activity_style if isinstance(activity_style, list) else []
    must_avoid = preferences.get("must_avoid")
    preferences["must_avoid"] = must_avoid if isinstance(must_avoid, list) else []
    normalized["preferences"] = preferences

    normalized["child_friendly"] = child_friendly
    normalized["diet_preference"] = diet_preferences
    normalized.setdefault("raw_query", user_input)
    normalized["is_leisure_planning"] = bool(
        normalized.get("is_leisure_planning", _heuristic_is_leisure(user_input))
    )
    normalized["need_retrieval"] = bool(normalized.get("need_retrieval", False))
    return normalized


class IntentAgent:
    def parse(self, user_input: str, runtime_origin_area: str = "") -> dict:
        print("[Intent Agent] 正在解析用户自然语言意图...")
        system_prompt = """
你是一个本地生活意图解析助手。请严格分析用户输入，并仅返回 JSON（不要包含任何其他文字、不要包裹 markdown 代码块）。

必须包含以下字段：
- scenario: 字符串，"family"（家庭场景）/ "friends"（朋友场景）/ "unknown"（是规划任务但场景不明）/ "none"（与本地生活无关）。
- child_friendly: bool，是否需要儿童友好的安排。
- diet_preference: 字符串或字符串列表，例如 "减脂" / "无辣" / "素食" / "无"。
- is_leisure_planning: bool，用户是否在做"半日 / 一日休闲活动规划"，如吃喝玩乐、家庭出游、朋友聚餐、约会安排等。
    判定指南：
    * "帮我安排周日全家出游" / "想找个适合带孩子的餐厅" / "下午想和朋友逛街吃饭" → true。
    * "今天天气怎样" / "帮我写一段 Python 代码" / "翻译这段英文" / "科普一下什么是 RAG" / "Python 字典怎么排序" → false。
- need_retrieval: bool，是否需要补充语义检索信息。当用户提到具体偏好（特定菜系 / 口味 / 活动类型 / 小众需求 / 特定地点或活动类别）时设为 true，否则 false。
- clarification_needed: bool，当前信息是否不足以继续后续规划。
- missing_slots: 对象，形如 {"global": ["scenario"]}，若无需澄清则返回空对象 {}。
- follow_up_message: 字符串，若 clarification_needed=true，给出一条最关键的追问；否则返回空字符串。
- location: 对象，包含 origin_area_hint 和 location_text；若无位置线索可为 null / 空字符串。
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
            intent = _normalize_intent(user_input, intent)
            clarification_needed, missing_slots, follow_up_message = _infer_missing_slots(
                user_input, intent, runtime_origin_area
            )
            intent["clarification_needed"] = clarification_needed
            intent["missing_slots"] = missing_slots
            intent["follow_up_message"] = follow_up_message
            print(f"[OK] 解析结果: {intent}")
            return intent
        except Exception as e:
            print(f"[WARN] 解析失败，使用默认家庭意图 + 关键词启发式: {e}")
            fallback_intent = _normalize_intent(user_input, {
                "scenario": "family" if any(kw in user_input for kw in _FAMILY_KEYWORDS) else "unknown",
                "child_friendly": True if any(kw in user_input for kw in _FAMILY_KEYWORDS) else False,
                "diet_preference": "减脂" if ("减脂" in user_input or "减肥" in user_input) else "无",
                "is_leisure_planning": _heuristic_is_leisure(user_input),
                "need_retrieval": False,
                "raw_query": user_input,
            })
            clarification_needed, missing_slots, follow_up_message = _infer_missing_slots(
                user_input, fallback_intent, runtime_origin_area
            )
            fallback_intent["clarification_needed"] = clarification_needed
            fallback_intent["missing_slots"] = missing_slots
            fallback_intent["follow_up_message"] = follow_up_message
            return fallback_intent
