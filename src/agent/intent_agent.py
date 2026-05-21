import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.model.factory import chat_model


_LEISURE_KEYWORDS = (
    "安排",
    "规划",
    "玩",
    "吃",
    "聚会",
    "带孩子",
    "遛",
    "约",
    "计划",
    "出门",
)
_LOCATION_SENSITIVE_KEYWORDS = (
    "离家近",
    "附近",
    "别跑太远",
    "别离家太远",
    "就在这边",
    "近一点",
)
_FAMILY_KEYWORDS = ("家人", "家庭", "亲子", "老婆", "孩子", "儿子", "女儿", "全家", "老公")
_FRIENDS_KEYWORDS = ("朋友", "同学", "同事", "聚会")
_CUISINE_KEYWORDS = ("烤肉", "烧烤", "火锅", "西餐", "日料", "韩餐", "川菜", "粤菜", "湘菜", "轻食", "自助")
_LOCATION_HINTS = ("国贸", "望京", "朝阳", "海淀", "家附近", "公司附近")
_WEEKDAY_TOKENS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日", "周天")
_VALID_DAYPARTS = {"上午", "下午", "晚上", "全天"}


def _heuristic_is_leisure(user_input: str) -> bool:
    text = user_input or ""
    return any(kw in text for kw in _LEISURE_KEYWORDS)


def _extract_time_semantics(text: str) -> tuple[str | None, str | None]:
    raw = text or ""
    date_label = None
    if "周末" in raw:
        date_label = "周末"
    elif "今天" in raw:
        date_label = "今天"
    elif "明天" in raw:
        date_label = "明天"
    elif "后天" in raw:
        date_label = "后天"
    else:
        for token in _WEEKDAY_TOKENS:
            if token in raw:
                prefix = "下周" if "下周" in raw else ("本周" if "本周" in raw else "")
                date_label = f"{prefix}{token}"
                break

    daypart = None
    if "晚上" in raw or "今晚" in raw:
        daypart = "晚上"
    elif "全天" in raw or "一整天" in raw or "整天" in raw:
        daypart = "全天"
    elif "上午" in raw or "早上" in raw:
        daypart = "上午"
    elif "下午" in raw:
        daypart = "下午"
    elif "中午" in raw or "白天" in raw:
        daypart = "下午"

    return date_label, daypart


def _build_time_phrase(date_label: str | None, daypart: str | None) -> str | None:
    if date_label and daypart:
        return f"{date_label}{daypart}"
    if date_label:
        return date_label
    return None


def _normalize_daypart(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value in _VALID_DAYPARTS:
        return value
    if value in {"早上", "中午", "白天"}:
        return "上午" if value == "早上" else ("下午" if value != "白天" else "上午")
    if value in {"中午", "白天"}:
        return "下午"
    if "晚" in value:
        return "晚上"
    if "下" in value:
        return "下午"
    return None


def _normalize_date_label(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    date_label, _ = _extract_time_semantics(value)
    return date_label or value


def _normalize_time_info(user_input: str, raw_time_info: dict) -> dict:
    time_info = dict(raw_time_info) if isinstance(raw_time_info, dict) else {}

    llm_date_label = _normalize_date_label(time_info.get("date_label"))
    llm_daypart = _normalize_daypart(time_info.get("daypart"))

    candidate_phrase = time_info.get("time_phrase")
    if candidate_phrase is None:
        candidate_phrase = ""
    if not isinstance(candidate_phrase, str):
        candidate_phrase = str(candidate_phrase)
    phrase_date_label, phrase_daypart = _extract_time_semantics(candidate_phrase)

    date_label = llm_date_label or phrase_date_label
    daypart = llm_daypart or phrase_daypart

    if not date_label:
        fallback_date_label, fallback_daypart = _extract_time_semantics(user_input)
        date_label = fallback_date_label
        if not daypart:
            daypart = fallback_daypart
    elif not daypart:
        _, fallback_daypart = _extract_time_semantics(user_input)
        daypart = fallback_daypart

    normalized_phrase = _build_time_phrase(date_label, daypart)
    time_info["time_phrase"] = normalized_phrase
    time_info["date_label"] = date_label
    time_info["daypart"] = daypart

    start_time_hint = time_info.get("start_time_hint")
    if not isinstance(start_time_hint, str) or not start_time_hint.strip():
        time_info["start_time_hint"] = None
    else:
        time_info["start_time_hint"] = start_time_hint.strip()

    duration_hours_hint = time_info.get("duration_hours_hint")
    if isinstance(duration_hours_hint, bool) or not isinstance(duration_hours_hint, int) or duration_hours_hint <= 0:
        time_info["duration_hours_hint"] = None

    return time_info


def _infer_missing_slots(
    user_input: str,
    intent: dict,
    runtime_origin_area: str = "",
) -> tuple[bool, dict[str, list[str]], str]:
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

    time_info = intent.get("time")
    time_info = time_info if isinstance(time_info, dict) else {}
    date_label = time_info.get("date_label")
    daypart = time_info.get("daypart")
    if not isinstance(date_label, str) or not date_label.strip():
        missing["global"].append("time_day")
    elif not isinstance(daypart, str) or not daypart.strip():
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
    if "time_day" in missing["global"]:
        return True, missing, "你想安排在今天、明天、周五，还是周末？"
    if "time_window" in missing["global"]:
        return True, missing, f"你想安排在{date_label}下午还是{date_label}晚上？"
    if "origin_area" in missing["global"]:
        return True, missing, "你现在大概想从哪个区域出发？比如家附近、国贸、望京这类位置，我可以尽量帮你安排得更近一点。"
    return True, missing, "你是想让我帮你安排一个本地半日活动吗？如果是，可以告诉我是和谁一起、什么时候出门。"


def _extract_location_info(user_input: str) -> dict:
    text = user_input or ""
    for hint in _LOCATION_HINTS:
        if hint in text:
            return {"origin_area_hint": hint, "location_text": hint}
    return {"origin_area_hint": None, "location_text": None}


def _normalize_intent(user_input: str, intent: dict) -> dict:
    normalized = dict(intent) if isinstance(intent, dict) else {}
    heuristic_leisure = _heuristic_is_leisure(user_input)
    scenario = normalized.get("scenario")
    if scenario not in {"family", "friends", "unknown", "none"}:
        scenario = "unknown" if heuristic_leisure else "none"
    normalized["scenario"] = scenario

    child_friendly = bool(normalized.get("child_friendly"))
    diet_preference = normalized.get("diet_preference", "无")
    if isinstance(diet_preference, str):
        diet_preferences = [] if diet_preference in {"", "无", "none", "None"} else [diet_preference]
    elif isinstance(diet_preference, list):
        diet_preferences = [
            str(item).strip()
            for item in diet_preference
            if str(item).strip() and str(item).strip() not in {"无", "none", "None"}
        ]
    else:
        diet_preferences = []

    for cuisine in _CUISINE_KEYWORDS:
        if cuisine in user_input and cuisine not in diet_preferences:
            diet_preferences.append(cuisine)

    participants = normalized.get("participants")
    if not isinstance(participants, dict):
        participants = {}
    participants.setdefault("people_count", None)
    participants.setdefault("has_child", child_friendly)
    participants.setdefault("child_age", None)
    normalized["participants"] = participants

    normalized["time"] = _normalize_time_info(user_input, normalized.get("time"))

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
        preferences["distance_preference"] = "别太远" if ("近" in user_input or "附近" in user_input) else ""
    preferences["diet_preference"] = diet_preferences
    activity_style = preferences.get("activity_style")
    preferences["activity_style"] = activity_style if isinstance(activity_style, list) else []
    must_avoid = preferences.get("must_avoid")
    preferences["must_avoid"] = must_avoid if isinstance(must_avoid, list) else []
    normalized["preferences"] = preferences

    normalized["child_friendly"] = child_friendly
    normalized["diet_preference"] = diet_preferences
    normalized["raw_query"] = user_input
    llm_leisure = normalized.get("is_leisure_planning")
    normalized["is_leisure_planning"] = heuristic_leisure if llm_leisure is not True else True
    normalized["need_retrieval"] = bool(normalized.get("need_retrieval", False))
    return normalized


class IntentAgent:
    def parse(self, user_input: str, runtime_origin_area: str = "") -> dict:
        print("[Intent Agent] 正在解析用户自然语言意图...")
        system_prompt = """
你是一个本地生活意图解析助手。请严格分析用户输入，并仅返回 JSON（不要包含任何其他文字，不要包裹 markdown 代码块）。
必须包含以下字段：
- scenario: 字符串，"family"（家庭场景）/ "friends"（朋友场景）/ "unknown"（是规划任务但场景不明）/ "none"（与本地生活无关）。
- child_friendly: bool，是否需要儿童友好的安排。
- diet_preference: 字符串或字符串列表，例如 "减脂" / "无辣" / "素食" / "无"。
- time: 对象，必须包含以下字段：
  - date_label: 字符串或 null。用于表示日期/星期语义，例如 "今天"、"明天"、"周五"、"本周五"、"周末"。
  - daypart: 字符串或 null。只允许返回 "下午" 或 "晚上"；如果用户没有明确说时段，就返回 null。
  - time_phrase: 字符串或 null。尽量保留用户原始时间表达，例如 "周五"、"周五晚上"、"本周五下午"、"今天下午"、"周末"。
  - start_time_hint: 字符串或 null。只有当用户明确给出具体出发时间时才填写，例如 "18:30"；否则为 null。
  - duration_hours_hint: 数字或 null。只有当用户明确给出时长时才填写，例如 4；否则为 null。
- is_leisure_planning: bool，用户是否在做"半日 / 一日休闲活动规划"，如吃喝玩乐、家庭出游、朋友聚餐、约会安排等。
  判定指南：
  * "帮我安排周日全家出游" / "想找一个适合带孩子的餐厅" / "下午想和朋友逛街吃饭" -> true。
  * "今天天气怎么样" / "帮我写一段 Python 代码" / "翻译这段英文" / "科普一下什么是 RAG" / "Python 字典怎么排序" -> false。
- need_retrieval: bool，是否需要补充语义检索信息。当用户提到具体偏好（特定菜系 / 口味 / 活动类型 / 小众需求 / 特定地点或活动类别）时设为 true，否则 false。
- clarification_needed: bool，当前信息是否不足以继续后续规划。
- missing_slots: 对象，形如 {"global": ["scenario"]}，若无需澄清则返回空对象 {}。
- follow_up_message: 字符串，若 clarification_needed=true，给出一条最关键的追问；否则返回空字符串。
- location: 对象，包含 origin_area_hint 和 location_text；若无位置线索可为 null / 空字符串。
- raw_query: 字符串，原样回写用户输入。

时间处理要求：
- 日期/星期和时段都是硬约束，不能省略后默认补齐。
- 如果用户只说了 "周五"，请返回 `date_label="周五"`、`daypart=null`、`time_phrase="周五"`。
- 如果用户说了 "周五晚上"、"周六下午"，请返回对应的 `date_label`、`daypart` 和完整 `time_phrase`。
- 不要因为缺少精确出发时间就把 `time_phrase` 设为 null。

请基于用户输入推断字段，不要编造与用户输入无关的偏好。仅输出 JSON。
"""

        try:
            response = chat_model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input),
                ]
            )
            content = response.content
            json_str = re.search(r"\{.*\}", content, re.DOTALL)
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
            fallback_intent = _normalize_intent(
                user_input,
                {
                    "scenario": "family" if any(kw in user_input for kw in _FAMILY_KEYWORDS) else "unknown",
                    "child_friendly": True if any(kw in user_input for kw in _FAMILY_KEYWORDS) else False,
                    "diet_preference": "减脂" if ("减脂" in user_input or "减肥" in user_input) else "无",
                    "is_leisure_planning": _heuristic_is_leisure(user_input),
                    "need_retrieval": False,
                    "raw_query": user_input,
                },
            )
            clarification_needed, missing_slots, follow_up_message = _infer_missing_slots(
                user_input, fallback_intent, runtime_origin_area
            )
            fallback_intent["clarification_needed"] = clarification_needed
            fallback_intent["missing_slots"] = missing_slots
            fallback_intent["follow_up_message"] = follow_up_message
            return fallback_intent
