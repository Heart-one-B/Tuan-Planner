from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.model.factory import chat_model


_LEISURE_KEYWORDS = ("安排", "规划", "玩", "聚会", "带娃", "约", "计划", "出门")
_LOCATION_SENSITIVE_KEYWORDS = ("离家远", "附近", "别跑太远", "别离家太远", "就在这边", "近一点")
_FAMILY_KEYWORDS = ("家人", "家庭", "亲子", "老婆", "孩子", "儿子", "女儿", "全家", "老公")
_FRIENDS_KEYWORDS = ("朋友", "同学", "同事", "聚会")
_CUISINE_KEYWORDS = ("烧肉", "烧烤", "火锅", "西餐", "日料", "韩餐", "川菜", "粤菜", "湘菜", "轻食", "自助")
_LOCATION_HINTS = ("国贸", "望京", "朝阳", "海淀", "家附近", "公司附近")
_WEEKDAY_TOKENS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日", "周天")
_VALID_DAYPARTS = {"上午", "下午", "晚上", "全天"}
_RESTAURANT_HINTS = ("火锅", "烧烤", "烤肉", "轻食", "沙拉", "简餐", "中餐", "日料", "韩餐", "西餐", "川菜", "米饭", "自助", "健康", "养生")
_ACTIVITY_HINTS = ("室内", "户外", "亲子", "拍照", "休闲", "逛街", "展览", "运动", "散步", "日落", "夜景")


def _heuristic_is_leisure(user_input: str) -> bool:
    text = user_input or ""
    return any(kw in text for kw in _LEISURE_KEYWORDS)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out


def _coerce_text_list(value) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value and value not in {"无", "none", "None"} else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text and text not in {"无", "none", "None"} and text not in out:
                    out.append(text)
        return out
    return []


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
    if "晚上" in raw:
        daypart = "晚上"
    elif "全天" in raw or "一整天" in raw or "整天" in raw:
        daypart = "全天"
    elif "上午" in raw or "早上" in raw:
        daypart = "上午"
    elif "下午" in raw or "中午" in raw or "白天" in raw:
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
    if value == "早上":
        return "上午"
    if value in {"中午", "白天"}:
        return "下午"
    if "晚上" in value:
        return "晚上"
    if "下午" in value:
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

    time_info["time_phrase"] = _build_time_phrase(date_label, daypart)
    time_info["date_label"] = date_label
    time_info["daypart"] = daypart

    start_time_hint = time_info.get("start_time_hint")
    time_info["start_time_hint"] = start_time_hint.strip() if isinstance(start_time_hint, str) and start_time_hint.strip() else None
    duration_hours_hint = time_info.get("duration_hours_hint")
    if isinstance(duration_hours_hint, bool) or not isinstance(duration_hours_hint, int) or duration_hours_hint <= 0:
        time_info["duration_hours_hint"] = None
    return time_info


def _extract_location_info(user_input: str) -> dict:
    text = user_input or ""
    for hint in _LOCATION_HINTS:
        if hint in text:
            return {"origin_area_hint": hint, "location_text": hint}
    return {"origin_area_hint": None, "location_text": None}


def _build_restaurant_candidates(user_input: str, scenario: str, diet_preferences: list[str]) -> tuple[list[str], list[str]]:
    text = user_input or ""
    explicit = [token for token in _RESTAURANT_HINTS if token in text]
    keywords: list[str] = []

    if any(token in text for token in ("减脂", "减肥", "轻食", "健康")):
        keywords.extend(["轻食", "健康餐", "沙拉"])
    if any(token in text for token in ("火锅", "烧烤", "烤肉")):
        keywords.extend(["火锅", "烧烤", "烤肉"])
    if any(token in text for token in ("日料", "韩餐", "西餐", "川菜", "粤菜", "湘菜")):
        keywords.extend([token for token in _CUISINE_KEYWORDS if token in text])
    keywords.extend(explicit)
    keywords.extend(diet_preferences)

    if scenario == "family":
        keywords.extend(["亲子餐厅", "简餐", "自助"])
    elif scenario == "friends":
        keywords.extend(["聚餐", "简餐", "特色餐厅"])
    else:
        keywords.extend(["简餐", "聚餐", "特色餐厅"])

    if any(token in text for token in ("早餐", "早午餐", "早饭")):
        keywords.extend(["早餐", "早午餐", "咖啡"])
    if any(token in text for token in ("晚餐", "晚饭", "约饭")):
        keywords.extend(["晚餐", "聚餐", "约饭"])

    keywords = _dedupe_keep_order(keywords)
    if len(keywords) < 3:
        keywords.extend(["简餐", "聚餐", "特色餐厅"])

    excludes: list[str] = []
    if any(token in text for token in ("不吃烧烤", "不吃火锅", "不吃烤肉")):
        excludes.extend(["烧烤", "火锅", "烤肉"])
    if any(token in text for token in ("不要自助", "不想自助")):
        excludes.extend(["自助"])
    return _dedupe_keep_order(keywords)[:5], _dedupe_keep_order(excludes)[:3]


def _build_activity_candidates(user_input: str, scenario: str) -> tuple[list[str], list[str]]:
    text = user_input or ""
    explicit = [token for token in _ACTIVITY_HINTS if token in text]
    keywords: list[str] = []

    if "室内" in text:
        keywords.append("室内")
    if "户外" in text:
        keywords.append("户外")
    if "亲子" in text:
        keywords.append("亲子")
    if "拍照" in text:
        keywords.append("拍照")
    if "休闲" in text:
        keywords.append("休闲")

    if scenario == "family":
        keywords.extend(["室内", "户外", "亲子"])
    elif scenario == "friends":
        keywords.extend(["室内", "户外", "拍照"])
    else:
        keywords.extend(["室内", "户外", "休闲"])

    if any(token in text for token in ("展览", "看展", "博物馆")):
        keywords.extend(["展览", "看展"])
    if any(token in text for token in ("逛街", "citywalk")):
        keywords.extend(["逛街", "散步"])
    if any(token in text for token in ("日落", "夜景")):
        keywords.extend(["日落", "夜景", "户外"])
    if any(token in text for token in ("运动", "亲子")):
        keywords.extend(["运动", "亲子"])

    keywords = _dedupe_keep_order(keywords)
    if len(keywords) < 3:
        keywords.extend(["室内", "户外", "休闲"])
    if "室内" not in keywords:
        keywords.insert(0, "室内")
    if "户外" not in keywords:
        keywords.insert(1 if keywords else 0, "户外")
    return _dedupe_keep_order(keywords)[:5], _dedupe_keep_order(explicit)[:3]


def _adapt_activity_search_keywords(activity_keywords: list[str]) -> list[str]:
    mapping = {
        "逛街": ["商场", "购物中心", "步行街"],
        "散步": ["公园", "绿道", "步道"],
        "展览": ["展览", "美术馆", "博物馆"],
        "看展": ["展览", "美术馆", "博物馆"],
        "拍照": ["商场", "艺术中心", "景观"],
        "休闲": ["商场", "公园", "步行街"],
        "亲子": ["亲子乐园", "儿童乐园", "商场"],
    }
    search_keywords: list[str] = []
    for keyword in activity_keywords:
        if keyword in mapping:
            search_keywords.extend(mapping[keyword])
        else:
            search_keywords.append(keyword)
    return _dedupe_keep_order(search_keywords)[:6]


def _infer_missing_slots(user_input: str, intent: dict, runtime_origin_area: str = "") -> tuple[bool, dict[str, list[str]], str]:
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
        return True, missing, "这次是想和家人出去，还是和朋友一起安排？"
    if "time_day" in missing["global"]:
        return True, missing, "你想安排在今天、明天、后天，还是周末？"
    if "time_window" in missing["global"]:
        return True, missing, "你想安排在上午、下午，还是晚上？"
    if "origin_area" in missing["global"]:
        return True, missing, "你大概想从哪个区域出发？比如家附近、国贸、望京这类位置。"
    return True, missing, "你是想让我帮你安排一个本地半日活动吗？"


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
        diet_preferences = [str(item).strip() for item in diet_preference if str(item).strip() and str(item).strip() not in {"无", "none", "None"}]
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

    llm_restaurant_keywords = _coerce_text_list(normalized.get("restaurant_keywords"))
    llm_activity_keywords = _coerce_text_list(normalized.get("activity_keywords"))
    llm_restaurant_explicit_types = _coerce_text_list(normalized.get("restaurant_explicit_types"))
    llm_activity_explicit_types = _coerce_text_list(normalized.get("activity_explicit_types"))

    fallback_restaurant_keywords, fallback_restaurant_explicit_types = _build_restaurant_candidates(user_input, scenario, diet_preferences)
    fallback_activity_keywords, fallback_activity_explicit_types = _build_activity_candidates(user_input, scenario)

    restaurant_keywords = llm_restaurant_keywords[:] if llm_restaurant_keywords else fallback_restaurant_keywords
    activity_keywords = llm_activity_keywords[:] if llm_activity_keywords else fallback_activity_keywords
    restaurant_explicit_types = llm_restaurant_explicit_types[:] if llm_restaurant_explicit_types else fallback_restaurant_explicit_types
    activity_explicit_types = llm_activity_explicit_types[:] if llm_activity_explicit_types else fallback_activity_explicit_types

    if len(restaurant_keywords) < 3:
        for token in fallback_restaurant_keywords:
            if token not in restaurant_keywords:
                restaurant_keywords.append(token)
            if len(restaurant_keywords) >= 3:
                break

    if len(activity_keywords) < 3:
        for token in fallback_activity_keywords:
            if token not in activity_keywords:
                activity_keywords.append(token)
            if len(activity_keywords) >= 3:
                break

    normalized["child_friendly"] = child_friendly
    normalized["diet_preference"] = diet_preferences
    normalized["restaurant_keywords"] = _dedupe_keep_order(restaurant_keywords)[:5]
    normalized["activity_keywords"] = _dedupe_keep_order(activity_keywords)[:5]
    normalized["activity_search_keywords"] = _adapt_activity_search_keywords(normalized["activity_keywords"])
    normalized["restaurant_explicit_types"] = _dedupe_keep_order(restaurant_explicit_types)[:3]
    normalized["activity_explicit_types"] = _dedupe_keep_order(activity_explicit_types)[:3]
    normalized["raw_query"] = user_input
    llm_leisure = normalized.get("is_leisure_planning")
    normalized["is_leisure_planning"] = heuristic_leisure if llm_leisure is not True else True
    normalized["need_retrieval"] = bool(normalized.get("need_retrieval", False))
    return normalized


class IntentAgent:
    def parse(self, user_input: str, runtime_origin_area: str = "") -> dict:
        print("[Intent Agent] 正在解析用户自然语言意图...")
        system_prompt = """
你是一个本地生活意图解析助手。请严格分析用户输入，并仅返回 JSON。
必须包含以下字段：
- scenario: "family" / "friends" / "unknown" / "none"
- child_friendly: bool
- diet_preference: string 或 string list
- time: {date_label, daypart, time_phrase, start_time_hint, duration_hours_hint}
- is_leisure_planning: bool
- need_retrieval: bool
- clarification_needed: bool
- missing_slots: object
- follow_up_message: string
- location: {origin_area_hint, location_text}
- restaurant_keywords: 餐厅关键词候选列表，尽量至少 3 类，可直接用于本地生活搜索，不要生成“菜系”，“不辣餐厅”这种词，这种词放到高德中是搜不出来结果的
- activity_keywords: 活动关键词候选列表，尽量至少 3 类，可直接作为关键词用于api搜索，因此“逛街”、“散步”这类是不行的，因为调用API搜不出来东西，需要直接说“商场”、“公园”这种
- restaurant_explicit_types: 用户明确提到的餐厅类型列表；没有就空列表
- activity_explicit_types: 用户明确提到的活动类型列表；没有就空列表
- raw_query: 原样返回用户输入

关键词生成要求：
- 当用户历史偏好与最新要求矛盾时，优先考虑最新要求，能同时满足历史偏好最好，比如“历史偏好火锅，但朋友不能吃辣”，这个时候就可以尝试鸳鸯锅
- 如果用户明确提到了某种餐厅或活动类型，必须保留到对应 explicit_types 中。
- 如果用户明确提到了某种餐厅或活动类型，也应保留到对应 keywords 列表中。
- 除了用户明确提到的类型，还应额外补充 1 到 2 个互补类型，避免候选池只有单一类型。
- 对于餐厅关键词，如果用户说“晚上吃火锅”，不要只返回火锅类关键词，应该补充 1 到 2 个适合作为其他餐次候选的类型。
- 对于活动关键词，如果用户没有明确限制，也应保持多样性。尽量包含室内和户外两种类型的活动，不是直接把“室内”、“户外”放到关键词列表中去。
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
            clarification_needed, missing_slots, follow_up_message = _infer_missing_slots(user_input, intent, runtime_origin_area)
            intent["clarification_needed"] = clarification_needed
            intent["missing_slots"] = missing_slots
            intent["follow_up_message"] = follow_up_message
            print(f"[OK] 解析结果: {intent}")
            return intent
        except Exception as e:
            print(f"[WARN] 解析失败，使用规则 fallback: {e}")
            fallback_intent = _normalize_intent(
                user_input,
                {
                    "scenario": "family" if any(kw in user_input for kw in _FAMILY_KEYWORDS) else "unknown",
                    "child_friendly": any(kw in user_input for kw in _FAMILY_KEYWORDS),
                    "diet_preference": "减脂" if ("减脂" in user_input or "减肥" in user_input) else "无",
                    "is_leisure_planning": _heuristic_is_leisure(user_input),
                    "need_retrieval": False,
                    "raw_query": user_input,
                },
            )
            clarification_needed, missing_slots, follow_up_message = _infer_missing_slots(user_input, fallback_intent, runtime_origin_area)
            fallback_intent["clarification_needed"] = clarification_needed
            fallback_intent["missing_slots"] = missing_slots
            fallback_intent["follow_up_message"] = follow_up_message
            return fallback_intent
