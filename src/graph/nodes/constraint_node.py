# src/graph/nodes/constraint_node.py
from __future__ import annotations

from typing import Any

from src.graph.state import AgentState

# 用户没说城市时的兜底。个人助手知道用户常驻哪儿是合理的产品假设，
# 不是猜测——而"不知道城市就全国范围检索"必然出现「春熙路→昭通」
# 这类误判（实测）。
# 从配置读而不是写死：换城市的用户不该改代码。
DEFAULT_CITY = "成都"


# ── 工具函数 ──────────────────────────────────────────────────────────

def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _coerce_positive_int(v: Any) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return v if v > 0 else None


def _text_list(v: Any) -> list[str]:
    if isinstance(v, str):
        s = v.strip()
        return [s] if s and s not in {"无", "none", "None"} else []
    if isinstance(v, list):
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                s = item.strip()
                if s and s not in {"无", "none", "None"} and s not in out:
                    out.append(s)
        return out
    return []


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if x and not (x in seen or seen.add(x))]


def _norm_intent(v: Any) -> str:
    """三值归一。模型偶尔会返回别的字符串，退回 open——
    三种误判里 open 的代价最小（多搜几个品类 vs 整类需求被跳过）。"""
    s = (v or "open")
    s = s.strip() if isinstance(s, str) else "open"
    return s if s in {"explicit", "open", "none"} else "open"


def _build_current_request(raw_query: str, adjust_history: list[str]) -> str:
    """合成"用户当前有效的需求"，给下游 prompt 用。

    raw_query 是最初那句话（含可能已被撤回的要求），intent_node 靠它
    拼接调整历史。但 Planning / Evaluation 的 prompt 直接印它，就会
    拿已撤回的需求去评判——实测过：用户说"不吃火锅了"，方案换成了
    不辣的茶餐厅，评估器却打了 58 分并写"核心需求未满足"。

    两个字段各自含义单一，比一个字段承担两种含义安全。
    """
    if not adjust_history:
        return raw_query
    lines = [f"最初需求：{raw_query}", "", "用户后续提出的调整（优先级高于最初需求）："]
    lines += [f"{i}. {t}" for i, t in enumerate(adjust_history, 1)]
    lines.append("")
    lines.append(
        "评估与规划时以调整后的需求为准。已被调整撤回的最初要求"
        "不再是需求，不要因为方案没有满足它而扣分。"
    )
    return "\n".join(lines)


def _build_plan_context(
    intent: dict[str, Any], adjust_history: list[str] | None = None,
) -> dict[str, Any]:
    _PARTY = {
        "family": "family_with_kids",
        "friends": "friends_group",
        "couple": "couple",
        "team": "team",
    }
    _DEFAULTS = {"max_traffic_minutes": 40}

    scenario = intent.get("scenario") or "unknown"
    party = _PARTY.get(scenario, "default")

    participants = _safe_dict(intent.get("participants"))
    people_count = _coerce_positive_int(participants.get("people_count"))
    has_child = bool(participants.get("has_child"))
    child_age = _coerce_positive_int(participants.get("child_age"))

    time_info = _safe_dict(intent.get("time"))
    date_label = time_info.get("date_label") or ""
    start_time = time_info.get("start_time") or ""
    end_time = time_info.get("end_time") or ""
    time_phrase = time_info.get("time_phrase") or ""

    location = _safe_dict(intent.get("location"))
    origin_area = location.get("origin_area_hint") or ""
    # 用户说了城市就用他说的，没说用默认。不猜——猜错的代价是
    # 整个候选池落在另一个省。
    origin_city = (location.get("origin_city") or "").strip() or DEFAULT_CITY

    preferences = _safe_dict(intent.get("preferences"))
    diet_preference = _text_list(preferences.get("diet_preference"))
    activity_style = _text_list(preferences.get("activity_style"))
    must_avoid = _text_list(preferences.get("must_avoid"))

    distance_raw = (preferences.get("distance_preference") or "").strip()
    distance_preference = (
        "nearby" if distance_raw in {"别太远", "附近", "近一点", "近点", "不要太远"}
        else "balanced"
    )
    pace = (
        "relaxed" if "轻松" in activity_style
        else "compact" if "紧凑" in activity_style
        else "balanced"
    )

    raw_query = intent.get("raw_query") or ""
    current_request = _build_current_request(raw_query, list(adjust_history or []))

    # ── 两侧意图分流：完全对称 ────────────────────────────────────
    #
    # 三种模式对应三种搜索行为，以前混成一种：不管用户说什么都发散搜。
    # 用户说"想吃凉皮"，模型照样搜"川菜馆、私房菜、本地特色餐厅"——
    # **多余的搜索本身就是错误**：它稀释候选池，让 Planning 可能挑一个
    # 用户根本没要的东西。
    #
    # 【活动侧本次新增】实测 L2-04：用户说"就吃个午饭"，原来靠字符串
    # 匹配（"只吃饭"/"找餐厅"/"约饭"）判断要不要安排活动，"就吃个午饭"
    # 不在表里 → plan_mode=activity_plus_meal → FactAgent 又因为
    # "要求活动但模型没规划活动搜索"盲补了"公园、美术馆"。
    # 一个错误的分类判断制造了两次多余搜索。字符串匹配穷举不了用户的
    # 说法，这个判断本来就该模型做。
    restaurant_intent = _norm_intent(intent.get("restaurant_intent"))
    activity_intent = _norm_intent(intent.get("activity_intent"))

    restaurant_explicit = _text_list(intent.get("restaurant_explicit_types"))
    activity_explicit = _text_list(intent.get("activity_explicit_types"))

    # explicit 却没给出具体品类 → 声明与内容矛盾，两者不可能同时为真。
    # 不静默接受也不报错中断，退回更安全的 open。
    if restaurant_intent == "explicit" and not restaurant_explicit:
        restaurant_intent = "open"
    if activity_intent == "explicit" and not activity_explicit:
        activity_intent = "open"

    # ── 餐饮关键词 ──
    if restaurant_intent == "none":
        restaurant_keywords: list[str] = []
    elif restaurant_intent == "explicit":
        restaurant_keywords = _dedupe(restaurant_explicit)
    else:
        restaurant_keywords = _dedupe(
            _text_list(intent.get("restaurant_keywords"))
            or ["简餐", "聚餐", "特色餐厅"]
        )
        # 不再用 diet_preference 兜底——"清淡""不辣"是口味不是品类，
        # 拿它当搜索词会搜出零结果（实测「清淡餐饮」0 家）。
        # 口味约束由 fact_node 的 no_spicy 和下游筛选执行。
        if len(restaurant_keywords) < 3:
            restaurant_keywords = _dedupe(
                restaurant_keywords + ["简餐", "聚餐", "特色餐厅"]
            )

    # ── 活动关键词 ──
    if activity_intent == "none":
        activity_keywords: list[str] = []
    elif activity_intent == "explicit":
        activity_keywords = _dedupe(activity_explicit)
    else:
        activity_keywords = _dedupe(
            _text_list(intent.get("activity_keywords"))
            or (
                ["儿童乐园", "科技馆", "公园", "博物馆", "商场"]
                if has_child
                else ["公园", "美术馆", "博物馆", "商场", "购物中心"]
            )
        )

    need_activity = activity_intent != "none"
    need_restaurant = restaurant_intent != "none"

    if not need_activity and not need_restaurant:
        # 两侧都判 none：用户既不吃也不玩，这不是一个规划需求。
        # 退回 activity_only 而不是空转——route_after_intent 那层
        # 已经用 is_leisure_planning 挡过一次，走到这里说明模型认为
        # 这是规划需求，只是两个子判断都说不需要，矛盾。取代价小的一侧。
        need_activity = True
        activity_intent = "open"
        activity_keywords = ["公园", "商场"]

    if scenario == "family" and has_child and need_activity and need_restaurant:
        plan_mode = "family_with_kids"
    elif not need_activity:
        plan_mode = "meal_only"
    elif not need_restaurant:
        plan_mode = "activity_only"
    else:
        plan_mode = "activity_plus_meal"

    max_traffic = _DEFAULTS["max_traffic_minutes"]
    if has_child or distance_preference == "nearby":
        max_traffic = min(max_traffic, 30)

    return {
        "scenario": scenario,
        "party": party,
        "plan_mode": plan_mode,

        # raw_query        最初原话（含可能已撤回的要求）
        # current_request  当前有效需求，下游 prompt 读这个
        "raw_query": raw_query,
        "current_request": current_request,

        "restaurant_intent": restaurant_intent,
        "activity_intent": activity_intent,

        "date_label": date_label,
        "start_time": start_time,
        "end_time": end_time,
        "time_phrase": time_phrase,

        "people_count": people_count,
        "child_friendly": has_child,
        "child_age": child_age,

        "origin_area": origin_area,
        "origin_city": origin_city,

        "activity_keywords": activity_keywords,
        "restaurant_keywords": restaurant_keywords,
        "activity_explicit_types": activity_explicit,
        "restaurant_explicit_types": restaurant_explicit,
        "waypoint_requests": intent.get("waypoint_requests") or [],

        "max_traffic_minutes": max_traffic,

        "preferences": {
            "diet": diet_preference,
            "activity": activity_style,
            "avoid": must_avoid,
            "pace": pace,
            "distance": distance_preference,
        },
    }


# ── 节点 ──────────────────────────────────────────────────────────────

async def constraint_node(state: AgentState) -> dict:
    task_log = list(state.get("task_log") or [])
    errors = list(state.get("errors") or [])

    try:
        plan_context = _build_plan_context(
            intent=state.get("intent") or {},
            adjust_history=state.get("adjust_history") or [],
        )
        n_adjust = len(state.get("adjust_history") or [])
        task_log.append(
            f"constraint: scenario={plan_context.get('scenario')} "
            f"plan_mode={plan_context.get('plan_mode')} "
            f"r_intent={plan_context.get('restaurant_intent')} "
            f"a_intent={plan_context.get('activity_intent')} "
            f"r_kw={plan_context.get('restaurant_keywords')} "
            f"a_kw={plan_context.get('activity_keywords')} "
            f"adjust={n_adjust} "
            f"origin={plan_context.get('origin_area')}@{plan_context.get('origin_city')} "
            f"{plan_context.get('start_time')}-{plan_context.get('end_time')}"
        )
        return {
            "plan_context": plan_context,
            "task_log": task_log,
            "errors": errors,
        }
    except Exception as e:
        errors.append({"node": "constraint", "error": str(e), "recoverable": False})
        task_log.append(f"constraint: failed {e}")
        return {"task_log": task_log, "errors": errors}