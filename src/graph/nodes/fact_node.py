# src/graph/nodes/fact_node.py
from __future__ import annotations

from agents.fact.agent import FactAgent
from src.graph.state import AgentState
from src.graph.tracing import parent_span_of
from src.model.factory import build_llm_client

# 用户表达"不吃辣"的说法。命中任一才启用 typecode 层面的辣度约束
# ——没说过不吃辣时，川菜是完全正当的推荐，凭空施加约束是替用户
# 做决定。
_NO_SPICY_SIGNALS = ("不辣", "不吃辣", "不能吃辣", "清淡", "微辣",
                     "少辣", "忌辣", "重辣", "辣")


def _wants_no_spicy(prefs: dict) -> bool:
    """同时看 diet（正向偏好，如"清淡"）和 avoid（负向排除，如"重辣"）
    ——两个字段都可能承载同一个意思，只看一个会漏。实测记忆召回后
    的结果是 diet=['不辣','清淡'] + avoid=['连锁店','重辣']，
    信息分散在两处。"""
    signals = list(prefs.get("diet") or []) + list(prefs.get("avoid") or [])
    return any(sig in item for item in signals for sig in _NO_SPICY_SIGNALS)


async def fact_node(state: AgentState) -> dict:
    task_log = list(state.get("task_log") or [])
    errors = list(state.get("errors") or [])
    agent_outputs = dict(state.get("agent_outputs") or {})

    plan_context = state.get("plan_context") or {}
    intent = state.get("intent") or {}

    task = _build_fact_task(intent, plan_context)

    # 用户点名时，那几个词是**硬约束**不是提示。
    # 走参数不走任务文本：文本是给模型看的，模型可以选择不理会；
    # 参数是给代码看的，代码会强制执行。"没有多余搜索"这个保证
    # 不能建立在模型听话上——实测过它会造"清淡川菜"这种词来调和
    # 冲突信号。
    explicit_restaurants = (
        list(plan_context.get("restaurant_keywords") or [])
        if plan_context.get("restaurant_intent") == "explicit" else []
    )
    explicit_activities = (
        list(plan_context.get("activity_keywords") or [])
        if plan_context.get("activity_intent") == "explicit" else []
    )

    # 辣度约束：只有它能靠 typecode 可靠执行（050102 川菜 /
    # 050108 湘菜 是确定的）。"不要连锁""想吃本地特色"没有可靠判据，
    # 不传下去假装能执行。
    no_spicy = _wants_no_spicy(plan_context.get("preferences") or {})

    try:
        result = await FactAgent(llm_client=build_llm_client()).run(
            task=task,
            plan_mode=plan_context.get("plan_mode", "activity_plus_meal"),
            explicit_restaurant_keywords=explicit_restaurants,
            explicit_activity_keywords=explicit_activities,
            origin_city=plan_context.get("origin_city") or "",
            no_spicy=no_spicy,
            parent_span=parent_span_of(state),
        )
        agent_outputs["fact"] = {
            "status": result.status,
            "summary": result.summary,
            "data": result.data,
        }
        task_log.append(
            f"fact: status={result.status} no_spicy={no_spicy} "
            f"summary={result.summary}"
        )
        if result.status == "error":
            errors.append({"node": "fact", "error": result.summary,
                          "recoverable": False})
    except Exception as e:
        agent_outputs["fact"] = {"status": "error", "summary": str(e), "data": {}}
        errors.append({"node": "fact", "error": str(e), "recoverable": False})
        task_log.append(f"fact: exception {e}")

    return {
        "agent_outputs": agent_outputs,
        "task_log": task_log,
        "errors": errors,
    }


def _build_fact_task(intent: dict, plan_context: dict) -> str:
    """把 plan_context 整理成自然语言任务描述喂给 Fact Agent。"""
    parts = []
    prefs = plan_context.get("preferences") or {}

    raw_query = plan_context.get("raw_query") or intent.get("raw_query") or ""
    if raw_query:
        parts.append(f"用户需求：{raw_query}")

    origin = plan_context.get("origin_area") or ""
    city = plan_context.get("origin_city") or ""
    if origin:
        # 城市写进同一行：FactAgent 的 _extract_line 按"出发地"取值，
        # 单独一行"城市：xx"它读不到。city 同时也通过参数传，
        # 文本这份是给模型看的上下文。
        parts.append(f"出发地：{origin}" + (f"（{city}）" if city else ""))

    scenario = plan_context.get("scenario") or ""
    people = plan_context.get("people_count") or ""
    has_child = plan_context.get("child_friendly") or False
    child_age = plan_context.get("child_age")
    if scenario or people:
        child_info = ""
        if has_child:
            child_info = "，有孩子随行"
            if child_age:
                child_info += f"（{child_age}岁）"
        parts.append(f"场景：{scenario}，人数：{people}{child_info}")

    start = plan_context.get("start_time") or ""
    end = plan_context.get("end_time") or ""
    if start or end:
        parts.append(f"时间：{start} 到 {end}")

    activity_intent = plan_context.get("activity_intent") or "open"
    restaurant_intent = plan_context.get("restaurant_intent") or "open"
    activity_keywords = plan_context.get("activity_keywords") or []
    restaurant_keywords = plan_context.get("restaurant_keywords") or []

    # 活动与餐饮各按各的 intent 描述，让模型看到的任务本身就是明确的。
    # 代码侧的硬约束（explicit_*_keywords）是保证，文字是配合——
    # 两者一致时模型不会产生"想扩展但被拦住"的拉扯，规划质量更好。
    if activity_intent == "none":
        parts.append("本次不需要安排活动，不要搜索任何活动场所。")
    elif activity_intent == "explicit":
        parts.append(
            f"用户明确点名要去：{'、'.join(activity_keywords)}。"
            f"活动只搜这些关键词，不要自行扩展成其它类型。"
        )
    elif activity_keywords:
        parts.append(f"活动搜索关键词：{'、'.join(activity_keywords)}")

    if restaurant_intent == "none":
        parts.append("本次不需要安排餐饮，不要搜索任何餐厅。")
    elif restaurant_intent == "explicit":
        parts.append(
            f"用户明确点名要吃：{'、'.join(restaurant_keywords)}。"
            f"餐饮只搜这些关键词，不要自行扩展成其它品类或菜系。"
        )
    elif restaurant_keywords:
        parts.append(f"餐饮搜索关键词：{'、'.join(restaurant_keywords)}")

    waypoint_requests = plan_context.get("waypoint_requests") or []
    if waypoint_requests:
        parts.append(f"途径小需求（需单独搜索）："
                    f"{'、'.join(str(w) for w in waypoint_requests)}")

    avoid = prefs.get("avoid") or []
    if avoid:
        parts.append(f"明确排除：{'、'.join(avoid)}")

    # 用户点名品类时不列饮食偏好：他已经说了要吃什么，历史偏好这时
    # 只会制造干扰（"想吃火锅"+"不吃辣"让模型不知所措，实测造出过
    # "清淡川菜"这种自相矛盾的关键词）。本轮原话优先于历史偏好。
    diet = prefs.get("diet") or []
    if diet and restaurant_intent != "explicit":
        parts.append(f"饮食偏好：{'、'.join(diet)}")

    activity_style = prefs.get("activity") or []
    if activity_style:
        parts.append(f"活动风格偏好：{'、'.join(activity_style)}")

    return "\n".join(parts)