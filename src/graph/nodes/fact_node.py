from __future__ import annotations

from src.graph.state import AgentState
from src.model.factory import build_llm_client
from agents.fact.agent import FactAgent


async def fact_node(state: AgentState) -> dict:
    task_log = list(state.get("task_log") or [])
    errors   = list(state.get("errors") or [])
    agent_outputs = dict(state.get("agent_outputs") or {})

    plan_context = state.get("plan_context") or {}
    intent       = state.get("intent") or {}

    task = _build_fact_task(intent, plan_context)

    try:
        result = await FactAgent(llm_client=build_llm_client()).run(
            task=task,
            plan_mode=plan_context.get("plan_mode", "activity_plus_meal"),
            trace_id=state.get("session_id"),
        )
        agent_outputs["fact"] = {
            "status":  result.status,
            "summary": result.summary,
            "data":    result.data,
        }
        task_log.append(f"fact: status={result.status} summary={result.summary}")

        if result.status == "error":
            errors.append({
                "node": "fact",
                "error": result.summary,
                "recoverable": False,
            })
    except Exception as e:
        agent_outputs["fact"] = {"status": "error", "summary": str(e), "data": {}}
        errors.append({"node": "fact", "error": str(e), "recoverable": False})
        task_log.append(f"fact: exception {e}")

    return {
        "agent_outputs": agent_outputs,
        "task_log":      task_log,
        "errors":        errors,
    }


def _build_fact_task(intent: dict, plan_context: dict) -> str:
    """把 plan_context 整理成自然语言任务描述喂给 Fact Agent。"""
    parts = []
    prefs = plan_context.get("preferences") or {}

    # 用户原始需求
    raw_query = plan_context.get("raw_query") or intent.get("raw_query") or ""
    if raw_query:
        parts.append(f"用户需求：{raw_query}")

    # 出发地
    origin = plan_context.get("origin_area") or ""
    if origin:
        parts.append(f"出发地：{origin}")

    # 场景和人员
    scenario = plan_context.get("scenario") or ""
    people   = plan_context.get("people_count") or ""
    has_child = plan_context.get("child_friendly") or False
    child_age = plan_context.get("child_age")
    if scenario or people:
        child_info = ""
        if has_child:
            child_info = f"，有孩子随行"
            if child_age:
                child_info += f"（{child_age}岁）"
        parts.append(f"场景：{scenario}，人数：{people}{child_info}")

    # 时间
    start = plan_context.get("start_time") or ""
    end   = plan_context.get("end_time") or ""
    if start or end:
        parts.append(f"时间：{start} 到 {end}")

    # 行程类型
    plan_mode = plan_context.get("plan_mode", "activity_plus_meal")
    need_activity   = plan_mode != "meal_only"
    need_restaurant = plan_mode != "activity_only"

    # 活动搜索关键词
    if need_activity:
        explicit_types   = plan_context.get("activity_explicit_types") or []
        activity_keywords = plan_context.get("activity_keywords") or []
        if explicit_types:
            parts.append(f"用户点名活动类型（必须搜到）：{'、'.join(explicit_types)}")
        if activity_keywords:
            parts.append(f"活动搜索关键词：{'、'.join(activity_keywords)}")

    # 餐厅搜索关键词
    if need_restaurant:
        explicit_rest    = plan_context.get("restaurant_explicit_types") or []
        restaurant_keywords = plan_context.get("restaurant_keywords") or []
        if explicit_rest:
            parts.append(f"用户点名餐厅类型（必须搜到）：{'、'.join(explicit_rest)}")
        if restaurant_keywords:
            parts.append(f"餐饮搜索关键词：{'、'.join(restaurant_keywords)}")

    # 途径小需求
    waypoint_requests = plan_context.get("waypoint_requests") or []
    if waypoint_requests:
        parts.append(f"途径小需求（需单独搜索）：{'、'.join(str(w) for w in waypoint_requests)}")

    # 排除项
    avoid = prefs.get("avoid") or []
    if avoid:
        parts.append(f"明确排除：{'、'.join(avoid)}")

    # 其他偏好
    diet = prefs.get("diet") or []
    if diet:
        parts.append(f"饮食偏好：{'、'.join(diet)}")

    activity_style = prefs.get("activity") or []
    if activity_style:
        parts.append(f"活动风格偏好：{'、'.join(activity_style)}")

    return "\n".join(parts)