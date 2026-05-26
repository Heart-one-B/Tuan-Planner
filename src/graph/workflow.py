from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    activity_search_node,
    candidate_planning_node,
    clarification_node,
    confirmation_node,
    constraint_collect_node,
    crowd_risk_node,
    execution_node,
    fact_gathering_node,
    final_message_node,
    final_plan_node,
    intent_node,
    llm_answer_node,
    interaction_wait_node,
    location_lookup_node,
    location_fallback_node,
    location_permission_node,
    presentation_node,
    queue_check_node,
    reject_node,
    repair_loop_node,
    retrieval_node,
    rule_validation_node,
    route_after_confirmation,
    schedule_timing_node,
    time_normalize_node,
    restaurant_search_node,
    traffic_eta_node,
    weather_check_node,
    scoring_node,
)
from src.graph.state import AgentState


_MAX_CLARIFICATION_ROUNDS = 5


def route_after_intent_with_clarification(state: AgentState) -> str:
    intent = state.get("intent")
    if not isinstance(intent, dict):
        return "constraint_build"
    if intent.get("is_leisure_planning") is False:
        return "llm_answer"
    missing_global = (intent.get("missing_slots", {}) or {}).get("global", [])
    if intent.get("clarification_needed") is True and "scenario" in missing_global:
        return "clarification"
    if intent.get("clarification_needed") is True and any(slot in missing_global for slot in ("time_day", "time_window")):
        return "clarification"
    location = intent.get("location")
    origin_area_hint = ""
    if isinstance(location, dict):
        hint = location.get("origin_area_hint")
        if isinstance(hint, str):
            origin_area_hint = hint.strip()
    # 若用户已经给出地点线索，就直接进入后续约束收集，不再强制打断询问定位权限。
    if not origin_area_hint and not state.get("runtime_origin_area") and "location_permission_granted" not in state:
        return "location_permission"
    if intent.get("is_leisure_planning") is True and intent.get("need_retrieval") is True:
        return "retrieval"
    if intent.get("clarification_needed") is True:
        return "clarification"
    return "constraint_build"


def route_after_clarification(state: AgentState) -> str:
    if state.get("pending_action") == "clarification":
        return "interaction_wait"
    clarification_round = state.get("clarification_round", 0)
    if isinstance(clarification_round, bool) or not isinstance(clarification_round, int):
        clarification_round = 0
    if clarification_round >= _MAX_CLARIFICATION_ROUNDS:
        return "llm_answer"
    return "intent"


def route_after_candidate_planning(state: AgentState) -> str:
    return "rule_validation"


def route_after_rule_validation(state: AgentState) -> str:
    result = state.get("rule_validation_result")
    valid_plans = result.get("valid_plans") if isinstance(result, dict) else None
    if isinstance(valid_plans, list) and len(valid_plans) >= 3:
        return "scoring"
    return "repair_loop"


def route_after_repair_loop_new(state: AgentState) -> str:
    repair_loop_result = state.get("repair_loop_result")
    if not isinstance(repair_loop_result, dict):
        repair_loop_result = {}
    next_constraint_build = repair_loop_result.get("next_constraint_build")
    if not isinstance(next_constraint_build, dict):
        next_constraint_build = {}
    context_memory = next_constraint_build.get("context_memory")
    if not isinstance(context_memory, dict):
        context_memory = {}
    repair_round = context_memory.get("repair_round")
    if isinstance(repair_round, int) and repair_round >= 2:
        return "final_plan"
    return "constraint_build"


def route_after_scoring(state: AgentState) -> str:
    return "final_plan"


def route_after_location_permission(state: AgentState) -> str:
    if state.get("pending_action") == "location_permission":
        return "interaction_wait"
    return "location_lookup" if state.get("location_permission_granted") else "constraint_build"


def route_after_location_fallback(state: AgentState) -> str:
    if state.get("pending_action") == "location_fallback":
        return "interaction_wait"
    return "constraint_build"


def route_after_confirmation(state: AgentState) -> str:
    if state.get("pending_action") == "confirmation":
        return "interaction_wait"
    if state.get("user_confirmed"):
        return "execute"
    return "replan"


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("time_normalize", time_normalize_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("llm_answer", llm_answer_node)
    graph.add_node("location_permission", location_permission_node)
    graph.add_node("location_lookup", location_lookup_node)
    graph.add_node("location_fallback", location_fallback_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("constraint_build", constraint_collect_node)

    graph.add_node("weather_check", weather_check_node)
    graph.add_node("activity_search", activity_search_node)
    graph.add_node("restaurant_search", restaurant_search_node)
    graph.add_node("traffic_eta", traffic_eta_node)
    graph.add_node("queue_check", queue_check_node)
    graph.add_node("crowd_risk", crowd_risk_node)
    graph.add_node("fact_gathering", fact_gathering_node)

    graph.add_node("candidate_planning", candidate_planning_node)
    graph.add_node("rule_validation", rule_validation_node)
    graph.add_node("repair_loop", repair_loop_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("final_plan", final_plan_node)
    graph.add_node("schedule_timing", schedule_timing_node)
    graph.add_node("presentation", presentation_node)
    graph.add_node("interaction_wait", interaction_wait_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("execution", execution_node)
    graph.add_node("final_message", final_message_node)
    graph.add_node("reject", reject_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "time_normalize")
    graph.add_conditional_edges(
        "time_normalize",
        route_after_intent_with_clarification,
        {
            "llm_answer": "llm_answer",
            "location_permission": "location_permission",
            "clarification": "clarification",
            "retrieval": "retrieval",
            "constraint_build": "constraint_build",
        },
    )
    graph.add_conditional_edges(
        "location_permission",
        route_after_location_permission,
        {
            "location_lookup": "location_lookup",
            "constraint_build": "constraint_build",
            "interaction_wait": "interaction_wait",
        },
    )
    graph.add_conditional_edges(
        "location_lookup",
        lambda state: "location_fallback" if not state.get("runtime_origin_area") else "constraint_build",
        {
            "location_fallback": "location_fallback",
            "constraint_build": "constraint_build",
        },
    )
    graph.add_conditional_edges(
        "location_fallback",
        route_after_location_fallback,
        {
            "interaction_wait": "interaction_wait",
            "constraint_build": "constraint_build",
        },
    )
    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "intent": "intent",
            "llm_answer": "llm_answer",
            "interaction_wait": "interaction_wait",
        },
    )
    graph.add_edge("llm_answer", END)
    graph.add_edge("retrieval", "constraint_build")
    graph.add_edge("constraint_build", "weather_check")
    graph.add_edge("constraint_build", "restaurant_search")
    graph.add_edge("constraint_build", "traffic_eta")
    graph.add_edge("constraint_build", "queue_check")
    graph.add_edge("constraint_build", "crowd_risk")
    graph.add_edge("weather_check", "activity_search")
    graph.add_edge(
        ["activity_search", "restaurant_search", "traffic_eta", "queue_check", "crowd_risk"],
        "fact_gathering",
    )
    graph.add_edge("fact_gathering", "candidate_planning")
    graph.add_conditional_edges(
        "candidate_planning",
        route_after_candidate_planning,
        {"rule_validation": "rule_validation"},
    )
    graph.add_conditional_edges(
        "rule_validation",
        route_after_rule_validation,
        {
            "scoring": "scoring",
            "repair_loop": "repair_loop",
        },
    )
    graph.add_conditional_edges(
        "repair_loop",
        route_after_repair_loop_new,
        {
            "constraint_build": "constraint_build",
            "final_plan": "final_plan",
        },
    )
    graph.add_conditional_edges(
        "scoring",
        route_after_scoring,
        {"final_plan": "final_plan"},
    )
    graph.add_edge("final_plan", "schedule_timing")
    graph.add_edge("schedule_timing", "presentation")
    graph.add_edge("presentation", "confirmation")
    graph.add_conditional_edges(
        "confirmation",
        route_after_confirmation,
        {
            "execute": "execution",
            "replan": "repair_loop",
            "interaction_wait": "interaction_wait",
        },
    )
    graph.add_edge("interaction_wait", END)
    graph.add_edge("execution", "final_message")
    graph.add_edge("final_message", END)
    graph.add_edge("reject", END)

    return graph.compile()


app = build_workflow()
