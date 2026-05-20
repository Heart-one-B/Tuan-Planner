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
    presentation_node,
    queue_check_node,
    reject_node,
    repair_loop_node,
    retrieval_node,
    rule_validation_node,
    route_after_confirmation,
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
    if intent.get("clarification_needed") is True:
        return "clarification"
    if intent.get("is_leisure_planning") is True and intent.get("need_retrieval") is True:
        return "retrieval"
    return "constraint_build"


def route_after_clarification(state: AgentState) -> str:
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
    if isinstance(result, dict) and isinstance(result.get("valid_plans"), list) and result.get("valid_plans"):
        return "scoring"
    return "repair_loop"


def route_after_repair_loop_new(state: AgentState) -> str:
    return "constraint_build"


def route_after_scoring(state: AgentState) -> str:
    return "final_plan"


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("llm_answer", llm_answer_node)
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
    graph.add_node("presentation", presentation_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("execution", execution_node)
    graph.add_node("final_message", final_message_node)
    graph.add_node("reject", reject_node)

    graph.set_entry_point("intent")
    graph.add_conditional_edges(
        "intent",
        route_after_intent_with_clarification,
        {
            "llm_answer": "llm_answer",
            "clarification": "clarification",
            "retrieval": "retrieval",
            "constraint_build": "constraint_build",
        },
    )
    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "intent": "intent",
            "llm_answer": "llm_answer",
        },
    )
    graph.add_edge("llm_answer", END)
    graph.add_edge("retrieval", "constraint_build")
    graph.add_edge("constraint_build", "weather_check")
    graph.add_edge("constraint_build", "activity_search")
    graph.add_edge("constraint_build", "restaurant_search")
    graph.add_edge("constraint_build", "traffic_eta")
    graph.add_edge("constraint_build", "queue_check")
    graph.add_edge("constraint_build", "crowd_risk")
    graph.add_edge("weather_check", "fact_gathering")
    graph.add_edge("activity_search", "fact_gathering")
    graph.add_edge("restaurant_search", "fact_gathering")
    graph.add_edge("traffic_eta", "fact_gathering")
    graph.add_edge("queue_check", "fact_gathering")
    graph.add_edge("crowd_risk", "fact_gathering")
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
        {"constraint_build": "constraint_build"},
    )
    graph.add_conditional_edges(
        "scoring",
        route_after_scoring,
        {"final_plan": "final_plan"},
    )
    graph.add_edge("final_plan", "presentation")
    graph.add_edge("presentation", "confirmation")
    graph.add_conditional_edges(
        "confirmation",
        route_after_confirmation,
        {
            "execute": "execution",
            "replan": "repair_loop",
        },
    )
    graph.add_edge("execution", "final_message")
    graph.add_edge("final_message", END)
    graph.add_edge("reject", END)

    return graph.compile()


app = build_workflow()
