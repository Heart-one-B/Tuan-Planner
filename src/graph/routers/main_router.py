from __future__ import annotations

from src.graph.state import AgentState


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent") or {}
    errors = state.get("errors") or []

    # intent 节点失败
    if any(e.get("node") == "intent" for e in errors):
        return "end"

    if not intent.get("is_leisure_planning"):
        return "chat"

    if intent.get("clarification_needed"):
        return "clarification"

    return "constraint"


def route_after_feedback(state: AgentState) -> str:
    route = state.get("feedback_route") or "new_plan"
    if route == "adjust":
        return "orchestrator"
    if route == "chat":
        return "chat"
    return "intent"


def route_after_fact(state: AgentState) -> str:
    agent_outputs = state.get("agent_outputs") or {}
    fact = agent_outputs.get("fact") or {}

    if fact.get("status") == "error":
        return "end"
    return "planning"