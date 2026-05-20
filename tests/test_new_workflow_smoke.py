from __future__ import annotations

from src.graph.workflow import build_workflow, route_after_clarification, route_after_intent_with_clarification


def test_new_workflow_compiles_and_has_core_nodes():
    app = build_workflow()
    graph = app.get_graph()
    node_names = set(getattr(graph, "nodes", {}).keys()) if hasattr(graph, "nodes") else set()

    assert "intent" in node_names
    assert "constraint_build" in node_names
    assert "fact_gathering" in node_names
    assert "candidate_planning" in node_names
    assert "rule_validation" in node_names
    assert "repair_loop" in node_names
    assert "scoring" in node_names
    assert "final_plan" in node_names
    assert "presentation" in node_names


def test_new_route_from_intent_goes_to_new_architecture():
    assert route_after_intent_with_clarification(
        {"intent": {"is_leisure_planning": True, "clarification_needed": False, "need_retrieval": False}}
    ) == "constraint_build"
    assert route_after_intent_with_clarification(
        {"intent": {"is_leisure_planning": True, "clarification_needed": True, "need_retrieval": False}}
    ) == "clarification"
    assert route_after_intent_with_clarification(
        {"intent": {"is_leisure_planning": False, "clarification_needed": False, "need_retrieval": False}}
    ) == "llm_answer"


def test_clarification_returns_to_intent_until_limit():
    assert route_after_clarification({"clarification_round": 0}) == "intent"
    assert route_after_clarification({"clarification_round": 4}) == "intent"
    assert route_after_clarification({"clarification_round": 5}) == "llm_answer"
