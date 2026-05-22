from __future__ import annotations

import builtins

from src.agent.execution_agent import ExecutionAgent
from src.agent.intent_agent import IntentAgent
from src.graph import workflow as wf


def test_new_workflow_compiles_and_has_core_nodes():
    app = wf.build_workflow()
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
    assert wf.route_after_intent_with_clarification(
        {"intent": {"is_leisure_planning": True, "clarification_needed": False, "need_retrieval": False}}
    ) == "location_permission"
    assert wf.route_after_intent_with_clarification(
        {
            "intent": {
                "is_leisure_planning": True,
                "clarification_needed": False,
                "need_retrieval": False,
                "location": {"origin_area_hint": "望京"},
            }
        }
    ) == "constraint_build"
    assert wf.route_after_intent_with_clarification(
        {
            "intent": {
                "is_leisure_planning": True,
                "clarification_needed": True,
                "need_retrieval": False,
                "missing_slots": {"global": ["scenario"]},
            }
        }
    ) == "clarification"
    assert wf.route_after_intent_with_clarification(
        {"intent": {"is_leisure_planning": False, "clarification_needed": False, "need_retrieval": False}}
    ) == "llm_answer"


def test_clarification_returns_to_intent_until_limit():
    assert wf.route_after_clarification({"clarification_round": 0}) == "intent"
    assert wf.route_after_clarification({"clarification_round": 4}) == "intent"
    assert wf.route_after_clarification({"clarification_round": 5}) == "llm_answer"


def test_smoke_new_workflow_does_not_execute_before_confirmation():
    calls: list[str] = []
    parse_calls = {"n": 0}

    original_parse = IntentAgent.parse
    original_execute = ExecutionAgent.execute
    original_input = builtins.input
    original_candidate = wf.candidate_planning_node
    original_rule_validation = wf.rule_validation_node
    original_scoring = wf.scoring_node
    original_final_plan = wf.final_plan_node
    original_presentation = wf.presentation_node

    def fake_parse(self, user_input, runtime_origin_area=""):
        parse_calls["n"] += 1
        if parse_calls["n"] == 1:
            return {
                "is_leisure_planning": True,
                "need_retrieval": False,
                "clarification_needed": True,
                "missing_slots": {"time": ["missing"]},
                "follow_up_message": "请补充时间",
                "scenario": "family",
                "conversation_turns": [],
                "raw_query": user_input,
            }
        return {
            "is_leisure_planning": True,
            "need_retrieval": False,
            "clarification_needed": False,
            "missing_slots": {},
            "follow_up_message": "",
            "scenario": "family",
            "conversation_turns": [],
            "raw_query": user_input,
            "time_window": "today_afternoon",
            "date_label": "今天",
            "daypart": "下午",
            "origin_area": "area_central",
            "time": {"date_label": "今天", "daypart": "下午", "time_phrase": "今天下午"},
            "participants": {"people_count": 3, "has_child": True, "child_age": 5},
            "diet_preference": ["减脂"],
            "location": {"origin_area_hint": "area_central", "location_text": "area_central"},
        }

    def fake_execute(self, plan):
        calls.append("execute")
        return {"status": "success", "message": "ok"}

    def fake_candidate_planning(state):
        return {
            "candidate_plans": {
                "request_type": "family_with_kids",
                "plan_mode": "activity_plus_meal",
                "candidates": [
                    {
                        "id": "plan_1",
                        "title": "室内亲子 + 轻食",
                        "timeline": ["下午活动", "晚餐"],
                        "activity": {"id": "A1", "name": "亲子活动", "type": "indoor"},
                        "restaurant": {"id": "R1", "name": "亲子餐厅"},
                        "reasoning": ["符合亲子需求"],
                    },
                    {
                        "id": "plan_2",
                        "title": "备用计划",
                        "timeline": [],
                        "activity": {},
                        "restaurant": {},
                        "reasoning": [],
                    },
                    {
                        "id": "plan_3",
                        "title": "备用计划2",
                        "timeline": [],
                        "activity": {},
                        "restaurant": {},
                        "reasoning": [],
                    },
                ],
            }
        }

    def fake_rule_validation(state):
        candidates = state.get("candidate_plans", {}).get("candidates", []) if isinstance(state.get("candidate_plans"), dict) else []
        valid = [candidates[0]] if candidates else []
        return {
            "rule_validation_result": {
                "request_type": "family_with_kids",
                "plan_mode": "activity_plus_meal",
                "valid_plans": valid,
                "invalid_plans": [],
            }
        }

    def fake_scoring(state):
        valid_plans = state.get("rule_validation_result", {}).get("valid_plans", []) if isinstance(state.get("rule_validation_result"), dict) else []
        scored = []
        for item in valid_plans:
            scored.append({
                "candidate_id": item.get("id", "plan_1"),
                "score_breakdown": {"semantic_match": 8.5, "time_relaxation": 7.0, "weather_fit": 9.0, "distance_fit": 8.0, "queue_fit": 6.5, "review_quality": 8.0},
                "final_score": 82.5,
            })
        return {
            "scoring_result": {
                "request_type": "family_with_kids",
                "plan_mode": "activity_plus_meal",
                "weights": {"semantic_match": 0.30, "time_relaxation": 0.20, "weather_fit": 0.15, "distance_fit": 0.15, "queue_fit": 0.10, "review_quality": 0.10},
                "scored_candidates": scored,
            }
        }

    def fake_final_plan(state):
        scored = state.get("scoring_result", {}).get("scored_candidates", []) if isinstance(state.get("scoring_result"), dict) else []
        best = scored[0] if scored else {}
        return {
            "final_plan_result": {
                "selected_candidate_id": best.get("candidate_id", "plan_1"),
                "selected_candidate": best,
                "final_score": best.get("final_score", 0),
                "all_scored_candidates": scored,
            }
        }

    def fake_presentation(state):
        return {
            "display_text": "方案展示",
            "plan": state.get("final_plan_result", {}).get("selected_candidate", {}),
        }

    inputs = iter(["补充时间：今天下午", "y"])

    try:
        IntentAgent.parse = fake_parse
        ExecutionAgent.execute = fake_execute
        wf.candidate_planning_node = fake_candidate_planning
        wf.rule_validation_node = fake_rule_validation
        wf.scoring_node = fake_scoring
        wf.final_plan_node = fake_final_plan
        wf.presentation_node = fake_presentation
        builtins.input = lambda prompt="": next(inputs)

        local_app = wf.build_workflow()
        state = local_app.invoke(
            {
                "user_input": "想出去玩",
                "runtime_origin_area": "area_central",
                "conversation_turns": ["想出去玩"],
                "clarification_round": 0,
                "errors": [],
            }
        )
    finally:
        IntentAgent.parse = original_parse
        ExecutionAgent.execute = original_execute
        wf.candidate_planning_node = original_candidate
        wf.rule_validation_node = original_rule_validation
        wf.scoring_node = original_scoring
        wf.final_plan_node = original_final_plan
        wf.presentation_node = original_presentation
        builtins.input = original_input

    assert state["clarification_round"] == 1
    assert state["user_confirmed"] is True
    assert calls == ["execute"]
