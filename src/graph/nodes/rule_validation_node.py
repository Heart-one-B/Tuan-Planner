# src/graph/nodes/rule_validation_node.py
from src.graph.state import AgentState
from src.utils.state_utils import _append_error, _candidate_activities
from src.utils.poi_utils import _activity_environment
from src.utils.weather_utils import _weather_requires_indoor
from src.utils.parsing_utils import _steps_cover_daypart

def rule_validation_node(state: AgentState) -> AgentState:
    """新架构规则校验节点骨架：逐个检查 candidate_plans，并输出合法/非法结果。"""
    print("[Rule Validation Node] 校验候选计划骨架...")
    try:
        candidate_plans = state.get("candidate_plans")
        if not isinstance(candidate_plans, dict):
            candidate_plans = {}

        constraint_build = state.get("constraint_build")
        if not isinstance(constraint_build, dict):
            constraint_build = {}

        fact_gathering_result = state.get("fact_gathering_result")
        if not isinstance(fact_gathering_result, dict):
            fact_gathering_result = {}

        validation_profile = constraint_build.get("validation_profile")
        if not isinstance(validation_profile, dict):
            validation_profile = {}
        hard_constraints = constraint_build.get("hard_constraints")
        if not isinstance(hard_constraints, dict):
            hard_constraints = {}
        daypart = hard_constraints.get("daypart") if isinstance(hard_constraints.get("daypart"), str) else ""

        candidates = candidate_plans.get("candidates")
        if not isinstance(candidates, list):
            candidates = []

        valid_plans = []
        invalid_plans = []

        weather = fact_gathering_result.get("weather") if isinstance(fact_gathering_result.get("weather"), dict) else state.get("weather") or {}
        weather_risk = weather.get("risk_level") or weather.get("risk") or ""

        for item in candidates:
            if not isinstance(item, dict):
                continue

            candidate_id = item.get("id") if isinstance(item.get("id"), str) else "unknown_plan"
            activity = item.get("activity") if isinstance(item.get("activity"), dict) else {}
            restaurant = item.get("restaurant") if isinstance(item.get("restaurant"), dict) else {}
            candidate_activities = _candidate_activities(item)

            violations = []
            repair_instructions = []
            steps = item.get("steps") if isinstance(item.get("steps"), list) else []

            if not _steps_cover_daypart(steps, daypart):
                violations.append("daypart_coverage_conflict")
                repair_instructions.append(
                    {"type": "regenerate_steps", "constraint": "full_day_requires_morning_lunch_afternoon_dinner"}
                )

            if validation_profile.get("check_weather_compatibility") is True:
                weather_conflict_activities = [
                    activity_item
                    for activity_item in candidate_activities
                    if _weather_requires_indoor(weather_risk)
                    and _activity_environment(activity_item) in {"outdoor", "mixed", "unknown"}
                ]
                if weather_conflict_activities:
                    violations.append("weather_outdoor_conflict")
                    repair_instructions.append(
                        {
                            "type": "replace_activity",
                            "constraint": "indoor_only",
                            "poi_ids": [
                                activity_item.get("id")
                                for activity_item in weather_conflict_activities
                                if isinstance(activity_item.get("id"), str)
                            ],
                            "poi_names": [
                                activity_item.get("name")
                                for activity_item in weather_conflict_activities
                                if isinstance(activity_item.get("name"), str)
                            ],
                        }
                    )

            if validation_profile.get("check_party_fit") is True:
                if not restaurant:
                    violations.append("restaurant_missing")
                    repair_instructions.append(
                        {"type": "replace_restaurant", "constraint": "party_fit_required"}
                    )

            if violations:
                invalid_plans.append(
                    {
                        "candidate_id": candidate_id,
                        "violations": violations,
                        "repair_instructions": repair_instructions,
                    }
                )
            else:
                valid_plans.append(item)

            print(
                f"[Rule Validation Node] candidate_id={candidate_id!r}, "
                f"weather_risk={weather_risk!r}, "
                f"activities={[(activity_item.get('name'), _activity_environment(activity_item)) for activity_item in candidate_activities]!r}, "
                f"violations={violations!r}"
            )

        return {
            "rule_validation_result": {
                "request_type": candidate_plans.get("request_type", "generic_local_plan"),
                "plan_mode": candidate_plans.get("plan_mode", "activity_plus_meal"),
                "valid_plans": valid_plans,
                "invalid_plans": invalid_plans,
            }
        }
    except Exception as exc:
        print(f"[Rule Validation Node][WARN] 节点异常，返回空骨架: {exc}")
        update = _append_error(state, f"Rule Validation node failed: {exc}")
        update["rule_validation_result"] = {
            "request_type": "generic_local_plan",
            "plan_mode": "activity_plus_meal",
            "valid_plans": [],
            "invalid_plans": [],
        }
        return update