# src/graph/nodes/presentation_node.py
from src.graph.state import AgentState
from src.agent.presentation_agent import PresentationAgent
from src.utils.state_utils import _derive_plan_compat_from_state, _append_error

def presentation_node(state: AgentState) -> AgentState:
    """Presentation Node：将选定的最优计划及其排期信息渲染为富文本供用户预览。"""
    print("[Presentation Node] 开始生成富文本最终预览排版...")
    try:
        final_plan_result = state.get("final_plan_result")
        if isinstance(final_plan_result, dict) and final_plan_result:
            selected_candidate = final_plan_result.get("selected_candidate") or {}
            plan = dict(selected_candidate)
            activities = plan.get("activities")
            if isinstance(activities, list):
                plan["activities"] = [item for item in activities if isinstance(item, dict)]
            else:
                activity = plan.get("activity")
                plan["activities"] = [activity] if isinstance(activity, dict) else []
            if not isinstance(plan.get("restaurant"), dict):
                plan["restaurant"] = {}
            schedule_timing_result = state.get("schedule_timing_result")
            if isinstance(schedule_timing_result, dict):
                plan["schedule_timing_result"] = schedule_timing_result
            plan["selected_candidate_id"] = final_plan_result.get("selected_candidate_id", "")
            plan["final_score"] = final_plan_result.get("final_score", 0)
            plan["all_scored_candidates"] = final_plan_result.get("all_scored_candidates", [])
        else:
            plan = _derive_plan_compat_from_state(state)

        display_text = PresentationAgent().generate_plan_display(plan, state.get("intent", {}))
        print("\n" + "=" * 20 + " 方案详情 " + "=" * 20)
        print(display_text)
        print("=" * 50)
        return {"display_text": display_text, "plan": plan}
    except Exception as exc:
        return _append_error(state, f"Presentation node failed: {exc}")