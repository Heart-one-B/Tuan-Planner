# src/graph/nodes/presentation_node.py
from typing import Any

from src.graph.state import AgentState
from src.agent.presentation_agent import PresentationAgent
from src.tools.cached_amap_client import CachedAmapClient
from src.utils.state_utils import  _append_error


def _location_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _origin_coordinates(state: AgentState) -> str:
    direct = _location_text(state.get("runtime_origin_coordinates"))
    if direct:
        return direct
    lookup = state.get("location_lookup_result")
    if isinstance(lookup, dict):
        return _location_text(lookup.get("coordinates") or lookup.get("location"))
    return ""


def _format_distance_text(result: dict | None, prefix: str = "") -> str:
    if not isinstance(result, dict):
        return ""
    raw_meters = result.get("distance_meters")
    try:
        meters = int(float(raw_meters))
    except (TypeError, ValueError):
        return ""

    if meters >= 1000:
        km = meters / 1000
        distance = f"约{km:.1f}km" if km < 10 else f"约{round(km)}km"
    else:
        distance = f"约{meters}m"
    if prefix:
        distance = f"{prefix}{distance}"

    eta = result.get("eta_minutes")
    try:
        eta_minutes = int(round(float(eta)))
    except (TypeError, ValueError):
        eta_minutes = 0
    if eta_minutes > 0:
        return f"{distance} / 约{eta_minutes}分钟车程"
    return distance


def enrich_selected_plan_distances(plan: dict, state: AgentState, api: CachedAmapClient | None = None) -> dict:
    details = plan.get("plan_poi_details")
    steps = [step for step in plan.get("steps") or [] if isinstance(step, dict)]
    if not isinstance(details, dict) or not steps:
        return plan

    enriched_details = {
        poi_id: dict(detail) if isinstance(detail, dict) else detail
        for poi_id, detail in details.items()
    }
    client = api or CachedAmapClient()
    previous_location = _origin_coordinates(state)

    for idx, step in enumerate(steps):
        poi_id = step.get("poi_id") or ""
        detail = enriched_details.get(poi_id)
        if not poi_id or not isinstance(detail, dict):
            continue
        destination = _location_text(detail.get("location"))
        if not destination:
            continue

        distance_text = ""
        if previous_location:
            try:
                prefix = "距出发地" if idx == 0 else "距上一站"
                distance_text = _format_distance_text(client.distance(previous_location, destination), prefix)
            except Exception:
                distance_text = ""
        elif idx == 0:
            distance_text = "行程起点"

        if distance_text:
            detail["distance"] = distance_text
        previous_location = destination

    plan["plan_poi_details"] = enriched_details
    return plan


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
            plan["plan_poi_details"] = state.get("plan_poi_details") or {}
            plan = enrich_selected_plan_distances(plan, state)
        else:
            return state

        display_text = PresentationAgent().generate_plan_display(plan, state.get("intent", {}))
        print("\n" + "=" * 20 + " 方案详情 " + "=" * 20)
        print(display_text)
        print("=" * 50)
        return {"display_text": display_text, "plan": plan}
    except Exception as exc:
        return _append_error(state, f"Presentation node failed: {exc}")
