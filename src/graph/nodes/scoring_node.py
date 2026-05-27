# src/graph/nodes/scoring_node.py
from src.graph.state import AgentState
from src.tools.mock_api import MockToolAPI
from src.utils.state_utils import _append_error
from src.utils.route_utils import _estimate_candidate_route_minutes


def scoring_node(state: AgentState) -> AgentState:
    """Scoring Node: 对合法候选方案进行出行时长最优性打分排序。"""
    print("[Scoring Node] 评估最终全流程通勤耗时并对方案排序...")
    try:
        rule_validation_result = state.get("rule_validation_result") or {}
        fact_gathering_result = state.get("fact_gathering_result") or {}
        traffic = fact_gathering_result.get("traffic") or {}
        valid_plans = rule_validation_result.get("valid_plans") or []

        origin_coordinates = state.get("runtime_origin_coordinates")
        if not origin_coordinates or not origin_coordinates.strip():
            origin_coordinates = traffic.get("origin_coordinates_used") or traffic.get("traffic_origin_used") or ""

        api = MockToolAPI()
        route_candidates = []

        for idx, item in enumerate(valid_plans):
            if not isinstance(item, dict):
                continue
            candidate_id = item.get("id") or "unknown_plan"
            route_estimation = _estimate_candidate_route_minutes(candidate=item, origin_coordinates=origin_coordinates,
                                                                 api=api)
            total_route_minutes = route_estimation.get("total_route_minutes")
            has_route_minutes = isinstance(total_route_minutes, (int, float)) and not isinstance(total_route_minutes,
                                                                                                 bool)

            route_candidates.append({
                "candidate": item, "candidate_id": candidate_id, "route_estimation": route_estimation,
                "total_route_minutes": total_route_minutes, "has_route_minutes": has_route_minutes,
                "original_index": idx,
            })

        # 排序：有估算用时的排在前面，且用时越少分越高
        route_candidates.sort(key=lambda x: (
        0 if x["has_route_minutes"] else 1, x["total_route_minutes"] if x["has_route_minutes"] else float("inf"),
        x["original_index"]))

        scored_candidates = []
        for rank, item in enumerate(route_candidates, start=1):
            route_estimation = item["route_estimation"]
            final_score = max(0, 100 - (rank - 1) * 20) if item["has_route_minutes"] else 0
            scored_candidates.append({
                "candidate_id": item["candidate_id"],
                "score_breakdown": {
                    "route_rank": rank, "total_route_minutes": item["total_route_minutes"],
                    "route_status": route_estimation.get("route_status"),
                    "known_segment_count": len(route_estimation.get("segments") or []),
                    "missing_segment_count": len(route_estimation.get("missing_segments") or []),
                },
                "route_estimation": route_estimation, "final_score": final_score,
            })

        return {
            "scoring_result": {
                "request_type": rule_validation_result.get("request_type", "generic_local_plan"),
                "plan_mode": rule_validation_result.get("plan_mode", "activity_plus_meal"),
                "scoring_method": "shortest_total_route_minutes", "weights": {"total_route_minutes": 1.0},
                "scored_candidates": scored_candidates,
            }
        }
    except Exception as exc:
        print(f"[Scoring Node][WARN] 打分过程失败: {exc}")
        update = _append_error(state, f"Scoring node failed: {exc}")
        update["scoring_result"] = {"request_type": "generic_local_plan", "plan_mode": "activity_plus_meal",
                                    "weights": {}, "scored_candidates": []}
        return update