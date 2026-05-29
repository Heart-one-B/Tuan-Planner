from src.graph.state import AgentState
from src.utils.state_utils import _append_error


def _get_poi_ids(candidate: dict) -> list[str]:
    """从候选方案的 steps 中提取所有 poi_id。"""
    return [
        step.get("poi_id")
        for step in (candidate.get("steps") or [])
        if isinstance(step, dict) and step.get("poi_id")
    ]


def _estimate_route_minutes(candidate: dict, eta: dict) -> dict:
    """
    用 fact_gathering_result["eta"] 估算该方案的总出行时间。

    eta 格式：{poi_id: {eta_minutes: int | None, ...}}
    """
    poi_ids = _get_poi_ids(candidate)
    segments = []
    missing = []

    for poi_id in poi_ids:
        record = eta.get(poi_id)
        if isinstance(record, dict) and record.get("eta_minutes") is not None:
            segments.append({
                "poi_id":      poi_id,
                "eta_minutes": record["eta_minutes"],
            })
        else:
            missing.append(poi_id)

    total = sum(s["eta_minutes"] for s in segments) if segments else None
    return {
        "total_route_minutes":  total,
        "route_status":         "partial" if missing else ("ok" if segments else "unknown"),
        "segments":             segments,
        "missing_segments":     missing,
    }


def scoring_node(state: AgentState) -> AgentState:
    """Scoring Node：对合法候选方案按总出行时间打分排序。"""
    print("[Scoring Node] 评估最终全流程通勤耗时并对方案排序...")
    try:
        rule_validation_result = state.get("rule_validation_result") or {}
        eta: dict = (state.get("fact_gathering_result") or {}).get("eta") or {}
        valid_plans: list[dict] = rule_validation_result.get("valid_plans") or []

        route_candidates = []
        for idx, candidate in enumerate(valid_plans):
            if not isinstance(candidate, dict):
                continue
            estimation = _estimate_route_minutes(candidate, eta)
            total = estimation.get("total_route_minutes")
            has_total = isinstance(total, (int, float)) and not isinstance(total, bool)
            route_candidates.append({
                "candidate":         candidate,
                "candidate_id":      candidate.get("id") or "unknown",
                "estimation":        estimation,
                "total":             total,
                "has_total":         has_total,
                "original_index":    idx,
            })

        # 有出行时间的排前面，时间越短分越高
        route_candidates.sort(key=lambda x: (
            0 if x["has_total"] else 1,
            x["total"] if x["has_total"] else float("inf"),
            x["original_index"],
        ))

        scored_candidates = []
        for rank, item in enumerate(route_candidates, start=1):
            est = item["estimation"]
            final_score = max(0, 100 - (rank - 1) * 20) if item["has_total"] else 0
            scored_candidates.append({
                "candidate_id": item["candidate_id"],
                "final_score":  final_score,
                "score_breakdown": {
                    "route_rank":            rank,
                    "total_route_minutes":   item["total"],
                    "route_status":          est.get("route_status"),
                    "known_segment_count":   len(est.get("segments") or []),
                    "missing_segment_count": len(est.get("missing_segments") or []),
                },
                "route_estimation": est,
            })

        return {
            "scoring_result": {
                "request_type":    rule_validation_result.get("request_type", "generic_local_plan"),
                "plan_mode":       rule_validation_result.get("plan_mode", "activity_plus_meal"),
                "scoring_method":  "shortest_total_route_minutes",
                "scored_candidates": scored_candidates,
            }
        }

    except Exception as exc:
        print(f"[Scoring Node][WARN] 打分失败: {exc}")
        update = _append_error(state, f"Scoring node failed: {exc}")
        update["scoring_result"] = {
            "request_type": "generic_local_plan",
            "plan_mode":    "activity_plus_meal",
            "scored_candidates": [],
        }
        return update