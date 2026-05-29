from src.graph.state import AgentState
from src.utils.state_utils import _append_error


def final_plan_node(state: AgentState) -> AgentState:
    """
    Final Plan Node：从评分结果中挑选最高分候选方案。

    时间轴已由 candidate_planning_node 计算完毕（每个 step 含 start_time/end_time），
    本节点不再重算，直接透传。
    """
    print("[Final Plan Node] 精选高评分方案...")
    try:
        scoring_result         = state.get("scoring_result") or {}
        scored_candidates      = scoring_result.get("scored_candidates") or []
        candidate_plans        = state.get("candidate_plans") or {}
        all_candidates: list   = candidate_plans.get("candidates") or []

        # 按 final_score 挑最高分
        best_score_item = None
        for item in scored_candidates:
            if not isinstance(item, dict):
                continue
            score = item.get("final_score")
            if not isinstance(score, (int, float)):
                continue
            if best_score_item is None or score > best_score_item["final_score"]:
                best_score_item = item

        # 从 candidate_plans 里找对应的完整候选（含 timeline/steps）
        best_candidate: dict = {}
        best_candidate_id    = ""
        best_score           = 0

        if best_score_item:
            best_candidate_id = best_score_item.get("candidate_id") or ""
            best_score        = best_score_item.get("final_score") or 0
            for c in all_candidates:
                if isinstance(c, dict) and c.get("id") == best_candidate_id:
                    best_candidate = dict(c)
                    break

        # 没有评分结果时兜底取第一个候选
        if not best_candidate and all_candidates:
            best_candidate    = dict(all_candidates[0])
            best_candidate_id = best_candidate.get("id") or ""

        timeline = best_candidate.get("timeline") or []

        print(f"[Final Plan Node] 选中方案：{best_candidate_id}，得分：{best_score}")
        for item in timeline:
            if isinstance(item, dict):
                print(f"  {item.get('time', '')} - {item.get('item', '')} ({item.get('type', '')})")

        return {
            "final_plan_result": {
                "selected_candidate":    best_candidate,
                "selected_candidate_id": best_candidate_id,
                "final_score":           best_score,
                "all_scored_candidates": scored_candidates,
            },
            "schedule_timing_result": {
                "timeline":              timeline,
                "segment_eta":           {},
                "normalized_date_label": (state.get("plan_context") or {}).get("date_label") or "",
            },
        }

    except Exception as exc:
        print(f"[Final Plan Node][WARN] 异常: {exc}")
        update = _append_error(state, f"Final Plan failed: {exc}")
        update["final_plan_result"] = {
            "selected_candidate": {}, "selected_candidate_id": "",
            "final_score": 0, "all_scored_candidates": [],
        }
        return update