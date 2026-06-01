from src.graph.state import AgentState
from src.utils.state_utils import _append_error


def final_plan_node(state: AgentState) -> AgentState:
    """
    Final Plan Node：从评分结果中挑选最高分候选方案。

    数据来源：
      - scoring_result["scored_candidates"]  → 排名和得分
      - rule_validation_result["valid_plans"] → 完整候选数据（含 steps/timeline）
    """
    print("[Final Plan Node] 精选高评分方案...")
    try:
        scoring_result  = state.get("scoring_result") or {}
        scored_candidates = scoring_result.get("scored_candidates") or []

        valid_plans: list[dict] = (
            (state.get("rule_validation_result") or {}).get("valid_plans") or []
        )
        valid_by_id = {
            c["id"]: c for c in valid_plans
            if isinstance(c, dict) and c.get("id")
        }

        # 挑最高分
        best_scored = max(
            (s for s in scored_candidates if isinstance(s, dict)),
            key=lambda s: s.get("final_score") or 0,
            default=None,
        )

        best_candidate: dict = {}
        best_id:    str  = ""
        best_score: int  = 0

        if best_scored:
            best_id    = best_scored.get("candidate_id") or ""
            best_score = best_scored.get("final_score") or 0
            best_candidate = valid_by_id.get(best_id) or {}

        # 兜底：直接取第一个 valid_plan
        if not best_candidate and valid_plans:
            best_candidate = valid_plans[0]
            best_id        = best_candidate.get("id") or ""

        timeline = best_candidate.get("timeline") or []
        date_label = (state.get("plan_context") or {}).get("date_label") or ""

        print(f"[Final Plan Node] 选中：{best_id}，得分：{best_score}")
        for item in timeline:
            if isinstance(item, dict):
                end = item.get("end_time", "")
                end_str = f"～{end}" if end else ""
                print(f"  {item.get('time','')}{end_str} {item.get('item','')} ({item.get('type','')})")

        return {
            "final_plan_result": {
                "selected_candidate":    best_candidate,
                "selected_candidate_id": best_id,
                "final_score":           best_score,
                "date_label":            date_label,
                "all_scored_candidates": scored_candidates,
            },
        }

    except Exception as exc:
        print(f"[Final Plan Node][ERROR] {exc}")
        update = _append_error(state, f"Final Plan failed: {exc}")
        update["final_plan_result"] = {
            "selected_candidate": {}, "selected_candidate_id": "",
            "final_score": 0, "all_scored_candidates": [],
        }
        return update