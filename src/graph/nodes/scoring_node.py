from __future__ import annotations

from src.graph.state import AgentState
from src.utils.state_utils import _append_error


def _first_activity_eta(candidate: dict, eta: dict) -> int | None:
    """返回出发地到第一个活动 POI 的 ETA（分钟），拿不到返回 None。"""
    for step in (candidate.get("steps") or []):
        if not isinstance(step, dict):
            continue
        if step.get("poi_type") == "activity":
            record = eta.get(step.get("poi_id") or "")
            if isinstance(record, dict):
                return record.get("eta_minutes")
    return None


def _total_eta(candidate: dict, eta: dict) -> int | None:
    """所有 POI ETA 之和，作为二级排序的粗略代理。"""
    total = 0
    has_any = False
    for step in (candidate.get("steps") or []):
        if not isinstance(step, dict):
            continue
        record = eta.get(step.get("poi_id") or "")
        if isinstance(record, dict) and record.get("eta_minutes") is not None:
            total += record["eta_minutes"]
            has_any = True
    return total if has_any else None


def scoring_node(state: AgentState) -> AgentState:
    """
    Scoring Node：按通勤时间对合法候选方案排序。

    主指标：出发地 → 第一个活动的 ETA（最影响体验的一段）
    次指标：所有 POI ETA 之和（粗略代理整体通勤量）
    """
    print("[Scoring Node] 评估通勤耗时并排序...")
    try:
        rule_validation_result = state.get("rule_validation_result") or {}
        eta: dict              = (state.get("fact_gathering_result") or {}).get("eta") or {}
        valid_plans: list[dict] = rule_validation_result.get("valid_plans") or []

        scored: list[dict] = []
        for idx, candidate in enumerate(valid_plans):
            if not isinstance(candidate, dict):
                continue
            first_eta = _first_activity_eta(candidate, eta)
            total_eta = _total_eta(candidate, eta)
            scored.append({
                "candidate":      candidate,
                "candidate_id":   candidate.get("id") or "unknown",
                "first_eta":      first_eta,
                "total_eta":      total_eta,
                "original_index": idx,
            })

        INF = float("inf")
        scored.sort(key=lambda x: (
            x["first_eta"] if x["first_eta"] is not None else INF,  # 主：第一段越短越好
            x["total_eta"] if x["total_eta"] is not None else INF,  # 次：总 ETA 越短越好
            x["original_index"],                                     # 三级：保持原顺序稳定
        ))

        scored_candidates = []
        for rank, item in enumerate(scored, start=1):
            has_eta = item["first_eta"] is not None
            final_score = max(0, 100 - (rank - 1) * 20) if has_eta else 0
            scored_candidates.append({
                "candidate_id": item["candidate_id"],
                "final_score":  final_score,
                "score_breakdown": {
                    "route_rank":       rank,
                    "first_eta_minutes": item["first_eta"],
                    "total_eta_minutes": item["total_eta"],
                },
            })
            print(
                f"[Scoring Node] rank={rank} {item['candidate_id']} "
                f"first_eta={item['first_eta']} total_eta={item['total_eta']} score={final_score}"
            )

        return {
            "scoring_result": {
                "plan_mode":         rule_validation_result.get("plan_mode", "activity_plus_meal"),
                "scoring_method":    "first_activity_eta",
                "scored_candidates": scored_candidates,
            }
        }

    except Exception as exc:
        print(f"[Scoring Node][WARN] 打分失败: {exc}")
        update = _append_error(state, f"Scoring node failed: {exc}")
        update["scoring_result"] = {
            "plan_mode": "activity_plus_meal",
            "scored_candidates": [],
        }
        return update