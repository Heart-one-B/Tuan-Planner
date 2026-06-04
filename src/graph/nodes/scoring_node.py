from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import AgentState
from src.model.factory import get_chat_model
from src.utils.state_utils import _append_error


def _first_activity_eta(candidate: dict, eta: dict) -> int | None:
    for step in candidate.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("poi_type") == "activity":
            record = eta.get(step.get("poi_id") or "")
            if isinstance(record, dict):
                return record.get("eta_minutes")
    return None


def _total_eta(candidate: dict, eta: dict) -> int | None:
    total = 0
    has_any = False
    for step in candidate.get("steps") or []:
        if not isinstance(step, dict):
            continue
        record = eta.get(step.get("poi_id") or "")
        if isinstance(record, dict) and record.get("eta_minutes") is not None:
            total += record["eta_minutes"]
            has_any = True
    return total if has_any else None


def _candidate_summary(candidate: dict, plan_poi_details: dict | None = None) -> dict:
    plan_poi_details = plan_poi_details or {}
    steps = []
    for step in candidate.get("steps") or []:
        if not isinstance(step, dict):
            continue
        poi_id = step.get("poi_id") or ""
        detail = plan_poi_details.get(poi_id) if isinstance(plan_poi_details, dict) else None
        if not isinstance(detail, dict):
            detail = {}
        steps.append({
            "label": step.get("label") or step.get("item") or step.get("poi_name") or detail.get("name") or "",
            "poi_id": poi_id,
            "poi_type": step.get("poi_type") or step.get("type") or "",
            "phase": step.get("phase") or "",
            "start_time": step.get("start_time") or "",
            "end_time": step.get("end_time") or "",
            "poi_name": detail.get("name") or "",
            "category": detail.get("category") or "",
            "business_area": detail.get("business_area") or "",
            "rating": detail.get("rating") or "",
            "distance": detail.get("distance") or "",
            "avg_price": detail.get("avg_price") or "",
            "tags": (detail.get("tags") or [])[:5] if isinstance(detail.get("tags"), list) else [],
            "highlights": (detail.get("highlights") or [])[:3] if isinstance(detail.get("highlights"), list) else [],
        })

    return {
        "id": candidate.get("id") or "unknown",
        "title": candidate.get("title") or candidate.get("name") or "",
        "summary": candidate.get("summary") or candidate.get("description") or "",
        "steps": steps[:10],
        "timeline": (candidate.get("timeline") or [])[:10],
    }


def _extract_json_block(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group() if match else text
    return json.loads(raw)


def _llm_relevance_scores(
    user_input: str,
    scored: list[dict],
    plan_poi_details: dict | None = None,
) -> dict[str, int]:
    if not user_input.strip() or not scored:
        return {}

    model = get_chat_model()
    candidate_payload = [
        _candidate_summary(item["candidate"], plan_poi_details)
        for item in scored
    ]

    system_prompt = (
        "你是一个本地生活方案打分器。你的任务是根据用户当前需求，"
        "评估候选方案与本轮需求的匹配程度。可以参考 POI 类别、商圈、评分、距离、价格、标签和推荐理由。"
        "不要参考用户历史偏好，也不要因为符合历史偏好而加分。"
        "分数范围必须是 0 到 100，分数越高代表越符合用户当前需求。"
        "你必须只输出严格 JSON，不要输出任何额外解释。"
        "输出格式如下："
        "{\"scores\":[{\"candidate_id\":\"plan_1\",\"score\":90,\"reason\":\"...\"}]}"
    )
    user_prompt = (
        f"用户当前需求：{user_input}\n\n"
        f"候选方案数据：{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
        "请对每个 candidate_id 生成一个 score。"
        "必须覆盖全部候选方案，scores 的数量要与候选方案数量一致。"
    )

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    content = getattr(response, "content", "") or ""
    parsed = _extract_json_block(content)

    scores = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(scores, list):
        raise ValueError("LLM score payload missing scores list")

    score_map: dict[str, int] = {}
    for item in scores:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        try:
            score_int = int(round(float(item.get("score"))))
        except (TypeError, ValueError):
            continue
        score_map[candidate_id] = max(0, min(100, score_int))

    expected_ids = {
        str(item["candidate_id"])
        for item in scored
        if item.get("candidate_id")
    }
    if expected_ids and not expected_ids.issubset(score_map.keys()):
        missing = sorted(expected_ids - score_map.keys())
        raise ValueError(f"LLM score missing candidate ids: {missing}")

    return score_map


def _eta_score_by_rank(rank: int, has_eta: bool) -> int:
    return max(0, 100 - (rank - 1) * 20) if has_eta else 0


def scoring_node(state: AgentState) -> AgentState:
    print("[Scoring Node] 评估需求匹配分与通勤耗时并排序...")
    try:
        rule_validation_result = state.get("rule_validation_result") or {}
        eta: dict = (state.get("fact_gathering_result") or {}).get("eta") or {}
        plan_poi_details: dict = state.get("plan_poi_details") or {}
        valid_plans: list[dict] = rule_validation_result.get("valid_plans") or []
        user_input = (state.get("user_input") or "").strip()
        if not user_input:
            intent = state.get("intent") or {}
            if isinstance(intent, dict):
                user_input = (intent.get("raw_query") or "").strip()

        scored: list[dict] = []
        for idx, candidate in enumerate(valid_plans):
            if not isinstance(candidate, dict):
                continue
            first_eta = _first_activity_eta(candidate, eta)
            total_eta = _total_eta(candidate, eta)
            scored.append({
                "candidate": candidate,
                "candidate_id": candidate.get("id") or "unknown",
                "first_eta": first_eta,
                "total_eta": total_eta,
                "original_index": idx,
            })

        inf = float("inf")
        scored.sort(key=lambda item: (
            item["first_eta"] if item["first_eta"] is not None else inf,
            item["total_eta"] if item["total_eta"] is not None else inf,
            item["original_index"],
        ))

        try:
            relevance_scores = _llm_relevance_scores(user_input, scored, plan_poi_details)
        except Exception as llm_exc:
            print(f"[Scoring Node][WARN] LLM 需求评分失败，回退到 ETA 评分：{llm_exc}")
            relevance_scores = {}

        scored_candidates = []
        for rank, item in enumerate(scored, start=1):
            has_eta = item["first_eta"] is not None
            eta_score = _eta_score_by_rank(rank, has_eta)
            llm_score = relevance_scores.get(item["candidate_id"])
            final_score = eta_score if llm_score is None else int(round(eta_score * 0.4 + llm_score * 0.6))
            scored_candidates.append({
                "candidate_id": item["candidate_id"],
                "final_score": final_score,
                "score_breakdown": {
                    "route_rank": rank,
                    "first_eta_minutes": item["first_eta"],
                    "total_eta_minutes": item["total_eta"],
                },
            })
            print(
                f"[Scoring Node] rank={rank} {item['candidate_id']} "
                f"first_eta={item['first_eta']} total_eta={item['total_eta']} "
                f"eta_score={eta_score} llm_score={llm_score if llm_score is not None else 'fallback'} "
                f"final_score={final_score}"
            )

        scored_candidates.sort(
            key=lambda item: (
                -(item.get("final_score") or 0),
                item.get("score_breakdown", {}).get("route_rank") or 0,
                item.get("candidate_id") or "",
            )
        )

        for idx, item in enumerate(scored_candidates, start=1):
            item["score_breakdown"]["route_rank"] = idx

        return {
            "scoring_result": {
                "plan_mode": rule_validation_result.get("plan_mode", "activity_plus_meal"),
                "scoring_method": "poi_detail_relevance_plus_first_activity_eta",
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
