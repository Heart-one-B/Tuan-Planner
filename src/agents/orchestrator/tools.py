from __future__ import annotations
import json


def build_orchestrator_context(
    user_feedback: str,
    plan_context: dict,
    fact_data: dict,
    selected_plan: dict,
) -> str:
    """把当前状态压缩成 Orchestrator 需要的最小上下文。"""

    # 当前方案 timeline 摘要
    timeline = selected_plan.get("timeline") or []
    current_plan_lines = [
        f"{t.get('time')}-{t.get('end_time')} "
        f"「{t.get('label')}」({t.get('ref_type')}) "
        f"id={t.get('ref_id')}"
        for t in timeline
    ]

    # POI 池压缩
    def slim_poi(p: dict) -> dict:
        return {
            "id":          p.get("id", ""),
            "name":        p.get("name", ""),
            "type":        p.get("type", ""),
            "rating":      p.get("rating", ""),
            "eta_minutes": p.get("eta_minutes"),
        }

    activities  = [slim_poi(a) for a in (fact_data.get("activities")  or []) if isinstance(a, dict)]
    restaurants = [slim_poi(r) for r in (fact_data.get("restaurants") or []) if isinstance(r, dict)]
    waypoints   = [slim_poi(w) for w in (fact_data.get("waypoints")   or []) if isinstance(w, dict)]

    # plan_context 精简
    prefs = plan_context.get("preferences") or {}
    constraints = {
        "scenario":            plan_context.get("scenario"),
        "origin_area":         plan_context.get("origin_area"),
        "start_time":          plan_context.get("start_time"),
        "end_time":            plan_context.get("end_time"),
        "people_count":        plan_context.get("people_count"),
        "raw_query":           plan_context.get("raw_query"),
        "activity_keywords":   plan_context.get("activity_keywords") or [],
        "restaurant_keywords": plan_context.get("restaurant_keywords") or [],
        "avoid":               prefs.get("avoid") or [],
    }

    return f"""\
## 用户反馈
「{user_feedback}」

## 当前方案时间轴
{chr(10).join(current_plan_lines) or "暂无方案"}

## 现有活动候选池（含通勤分钟数）
{json.dumps(activities, ensure_ascii=False, indent=2)}

## 现有餐厅候选池
{json.dumps(restaurants, ensure_ascii=False, indent=2)}

## 途径小需求候选
{json.dumps(waypoints, ensure_ascii=False, indent=2) if waypoints else "无"}

## 规划约束
{json.dumps(constraints, ensure_ascii=False, indent=2)}
"""