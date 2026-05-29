from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import AgentState
from src.model.factory import get_chat_model
from src.utils.state_utils import _append_error

_SYSTEM_PROMPT = """\
你是本地生活行程规划助手，负责根据用户需求和候选 POI 池生成 3 个候选行程方案。

## 输出要求
只输出 JSON，不输出任何其他内容。格式如下：
{
  "candidates": [
    {
      "id": "plan_1",
      "title": "方案标题",
      "steps": [
        {"step_id": "step_1", "phase": "morning",   "poi_type": "activity",    "poi_id": "...", "label": "上午活动", "duration_minutes": 90},
        {"step_id": "step_2", "phase": "lunch",     "poi_type": "restaurant",  "poi_id": "...", "label": "午餐",    "duration_minutes": 75},
        {"step_id": "step_3", "phase": "afternoon", "poi_type": "activity",    "poi_id": "...", "label": "下午活动", "duration_minutes": 90},
        {"step_id": "step_4", "phase": "dinner",    "poi_type": "restaurant",  "poi_id": "...", "label": "晚餐",    "duration_minutes": 75}
      ],
      "reasoning": ["理由1", "理由2"]
    }
  ]
}

## 选址规则
- 只能使用候选池中提供的 poi_id，禁止编造
- 3 个方案的活动组合要尽量不同，不要三个方案都选同一个活动
- 相似活动类型视为同一类（剧本杀/密室逃脱/桌游属于同一类），同一方案内活动不重复

## 时间段规则（phase）
根据 start_time 和 end_time 判断需要安排哪些时间段：
- 包含上午（start_time <= 11:00）→ 需要 morning 活动
- 跨越午饭时间（start_time < 13:00 且 end_time > 12:00）→ 需要 lunch 餐厅
- 包含下午（end_time >= 15:00 且中间有空档）→ 需要 afternoon 活动
- 包含晚饭时间（end_time >= 19:00）→ 需要 dinner 餐厅
- 严格按照时间段决定 steps 数量，不要遗漏任何时间段

## 活动数量规则
- 用户明确要求的活动数量必须满足
- 时间足够时（超过 6 小时）默认安排 2 个活动

## 天气规则
- 天气 risk_level 为 High 时，优先选择 type=indoor 的活动
- 没有 indoor 活动时才考虑 outdoor

## 其他
- reasoning 只写 2-3 条简短理由
- duration_minutes：活动默认 90 分钟，餐厅默认 75 分钟
"""


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else ""


def _build_step_index(
    steps: list[dict],
    activities_by_id: dict[str, dict],
    restaurants_by_id: dict[str, dict],
) -> dict[str, Any]:
    """从 steps 中提取 activities / restaurants 列表及 timeline。"""
    activities_out: list[dict] = []
    restaurants_out: list[dict] = []
    timeline: list[dict] = []

    _phase_label = {
        "morning": "上午", "noon": "中午", "afternoon": "下午",
        "evening": "晚上", "lunch": "午餐", "dinner": "晚餐", "flex": "待定",
    }

    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        poi_type = step.get("poi_type")
        poi_id   = step.get("poi_id") or ""
        phase    = step.get("phase") or "flex"
        label    = step.get("label") or ("活动" if poi_type == "activity" else "用餐")
        duration = step.get("duration_minutes")
        if not isinstance(duration, int) or duration <= 0:
            duration = 90 if poi_type == "activity" else 75

        if poi_type == "activity":
            poi = activities_by_id.get(poi_id, {})
            if poi:
                activities_out.append(poi)
        elif poi_type == "restaurant":
            poi = restaurants_by_id.get(poi_id, {})
            if poi:
                restaurants_out.append(poi)
        else:
            poi = {}

        timeline.append({
            "time":     _phase_label.get(phase, phase),
            "item":     poi.get("name") or "待确认",
            "type":     f"{poi_type}_{phase}" if poi_type else phase,
            "ref_type": poi_type or "",
            "ref_id":   poi_id,
        })

    # 主活动 / 主餐厅（用于下游兼容字段）
    primary_activity    = activities_out[0] if activities_out else {}
    secondary_activity  = activities_out[1] if len(activities_out) > 1 else {}
    # 晚餐优先作为 primary_restaurant
    dinner_restaurants  = [r for s, r in zip(steps, restaurants_out) if isinstance(s, dict) and s.get("phase") == "dinner"]
    primary_restaurant  = dinner_restaurants[0] if dinner_restaurants else (restaurants_out[0] if restaurants_out else {})

    return {
        "activities":          activities_out,
        "activity":            primary_activity,
        "secondary_activity":  secondary_activity,
        "restaurants":         restaurants_out,
        "restaurant":          primary_restaurant,
        "timeline":            timeline,
    }


def candidate_planning_node(state: AgentState) -> AgentState:
    """
    Candidate Planning Node：调用 LLM 基于真实候选池生成 3 个结构化候选方案。

    输入：state["plan_context"] + state["fact_gathering_result"]
    输出：state["candidate_plans"]
    """
    print("[Candidate Planning Node] 基于真实候选池生成 3 个结构化候选方案...")

    plan    = state.get("plan_context") or {}
    facts   = state.get("fact_gathering_result") or {}
    errors  = list(state.get("errors") or [])

    activities:  list[dict] = facts.get("activities") or []
    restaurants: list[dict] = facts.get("restaurants") or []
    weather:     dict       = facts.get("weather") or {}

    activities_by_id  = {a["id"]: a for a in activities  if isinstance(a, dict) and a.get("id")}
    restaurants_by_id = {r["id"]: r for r in restaurants if isinstance(r, dict) and r.get("id")}

    # ── 构造给模型的 prompt ───────────────────────────────────────────────────
    user_message = f"""
用户原始需求：{plan.get("raw_query") or state.get("user_input") or ""}

## 时间信息
- 日期：{plan.get("date_label") or "未指定"}
- 开始时间：{plan.get("start_time") or "未指定"}
- 结束时间：{plan.get("end_time") or "未指定"}
- 时间描述：{plan.get("time_phrase") or ""}

## 人员信息
- 场景：{plan.get("scenario") or "family"}
- 人数：{plan.get("people_count") or 2}
- 是否有孩子：{plan.get("child_friendly") or False}
- 孩子年龄：{plan.get("child_age") or "未知"}

## 出发地
{plan.get("origin_area") or "未指定"}

## 用户偏好
- 饮食偏好：{plan.get("diet_preference") or []}
- 活动风格：{plan.get("activity_style") or []}
- 避免：{plan.get("must_avoid") or []}

## 天气
- 状况：{weather.get("day_weather") or "未知"}
- 风险等级：{weather.get("risk_level") or "unknown"}
- 建议：{weather.get("advice") or ""}

## 可选活动候选池
{json.dumps([
    {
        "id":      a.get("id"),
        "name":    a.get("name"),
        "type":    a.get("type"),
        "rating":  a.get("rating"),
        "tags":    a.get("tags") or [],
        "child_friendly": a.get("child_friendly"),
    }
    for a in activities
], ensure_ascii=False, indent=2)}

## 可选餐厅候选池
{json.dumps([
    {
        "id":       r.get("id"),
        "name":     r.get("name"),
        "rating":   r.get("rating"),
        "tags":     r.get("tags") or [],
        "distance": r.get("distance"),
        "child_friendly": r.get("child_friendly"),
    }
    for r in restaurants
], ensure_ascii=False, indent=2)}

请严格按照 System Prompt 的格式和规则生成 3 个候选方案。
"""

    # ── 调模型 ────────────────────────────────────────────────────────────────
    raw_candidates: list[dict] = []
    try:
        response = get_chat_model().invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        json_text = _extract_json(response.content or "")
        payload = json.loads(json_text) if json_text else {}
        raw_candidates = payload.get("candidates") or []
    except Exception as exc:
        errors.append(f"Candidate planning LLM failed: {exc}")
        print(f"[Candidate Planning Node][ERROR] {exc}")

    # ── 标准化候选结构 ────────────────────────────────────────────────────────
    normalized: list[dict] = []
    for i, raw in enumerate(raw_candidates[:3], 1):
        if not isinstance(raw, dict):
            continue

        steps = raw.get("steps") or []
        # 过滤非法 step（poi_id 不在候选池中的直接跳过，避免模型幻觉）
        valid_steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            poi_type = step.get("poi_type")
            poi_id   = step.get("poi_id") or ""
            if poi_type == "activity"   and poi_id not in activities_by_id:
                print(f"[Candidate Planning Node][WARN] plan_{i} 幻觉 poi_id={poi_id!r}，已丢弃")
                continue
            if poi_type == "restaurant" and poi_id not in restaurants_by_id:
                print(f"[Candidate Planning Node][WARN] plan_{i} 幻觉 poi_id={poi_id!r}，已丢弃")
                continue
            valid_steps.append(step)

        index_result = _build_step_index(valid_steps, activities_by_id, restaurants_by_id)
        normalized.append({
            "id":       raw.get("id") or f"plan_{i}",
            "title":    raw.get("title") or f"候选方案 {i}",
            "steps":    valid_steps,
            "reasoning": [r for r in (raw.get("reasoning") or []) if isinstance(r, str)][:3],
            **index_result,
        })

    # 不足 3 个时补空骨架（下游节点需要固定 3 个）
    while len(normalized) < 3:
        normalized.append({
            "id": f"plan_{len(normalized)+1}", "title": "", "steps": [], "timeline": [],
            "activity": {}, "secondary_activity": {}, "activities": [],
            "restaurant": {}, "restaurants": [], "reasoning": [],
        })

    # ── 打印摘要 ──────────────────────────────────────────────────────────────
    for c in normalized:
        act_names  = [a.get("name") for a in c.get("activities") or [] if a]
        rest_names = [r.get("name") for r in c.get("restaurants") or [] if r]
        print(f"[Candidate Planning Node] {c['id']}: 活动={act_names}, 餐厅={rest_names}")

    return {
        "candidate_plans": {
            "plan_mode":  plan.get("plan_mode") or "activity_plus_meal",
            "candidates": normalized,
        },
        "errors": errors,
    }