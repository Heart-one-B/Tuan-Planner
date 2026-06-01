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
        {"step_id": "step_1", "phase": "morning",   "poi_type": "activity",   "poi_id": "...", "label": "上午活动", "start_time": "10:00", "duration_minutes": 120},
        {"step_id": "step_2", "phase": "lunch",     "poi_type": "restaurant", "poi_id": "...", "label": "午餐",    "start_time": "12:20", "duration_minutes": 75},
        {"step_id": "step_3", "phase": "afternoon", "poi_type": "activity",   "poi_id": "...", "label": "下午活动", "start_time": "14:15", "duration_minutes": 120},
        {"step_id": "step_4", "phase": "dinner",    "poi_type": "restaurant", "poi_id": "...", "label": "晚餐",    "start_time": "18:00", "duration_minutes": 90}
      ],
      "reasoning": ["理由1", "理由2"]
    }
  ]
}

## 选址规则
- 只能使用候选池中提供的 poi_id，禁止编造
- 3 个方案的活动组合要尽量不同
- 相似活动类型视为同一类（剧本杀/密室逃脱/桌游属于同一类），同一方案内活动不重复

## 时间段规则（phase）
根据 start_time 和 end_time 判断需要安排哪些时间段：
- 包含上午（start_time <= 11:00）→ 需要 morning 活动
- 跨越午饭时间（start_time < 13:00 且 end_time > 12:00）→ 需要 lunch 餐厅
- 包含下午（end_time >= 15:00）→ 需要 afternoon 活动（同一下午有多个活动时 phase 都填 afternoon）
- 包含晚饭时间（end_time >= 19:00）→ 需要 dinner 餐厅
- 晚饭后还有时间（end_time >= 21:00）且场景适合 → 可安排 evening 活动
- 严格按照时间段决定 steps 数量，不要遗漏任何时间段
- 可用的 phase 值：morning / lunch / afternoon / dinner / evening

## 时间排期规则（重要）
每个 step 必须填写 start_time，按以下规则推算：

**时间计算：**
- 第一个 step 的 start_time = 用户给定的开始时间
- 后续 step 的 start_time = 上一 step 的 start_time + duration_minutes + 通勤时间
- 同一建筑内通勤 0 分钟，步行可达 10 分钟，需驾车 20～30 分钟
- 所有 step 必须在用户给定的结束时间前完成

**各 phase 最早开始时间：**
- morning：06:00，lunch：11:30，afternoon：13:30
- dinner：最早 17:00，但应根据下午最后一个活动的结束时间自然推算（结束时间 + 通勤时间），不要强行卡在 17:00；活动排得晚时 18:00、19:00 吃晚饭都合理
- evening：19:30
- 累加时间早于地板时，等到地板时间再开始（中间视为自由活动）

**活动时长估算：**
- 儿童乐园 / 游乐场：120～150 分钟；公园 / 景区：90～120 分钟
- 博物馆 / 科技馆：90～120 分钟；商场逛街：60～90 分钟
- 正餐：60～90 分钟；快餐：45～60 分钟

**活动数量：**
- 用户明确要求的活动数量必须满足
- 没有明确要求时，根据可用时间自行推断：
  一个活动结束后，若距下一个 phase 地板（或结束时间）还有 ≥ 90 分钟，则安排下一个活动
  时间不够就不安排，宁可留白也不要硬塞

## evening 场景规则
根据出行场景决定是否安排 evening 活动：
- family（有孩子）→ 不安排 evening，dinner 是行程终点；孩子需要早休息
- couple / friends / team → evening 可正常安排（KTV、酒吧、夜市、演出等）

## 天气规则
根据天气状况自行判断：
- 雨天 / 雷雨 / 大风 → 优先选择 environment=indoor 的活动
- 阴天 / 多云 → 室内外均可，优先用户偏好
- 晴天 → 室内外均可

## 其他
- reasoning 只写 2-3 条简短理由
"""


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else ""


def _minutes_to_hhmm(total: int) -> str:
    total = max(0, min(total, 23 * 60 + 59))
    return f"{total // 60:02d}:{total % 60:02d}"


def _hhmm_to_minutes(t: str) -> int:
    try:
        h, m = map(int, t.split(":"))
        return h * 60 + m
    except Exception:
        return 10 * 60  # fallback: 10:00


_TRAVEL_BUFFER = 20  # 两个 step 之间估算的交通时间（分钟）

# 各 phase 的最早允许开始时间，防止累加出现 16:15 吃晚饭的情况
_PHASE_FLOOR: dict[str, int] = {
    "morning":    6 * 60,
    "lunch":     11 * 60 + 30,
    "afternoon": 13 * 60 + 30,
    "dinner":    17 * 60,
    "evening":   19 * 60 + 30,
}


def _assign_step_times(steps: list[dict], start_time: str) -> list[dict]:
    """
    优先使用模型给出的 start_time，补全 end_time，并应用 phase 时间地板做修正。
    模型未给 start_time 时兜底用累加计算。
    """
    cursor = _hhmm_to_minutes(start_time)
    result = []
    for i, step in enumerate(steps):
        step     = dict(step)
        phase    = step.get("phase") or "afternoon"
        duration = step.get("duration_minutes")
        if not isinstance(duration, int) or duration <= 0:
            duration = 120 if step.get("poi_type") == "activity" else 75
        floor = _PHASE_FLOOR.get(phase, 0)

        if step.get("start_time"):
            # 模型给了时间：用模型的，但检查地板
            model_start  = _hhmm_to_minutes(step["start_time"])
            actual_start = max(model_start, floor)
            if actual_start != model_start:
                print(f"[Candidate Planning Node] 时间修正: {step.get('label')} {step['start_time']} → {_minutes_to_hhmm(actual_start)}")
            cursor = actual_start
        else:
            # 模型没给：兜底累加
            cursor = max(cursor, floor)

        step["start_time"] = _minutes_to_hhmm(cursor)
        step["end_time"]   = _minutes_to_hhmm(cursor + duration)
        cursor += duration + (0 if i == len(steps) - 1 else _TRAVEL_BUFFER)
        result.append(step)
    return result


def _build_step_index(
    steps: list[dict],
    activities_by_id: dict[str, dict],
    restaurants_by_id: dict[str, dict],
) -> dict[str, Any]:
    activities_out: list[dict] = []
    restaurants_out: list[dict] = []
    timeline: list[dict] = []

    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        poi_type = step.get("poi_type")
        poi_id   = step.get("poi_id") or ""
        label    = step.get("label") or ("活动" if poi_type == "activity" else "用餐")

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
            "time":       step.get("start_time") or "",
            "end_time":   step.get("end_time") or "",
            "item":       poi.get("name") or "待确认",
            "label":      label,
            "type":       f"{poi_type}_{step.get('phase', '')}" if poi_type else step.get("phase", ""),
            "ref_type":   poi_type or "",
            "ref_id":     poi_id,
            "duration":   step.get("duration_minutes"),
        })

    primary_activity   = activities_out[0] if activities_out else {}
    secondary_activity = activities_out[1] if len(activities_out) > 1 else {}
    dinner_steps       = [s for s in steps if s.get("phase") == "dinner"]
    dinner_ids         = [s.get("poi_id") for s in dinner_steps if s.get("poi_id")]
    primary_restaurant = (
        restaurants_by_id.get(dinner_ids[0])
        if dinner_ids else
        (restaurants_out[0] if restaurants_out else {})
    )

    return {
        "activities":         activities_out,
        "activity":           primary_activity,
        "secondary_activity": secondary_activity,
        "restaurants":        restaurants_out,
        "restaurant":         primary_restaurant,
        "timeline":           timeline,
    }


def candidate_planning_node(state: AgentState) -> AgentState:
    """
    Candidate Planning Node：调用 LLM 生成 3 个候选方案，并计算每个 step 的实际时刻。

    输入：state["plan_context"] + state["fact_gathering_result"]
    输出：state["candidate_plans"]
    """
    print("[Candidate Planning Node] 基于真实候选池生成 3 个结构化候选方案...")

    plan          = state.get("plan_context") or {}
    facts         = state.get("fact_gathering_result") or {}
    errors        = list(state.get("errors") or [])
    start_time    = plan.get("start_time") or "10:00"
    replan_reason = state.get("replan_reason") or ""

    # ── 打印节点实际收到的输入，便于判断哪些字段有效 ──────────────────────────
    print("\n" + "=" * 40 + " [CANDIDATE PLANNING INPUT] " + "=" * 40)
    print(json.dumps({
        "plan_context": {k: plan.get(k) for k in [
            "raw_query", "scenario", "date_label", "start_time", "end_time",
            "time_phrase", "people_count", "child_friendly", "child_age",
            "origin_area", "diet_preference", "activity_style", "must_avoid", "plan_mode",
        ]},
        "weather":          facts.get("weather") or {},
        "activities_count": len(facts.get("activities") or []),
        "restaurants_count": len(facts.get("restaurants") or []),
        "eta_count":        len(facts.get("eta") or {}),
    }, ensure_ascii=False, indent=2))
    print("=" * 108 + "\n")

    activities:  list[dict] = facts.get("activities") or []
    restaurants: list[dict] = facts.get("restaurants") or []
    weather:     dict       = facts.get("weather") or {}

    activities_by_id  = {a["id"]: a for a in activities  if isinstance(a, dict) and a.get("id")}
    restaurants_by_id = {r["id"]: r for r in restaurants if isinstance(r, dict) and r.get("id")}

    user_message = f"""
用户原始需求：{plan.get("raw_query") or state.get("user_input") or ""}

## 时间信息
- 日期：{plan.get("date_label") or "未指定"}
- 开始时间：{start_time}（请从这个时间开始安排第一个 step）
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
- 温度：{weather.get("day_temp") or "未知"}℃，风向：{weather.get("day_wind") or "未知"}
（请根据以上天气自行判断是否适合户外活动）

## 可选活动候选池
{json.dumps([
    {
        "id":          a.get("id"),
        "name":        a.get("name"),
        "type":        a.get("type"),
        "environment": a.get("environment") or "unknown",
        "rating":      a.get("rating"),
        "child_friendly": a.get("child_friendly"),
    }
    for a in activities
], ensure_ascii=False, indent=2)}

## 可选餐厅候选池
{json.dumps([
    {
        "id":       r.get("id"),
        "name":     r.get("name"),
        "type":     r.get("type"),
        "rating":   r.get("rating"),
        "distance": r.get("distance"),
    }
    for r in restaurants
], ensure_ascii=False, indent=2)}

请根据以上信息，严格按照 System Prompt 的格式和规则生成 3 个候选方案。
注意：duration_minutes 要根据场所类型合理估算，不要全部使用同一个默认值。
{f"""
## 上一轮校验失败原因（必须修正）
{replan_reason}
""" if replan_reason else ""}"""

    raw_candidates: list[dict] = []
    try:
        print("[Candidate Planning Node] 模型生成中...\n")
        full_content = ""
        for chunk in get_chat_model().stream([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]):
            token = chunk.content or ""
            print(token, end="", flush=True)
            full_content += token
        print()  # 换行

        json_text = _extract_json(full_content)
        payload = json.loads(json_text) if json_text else {}
        raw_candidates = payload.get("candidates") or []
    except Exception as exc:
        errors.append(f"Candidate planning LLM failed: {exc}")
        print(f"\n[Candidate Planning Node][ERROR] {exc}")

    normalized: list[dict] = []
    for i, raw in enumerate(raw_candidates[:3], 1):
        if not isinstance(raw, dict):
            continue

        steps = raw.get("steps") or []

        # 过滤幻觉 poi_id
        valid_steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            poi_type = step.get("poi_type")
            poi_id   = step.get("poi_id") or ""
            if poi_type == "activity" and poi_id not in activities_by_id:
                print(f"[Candidate Planning Node][WARN] plan_{i} 幻觉 poi_id={poi_id!r}，已丢弃")
                continue
            if poi_type == "restaurant" and poi_id not in restaurants_by_id:
                print(f"[Candidate Planning Node][WARN] plan_{i} 幻觉 poi_id={poi_id!r}，已丢弃")
                continue
            valid_steps.append(step)

        # 计算每个 step 的实际时刻
        timed_steps = _assign_step_times(valid_steps, start_time)

        index_result = _build_step_index(timed_steps, activities_by_id, restaurants_by_id)
        normalized.append({
            "id":        raw.get("id") or f"plan_{i}",
            "title":     raw.get("title") or f"候选方案 {i}",
            "steps":     timed_steps,
            "reasoning": [r for r in (raw.get("reasoning") or []) if isinstance(r, str)][:3],
            **index_result,
        })

    while len(normalized) < 3:
        normalized.append({
            "id": f"plan_{len(normalized)+1}", "title": "", "steps": [], "timeline": [],
            "activity": {}, "secondary_activity": {}, "activities": [],
            "restaurant": {}, "restaurants": [], "reasoning": [],
        })

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