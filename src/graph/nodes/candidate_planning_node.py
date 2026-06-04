from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import AgentState
from src.model.factory import get_chat_model
from src.utils.state_utils import _append_error

# ──────────────────────────────────────────────────────────────────────────────
# v3 重构：从"档位 + 求解器弹性填充"改为"模型估真实时长 + 求解器只收口"。
#
# 核心理念：
#   - 模型对活动时长有可靠常识（剧本杀≈3h、咖啡≈1h、正餐≈1h），把这个常识用起来。
#   - 活动：模型直接给 duration_minutes（真实估计）+ duration_flex（弹性区间）。
#   - 餐饮：时长方差小、规则清晰，不交给模型，用固定区间 _MEAL_RANGE（按 meal_tier）。
#   - 求解器不再"弹性拉伸填满时间"，只做三件事：
#       1) 按模型给的时长顺序排布 + 真实通勤
#       2) 餐饮窗口顺延（午饭 11:30~13:00、晚饭 17:30~19:00，活动顶则顺延）
#       3) 超时收口（末段超出 end_time 时，在 flex 下限内压缩活动，再不行压末段）
#   - "下午留了大空档"不再是求解器算不准，而是模型自己没把时段填合理 → 交 rule_validation
#     上报 spare_gap，让模型重排（新架构下无系统偏差，repair 收敛）。
#
# 不变：waypoint 处理、poi_type 自动修正、幻觉 poi_id 过滤、_build_step_index、三方案兜底。
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是本地生活行程规划助手，负责根据用户需求和候选 POI 池生成 3 个候选行程方案。

## 最高准则（务必遵守）

### 原话中有两类信息，作用不同，不能混淆
**第一类：时间信息**（几点出发、几点结束）→ 定义时间窗，是排期的**硬约束**。
- 时间窗内每个时段都需要覆盖，不能因为用户说"下午聚"就跳过上午。
- 例："早上九点到晚上九点" → 时间窗从 09:00 开始，必须从上午开始排，
  不能因为用户说"下午聚一聚"就把上午空着。

**第二类：活动描述**（下午聚一聚、晚上喝酒、想吃 DQ）→ 描述活动重心和偏好，
影响各时段安排什么，但**不改变时间窗的起止**。
- "下午和朋友聚一聚" = 活动重心在下午，不代表上午不需要安排。
- "晚上想喝点酒" = 喝酒安排在晚上，不代表下午不需要活动。

### 其他准则
- 不要预设用户想把时间填满。**段数由用户需求和时间窗自然决定。留白优于硬塞，但也不要留下大段空白（见下方"时段覆盖自检"）。**
- 用户只在某个时段点名的活动（如"晚上想喝点酒"），就安排在那个时段，不要扩散到其他时段。

## 输出要求
只输出 JSON，不输出任何其他内容。steps 的数量和 phase 由你根据需求决定。

格式如下（实际 steps 数组长度自定）：
{
  "candidates": [
    {
      "id": "plan_1",
      "title": "方案标题",
      "steps": [
        {"step_id": "step_1", "phase": "afternoon", "poi_type": "activity", "poi_id": "...", "label": "剧本杀", "duration_minutes": 180, "duration_flex": [150, 210]},
        {"step_id": "step_2", "phase": "dinner", "poi_type": "restaurant", "poi_id": "...", "label": "晚餐", "meal_tier": "medium"}
      ],
      "reasoning": ["理由1", "理由2"]
    }
  ]
}
可用的 phase 值：morning / lunch / afternoon / dinner / evening。
poi_type 值：activity / restaurant。

## 时长怎么填（关键）
**活动（poi_type=activity）**：你需要根据生活常识，给出普通人实际会花费的真实时长。
- duration_minutes：典型时长（分钟），按你对这个活动的常识估计
- duration_flex：[下限, 上限]，可接受的弹性范围（系统超时时会在此范围内压缩）
- 常识参考（你应按实际活动微调，不要照抄）：
    剧本杀 150~210（新本可到 240）、密室逃脱 60~90、电影 120~150、桌游 90~150、
    台球 60~120、KTV 120~180、咖啡馆 45~90、美术馆/展馆 60~120、公园散步 30~60、
    商场逛街 90~180、酒吧/精酿小酌 60~120
- 单个活动不要超过 240 分钟。

**餐饮（poi_type=restaurant）**：分两种情况：

普通餐厅（正餐/快餐/轻食）→ 只填 meal_tier，不填 duration_minutes：
- "short"：快餐、轻食（约 20~45 分钟）
- "medium"：普通正餐、火锅、串串（约 45~75 分钟）
- "long"：自助餐、大桌宴（约 90~120 分钟）

**酒吧、小酒馆、清吧、精酿店等喝酒场所** → 填 duration_minutes + duration_flex，不填 meal_tier：
- 这类场所本质是"喝酒活动"，时长按活动估，不按快餐算
- 典型时长：60~120 分钟，建议给 duration_minutes: 90，duration_flex: [60, 120]
- 例：{"poi_type":"restaurant","poi_id":"...","label":"小酒馆","duration_minutes":90,"duration_flex":[60,120]}

## 你不需要计算具体时刻
不要填 start_time / end_time。系统会根据真实通勤时间和上面的时长精确计算每个 step 的时刻。

## 选址规则
- 只能使用候选池中提供的 poi_id，禁止编造
- 3 个方案的活动组合要尽量不同
- 相似活动类型视为同一类（剧本杀/密室逃脱/桌游属于同一类），同一方案内活动不重复

## 时间段规则（phase）—— 用"窗口重叠"判断
根据用户的开始/结束时间，判断时间窗真正覆盖了哪些时段，只为被覆盖的时段安排 step：
- 覆盖上午（start_time <= 11:00）→ 可安排 morning 活动
- 跨越午饭（start_time < 13:00 且 end_time > 12:00）→ 安排 lunch 餐厅
- 覆盖下午（start_time < 17:00 且 end_time >= 14:00）→ 安排 afternoon 活动
- 覆盖晚饭（start_time < 20:00 且 end_time >= 18:30）→ **必须**安排 dinner 餐厅
- 覆盖夜间（end_time >= 21:00）→ 可安排 evening 活动
- 仅当时间窗真正覆盖某时段时才安排它。例如 14:00-22:00 不含上午，就不要排 morning。

## 餐饮是必须项
**午饭（lunch）**：时间窗跨越 11:30~13:00 时，必须安排 lunch 餐厅。用户没提午饭不代表不吃饭。
**晚饭（dinner）**：时间窗跨越 17:30~19:00 时，必须安排 dinner 餐厅。同上。
用户没有特别要求的情况下，午饭可以给 meal_tier="short"（快餐/轻食，节省时间）；
晚饭通常给 meal_tier="medium" 或根据用户偏好选。

## 时段覆盖自检（输出每个方案前必做，重要）

user_message 里会给出每个时段的可用分钟数（代码已算好）。
根据这个数字选出时长匹配的活动，不能让活动总时长和可用时长差距过大。

### 每个时段的选活动规则（按可用分钟数判断）

**上午（morning）**
- 可用 < 90min → 最多 1 个 short，或不排直接从午饭开始
- 可用 90~180min → 优先 1 个 medium；没合适的再考虑 short+short
- 可用 > 180min → 优先 1 个 long；没合适的再考虑 medium+short

**下午（afternoon）**
- 可用 90~180min → 优先 1 个 medium，时长够就不加第二个
- 可用 180~270min → 优先 1 个 long（剧本杀/密室/KTV）；没合适的 long 再选 medium+short
- 可用 > 270min → 必须 2 个活动（long+short 或 medium+medium）

**选一个还是两个的权衡原则：**
- 用户原话点名了该时段的活动 → 先排点名的，剩余不足 90min 就不加了
- 用户没点名 → 优先选 1 个时长匹配的活动（long > medium），而不是凑两个短的
- 已排活动结束后距下一个饭点不足 90min → 不必再加，等饭点是合理留白

**覆盖率验证（输出前必查）：**
该时段活动总时长 ÷ 可用分钟数 ≥ 60%，否则补活动或换更长的活动。
例：上午可用 150min，只排了 60min 咖啡 → 覆盖率 40%，不合格，必须换 medium 活动或再加一个。

## evening 场景规则
- family（有孩子）→ 不安排 evening，dinner 是终点
- couple / friends / team → evening 可正常安排，晚饭后可直接开始，无需强制等待

## 天气规则
- 雨/雷雨/大风 → 优先 environment=indoor
- 阴天/多云/晴 → 室内外均可，优先贴合用户原话

## waypoint（途径小需求）
用户有时提出非主要活动的小需求（吃DQ、买奶茶、顺路甜品等）。系统会单独提供 waypoint POI：
- waypoint 不独占完整时段，插在两个活动之间作为过渡，duration_minutes 给 30~45，duration_flex 给 [20, 60]
- 若提供了 waypoint 但 not_found=true，在 reasoning 里说明"附近未找到XXX，已跳过"
- waypoint step 的 phase 填它所在时段，poi_type 填 "activity"

## 其他
- reasoning 只写 2-3 条简短理由，体现"为什么这样安排贴合用户原话"
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


_DEFAULT_TRAVEL = 12      # 拿不到 ETA 时的兜底通勤（分钟）

# 餐饮固定时长区间（不交给模型估）。取值偏区间下限，符合"一顿饭不会太久"的直觉。
_MEAL_RANGE: dict[str, tuple[int, int]] = {
    "short":  (30,  45),   # 快餐/轻食/小酌
    "medium": (45,  75),   # 普通正餐/火锅/串串
    "long":   (90,  120),  # 自助/大桌宴
}
_DEFAULT_MEAL_TIER = "medium"

# 活动时长 sanity 边界（裁剪模型给的离谱值）
_ACTIVITY_MIN = 20
_ACTIVITY_MAX = 240


def _calc_slot_budgets(start_time: str, end_time: str) -> dict:
    """根据用户时间窗，计算各时段的可用分钟数，传给模型让它选出时长匹配的活动。"""
    start = _hhmm_to_minutes(start_time)
    end   = _hhmm_to_minutes(end_time) if end_time else 21 * 60

    LUNCH_FLOOR    = 11 * 60 + 30   # 午饭最早 11:30
    LUNCH_END_EST  = 13 * 60        # 午饭预计结束（保守）
    DINNER_FLOOR   = 17 * 60 + 30   # 晚饭最早 17:30
    DINNER_END_EST = 19 * 60        # 晚饭预计结束

    budgets: dict = {}

    if start < LUNCH_FLOOR:
        budgets["morning"] = LUNCH_FLOOR - start

    if start < DINNER_FLOOR and end > LUNCH_END_EST:
        budgets["afternoon"] = DINNER_FLOOR - LUNCH_END_EST  # 约 270min

    if end > DINNER_END_EST:
        budgets["evening"] = max(0, end - DINNER_END_EST)

    return budgets

# 餐饮弹性窗口（分钟）。cursor < 下限 → 等到下限；落在窗口内 → 直接用（顺延）。
_LUNCH_WINDOW  = (11 * 60 + 30, 13 * 60)       # 11:30 ~ 13:00
_DINNER_WINDOW = (17 * 60 + 30, 19 * 60)       # 17:30 ~ 19:00
_EVENING_FLOOR = 19 * 60                        # 夜间软地板

_OVERFLOW_TOLERANCE = 15   # 末段超出 end_time 15 分钟内不收口（用户可自行提前结束）


def _meal_duration(step: dict) -> int:
    """
    餐饮时长：
    - 若 step 提供了 duration_minutes（如酒吧/小酒馆），直接用（裁剪到 sanity 边界）
    - 否则按 meal_tier 取固定区间典型值（偏下，约下限 + 1/3 区间）
    """
    if step.get("duration_minutes") is not None:
        d = step["duration_minutes"]
        if isinstance(d, (int, float)):
            return max(_ACTIVITY_MIN, min(int(d), _ACTIVITY_MAX))
    tier = step.get("meal_tier") or _DEFAULT_MEAL_TIER
    lo, hi = _MEAL_RANGE.get(tier, _MEAL_RANGE[_DEFAULT_MEAL_TIER])
    return lo + (hi - lo) // 3


def _activity_duration(step: dict) -> int:
    """活动时长：用模型给的 duration_minutes，裁剪到 sanity 边界。"""
    d = step.get("duration_minutes")
    if not isinstance(d, (int, float)):
        d = 90  # 模型没给时的兜底
    return max(_ACTIVITY_MIN, min(int(d), _ACTIVITY_MAX))


def _activity_floor(step: dict, current: int) -> int:
    """活动可压缩到的下限：取 duration_flex[0]，无则不压（等于当前时长）。"""
    flex = step.get("duration_flex")
    if isinstance(flex, list) and len(flex) == 2 and isinstance(flex[0], (int, float)):
        return max(_ACTIVITY_MIN, min(int(flex[0]), current))
    return current  # 没给 flex 就不压


def _travel_to(prev_step: dict | None, step: dict, eta: dict) -> int:
    rec = eta.get(step.get("poi_id") or "")
    if isinstance(rec, dict) and rec.get("eta_minutes") is not None:
        return int(rec["eta_minutes"])
    return _DEFAULT_TRAVEL


def _assign_step_times(
    steps: list[dict],
    start_time: str,
    end_time: str = "",
    eta: dict | None = None,
) -> list[dict]:
    """
    时刻求解器（v3）：模型给真实时长，这里只排布 + 餐饮顺延 + 超时收口，不再弹性拉伸。

    流程：
      1. 确定每段时长：活动用模型的 duration_minutes（裁剪到 sanity），餐饮用 _MEAL_RANGE。
      2. 布局：按时长顺序排，加真实通勤；餐饮 cursor 不低于窗口下限（活动顶则顺延）；
         evening 不低于软地板 19:00。
      3. 超时收口：若末段结束超出 end_time 超过容忍值，从后往前在活动 flex 下限内压缩；
         仍超则直接压缩末段时长。
    不做任何"填满空档"的拉伸——空档是否过大由 rule_validation 判定，交模型重排。
    """
    eta = eta or {}
    plan_start = _hhmm_to_minutes(start_time)
    plan_end   = _hhmm_to_minutes(end_time) if end_time else None

    n = len(steps)
    if n == 0:
        return []

    # ── 1. 各段时长 + 通勤 ──────────────────────────────────────────────────
    durations: list[int] = []
    travels:   list[int] = []
    for i, step in enumerate(steps):
        if step.get("poi_type") == "restaurant":
            durations.append(_meal_duration(step))
        else:
            durations.append(_activity_duration(step))
        prev = steps[i - 1] if i > 0 else None
        travels.append(0 if i == 0 else _travel_to(prev, step, eta))

    # ── 2. 布局 ──────────────────────────────────────────────────────────────
    def _layout(durs: list[int]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = plan_start
        for i, step in enumerate(steps):
            phase = step.get("phase") or "afternoon"
            cursor += travels[i]
            if phase == "lunch":
                cursor = max(cursor, _LUNCH_WINDOW[0])
            elif phase == "dinner":
                cursor = max(cursor, _DINNER_WINDOW[0])
            elif phase == "evening":
                cursor = max(cursor, _EVENING_FLOOR)
            start = cursor
            end   = start + durs[i]
            spans.append((start, end))
            cursor = end
        return spans

    spans = _layout(durations)

    # ── 2b. 饭点保护：如果某活动结束 + 通勤会超过餐饮窗口上限，压缩该活动 ──────
    # 规则：activity 结束时间 + 下一个 step 的通勤 <= 餐饮窗口上限
    #   午饭上限 13:00，晚饭上限 19:00
    # 压缩不低于 flex 下限；压到底还超的话交 rule_validation 报 lunch/dinner_too_late
    _LUNCH_END_HARD  = 13 * 60
    _DINNER_END_HARD = 19 * 60
    changed_meal = True
    while changed_meal:
        changed_meal = False
        for i in range(len(steps) - 1):
            if steps[i].get("poi_type") == "restaurant":
                continue
            next_phase = steps[i + 1].get("phase") or ""
            if next_phase not in ("lunch", "dinner"):
                continue
            meal_hard = _LUNCH_END_HARD if next_phase == "lunch" else _DINNER_END_HARD
            act_start = spans[i][0]
            act_end   = spans[i][1]
            transit   = travels[i + 1]
            if act_end + transit <= meal_hard:
                continue  # 没问题
            # 需要压缩
            target_end = meal_hard - transit
            new_dur = max(_activity_floor(steps[i], durations[i]),
                          target_end - act_start)
            if new_dur < durations[i]:
                print(
                    f"[Candidate Planning Node] 饭点保护压缩: {steps[i].get('label')} "
                    f"{durations[i]}min → {new_dur}min（保证{next_phase}在{meal_hard//60}:00前开始）"
                )
                durations[i] = new_dur
                spans = _layout(durations)
                changed_meal = True
                break

    # ── 3. 超时收口 ──────────────────────────────────────────────────────────
    if plan_end is not None and spans[-1][1] > plan_end + _OVERFLOW_TOLERANCE:
        overshoot = spans[-1][1] - plan_end
        # 3a. 从后往前压缩活动（餐饮不压），各自压到 flex 下限
        for i in range(n - 1, -1, -1):
            if overshoot <= 0:
                break
            if steps[i].get("poi_type") == "restaurant":
                continue
            floor = _activity_floor(steps[i], durations[i])
            cut = min(durations[i] - floor, overshoot)
            if cut > 0:
                print(
                    f"[Candidate Planning Node] 超时压缩活动: {steps[i].get('label')} "
                    f"{durations[i]}min → {durations[i]-cut}min"
                )
                durations[i] -= cut
                overshoot    -= cut
        spans = _layout(durations)

        # 3b. 仍超出 → 直接压末段（保底，不强制下限）
        if spans[-1][1] > plan_end:
            last_start = spans[-1][0]
            squeezed = max(_ACTIVITY_MIN, plan_end - last_start)
            if squeezed != durations[-1]:
                print(
                    f"[Candidate Planning Node] 末段压缩: {steps[-1].get('label')} "
                    f"{durations[-1]}min → {squeezed}min（避免超出 {end_time}）"
                )
                durations[-1] = squeezed
                spans = _layout(durations)

    # ── 4. 写回 ─────────────────────────────────────────────────────────────
    result = []
    for i, step in enumerate(steps):
        step = dict(step)
        s, e = spans[i]
        step["start_time"]       = _minutes_to_hhmm(s)
        step["end_time"]         = _minutes_to_hhmm(e)
        step["duration_minutes"] = durations[i]   # 回填实际换算时长（餐饮也回填）
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
    print("[Candidate Planning Node] 基于真实候选池生成 3 个结构化候选方案...")

    plan          = state.get("plan_context") or {}
    facts         = state.get("fact_gathering_result") or {}
    errors        = list(state.get("errors") or [])
    start_time    = plan.get("start_time") or "10:00"
    replan_reason = state.get("replan_reason") or ""

    raw_query = plan.get("raw_query") or state.get("user_input") or ""

    print("\n" + "=" * 40 + " [CANDIDATE PLANNING INPUT] " + "=" * 40)
    print(json.dumps({
        "raw_query": raw_query,
        "plan_context": {k: plan.get(k) for k in [
            "scenario", "date_label", "start_time", "end_time",
            "time_phrase", "people_count", "child_friendly", "child_age",
            "origin_area", "activity_style", "must_avoid", "plan_mode",
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

    waypoints: list[dict] = facts.get("waypoints") or []
    waypoints_by_id = {w["id"]: w for w in waypoints if isinstance(w, dict) and w.get("id") and not w.get("not_found")}

    if waypoints:
        wp_lines = []
        for w in waypoints:
            if w.get("not_found"):
                wp_lines.append(f"- 用户想要「{w.get('waypoint_raw')}」，附近未找到，请在 reasoning 中说明已跳过")
            else:
                hint = f"（时间提示：{w['waypoint_time_hint']}）" if w.get("waypoint_time_hint") else ""
                wp_lines.append(f"- 用户想要「{w.get('waypoint_raw')}」{hint}，可用 POI：{w.get('name')}（id={w.get('id')}），请插入方案合适位置，duration_minutes=30、duration_flex=[20,60]，poi_type=activity")
        waypoint_section = "\n".join(wp_lines)
    else:
        waypoint_section = "无途径小需求"

    activity_style: list = plan.get("activity_style") or []
    explicit = "、".join(activity_style) if activity_style else "（无明确点名，由你根据场景自由推荐）"

    # 计算各时段可用分钟数，传给模型用于选出时长匹配的活动
    _budgets = _calc_slot_budgets(start_time, plan.get("end_time") or "")
    _label_map = {"morning": "上午", "afternoon": "下午", "evening": "夜间"}
    _guide_map = {
        "morning": (
            "可用 < 90min → 最多 1 个 short；"
            "90~180min → 优先 1 个 medium；"
            "> 180min → 优先 1 个 long"
        ),
        "afternoon": (
            "90~180min → 优先 1 个 medium；"
            "180~270min → 优先 1 个 long，没合适 long 再选 medium+short；"
            "> 270min → 必须 2 个活动"
        ),
        "evening": "按剩余时间和用户需求自由安排",
    }
    _slot_lines = []
    for _slot, _mins in _budgets.items():
        _lbl = _label_map.get(_slot, _slot)
        _gd  = _guide_map.get(_slot, "")
        _slot_lines.append(f"- {_lbl}（{_slot}）：可用 {_mins} 分钟 → {_gd}")
    slot_budget_hint = "\n".join(_slot_lines) if _slot_lines else "- 无需安排活动时段"

    user_message = f"""\
# 用户原始需求（唯一权威，请逐字理解，下面所有结构化字段仅供参考）
「{raw_query}」

请基于以上原话，判断用户想做几件事、各在什么时段、哪些是明确点名的，然后规划。
不要预设要填满时间，段数由原话和时间窗自然决定；但完成"时段覆盖自检"，别留过大空档。

## 用户明确点名的活动（硬约束，必须全部安排）
- 点名活动：{explicit}
- 若原话中还有"再安排一个活动""另外去 X""然后想玩 Y"等表述，同样必须体现。
- **活动（poi_type=activity）和餐饮（poi_type=restaurant）是两类不同的 step，不能互相替代。**
- 这是硬约束，不是建议。

## 时间信息（用于排期计算）
- 日期：{plan.get("date_label") or "未指定"}
- 开始时间：{start_time}（第一个 step 从此时间开始）
- 结束时间：{plan.get("end_time") or "未指定"}
- 时间描述：{plan.get("time_phrase") or ""}

## 各时段可用分钟数（代码预算，直接用于选活动时长）
{slot_budget_hint}

## 人员信息
- 场景：{plan.get("scenario") or "friends"}
- 人数：{plan.get("people_count") or 2}
- 是否有孩子：{plan.get("child_friendly") or False}
- 孩子年龄：{plan.get("child_age") or "未知"}

## 出发地
{plan.get("origin_area") or "未指定"}

## 用户偏好（参考，原话优先）
- 活动风格：{plan.get("activity_style") or []}
- 明确避免：{plan.get("must_avoid") or []}

## 天气
- 状况：{weather.get("day_weather") or "未知"}，温度：{weather.get("day_temp") or "未知"}℃，风向：{weather.get("day_wind") or "未知"}

## 可选活动候选池
{json.dumps([
    {
        "id": a.get("id"), "name": a.get("name"), "type": a.get("type"),
        "environment": a.get("environment") or "unknown",
        "rating": a.get("rating"), "child_friendly": a.get("child_friendly"),
    }
    for a in activities
], ensure_ascii=False, indent=2)}

## 可选餐厅候选池
{json.dumps([
    {
        "id": r.get("id"), "name": r.get("name"), "type": r.get("type"),
        "rating": r.get("rating"), "distance": r.get("distance"),
    }
    for r in restaurants
], ensure_ascii=False, indent=2)}

## 途径小需求（waypoints）
{waypoint_section}

请生成 3 个候选方案。活动 step 给出 duration_minutes 和 duration_flex（按你的时长常识估），
餐饮 step 只给 meal_tier（short/medium/long）。不要填写任何具体时刻。
输出每个方案前，务必完成 system 中的"时段覆盖自检"。
{f'''
## 上一轮校验失败原因（必须修正）
{replan_reason}
''' if replan_reason else ""}"""

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
        print()

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
        valid_steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            poi_type = step.get("poi_type")
            poi_id   = step.get("poi_id") or ""

            in_activities  = poi_id in activities_by_id
            in_restaurants = poi_id in restaurants_by_id
            in_waypoints   = poi_id in waypoints_by_id

            if not in_activities and not in_restaurants and not in_waypoints:
                print(f"[Candidate Planning Node][WARN] plan_{i} 幻觉 poi_id={poi_id!r}，已丢弃")
                continue

            # 自动修正 poi_type
            if poi_type == "restaurant" and not in_restaurants and in_activities:
                print(f"[Candidate Planning Node] poi_type 修正: {poi_id!r} restaurant→activity")
                step = dict(step)
                step["poi_type"] = "activity"
            elif poi_type == "activity" and not in_activities and in_restaurants:
                print(f"[Candidate Planning Node] poi_type 修正: {poi_id!r} activity→restaurant")
                step = dict(step)
                step["poi_type"] = "restaurant"

            valid_steps.append(step)

        timed_steps = _assign_step_times(
            valid_steps, start_time, plan.get("end_time") or "", facts.get("eta") or {}
        )
        index_result = _build_step_index(timed_steps, {**activities_by_id, **waypoints_by_id}, restaurants_by_id)
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