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

## 最高准则（务必遵守）
- **用户的原始需求（原话）是唯一权威**。请逐字理解原话想要什么：想做几件事、每件事大概在什么时段、哪些是明确点名的、哪些是开放的。
- 结构化字段（场景、时间、偏好等）只是辅助参考，**当它与原话语气/细节有出入时，以原话为准**。
- 不要预设用户想把时间填满。**段数由用户需求和时间窗自然决定，可能只有 1 段，也可能多段。留白优于硬塞。**
- 用户只在某个时段点名的活动（如"晚上想喝点酒"），就安排在那个时段，不要扩散到其他时段；其他时段安排什么，由你根据原话和场景自行判断。

## 输出要求
只输出 JSON，不输出任何其他内容。steps 的数量和 phase 由你根据需求决定。
**重要：你只负责"语义规划"——决定做哪些活动、顺序、各自属于哪个时段、以及每个活动的时长档位。
你不需要、也不要计算或填写任何具体时刻（start_time / end_time）和分钟数，这些由系统根据真实通勤时间精确计算。**

格式如下（下面只示意单个 step 的字段，实际 steps 数组长度自定）：
{
  "candidates": [
    {
      "id": "plan_1",
      "title": "方案标题",
      "steps": [
        {"step_id": "step_1", "phase": "dinner", "poi_type": "restaurant", "poi_id": "...", "label": "晚餐", "duration_tier": "medium"}
      ],
      "reasoning": ["理由1", "理由2"]
    }
  ]
}
可用的 phase 值：morning / lunch / afternoon / dinner / evening。
poi_type 值：activity / restaurant。

## 时长档位（duration_tier）—— 用定性判断，不要报具体分钟
休闲行程的时长本就因人而异、无法精确，请按活动的强度给一个**档位**，系统会结合可用时间换算成实际时长：
- "short"：约 1 小时（咖啡馆小坐、逛商场、快餐、小酌一杯）
- "medium"：约 2 小时（正餐、台球、桌游、KTV、一般景点）
- "long"：约 2-3 小时（剧本杀、密室逃脱、电影、大型场馆/乐园）——电影至少2小时，请给 long
按"这是个轻松/常规/深度的活动"来选档，不要纠结具体分钟。

## 选址规则
- 只能使用候选池中提供的 poi_id，禁止编造
- 3 个方案的活动组合要尽量不同
- 相似活动类型视为同一类（剧本杀/密室逃脱/桌游属于同一类），同一方案内活动不重复

## 时间段规则（phase）—— 用"窗口重叠"判断，不要用单端点
根据用户的开始/结束时间，判断时间窗真正覆盖了哪些时段，只为被覆盖的时段安排 step：
- 覆盖上午（start_time <= 11:00）→ 可安排 morning 活动
- 跨越午饭（start_time < 13:00 且 end_time > 12:00）→ 可安排 lunch 餐厅
- 覆盖下午（start_time < 17:00 且 end_time >= 14:00）→ 可安排 afternoon 活动
- 覆盖晚饭（start_time < 20:00 且 end_time >= 18:30）→ **必须**安排 dinner 餐厅
- 覆盖夜间（end_time >= 21:00）→ 可安排 evening 活动
- **注意：仅当时间窗真正覆盖某时段时才安排它。** 例如 14:00-22:00 不包含上午，就不要安排 morning；18:00-22:00 不包含下午，就不要安排 afternoon。

## dinner 是必须项（重要）
**时间窗跨越晚饭时段（覆盖 17:30-20:00 之间的任意时间）时，dinner 餐厅是必须安排的，不可省略。**
人需要吃饭——不管下午安排了多少活动，只要时间窗包含晚饭时间，就必须有一个 poi_type=restaurant 的 dinner step。
用户原话没有提到晚饭，不代表不需要吃饭；这是基本的生活常识，不是可选项。

## 时段密度——输出前必须完成的推算步骤（硬规则）

档位时长参考：short≈1h，medium≈2h，long≈2.5h。

**每次输出前，按以下步骤逐项推算，不能跳过：**

### 上午（morning）
可用时间 = 11:30 - 用户开始时间
- 可用 < 1.5h → 最多 1 个 short
- 1.5h ≤ 可用 < 3h → 1 个 medium 或 1short+1short
- 可用 ≥ 3h → 可安排 1 个 long，或 1medium+1short
- **绝对不要在上午塞 2 个 long**，否则午饭会被顶过 13:00

### 下午（afternoon）——最容易出错，重点检查
可用时间 = 17:30 - 午饭结束时间（午饭 medium 约 1h，结束约 12:30~13:00）

**下午所有活动时长之和必须覆盖到 16:00 以后**，否则到 17:30 晚饭前会出现超过 90 分钟空档，触发重规划。

推算示例（必须照此执行）：
- 午饭 12:30 结束 → 下午可用约 5h → 需要 2~3 个活动（如 long+short，或 medium+medium，或 medium+short+short）
- 午饭 13:00 结束 → 下午可用约 4.5h → 需要 2 个活动（如 long+short，或 medium+medium）
- 午饭 13:30 结束 → 下午可用约 4h → 需要 2 个活动（如 medium+medium，或 medium+short）

**结论：只要午饭在 13:30 前结束，下午就必须安排至少 2 个活动。只安排 1 个必然触发重规划。**

### 验证（输出前自查）
排完 steps 后，心算最后一个下午活动的结束时间：
- 结束时间 ≥ 16:00 → 通过，到晚饭等待 ≤ 90 分钟合理
- 结束时间 < 16:00 → 不通过，必须再加一个 short 活动

**不要在上午/下午堆太多 long**，否则会把午饭顶过 13:00 或晚饭顶过 19:00。

## evening 场景规则
- family（有孩子）→ 不安排 evening，dinner 是终点
- couple / friends / team → evening 可正常安排
- **晚饭结束后可以直接开始 evening 活动，不需要强制等待消化时间**

## 天气规则
- 雨/雷雨/大风 → 优先 environment=indoor
- 阴天/多云/晴 → 室内外均可，优先贴合用户原话

## waypoint（途径小需求，重要）
用户有时会提出非主要活动的小需求（吃DQ、买奶茶、顺路买甜品等）。
系统会把搜索到的 waypoint POI 单独提供给你，你需要把它安排进方案里：
- waypoint 不是主要活动，不独占完整时段，而是**插在两个活动之间作为过渡**
- 选择最自然的位置（根据 time_hint 或上下文判断），时长给 short 档
- 如果提供了 waypoint 但 not_found=true，在 reasoning 里说明"附近未找到XXX，已跳过"
- waypoint step 的 phase 填它所在的时段（afternoon/evening 等），poi_type 填 "activity"

## 其他
- reasoning 只写 2-3 条简短理由，并体现"为什么这样安排贴合用户原话"
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


_DEFAULT_TRAVEL = 15      # 拿不到 ETA 时的兜底通勤（分钟）
_MIN_STEP_DURATION = 45   # 超时压缩时的时长下限
_MIN_TRANSITION_MINUTES = 10  # 相邻 step 最小间隔，与 rule_validation 保持一致

# 时长档位 → (下限, 上限) 分钟区间。活动和餐饮分开，避免一顿饭被拉到 140 分钟。
_TIER_RANGE_ACTIVITY: dict[str, tuple[int, int]] = {
    "short":  (45,  75),
    "medium": (90,  150),
    "long":   (120, 180),  # 电影/剧本杀等需要 2 小时+的活动
}
_TIER_RANGE_MEAL: dict[str, tuple[int, int]] = {
    "short":  (30,  45),   # 快餐/轻食
    "medium": (45,  75),   # 普通正餐
    "long":   (90,  120),  # 自助/火锅/大桌宴
}
_DEFAULT_TIER = "medium"

# ── 餐饮时间窗口（自适应窗口，不是固定开始时间）────────────────────────────
# 餐饮开始时间 = max(上一活动结束 + 通勤, 窗口下限)
# 若超出窗口上限才触发 repair
_LUNCH_WINDOW  = (11 * 60 + 30, 13 * 60)       # 午饭合理区间 11:30 ~ 13:00
_DINNER_WINDOW = (17 * 60 + 30, 19 * 60)       # 晚饭合理区间 17:30 ~ 19:00


def _tier_range(step: dict) -> tuple[int, int]:
    poi_type = step.get("poi_type")
    table = _TIER_RANGE_MEAL if poi_type == "restaurant" else _TIER_RANGE_ACTIVITY
    tier = step.get("duration_tier")
    return table.get(tier) or table[_DEFAULT_TIER]


def _travel_to(prev_step: dict | None, step: dict, eta: dict) -> int:
    if prev_step is None:
        rec = eta.get(step.get("poi_id") or "")
        if isinstance(rec, dict) and rec.get("eta_minutes") is not None:
            return int(rec["eta_minutes"])
        return _DEFAULT_TRAVEL
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
    纯代码时刻求解器。核心理念：活动驱动、餐饮跟随。
    - lunch:  start = max(prev_end + transit, LUNCH_WINDOW[0])，超出上限由 rule_validation 上报
    - dinner: start = max(prev_end + transit, DINNER_WINDOW[0])，超出上限优先压缩下午活动，
              压缩不动（活动已在下限）则交 rule_validation 上报 dinner_too_late
    - evening: start = dinner_end + transit（不强制消化间隔）

    时长分配分三层（顺序很重要）：
      1. 餐前空档填充：把餐饮前的活动拉长以消除"活动早早结束、干等饭点"的空档，
         但**绝不把餐饮顶出窗口上限**（顶出则回退多加的部分）。
      2. 末段 slack 填充：剩余空闲按末段优先分配。
      3. dinner 超时压缩 + 末段超时压缩：兜底收口。
      下午活动太少（拉满仍有大空档）或太多（压不动仍超时）属于"语义层面排得不合理"，
      不在求解器内强行扭曲，交由 rule_validation + repair_loop 让模型重排。
    """
    eta = eta or {}
    plan_start = _hhmm_to_minutes(start_time)
    plan_end = _hhmm_to_minutes(end_time) if end_time else None

    n = len(steps)
    if n == 0:
        return []

    # ── 1. 初始：各段取区间下限，通勤取 ETA ──────────────────────────────────
    durations = []
    travels = []
    for i, step in enumerate(steps):
        lo, _hi = _tier_range(step)
        durations.append(lo)
        prev = steps[i - 1] if i > 0 else None
        travels.append(0 if i == 0 else _travel_to(prev, step, eta))

    # ── 2. 排布函数：活动驱动、餐饮自适应跟随 ──────────────────────────────
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
                # 紧跟 dinner 结束 + 通勤，不强制额外等待
                pass

            start = cursor
            end = start + durs[i]
            spans.append((start, end))
            cursor = end

        return spans

    # ── 3a. 餐前空档填充：消除"活动早结束、干等饭点"的空档 ──────────────────
    def _fill_pre_meal_gaps() -> list[tuple[int, int]]:
        spans = _layout(durations)
        for meal_idx, step in enumerate(steps):
            phase = step.get("phase")
            if phase not in ("lunch", "dinner") or meal_idx == 0:
                continue
            window = _LUNCH_WINDOW if phase == "lunch" else _DINNER_WINDOW

            # 餐饮已自然顺延到窗口下限之后（前序活动够长）→ 无需填充
            if spans[meal_idx][0] > window[0]:
                continue

            prev_end = spans[meal_idx - 1][1]
            transit = travels[meal_idx]
            gap = window[0] - (prev_end + transit)
            if gap <= 0:
                continue

            # 把空档分配给餐饮前面的活动（从紧邻的往前），不超过各自上限
            remaining = gap
            for i in range(meal_idx - 1, -1, -1):
                if steps[i].get("phase") in ("lunch", "dinner", "evening"):
                    continue
                lo, hi = _tier_range(steps[i])
                headroom = hi - durations[i]
                if headroom <= 0:
                    continue
                add = min(headroom, remaining)
                durations[i] += add
                remaining -= add
                if remaining <= 0:
                    break
            spans = _layout(durations)

            # 关键保护：填充后若把餐饮顶出窗口上限，回退多加的部分
            if spans[meal_idx][0] > window[1]:
                overshoot = spans[meal_idx][0] - window[1]
                for i in range(meal_idx - 1, -1, -1):
                    if steps[i].get("phase") in ("lunch", "dinner", "evening"):
                        continue
                    lo, _ = _tier_range(steps[i])
                    back = min(durations[i] - lo, overshoot)
                    if back > 0:
                        durations[i] -= back
                        overshoot -= back
                    if overshoot <= 0:
                        break
                spans = _layout(durations)
        return spans

    spans = _fill_pre_meal_gaps()

    # ── 3b. 末段 slack 填充：按 headroom 比例分配，活动与餐饮平等竞争 ───────────
    # 比例分配：每段能拿到的 slack 与自己的 headroom（上限-下限）成正比，
    # 避免倒序时末段（evening）把 slack 独吞，餐饮和前面的活动一无所获。
    # 这样 long 餐饮（自助/火锅）和 long 活动（电影/KTV）都能按比例拿到合理时长。
    if plan_end is not None:
        slack = plan_end - spans[-1][1]
        if slack > 0:
            headrooms = [max(0, _tier_range(steps[i])[1] - durations[i]) for i in range(n)]
            total_headroom = sum(headrooms)
            if total_headroom > 0:
                for i in range(n):
                    if headrooms[i] <= 0:
                        continue
                    # 按比例分配，向下取整；余数在后面补
                    add_i = int(slack * headrooms[i] / total_headroom)
                    add_i = min(add_i, headrooms[i])
                    if add_i > 0:
                        durations[i] += add_i
                # 分配完后把整除余数补给 headroom 最大的段
                spans = _layout(durations)
                remaining = plan_end - spans[-1][1]
                if remaining > 0:
                    best = max(range(n), key=lambda i: headrooms[i])
                    extra = min(remaining, headrooms[best] - (durations[best] - (int(slack * headrooms[best] / total_headroom) if total_headroom > 0 else 0)))
                    extra = min(remaining, _tier_range(steps[best])[1] - durations[best])
                    if extra > 0:
                        durations[best] += extra
            spans = _layout(durations)

            # 补充后检查：若相邻间隔被挤压到 < _MIN_TRANSITION_MINUTES，收缩该段
            changed = True
            while changed:
                changed = False
                for i in range(n - 1):
                    gap = spans[i + 1][0] - spans[i][1]
                    if gap < _MIN_TRANSITION_MINUTES and durations[i] > _tier_range(steps[i])[0]:
                        shrink = min(_MIN_TRANSITION_MINUTES - gap, durations[i] - _tier_range(steps[i])[0])
                        if shrink > 0:
                            durations[i] -= shrink
                            spans = _layout(durations)
                            changed = True
                            break

    # ── 4. dinner 超时压缩：dinner 超出窗口上限则压缩前序下午活动 ────────────
    dinner_idx = next((i for i, s in enumerate(steps) if s.get("phase") == "dinner"), None)
    if dinner_idx is not None and spans[dinner_idx][0] > _DINNER_WINDOW[1]:
        overshoot = spans[dinner_idx][0] - _DINNER_WINDOW[1]
        for i in range(dinner_idx - 1, -1, -1):
            if steps[i].get("phase") not in ("morning", "lunch", "afternoon"):
                continue
            lo, _ = _tier_range(steps[i])
            shrink = min(durations[i] - lo, overshoot)
            if shrink > 0:
                print(
                    f"[Candidate Planning Node] dinner超时压缩: "
                    f"{steps[i].get('label')} {durations[i]}min → {durations[i]-shrink}min"
                )
                durations[i] -= shrink
                overshoot -= shrink
            if overshoot <= 0:
                break
        spans = _layout(durations)
        # 压缩不动（活动已全在下限、overshoot 仍 > 0）→ 不强扭，
        # dinner 仍超窗，由 rule_validation 报 dinner_too_late 让模型减少下午活动

    # ── 5. 末段超时压缩 ──────────────────────────────────────────────────────
    _COMPRESS_THRESHOLD = 15
    if plan_end is not None and spans[-1][1] > plan_end + _COMPRESS_THRESHOLD:
        last_start = spans[-1][0]
        squeezed = max(1, plan_end - last_start)
        if squeezed != durations[-1]:
            print(
                f"[Candidate Planning Node] 时长压缩: {steps[-1].get('label')} "
                f"{durations[-1]}min → {squeezed}min（避免超出结束时间 {end_time}）"
            )
        durations[-1] = squeezed
        spans = _layout(durations)

    # ── 6. 写回 ─────────────────────────────────────────────────────────────
    result = []
    for i, step in enumerate(steps):
        step = dict(step)
        s, e = spans[i]
        step["start_time"] = _minutes_to_hhmm(s)
        step["end_time"] = _minutes_to_hhmm(e)
        step["duration_minutes"] = durations[i]
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
                wp_lines.append(f"- 用户想要「{w.get('waypoint_raw')}」{hint}，可用 POI：{w.get('name')}（id={w.get('id')}），请插入方案中合适位置，duration_tier=short，poi_type=activity")
        waypoint_section = "\n".join(wp_lines)
    else:
        waypoint_section = "无途径小需求"

    activity_style: list = plan.get("activity_style") or []
    explicit = "、".join(activity_style) if activity_style else "（无明确点名，由你根据场景自由推荐）"

    user_message = f"""\
# 用户原始需求（唯一权威，请逐字理解，下面所有结构化字段仅供参考）
「{raw_query}」

请基于以上原话，判断用户想做几件事、各在什么时段、哪些是明确点名的，然后规划。
不要预设要填满时间，段数由原话和时间窗自然决定。

## 用户明确点名的活动（硬约束，必须全部安排）
以下是从原话中识别出的明确点名活动，必须全部出现在 steps 中，不可遗漏：
- 点名活动：{explicit}
- 若原话中还有"再安排一个活动""另外去 X""然后想玩 Y"等表述，同样必须体现。
- **活动（poi_type=activity）和餐饮（poi_type=restaurant）是两类不同的 step，不能互相替代。**
  用户说"再安排一个活动"，必须对应一个 poi_type=activity 的 step，不能用晚餐代替。
- 这是硬约束，不是建议；候选方案中每个这样的活动都必须有对应的 step。

## 时间信息（用于排期计算）
- 日期：{plan.get("date_label") or "未指定"}
- 开始时间：{start_time}（第一个 step 从此时间开始）
- 结束时间：{plan.get("end_time") or "未指定"}
- 时间描述：{plan.get("time_phrase") or ""}

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

请生成 3 个候选方案。每个 step 给出 phase、poi_id、label、duration_tier（short/medium/long），不要填写任何具体时刻或分钟数。
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