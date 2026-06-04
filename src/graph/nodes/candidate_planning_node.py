from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import AgentState
from src.model.factory import get_chat_model
from src.utils.state_utils import _append_error

# ──────────────────────────────────────────────────────────────────────────────
# 最小化验证版本：核心改动 = 把完整语境（用户原话）还给模型，不再用结构化字段架空它。
#
# 改动点：
#   1. System Prompt：删掉"满天行程"四步示例（morning→lunch→afternoon→dinner），
#      改成单步极简示例 + 明确声明"段数随需求和时间窗变化，留白优于硬塞"。
#   2. System Prompt：新增一条总则——以用户原始需求为唯一权威，结构化字段仅供参考。
#   3. System Prompt：afternoon 相位判断从"单端点 end_time>=15:00"改为"重叠判断"，
#      避免 18:00-22:00 这种纯晚间窗口凭空多出下午段。
#   4. user_message：raw_query 提到最顶、强措辞声明为权威需求；
#      删除 diet_preference 字段（它把"晚上喝点酒"压成全局"喝酒"标签污染所有时段）。
#      activity_style / must_avoid 保留（未被污染，是有效约束）。
#
# 不改：schema、意图节点输出、时间地板修正逻辑、池子结构。
# ──────────────────────────────────────────────────────────────────────────────

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
      "recommendation_mode": "preference_fit",
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
- "long"：约 3 小时（剧本杀、密室逃脱、深度局、大型场馆/乐园）
按"这是个轻松/常规/深度的活动"来选档，不要纠结具体分钟。

## 选址规则
- 只能使用候选池中提供的 poi_id，禁止编造
- 3 个方案的活动组合要尽量不同
- 相似活动类型视为同一类（剧本杀/密室逃脱/桌游属于同一类），同一方案内活动不重复
- 如果提供了历史偏好，plan_1 和 plan_2 的 recommendation_mode 设为 "preference_fit"，可以参考历史偏好生成稳妥方案；plan_3 的 recommendation_mode 设为 "exploration"，必须满足本轮用户原话和硬约束，但可以合理偏离历史偏好，提供新鲜感。
- 历史偏好只能启发候选生成，不能覆盖本轮用户原话，不能违反明确排除项、时间、天气、距离、安全等硬约束。

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

## 时段密度（同一时段安排几个活动）
判断每个时段适合安排几个活动：先大致估算各活动占用的时长档位（short≈1h / medium≈2h / long≈3h），
再看这个时段的可用时间能容纳几个（活动之间还需留通勤时间）。
例如下午 14:00-17:30 约 3.5 小时，可安排两个 medium 活动，或一个 long 活动。
宁可排得舒展，也不要硬塞到时间紧张；但明显有大段空白时应再加一个活动填充。
**你只决定"排几个、各是什么档位"，具体时刻由系统计算。**

## evening 场景规则
- family（有孩子）→ 不安排 evening，dinner 是终点
- couple / friends / team → evening 可正常安排

## 天气规则
- 雨/雷雨/大风 → 优先 environment=indoor
- 阴天/多云/晴 → 室内外均可，优先贴合用户原话

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

# 时长档位 → (下限, 上限) 分钟区间。代码按可用时间在区间内弹性取值。
_TIER_RANGE: dict[str, tuple[int, int]] = {
    "short":  (45, 75),
    "medium": (90, 150),
    "long":   (150, 210),
}
_DEFAULT_TIER = "medium"

# 常识 sanity 规则：各 phase 的最早开始时间（只拦离谱值，不硬拉到这个点）。
_PHASE_EARLIEST: dict[str, int] = {
    "lunch":   11 * 60 + 30,   # 午餐不早于 11:30
    "dinner":  17 * 60 + 30,   # 晚餐不早于 17:30
    "evening": 19 * 60 + 30,   # 夜间活动不早于 19:30
}
# 晚餐与夜间活动之间的消化/转场间隔（分钟）
_DINNER_TO_EVENING_GAP = (60, 120)


def _tier_range(step: dict) -> tuple[int, int]:
    tier = step.get("duration_tier")
    if tier not in _TIER_RANGE:
        # 兼容/兜底：餐厅默认 medium 偏短，活动默认 medium
        tier = _DEFAULT_TIER
    return _TIER_RANGE[tier]


def _travel_to(prev_step: dict | None, step: dict, eta: dict) -> int:
    """相邻两段之间的通勤分钟。优先用高德真实 ETA，拿不到用兜底值。

    说明：当前 eta 是 origin→POI 的单程时间。第一段（出发地→首个 POI）精确；
    后续段用目标 POI 的 origin-ETA 作为近似（点到点 ETA 需另算，见 TODO）。
    """
    if prev_step is None:
        # 第一段：出发地 → 首个 POI，直接用该 POI 的 origin-ETA
        rec = eta.get(step.get("poi_id") or "")
        if isinstance(rec, dict) and rec.get("eta_minutes") is not None:
            return int(rec["eta_minutes"])
        return _DEFAULT_TRAVEL
    # 后续段：用目标 POI 的 origin-ETA 近似（TODO: 改为 prev→curr 点到点 ETA）
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
    纯代码时刻求解器。模型只给"有序活动 + 档位 + phase"，这里负责算出所有时刻。

    流程：
      1. 每段时长先取档位区间下限，通勤用高德 ETA（拿不到用兜底）。
      2. 应用常识 sanity：午餐不早于 11:30、晚餐不早于 17:30、
         晚餐与其后的夜间活动至少留 60 分钟间隔。
      3. 若总时长未填满用户时间窗，按比例把各段在档位区间内弹性拉长，填充富余。
      4. 若超出 end_time，压缩最后一段（下限 _MIN_STEP_DURATION）；
         仍放不下交 rule_validation 的 overflow 校验兜底。
    模型不再输出 start_time，故此处不读取模型时刻。
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

    # ── 2. 先排一遍，应用 sanity 规则得到每段起始 ───────────────────────────
    def _layout(durs: list[int]) -> list[tuple[int, int]]:
        """给定各段时长，结合通勤与 sanity 规则，算出 [(start, end), ...]。"""
        spans: list[tuple[int, int]] = []
        cursor = plan_start
        prev_phase = ""
        prev_end = None
        for i, step in enumerate(steps):
            phase = step.get("phase") or "afternoon"
            cursor += travels[i]
            # sanity：phase 最早开始
            earliest = _PHASE_EARLIEST.get(phase)
            if earliest is not None and cursor < earliest:
                cursor = earliest
            # sanity：晚餐 → 夜间活动 至少留间隔
            if prev_phase == "dinner" and phase == "evening" and prev_end is not None:
                min_start = prev_end + _DINNER_TO_EVENING_GAP[0]
                if cursor < min_start:
                    cursor = min_start
            start = cursor
            end = start + durs[i]
            spans.append((start, end))
            cursor = end
            prev_phase = phase
            prev_end = end
        return spans

    spans = _layout(durations)

    # ── 3. 弹性填充：末段优先策略 ────────────────────────────────────────────
    # 优先拉长最后一段（在酒吧多待一会儿），而非均摊到所有段。
    # 均摊会让中间段结束时间后移，压缩后续段的通勤缓冲，触发 transition_too_rushed。
    # 末段拉长后若仍有剩余 slack，再反向逐段补充（倒数第二、第三…），
    # 但每段补充后都重新校验相邻间隔，保证通勤空间不被压垮。
    if plan_end is not None:
        slack = plan_end - spans[-1][1]
        if slack > 0:
            headrooms = []
            for step in steps:
                lo, hi = _tier_range(step)
                headrooms.append(hi - lo)

            # 从最后一段开始向前逐段消化 slack
            for i in range(n - 1, -1, -1):
                if slack <= 0:
                    break
                add_i = min(slack, headrooms[i])
                if add_i > 0:
                    durations[i] += add_i
                    slack -= add_i
            spans = _layout(durations)

            # 补充后检查：若有相邻 step 间隔 < _MIN_TRANSITION_MINUTES，
            # 说明某段被拉长后挤压了通勤，把该段时长收缩回来
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

    # ── 4. 超时压缩：末段超出 end_time 超过 15 分钟才压缩 ─────────────────────
    # 用户不是机器人，15 分钟内的超时可以自行提前结束，不需要系统强制压缩
    _COMPRESS_THRESHOLD = 15
    if plan_end is not None and spans[-1][1] > plan_end + _COMPRESS_THRESHOLD:
        last_start = spans[-1][0]
        # 超时压缩：直接用可用时间，不强制 _MIN_STEP_DURATION 下限
        # 宁可最后一段短一点，也不能让行程超出用户的结束时间
        squeezed = max(1, plan_end - last_start)
        if squeezed != durations[-1]:
            print(
                f"[Candidate Planning Node] 时长压缩: {steps[-1].get('label')} "
                f"{durations[-1]}min → {squeezed}min（避免超出结束时间 {end_time}）"
            )
        durations[-1] = squeezed
        spans = _layout(durations)

    # ── 5. 写回 ─────────────────────────────────────────────────────────────
    result = []
    for i, step in enumerate(steps):
        step = dict(step)
        s, e = spans[i]
        step["start_time"] = _minutes_to_hhmm(s)
        step["end_time"] = _minutes_to_hhmm(e)
        step["duration_minutes"] = durations[i]  # 回填实际换算时长，供下游展示
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
    Candidate Planning Node（最小化验证版）：
    以用户原话为主输入生成 3 个候选方案，再计算每个 step 的实际时刻。
    """
    print("[Candidate Planning Node] 基于真实候选池生成 3 个结构化候选方案...")

    plan          = state.get("plan_context") or {}
    facts         = state.get("fact_gathering_result") or {}
    errors        = list(state.get("errors") or [])
    start_time    = plan.get("start_time") or "10:00"
    replan_reason = state.get("replan_reason") or ""
    preference_profile = state.get("user_preference_profile") or ""
    if not isinstance(preference_profile, str):
        preference_profile = ""

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
        "has_preference_profile": bool(preference_profile.strip()),
    }, ensure_ascii=False, indent=2))
    print("=" * 108 + "\n")

    activities:  list[dict] = facts.get("activities") or []
    restaurants: list[dict] = facts.get("restaurants") or []
    weather:     dict       = facts.get("weather") or {}

    activities_by_id  = {a["id"]: a for a in activities  if isinstance(a, dict) and a.get("id")}
    restaurants_by_id = {r["id"]: r for r in restaurants if isinstance(r, dict) and r.get("id")}

    # 提取用户明确点名的活动，作为硬约束传给 planner
    activity_style: list = plan.get("activity_style") or []
    explicit = "、".join(activity_style) if activity_style else "（无明确点名，由你根据场景自由推荐）"
    activity_explicit_types: list = plan.get("activity_explicit_types") or []
    activity_explicit_search: dict = facts.get("activity_explicit_search") or {}
    explicit = "、".join(activity_explicit_types) if activity_explicit_types else "（无明确点名，由你根据场景自由推荐）"
    explicit_search_block = f"""\
## 显式活动搜索结果
{json.dumps(activity_explicit_search, ensure_ascii=False, indent=2)}

规划规则：
- 如果 matched 中有显式活动类型，候选方案不能全部忽略对应 POI。
- 如果 missing 中有显式活动类型，说明真实搜索没有结果，可以不安排该类型。
"""
    preference_block = ""
    if preference_profile.strip():
        preference_block = f"""\
## 历史偏好档案（只用于启发候选生成，不参与最终打分）
{preference_profile.strip()}

使用规则：
- 本轮用户原话优先级最高。
- plan_1 和 plan_2 可以参考历史偏好，生成稳妥方案。
- plan_3 必须是 exploration，可以合理偏离历史偏好，提供新鲜感。
- 历史偏好不能覆盖本轮明确需求，不能违反明确排除项、时间、天气、距离、安全等硬约束。
"""

    # ── user_message：raw_query 置顶且声明为权威；删除 diet_preference 污染字段 ──
    user_message = f"""\
# 用户原始需求（唯一权威，请逐字理解，下面所有结构化字段仅供参考）
「{raw_query}」

请基于以上原话，判断用户想做几件事、各在什么时段、哪些是明确点名的，然后规划。
不要预设要填满时间，段数由原话和时间窗自然决定。

{preference_block}
{explicit_search_block}

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
        "explicit_activity_type": a.get("explicit_activity_type"),
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

            if not in_activities and not in_restaurants:
                # poi_id 在两个池子都找不到，才是真幻觉
                print(f"[Candidate Planning Node][WARN] plan_{i} 幻觉 poi_id={poi_id!r}，已丢弃")
                continue

            # 自动修正 poi_type：模型给的类型和实际池子归属不一致时纠偏
            # 常见场景：酒吧被模型标为 restaurant，但实际在 activities 池（高德 typecode 080304）
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
        index_result = _build_step_index(timed_steps, activities_by_id, restaurants_by_id)
        recommendation_mode = raw.get("recommendation_mode")
        if recommendation_mode not in {"preference_fit", "exploration"}:
            recommendation_mode = "exploration" if i == 3 else "preference_fit"
        normalized.append({
            "id":        raw.get("id") or f"plan_{i}",
            "title":     raw.get("title") or f"候选方案 {i}",
            "recommendation_mode": recommendation_mode,
            "steps":     timed_steps,
            "reasoning": [r for r in (raw.get("reasoning") or []) if isinstance(r, str)][:3],
            **index_result,
        })

    while len(normalized) < 3:
        normalized.append({
            "id": f"plan_{len(normalized)+1}", "title": "", "steps": [], "timeline": [],
            "recommendation_mode": "exploration" if len(normalized) == 2 else "preference_fit",
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
