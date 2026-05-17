"""Planning Agent：候选方案合成器（T8 起启用）。

设计要点
--------
* 主入口 :py:meth:`PlanningAgent.compose` 是 **纯函数**：不调 LLM、不调
  ``MockToolAPI``、不读外部状态。同输入恒等输出，便于单测。
* 旧入口 :py:meth:`PlanningAgent.plan` 改为薄薄的兼容 shim：内部一次性查全
  6 类数据，调用 ``compose``，再把结果适配为旧返回结构
  （``activities`` / ``restaurant`` / ``exceptions_handled``），
  让旧 PresentationAgent / ExecutionAgent 保持可用。
* 选择策略遵循 T8 约定：
    1. 活动：天气 ``risk_level``/``risk`` 命中 {"High","high"} 或
       ``constraints.indoor_preferred`` 为 True → 仅保留 ``type=="indoor"``；
       按 ``traffic.eta_by_target[id].eta_minutes > max_traffic_minutes`` 过滤；
       按 ``crowd.crowd_by_activity[id].risk_level`` 升序
       （low<medium<high<unknown）。
    2. 餐厅：按
       ``queue.wait_by_restaurant[id].wait_minutes > max_queue_minutes`` 过滤；
       按 (eta_minutes asc, wait_minutes asc) 升序。
* 防御：候选列表全为空时直接返回 ``{}``，不抛异常。
"""

from __future__ import annotations

from typing import Any

from src.tools.mock_api import MockToolAPI


# crowd risk_level → 排序键。越小越优先。未知 / 缺失统一归为 "unknown"。
_CROWD_RANK = {"low": 0, "medium": 1, "high": 2, "unknown": 3}

# 默认阈值（与 ConstraintAgent.DEFAULT_POLICY 对齐，作为兜底）。
_DEFAULT_MAX_TRAFFIC = 40
_DEFAULT_MAX_QUEUE = 30

# 天气切 indoor 的 risk 命中集合（兼容大小写）。
_WEATHER_HIGH = {"High", "high"}


# ---------------------------------------------------------------------------
# 内部小工具
# ---------------------------------------------------------------------------

def _safe_dict(value):
    return value if isinstance(value, dict) else {}


def _safe_list(value):
    return value if isinstance(value, list) else []


def _weather_high(weather):
    """读取天气风险等级；优先 ``risk_level``，回退 ``risk``。"""
    level = weather.get("risk_level")
    if not isinstance(level, str) or not level:
        level = weather.get("risk")
    return isinstance(level, str) and level in _WEATHER_HIGH


def _eta_minutes(traffic, target_id):
    record = _safe_dict(traffic.get("eta_by_target")).get(target_id)
    if not isinstance(record, dict):
        return None
    return record.get("eta_minutes")


def _wait_minutes(queue, target_id):
    record = _safe_dict(queue.get("wait_by_restaurant")).get(target_id)
    if not isinstance(record, dict):
        return None
    return record.get("wait_minutes")


def _crowd_level(crowd, target_id):
    record = _safe_dict(crowd.get("crowd_by_activity")).get(target_id)
    if not isinstance(record, dict):
        return "unknown"
    level = record.get("risk_level")
    if not isinstance(level, str) or not level:
        return "unknown"
    return level


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_filtered_by_traffic(eta, threshold):
    return _is_num(eta) and eta > threshold


def _is_filtered_by_queue(wait, threshold):
    return _is_num(wait) and wait > threshold


def _ordered_unique(items):
    """保持首次出现顺序去重（仅保留非空字符串）。"""
    seen = set()
    out = []
    for s in items:
        if not isinstance(s, str) or not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _format_timeline(activity, restaurant, constraints):
    activity_name = (activity or {}).get("name") or "待确认活动"
    restaurant_name = (restaurant or {}).get("name") or "待确认餐厅"
    constraints = _safe_dict(constraints)

    time_phrase = constraints.get("time_phrase")
    if not isinstance(time_phrase, str) or not time_phrase.strip():
        time_phrase = "待确认时间"

    start_time = constraints.get("start_time")
    if isinstance(start_time, str) and start_time.strip():
        return f"{time_phrase} {start_time.strip()} 出发：{activity_name} → {restaurant_name}"
    return f"{time_phrase}：{activity_name} → {restaurant_name}"


def _derive_time_window_from_intent_time(time_info):
    time_info = _safe_dict(time_info)
    explicit_time_window = time_info.get("time_window")
    if isinstance(explicit_time_window, str) and explicit_time_window.strip():
        return explicit_time_window.strip()

    time_phrase = time_info.get("time_phrase")
    if not isinstance(time_phrase, str):
        time_phrase = ""
    is_weekend = any(token in time_phrase for token in ("周末", "周六", "周日", "周天"))
    if "晚上" in time_phrase or "今晚" in time_phrase:
        return "weekend_evening" if is_weekend else "today_evening"
    if "下午" in time_phrase or "中午" in time_phrase or "白天" in time_phrase:
        return "weekend_afternoon" if is_weekend else "today_afternoon"
    return ""


def _gather_fallbacks(weather, traffic, queue, crowd):
    hints = []
    weather_hint = weather.get("fallback_hint")
    if isinstance(weather_hint, str):
        hints.append(weather_hint)
    for record in _safe_dict(traffic.get("eta_by_target")).values():
        if isinstance(record, dict):
            hint = record.get("fallback_hint")
            if isinstance(hint, str):
                hints.append(hint)
    for record in _safe_dict(queue.get("wait_by_restaurant")).values():
        if isinstance(record, dict):
            hint = record.get("fallback_hint")
            if isinstance(hint, str):
                hints.append(hint)
    for record in _safe_dict(crowd.get("crowd_by_activity")).values():
        if isinstance(record, dict):
            hint = record.get("fallback_hint")
            if isinstance(hint, str):
                hints.append(hint)
    return _ordered_unique(hints)


# ---------------------------------------------------------------------------
# 主选择逻辑
# ---------------------------------------------------------------------------

def _select_activities(*, activities, constraints, weather, traffic, crowd):
    """返回 (primary, backup, exceptions_for_activity)。"""
    flags = []
    if not activities:
        return None, None, flags

    indoor_required = bool(constraints.get("indoor_preferred")) or _weather_high(weather)

    # 仅当确实切换且原列表存在被过滤掉的非 indoor 项时，记一笔天气切换说明
    if indoor_required:
        had_outdoor = any(
            isinstance(a, dict) and a.get("type") != "indoor" for a in activities
        )
        if had_outdoor:
            flags.append("因天气切换 indoor")

    indoor_filtered = []
    for act in activities:
        if not isinstance(act, dict):
            continue
        if indoor_required and act.get("type") != "indoor":
            continue
        indoor_filtered.append(act)

    max_traffic = constraints.get("max_traffic_minutes")
    if (
        not isinstance(max_traffic, int)
        or isinstance(max_traffic, bool)
        or max_traffic <= 0
    ):
        max_traffic = _DEFAULT_MAX_TRAFFIC

    eligible = []
    eta_filter_dropped_first = False
    first_after_indoor = indoor_filtered[0] if indoor_filtered else None
    for act in indoor_filtered:
        eta = _eta_minutes(traffic, act.get("id"))
        if _is_filtered_by_traffic(eta, max_traffic):
            if act is first_after_indoor:
                eta_filter_dropped_first = True
            continue
        eligible.append(act)

    if eta_filter_dropped_first and eligible:
        flags.append("因活动 ETA 切换备选")

    if not eligible:
        return None, None, flags

    indexed = list(enumerate(eligible))
    indexed.sort(
        key=lambda pair: (
            _CROWD_RANK.get(_crowd_level(crowd, pair[1].get("id")), _CROWD_RANK["unknown"]),
            pair[0],
        )
    )
    sorted_eligible = [pair[1] for pair in indexed]

    primary = sorted_eligible[0] if sorted_eligible else None
    backup = sorted_eligible[1] if len(sorted_eligible) >= 2 else None
    return primary, backup, flags


def _select_restaurants(*, restaurants, constraints, queue, traffic):
    """返回 (primary, backup, exceptions_for_restaurant)。"""
    flags = []
    if not restaurants:
        return None, None, flags

    max_queue = constraints.get("max_queue_minutes")
    if (
        not isinstance(max_queue, int)
        or isinstance(max_queue, bool)
        or max_queue <= 0
    ):
        max_queue = _DEFAULT_MAX_QUEUE

    eligible = []
    queue_dropped_first = False
    first_dict = next((r for r in restaurants if isinstance(r, dict)), None)
    for r in restaurants:
        if not isinstance(r, dict):
            continue
        wait = _wait_minutes(queue, r.get("id"))
        if _is_filtered_by_queue(wait, max_queue):
            if r is first_dict:
                queue_dropped_first = True
            continue
        eligible.append(r)

    if queue_dropped_first and eligible:
        flags.append("因餐厅排队切换备选")

    if not eligible:
        return None, None, flags

    INF = float("inf")

    def _sort_key(item):
        eta = _eta_minutes(traffic, item.get("id"))
        wait = _wait_minutes(queue, item.get("id"))
        eta_key = eta if _is_num(eta) else INF
        wait_key = wait if _is_num(wait) else INF
        return (eta_key, wait_key)

    indexed = list(enumerate(eligible))
    indexed.sort(key=lambda pair: (_sort_key(pair[1]), pair[0]))
    sorted_eligible = [pair[1] for pair in indexed]

    primary = sorted_eligible[0] if sorted_eligible else None
    backup = sorted_eligible[1] if len(sorted_eligible) >= 2 else None
    return primary, backup, flags


# ---------------------------------------------------------------------------
# PlanningAgent
# ---------------------------------------------------------------------------

class PlanningAgent:
    """Plan Candidate 合成器：纯函数 ``compose`` + 兼容 shim ``plan``。"""

    def __init__(self):
        # 仅供 ``plan`` shim 内部一次性查工具数据使用；compose 不会用到。
        self._tools_lazy = None

    # ---- 主入口：纯函数 ---------------------------------------------------
    def compose(
        self,
        *,
        constraints,
        weather,
        activities,
        restaurants,
        traffic,
        queue,
        crowd,
    ):
        """根据 6 个并行节点的输出合成 primary + backup 两套候选。

        参数全部 keyword-only，全部允许 None / 缺键 / 错类型，按"安全默认"路径
        处理；候选为空时返回 ``{}``，绝不抛异常。
        """
        constraints = _safe_dict(constraints)
        weather = _safe_dict(weather)
        activities = _safe_list(activities)
        restaurants = _safe_list(restaurants)
        traffic = _safe_dict(traffic)
        queue = _safe_dict(queue)
        crowd = _safe_dict(crowd)

        act_primary, act_backup, act_flags = _select_activities(
            activities=activities,
            constraints=constraints,
            weather=weather,
            traffic=traffic,
            crowd=crowd,
        )
        res_primary, res_backup, res_flags = _select_restaurants(
            restaurants=restaurants,
            constraints=constraints,
            queue=queue,
            traffic=traffic,
        )

        # 兜底：双侧都空 → 返回 {}
        if act_primary is None and res_primary is None:
            return {}

        exceptions_handled = []
        # 顺序：天气切 indoor → 餐厅排队切备选 → 活动 ETA 切备选
        for label in ("因天气切换 indoor", "因活动 ETA 切换备选"):
            if label in act_flags and label not in exceptions_handled:
                exceptions_handled.append(label)
        for label in ("因餐厅排队切换备选",):
            if label in res_flags and label not in exceptions_handled:
                exceptions_handled.append(label)

        fallbacks = _gather_fallbacks(weather, traffic, queue, crowd)

        primary = {
            "activity": act_primary,
            "restaurant": res_primary,
            "weather": weather,
            "time_window": constraints.get("time_window"),
            "time_phrase": constraints.get("time_phrase"),
            "start_time": constraints.get("start_time"),
            "duration_hours": constraints.get("duration_hours"),
            "exceptions_handled": exceptions_handled,
            "fallbacks": fallbacks,
            "timeline": _format_timeline(act_primary, res_primary, constraints),
        }
        backup = {
            "activity": act_backup,
            "restaurant": res_backup,
            "time_window": constraints.get("time_window"),
            "time_phrase": constraints.get("time_phrase"),
            "start_time": constraints.get("start_time"),
            "duration_hours": constraints.get("duration_hours"),
            "exceptions_handled": [],
            "timeline": _format_timeline(act_backup, res_backup, constraints),
        }
        return {"primary": primary, "backup": backup}

    # ---- 兼容 shim：旧 plan(intent) -------------------------------------
    def plan(self, intent):
        """旧入口：保留方法签名，内部一次性查工具数据后委托给 ``compose``。

        返回结构与历史一致：
            {"activities": [primary.activity], "restaurant": primary.restaurant,
             "exceptions_handled": [...]}
        以便老的 PresentationAgent / ExecutionAgent 不需修改。
        """
        intent = intent if isinstance(intent, dict) else {}
        print("\n[Planning Agent] (shim) 一次性查询全数据并调用 compose...")

        if self._tools_lazy is None:
            self._tools_lazy = MockToolAPI()
        api = self._tools_lazy

        scenario = intent.get("scenario") if isinstance(intent.get("scenario"), str) else "family"
        diet = intent.get("diet_preference")
        if isinstance(diet, str):
            diet_pref = diet
        elif isinstance(diet, list):
            diet_pref = " ".join(str(item) for item in diet)
        else:
            diet_pref = ""
        time_info = intent.get("time")
        time_window = (
            intent.get("time_window")
            if isinstance(intent.get("time_window"), str)
            else _derive_time_window_from_intent_time(time_info)
        )
        time_info = _safe_dict(time_info)
        time_phrase = time_info.get("time_phrase")
        if not isinstance(time_phrase, str):
            time_phrase = ""
        start_time = time_info.get("start_time_hint")
        if not isinstance(start_time, str):
            start_time = ""
        duration_hours = time_info.get("duration_hours_hint")
        if isinstance(duration_hours, bool) or not isinstance(duration_hours, int) or duration_hours <= 0:
            duration_hours = None
        origin_area = (
            intent.get("origin_area")
            if isinstance(intent.get("origin_area"), str) and intent.get("origin_area")
            else "area_central"
        )

        # 仿照 nodes.py 中并行节点的策略
        weather = api.get_weather("default")
        activities_list = api.search_activities(scenario)
        if not isinstance(activities_list, list):
            activities_list = []
        restaurants_list = api.search_restaurants(diet_pref)
        if not isinstance(restaurants_list, list):
            restaurants_list = []

        # traffic：遍历可见 activity / restaurant id
        all_targets = []
        for a in activities_list:
            if isinstance(a, dict) and a.get("id"):
                all_targets.append(a["id"])
        for r in restaurants_list:
            if isinstance(r, dict) and r.get("id"):
                all_targets.append(r["id"])
        eta_by_target = {}
        for tid in all_targets:
            rec = api.get_traffic_eta(origin_area, tid)
            eta_by_target[tid] = {
                "eta_minutes": rec.get("eta_minutes"),
                "congestion": rec.get("congestion"),
                "fallback_hint": rec.get("fallback_hint"),
            }
        traffic = {"eta_by_target": eta_by_target}

        # queue：默认 lunch（与 _infer_queue_time_slot 默认一致）
        queue_slot = "dinner" if time_window in {"today_evening", "weekend_evening"} else "lunch"
        wait_by_restaurant = {}
        for r in restaurants_list:
            if isinstance(r, dict) and r.get("id"):
                rec = api.estimate_restaurant_queue(r["id"], queue_slot)
                wait_by_restaurant[r["id"]] = {
                    "wait_minutes": rec.get("wait_minutes"),
                    "party_acceptable": rec.get("party_acceptable"),
                    "fallback_hint": rec.get("fallback_hint"),
                }
        queue = {"wait_by_restaurant": wait_by_restaurant}

        # crowd：默认 weekend_morning
        crowd_slot_mapping = {
            "today_afternoon": "weekend_morning",
            "weekend_afternoon": "weekend_morning",
            "today_evening": "weekday_evening",
            "weekend_evening": "weekend_evening",
        }
        crowd_slot = crowd_slot_mapping.get(time_window, "weekend_morning")
        crowd_by_activity = {}
        for a in activities_list:
            if isinstance(a, dict) and a.get("id"):
                rec = api.evaluate_crowd_risk(a["id"], crowd_slot)
                crowd_by_activity[a["id"]] = {
                    "risk_level": rec.get("risk_level"),
                    "fallback_hint": rec.get("fallback_hint"),
                }
        crowd = {"crowd_by_activity": crowd_by_activity}

        constraints = {
            "scenario": scenario,
            "diet_preference": diet_pref,
            "time_window": time_window,
            "time_phrase": time_phrase,
            "start_time": start_time,
            "duration_hours": duration_hours,
            "origin_area": origin_area,
            "max_traffic_minutes": _DEFAULT_MAX_TRAFFIC,
            "max_queue_minutes": _DEFAULT_MAX_QUEUE,
            "indoor_preferred": False,
        }

        candidates = self.compose(
            constraints=constraints,
            weather=weather,
            activities=activities_list,
            restaurants=restaurants_list,
            traffic=traffic,
            queue=queue,
            crowd=crowd,
        )
        if not candidates:
            return {"activities": [], "restaurant": None, "exceptions_handled": []}

        primary = candidates.get("primary") or {}
        activity = primary.get("activity")
        restaurant = primary.get("restaurant")
        return {
            "activities": [activity] if activity else [],
            "restaurant": restaurant,
            "exceptions_handled": list(primary.get("exceptions_handled") or []),
        }
