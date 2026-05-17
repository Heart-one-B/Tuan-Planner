from langchain_core.messages import HumanMessage

from src.agent.constraint_agent import ConstraintAgent
from src.agent.execution_agent import ExecutionAgent
from src.agent.intent_agent import IntentAgent
from src.agent.planning_agent import PlanningAgent
from src.agent.presentation_agent import PresentationAgent
from src.agent.retrieval_agent import RetrievalAgent
from src.graph.state import AgentState
from src.model.factory import chat_model
from src.tools.mock_api import MockToolAPI


_LLM_ANSWER_FALLBACK = (
    "该问题不属于本地生活规划范畴，建议直接咨询通用助手或搜索引擎。"
)


def _append_error(state: AgentState, error: str) -> AgentState:
    errors = list(state.get("errors", []))
    errors.append(error)
    return {"errors": errors}


def intent_node(state: AgentState) -> AgentState:
    try:
        intent = IntentAgent().parse(state["user_input"])
        return {"intent": intent}
    except Exception as exc:
        return _append_error(state, f"Intent node failed: {exc}")


def llm_answer_node(state: AgentState) -> AgentState:
    """非规划任务直答节点：让 LLM 直接回答用户输入。失败时写入固定 fallback 文案。"""
    user_input = state.get("user_input", "")
    print("[LLM Answer Node] 检测到非本地生活规划任务，调用 LLM 直接回答...")
    try:
        response = chat_model.invoke([HumanMessage(content=user_input)])
        answer = getattr(response, "content", None)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("LLM 返回为空或不是字符串")
        print(f"[OK] LLM 直答完成，长度={len(answer)}")
        return {"llm_answer": answer}
    except Exception as exc:
        print(f"[WARN] LLM 直答失败，使用固定 fallback: {exc}")
        update = _append_error(state, f"LLM Answer node failed: {exc}")
        update["llm_answer"] = _LLM_ANSWER_FALLBACK
        return update


# DEPRECATED (T8 起替换为 plan_candidate_node)：保留实现以便向后兼容旧链路 / 测试。
# 不再被 workflow.py 主图连入，移除前请确认无外部调用方依赖。
def planning_node(state: AgentState) -> AgentState:
    try:
        plan = PlanningAgent().plan(state.get("intent", {}))
        return {"plan": plan}
    except Exception as exc:
        return _append_error(state, f"Planning node failed: {exc}")


def plan_candidate_node(state: AgentState) -> AgentState:
    """Plan Candidate Node：消费 6 个并行节点的输出，合成 primary + backup。

    设计：
        * 完全不调用 LLM / MockToolAPI；纯函数代理 PlanningAgent().compose。
        * 同时写入：
              - state.candidates = 完整 {primary, backup} 结构
              - state.plan = candidates["primary"]，向后兼容旧
                PresentationAgent / ExecutionAgent。
        * 防御：
              - 任意上游字段缺失 / 非法 → 由 compose 内部按空安全默认处理；
              - compose 返回 {} → state.candidates={}, state.plan={}，不写错误；
              - compose 抛异常 → 兜底 catch，写 errors。
    """
    print("[Plan Candidate Node] 合成 primary + backup 候选方案...")
    try:
        candidates = PlanningAgent().compose(
            constraints=state.get("constraints") or {},
            weather=state.get("weather") or {},
            activities=state.get("activities") or [],
            restaurants=state.get("restaurants") or [],
            traffic=state.get("traffic") or {},
            queue=state.get("queue") or {},
            crowd=state.get("crowd") or {},
        )
        if not isinstance(candidates, dict):
            candidates = {}
        primary = candidates.get("primary") if candidates else None
        if isinstance(primary, dict):
            # state.plan = candidates["primary"] 的同时，补齐旧 PresentationAgent /
            # ExecutionAgent 期望的 activities 列表键，避免破坏向后兼容。
            plan_compat = dict(primary)
            activity = primary.get("activity")
            plan_compat["activities"] = [activity] if isinstance(activity, dict) else []
        else:
            plan_compat = {}
        return {"candidates": candidates, "plan": plan_compat}
    except Exception as exc:
        print(f"[Plan Candidate Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Plan Candidate node failed: {exc}")


def retrieval_node(state: AgentState) -> AgentState:
    """Retrieval Node：基于 intent 做 mock RAG 检索，写入 ``state.retrieval_context``。

    任意异常都被吞掉、写入 ``state.errors``，避免阻塞主链路。RetrievalAgent 自身
    设计为"任意失败返回空骨架"，因此正常路径不会抛；本 try 仅是兜底护栏。
    """
    print("[Retrieval Node] 开始 mock RAG 检索...")
    try:
        result = RetrievalAgent().retrieve(state.get("intent", {}))
        return {"retrieval_context": result}
    except Exception as exc:
        print(f"[Retrieval Node][WARN] 检索节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Retrieval node failed: {exc}")


def constraint_collect_node(state: AgentState) -> AgentState:
    """Constraint Collect Node：把 intent + retrieval_context + replan_reason
    汇总为 ``state.constraints``，作为后续 6 个并行工具节点的统一输入。

    设计：
        * ConstraintAgent 是纯函数（不调用 LLM）。
        * 任意异常都被吞掉，写入 ``state.errors``，**不**写 ``constraints``，
          让下游节点以"缺少 constraints"路径继续兜底，不阻塞图执行。
    """
    print("[Constraint Collect Node] 汇总意图 / 检索 / 默认策略 / 重规划反馈...")
    try:
        constraints = ConstraintAgent().collect(
            state.get("intent", {}),
            state.get("retrieval_context", {}),
            state.get("replan_reason", ""),
        )
        print(
            f"[Constraint Collect Node][OK] scenario={constraints.get('scenario')}, "
            f"party={constraints.get('party')}, "
            f"max_traffic_minutes={constraints.get('max_traffic_minutes')}, "
            f"max_queue_minutes={constraints.get('max_queue_minutes')}, "
            f"replan_hints_n={len(constraints.get('replan_hints', []))}, "
            f"retrieval_pois_n={len(constraints.get('retrieval_pois', []))}, "
            f"retrieval_notes_n={len(constraints.get('retrieval_notes', []))}"
        )
        return {"constraints": constraints}
    except Exception as exc:
        print(f"[Constraint Collect Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Constraint Collect node failed: {exc}")


# ---------------------------------------------------------------------------
# T7：Constraint Collect 之后的 6 个并行工具节点
# ---------------------------------------------------------------------------
#
# 设计原则：
#   * 6 个节点之间互不读对方写入的字段，保持真正并行；
#   * 仅依赖 ``state.constraints``（缺失时使用安全默认）或 mock 数据全集；
#   * 任意异常都被吞掉，写入 ``state.errors``，不阻塞主链路；
#   * 每节点最多一两条 print，避免噪音。
#
# 这一阶段 PlanningAgent 仍然内部独立调工具（暂时桥接），下一任务 T8 会用
# plan_candidate_node 替换它，让 planning 节点直接消费这里写入的 6 个字段。


def _safe_constraints(state: AgentState) -> dict:
    constraints = state.get("constraints")
    return constraints if isinstance(constraints, dict) else {}


def _infer_queue_time_slot(time_window: str) -> str:
    """time_window → estimate_restaurant_queue 所需 slot。

    规则：
        * 含 "evening" → "dinner"；
        * 其它（含 "morning" / 缺省 / 非字符串） → "lunch"。
    """
    if isinstance(time_window, str) and "evening" in time_window:
        return "dinner"
    return "lunch"


def _infer_crowd_time_slot(time_window: str) -> str:
    """time_window → evaluate_crowd_risk 所需 slot。

    支持三种 mock 表内键：weekend_morning / weekend_evening / weekday_evening。
    其它输入回退 "weekend_morning"，与 DEFAULT_POLICY 默认一致。
    """
    valid = {"weekend_morning", "weekend_evening", "weekday_evening"}
    if isinstance(time_window, str) and time_window in valid:
        return time_window
    return "weekend_morning"


def weather_check_node(state: AgentState) -> AgentState:
    """并行节点 1：查询天气，写 ``state.weather``。"""
    print("[Weather Check Node] 查询天气...")
    try:
        constraints = _safe_constraints(state)
        scenario_key = constraints.get("weather_scenario") or "default"
        weather = MockToolAPI().get_weather(scenario_key)
        return {"weather": weather}
    except Exception as exc:
        print(f"[Weather Check Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Weather Check node failed: {exc}")


def activity_search_node(state: AgentState) -> AgentState:
    """并行节点 2：根据场景搜活动，写 ``state.activities``。"""
    print("[Activity Search Node] 搜索候选活动...")
    try:
        constraints = _safe_constraints(state)
        scenario = constraints.get("scenario") or "family"
        activities = MockToolAPI().search_activities(scenario)
        if not isinstance(activities, list):
            activities = []
        return {"activities": activities}
    except Exception as exc:
        print(f"[Activity Search Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Activity Search node failed: {exc}")


def restaurant_search_node(state: AgentState) -> AgentState:
    """并行节点 3：根据饮食偏好搜餐厅，写 ``state.restaurants``。"""
    print("[Restaurant Search Node] 搜索候选餐厅...")
    try:
        constraints = _safe_constraints(state)
        diet_preference = constraints.get("diet_preference") or ""
        restaurants = MockToolAPI().search_restaurants(diet_preference)
        if not isinstance(restaurants, list):
            restaurants = []
        return {"restaurants": restaurants}
    except Exception as exc:
        print(f"[Restaurant Search Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Restaurant Search node failed: {exc}")


def traffic_eta_node(state: AgentState) -> AgentState:
    """并行节点 4：批量查通勤 ETA，写 ``state.traffic``。

    为保持与其它并行节点真正独立（不读 state.activities / state.restaurants），
    直接遍历 mock_db 中的活动 (family+friends) 和餐厅全集 id 做 ETA 查询。
    """
    print("[Traffic ETA Node] 批量查询通勤 ETA...")
    try:
        api = MockToolAPI()
        activity_ids: list[str] = []
        activities_by_scenario = api.db.get("activities", {}) or {}
        for scenario_key in ("family", "friends"):
            for item in activities_by_scenario.get(scenario_key, []) or []:
                if isinstance(item, dict) and item.get("id"):
                    activity_ids.append(item["id"])
        restaurant_ids = [
            item["id"]
            for item in (api.db.get("restaurants", []) or [])
            if isinstance(item, dict) and item.get("id")
        ]
        all_targets = activity_ids + restaurant_ids

        eta_by_target: dict[str, dict] = {}
        for target_id in all_targets:
            record = api.get_traffic_eta("area_central", target_id)
            eta_by_target[target_id] = {
                "eta_minutes": record.get("eta_minutes"),
                "congestion": record.get("congestion"),
                "fallback_hint": record.get("fallback_hint"),
            }
        return {"traffic": {"eta_by_target": eta_by_target}}
    except Exception as exc:
        print(f"[Traffic ETA Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Traffic ETA node failed: {exc}")


def queue_check_node(state: AgentState) -> AgentState:
    """并行节点 5：批量查餐厅排队，写 ``state.queue``。"""
    print("[Queue Check Node] 批量查询餐厅排队...")
    try:
        constraints = _safe_constraints(state)
        time_window = constraints.get("time_window") or ""
        time_slot = _infer_queue_time_slot(time_window)

        api = MockToolAPI()
        restaurant_ids = [
            item["id"]
            for item in (api.db.get("restaurants", []) or [])
            if isinstance(item, dict) and item.get("id")
        ]

        wait_by_restaurant: dict[str, dict] = {}
        for rid in restaurant_ids:
            record = api.estimate_restaurant_queue(rid, time_slot)
            wait_by_restaurant[rid] = {
                "wait_minutes": record.get("wait_minutes"),
                "party_acceptable": record.get("party_acceptable"),
                "fallback_hint": record.get("fallback_hint"),
            }
        return {"queue": {"wait_by_restaurant": wait_by_restaurant}}
    except Exception as exc:
        print(f"[Queue Check Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Queue Check node failed: {exc}")


def crowd_risk_node(state: AgentState) -> AgentState:
    """并行节点 6：批量评估活动人流风险，写 ``state.crowd``。"""
    print("[Crowd Risk Node] 批量评估活动人流风险...")
    try:
        constraints = _safe_constraints(state)
        time_window = constraints.get("time_window") or ""
        time_slot = _infer_crowd_time_slot(time_window)

        api = MockToolAPI()
        activity_ids: list[str] = []
        activities_by_scenario = api.db.get("activities", {}) or {}
        for scenario_key in ("family", "friends"):
            for item in activities_by_scenario.get(scenario_key, []) or []:
                if isinstance(item, dict) and item.get("id"):
                    activity_ids.append(item["id"])

        crowd_by_activity: dict[str, dict] = {}
        for aid in activity_ids:
            record = api.evaluate_crowd_risk(aid, time_slot)
            crowd_by_activity[aid] = {
                "risk_level": record.get("risk_level"),
                "fallback_hint": record.get("fallback_hint"),
            }
        return {"crowd": {"crowd_by_activity": crowd_by_activity}}
    except Exception as exc:
        print(f"[Crowd Risk Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Crowd Risk node failed: {exc}")


def validate_plan_node(state: AgentState) -> AgentState:
    """Validate Plan Node：纯校验，不调 LLM。

    输入：``state.candidates.primary`` 与 ``state.constraints`` /
    ``state.traffic`` / ``state.queue`` / ``state.weather``。
    输出：``state.validation_result = {"passed": bool, "violations": [str],
    "suggested_fixes": [str]}``，**任何分支都必须写入** validation_result。

    校验项（命中即追加一条人类可读 violation 和一条对应 suggested_fix）：
        1. 若 primary.activity 非空：traffic.eta_by_target[id].eta_minutes
           ≤ constraints.max_traffic_minutes
        2. 若 primary.restaurant 非空：queue.wait_by_restaurant[id].wait_minutes
           ≤ constraints.max_queue_minutes
        3. weather.risk_level ∈ {"High","high"} 时 activity.type 必须为 "indoor"
        4. queue.wait_by_restaurant[id].party_acceptable 不能为 False
        * primary 为空 / 非 dict → 直接 passed=False，violations=["主方案为空"]

    异常时追加 errors，但 validation_result 仍以 passed=False 写入。
    """
    print("[Validate Plan Node] 校验主方案...")
    violations: list[str] = []
    suggested_fixes: list[str] = []
    try:
        candidates = state.get("candidates")
        primary = candidates.get("primary") if isinstance(candidates, dict) else None

        if not isinstance(primary, dict) or not primary:
            violations.append("主方案为空")
            suggested_fixes.append("重新生成候选方案或放宽约束")
        else:
            constraints = state.get("constraints") or {}
            constraints = constraints if isinstance(constraints, dict) else {}
            traffic = state.get("traffic") or {}
            traffic = traffic if isinstance(traffic, dict) else {}
            queue = state.get("queue") or {}
            queue = queue if isinstance(queue, dict) else {}
            weather = state.get("weather") or {}
            weather = weather if isinstance(weather, dict) else {}

            max_traffic = constraints.get("max_traffic_minutes")
            max_queue = constraints.get("max_queue_minutes")

            def _is_num(v):
                return isinstance(v, (int, float)) and not isinstance(v, bool)

            activity = primary.get("activity")
            if isinstance(activity, dict) and activity:
                # 1) ETA 阈值
                aid = activity.get("id")
                eta_table = traffic.get("eta_by_target")
                eta_record = eta_table.get(aid) if isinstance(eta_table, dict) else None
                eta = eta_record.get("eta_minutes") if isinstance(eta_record, dict) else None
                if _is_num(eta) and _is_num(max_traffic) and eta > max_traffic:
                    violations.append(
                        f"首选活动 ETA {eta} 分钟超过阈值 {max_traffic} 分钟"
                    )
                    suggested_fixes.append("放宽 max_traffic_minutes 或更换更近活动")

                # 3) 天气 risk_level vs activity.type
                risk_level = weather.get("risk_level")
                if isinstance(risk_level, str) and risk_level in {"High", "high"}:
                    act_type = activity.get("type")
                    if act_type != "indoor":
                        violations.append(
                            f"天气风险等级 {risk_level} 与首选活动类型 {act_type!r} 冲突，应为 indoor"
                        )
                        suggested_fixes.append("更换 indoor 类型活动或调整时间窗口")

            restaurant = primary.get("restaurant")
            if isinstance(restaurant, dict) and restaurant:
                rid = restaurant.get("id")
                wait_table = queue.get("wait_by_restaurant")
                wait_record = wait_table.get(rid) if isinstance(wait_table, dict) else None
                if isinstance(wait_record, dict):
                    wait_minutes = wait_record.get("wait_minutes")
                    # 2) 排队阈值
                    if _is_num(wait_minutes) and _is_num(max_queue) and wait_minutes > max_queue:
                        violations.append(
                            f"首选餐厅排队 {wait_minutes} 分钟超过阈值 {max_queue} 分钟"
                        )
                        suggested_fixes.append("放宽 max_queue_minutes 或更换排队更短餐厅")
                    # 4) party_acceptable
                    if wait_record.get("party_acceptable") is False:
                        violations.append("首选餐厅不接受当前人数")
                        suggested_fixes.append("更换适合该人数的餐厅")

        passed = len(violations) == 0
        validation_result = {
            "passed": passed,
            "violations": violations,
            "suggested_fixes": suggested_fixes,
        }
        print(
            f"[Validate Plan Node][OK] passed={passed}, "
            f"violations_n={len(violations)}"
        )
        return {"validation_result": validation_result}
    except Exception as exc:
        print(f"[Validate Plan Node][WARN] 节点异常，记录错误并强制 passed=False: {exc}")
        errors = list(state.get("errors", []))
        errors.append(f"Validate Plan node failed: {exc}")
        if not violations:
            violations = [f"校验异常: {exc}"]
            suggested_fixes = ["请检查上游节点输出"]
        return {
            "errors": errors,
            "validation_result": {
                "passed": False,
                "violations": violations,
                "suggested_fixes": suggested_fixes,
            },
        }


def route_after_validate(state: AgentState) -> str:
    """Validate 之后的条件路由。

    * validation_result.passed 显式为 True → ``presentation``；
    * 其它（缺失 / 非 dict / passed 非 True） → ``replan``，由上游图把 ``replan``
      映射到合适的下游节点（T9 暂占位映射到 presentation，T10 改为真正的 replan 节点）。
    """
    result = state.get("validation_result")
    if isinstance(result, dict) and result.get("passed") is True:
        return "presentation"
    return "replan"


# ---------------------------------------------------------------------------
# T10：Replan Node + 路由
# ---------------------------------------------------------------------------
#
# 设计要点：
#   * 纯函数，不调 LLM。从 ``state.replan_reason`` 或 ``validation_result.violations``
#     取一段人类可读的失败原因，按关键字 → 约束变更映射收紧 constraints；
#   * 通过浅拷贝写新 dict / list，绝不 in-place 修改入参；
#   * ``replan_count`` 自增；命中上限（>= 3）时不再触发新一轮，写降级说明到
#     ``state.errors``，由路由把后续走向 ``final_message``（T15 占位 → presentation）。
#   * 节点输出**必须**把 ``replan_reason`` 重置为空串，避免下一轮 validate 失败时
#     旧 reason 累积污染；
#   * 异常时仅追加 errors，不修改 constraints / replan_count / replan_reason，
#     让上层评估到底是节点出错还是约束没法再收紧。

# 关键字 → 约束变更的安全默认：当 current 不是合法数值时回退此基线，
# 与 ConstraintAgent 默认 max_traffic_minutes / max_queue_minutes 保持一致。
_REPLAN_TRAFFIC_FALLBACK_BASE = 60
_REPLAN_QUEUE_FALLBACK_BASE = 60
_REPLAN_TRAFFIC_FLOOR = 20
_REPLAN_QUEUE_FLOOR = 10
_REPLAN_BUDGET = 3


def _is_pos_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _resolve_replan_reason(state: AgentState) -> str:
    """优先 state.replan_reason，否则把 validation_result.violations 拼成一段。"""
    reason = state.get("replan_reason")
    if isinstance(reason, str) and reason.strip():
        return reason
    result = state.get("validation_result")
    if isinstance(result, dict):
        violations = result.get("violations")
        if isinstance(violations, list) and violations:
            parts = [v for v in violations if isinstance(v, str) and v.strip()]
            if parts:
                return "; ".join(parts)
    return ""


def replan_node(state: AgentState) -> AgentState:
    """Replan Node：把失败原因或拒绝反馈结构化成新约束，准备回到 plan_candidate。

    输出字段（命中即写入）：
        * constraints       —— 浅拷贝并按关键字收紧 max_traffic_minutes /
                                max_queue_minutes / indoor_preferred；append replan_hints
        * replan_count      —— +1
        * replan_reason     —— 重置为 ""
        * errors            —— 命中预算上限或异常时追加
    """
    print("[Replan Node] 把失败原因 / 拒绝反馈映射成新约束...")
    try:
        prev_count = state.get("replan_count", 0)
        if not isinstance(prev_count, int) or isinstance(prev_count, bool):
            prev_count = 0
        new_count = prev_count + 1

        reason = _resolve_replan_reason(state)

        # 浅拷贝 constraints / replan_hints，避免污染入参
        old_constraints = state.get("constraints")
        if not isinstance(old_constraints, dict):
            old_constraints = {}
        new_constraints = dict(old_constraints)
        old_hints = new_constraints.get("replan_hints")
        if not isinstance(old_hints, list):
            old_hints = []
        new_hints = list(old_hints)

        if reason:
            # 1) ETA / 通勤 → 收紧 max_traffic_minutes
            if ("ETA" in reason) or ("通勤" in reason):
                cur = new_constraints.get("max_traffic_minutes")
                base = cur if _is_pos_num(cur) else _REPLAN_TRAFFIC_FALLBACK_BASE
                new_constraints["max_traffic_minutes"] = max(
                    _REPLAN_TRAFFIC_FLOOR, int(base) - 10
                )
            # 2) 排队 → 收紧 max_queue_minutes
            if "排队" in reason:
                cur = new_constraints.get("max_queue_minutes")
                base = cur if _is_pos_num(cur) else _REPLAN_QUEUE_FALLBACK_BASE
                new_constraints["max_queue_minutes"] = max(
                    _REPLAN_QUEUE_FLOOR, int(base) - 10
                )
            # 3) 天气 / 户外 / outdoor → 强制室内偏好
            if ("天气" in reason) or ("户外" in reason) or ("outdoor" in reason):
                new_constraints["indoor_preferred"] = True
            # 4) 不论是否命中关键字，都把 reason 加进 hints 历史，便于追溯
            new_hints.append(reason)

        new_constraints["replan_hints"] = new_hints

        update: AgentState = {
            "constraints": new_constraints,
            "replan_count": new_count,
            "replan_reason": "",
        }

        if new_count >= _REPLAN_BUDGET:
            errors = list(state.get("errors", []))
            errors.append(
                f"已尝试重规划 {new_count} 次，仍无法满足全部约束，进入兜底展示"
            )
            update["errors"] = errors
            print(
                f"[Replan Node][WARN] replan_count={new_count} 命中上限 "
                f"{_REPLAN_BUDGET}，进入兜底"
            )
        else:
            print(
                f"[Replan Node][OK] replan_count={new_count}, "
                f"max_traffic_minutes={new_constraints.get('max_traffic_minutes')}, "
                f"max_queue_minutes={new_constraints.get('max_queue_minutes')}, "
                f"indoor_preferred={new_constraints.get('indoor_preferred')}, "
                f"replan_hints_n={len(new_hints)}"
            )

        return update
    except Exception as exc:
        print(f"[Replan Node][WARN] 节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Replan node failed: {exc}")


def route_after_replan(state: AgentState) -> str:
    """Replan 之后的条件路由。

    * replan_count >= 3 → ``final_message``（T15 之前由 workflow 占位映射到
      presentation，避免阻塞图编译）；
    * 否则 → ``constraint_collect``，由 6 个并行节点 + plan_candidate + validate
      重新执行（这是 LangGraph 中复跑前置并行节点最直接的做法）。
    """
    count = state.get("replan_count", 0)
    if isinstance(count, bool) or not isinstance(count, int):
        count = 0
    if count >= _REPLAN_BUDGET:
        return "final_message"
    return "constraint_collect"



def presentation_node(state: AgentState) -> AgentState:
    try:
        display_text = PresentationAgent().generate_plan_display(
            state.get("plan", {}),
            state.get("intent", {}),
        )
        print("\n" + "=" * 20 + " 方案详情 " + "=" * 20)
        print(display_text)
        print("=" * 50)
        return {"display_text": display_text}
    except Exception as exc:
        return _append_error(state, f"Presentation node failed: {exc}")


def confirmation_node(state: AgentState) -> AgentState:
    if "user_confirmed" in state:
        return {"user_confirmed": state["user_confirmed"]}
    confirm = input("\n[系统提示] 确定按照此方案执行一键下单吗？(y/n): ")
    return {"user_confirmed": confirm.lower() == "y"}


def execution_node(state: AgentState) -> AgentState:
    try:
        result = ExecutionAgent().execute(state.get("plan", {}))
        if result is None:
            result = {"status": "success", "message": "Execution completed"}
        return {"execution_result": result}
    except Exception as exc:
        return _append_error(state, f"Execution node failed: {exc}")


def reject_node(state: AgentState) -> AgentState:
    return {
        "execution_result": {
            "status": "cancelled",
            "message": "用户未确认方案，未执行任何预约或下单动作。",
        }
    }


def route_after_confirmation(state: AgentState) -> str:
    if state.get("user_confirmed"):
        return "execute"
    return "reject"


def route_after_intent(state: AgentState) -> str:
    """Intent 节点之后的条件路由。

    判定逻辑：
        * 仅当 ``intent.is_leisure_planning`` 显式为 False 时，路由到直答 ``llm_answer``。
        * 字段缺失 / intent 缺失 / 任意异常 → 默认走 ``planning``，避免误把规划任务路由到直答。
    """
    intent = state.get("intent") or {}
    if not isinstance(intent, dict):
        return "planning"
    if intent.get("is_leisure_planning") is False:
        return "llm_answer"
    return "planning"


def route_after_intent_for_retrieval(state: AgentState) -> str:
    """Intent 节点之后的条件路由（T5 起启用，含 retrieval 分支）。

    判定逻辑：
        * intent 缺失 / 非 dict → 保守走 ``planning``（与旧路由保持一致的兜底策略）；
        * ``is_leisure_planning is False`` → ``llm_answer``；
        * ``is_leisure_planning is True`` 且 ``need_retrieval is True`` → ``retrieval``；
        * 其它（含 ``need_retrieval`` 缺失 / 非 True） → ``planning``，
          这样默认家庭场景不会被多绕一次检索，主链路速度不变。
    """
    intent = state.get("intent")
    if not isinstance(intent, dict):
        return "planning"
    if intent.get("is_leisure_planning") is False:
        return "llm_answer"
    if intent.get("is_leisure_planning") is True and intent.get("need_retrieval") is True:
        return "retrieval"
    return "planning"
