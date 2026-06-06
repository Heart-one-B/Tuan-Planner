# src/graph/nodes/execution_node.py
from src.graph.state import AgentState
from src.agent.execution_agent import ExecutionAgent
# from src.utils.state_utils import _derive_plan_compat_from_state

def execution_node(state: AgentState) -> AgentState:
    """Execution Node: 事务性的一键预约、预订下单与亲友最终版本通知组装。
    合并了原：
      - execution_node (系统进行预订预约)
      - reject_node (中止、取消预订时的退回标记)
      - final_message_node (整理最终可转发给亲友的通知文本)
    """
    print("[Execution Node] 正在根据确认状态进行预订事务执行与最终收尾通知拼装...")
    errors = list(state.get("errors", []))

    # 1. 读取确认状态，做出分路下单执行
    user_confirmed = state.get("user_confirmed", False)

    # ── 修复：从 state["plan"] 取最终展示方案，而非整个 state ──
    # state["activities"] / state["restaurant"] 是 fact_gathering 阶段的全量候选结果，
    # 与经过评分筛选后写入 state["plan"] 的最终方案无关，直接使用 state 会导致
    # 反馈里显示的地点与规划卡片里展示的地点不一致。
    plan = state.get("plan") or {}
    if not plan:
        # 兜底：尝试从 final_plan_result.selected_candidate 取
        final_plan_result = state.get("final_plan_result") or {}
        plan = final_plan_result.get("selected_candidate") or {}

    if user_confirmed is True:
        try:
            result = ExecutionAgent().execute(plan)
            if result is None:
                result = {"status": "success", "message": "Execution completed"}
            execution_result = result
        except Exception as exc:
            print(f"[Execution Node][WARN] 下单组件抛出异常: {exc}")
            errors.append(f"Execution failed: {exc}")
            execution_result = {"status": "error", "message": str(exc)}
    else:
        # 取消 (原 reject_node)
        execution_result = {
            "status": "cancelled",
            "message": "用户未确认方案，未执行任何预约或下单动作。",
        }

    # 2. 组装格式化统一最终通知 (原 final_message_node)
    intent = state.get("intent") or {}
    status = execution_result.get("status")
    scenario = intent.get("scenario", "family")

    activity_name = "待确认活动"
    restaurant_name = "待确认餐厅"
    if isinstance(plan, dict):
        activities = plan.get("activities") or []
        if isinstance(activities, list) and activities and isinstance(activities[0], dict):
            activity_name = activities[0].get("name") or activity_name
        restaurant = plan.get("restaurant") or {}
        if isinstance(restaurant, dict):
            restaurant_name = restaurant.get("name") or restaurant_name

    if status == "success":
        orders = execution_result.get("orders") or []
        order_lines = []
        if isinstance(orders, list):
            for item in orders:
                if not isinstance(item, dict):
                    continue
                order_lines.append(f"- {item.get('type', 'order')}: {item.get('order_id', 'N/A')}")
        order_text = "\n".join(order_lines) if order_lines else "- 已完成关键预约"
        message = (
            f"搞定了。{scenario} 场景下的本次安排已经完成。\n"
            f"活动：{activity_name}\n"
            f"餐厅：{restaurant_name}\n"
            f"执行结果：\n{order_text}\n"
            f"如果你要，我也可以继续帮你整理成可直接发给家人/朋友的版本。"
        )
    elif status == "cancelled":
        message = f"你刚才选择了不执行当前方案。如果你想重新安排 {scenario} 场景的本地活动，我可以继续帮你重新规划。"
    elif status == "error":
        msg = execution_result.get("message") or "执行阶段发生异常。"
        message = f"本次执行没有完全完成：{msg}\n如果你愿意，我可以基于当前结果帮你重新规划一版更稳妥的方案。"
    else:
        if errors:
            message = f"当前方案暂时没有收敛到可执行结果。\n原因：{errors[-1]}"
        else:
            message = f"本次安排已走到执行后的结果整理阶段。当前状态：{status or 'unknown'}。"

    return {
        "execution_result": execution_result, "plan": plan, "final_message": message, "errors": errors,
    }