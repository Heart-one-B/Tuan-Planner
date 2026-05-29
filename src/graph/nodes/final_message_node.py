# src/graph/nodes/final_message_node.py
from src.graph.state import AgentState
from src.utils.state_utils import _derive_plan_compat_from_state

def final_message_node(state: AgentState) -> AgentState:
    """Final Message Node：把执行结果转成可转发的最终消息。"""
    execution_result = state.get("execution_result") or {}
    plan = _derive_plan_compat_from_state(state)
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
                order_type = item.get("type", "order")
                order_id = item.get("order_id", "N/A")
                order_lines.append(f"- {order_type}: {order_id}")
        order_text = "\n".join(order_lines) if order_lines else "- 已完成关键预约"
        message = (
            f"搞定了。{scenario} 场景下的本次安排已经完成。\n"
            f"活动：{activity_name}\n"
            f"餐厅：{restaurant_name}\n"
            f"执行结果：\n{order_text}\n"
            f"如果你要，我也可以继续帮你整理成可直接发给家人/朋友的版本。"
        )
    elif status == "cancelled":
        message = (
            f"你刚才选择了不执行当前方案。"
            f"如果你想重新安排 {scenario} 场景的本地活动，我可以继续帮你重新规划。"
        )
    elif status == "error":
        msg = execution_result.get("message") or "执行阶段发生异常。"
        message = (
            f"本次执行没有完全完成：{msg}\n"
            f"如果你愿意，我可以基于当前结果帮你重新规划一版更稳妥的方案。"
        )
    else:
        errors = state.get("errors") or []
        if isinstance(errors, list) and errors:
            message = (
                "当前方案暂时没有收敛到可执行结果。\n"
                f"原因：{errors[-1]}"
            )
        else:
            message = (
                f"本次安排已走到执行后的结果整理阶段。"
                f"当前状态：{status or 'unknown'}。"
            )

    return {"final_message": message}