from src.graph.state import AgentState


def confirmation_node(state: AgentState) -> AgentState:
    """
    Confirmation Node：纯状态机，不含任何 I/O。

    图在此节点后 interrupt，等待外部把用户回复注入 state["user_reply"]。
    外部通过 Command(resume=None, update={"user_reply": "y/n/用户反馈"}) 恢复。
    """
    # 读取外部注入的回复
    user_reply: str = (state.get("user_reply") or "").strip().lower()

    if not user_reply:
        # 首次到达，还没有用户回复，图在此 interrupt
        return {"pending_confirmation": True}

    # 有回复了：解析用户意图
    confirmed = user_reply in {"y", "yes", "是", "确认", "好", "ok", "确定"}

    if confirmed:
        return {
            "user_confirmed":      True,
            "pending_confirmation": False,
            "user_reply":          "",
        }
    else:
        # 用户拒绝或提出修改意见，把原文作为重规划原因
        replan_reason = (
            user_reply
            if user_reply not in {"n", "no", "否", "不", "不要", "取消"}
            else "用户未确认当前方案，希望重新规划"
        )
        return {
            "user_confirmed":      False,
            "pending_confirmation": False,
            "user_reply":          "",
            "replan_reason":       replan_reason,
            "replan_reason_type":  "user_feedback",
        }