# src/graph/nodes/confirmation_node.py
from src.graph.state import AgentState

def confirmation_node(state: AgentState) -> AgentState:
    """确认节点：在终端阻塞等待用户输入以确认是否按照当前排版方案执行一键下单。"""
    if "user_confirmed" in state:
        return {"user_confirmed": state["user_confirmed"]}
    confirm = input("\n[系统提示] 确定按照此方案执行一键下单吗？(y/n): ")
    confirmed = confirm.lower() == "y"
    if confirmed:
        return {"user_confirmed": True}
    return {
        "user_confirmed": False,
        "replan_reason": "用户未确认当前方案",
        "replan_reason_type": "user_feedback",
    }