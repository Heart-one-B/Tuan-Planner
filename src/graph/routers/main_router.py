from __future__ import annotations

from src.graph.state import AgentState


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent") or {}
    errors = state.get("errors") or []

    # intent 节点失败
    if any(e.get("node") == "intent" for e in errors):
        return "end"

    if not intent.get("is_leisure_planning"):
        return "chat"

    if intent.get("clarification_needed"):
        return "clarification"

    return "constraint"


def route_after_feedback(state: AgentState) -> str:
    route = state.get("feedback_route") or "new_plan"
    if route == "adjust":
        return "orchestrator"
    if route == "chat":
        return "chat"
    # new_plan 和 clarify_reply 都走这里，统一重新解析意图
    return "intent"


def route_after_fact(state: AgentState) -> str:
    agent_outputs = state.get("agent_outputs") or {}
    fact = agent_outputs.get("fact") or {}

    if fact.get("status") == "error":
        return "end"
    return "planning"


def route_after_clarification(state: AgentState) -> str:
    """
    clarification_node 跑完后的路由:
    - pending_clarification 非空 → 还在等用户回答，中断到 END（等待下一轮输入）
    - pending_clarification 为空（要么强制兜底完成，要么本不该来这里）→ 进入 constraint
    """
    pending = state.get("pending_clarification") or ""
    if pending:
        return "end"
    return "constraint"

def route_after_evaluation(state: AgentState) -> str:
    """
    evaluation_node 跑完后的路由:
    - status=ok 且有选中方案 → presentation 渲染展示
    - status=error（全部不合格）→ END，告知用户
    """
    agent_outputs = state.get("agent_outputs") or {}
    eval_output = agent_outputs.get("evaluation") or {}

    if eval_output.get("status") == "ok" and (eval_output.get("data") or {}).get("selected_plan_id"):
        return "presentation"
    return "end"

def route_after_orchestrator(state: AgentState) -> str:
    """
    orchestrator_node 跑完后的路由:
    - 产出了新的合格方案 → presentation 重新渲染展示
    - 没有产出新方案(调度失败/异常/未触发replan) → END
    """
    if state.get("has_new_plan"):
        return "presentation"
    return "end"