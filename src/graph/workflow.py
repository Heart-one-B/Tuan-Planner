# src/graph/workflow.py
from langgraph.graph import END, StateGraph

from src.graph.nodes.intent_node import intent_node
from src.graph.nodes.clarification_node import clarification_node, receive_clarification_node
from src.graph.nodes.llm_answer_node import llm_answer_node
from src.graph.nodes.constraint_build_node import constraint_build_node
from src.graph.nodes.fact_gathering_node import fact_gathering_node
from src.graph.nodes.candidate_planning_node import candidate_planning_node
from src.graph.nodes.plan_poi_detail_sync_node import plan_poi_detail_sync_node
from src.graph.nodes.rule_validation_node import rule_validation_node
from src.graph.nodes.repair_loop_node import repair_loop_node
from src.graph.nodes.scoring_node import scoring_node
from src.graph.nodes.final_plan_node import final_plan_node
from src.graph.nodes.presentation_node import presentation_node
from src.graph.nodes.confirmation_node import confirmation_node
from src.graph.nodes.execution_node import execution_node
from src.graph.nodes.final_message_node import final_message_node
from src.graph.nodes.feedback_router_node import feedback_router_node
from src.graph.nodes.adjust_conflict_node import adjust_conflict_node

from src.graph.routers import route_after_confirmation
from src.graph.state import AgentState

_MAX_CLARIFICATION_ROUNDS = 5


def route_after_feedback(state: AgentState) -> str:
    """
    feedback_router 之后的条件路由。

    new_plan  → intent（重走完整规划流程）
    adjust    → constraint_build（增量合并反馈 + 增量搜索）
    chat      → llm_answer（直接回复，不修改方案）
    """
    route = state.get("feedback_route") or "new_plan"
    if route == "adjust":
        return "constraint_build"
    if route == "chat":
        return "llm_answer"
    return "intent"  # new_plan 或兜底


def route_after_intent_with_clarification(state: AgentState) -> str:
    intent = state.get("intent")
    if not isinstance(intent, dict):
        return "constraint_build"
    if intent.get("is_leisure_planning") is False:
        return "llm_answer"
    if intent.get("clarification_needed") is True:
        clarification_round = state.get("clarification_round", 0)
        if isinstance(clarification_round, bool) or not isinstance(clarification_round, int):
            clarification_round = 0
        if clarification_round >= _MAX_CLARIFICATION_ROUNDS:
            print(f"[Workflow] 达到追问上限轮次 ({_MAX_CLARIFICATION_ROUNDS})，转向直答。")
            return "llm_answer"
        return "clarification"
    return "constraint_build"


def route_after_rule_validation(state: AgentState) -> str:
    """
    校验后分流：
    - adjust 模式：用户指定的局部修改，不重新打分（打分可能违背用户意愿，
      比如把用户点名的鸡公煲又换成评分更高的别的店）。
        校验通过（>=1 合法）→ final_plan（直接选，final_plan 会兜底取 valid_plans[0]）
        校验失败 → repair_loop（修改导致物理冲突，尝试修复）
    - 正常模式：valid_plans >= 3 → scoring，否则 repair_loop
    """
    result = state.get("rule_validation_result")
    valid_plans = result.get("valid_plans") if isinstance(result, dict) else None
    valid_count = len(valid_plans) if isinstance(valid_plans, list) else 0

    if state.get("feedback_route") == "adjust":
        return "final_plan" if valid_count >= 1 else "adjust_conflict"

    if valid_count >= 3:
        return "scoring"
    return "repair_loop"


def route_after_repair_loop_new(state: AgentState) -> str:
    count = state.get("replan_count", 0)
    if isinstance(count, bool) or not isinstance(count, int):
        count = 0
    if count >= 3:
        return "final_plan"
    return "candidate_planning"


def build_workflow(checkpointer=None):
    graph = StateGraph(AgentState)

    # === 注册节点 ===
    graph.add_node("feedback_router", feedback_router_node)   # 入口节点
    graph.add_node("intent", intent_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("receive_clarification", receive_clarification_node)
    graph.add_node("llm_answer", llm_answer_node)
    graph.add_node("constraint_build", constraint_build_node)
    graph.add_node("fact_gathering", fact_gathering_node)
    graph.add_node("candidate_planning", candidate_planning_node)
    graph.add_node("plan_poi_detail_sync", plan_poi_detail_sync_node)
    graph.add_node("rule_validation", rule_validation_node)
    graph.add_node("repair_loop", repair_loop_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("final_plan", final_plan_node)
    graph.add_node("presentation", presentation_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("execution", execution_node)
    graph.add_node("final_message", final_message_node)
    graph.add_node("adjust_conflict", adjust_conflict_node)

    # === 入口 ===
    graph.set_entry_point("feedback_router")

    # feedback_router 分流
    graph.add_conditional_edges(
        "feedback_router",
        route_after_feedback,
        {
            "intent":           "intent",
            "constraint_build": "constraint_build",
            "llm_answer":       "llm_answer",
        },
    )

    # intent 分流
    graph.add_conditional_edges(
        "intent",
        route_after_intent_with_clarification,
        {
            "llm_answer":       "llm_answer",
            "clarification":    "clarification",
            "constraint_build": "constraint_build",
        },
    )

    # clarification 闭环
    graph.add_edge("clarification", "receive_clarification")
    graph.add_edge("receive_clarification", "intent")

    # 直答出口
    graph.add_edge("llm_answer", END)

    # 规划主链路
    graph.add_edge("constraint_build", "fact_gathering")
    graph.add_edge("fact_gathering", "candidate_planning")
    graph.add_edge("candidate_planning", "plan_poi_detail_sync")
    graph.add_edge("plan_poi_detail_sync", "rule_validation")

    graph.add_conditional_edges(
        "rule_validation",
        route_after_rule_validation,
        {
            "scoring":         "scoring",
            "repair_loop":     "repair_loop",
            "final_plan":      "final_plan",      # adjust 校验通过，跳过 scoring 直接选方案
            "adjust_conflict": "adjust_conflict", # adjust 校验失败，提示用户冲突
        },
    )

    graph.add_conditional_edges(
        "repair_loop",
        route_after_repair_loop_new,
        {"candidate_planning": "candidate_planning", "final_plan": "final_plan"},
    )

    graph.add_edge("scoring", "final_plan")
    graph.add_edge("final_plan", "presentation")
    graph.add_edge("presentation", "confirmation")

    graph.add_conditional_edges(
        "confirmation",
        route_after_confirmation,
        {
            "execute":            "execution",
            "replan":             "repair_loop",
            "await_confirmation": END,
        },
    )

    graph.add_edge("execution", "final_message")
    graph.add_edge("final_message", END)
    graph.add_edge("adjust_conflict", END)

    # interrupt_after：
    # - clarification：等待用户回答追问
    # - confirmation：等待用户确认方案
    # feedback_router 不需要 interrupt，它在每次 invoke 开头执行，
    # main.py 的多轮循环负责重新 invoke 并注入新的 user_input
    return graph.compile(
        interrupt_after=["clarification", "confirmation"],
        checkpointer=checkpointer,
    )


app = build_workflow()