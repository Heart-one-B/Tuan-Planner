# src/graph/workflow.py
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.nodes.chat_node import chat_node
from src.graph.nodes.clarification_node import clarification_node
from src.graph.nodes.constraint_node import constraint_node
from src.graph.nodes.evaluation_node import evaluation_node
from src.graph.nodes.fact_node import fact_node
from src.graph.nodes.intent_node import intent_node
from src.graph.nodes.orchestrator_node import orchestrator_node
from src.graph.nodes.planning_node import planning_node
from src.graph.nodes.presentation_node import presentation_node
from src.graph.nodes.routing_node import routing_node
from src.graph.state import AgentState
from src.graph.routers.main_router import (
    route_after_clarification,
    route_after_constraint,
    route_after_evaluation,
    route_after_fact,
    route_after_feedback,
    route_after_intent,
    route_after_orchestrator,
)


def build_workflow(checkpointer=None):
    """图的形状。

    【本次改动：adjust 路径也过 intent + constraint】

        原来   routing --adjust--> orchestrator
        现在   routing --adjust--> intent -> constraint --> orchestrator

    原因见 route_after_feedback 的说明：跳过 intent/constraint 意味着
    plan_context 永远停留在第一轮的理解上，用户后续说的每一条约束
    都只是 replan 的一句 hint，evaluation 仍然拿旧标准打分。

    分流点从 routing 之后挪到 constraint 之后（route_after_constraint）：
    adjust 和 new_plan 在"要不要重新理解需求"上是一样的（都要），
    只在"要不要从头收集事实"上不同（adjust 复用现有候选池，
    由 orchestrator 决定是否补搜）。把分流点放在两者真正开始
    分岔的地方，而不是更早。
    """
    graph = StateGraph(AgentState)

    graph.add_node("routing", routing_node)
    graph.add_node("intent", intent_node)
    graph.add_node("constraint", constraint_node)
    graph.add_node("fact", fact_node)
    graph.add_node("planning", planning_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("presentation", presentation_node)
    graph.add_node("chat", chat_node)

    graph.set_entry_point("routing")

    # routing 分流：new_plan / clarify_reply / adjust 都先去 intent
    graph.add_conditional_edges(
        "routing",
        route_after_feedback,
        {
            "intent": "intent",
            "chat": "chat",
        },
    )

    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "constraint": "constraint",
            "clarification": "clarification",
            "chat": "chat",
            "end": END,
        },
    )

    # constraint 之后才分岔：调整走 orchestrator（复用候选池），
    # 其余走 fact（从头收集）
    graph.add_conditional_edges(
        "constraint",
        route_after_constraint,
        {
            "fact": "fact",
            "orchestrator": "orchestrator",
        },
    )

    graph.add_conditional_edges(
        "fact",
        route_after_fact,
        {
            "planning": "planning",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "constraint": "constraint",
            "end": END,
        },
    )

    graph.add_edge("planning", "evaluation")

    graph.add_conditional_edges(
        "evaluation",
        route_after_evaluation,
        {
            "presentation": "presentation",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "presentation": "presentation",
            "end": END,
        },
    )

    graph.add_edge("presentation", END)
    graph.add_edge("chat", END)

    return graph.compile(checkpointer=checkpointer)


workflow = build_workflow()