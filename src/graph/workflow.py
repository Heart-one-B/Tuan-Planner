from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.nodes.clarification_node import clarification_node
from src.graph.nodes.orchestrator_node import orchestrator_node
from src.graph.nodes.evaluation_node import evaluation_node
from src.graph.nodes.presentation_node import presentation_node
from src.graph.nodes.chat_node import chat_node
from src.graph.state import AgentState
from src.graph.nodes.routing_node import routing_node
from src.graph.nodes.intent_node import intent_node
from src.graph.nodes.constraint_node import constraint_node
from src.graph.nodes.fact_node import fact_node
from src.graph.nodes.planning_node import planning_node
from src.graph.routers.main_router import (
    route_after_intent,
    route_after_feedback,
    route_after_fact,
    route_after_clarification,
    route_after_evaluation, route_after_orchestrator,  # 新增
)


def build_workflow(checkpointer=None):
    graph = StateGraph(AgentState)

    # 节点注册
    graph.add_node("routing",        routing_node)
    graph.add_node("intent",         intent_node)
    graph.add_node("constraint",     constraint_node)
    graph.add_node("fact",           fact_node)
    graph.add_node("planning",       planning_node)
    graph.add_node("evaluation",     evaluation_node)
    graph.add_node("orchestrator",   orchestrator_node)
    graph.add_node("clarification",  clarification_node)
    graph.add_node("presentation",   presentation_node)
    graph.add_node("chat",           chat_node)

    # 入口
    graph.set_entry_point("routing")

    # routing 分流
    graph.add_conditional_edges(
        "routing",
        route_after_feedback,
        {
            "intent":       "intent",
            "orchestrator": "orchestrator",
            "chat":         "chat",   # 接上了
        },
    )

    # intent 分流
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "constraint":    "constraint",
            "clarification": "clarification",
            "chat":          "chat",
            "end":           END,
        },
    )

    # 主链路
    graph.add_edge("constraint", "fact")

    # fact 分流
    graph.add_conditional_edges(
        "fact",
        route_after_fact,
        {
            "planning": "planning",
            "end":      END,
        },
    )

    # clarification 跑完后的分流
    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "constraint": "constraint",
            "end":        END,
        },
    )

    graph.add_edge("planning", "evaluation")

    # evaluation 分流（新增）：有方案 → presentation，无方案 → END
    graph.add_conditional_edges(
        "evaluation",
        route_after_evaluation,
        {
            "presentation": "presentation",
            "end":          END,
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

    graph.add_edge("presentation",  END)
    graph.add_edge("chat",          END)

    return graph.compile(checkpointer=checkpointer)


workflow = build_workflow()