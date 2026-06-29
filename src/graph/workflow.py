from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.nodes.orchestrator_node import orchestrator_node
from src.graph.nodes.evaluation_node import evaluation_node
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
)


def build_workflow(checkpointer=None):
    graph = StateGraph(AgentState)

    # 节点注册
    graph.add_node("routing",      routing_node)
    graph.add_node("intent",       intent_node)
    graph.add_node("constraint",   constraint_node)
    graph.add_node("fact",         fact_node)
    graph.add_node("planning",     planning_node)
    graph.add_node("evaluation",   evaluation_node)
    graph.add_node("orchestrator", orchestrator_node)

    # 入口
    graph.set_entry_point("routing")

    # routing 分流
    graph.add_conditional_edges(
        "routing",
        route_after_feedback,
        {
            "intent":       "intent",
            "orchestrator": "orchestrator",  # adjust 路径接入 Orchestrator
            "chat":         END,             # 阶段6再接
        },
    )

    # intent 分流
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "constraint":    "constraint",
            "clarification": END,   # 阶段6再接
            "chat":          END,
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

    graph.add_edge("planning",     "evaluation")
    graph.add_edge("evaluation",   END)
    graph.add_edge("orchestrator", END)

    return graph.compile(checkpointer=checkpointer)


workflow = build_workflow()