from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    confirmation_node,
    execution_node,
    intent_node,
    planning_node,
    presentation_node,
    reject_node,
    route_after_confirmation,
)
from src.graph.state import AgentState


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("planning", planning_node)
    graph.add_node("presentation", presentation_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("execution", execution_node)
    graph.add_node("reject", reject_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "planning")
    graph.add_edge("planning", "presentation")
    graph.add_edge("presentation", "confirmation")
    graph.add_conditional_edges(
        "confirmation",
        route_after_confirmation,
        {
            "execute": "execution",
            "reject": "reject",
        },
    )
    graph.add_edge("execution", END)
    graph.add_edge("reject", END)

    return graph.compile()


app = build_workflow()

