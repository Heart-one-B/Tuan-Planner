from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    activity_search_node,
    confirmation_node,
    constraint_collect_node,
    crowd_risk_node,
    execution_node,
    intent_node,
    llm_answer_node,
    plan_candidate_node,
    presentation_node,
    queue_check_node,
    reject_node,
    replan_node,
    restaurant_search_node,
    retrieval_node,
    route_after_confirmation,
    route_after_intent_for_retrieval,
    route_after_replan,
    route_after_validate,
    traffic_eta_node,
    validate_plan_node,
    weather_check_node,
)
from src.graph.state import AgentState


# T7 起的并行工具节点名常量（注册名 / 边名 一处定义，避免拼写漂移）。
_PARALLEL_TOOL_NODES: tuple[str, ...] = (
    "weather_check",
    "activity_search",
    "restaurant_search",
    "traffic_eta",
    "queue_check",
    "crowd_risk",
)


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("llm_answer", llm_answer_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("constraint_collect", constraint_collect_node)

    # T7：6 个并行工具节点，均从 constraint_collect 扇出。
    graph.add_node("weather_check", weather_check_node)
    graph.add_node("activity_search", activity_search_node)
    graph.add_node("restaurant_search", restaurant_search_node)
    graph.add_node("traffic_eta", traffic_eta_node)
    graph.add_node("queue_check", queue_check_node)
    graph.add_node("crowd_risk", crowd_risk_node)

    # T8：plan_candidate 取代旧 planning，作为 6 个并行节点的汇聚点。
    graph.add_node("plan_candidate", plan_candidate_node)
    # T9：在 plan_candidate 与 presentation 之间插入 Validate Plan 节点。
    graph.add_node("validate_plan", validate_plan_node)
    # T10：Replan Node 接住 validate_plan 失败 / 用户拒绝时的反馈，
    # 把 reason 结构化成新约束并回到 constraint_collect 重新走 6 个并行节点。
    graph.add_node("replan", replan_node)
    graph.add_node("presentation", presentation_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("execution", execution_node)
    graph.add_node("reject", reject_node)

    graph.set_entry_point("intent")
    # route_after_intent_for_retrieval 仍返回 "planning" / "retrieval" / "llm_answer"
    # 三个语义；映射表把 "planning" 路由到 constraint_collect，确保所有进入规划链路的请求
    # 都先经过 constraint_collect 再到 plan_candidate。
    graph.add_conditional_edges(
        "intent",
        route_after_intent_for_retrieval,
        {
            "llm_answer": "llm_answer",
            "retrieval": "retrieval",
            "planning": "constraint_collect",
        },
    )
    graph.add_edge("llm_answer", END)
    graph.add_edge("retrieval", "constraint_collect")
    # T8：constraint_collect 扇出到 6 个并行工具节点；6 个节点全部汇聚到
    # plan_candidate（旧 planning 节点已不再连入主图）。
    for node_name in _PARALLEL_TOOL_NODES:
        graph.add_edge("constraint_collect", node_name)
        graph.add_edge(node_name, "plan_candidate")
    graph.add_edge("plan_candidate", "validate_plan")
    # T9 / T10：validate_plan 之后做条件路由：通过 → presentation；不通过 → replan。
    graph.add_conditional_edges(
        "validate_plan",
        route_after_validate,
        {
            "presentation": "presentation",
            "replan": "replan",
        },
    )
    # T10：replan 之后的条件路由。
    # constraint_collect → 回边重跑 6 个并行节点 + plan_candidate + validate；
    # final_message → 命中 replan_count 上限时进入兜底展示。T15 之前 final_message
    # 节点尚未实现，这里占位映射到 presentation，待 T15 上线后改为真正的
    # final_message 节点。
    graph.add_conditional_edges(
        "replan",
        route_after_replan,
        {
            "constraint_collect": "constraint_collect",
            "final_message": "presentation",  # TODO(T15): 切到 final_message_node
        },
    )
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
