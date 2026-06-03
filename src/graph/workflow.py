# src/graph/workflow.py
from langgraph.graph import END, StateGraph

from src.graph.nodes.intent_node import intent_node
from src.graph.nodes.clarification_node import clarification_node, receive_clarification_node
from src.graph.nodes.llm_answer_node import llm_answer_node
from src.graph.nodes.constraint_build_node import constraint_build_node
from src.graph.nodes.fact_gathering_node import fact_gathering_node
from src.graph.nodes.candidate_planning_node import candidate_planning_node
from src.graph.nodes.rule_validation_node import rule_validation_node
from src.graph.nodes.repair_loop_node import repair_loop_node
from src.graph.nodes.scoring_node import scoring_node
from src.graph.nodes.final_plan_node import final_plan_node
from src.graph.nodes.presentation_node import presentation_node
from src.graph.nodes.confirmation_node import confirmation_node
from src.graph.nodes.execution_node import execution_node
from src.graph.nodes.final_message_node import final_message_node

from src.graph.routers import route_after_confirmation
from src.graph.state import AgentState

_MAX_CLARIFICATION_ROUNDS = 5


# ---------------------------------------------------------------------------
# 对应工作流专用的条件路由函数定义
# ---------------------------------------------------------------------------
def route_after_intent_with_clarification(state: AgentState) -> str:
    """对应第一部分流程图中的 Leisure Planning Task? 与 Need Clarification? 判断流。"""
    intent = state.get("intent")
    if not isinstance(intent, dict):
        return "constraint_build"

    # 1. Leisure Planning Task? (No -> 走直答)
    if intent.get("is_leisure_planning") is False:
        return "llm_answer"

    # 2. Need Clarification? (Yes -> 追问澄清分流)
    if intent.get("clarification_needed") is True:
        # 将轮次上限判定前置到路由中，避免输出无意义的空白追问状态
        clarification_round = state.get("clarification_round", 0)
        if isinstance(clarification_round, bool) or not isinstance(clarification_round, int):
            clarification_round = 0

        if clarification_round >= _MAX_CLARIFICATION_ROUNDS:
            print(f"[Workflow] 达到追问上限轮次 ({_MAX_CLARIFICATION_ROUNDS})，转向直答。")
            return "llm_answer"

        return "clarification"

    # 3. 正常休闲规划路径：去除了数据检索判断，直接流向约束构建
    return "constraint_build"


def route_after_rule_validation(state: AgentState) -> str:
    """对应流程图中的 Enough Valid Plans? 合法候选充足性校验决策。"""
    result = state.get("rule_validation_result")
    valid_plans = result.get("valid_plans") if isinstance(result, dict) else None
    if isinstance(valid_plans, list) and len(valid_plans) >= 3:
        return "scoring"
    return "repair_loop"


def route_after_repair_loop_new(state: AgentState) -> str:
    """对应流程图中的 Repair Loop 重试计数控制路由（上限 3 次重规划）。"""
    count = state.get("replan_count", 0)
    if isinstance(count, bool) or not isinstance(count, int):
        count = 0
    if count >= 3:
        return "final_plan"  # 达到上限时，走 final_plan 降级兜底展示
    return "constraint_build"


# ---------------------------------------------------------------------------
# 组装并编译工作流图 (LangGraph Build)
# ---------------------------------------------------------------------------
def build_workflow(checkpointer=None):
    graph = StateGraph(AgentState)

    # === 注册节点 (Nodes) ===
    graph.add_node("intent", intent_node)

    # 分别注册发问节点与回复接收合并节点
    graph.add_node("clarification", clarification_node)
    graph.add_node("receive_clarification", receive_clarification_node)

    graph.add_node("llm_answer", llm_answer_node)
    graph.add_node("constraint_build", constraint_build_node)

    graph.add_node("fact_gathering", fact_gathering_node)

    graph.add_node("candidate_planning", candidate_planning_node)
    graph.add_node("rule_validation", rule_validation_node)
    graph.add_node("repair_loop", repair_loop_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("final_plan", final_plan_node)
    graph.add_node("presentation", presentation_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("execution", execution_node)
    graph.add_node("final_message", final_message_node)

    # === 建立连线与跳转 (Edges & Routers) ===

    # 图入口
    graph.set_entry_point("intent")

    graph.add_conditional_edges(
        "intent",
        route_after_intent_with_clarification,
        {
            "llm_answer": "llm_answer",
            "clarification": "clarification",
            "constraint_build": "constraint_build",
        },
    )

    # 通过静态连线直接串联澄清与接收闭环，原 route_after_clarification 条件路由废弃
    # 澄清节点执行完毕后，执行 receive_clarification，此时由于 interrupt_after 的作用，图会在 clarification 之后自动挂起
    graph.add_edge("clarification", "receive_clarification")

    # 接收完毕并合并对话历史后，自动路由回 intent 节点重新执行完整解析
    graph.add_edge("receive_clarification", "intent")

    # 非规划直答出口
    graph.add_edge("llm_answer", END)

    # 形成统一规划问题后，进行事实采集
    graph.add_edge("constraint_build", "fact_gathering")

    # 事实采集完毕 -> LLM规划初始方案
    graph.add_edge("fact_gathering", "candidate_planning")

    # 候选生成直接送审合规校验
    graph.add_edge("candidate_planning", "rule_validation")

    # 规则合规分流 (Enough Valid Plans?)
    graph.add_conditional_edges(
        "rule_validation",
        route_after_rule_validation,
        {
            "scoring": "scoring",
            "repair_loop": "repair_loop",
        },
    )

    # 修复及重规划回环退让控制
    graph.add_conditional_edges(
        "repair_loop",
        route_after_repair_loop_new,
        {
            "constraint_build": "constraint_build",
            "final_plan": "final_plan",
        },
    )

    # 耗时最优打分 -> 挑选高分计划并解算时刻排期
    graph.add_edge("scoring", "final_plan")

    # 排期解算完毕后渲染富文本呈现
    graph.add_edge("final_plan", "presentation")

    # 方案呈现 -> 等待 CLI 终端用户一键确认
    graph.add_edge("presentation", "confirmation")

    # 用户确认状态分流 (User Confirm?)
    graph.add_conditional_edges(
        "confirmation",
        route_after_confirmation,
        {
            "execute": "execution",
            "replan": "repair_loop",
            "await_confirmation": END,
        },
    )

    # 执行完直接进入整理消息收尾，出图
    graph.add_edge("execution", "final_message")
    graph.add_edge("final_message", END)

    # 设置图在 clarification 执行完毕、填充完 pending_clarification 之后，原地中断并保存状态
    return graph.compile(
        interrupt_after=["clarification", "confirmation"],
        checkpointer=checkpointer,
    )


app = build_workflow()