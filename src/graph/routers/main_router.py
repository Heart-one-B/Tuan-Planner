# src/graph/routers/main_router.py
from __future__ import annotations

from src.graph.state import AgentState


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent") or {}
    errors = state.get("errors") or []

    if any(e.get("node") == "intent" for e in errors):
        return "end"

    if not intent.get("is_leisure_planning"):
        return "chat"

    if intent.get("clarification_needed"):
        return "clarification"

    return "constraint"


def route_after_feedback(state: AgentState) -> str:
    """routing 之后的分流。

    【本次改动：adjust 不再直接跳到 orchestrator】
    原来 adjust → orchestrator，跳过了 intent 和 constraint，
    于是用户的调整意见只以自由文本形式进入 replan 的 hint，
    **plan_context 一个字没变**。

    实测后果：用户说"晚上不吃火锅了我们都吃不了辣"，方案确实换成了
    非辣的羊肉汤锅（replan 收到了 hint），但 EvaluationAgent 拿着
    旧 plan_context（restaurant_intent=explicit / 火锅）去评判，
    给出 72 分并写下"用户明确要求晚上吃火锅，但方案提供的是羊肉
    汤锅"——系统在用用户已经撤回的需求批评自己刚做对的事。

    现在 adjust 也走 intent → constraint，重新推导出正确的
    plan_context，再交给 orchestrator。代价是每次调整多一次
    IntentAgent 调用；换来的是下游全部环节（planning 的候选池渲染、
    evaluation 的打分依据、fact 的 no_spicy 约束）对"用户现在到底
    想要什么"有一致的理解。
    """
    route = state.get("feedback_route") or "new_plan"
    if route == "chat":
        return "chat"
    # new_plan / clarify_reply / adjust 统一先重新解析意图，
    # 三者的差别体现在 intent_node 内部怎么拼接输入，
    # 以及 constraint 之后往哪走（见 route_after_constraint）。
    return "intent"


def route_after_constraint(state: AgentState) -> str:
    """constraint 之后：调整现有方案 还是 从头收集事实。

    【为什么在这里分流而不是更早】
    adjust 需要的是"更新后的 plan_context + 现有候选池"，
    所以 constraint 必须先跑完。而 fact→planning→evaluation 那条
    全量链路对 adjust 是浪费——候选池大部分还能用，orchestrator
    自己会判断要不要补搜（search_pois 工具）。

    兜底条件：feedback_route=adjust 但**没有**已选中方案时，
    仍然走全量链路。这种情况理论上不该发生（routing_node 在没有
    selected_plan_id 时就直接判 new_plan 了），但 state 是跨轮
    累积的，防御一下不亏——没有方案可调却进了 orchestrator，
    它会拿着空的 selected_plan 空转一轮。
    """
    if (state.get("feedback_route") or "") != "adjust":
        return "fact"

    eval_data = ((state.get("agent_outputs") or {}).get("evaluation") or {}).get("data") or {}
    if not eval_data.get("selected_plan_id"):
        return "fact"

    return "orchestrator"


def route_after_fact(state: AgentState) -> str:
    agent_outputs = state.get("agent_outputs") or {}
    fact = agent_outputs.get("fact") or {}

    if fact.get("status") == "error":
        return "end"
    return "planning"


def route_after_clarification(state: AgentState) -> str:
    """clarification_node 跑完后的路由：
    - pending_clarification 非空 → 还在等用户回答，中断到 END
    - 为空（强制兜底完成，或本不该来这里）→ 进入 constraint
    """
    pending = state.get("pending_clarification") or ""
    if pending:
        return "end"
    return "constraint"


def route_after_evaluation(state: AgentState) -> str:
    """- status=ok 且有选中方案 → presentation 渲染展示
    - status=error（全部不合格）→ END，告知用户
    """
    agent_outputs = state.get("agent_outputs") or {}
    eval_output = agent_outputs.get("evaluation") or {}

    if eval_output.get("status") == "ok" and (eval_output.get("data") or {}).get("selected_plan_id"):
        return "presentation"
    return "end"


def route_after_orchestrator(state: AgentState) -> str:
    """- 产出了新的合格方案 → presentation 重新渲染
    - 没产出（调度失败/异常/未触发 replan）→ END，保持原展示
    """
    if state.get("has_new_plan"):
        return "presentation"
    return "end"