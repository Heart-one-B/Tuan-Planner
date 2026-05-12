from src.agent.execution_agent import ExecutionAgent
from src.agent.intent_agent import IntentAgent
from src.agent.planning_agent import PlanningAgent
from src.agent.presentation_agent import PresentationAgent
from src.graph.state import AgentState


def _append_error(state: AgentState, error: str) -> AgentState:
    errors = list(state.get("errors", []))
    errors.append(error)
    return {"errors": errors}


def intent_node(state: AgentState) -> AgentState:
    try:
        intent = IntentAgent().parse(state["user_input"])
        return {"intent": intent}
    except Exception as exc:
        return _append_error(state, f"Intent node failed: {exc}")


def planning_node(state: AgentState) -> AgentState:
    try:
        plan = PlanningAgent().plan(state.get("intent", {}))
        return {"plan": plan}
    except Exception as exc:
        return _append_error(state, f"Planning node failed: {exc}")


def presentation_node(state: AgentState) -> AgentState:
    try:
        display_text = PresentationAgent().generate_plan_display(
            state.get("plan", {}),
            state.get("intent", {}),
        )
        print("\n" + "=" * 20 + " 方案详情 " + "=" * 20)
        print(display_text)
        print("=" * 50)
        return {"display_text": display_text}
    except Exception as exc:
        return _append_error(state, f"Presentation node failed: {exc}")


def confirmation_node(state: AgentState) -> AgentState:
    if "user_confirmed" in state:
        return {"user_confirmed": state["user_confirmed"]}
    confirm = input("\n[系统提示] 确定按照此方案执行一键下单吗？(y/n): ")
    return {"user_confirmed": confirm.lower() == "y"}


def execution_node(state: AgentState) -> AgentState:
    try:
        result = ExecutionAgent().execute(state.get("plan", {}))
        if result is None:
            result = {"status": "success", "message": "Execution completed"}
        return {"execution_result": result}
    except Exception as exc:
        return _append_error(state, f"Execution node failed: {exc}")


def reject_node(state: AgentState) -> AgentState:
    return {
        "execution_result": {
            "status": "cancelled",
            "message": "用户未确认方案，未执行任何预约或下单动作。",
        }
    }


def route_after_confirmation(state: AgentState) -> str:
    if state.get("user_confirmed"):
        return "execute"
    return "reject"
