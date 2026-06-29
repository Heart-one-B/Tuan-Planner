from __future__ import annotations

from agents.orchestrator.agent import OrchestratorAgent
from agents.fact.agent import FactAgent
from agents.planning.agent import PlanningAgent
from agents.evaluation.agent import EvaluationAgent
from src.model.factory import build_llm_client
from src.graph.state import AgentState


async def orchestrator_node(state: AgentState) -> dict:
    task_log      = list(state.get("task_log") or [])
    errors        = list(state.get("errors") or [])
    agent_outputs = dict(state.get("agent_outputs") or {})

    # 取当前状态
    plan_context   = state.get("plan_context") or {}
    fact_data      = (agent_outputs.get("fact") or {}).get("data") or {}
    eval_data      = (agent_outputs.get("evaluation") or {}).get("data") or {}
    selected_plan_id = eval_data.get("selected_plan_id") or ""

    # 找到选中的方案
    planning_data = (agent_outputs.get("planning") or {}).get("data") or {}
    candidates    = planning_data.get("candidates") or []
    selected_plan = next(
        (c for c in candidates if c.get("id") == selected_plan_id),
        candidates[0] if candidates else {}
    )

    user_feedback = state.get("user_input") or ""

    if not user_feedback:
        task_log.append("orchestrator: no feedback, skipped")
        return {"task_log": task_log, "errors": errors}

    try:
        llm = build_llm_client()
        result = await OrchestratorAgent(
            llm_client=llm,
            fact_agent=FactAgent(llm_client=llm),
            planning_agent=PlanningAgent(llm_client=llm),
            evaluation_agent=EvaluationAgent(llm_client=llm),
        ).run(
            user_feedback=user_feedback,
            plan_context=plan_context,
            fact_data=fact_data,
            selected_plan=selected_plan,
            trace_id=state.get("session_id"),
        )

        # 只更新变动的字段
        data = result.data
        if data.get("updated_fact"):
            agent_outputs["fact"] = data["updated_fact"]
        if data.get("updated_planning"):
            agent_outputs["planning"] = data["updated_planning"]
        if data.get("updated_evaluation"):
            agent_outputs["evaluation"] = data["updated_evaluation"]

        task_log.append(f"orchestrator: {result.summary}")

    except Exception as e:
        errors.append({"node": "orchestrator", "error": str(e), "recoverable": True})
        task_log.append(f"orchestrator: exception {e}")

    return {
        "agent_outputs": agent_outputs,
        "task_log":      task_log,
        "errors":        errors,
    }