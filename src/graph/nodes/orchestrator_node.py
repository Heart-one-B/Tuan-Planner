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

    plan_context     = state.get("plan_context") or {}
    fact_data        = (agent_outputs.get("fact") or {}).get("data") or {}
    eval_data        = (agent_outputs.get("evaluation") or {}).get("data") or {}
    selected_plan_id = eval_data.get("selected_plan_id") or ""

    planning_data = (agent_outputs.get("planning") or {}).get("data") or {}
    candidates    = planning_data.get("candidates") or []
    selected_plan = next(
        (c for c in candidates if c.get("id") == selected_plan_id),
        candidates[0] if candidates else {}
    )

    user_feedback = state.get("user_input") or ""

    if not user_feedback:
        task_log.append("orchestrator: no feedback, skipped")
        return {"task_log": task_log, "errors": errors, "has_new_plan": False}

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

        data = result.data
        if data.get("updated_fact"):
            agent_outputs["fact"] = data["updated_fact"]
        if data.get("updated_planning"):
            agent_outputs["planning"] = data["updated_planning"]
        if data.get("updated_evaluation"):
            agent_outputs["evaluation"] = data["updated_evaluation"]

        task_log.append(f"orchestrator: {result.summary}")

        # 只有真正产出了新的 evaluation 结果，且有选中方案，
        # 才需要走向 presentation 重新渲染；否则(比如只是闲聊式的
        # 调整意见、或者没有触发 replan)保持原有展示不变
        new_eval = agent_outputs.get("evaluation") or {}
        has_new_plan = bool(
            data.get("updated_evaluation")
            and new_eval.get("status") == "ok"
            and (new_eval.get("data") or {}).get("selected_plan_id")
        )

    except Exception as e:
        errors.append({"node": "orchestrator", "error": str(e), "recoverable": True})
        task_log.append(f"orchestrator: exception {e}")
        has_new_plan = False

    return {
        "agent_outputs": agent_outputs,
        "task_log":      task_log,
        "errors":        errors,
        "has_new_plan":  has_new_plan,
    }