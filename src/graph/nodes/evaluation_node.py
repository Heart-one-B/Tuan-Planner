from __future__ import annotations

from agents.evaluation.agent import EvaluationAgent
from src.model.factory import build_llm_client
from src.graph.state import AgentState


async def evaluation_node(state: AgentState) -> dict:
    task_log = list(state.get("task_log") or [])
    errors   = list(state.get("errors") or [])
    agent_outputs = dict(state.get("agent_outputs") or {})

    plan_output  = agent_outputs.get("planning") or {}
    plan_data    = plan_output.get("data") or {}
    plan_context = state.get("plan_context") or {}
    fact_data    = (agent_outputs.get("fact") or {}).get("data") or {}

    if not plan_data:
        errors.append({"node": "evaluation", "error": "plan_data 缺失", "recoverable": False})
        task_log.append("evaluation: skipped, plan_data missing")
        return {"task_log": task_log, "errors": errors}

    try:
        result = await EvaluationAgent(llm_client=build_llm_client()).run(
            plan_data=plan_data,
            plan_context=plan_context,
            fact_data=fact_data,
            trace_id=state.get("session_id"),
        )
        agent_outputs["evaluation"] = {
            "status": result.status,
            "summary": result.summary,
            "data": result.data,
        }
        task_log.append(
            f"evaluation: status={result.status} "
            f"selected={result.data.get('selected_plan_id')} "
            f"score={result.data.get('selected_score')}"
        )
        if result.status == "error":
            errors.append({
                "node": "evaluation",
                "error": result.summary,
                "recoverable": False,
            })
    except Exception as e:
        agent_outputs["evaluation"] = {"status": "error", "summary": str(e), "data": {}}
        errors.append({"node": "evaluation", "error": str(e), "recoverable": False})
        task_log.append(f"evaluation: exception {e}")

    return {"agent_outputs": agent_outputs, "task_log": task_log, "errors": errors}