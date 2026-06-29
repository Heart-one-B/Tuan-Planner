from __future__ import annotations

from agents.intent.agent import IntentAgent
from src.model.factory import build_llm_client
from src.graph.state import AgentState


async def intent_node(state: AgentState) -> dict:
    user_input = (state.get("user_input") or "").strip()
    task_log = list(state.get("task_log") or [])
    errors = list(state.get("errors") or [])

    try:
        result = await IntentAgent(llm_client=build_llm_client()).parse(
            user_input=user_input,
            trace_id=state.get("session_id"),
        )
        intent = result.model_dump()
        task_log.append(
            f"intent: scenario={intent.get('scenario')} "
            f"clarification_needed={intent.get('clarification_needed')} "
            f"missing_slots={intent.get('missing_slots')}"
        )
        return {"intent": intent, "task_log": task_log, "errors": errors}

    except Exception as e:
        errors.append({"node": "intent", "error": str(e), "recoverable": False})
        task_log.append(f"intent: failed {e}")
        return {"task_log": task_log, "errors": errors}