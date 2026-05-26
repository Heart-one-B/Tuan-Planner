from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from src.graph.workflow import app as workflow_app


class WorkflowSessionService:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create_session(self, user_input: str, runtime_origin_area: str = "") -> str:
        session_id = uuid4().hex[:12]
        self._sessions[session_id] = self._build_initial_state(
            user_input=user_input,
            runtime_origin_area=runtime_origin_area,
        )
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        state = self._sessions.get(session_id)
        if state is None:
            return None
        return deepcopy(state)

    def run_turn(
        self,
        *,
        session_id: str | None,
        user_input: str | None = None,
        runtime_origin_area: str | None = None,
        location_permission_granted: bool | None = None,
        clarification_response: str | None = None,
        user_confirmed: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if session_id and session_id in self._sessions:
            state = deepcopy(self._sessions[session_id])
        else:
            state = self._build_initial_state(
                user_input=user_input or "",
                runtime_origin_area=runtime_origin_area or "",
            )
            session_id = session_id or uuid4().hex[:12]

        if user_input is not None and user_input.strip():
            state = self._build_initial_state(
                user_input=user_input.strip(),
                runtime_origin_area=runtime_origin_area or state.get("runtime_origin_area", ""),
            )
        else:
            if isinstance(runtime_origin_area, str) and runtime_origin_area.strip():
                state["runtime_origin_area"] = runtime_origin_area.strip()
            if location_permission_granted is not None:
                state["location_permission_granted"] = bool(location_permission_granted)
            if isinstance(clarification_response, str) and clarification_response.strip():
                turns = list(state.get("conversation_turns", []))
                if not turns:
                    first_turn = state.get("user_input", "")
                    if isinstance(first_turn, str) and first_turn.strip():
                        turns = [first_turn.strip()]
                turns.append(clarification_response.strip())
                state["conversation_turns"] = turns
                state["user_input"] = "\n".join(turns)
                state["clarification_response"] = clarification_response.strip()
            if user_confirmed is not None:
                state["user_confirmed"] = bool(user_confirmed)

        result = workflow_app.invoke(state)
        self._sessions[session_id] = deepcopy(result)
        return session_id, result

    def snapshot(self, session_id: str) -> dict[str, Any] | None:
        state = self._sessions.get(session_id)
        if state is None:
            return None
        return self._build_response(session_id=session_id, state=state)

    def _build_initial_state(self, *, user_input: str, runtime_origin_area: str = "") -> dict[str, Any]:
        cleaned_input = user_input.strip()
        initial_state: dict[str, Any] = {
            "user_input": cleaned_input,
            "conversation_turns": [cleaned_input] if cleaned_input else [],
            "clarification_round": 0,
            "runtime_origin_area": runtime_origin_area.strip(),
            "errors": [],
            "interactive_mode": False,
        }
        return initial_state

    def _build_ui(self, state: dict[str, Any]) -> dict[str, Any]:
        ui_state = state.get("ui_state") if isinstance(state.get("ui_state"), dict) else {}
        return {
            "step": state.get("pending_action") or ui_state.get("step") or "",
            "prompt": state.get("ui_prompt") or ui_state.get("prompt") or "",
            "kind": state.get("ui_kind") or ui_state.get("kind") or "",
            "options": state.get("ui_options") or ui_state.get("options") or [],
        }

    def _status_from_state(self, state: dict[str, Any]) -> str:
        if state.get("pending_action"):
            return "awaiting_input"
        if state.get("llm_answer"):
            return "done"
        execution_result = state.get("execution_result")
        if isinstance(execution_result, dict) and execution_result.get("status"):
            return str(execution_result.get("status"))
        if state.get("final_message"):
            return "done"
        if state.get("errors"):
            return "warning"
        return "running"

    def _build_response(self, *, session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "status": self._status_from_state(state),
            "ui": self._build_ui(state),
            "state": deepcopy(state),
            "display_text": state.get("display_text") if isinstance(state.get("display_text"), str) else "",
            "final_message": state.get("final_message") if isinstance(state.get("final_message"), str) else "",
            "llm_answer": state.get("llm_answer") if isinstance(state.get("llm_answer"), str) else "",
            "execution_result": state.get("execution_result") if isinstance(state.get("execution_result"), dict) else {},
            "errors": state.get("errors") if isinstance(state.get("errors"), list) else [],
        }

    def format_response(self, session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        return self._build_response(session_id=session_id, state=state)
