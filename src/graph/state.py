from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    intent: dict[str, Any]
    plan: dict[str, Any]
    display_text: str
    user_confirmed: bool
    execution_result: dict[str, Any]
    errors: list[str]

