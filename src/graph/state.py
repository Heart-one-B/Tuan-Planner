"""LangGraph shared state contract for the local life planning workflow."""

from operator import add
from typing import Any, Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # --- entry ---
    user_input: str
    conversation_turns: list[str]
    clarification_round: int
    interactive_mode: bool
    runtime_origin_area: str
    runtime_origin_coordinates: str
    location_permission_granted: bool
    location_lookup_result: dict[str, Any]
    normalized_time: dict[str, Any]
    time_normalization_result: dict[str, Any]

    # --- intent and routing ---
    intent: dict[str, Any]
    is_leisure_planning: bool
    need_retrieval: bool
    clarification_needed: bool
    missing_slots: dict[str, list[str]]
    follow_up_message: str
    llm_answer: str
    clarification_response: str
    pending_action: str
    ui_state: dict[str, Any]
    ui_prompt: str
    ui_kind: str
    ui_options: list[dict[str, Any]]

    # --- retrieval and unified constraints ---
    retrieval_context: dict[str, Any]
    constraints: dict[str, Any]
    constraint_build: dict[str, Any]
    # constraints currently carries at least:
    # scenario / party / time_window / date_label / daypart / time_phrase /
    # start_time / duration_hours / diet_preference / origin_area /
    # max_traffic_minutes / max_queue_minutes / indoor_preferred / replan_hints
    # Minimum node-level time requirements are centralized in
    # ``src.graph.time_contract.MIN_TIME_FIELDS_BY_NODE``.

    # --- parallel fact gathering ---
    weather: dict[str, Any]
    activities: list[dict[str, Any]]
    restaurants: list[dict[str, Any]]
    traffic: dict[str, Any]
    queue: dict[str, Any]
    crowd: dict[str, Any]
    fact_gathering_result: dict[str, Any]

    # --- planning and validation ---
    candidates: dict[str, Any]
    candidate_plans: dict[str, Any]
    validation_result: dict[str, Any]
    rule_validation_result: dict[str, Any]
    repair_loop_result: dict[str, Any]
    scoring_result: dict[str, Any]
    final_plan_result: dict[str, Any]
    final_plan_selection: dict[str, Any]
    schedule_timing_result: dict[str, Any]

    # --- replan loop ---
    replan_reason: str
    replan_reason_type: str
    replan_count: int
    repair_round: int

    # --- compatibility display/execution view ---
    plan: dict[str, Any]
    display_text: str
    user_confirmed: bool

    # --- execution and final output ---
    execution_result: dict[str, Any]
    recovery_result: dict[str, Any]
    final_message: str

    # --- aggregated errors ---
    errors: Annotated[list[str], add]
