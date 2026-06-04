from operator import add
from typing import Any, Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # --- entry ---
    user_input: str
    conversation_turns: list[str]
    clarification_round: int
    user_preference_profile: str
    
    # 🌟 新增：支持异步澄清交互的状态字段 🌟
    pending_clarification: str  # 挂起时输出给前端的追问消息
    user_reply: str            # 前端注入的用户回复
    
    runtime_origin_area: str
    runtime_origin_coordinates: str
    location_permission_granted: bool
    location_lookup_result: dict[str, Any]

    # --- intent and routing ---
    intent: dict[str, Any]
    is_leisure_planning: bool
    need_retrieval: bool
    clarification_needed: bool
    
    # 🌟 修改：由 dict 修改为 list[str]，扁平化管理缺失槽位 🌟
    missing_slots: list[str]
    
    # 🌟 新增：显示当前正在澄清的特定槽位名（如 scenario / time_day / time_window 等） 🌟
    current_asking_slot: str | None
    
    follow_up_message: str
    llm_answer: str

    # --- retrieval and unified constraints ---
    retrieval_context: dict[str, Any]
    plan_context: dict[str, Any]
    # constraints currently carries at least:
    # scenario / party / time_window / date_label / daypart / time_phrase /
    # start_time / end_time / diet_preference / origin_area /
    # max_traffic_minutes / max_queue_minutes / indoor_preferred / replan_hints

    # --- parallel fact gathering ---
    weather: dict[str, Any]
    activities: list[dict[str, Any]]
    restaurants: list[dict[str, Any]]
    traffic: dict[str, Any]
    queue: dict[str, Any]
    crowd: dict[str, Any]
    fact_gathering_result: dict[str, Any]
    activity_explicit_search: dict[str, Any]

    # --- planning and validation ---
    candidates: dict[str, Any]
    candidate_plans: dict[str, Any]
    plan_poi_details: dict[str, Any]
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
    web_preview_mode: bool
    pending_confirmation: dict[str, Any]

    # --- execution and final output ---
    execution_result: dict[str, Any]
    recovery_result: dict[str, Any]
    final_message: str

    # --- aggregated errors ---
    errors: Annotated[list[str], add]
