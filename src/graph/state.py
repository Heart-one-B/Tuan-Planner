"""LangGraph 共享状态契约。

本模块定义 ``AgentState``，是所有节点之间唯一的共享数据契约。
目标流程见 ``docs/langgraph-flow.md``。

设计原则：
    * 仍保持 ``TypedDict + total=False`` 的纯结构风格，不引入 pydantic、
      dataclass 或运行期校验，节点写入时只需返回需要更新的字段。
    * 字段顺序与 LangGraph 节点流向一致，便于对照流程图阅读。
    * 下表给出每个字段的「写入节点 / 读取节点」设计期意图。后续节点实现
      可以扩展读取方而不破坏本契约（写入方应保持唯一，避免冲突）。

字段写入 / 读取映射
-------------------

旧字段（迁移期保留，向后兼容现有 ``nodes.py`` / ``workflow.py``）：

    user_input
        写入: 入口（外部调用方传入）
        读取: Intent Node, LLM Answer Node, Retrieval Node
    intent
        写入: Intent Node
        读取: Need Retrieval?, Constraint Collect Node, Presentation Node
    plan
        写入: Plan Candidate Node（迁移期可由 candidates 派生）
        读取: Presentation Node, Execution Node
    display_text
        写入: Presentation Node
        读取: User Confirm?
    user_confirmed
        写入: User Confirm?（外部输入）
        读取: route_after_confirmation
    execution_result
        写入: Execution Node, Reject Node
        读取: Execution Result 路由, Final Message Node, Execution Recovery Node
    errors
        写入: 任意节点的异常分支（追加写入）
        读取: 任意下游节点 / 调用方

新增字段（按 LangGraph 流程顺序排列）：

    is_leisure_planning
        写入: Intent Node
        读取: Leisure Planning Task? 路由
    need_retrieval
        写入: Intent Node（或独立判定节点）
        读取: Need Retrieval? 路由
    clarification_needed
        写入: Intent Node
        读取: Clarification 路由 / 调用方
    missing_slots
        写入: Intent Node
        读取: Clarification 路由 / 调用方
        形如: {"global": ["scenario", "origin_area"]}
    follow_up_message
        写入: Intent Node
        读取: Clarification 路由 / 调用方
    llm_answer
        写入: LLM Answer Node
        读取: 终止前调用方
    retrieval_context
        写入: Retrieval Node
        读取: Constraint Collect Node, Activity Search, Restaurant Search,
              Presentation Node
        形如: {"pois": [...], "notes": [...]}
    constraints
        写入: Constraint Collect Node
        读取: Weather Check, Activity Search, Restaurant Search,
              Traffic ETA, Queue Check, Crowd Risk, Plan Candidate Node,
              Validate Plan, Replan Node
        包含: scenario / party / time_window / diet_preference /
              max_traffic_minutes / max_queue_minutes / indoor_preferred /
              replan_hints
    weather
        写入: Weather Check
        读取: Plan Candidate Node, Validate Plan
    activities
        写入: Activity Search
        读取: Plan Candidate Node, Validate Plan
    restaurants
        写入: Restaurant Search
        读取: Plan Candidate Node, Validate Plan
    traffic
        写入: Traffic ETA
        读取: Plan Candidate Node, Validate Plan
    queue
        写入: Queue Check
        读取: Plan Candidate Node, Validate Plan
    crowd
        写入: Crowd Risk
        读取: Plan Candidate Node, Validate Plan
    candidates
        写入: Plan Candidate Node, Replan Node
        读取: Validate Plan, Presentation Node, Execution Node
        形如: {"primary": {...}, "backup": {...}}
    validation_result
        写入: Validate Plan
        读取: Validate 路由, Replan Node, Presentation Node
        形如: {"passed": bool, "violations": [str], "suggested_fixes": [str]}
    replan_reason
        写入: Replan Node, User Confirm? 拒绝分支, Execution Result 核心变化分支
        读取: Replan Node, Plan Candidate Node
    replan_reason_type
        写入: Replan Node, User Confirm? 拒绝分支, Execution Result 核心变化分支
        读取: Replan Node
    replan_count
        写入: Replan Node（自增；读取方应使用 ``state.get("replan_count", 0)`` 兜底）
        读取: Replan Node 防死循环判定
    recovery_result
        写入: Execution Recovery Node
        读取: Final Message Node
    final_message
        写入: Final Message Node
        读取: 终止前调用方
"""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # --- 入口与意图 ---
    user_input: str
    runtime_origin_area: str
    intent: dict[str, Any]

    # --- 路由布尔 ---
    is_leisure_planning: bool
    need_retrieval: bool
    clarification_needed: bool
    missing_slots: dict[str, list[str]]
    follow_up_message: str

    # --- 非规划直答 ---
    llm_answer: str

    # --- 检索与约束 ---
    retrieval_context: dict[str, Any]
    constraints: dict[str, Any]

    # --- 并行工具结果（Constraint Collect 之后扇出）---
    weather: dict[str, Any]
    activities: list[dict[str, Any]]
    restaurants: list[dict[str, Any]]
    traffic: dict[str, Any]
    queue: dict[str, Any]
    crowd: dict[str, Any]

    # --- 候选方案与校验 ---
    candidates: dict[str, Any]
    validation_result: dict[str, Any]

    # --- 重规划 ---
    replan_reason: str
    replan_reason_type: str
    replan_count: int

    # --- 展示与确认（旧字段，迁移期保留）---
    plan: dict[str, Any]
    display_text: str
    user_confirmed: bool

    # --- 执行、恢复与最终消息 ---
    execution_result: dict[str, Any]
    recovery_result: dict[str, Any]
    final_message: str

    # --- 错误聚合（任意节点追加）---
    errors: list[str]
