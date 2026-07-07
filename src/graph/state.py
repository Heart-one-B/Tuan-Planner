# src/graph/state.py
from __future__ import annotations
from typing import Any, Literal
from typing_extensions import TypedDict


# ── 局部状态:各 Agent 自己的产出,不跨 Agent 直接读取 ──────────────────────
class AgentOutput(TypedDict, total=False):
    status: str              # ok / partial / empty / error
    summary: str             # 一句话结论
    data: dict[str, Any]     # 业务数据,形状由各 Agent 自定义
    error: str               # 失败原因,供 Orchestrator 决策用


# ── 全局状态:所有节点/Agent 都可读,严格只追加不覆盖 ──────────────────────
class AgentState(TypedDict, total=False):

    # 用户输入(只写一次,所有 Agent 只读)
    user_input: str
    session_id: str
    conversation_turns: list[str]      # 追加,不覆盖

    # 路由结果
    feedback_route: Literal["new_plan", "adjust", "chat"]

    # 意图与约束(intent 节点和 constraint 节点各写一次)
    intent: dict[str, Any]
    plan_context: dict[str, Any]

    # 各 Agent 产出(按 agent 名称 key 隔离,只追加新 key 不覆盖旧 key)
    agent_outputs: dict[str, dict]
    # 例:
    # agent_outputs["fact"]     → FactAgent 的产出
    # agent_outputs["planning"] → PlanningAgent 的产出
    # agent_outputs["eval"]     → EvaluationAgent 的产出

    # 任务进展(追加,供 Orchestrator 读取决策)
    task_log: list[str]               # 每个节点完成后追加一条

    # 错误收集(追加,不覆盖,Orchestrator 据此决策降级或终止)
    errors: list[dict[str, Any]]
    # 每条格式: {"node": "fact", "error": "...", "recoverable": True/False}

    # 最终输出
    display_text: str
    user_confirmed: bool
    pending_clarification: str
    final_message: str

    clarification_round: int  # 当前追问轮次，默认0
    clarification_history: list[dict]  # [{"question": str, "answer": str}]
    clarification_forced: bool
    has_new_plan: bool