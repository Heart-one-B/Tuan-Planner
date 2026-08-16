# src/graph/state.py
from __future__ import annotations

from typing import Any, Literal
from typing_extensions import TypedDict


# ── 局部状态：各 Agent 自己的产出，不跨 Agent 直接读取 ──────────────────
class AgentOutput(TypedDict, total=False):
    status: str              # ok / partial / empty / error
    summary: str             # 一句话结论
    data: dict[str, Any]     # 业务数据，形状由各 Agent 自定义
    error: str               # 失败原因，供 Orchestrator 决策用


# ── 全局状态：所有节点/Agent 都可读，严格只追加不覆盖 ──────────────────
#
# ⚠️ 【必须在这里声明，否则静默丢失】
# LangGraph 按这个 TypedDict 过滤 state：**没在这里声明的键会被
# 直接丢弃**，不报错、不告警，节点里永远读不到。
#
# 这个坑今天踩过：ainvoke 时传了 root_trace_id，节点里读到的是
# None，于是每个子 Agent 各开各的根 trace，Span 树散成一地——
# 而全程没有任何一条错误信息。
#
# 同一天还踩过同构的另外三次：POIItem 没声明 cost（pydantic 丢弃）、
# maps_around_search 没有 types 参数（MCP 丢弃）、_hid 出网前要剥离。
# 共同结构是"发送方以为传了，接收方按 schema 过滤掉了"。
# 加字段时先来这里声明，不要只在传参处写。
class AgentState(TypedDict, total=False):

    # 用户输入（只写一次，所有 Agent 只读）
    user_input: str
    session_id: str
    root_trace_id: str                 # 本次请求的根 Span，子 Agent 挂它下面
    conversation_turns: list[str]      # 追加，不覆盖

    # 路由结果
    feedback_route: Literal["new_plan", "adjust", "chat", "clarify_reply"]

    # 意图与约束
    intent: dict[str, Any]
    plan_context: dict[str, Any]

    # 各 Agent 产出（按 agent 名称 key 隔离，只追加新 key 不覆盖旧 key）
    agent_outputs: dict[str, dict]
    # 例：
    # agent_outputs["fact"]       → FactAgent 的产出
    # agent_outputs["planning"]   → PlanningAgent 的产出
    # agent_outputs["evaluation"] → EvaluationAgent 的产出

    # 任务进展（追加，供 Orchestrator 读取决策）
    task_log: list[str]

    # 错误收集（追加，不覆盖，Orchestrator 据此决策降级或终止）
    errors: list[dict[str, Any]]
    # 每条格式: {"node": "fact", "error": "...", "recoverable": True/False}
    # recoverable 的判据：**这次失败重试有没有可能成功**。
    # TypeError/AttributeError 这类签名或属性错误每次都会以相同方式
    # 失败，必须标 False——标成 True 只会让 bug 活得更久（实测过：
    # orchestrator 的一个签名不匹配被当成可恢复失败，导致"用户说
    # 换一家、系统回同一份方案"这个体验缺陷存活了很久）。

    # 最终输出
    display_text: str
    user_confirmed: bool
    pending_clarification: str
    final_message: str

    # 澄清追问
    clarification_round: int
    clarification_history: list[dict]   # [{"question": str, "answer": str}]
    clarification_forced: bool

    # 调整历史：用户对当前方案提过的全部修改意见，按顺序累积。
    # 只留最后一条会让先前的调整被静默丢弃——用户先说"不吃辣"、
    # 再说"换个近点的"，第二条一提交第一条就失效，是最容易被察觉
    # 也最恼火的那种"说了不听"。
    adjust_history: list[str]

    has_new_plan: bool

    # 本会话已经推送过的记忆名，避免同一条记忆每轮重复注入
    surfaced_memory_names: list[str]