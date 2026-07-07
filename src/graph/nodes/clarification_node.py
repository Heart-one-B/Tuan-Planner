# src/graph/nodes/clarification_node.py

from __future__ import annotations

from src.model.factory import build_llm_client
from src.graph.state import AgentState

MAX_CLARIFICATION_ROUNDS = 3

# 缺失字段的默认兜底值，超过轮次上限时使用
_DEFAULT_FALLBACKS = {
    "origin_area":   "用户附近",
    "people_count":  2,
    "start_time":    "14:00",
    "end_time":      "21:00",
    "scenario":      "friends",
}

_LABEL_MAP = {
    "origin_area":  "出发地",
    "people_count": "人数",
    "start_time":   "开始时间",
    "end_time":     "结束时间",
    "scenario":     "出行场景",
}

CLARIFICATION_SYSTEM_PROMPT = """\
你是行程规划助手的追问模块。用户的需求里缺少一些必要信息，
你需要生成一句自然、口语化的追问，一次性覆盖所有缺失的信息点。

## 规则

不要逐个字段生硬地列举（如"请提供：1.出发地 2.人数"），
要组织成一句自然的问句，像朋友间对话一样。

例：缺失字段是 ["origin_area", "people_count"]
好的追问："你们打算从哪边出发呀？大概几个人一起去呢？"
不好的追问："请提供出发地和人数信息。"

如果只缺一个字段，就只问那一个，不要画蛇添足问已知信息。

如果这是第2轮或以上的追问（上下文会标注轮次），语气要体现"还差一点点"，
不要让用户觉得在反复被盘问。

只输出追问文本本身，不要任何前缀、引号或多余说明。
"""


async def clarification_node(state: AgentState) -> dict:
    """生成追问，中断等待用户回复。

    第一次进入：intent 解析完发现缺失字段，生成追问，中断。
    恢复时的逻辑不在这个节点里 —— 由 routing_node 判断
    "当前是否在等待澄清回复"，如果是则重新走 intent_node。
    这个节点只负责"生成追问"这一件事。

    不做成 Agent：这里没有自主决策、没有工具调用、没有多轮推理，
    只是一次结构化输入 → 一次 LLM 调用 → 一段文本输出。
    用 StructuredAgent 的 ReAct loop + finish tool 是过度设计。
    """
    task_log = list(state.get("task_log") or [])
    errors   = list(state.get("errors") or [])

    intent = state.get("intent") or {}
    missing_slots = intent.get("missing_slots") or []
    round_num = state.get("clarification_round") or 0

    # ── 异常1：超过轮次上限，强制兜底，不再追问 ──────────────────────────
    if round_num >= MAX_CLARIFICATION_ROUNDS:
        filled_intent = _apply_fallbacks(intent, missing_slots)
        task_log.append(
            f"clarification: 已达{MAX_CLARIFICATION_ROUNDS}轮上限，"
            f"使用默认值兜底 {missing_slots}"
        )
        return {
            "intent": filled_intent,
            "clarification_forced": True,
            "pending_clarification": "",  # 清空，不再中断
            "task_log": task_log,
            "errors":   errors,
        }

    # ── 异常2：missing_slots 为空但被路由到这里（防御性检查）─────────────
    if not missing_slots:
        task_log.append("clarification: missing_slots为空，跳过追问直接放行")
        return {
            "pending_clarification": "",
            "task_log": task_log,
            "errors":   errors,
        }

    raw_query = intent.get("raw_query") or state.get("user_input") or ""
    history   = state.get("clarification_history") or []

    try:
        question = await _generate_clarification_question(
            llm_client=build_llm_client(),
            raw_query=raw_query,
            missing_slots=missing_slots,
            round_num=round_num,
            history=history,
        )

        if not question or not question.strip():
            # ── 异常3：LLM 返回空内容，降级为规则生成的通用追问 ─────────
            question = _fallback_question(missing_slots)
            task_log.append("clarification: LLM返回空内容，使用降级追问")
        else:
            question = question.strip()
            task_log.append(f"clarification: round={round_num+1} question生成成功")

        return {
            "pending_clarification": question,
            "clarification_round":   round_num + 1,
            "task_log": task_log,
            "errors":   errors,
        }

    except Exception as e:
        # ── 异常4：LLM 调用异常（超时/网络/额度耗尽等），降级追问 ──────
        # 不让用户卡在无响应状态，流程必须能继续往下走
        question = _fallback_question(missing_slots)
        errors.append({"node": "clarification", "error": str(e), "recoverable": True})
        task_log.append(f"clarification: exception {e}，使用降级追问")
        return {
            "pending_clarification": question,
            "clarification_round":   round_num + 1,
            "task_log": task_log,
            "errors":   errors,
        }


async def _generate_clarification_question(
    llm_client,
    raw_query: str,
    missing_slots: list[str],
    round_num: int,
    history: list[dict],
) -> str:
    """生成一次性追问，覆盖所有缺失字段。一次 LLM 调用，无工具、无决策。"""

    history_text = ""
    if history:
        history_lines = [
            f"第{i+1}轮 - 问：{h.get('question','')} 答：{h.get('answer','')}"
            for i, h in enumerate(history)
        ]
        history_text = "## 之前的追问历史\n" + "\n".join(history_lines)

    messages = [
        {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"""\
用户原始需求：「{raw_query}」
当前缺失的信息字段：{missing_slots}
当前是第 {round_num + 1} 轮追问。
{history_text}

请生成一句自然的追问，覆盖上述缺失字段。只输出追问文本，不要任何其他内容。
"""},
    ]

    resp = await llm_client.call(trace_id="clarification", messages=messages)
    content = resp.choices[0].message.content
    return content or ""


def _fallback_question(missing_slots: list[str]) -> str:
    """LLM 调用失败或返回空时的规则降级追问，保证流程不卡死。"""
    labels = [_LABEL_MAP.get(s, s) for s in missing_slots]
    return f"为了帮你规划，还需要确认一下：{('、'.join(labels))}，方便告诉我吗？"


def _apply_fallbacks(intent: dict, missing_slots: list[str]) -> dict:
    """超过追问轮次上限时，给缺失字段填默认值，让流程能继续。"""
    filled = dict(intent)
    location     = dict(filled.get("location") or {})
    participants = dict(filled.get("participants") or {})
    time_info    = dict(filled.get("time") or {})

    for slot in missing_slots:
        default = _DEFAULT_FALLBACKS.get(slot)
        if default is None:
            continue
        if slot == "origin_area":
            location["origin_area_hint"] = default
        elif slot == "people_count":
            participants["people_count"] = default
        elif slot == "start_time":
            time_info["start_time"] = default
        elif slot == "end_time":
            time_info["end_time"] = default
        elif slot == "scenario":
            filled["scenario"] = default

    filled["location"]      = location
    filled["participants"]  = participants
    filled["time"]          = time_info
    filled["missing_slots"] = []   # 清空，表示已"解决"（即使是兜底）
    filled["clarification_needed"] = False
    return filled