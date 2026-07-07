from __future__ import annotations

from agents.intent.agent import IntentAgent
from src.model.factory import build_llm_client
from src.graph.state import AgentState


async def intent_node(state: AgentState) -> dict:
    task_log = list(state.get("task_log") or [])
    errors   = list(state.get("errors") or [])

    user_input = (state.get("user_input") or "").strip()
    feedback_route = state.get("feedback_route") or ""

    # ── clarify_reply 路径：把原始需求 + 历史澄清 + 这次回复拼成完整输入 ──
    # 不能只用这次回复单独解析，因为"国贸附近，四个人"脱离上下文
    # 无法判断这是在回答什么问题；必须带着原始需求重新理解整句话。
    if feedback_route == "clarify_reply":
        history = state.get("clarification_history") or []
        # 上一轮的 intent 里有 raw_query，这是用户最初的完整需求
        prev_intent = state.get("intent") or {}
        original_query = prev_intent.get("raw_query") or user_input

        history_text = "\n".join(
            f"[追问]{h.get('question','')} [回答]{h.get('answer','')}"
            for h in history
        )

        combined_input = (
            f"{original_query}\n{history_text}"
            if history_text else original_query
        )
        task_log.append(
            f"intent: clarify_reply模式，拼接{len(history)}轮历史重新解析"
        )
    else:
        combined_input = user_input

    # ── 防御性检查：拼接后的输入为空，无法解析 ──────────────────────────
    if not combined_input.strip():
        errors.append({
            "node": "intent",
            "error": "用户输入为空，无法解析意图",
            "recoverable": False,
        })
        task_log.append("intent: combined_input为空，跳过解析")
        return {"task_log": task_log, "errors": errors}

    try:
        result = await IntentAgent(llm_client=build_llm_client()).parse(
            user_input=combined_input,
            trace_id=state.get("session_id"),
        )
        intent = result.model_dump()

        # ── 防御性检查：clarify_reply 模式下，raw_query 应保留最初的需求 ──
        # 防止 Intent Agent 把 raw_query 重写成只包含这次回复的片段
        if feedback_route == "clarify_reply" and not intent.get("raw_query"):
            intent["raw_query"] = original_query

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