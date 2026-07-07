from __future__ import annotations

import json
import re

from src.model.factory import build_llm_client
from src.graph.state import AgentState

_SYSTEM_PROMPT = """\
你是对话路由模块。你会看到"当前方案"和"用户刚说的话"，判断用户是在
对当前方案提修改意见，还是在提出一个和当前方案无关的全新需求。

只输出 JSON，不输出其他内容：
{
  "route": "new_plan" | "adjust" | "chat",
  "reason": "一句话说明判断依据"
}

adjust：用户的话是针对当前方案里某个具体环节的修改、补充或调整
（不管改动大小，只要还是基于当前方案在提意见，就算 adjust）

new_plan：用户的话和当前方案完全无关，是一个全新的、独立的出行需求
（比如出行目的、出发地、日期这类根本前提变了）

chat：闲聊、确认（"可以了"、"谢谢"）、单纯问答，不涉及方案修改
"""


async def routing_node(state: AgentState) -> dict:
    user_input = (state.get("user_input") or "").strip()
    task_log = list(state.get("task_log") or [])
    errors   = list(state.get("errors") or [])

    # ── 优先级最高：上一轮在等待澄清回复 ──────────────────────────────
    pending_clarification = state.get("pending_clarification") or ""
    if pending_clarification:
        history = list(state.get("clarification_history") or [])
        history.append({"question": pending_clarification, "answer": user_input})
        task_log.append(
            f"routing: 检测到待澄清状态，本轮输入作为澄清回复，"
            f"累计{len(history)}轮历史"
        )
        return {
            "feedback_route":        "clarify_reply",
            "clarification_history": history,
            "pending_clarification": "",
            "task_log": task_log,
        }

    # ── 没有已选中方案 → 新会话，默认 new_plan，不调LLM ─────────────────
    agent_outputs = state.get("agent_outputs") or {}
    eval_data = (agent_outputs.get("evaluation") or {}).get("data") or {}
    selected_plan_id = eval_data.get("selected_plan_id") or ""

    if not selected_plan_id:
        task_log.append("routing: no existing plan, route=new_plan")
        return {"feedback_route": "new_plan", "task_log": task_log}

    # ── 有已选中方案 → 把方案摘要作为上下文，让模型真正比较，不是猜测 ────
    planning_data = (agent_outputs.get("planning") or {}).get("data") or {}
    candidates = planning_data.get("candidates") or []
    selected_plan = next((c for c in candidates if c.get("id") == selected_plan_id), {})
    plan_summary = _summarize_plan(selected_plan)

    try:
        llm = build_llm_client()
        resp = await llm.call(
            trace_id=state.get("session_id") or "routing",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"当前方案：{plan_summary}\n\n用户刚说：{user_input}"},
            ],
        )
        raw = resp.choices[0].message.content or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {}
        route = result.get("route") or "adjust"  # 有方案时缺省更该偏向adjust，不是new_plan

        if route not in ("new_plan", "adjust", "chat"):
            task_log.append(f"routing: LLM返回非法route={route!r}，降级为adjust")
            route = "adjust"
        else:
            task_log.append(f"routing: route={route} reason={result.get('reason', '')}")

    except Exception as e:
        errors.append({"node": "routing", "error": str(e), "recoverable": True})
        route = "adjust"  # 有方案时LLM调用失败，降级为adjust比new_plan更安全
        task_log.append(f"routing: LLM failed ({e})，fallback to adjust")

    return {
        "feedback_route": route,
        "task_log": task_log,
        "errors": errors,
    }


def _summarize_plan(plan: dict) -> str:
    """把当前方案压缩成一行摘要，给路由判断提供真实比对上下文。"""
    timeline = plan.get("timeline") or []
    items = [t.get("label", "") for t in timeline if t.get("label")]
    return "、".join(items) if items else "（无）"