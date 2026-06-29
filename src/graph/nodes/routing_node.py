from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.model.factory import build_llm_client
from src.graph.state import AgentState

_SYSTEM_PROMPT = """\
你是对话路由模块,判断用户这句话的意图,只输出 JSON 不输出其他内容。

{
  "route": "new_plan" | "adjust" | "chat",
  "reason": "一句话说明判断依据"
}

new_plan: 全新规划需求,或换了场景/日期/人员
adjust:   对当前方案提意见、要求修改、补充需求
chat:     闲聊、确认、感谢、单纯问答
"""


async def routing_node(state: AgentState) -> dict:
    user_input = (state.get("user_input") or "").strip()
    task_log = list(state.get("task_log") or [])
    errors = list(state.get("errors") or [])

    # 没有已有方案直接走 new_plan,不调 LLM
    agent_outputs = state.get("agent_outputs") or {}
    if not agent_outputs.get("planning"):
        task_log.append("routing: no existing plan, route=new_plan")
        return {
            "feedback_route": "new_plan",
            "task_log": task_log,
        }

    try:
        llm = build_llm_client()
        resp = await llm.call(
            trace_id=state.get("session_id") or "routing",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        raw = resp.choices[0].message.content or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {}
        route = result.get("route") or "new_plan"
        task_log.append(f"routing: route={route} reason={result.get('reason', '')}")
    except Exception as e:
        errors.append({"node": "routing", "error": str(e), "recoverable": True})
        route = "new_plan"
        task_log.append("routing: LLM failed, fallback to new_plan")

    return {
        "feedback_route": route,
        "task_log": task_log,
        "errors": errors,
    }