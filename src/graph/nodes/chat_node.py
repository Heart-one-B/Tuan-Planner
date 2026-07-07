# src/graph/nodes/chat_node.py

from __future__ import annotations

from src.model.factory import build_llm_client
from src.graph.state import AgentState

CHAT_SYSTEM_PROMPT = """\
你是一个本地生活行程规划助手。用户刚才说的话不是一个具体的规划需求，
可能是打招呼、闲聊、或者单纯的提问。

请自然地回应，并在合适的时候引导用户告诉你他们的出行需求
（比如时间、人数、想做什么）。回复保持简短，1-3句话即可。

不要假装自己能做规划之外的事情（比如订票、查违章）。
"""


async def chat_node(state: AgentState) -> dict:
    """处理闲聊/非规划类输入，单轮 LLM 回复，无状态依赖。

    不调用任何业务 Agent，不读 plan_context / fact_data，
    因为这个分支本身就是"用户没有提出规划需求"。
    """
    task_log = list(state.get("task_log") or [])
    errors   = list(state.get("errors") or [])

    user_input = (state.get("user_input") or "").strip()

    # ── 防御性检查：空输入直接给默认欢迎语，不调 LLM ──────────────────
    if not user_input:
        task_log.append("chat: 空输入，返回默认欢迎语")
        return {
            "final_message": "你好，我可以帮你规划本地行程，告诉我想做什么、什么时候、几个人吧。",
            "task_log": task_log,
            "errors":   errors,
        }

    try:
        llm = build_llm_client()
        resp = await llm.call(
            trace_id=state.get("session_id") or "chat",
            messages=[
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        reply = (resp.choices[0].message.content or "").strip()

        # ── 防御性检查：LLM 返回空内容，降级为通用回复 ──────────────────
        if not reply:
            reply = "我在的，有什么出行计划想让我帮你安排吗？"
            task_log.append("chat: LLM返回空内容，使用降级回复")
        else:
            task_log.append("chat: 回复生成成功")

        return {
            "final_message": reply,
            "task_log": task_log,
            "errors":   errors,
        }

    except Exception as e:
        # ── 异常：LLM 调用失败，降级为固定文案，不让用户卡住 ──────────
        errors.append({"node": "chat", "error": str(e), "recoverable": True})
        task_log.append(f"chat: exception {e}，使用降级回复")
        return {
            "final_message": "抱歉刚才走神了，能再说一次你想安排的行程吗？",
            "task_log": task_log,
            "errors":   errors,
        }