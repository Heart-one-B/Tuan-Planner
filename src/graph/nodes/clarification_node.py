"""
    IntentNode
        ↓ clarification_needed=True
    clarification_node          # 写入 pending_clarification，图暂停/返回前端
        ↓ 外部框架注入 user_reply
    receive_clarification_node  # 合并输入，清空挂起，图路由回 IntentNode
        ↓
    IntentNode（重新解析，直到 missing_slots 为空）
"""
from __future__ import annotations

from src.graph.state import AgentState

def clarification_node(state: AgentState) -> AgentState:
    """
    发问阶段：将追问消息写入 pending_clarification，供前端读取后展示给用户。
    图在此节点后暂停，等待外部把用户回复写入 state["user_reply"]。

    写入 state 的字段：
      pending_clarification   str  向用户展示的追问消息
      clarification_round     int  追问轮次 +1
    """
    follow_up = (state.get("follow_up_message") or "").strip()
    if not follow_up:
        follow_up = "为了帮你规划，还需要补充一点信息。"

    prev_round = state.get("clarification_round", 0)
    if not isinstance(prev_round, int) or isinstance(prev_round, bool):
        prev_round = 0

    print(f"[Clarification Node] 发问（轮次 {prev_round + 1}）：{follow_up!r}")

    return {
        "pending_clarification": follow_up,
        "clarification_round": prev_round + 1,
    }


def receive_clarification_node(state: AgentState) -> AgentState:
    """
    接收阶段：外部框架把用户回复注入 state["user_reply"] 后调用此节点。

    职责：
    1. 把本轮回复追加进 conversation_turns
    2. 把完整对话历史拼合为新的 user_input，交还给 IntentNode 重新解析
    3. 清空挂起状态

    写入 state 的字段：
      user_input              str        合并后的完整输入
      conversation_turns      list[str]  含本轮回复的对话历史
      pending_clarification   str        清空为 ""
      user_reply              str        清空为 ""
      clarification_needed    bool       重置为 False（IntentNode 会重新判断）
      missing_slots           list       重置为 []
      follow_up_message       str        重置为 ""
      current_asking_slot     None       重置为 None
    """
    user_reply = (state.get("user_reply") or "").strip()
    if not user_reply:
        print("[Clarification Node][WARN] user_reply 为空，跳过合并。")
        return {}

    turns: list[str] = list(state.get("conversation_turns") or [])
    if not turns:
        first = (state.get("user_input") or "").strip()
        if first:
            turns = [first]

    turns.append(user_reply)
    combined_input = "\n".join(t for t in turns if t.strip())

    print(f"[Clarification Node] 接收回复，合并输入（共 {len(turns)} 轮）")

    return {
        "user_input": combined_input,
        "conversation_turns": turns,
        "pending_clarification": "",
        "user_reply": "",
        "clarification_needed": False,
        "missing_slots": [],
        "follow_up_message": "",
        "current_asking_slot": None,
    }