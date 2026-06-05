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
        "clarification_round":   prev_round + 1,
    }


def receive_clarification_node(state: AgentState) -> AgentState:
    """
    接收阶段：外部框架把用户回复注入 state["user_reply"] 后调用此节点。

    职责：
    1. 把本轮回复追加进 conversation_turns
    2. 把完整对话历史拼合为新的 user_input，交还给 IntentNode 重新解析
    3. 把上一轮已识别的关键实体（waypoints / explicit_types）作为已知信息
       注入 user_input，防止模型重新解析时丢失这些实体
    4. 清空挂起状态

    写入 state 的字段：
      user_input              str        合并后的完整输入（含已知实体提示）
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

    pending_q = (state.get("pending_clarification") or "").strip()
    if pending_q:
        turns.append(f"[系统追问]{pending_q}")
    turns.append(f"[用户回复]{user_reply}")
    combined_input = "\n".join(t for t in turns if t.strip())

    # ── 把上一轮已识别的关键实体注入，防止重新解析时丢失 ──────────────────
    # 从上一轮 intent 解析结果里提取已知实体
    prev_intent: dict = state.get("intent") or {}

    known_hints: list[str] = []

    # 1. waypoint_requests（DQ / 星巴克 / 奶茶等）
    prev_waypoints: list[dict] = prev_intent.get("waypoint_requests") or []
    if prev_waypoints:
        wp_descs = [
            f"{w.get('raw_text', '')}→keyword={w.get('keyword', '')}"
            for w in prev_waypoints if w.get("keyword")
        ]
        if wp_descs:
            known_hints.append(f"已识别途径需求（waypoint_requests）：{', '.join(wp_descs)}，请保留在输出中，不要丢失。")

    # 2. activity_explicit_types（用户明确点名的活动）
    prev_explicit_acts: list[str] = prev_intent.get("activity_explicit_types") or []
    if prev_explicit_acts:
        known_hints.append(f"已识别明确活动类型（activity_explicit_types）：{', '.join(prev_explicit_acts)}，请保留在输出中。")

    # 3. restaurant_explicit_types（用户明确点名的餐厅类型）
    prev_explicit_rests: list[str] = prev_intent.get("restaurant_explicit_types") or []
    if prev_explicit_rests:
        known_hints.append(f"已识别明确餐厅类型（restaurant_explicit_types）：{', '.join(prev_explicit_rests)}，请保留在输出中。")

    if known_hints:
        hint_block = "\n".join(f"[已知实体]{h}" for h in known_hints)
        combined_input = f"{combined_input}\n{hint_block}"

    print(f"[Clarification Node] 接收回复，合并输入（共 {len(turns)} 轮）")
    if known_hints:
        print(f"[Clarification Node] 注入已知实体提示：{known_hints}")

    return {
        "user_input":            combined_input,
        "conversation_turns":    turns,
        "pending_clarification": "",
        "user_reply":            "",
        "clarification_needed":  False,
        "missing_slots":         [],
        "follow_up_message":     "",
        "current_asking_slot":   None,
    }