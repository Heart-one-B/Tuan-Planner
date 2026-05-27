# src/graph/nodes/clarification_node.py
from src.graph.state import AgentState

def clarification_node(state: AgentState) -> AgentState:
    """Clarification Node：追问关键缺口。"""
    follow_up = state.get("follow_up_message")
    if not isinstance(follow_up, str) or not follow_up.strip():
        follow_up = "为了继续规划，请补充一下关键信息。"

    clarification_round = state.get("clarification_round", 0)
    if isinstance(clarification_round, bool) or not isinstance(clarification_round, int):
        clarification_round = 0

    print(f"[Clarification Node] {follow_up}")
    extra_input = input("> ")
    while not extra_input.strip():
        extra_input = input("> ")
    extra_input = extra_input.strip()

    turns = list(state.get("conversation_turns", []))
    if not turns:
        first_turn = state.get("user_input", "")
        if isinstance(first_turn, str) and first_turn.strip():
            turns = [first_turn]

    turns.append(extra_input)
    combined_input = "\n".join([turn for turn in turns if isinstance(turn, str) and turn.strip()])

    return {
        "clarification_round": clarification_round + 1,
        "conversation_turns": turns,
        "user_input": combined_input,
        "clarification_needed": False,
        "missing_slots": {},
        "follow_up_message": "",
    }