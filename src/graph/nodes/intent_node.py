from src.agent.intent_agent import IntentAgent
from src.graph.state import AgentState
from src.utils.state_utils import _append_error


def intent_node(state: AgentState) -> AgentState:
    """
    Intent Node：解析用户输入，识别规划意图、场景、时间、偏好。

    写入 state 的字段：
      intent                完整 IntentResult 字典
      is_leisure_planning   bool
      need_retrieval        bool
      clarification_needed  bool
      missing_slots         list[str]  按优先级排列的缺口槽位名
      current_asking_slot   str | None 本轮要追问的槽位
      follow_up_message     str | None 追问话术
    """
    print("[Intent Node] 开始解析意图...")

    try:
        result = IntentAgent().parse(state["user_input"])
        intent_dict = result.model_dump()
    except Exception as exc:
        print(f"[Intent Node][ERROR] 意图解析失败: {exc}")
        return _append_error(state, f"Intent parse failed: {exc}")

    return {
        "intent": intent_dict,
        "is_leisure_planning": intent_dict.get("is_leisure_planning"),
        "clarification_needed": intent_dict.get("clarification_needed"),
        "missing_slots": intent_dict.get("missing_slots", []),
        "current_asking_slot": intent_dict.get("current_asking_slot"),
        "follow_up_message": intent_dict.get("follow_up_message"),
    }