# src/graph/nodes/intent_node.py
from datetime import datetime
from src.graph.state import AgentState
from src.agent.intent_agent import IntentAgent
from src.utils.state_utils import _append_error


def intent_node(state: AgentState) -> AgentState:
    """Intent Node：识别规划意图、场景、时间、偏好。
    合并了原：
      - intent_node (基础意图识别)
      - time_normalize_node (时间语义规范化)
    """
    print("[Intent Node] 识别规划意图、场景、时间、偏好并规范化...")

    # 1. 识别基础意图
    try:
        intent = IntentAgent().parse(
            state["user_input"],
            state.get("runtime_origin_area", ""),
        )
    except Exception as exc:
        return _append_error(state, f"Intent parse failed: {exc}")

    # 2. 规范化时间偏好
    try:
        time_info = intent.get("time") or {}
        date_label = time_info.get("date_label") if isinstance(time_info.get("date_label"), str) else ""
        daypart = time_info.get("daypart") if isinstance(time_info.get("daypart"), str) else ""
        time_phrase = time_info.get("time_phrase") if isinstance(time_info.get("time_phrase"), str) else ""

        normalized_date_label = date_label
        normalized_daypart = daypart
        normalized_time_phrase = time_phrase or (f"{date_label}{daypart}" if date_label and daypart else "")

        now = datetime.now()
        now_minutes = now.hour * 60 + now.minute
        cutoff_minutes = 18 * 60 if normalized_daypart in {"下午", "上午", "全天"} else 21 * 60
        if normalized_date_label == "今天" and now_minutes > cutoff_minutes:
            normalized_date_label = "明天"
            if normalized_time_phrase:
                normalized_time_phrase = normalized_time_phrase.replace("今天", "明天", 1)

        if normalized_daypart == "上午":
            base_start_minutes = 9 * 60 + 30
        elif normalized_daypart == "晚上":
            base_start_minutes = 18 * 60 + 30
        elif normalized_daypart == "全天":
            base_start_minutes = 9 * 60 + 30
        else:
            base_start_minutes = 14 * 60
        if normalized_date_label == "今天" and normalized_daypart in {"下午",
                                                                      "晚上"} and now_minutes + 15 > cutoff_minutes:
            normalized_date_label = "明天"

        normalized_time = {
            "normalized_date_label": normalized_date_label,
            "normalized_daypart": normalized_daypart,
            "normalized_time_phrase": normalized_time_phrase,
            "base_start_minutes": base_start_minutes,
            "current_minutes": now_minutes,
            "current_time": now.strftime("%H:%M"),
        }

        return {
            "intent": intent,
            "is_leisure_planning": intent.get("is_leisure_planning"),
            "need_retrieval": intent.get("need_retrieval"),
            "clarification_needed": intent.get("clarification_needed"),
            "missing_slots": intent.get("missing_slots"),
            "follow_up_message": intent.get("follow_up_message"),
            "normalized_time": normalized_time,
            "time_normalization_result": normalized_time,
        }
    except Exception as exc:
        print(f"[Intent Node][WARN] 时间规范化失败: {exc}")
        state_update = _append_error(state, f"Time Normalize failed: {exc}")
        state_update["intent"] = intent
        return state_update