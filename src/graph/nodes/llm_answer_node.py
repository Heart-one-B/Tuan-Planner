# src/graph/nodes/llm_answer_node.py
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.model.factory import get_chat_model
from src.utils.state_utils import _append_error

_LLM_ANSWER_FALLBACK = "该问题不属于本地生活规划范畴，建议直接咨询通用助手或搜索引擎。"

def llm_answer_node(state: AgentState) -> AgentState:
    """LLM Answer Node: 意图非休闲任务直答。"""
    user_input = state.get("user_input", "")
    print("[LLM Answer Node] 检测到非本地生活规划任务，调用 LLM 直接回答...")
    try:
        response = get_chat_model().invoke([HumanMessage(content=user_input)])
        answer = getattr(response, "content", None)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("LLM 返回为空或不是字符串")
        print(f"[OK] LLM 直答完成，长度={len(answer)}")
        return {"llm_answer": answer}
    except Exception as exc:
        print(f"[WARN] LLM 直答失败，使用固定 fallback: {exc}")
        update = _append_error(state, f"LLM Answer node failed: {exc}")
        update["llm_answer"] = _LLM_ANSWER_FALLBACK
        return update