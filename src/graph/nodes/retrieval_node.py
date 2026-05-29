# src/graph/nodes/retrieval_node.py
from src.graph.state import AgentState
from src.agent.retrieval_agent import RetrievalAgent
from src.utils.state_utils import _append_error

def retrieval_node(state: AgentState) -> AgentState:
    """Retrieval Node: 检索背景知识/候选数据。"""
    print("[Retrieval Node] 开始 mock RAG 检索...")
    try:
        result = RetrievalAgent().retrieve(state.get("intent", {}))
        return {"retrieval_context": result}
    except Exception as exc:
        print(f"[Retrieval Node][WARN] 检索节点异常，跳过并记录错误: {exc}")
        return _append_error(state, f"Retrieval node failed: {exc}")