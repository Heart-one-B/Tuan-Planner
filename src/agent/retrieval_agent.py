# src/agent/retrieval_agent.py
from typing import Any


class RetrievalAgent:
    """RAG 知识库检索智能体 (当前为占位桩)"""
    
    def retrieve(self, query: str, keywords: list[str]) -> dict[str, Any]:
        print(f"[Retrieval Agent] 开始 RAG 检索，关键词={keywords}")
        
        # 因为我们已经切换到真实的高德 LBS 链路，不再需要本地的假数据知识库。
        # 实际生产中，这里应当是请求您的向量数据库 (Vector DB)。
        # 当前作为占位符，直接返回空结果，避免假数据干扰真实的高德 API 规划。
        print("[Retrieval Agent] 检索完成：pois=0 条，notes=0 条")
        
        return {
            "pois": [],
            "notes": []
        }