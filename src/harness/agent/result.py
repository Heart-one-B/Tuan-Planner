from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 子 Agent 的统一状态信号。上层(流程 / Orchestrator)据此做决策。
#   ok      理想结果,可直接用
#   partial 只拿到部分,可用但不完整
#   empty   确实查不到(参数没错,就是没有)
#   error   持续失败,无法完成
AgentStatus = Literal["ok", "partial", "empty", "error"]


class AgentResult(BaseModel):
    """所有子 Agent 的统一返回契约。

    这是多 Agent 协作的核心规范:无论 Fact / Planning / Evaluation,
    还是未来任何子 Agent,都返回这个形状。上层永远知道自己会收到什么。

    - status:  最小决策信号,让上层判断该怎么走下一步
    - summary: 一句话结论,供上层快速理解、供 trace 记录
    - data:    结构化结果本体,形状由各 Agent 的 output_schema 决定
    """

    status: AgentStatus = Field(description="ok / partial / empty / error")
    summary: str = Field(default="", description="一句话结论")
    data: dict[str, Any] = Field(default_factory=dict, description="结构化结果本体")

    def as_tool_output(self) -> str:
        """序列化成字符串,供上层 Agent 当作工具返回结果塞回上下文。

        这是 'Agent 即工具' 的关键:子 Agent 的结构化结果在这里
        变回字符串,和普通工具的返回值走同一条路径回到上层模型。
        """
        import json
        return json.dumps(
            {"status": self.status, "summary": self.summary, "data": self.data},
            ensure_ascii=False,
        )