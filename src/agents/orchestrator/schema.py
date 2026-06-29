from __future__ import annotations
from pydantic import BaseModel, Field


class OrchestratorData(BaseModel):
    """Orchestrator Agent 的输出契约。

    只记录被调用的操作结果,未调用的字段保持默认空值。
    orchestrator_node 根据是否为空决定要不要写回 state。
    """
    updated_fact: dict = Field(default_factory=dict)
    updated_planning: dict = Field(default_factory=dict)
    updated_evaluation: dict = Field(default_factory=dict)
    action_log: list[str] = Field(default_factory=list)