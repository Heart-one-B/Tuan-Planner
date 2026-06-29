from __future__ import annotations
from pydantic import BaseModel, Field


class PlanScore(BaseModel):
    candidate_id: str = Field(default="")
    score: float = Field(default=0.0)
    reason: str = Field(default="")
    passed_validation: bool = Field(default=True)
    violations: list[str] = Field(default_factory=list)


class EvalData(BaseModel):
    """Evaluation Agent 的输出契约。

    selected_plan_id 非空 → 有合格方案
    selected_plan_id 为空 → 全部不合格,上层决定怎么处理
    """
    selected_plan_id: str = Field(default="")
    selected_score: float = Field(default=0.0)
    selected_reason: str = Field(default="")
    all_scores: list[PlanScore] = Field(default_factory=list)