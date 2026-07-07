from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StepItem(BaseModel):
    """行程中的一个步骤。"""
    step_id: str = Field(default="")
    category: Literal["meal", "snack_drink", "activity"] = Field(
        default="activity",
        description="POI性质分类：meal=能吃饱的正餐场所，"
                     "snack_drink=甜品饮品类场所，activity=其他活动场所"
    )
    phase: str = Field(default="")           # morning/lunch/afternoon/dinner/evening
    poi_type: str = Field(default="")        # activity/restaurant/waypoint
    poi_id: str = Field(default="")
    label: str = Field(default="")
    duration_minutes: int = Field(default=60)
    duration_flex: list[int] = Field(default_factory=list)
    # 时刻求解器回填
    start_time: str = Field(default="")
    end_time: str = Field(default="")


class TimelineItem(BaseModel):
    """面向展示的时间轴条目。"""
    time: str = Field(default="")
    end_time: str = Field(default="")
    item: str = Field(default="")            # POI 名称
    label: str = Field(default="")
    type: str = Field(default="")            # 如 activity_afternoon
    ref_type: str = Field(default="")        # activity/restaurant
    ref_id: str = Field(default="")          # poi_id
    duration: int | None = Field(default=None)


class CandidatePlan(BaseModel):
    """单个候选方案。"""
    id: str = Field(default="")
    title: str = Field(default="")
    steps: list[StepItem] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    # _build_step_index 产出
    activities: list[dict] = Field(default_factory=list)
    activity: dict = Field(default_factory=dict)
    secondary_activity: dict = Field(default_factory=dict)
    restaurants: list[dict] = Field(default_factory=list)
    restaurant: dict = Field(default_factory=dict)
    timeline: list[TimelineItem] = Field(default_factory=list)


class PlanData(BaseModel):
    """Planning Agent 的输出契约。

    包含 3 个候选方案,上层(Evaluation Agent / Orchestrator)
    从这里拿方案做评估和选择。
    """
    plan_mode: str = Field(default="activity_plus_meal")
    candidates: list[CandidatePlan] = Field(default_factory=list)