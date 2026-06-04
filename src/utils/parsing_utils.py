# src/utils/parsing_utils.py
import re
from pydantic import BaseModel, Field

class CandidatePlanDraftItem(BaseModel):
    id: str = Field(..., description="candidate id such as plan_1")
    title: str = Field(..., description="short plan title")
    steps: list[dict] = Field(default_factory=list, description="ordered plan steps")
    activity_id: str = Field(default="", description="legacy single activity id")
    restaurant_ids: list[str] = Field(default_factory=list, description="legacy restaurant ids")
    reasoning: list[str] = Field(default_factory=list, description="2-3 short reasons")


class CandidatePlanDraftEnvelope(BaseModel):
    candidates: list[CandidatePlanDraftItem] = Field(default_factory=list)


def _extract_step_phases(steps: list[dict]) -> set[str]:
    phases: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        phase = step.get("phase")
        if isinstance(phase, str) and phase.strip():
            phases.add(phase.strip())
    return phases


def _steps_cover_daypart(steps: list[dict], daypart: str) -> bool:
    """语义校验守卫：检验计划骨架是否完整覆盖了所需的时段"""
    if daypart != "全天":
        return True
    phases = _extract_step_phases(steps)
    return {"morning", "lunch", "afternoon", "dinner"}.issubset(phases)


def _extract_json_object(raw_text: str) -> str | None:
    """提取文本中的标准 JSON 字符串块"""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return None
    return text[start_index : end_index + 1]