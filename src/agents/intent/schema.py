from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ParticipantsInfo(BaseModel):
    people_count: int | None = Field(None)
    has_child: bool = Field(False)
    child_age: int | None = Field(None)


class TimeInfo(BaseModel):
    date_label: str | None = Field(None)
    start_time: str | None = Field(None)
    end_time: str | None = Field(None)
    time_phrase: str | None = Field(None)


class LocationInfo(BaseModel):
    origin_area_hint: str | None = Field(None)
    location_text: str | None = Field(None)


class PreferencesInfo(BaseModel):
    distance_preference: str | None = Field(None)
    diet_preference: list[str] = Field(default_factory=list)
    activity_style: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)


class WaypointRequest(BaseModel):
    raw_text: str
    keyword: str
    time_hint: str | None = Field(None)


class IntentResult(BaseModel):
    is_leisure_planning: bool
    scenario: Literal["family", "friends", "couple", "team", "unknown", "none"]
    participants: ParticipantsInfo = Field(default_factory=ParticipantsInfo)
    time: TimeInfo = Field(default_factory=TimeInfo)
    location: LocationInfo = Field(default_factory=LocationInfo)
    preferences: PreferencesInfo = Field(default_factory=PreferencesInfo)

    restaurant_keywords: list[str] = Field(default_factory=list)
    restaurant_explicit_types: list[str] = Field(default_factory=list)
    activity_keywords: list[str] = Field(default_factory=list)
    activity_explicit_types: list[str] = Field(default_factory=list)
    waypoint_requests: list[WaypointRequest] = Field(default_factory=list)

    need_retrieval: bool = Field(default=False)
    clarification_needed: bool = Field(default=False)
    missing_slots: list[str] = Field(default_factory=list)
    current_asking_slot: str | None = Field(None)
    follow_up_message: str | None = Field(None)
    raw_query: str = Field(default="")