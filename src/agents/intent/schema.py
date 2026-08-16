# agents/intent/schema.py
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

    # 【新增】出发地所在城市。
    #
    # 修的是一个实测到的真 bug：「从春熙路出发」被解析到了云南昭通
    # （更早还出现过「陆家嘴」→ 昆明）。根因是 maps_geo 调用时没传
    # city 参数——高德 schema 里有这个可选参数，不给城市范围，
    # 全国重名地点就是碰运气。
    #
    # 用户不一定会说城市（"从春熙路出发"就没说），此时留 None，
    # 由下游用默认城市兜底（个人助手知道用户常驻哪儿，这是合理的
    # 产品假设，不是猜测）。
    origin_city: str | None = Field(
        None, description="出发地所在城市，用户明确提到才填，没提留 null"
    )


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

    # 用户对"吃什么"的表达方式，决定 FactAgent 该精确搜索还是自由发挥。
    #
    # 三值而不是布尔："没点名"和"没提吃饭"是两种不同的下游行为
    # （前者要搜、后者不搜），合并成布尔会丢掉这个区别。
    #
    # 默认 open：三种误判的下游代价不对称。
    #   explicit 误判成 open → 多搜几个品类，用户从更大的池子里挑
    #   open 误判成 none     → 完全不安排餐厅，用户没饭吃（代价最大）
    restaurant_intent: Literal["explicit", "open", "none"] = Field(
        default="open",
        description=(
            "explicit=用户点名了具体想吃什么（可检索的菜系/品类/店名）；"
            "open=想吃饭但没指定，或只说了口味风格；none=完全没提到吃饭"
        ),
    )

    # 【新增】与 restaurant_intent 完全对称的活动侧分流。
    #
    # 修的是 L2-04 实测暴露的缺口：用户说"就吃个午饭"，
    # _build_plan_context 靠字符串匹配（"只吃饭"/"找餐厅"/"约饭"）
    # 判断要不要安排活动——"就吃个午饭"不在那张表里，于是
    # plan_mode 判成 activity_plus_meal，FactAgent 又因为
    # "plan_mode 要求活动但模型没规划活动搜索"盲补了"公园、美术馆"。
    #
    # 一个错误的分类判断，制造了两次多余搜索，还稀释了候选池。
    # 字符串匹配穷举不了用户的说法，这个判断本来就该模型做。
    activity_intent: Literal["explicit", "open", "none"] = Field(
        default="open",
        description=(
            "explicit=用户点名了具体活动类型（博物馆/剧本杀/KTV 等）；"
            "open=想找地方玩但没指定；none=完全没有活动需求（只吃饭）"
        ),
    )

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