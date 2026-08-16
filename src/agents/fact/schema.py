# agents/fact/schema.py
from __future__ import annotations

from pydantic import BaseModel, Field


class POIItem(BaseModel):
    """精炼后的 POI —— 只保留规划够用的字段。
    详情(商圈/电话/营业时间)留到选定方案后，拿 id 再查。
    """
    id: str = Field(description="高德 POI id，后续查详情的钥匙")
    name: str = Field(description="POI 名称")
    type: str = Field(default="", description="POI 类型")
    location: str = Field(default="", description="经纬度 'lng,lat'")
    rating: str = Field(default="", description="评分，可能为空")

    # 【必须显式声明，否则静默丢失】_synthesize 传进来的是 dict，
    # pydantic 默认忽略未声明的键——不加这个字段，cost 会被安静地
    # 扔掉，不报错、不告警，下游只看到"所有餐厅都没有价格"。
    # 这正是 cost 覆盖率排查过程中已经踩过一次的那类失败模式
    # （当时是字段层级找错，这次是字段没声明，表现完全一样：
    # 全空、无异常、看起来像"数据源没有"）。
    #
    # 默认 None 不是 0.0：None 的语义是"高德没给价格"，0 的语义是
    # "免费"。混淆二者会让 `cost <= 预算` 这类判断静默通过。
    cost: float | None = Field(
        default=None, description="人均消费(元)，未知为 null，不是 0"
    )

    environment: str = Field(default="unknown", description="indoor/outdoor/mixed/unknown")
    eta_minutes: int | None = Field(default=None, description="从出发地驾车分钟数，未知为 null")


class WaypointItem(BaseModel):
    """途径小需求 POI(DQ/奶茶等)。not_found=True 表示搜了但附近没有。"""
    id: str = Field(default="", description="POI id，not_found 时为空")
    name: str = Field(default="", description="POI 名称")
    location: str = Field(default="", description="经纬度")
    eta_minutes: int | None = Field(default=None)
    keyword: str = Field(description="对应的途径需求关键词，如 'DQ'")
    not_found: bool = Field(default=False, description="附近是否未找到")


class Weather(BaseModel):
    city: str = Field(default="")
    day_weather: str = Field(default="", description="如 '晴' '小雨'")
    day_temp: str = Field(default="", description="温度℃")
    day_wind: str = Field(default="")


class FactData(BaseModel):
    """Fact Gathering Agent 的输出契约(信封里的 data 部分)。

    交付一个"够规划用的精炼事实包"：出发地解析、天气、
    活动/餐厅候选池(已含 eta 和价格)、途径小需求。
    """
    origin_city: str = Field(default="", description="解析出的城市")
    origin_coordinates: str = Field(default="", description="出发地坐标 'lng,lat'")

    weather: Weather = Field(default_factory=Weather)

    activities: list[POIItem] = Field(
        default_factory=list, description="活动候选池(精炼)"
    )
    restaurants: list[POIItem] = Field(
        default_factory=list, description="餐厅候选池(精炼)"
    )
    waypoints: list[WaypointItem] = Field(
        default_factory=list, description="途径小需求 POI"
    )