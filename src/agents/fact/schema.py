from __future__ import annotations

from pydantic import BaseModel, Field


class POIItem(BaseModel):
    """精炼后的 POI —— 只保留规划够用的字段。
    详情(商圈/电话/营业时间)留到选定方案后,拿 id 再查。
    """
    id: str = Field(description="高德 POI id,后续查详情的钥匙")
    name: str = Field(description="POI 名称")
    type: str = Field(default="", description="POI 类型")
    location: str = Field(default="", description="经纬度 'lng,lat'")
    rating: str = Field(default="", description="评分,可能为空")
    environment: str = Field(default="unknown", description="indoor/outdoor/mixed/unknown")
    eta_minutes: int | None = Field(default=None, description="从出发地驾车分钟数,未知为 null")


class WaypointItem(BaseModel):
    """途径小需求 POI(DQ/奶茶等)。not_found=True 表示搜了但附近没有。"""
    id: str = Field(default="", description="POI id,not_found 时为空")
    name: str = Field(default="", description="POI 名称")
    location: str = Field(default="", description="经纬度")
    eta_minutes: int | None = Field(default=None)
    keyword: str = Field(description="对应的途径需求关键词,如 'DQ'")
    not_found: bool = Field(default=False, description="附近是否未找到")


class Weather(BaseModel):
    city: str = Field(default="")
    day_weather: str = Field(default="", description="如 '晴' '小雨'")
    day_temp: str = Field(default="", description="温度℃")
    day_wind: str = Field(default="")


class FactData(BaseModel):
    """Fact Gathering Agent 的输出契约(信封里的 data 部分)。

    交付一个'够规划用的精炼事实包':出发地解析、天气、
    活动/餐厅候选池(已含 eta 和室内外)、途径小需求。
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