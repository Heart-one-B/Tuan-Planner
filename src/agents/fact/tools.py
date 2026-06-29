from __future__ import annotations

import json

from harness.tools.tool_definition import ToolDefinition
from harness.tools.exceptions import RetryableError, DataNotFoundError
from src.tools.cached_amap_client import CachedAmapClient


class FactToolset:
    """Fact Agent 的工具集 + 会话级坐标暂存。

    pool 的唯一职责：search_pois 搜到的 POI 按 id 存坐标，
    供 get_distance 按 id 查坐标算车程。
    不参与 finish 组装——模型直接从上下文摘抄精炼 POI 填进 FactData。

    每次 FactAgent.run() 新建一个实例，保证会话隔离。
    """

    RESTAURANT_TYPE = "050000"

    def __init__(self):
        self._api = CachedAmapClient()
        # id -> {"location": "lng,lat"}，仅供 get_distance 查坐标
        self._pool: dict[str, dict] = {}
        self.origin_city: str = ""
        self.origin_coordinates: str = ""

    # ── 内部：精炼 POI + 存坐标进 pool ──────────────────────────────────────
    def _refine(self, poi: dict, *, infer_env: bool) -> dict:
        """标准化一条原始 POI，存坐标进 pool，返回精炼版供上下文展示。"""
        pid = poi.get("id") or ""
        location = poi.get("location") or ""
        rating = poi.get("rating") or ""

        if pid and not location:
            try:
                detail = self._api.poi_detail(pid)
                if isinstance(detail, dict):
                    location = detail.get("location") or ""
                    rating = (
                            (detail.get("biz_ext") or {}).get("rating")
                            or detail.get("rating")
                            or poi.get("rating")
                            or ""
                    )
            except Exception:
                pass

        refined = {
            "id": pid,
            "name": poi.get("name") or "",
            "type": poi.get("type") or "",
            "location": location,
            "rating": rating,
            "environment": (
                CachedAmapClient.infer_environment(poi) if infer_env else "unknown"
            ),
            "eta_minutes": None,
        }
        # pool 只存 location，其余字段模型从上下文摘抄
        if pid and location:
            self._pool[pid] = {"location": location}
        return refined

    # ── 工具实现 ─────────────────────────────────────────────────────────────

    def geocode(self, address: str) -> str:
        """把出发地名称解析成城市 + 坐标，后续所有搜索依赖它。"""
        result = self._api.geocode(address)
        city = result.get("city") or ""
        coord = result.get("coordinates") or ""
        if not city and not coord:
            raise DataNotFoundError(f"无法解析地址 '{address}'，请换个更具体的地名。")
        self.origin_city = city
        self.origin_coordinates = coord
        return json.dumps(
            {
                "city": city,
                "coordinates": coord,
                "district": result.get("district", ""),
            },
            ensure_ascii=False,
        )

    def get_weather(self, city: str) -> str:
        """查城市当天天气，用于判断活动是否需要优先选室内。"""
        result = self._api.weather(city)
        return json.dumps(
            {
                "city": result.get("city") or city,
                "day_weather": result.get("day_weather") or "",
                "day_temp": result.get("day_temp") or "",
                "day_wind": result.get("day_wind") or "",
            },
            ensure_ascii=False,
        )

    def search_pois(self, keywords: str, is_restaurant: bool = False) -> str:
        """用一个关键词搜索 POI（一次一个词）。

        搜几轮、换什么词、结果够不够——由 Agent 自己决定，工具只执行一次搜索。
        返回精炼列表（含 id/name/type/rating/location/environment），
        Agent 可直接把满意的条目纳入 finish 的 FactData。
        """
        if not self.origin_coordinates and not self.origin_city:
            raise RetryableError(
                "出发地未解析，请先调用 geocode，再搜索 POI。"
            )

        pois = self._api.search_pois(
            keywords,
            location=self.origin_coordinates,
            city=self.origin_city,
            radius="5000",
            poi_type=self.RESTAURANT_TYPE if is_restaurant else "",
        )

        if not pois:
            return json.dumps(
                {"keyword": keywords, "count": 0, "pois": [],
                 "hint": "未搜到结果，建议换一个更通用的关键词再试。"},
                ensure_ascii=False,
            )

        refined_list = [
            self._refine(p, infer_env=not is_restaurant)
            for p in pois[:8]
            if p.get("id")
        ]
        return json.dumps(
            {"keyword": keywords, "count": len(refined_list), "pois": refined_list},
            ensure_ascii=False,
        )

    def get_distance(self, poi_id: str) -> str:
        if poi_id not in self._pool:
            raise DataNotFoundError(
                f"poi_id '{poi_id}' 不在已搜索的结果中，"
                "只能对 search_pois 返回过的 POI 计算车程。"
            )
        if not self.origin_coordinates:
            raise DataNotFoundError("缺少出发地坐标，无法计算车程。")

        detail = self._api.poi_detail(poi_id)
        if not detail or not detail.get("location"):
            raise DataNotFoundError(f"poi_id '{poi_id}' 无法获取坐标。")

        result = self._api.distance(self.origin_coordinates, detail["location"])
        eta = result.get("eta_minutes") if isinstance(result, dict) else None
        return json.dumps({"poi_id": poi_id, "eta_minutes": eta}, ensure_ascii=False)

    def get_distance_batch(self, poi_ids: list[str]) -> str:
        if not self.origin_coordinates:
            raise DataNotFoundError("出发地坐标未解析,请先调用 geocode。")

        valid = [(pid, self._pool[pid]["location"])
                 for pid in poi_ids
                 if pid in self._pool and self._pool[pid].get("location")]

        if not valid:
            return json.dumps({"results": []}, ensure_ascii=False)

        # 一次调用:origins 用 | 分隔,destination 是出发地
        origins_str = "|".join(loc for _, loc in valid)
        try:
            raw = self._api._amap.maps_distance(
                origins=origins_str,
                destination=self.origin_coordinates,
                type_="1"
            )
            # 高德返回 results 列表,按顺序对应 origins
            items = (raw or {}).get("results") or []
            results = []
            for i, (pid, _) in enumerate(valid):
                r = items[i] if i < len(items) else {}
                eta = round(int(r["duration"]) / 60) if r.get("duration") else None
                results.append({"poi_id": pid, "eta_minutes": eta})
        except Exception as e:
            results = [{"poi_id": pid, "eta_minutes": None} for pid, _ in valid]

        # 补上 pool 里没坐标的
        valid_ids = {pid for pid, _ in valid}
        for pid in poi_ids:
            if pid not in valid_ids:
                results.append({"poi_id": pid, "eta_minutes": None,
                                "hint": "无坐标"})

        return json.dumps({"results": results}, ensure_ascii=False)


def build_fact_tools(toolset: FactToolset) -> list[ToolDefinition]:
    """把 FactToolset 的方法包成 ToolDefinition 列表注册进 ToolExecutor。"""
    return [
        ToolDefinition(
            name="geocode",
            description=(
                "把出发地名称（如'国贸''三里屯''望京'）解析成城市和坐标。"
                "必须最先调用，后续所有 search_pois 和 get_distance 都依赖它。"
            ),
            parameters={
                "address": {
                    "type": "string",
                    "description": "出发地名称，尽量具体，如'北京国贸'而非'北京'",
                },
            },
            required=["address"],
            func=toolset.geocode,
        ),
        ToolDefinition(
            name="get_weather",
            description=(
                "查询城市当天天气（天气状况、温度、风向）。"
                "用于判断活动是否优先选室内场所。"
                "geocode 完成后调用，用解析出的 city 字段。"
            ),
            parameters={
                "city": {
                    "type": "string",
                    "description": "城市名，来自 geocode 返回的 city 字段，如'北京'",
                },
            },
            required=["city"],
            func=toolset.get_weather,
        ),
        ToolDefinition(
            name="search_pois",
            description=(
                "用一个关键词搜索附近 POI（每次只能一个关键词）。\n"
                "- 搜活动：is_restaurant=false，关键词如'剧本杀''咖啡馆''美术馆'\n"
                "- 搜餐厅：is_restaurant=true，关键词如'火锅''日料''川菜'\n"
                "结果不理想时（太少、类型不符）你应自行判断并换词再次调用。\n"
                "返回的每条 POI 含 id/name/type/rating/location/environment，"
                "满意的条目可直接纳入 finish 的 FactData。"
            ),
            parameters={
                "keywords": {
                    "type": "string",
                    "description": "单个搜索关键词，如'剧本杀'",
                },
                "is_restaurant": {
                    "type": "boolean",
                    "description": "是否搜餐厅，默认 false",
                },
            },
            required=["keywords"],
            func=toolset.search_pois,
        ),
        ToolDefinition(
            name="get_distance",
            description=(
                "计算某个已搜到的 POI 到出发地的驾车分钟数。\n"
                "只能用于 search_pois 已返回过的 POI id。\n"
                "建议对进入候选的 POI 都算一遍，填入 FactData 的 eta_minutes。"
            ),
            parameters={
                "poi_id": {
                    "type": "string",
                    "description": "search_pois 返回的 POI id",
                },
            },
            required=["poi_id"],
            func=toolset.get_distance,
        ),
        ToolDefinition(
            name="get_distance_batch",
            description=(
                "批量计算多个已搜到的 POI 到出发地的驾车分钟数。"
                "算车程时优先用这个，传入所有候选 POI 的 id 列表，一次完成。"
                "只能用于 search_pois 已返回过的 POI id。"
            ),
            parameters={
                "poi_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "POI id 列表，来自 search_pois 的返回结果",
                },
            },
            required=["poi_ids"],
            func=toolset.get_distance_batch,
        ),
    ]