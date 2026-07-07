from __future__ import annotations

import asyncio
import json

from harness.tools.tool_definition import ToolDefinition
from harness.tools.exceptions import RetryableError, DataNotFoundError
from src.tools.cached_amap_client import CachedAmapClient


class FactToolset:
    """Fact Agent 的工具集 + 会话级坐标暂存（异步版本）。"""

    RESTAURANT_TYPE = "050000"

    def __init__(self):
        self._api = CachedAmapClient()
        self._pool: dict[str, dict] = {}
        self._pool_lock = asyncio.Lock()
        self.origin_city: str = ""
        self.origin_coordinates: str = ""

    async def _refine(self, poi: dict) -> dict:
        """标准化一条原始 POI，存坐标进 pool。

        不再判断 environment —— 这个判断需要理解POI名称/类型的
        语义，之前用关键词匹配做（"馆"→室内，"公园"→室外）是
        规则模拟语义理解，判断质量很差。现在交给 Planning Agent
        在真正需要判断室内室外时（天气不好时），结合完整上下文
        自己判断，不在这里预先算一个不准的标签。
        """
        pid = poi.get("id") or ""
        location = poi.get("location") or ""
        rating = poi.get("rating") or ""

        if pid and not location:
            try:
                detail = await self._api.poi_detail(pid)
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
            "eta_minutes": None,
        }
        if pid and location:
            async with self._pool_lock:
                self._pool[pid] = {"location": location}
        return refined

    async def geocode(self, address: str) -> str:
        result = await self._api.geocode(address)
        city = result.get("city") or ""
        coord = result.get("coordinates") or ""
        if not city and not coord:
            raise DataNotFoundError(f"无法解析地址 '{address}'，请换个更具体的地名。")
        self.origin_city = city
        self.origin_coordinates = coord
        return json.dumps(
            {"city": city, "coordinates": coord, "district": result.get("district", "")},
            ensure_ascii=False,
        )

    async def get_weather(self, city: str) -> str:
        result = await self._api.weather(city)
        return json.dumps(
            {
                "city": result.get("city") or city,
                "day_weather": result.get("day_weather") or "",
                "day_temp": result.get("day_temp") or "",
                "day_wind": result.get("day_wind") or "",
            },
            ensure_ascii=False,
        )

    async def search_pois(self, keywords: str, is_restaurant: bool = False) -> str:
        if not self.origin_coordinates and not self.origin_city:
            raise RetryableError("出发地未解析，请先调用 geocode，再搜索 POI。")

        pois = await self._api.search_pois(
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
            await self._refine(p)
            for p in pois[:8]
            if p.get("id")
        ]
        return json.dumps(
            {"keyword": keywords, "count": len(refined_list), "pois": refined_list},
            ensure_ascii=False,
        )

    async def get_distance(self, poi_id: str) -> str:
        async with self._pool_lock:
            in_pool = poi_id in self._pool

        if not in_pool:
            raise DataNotFoundError(
                f"poi_id '{poi_id}' 不在已搜索的结果中，只能对 search_pois 返回过的 POI 计算车程。"
            )
        if not self.origin_coordinates:
            raise DataNotFoundError("缺少出发地坐标，无法计算车程。")

        detail = await self._api.poi_detail(poi_id)
        if not detail or not detail.get("location"):
            raise DataNotFoundError(f"poi_id '{poi_id}' 无法获取坐标。")

        result = await self._api.distance(self.origin_coordinates, detail["location"])
        eta = result.get("eta_minutes") if isinstance(result, dict) else None
        return json.dumps({"poi_id": poi_id, "eta_minutes": eta}, ensure_ascii=False)

    async def get_distance_batch(self, poi_ids: list[str]) -> str:
        if not self.origin_coordinates:
            raise DataNotFoundError("出发地坐标未解析,请先调用 geocode。")

        async with self._pool_lock:
            valid = [(pid, self._pool[pid]["location"])
                     for pid in poi_ids
                     if pid in self._pool and self._pool[pid].get("location")]

        if not valid:
            return json.dumps({"results": []}, ensure_ascii=False)

        origins_str = "|".join(loc for _, loc in valid)
        try:
            mcp = await self._api._get_mcp()
            raw = await mcp.call("maps_distance", {
                "origins": origins_str,
                "destination": self.origin_coordinates,
                "type": "1",
            })
            items = (raw or {}).get("results") or []
            results = []
            for i, (pid, _) in enumerate(valid):
                r = items[i] if i < len(items) else {}
                eta = round(int(r["duration"]) / 60) if r.get("duration") else None
                results.append({"poi_id": pid, "eta_minutes": eta})
        except Exception:
            results = [{"poi_id": pid, "eta_minutes": None} for pid, _ in valid]

        valid_ids = {pid for pid, _ in valid}
        for pid in poi_ids:
            if pid not in valid_ids:
                results.append({"poi_id": pid, "eta_minutes": None, "hint": "无坐标"})

        return json.dumps({"results": results}, ensure_ascii=False)


def build_fact_tools(toolset: FactToolset) -> list[ToolDefinition]:
    """把 FactToolset 的方法包成 ToolDefinition 列表。

    Plan-and-Execute 架构下，DAG 的每个节点具体怎么执行是可选的
    ——可以用 FunctionExecutor（纯函数调用），也可以用
    StructuredAgentExecutor（包装成 ReAct 子 Agent）。
    目前所有节点都用 FunctionExecutor，这个函数是给"未来某个
    节点需要 ReAct 子 Agent 执行"预留的桥梁。
    """
    return [
        ToolDefinition(
            name="geocode",
            description="把出发地名称解析成城市和坐标。必须最先调用。",
            parameters={"address": {"type": "string", "description": "出发地名称"}},
            required=["address"],
            func=toolset.geocode,
        ),
        ToolDefinition(
            name="get_weather",
            description="查询城市当天天气。",
            parameters={"city": {"type": "string", "description": "城市名"}},
            required=["city"],
            func=toolset.get_weather,
        ),
        ToolDefinition(
            name="search_pois",
            description="用一个关键词搜索附近POI。",
            parameters={
                "keywords": {"type": "string", "description": "搜索关键词"},
                "is_restaurant": {"type": "boolean", "description": "是否搜餐厅"},
            },
            required=["keywords"],
            func=toolset.search_pois,
        ),
        ToolDefinition(
            name="get_distance",
            description="计算某个POI到出发地的驾车分钟数。",
            parameters={"poi_id": {"type": "string", "description": "POI id"}},
            required=["poi_id"],
            func=toolset.get_distance,
        ),
        ToolDefinition(
            name="get_distance_batch",
            description="批量计算多个POI到出发地的驾车分钟数。",
            parameters={
                "poi_ids": {"type": "array", "items": {"type": "string"}, "description": "POI id列表"},
            },
            required=["poi_ids"],
            func=toolset.get_distance_batch,
        ),
    ]