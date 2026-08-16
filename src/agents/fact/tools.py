# agents/fact/tools.py
from __future__ import annotations

import asyncio
import json

from agents.fact.cost import extract_cost
from harness.tools.tool_definition import ToolDefinition
from harness.tools.exceptions import RetryableError, DataNotFoundError
from src.tools.cached_amap_client import CachedAmapClient

# ══════════════════════════════════════════════════════════════════
# 请求量控制（实测驱动，不是拍脑袋）
# ══════════════════════════════════════════════════════════════════
#
# 【问题】高德 MCP 有 QPS 限制，超了返回一个**字符串**：
#     "Around Search failed: CUQPS_HAS_EXCEEDED_THE_LIMIT"
# 不是异常、不是 HTTP 错误码、不是 {"error": ...} 结构。解析层拿到它
# 得到空列表 → 被当成"这个词搜不到" → status=ok, count=0
# → 用户得到一个错误的世界模型（"附近没有火锅"，实际是没查到）。
#
# 【实测】scripts/probe_search_stability.py：
#     串行 5 次 × 6 个词  → 全部成功，0 失败
#     同词并发 8 次        → 3~5 次被限流
#     6 个不同词同时发     → 只有 2 个成功
# 最后一种正是 FactAgent._run_searches 的真实形态。
#
# 【为什么搜索直接串行，不用并发 2】
# 6 并发只活 2 个，说明 QPS 窗口很窄；并发 2 大概率仍会丢，
# 丢了要重试，重试同样占额度——不如一开始就串行。
# 代价可算：一次规划 4-6 个搜索 × 400ms ≈ 2.4 秒，可接受。
#
# registry 的 MCPClient 有 max_concurrent=2，但显然没兜住
# （实测 6 个请求都打出去了），所以限流必须加在这一层。
_SEARCH_SEM = asyncio.Semaphore(1)      # 搜索串行
_SEARCH_GAP = 0.15                      # 每次搜索后的间隔，给 QPS 窗口留余量
_DETAIL_SEM = asyncio.Semaphore(2)      # 详情并发上限

# 信号量必须是**模块级**：FactAgent 每次 run 都新建 FactToolset，
# 实例级信号量等于没有限制——真正的并发发生在 _run_searches 的
# gather 层，跨越多个 toolset 实例。

_DETAIL_TOP_N = 5      # 每次搜索只对前 N 家餐厅拉详情
_POI_PER_SEARCH = 8    # 每次搜索保留的 POI 数（不变，保多样性）


class FactToolset:
    """Fact Agent 的工具集 + 会话级坐标暂存（异步版本）。"""

    RESTAURANT_TYPE = "050000"

    def __init__(self):
        self._api = CachedAmapClient()
        self._pool: dict[str, dict] = {}
        self._pool_lock = asyncio.Lock()
        self.origin_city: str = ""
        self.origin_coordinates: str = ""

    async def _refine(self, poi: dict, *, want_detail: bool = False) -> dict:
        """标准化一条原始 POI，存坐标进 pool。

        不判断 environment —— 那需要理解 POI 名称/类型的语义，
        用关键词匹配做（"馆"→室内，"公园"→室外）是规则模拟语义理解，
        判断质量很差。交给 PlanningAgent 结合完整上下文自己判断。

        【want_detail 的取舍，需要显式接受的代价】
        详情调用只为拿 cost（人均消费）。改动前对每家餐厅都拉，
        峰值 8 搜索 × 8 家 = 64 个并发请求，稳定触发 QPS 限流。
        而这些详情大部分白拉——搜到 30-40 家，Planning 最终只用 1-2 家。

        现在只对**前 N 家餐厅**拉，后面的 cost 为 None。
        代价明确：候选池里有一部分餐厅没有价格数据。

        这个代价可以接受，因为 cost=None 的语义是"未知"而不是 0
        （见 cost.py）——预算断言会把它计入"无价格数据"那一桶，
        **不会被误判成满足预算**。如果 cost 缺失会导致静默通过，
        这个取舍就不成立了。

        活动完全不拉详情：活动不需要价格，location 搜索结果里已有。
        """
        pid = poi.get("id") or ""
        location = poi.get("location") or ""
        rating = poi.get("rating") or ""
        cost: float | None = None

        if pid and want_detail:
            # 信号量在这里而不是 gather 层：这是唯一真正打到高德的
            # 地方，限在源头最准确。
            async with _DETAIL_SEM:
                try:
                    detail = await self._api.poi_detail(pid)
                    if isinstance(detail, dict):
                        location = location or detail.get("location") or ""
                        rating = detail.get("rating") or rating
                        cost = extract_cost(detail)
                except Exception:
                    # 详情拉不到不该让整条 POI 作废——名称和 id 仍然有用，
                    # cost 保持 None（诚实的未知）。
                    pass

        refined = {
            "id": pid,
            "name": poi.get("name") or "",
            "type": poi.get("type") or "",
            "location": location,
            "rating": rating,
            "cost": cost,          # float | None，None = 未知，不是 0
            "eta_minutes": None,
        }
        if pid and location:
            async with self._pool_lock:
                self._pool[pid] = {"location": location}
        return refined

    async def geocode(self, address: str, city: str = "") -> str:
        """出发地 → 城市 + 坐标。

        city 交给 CachedAmapClient 处理（它知道"maps_geo 的 city
        参数无效、要拼进地址"这个数据源怪癖，那是适配层的职责）。
        """
        result = await self._api.geocode(address, city=city)
        resolved_city = result.get("city") or ""
        coord = result.get("coordinates") or ""
        if not resolved_city and not coord:
            hint = f"（已限定城市：{city}）" if city else "（未限定城市）"
            raise DataNotFoundError(
                f"无法解析地址 '{address}'{hint}，请换个更具体的地名。"
            )
        self.origin_city = resolved_city
        self.origin_coordinates = coord
        return json.dumps(
            {"city": resolved_city, "coordinates": coord,
             "district": result.get("district", "")},
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
        """搜索附近 POI。keywords 必须是**单个字符串**，不是列表——
        它会被原样拼进高德的 keywords 查询参数。调用方拿到的如果是
        列表，责任在调用方展开成多次调用，不在这里猜。

        【搜索串行 + 间隔】见文件顶部说明。sleep 放在信号量**内部**
        ——放外面就没有节流效果，下一个请求会立刻进来。

        【详情只拉前 N 家】搜索结果已按相关性排序，前几家是最可能
        被 Planning 选中的。为全部 8 家拉详情而最终只用 1-2 家，
        是 6 倍的浪费——在有 QPS 限制的外部依赖上，浪费直接转化为
        失败率。
        """
        if not self.origin_coordinates and not self.origin_city:
            raise RetryableError("出发地未解析，请先调用 geocode，再搜索 POI。")

        async with _SEARCH_SEM:
            pois = await self._api.search_pois(
                keywords,
                location=self.origin_coordinates,
                city=self.origin_city,
                radius="5000",
                poi_type=self.RESTAURANT_TYPE if is_restaurant else "",
            )
            await asyncio.sleep(_SEARCH_GAP)

        if not pois:
            return json.dumps(
                {"keyword": keywords, "count": 0, "with_cost": 0, "pois": [],
                 "hint": "未搜到结果，建议换一个更通用的关键词再试。"},
                ensure_ascii=False,
            )

        candidates = [p for p in pois[:_POI_PER_SEARCH] if p.get("id")]
        # 只有餐厅需要 cost，且只拉前 N 家。活动一律不拉详情。
        detail_n = _DETAIL_TOP_N if is_restaurant else 0

        refined_list = list(await asyncio.gather(*(
            self._refine(p, want_detail=(i < detail_n))
            for i, p in enumerate(candidates)
        )))

        with_cost = sum(1 for r in refined_list if r["cost"] is not None)
        return json.dumps(
            {
                "keyword": keywords,
                "count": len(refined_list),
                # 如实报告有几家带价格：下游（尤其预算约束）需要知道
                # "这个池子里有多少家是可判定的"，而不是把 cost 缺失
                # 当成"没有超预算"。
                "with_cost": with_cost,
                "pois": refined_list,
            },
            ensure_ascii=False,
        )

    async def get_distance(self, poi_id: str) -> str:
        async with self._pool_lock:
            in_pool = poi_id in self._pool

        if not in_pool:
            raise DataNotFoundError(
                f"poi_id '{poi_id}' 不在已搜索的结果中，"
                f"只能对 search_pois 返回过的 POI 计算车程。"
            )
        if not self.origin_coordinates:
            raise DataNotFoundError("缺少出发地坐标，无法计算车程。")

        async with _DETAIL_SEM:
            detail = await self._api.poi_detail(poi_id)
        if not detail or not detail.get("location"):
            raise DataNotFoundError(f"poi_id '{poi_id}' 无法获取坐标。")

        result = await self._api.distance(self.origin_coordinates, detail["location"])
        eta = result.get("eta_minutes") if isinstance(result, dict) else None
        return json.dumps({"poi_id": poi_id, "eta_minutes": eta}, ensure_ascii=False)

    async def get_distance_batch(self, poi_ids: list[str]) -> str:
        """批量算车程：一次调用覆盖全部 POI。

        这是本文件里唯一**天然省请求**的设计——N 个 POI 一次调用。
        _refine 的详情拉取正好相反（N 个 POI = N 次调用），
        所以限制加在那边。
        """
        if not self.origin_coordinates:
            raise DataNotFoundError("出发地坐标未解析，请先调用 geocode。")

        async with self._pool_lock:
            valid = [(pid, self._pool[pid]["location"])
                     for pid in poi_ids
                     if pid in self._pool and self._pool[pid].get("location")]

        if not valid:
            return json.dumps({"results": []}, ensure_ascii=False)

        origins_str = "|".join(loc for _, loc in valid)
        try:
            mcp = await self._api._get_mcp()
            async with _SEARCH_SEM:
                raw = await mcp.call("maps_distance", {
                    "origins": origins_str,
                    "destination": self.origin_coordinates,
                    "type": "1",
                })
                await asyncio.sleep(_SEARCH_GAP)
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

    当前 ReWOO 架构下没有消费者——它是给"FactAgent 改成 harness Agent
    的 ReAct 循环"预留的桥梁。read_only 全部标 True：这五个都是纯查询，
    没有任何副作用。这个字段目前只是声明，等 harness 落地并行工具执行
    之后才会被真正消费。
    """
    return [
        ToolDefinition(
            name="geocode",
            description="把出发地名称解析成城市和坐标。必须最先调用。",
            parameters={
                "address": {"type": "string", "description": "出发地名称"},
                "city": {"type": "string", "description": "所在城市，可选"},
            },
            required=["address"],
            func=toolset.geocode,
            read_only=True,
        ),
        ToolDefinition(
            name="get_weather",
            description="查询城市当天天气。",
            parameters={"city": {"type": "string", "description": "城市名"}},
            required=["city"],
            func=toolset.get_weather,
            read_only=True,
        ),
        ToolDefinition(
            name="search_pois",
            description="用一个关键词搜索附近POI。一次只能传一个关键词字符串。",
            parameters={
                "keywords": {"type": "string", "description": "单个搜索关键词，不是列表"},
                "is_restaurant": {"type": "boolean", "description": "是否搜餐厅"},
            },
            required=["keywords"],
            func=toolset.search_pois,
            read_only=True,
        ),
        ToolDefinition(
            name="get_distance",
            description="计算某个POI到出发地的驾车分钟数。",
            parameters={"poi_id": {"type": "string", "description": "POI id"}},
            required=["poi_id"],
            func=toolset.get_distance,
            read_only=True,
        ),
        ToolDefinition(
            name="get_distance_batch",
            description="批量计算多个POI到出发地的驾车分钟数。",
            parameters={
                "poi_ids": {"type": "array", "items": {"type": "string"},
                            "description": "POI id列表"},
            },
            required=["poi_ids"],
            func=toolset.get_distance_batch,
            read_only=True,
        ),
    ]