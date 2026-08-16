"""
CachedAmapClient：高德 MCP API 的本地缓存层（异步版本）。

三级架构：内存/文件缓存 → MCP 远程调用 → 断网 fallback。
底层通过 registry 获取共享的长连接 MCPClient，不再每次调用都握手。

weather 走独立的和风天气 API（不消耗高德额度），不经过 MCP。

═══════════════════════════════════════════════════════════════
【三条铁律，全部是踩过坑之后加的】

  ① 空结果不写缓存
     "搜到 0 家" 和 "这次调用失败了" 在返回值上长得一模一样，
     而后者是暂时的。把空结果缓存 1 天，等于让一次偶发失败在
     之后 24 小时内持续复现——而且表现为"这个词就是搜不到"。
     实测代价：keywords='火锅' 被毒了整整一天。

  ② 缓存 key 只包含**真正发出去的参数**
     多一个维度 → 同一个查询有两个缓存条目 → 其中一个被污染时
     另一个还是好的 → 表现为"参数 A 搜不到、参数 B 搜得到"，
     而两者本该完全等价。实测：poi_type 只在 text_search 分支
     使用，around_search 分支不传它，却被放进了 key。

  ③ 错误响应必须被识别，不能降级为空结果
     高德 MCP 超 QPS 时返回的是一个**字符串**：
         "Around Search failed: CUQPS_HAS_EXCEEDED_THE_LIMIT"
     不是异常、不是 HTTP 错误码、不是 {"error": ...} 结构。
     不识别它就会把"被限流"当成"搜不到"——这正是 P0-A：
     系统声称成功、实际失败、无告警。
═══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from harness.mcp.registry import get_mcp_client
from harness.mcp.mcp_client import MCPClient
from harness.tools.exceptions import RetryableError
from src.utils.config_handler import tools_conf
from src.utils.path_tool import get_abs_path

_CACHE_FILE = get_abs_path("data/amap_cache.json")
_POI_DETAIL_CACHE = get_abs_path("data/poi_detail_cache.json")

_TTL: dict[str, timedelta] = {
    "geocode": timedelta(days=30),
    "weather": timedelta(hours=6),
    "search": timedelta(days=1),
    "detail": timedelta(days=7),
    "distance": timedelta(hours=2),
}

_MEM_CACHE: dict[str, dict] = {}
_MEM_CACHE_LOCK = threading.RLock()


class CachedAmapClient:
    """高德 MCP API + 本地文件缓存的异步客户端。"""

    def __init__(self):
        self._mcp_url = tools_conf.get("amap_mcp_url", "")
        self._cache_only: bool = bool(tools_conf.get("amap_use_cache_only", False))
        self._mcp: MCPClient | None = None

    async def _get_mcp(self) -> MCPClient:
        if self._mcp is None:
            self._mcp = await get_mcp_client(self._mcp_url)
        return self._mcp

    # ── 缓存基础设施 ──────────────────────────────────────────────────

    @staticmethod
    def _key(op: str, *parts: str) -> str:
        raw = "|".join(parts)
        short = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{op}:{short}"

    def _load(self) -> dict[str, dict]:
        with _MEM_CACHE_LOCK:
            if _MEM_CACHE:
                return _MEM_CACHE
            try:
                with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    _MEM_CACHE.update(data)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            return _MEM_CACHE

    def _save(self) -> None:
        with _MEM_CACHE_LOCK:
            Path(_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_MEM_CACHE, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat()

    def _fresh(self, entry: dict, op: str) -> bool:
        ts = entry.get("cached_at", "")
        if not ts:
            return False
        try:
            return datetime.utcnow() - datetime.fromisoformat(ts) <= _TTL[op]
        except (ValueError, KeyError):
            return False

    def _get(self, key: str, op: str) -> Any:
        with _MEM_CACHE_LOCK:
            entry = self._load().get(key)
            if isinstance(entry, dict) and self._fresh(entry, op):
                return entry["data"]
            return None

    def _set(self, key: str, data: Any) -> None:
        with _MEM_CACHE_LOCK:
            cache = self._load()
            cache[key] = {"cached_at": self._now(), "data": data}
        self._save()

    # ── 错误识别 ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_api_error(raw) -> str | None:
        """识别 MCP 以**字符串形式**返回的错误（铁律③）。

        高德 MCP 超 QPS 时返回：
            "Around Search failed: CUQPS_HAS_EXCEEDED_THE_LIMIT"

        原解析逻辑是 `raw.get("pois") or ... or []`——字符串走不到
        任何分支，pois 保持 []，**限流被解析成"这个词搜不到"**。

        后果链（P0-A 的完整实例，每一环都不报错）：
            外部限流 → 错误以非结构化形式返回 → 解析层未识别
            → 降级为空结果 → status=ok, count=0
            → 空结果被缓存 1 天 → 该关键词整天搜不到
            → 用户看到"这附近没有火锅"，得到一个错误的世界模型

        实测（scripts/probe_search_stability.py）：6 个关键词同时
        发出，只有 2 个成功；而系统全程报告正常。

        限流本身消除不了（额度不由客户端决定），但**静默可以消除**。
        这就是 P0-A 的定义：不是消除失败，是消除"声称成功的失败"。
        """
        if isinstance(raw, str):
            low = raw.lower()
            if "failed" in low or "limit" in low or "error" in low:
                return raw[:300]
        if isinstance(raw, dict):
            # 有些形态是 {"content": "...failed..."} 或 {"error": "..."}
            for key in ("error", "message", "content"):
                v = raw.get(key)
                if isinstance(v, str) and (
                    "failed" in v.lower() or "LIMIT" in v.upper()
                ):
                    return v[:300]
        return None

    # ── 地理编码 ──────────────────────────────────────────────────────

    async def geocode(self, address: str, city: str = "") -> dict:
        """地址 → 城市 + 坐标。两阶段：先原样查，城市不对再拼前缀重试。

        ══════════════════════════════════════════════════════════
        每一行都建立在对照实验上，不是文档
        （scripts/probe_geocode.py / probe_geocode2.py）
        ══════════════════════════════════════════════════════════

        【实验一：city 参数无效】

            address='春熙路' city=''     → 10 个结果，云南昭通排第一
            address='春熙路' city='成都' → 10 个结果，**逐字相同**

        maps_geo 的 city 参数被完全忽略——MCP schema 里声明了，
        实际不生效。这是本项目第二次撞到"schema 声明 ≠ 实际生效"
        （第一次是 maps_around_search 的 types）。

        纪律：**跨系统边界传参数时，验证方式是对照实验，不是读文档**。

        【实验二：拼前缀有效，但会误伤】

            四川大学江安校区 / 成都四川大学江安校区  坐标逐字相同 ✅
            双流机场       / 成都双流机场         坐标逐字相同 ✅
            太古里         → 新疆库车市（偏 2254km）
            成都太古里      → 成都锦江区 ✅
            人民南路        → 新疆图木舒克市（偏 2489km）
            天府广场        → 成都青羊区 ✅
            成都天府广场     → **无结果** 🔴

        最后一条是关键：加上**正确的城市**反而查不到。说明高德不是
        "城市前缀过滤 + 地名匹配"，而是整串做模糊匹配。所以无条件
        拼接是拆东墙补西墙：修好商圈俗称，坏掉部分地标。

        【策略：先不拼，城市不对才拼】
        样本里 4/8 不拼就对（机构名、交通枢纽、地标），零额外开销；
        另外 4 个多一次调用。"天府广场"这种"不拼对、拼了错"的，
        第一次就命中，不会走到重试。
        """
        first = await self._geocode_once(address)

        if not city:
            return first

        got = first.get("city") or ""
        if got and city.replace("市", "") in got:
            return first

        prefixed = address if city in address else f"{city}{address}"
        if prefixed == address:
            return first

        second = await self._geocode_once(prefixed)
        got2 = second.get("city") or ""
        if got2 and city.replace("市", "") in got2:
            return second

        # 两次都不对：返回信息更多的那个，如实告警。
        # 不静默改写 city 字段成期望值——坐标还是错的，只是错得更
        # 隐蔽，下游拿着错坐标搜出一堆无关 POI，日志里却看不出问题。
        better = second if second.get("coordinates") else first
        print(f"[CachedAmapClient][WARN] geocode('{address}', 期望城市='{city}') "
              f"两次尝试都未命中：原样→'{got or '无结果'}'，"
              f"加前缀→'{got2 or '无结果'}'。返回坐标="
              f"{better.get('coordinates') or '空'}")
        return better

    async def _geocode_once(self, query: str) -> dict:
        """单次地理编码 + 缓存。

        缓存 key 用**实际发出的查询串**（可能已拼过城市前缀），
        不是调用方传的原始地址。否则"春熙路"在成都和昆明会共用
        同一个缓存条目，第一次查询的结果污染之后所有城市。
        """
        key = self._key("geocode", query)
        hit = self._get(key, "geocode")
        if hit is not None:
            return hit

        result: dict = {
            "city": "", "coordinates": "", "district": "",
            "formatted_address": query,
        }
        if self._cache_only:
            print("[CachedAmapClient] cache_only=true，跳过 geocode API")
            return result

        try:
            mcp = await self._get_mcp()
            # 刻意不传 city：实测无效。传一个不生效的参数，会让读代码
            # 的人以为城市范围已经收窄了，比不传更有误导性。
            raw = await mcp.call("maps_geo", {"address": query})

            api_err = self._detect_api_error(raw)
            if api_err:
                print(f"[CachedAmapClient][WARN] geocode({query!r}) API 错误: {api_err}")
                return result

            data = raw if isinstance(raw, dict) else {}
            if "return" in data:
                data = {"geocodes": data["return"]}
            geocodes = data.get("geocodes") or []

            if geocodes and isinstance(geocodes[0], dict):
                g = geocodes[0]
                raw_city = g.get("city") or g.get("province") or ""
                result = {
                    "city": raw_city.replace("市", "").replace("省", ""),
                    "coordinates": g.get("location") or "",
                    "district": g.get("district") or "",
                    "formatted_address": g.get("formatted_address") or query,
                }
            else:
                print(f"[CachedAmapClient][WARN] geocode({query!r}) 无可用结果, "
                      f"raw={str(raw)[:200]}")
        except Exception as exc:
            print(f"[CachedAmapClient][ERROR] geocode({query!r}) failed: {exc}")
            result["error"] = str(exc)

        # 只缓存成功解析的结果（铁律①）
        if result.get("city"):
            self._set(key, result)
        return result

    # ── 天气 ──────────────────────────────────────────────────────────

    async def weather(self, city: str) -> dict:
        key = self._key("weather", city)
        hit = self._get(key, "weather")
        if hit is not None:
            return hit

        result: dict = {
            "city": city, "day_weather": "", "day_temp": "",
            "day_wind": "", "humidity": "",
        }
        if self._cache_only:
            print("[CachedAmapClient] cache_only=true，跳过 weather API")
            return result

        try:
            import asyncio
            import re
            from src.tools.get_weather import get_weather

            loop = asyncio.get_event_loop()
            raw_text: str = await loop.run_in_executor(
                None, lambda: get_weather.invoke({"city": city})
            )

            m = re.search(
                r"(.+?)当前天气：(.+?)，温度\s*(-?\d+)℃，(.+?)，湿度\s*(\d+)%",
                raw_text,
            )
            if m:
                result = {
                    "city": m.group(1).strip(),
                    "day_weather": m.group(2).strip(),
                    "day_temp": m.group(3).strip(),
                    "day_wind": m.group(4).strip(),
                    "humidity": m.group(5).strip(),
                }
            else:
                print(f"[CachedAmapClient][WARN] weather parse failed, raw={raw_text!r}")
                result["raw"] = raw_text
        except Exception as exc:
            print(f"[CachedAmapClient][ERROR] weather('{city}') failed: {exc}")
            result["error"] = str(exc)

        if result.get("day_weather"):
            self._set(key, result)
        return result

    # ── POI 搜索 ──────────────────────────────────────────────────────

    def _fallback_from_poi_detail_cache(
        self, keywords: str, is_restaurant: bool = False,
    ) -> list[dict]:
        try:
            with open(_POI_DETAIL_CACHE, "r", encoding="utf-8") as f:
                poi_cache: dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        kw_parts = [k for k in keywords.lower().split() if k]
        results: list[dict] = []

        for poi_id, entry in poi_cache.items():
            if not isinstance(entry, dict):
                continue
            detail = entry.get("detail")
            if not isinstance(detail, dict):
                continue
            name = detail.get("name") or ""
            poi_type = detail.get("type") or ""
            text = (name + " " + poi_type).lower()

            if is_restaurant:
                if "餐饮" not in poi_type:
                    continue
            else:
                if not any(kw in text for kw in kw_parts):
                    continue

            results.append({
                "id": detail.get("id") or poi_id,
                "name": name,
                "address": detail.get("address") or "",
                "location": detail.get("location") or "",
                "type": poi_type,
                "rating": entry.get("rating") or detail.get("rating") or "",
                "distance": "",
                "tel": detail.get("tel") or "",
                "business_area": (detail.get("business_area")
                                  or detail.get("businessarea") or ""),
                "keyword_source": keywords,
                "source": "poi_detail_cache",
            })

        return results[:10]

    async def search_pois(
        self,
        keywords: str,
        *,
        location: str = "",
        city: str = "",
        radius: str = "5000",
        poi_type: str = "",
    ) -> list[dict]:
        """周边/关键词搜索 POI。

        【缓存 key 只包含真正发出去的参数】（铁律②）
        有 location 时走 maps_around_search，它的 schema 只有
        keywords/location/radius——**poi_type 根本不会被发出去**。
        原实现把它无条件放进 key，于是同一个查询有两个缓存条目，
        其中一个被污染时另一个还是好的，表现为
        **is_restaurant=True 搜到 0 家、False 搜到 8 家**——
        看起来像"那个参数改变了查询结果"，而它根本没发出去。

        【限流错误必须抛，不能降级为空】（铁律③）
        RetryableError 会走到 FactAgent 的 outcomes[].error，
        最终反映在 status=partial 上——用户知道"这次结果不完整"，
        而不是以为"附近没有"。

        限流时**不走 fallback**：fallback 会返回一批陈旧的、和当前
        查询无关的 POI，那比空结果更糟——它看起来像有效结果。
        """
        if location:
            key = self._key("search", keywords, location, radius)
        else:
            key = self._key("search", keywords, city, poi_type)

        hit = self._get(key, "search")
        if hit is not None:
            return hit

        results: list[dict] = []
        try:
            if self._cache_only:
                raise ValueError("cache_only=true，直接走 poi_detail_cache fallback")
            if not location and not city:
                raise ValueError("both location and city are empty")

            mcp = await self._get_mcp()
            if location:
                raw = await mcp.call("maps_around_search", {
                    "keywords": keywords, "location": location, "radius": radius,
                })
            else:
                payload = {"keywords": keywords, "city": city}
                if poi_type:
                    payload["types"] = poi_type
                raw = await mcp.call("maps_text_search", payload)

            # 先查错误再解析。顺序不能反——错误字符串解析出来是空列表，
            # 和"真的没有"完全一样。
            api_err = self._detect_api_error(raw)
            if api_err:
                raise RetryableError(
                    f"高德 API 错误（keywords={keywords!r}）: {api_err}")

            pois: list = []
            if isinstance(raw, dict):
                if "return" in raw:
                    pois = raw["return"]
                else:
                    pois = raw.get("pois") or raw.get("results") or []
            elif isinstance(raw, list):
                pois = raw

            for poi in pois:
                if not isinstance(poi, dict):
                    continue
                results.append({
                    "id": poi.get("id") or poi.get("uid") or "",
                    "name": poi.get("name") or "",
                    "address": poi.get("address") or "",
                    "location": poi.get("location") or "",
                    "type": poi.get("type") or poi.get("typecode") or "",
                    "rating": ((poi.get("biz_ext") or {}).get("rating")
                               or poi.get("rating") or ""),
                    "distance": poi.get("distance") or "",
                    "tel": poi.get("tel") or "",
                    "business_area": (
                        poi.get("business_area")
                        or poi.get("businessarea")
                        or (poi.get("biz_ext") or {}).get("business_area")
                        or ""
                    ),
                    "keyword_source": keywords,
                })

        except RetryableError:
            # 限流/临时故障必须往上抛，不能走 fallback（见 docstring）
            raise
        except Exception as exc:
            is_restaurant = poi_type == "050000" or "餐饮" in poi_type
            fallback = self._fallback_from_poi_detail_cache(
                keywords, is_restaurant=is_restaurant)
            if fallback:
                print(f"[CachedAmapClient] API 不可用（{type(exc).__name__}），"
                      f"从 poi_detail_cache 返回 {len(fallback)} 条 fallback")
            else:
                print(f"[CachedAmapClient][WARN] API 不可用且无 fallback 匹配, "
                      f"keywords={keywords!r} err={exc}")
            return fallback

        if results:
            self._set(key, results)
        else:
            # API 正常返回但确实没有结果。留痕但不缓存（铁律①）——
            # 下次重试可能不一样。
            print(f"[CachedAmapClient][WARN] 搜索返回空结果（不缓存）: "
                  f"keywords={keywords!r} location={location or city!r}")
        return results

    # ── POI 详情 ──────────────────────────────────────────────────────

    async def poi_detail(self, poi_id: str) -> dict | None:
        key = self._key("detail", poi_id)
        hit = self._get(key, "detail")
        if hit is not None:
            return hit

        result: dict | None = None
        try:
            mcp = await self._get_mcp()
            raw = await mcp.call("maps_search_detail", {"id": poi_id})

            api_err = self._detect_api_error(raw)
            if api_err:
                raise RetryableError(f"高德 API 错误（poi_detail {poi_id}）: {api_err}")

            if isinstance(raw, dict):
                result = raw
        except RetryableError:
            # 详情的限流不往上抛：详情只影响 cost/rating 这类增益字段，
            # 拿不到时 cost=None（诚实的未知），整条 POI 仍然可用。
            # 与 search 不同——搜索失败会让整个候选池为空，那是主流程。
            print(f"[CachedAmapClient][WARN] poi_detail({poi_id}) 被限流，"
                  f"该 POI 的 cost/rating 将为空")
        except Exception as exc:
            print(f"[CachedAmapClient][WARN] poi_detail({poi_id}) failed: {exc}")

        # 只缓存成功的详情（铁律①）。原实现连 None 都缓存 7 天——
        # 一次网络抖动会让这个 POI 在一周内永远没有 cost/rating，
        # 而 cost 缺失又会被下游正确地理解成"高德没给价格"，
        # 于是一个临时故障伪装成了数据源的事实。
        if result:
            self._set(key, result)
        return result

    # ── 距离 ──────────────────────────────────────────────────────────

    async def distance(self, origin: str, destination: str) -> dict | None:
        key = self._key("distance", origin, destination)
        hit = self._get(key, "distance")
        if hit is not None:
            return hit

        result: dict | None = None
        try:
            mcp = await self._get_mcp()
            raw = await mcp.call("maps_distance", {
                "origins": origin, "destination": destination, "type": "1",
            })

            api_err = self._detect_api_error(raw)
            if api_err:
                print(f"[CachedAmapClient][WARN] distance 被限流: {api_err}")
                return None

            if isinstance(raw, dict):
                items = raw.get("results") or []
                if items and isinstance(items[0], dict):
                    r = items[0]
                    result = {
                        "distance_meters": r.get("distance"),
                        "duration_seconds": r.get("duration"),
                        "eta_minutes": (round(int(r["duration"]) / 60)
                                        if r.get("duration") else None),
                    }
        except Exception as exc:
            print(f"[CachedAmapClient][WARN] distance failed: {exc}")

        if result:
            self._set(key, result)
        return result

    # ── 工具 ──────────────────────────────────────────────────────────

    @staticmethod
    def infer_environment(poi: dict) -> str:
        """室内/室外推断。

        ⚠️ 关键词匹配模拟语义理解，判断质量差。
        FactToolset._refine 已经不再调用它——室内外判断交给
        PlanningAgent 结合完整上下文自己做。保留是因为可能还有
        别的调用方，新代码不要用它。
        """
        text = " ".join(filter(None, [
            poi.get("name"), poi.get("type"), poi.get("address")
        ])).lower()
        if any(t in text for t in ("室内", "馆", "乐园", "商场", "影院",
                                   "水族", "海洋", "展览")):
            return "indoor"
        if any(t in text for t in ("公园", "森林", "广场", "湖", "山",
                                   "户外", "草坪", "绿道")):
            return "outdoor"
        if any(t in text for t in ("动物园", "植物园", "游乐园", "景区")):
            return "mixed"
        return "unknown"