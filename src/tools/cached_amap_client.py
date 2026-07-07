"""
CachedAmapClient：高德 MCP API 的本地缓存层（异步版本）。

三级架构不变：内存/文件缓存 → MCP远程调用 → 断网fallback。
底层从 AmapMCPClient 换成通用 MCPClient，通过 registry
获取共享的长连接实例，不再每次调用都握手。

weather 走独立的和风天气API（不消耗高德额度），不经过MCP。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from harness.mcp.registry import get_mcp_client
from harness.mcp.mcp_client import MCPClient
from src.utils.config_handler import tools_conf
from src.utils.path_tool import get_abs_path

_CACHE_FILE        = get_abs_path("data/amap_cache.json")
_POI_DETAIL_CACHE  = get_abs_path("data/poi_detail_cache.json")

_TTL: dict[str, timedelta] = {
    "geocode":  timedelta(days=30),
    "weather":  timedelta(hours=6),
    "search":   timedelta(days=1),
    "detail":   timedelta(days=7),
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

    # ── 缓存基础设施 ──────────────────────────────────────────────────────

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

    # ── 公开 API（全部 async）────────────────────────────────────────────

    async def geocode(self, address: str) -> dict:
        key = self._key("geocode", address)
        hit = self._get(key, "geocode")
        if hit is not None:
            return hit

        result: dict = {
            "city": "", "coordinates": "", "district": "", "formatted_address": address
        }
        if self._cache_only:
            print(f"[CachedAmapClient] cache_only=true，跳过 geocode API")
            return result
        try:
            mcp = await self._get_mcp()
            raw = await mcp.call("maps_geo", {"address": address})
            data = raw if isinstance(raw, dict) else {}
            if "return" in data:
                data = {"geocodes": data["return"]}
            geocodes = data.get("geocodes") or []
            if geocodes and isinstance(geocodes[0], dict):
                g = geocodes[0]
                raw_city = g.get("city") or g.get("province") or ""
                result = {
                    "city":              raw_city.replace("市", "").replace("省", ""),
                    "coordinates":       g.get("location") or "",
                    "district":          g.get("district") or "",
                    "formatted_address": g.get("formatted_address") or address,
                }
            else:
                print(f"[CachedAmapClient][WARN] geocode no usable data, raw={raw!r}")
        except Exception as exc:
            print(f"[CachedAmapClient][ERROR] geocode('{address}') failed: {exc}")
            result["error"] = str(exc)

        if result.get("city"):
            self._set(key, result)
        return result

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
            print(f"[CachedAmapClient] cache_only=true，跳过 weather API")
            return result

        try:
            from src.tools.get_weather import get_weather

            loop = asyncio.get_event_loop()
            raw_text: str = await loop.run_in_executor(
                None, lambda: get_weather.invoke({"city": city})
            )

            import re
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

    def _fallback_from_poi_detail_cache(self, keywords: str, is_restaurant: bool = False) -> list[dict]:
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
            name     = detail.get("name") or ""
            poi_type = detail.get("type") or ""
            text     = (name + " " + poi_type).lower()

            if is_restaurant:
                if "餐饮" not in poi_type:
                    continue
            else:
                if not any(kw in text for kw in kw_parts):
                    continue

            results.append({
                "id":             detail.get("id") or poi_id,
                "name":           name,
                "address":        detail.get("address") or "",
                "location":       detail.get("location") or "",
                "type":           poi_type,
                "rating":         entry.get("rating") or detail.get("rating") or "",
                "distance":       "",
                "tel":            detail.get("tel") or "",
                "business_area":  detail.get("business_area") or detail.get("businessarea") or "",
                "keyword_source": keywords,
                "source":         "poi_detail_cache",
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
        loc_key = location or city
        key = self._key("search", keywords, loc_key, radius, poi_type)
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
                    "id":             poi.get("id") or poi.get("uid") or "",
                    "name":           poi.get("name") or "",
                    "address":        poi.get("address") or "",
                    "location":       poi.get("location") or "",
                    "type":           poi.get("type") or poi.get("typecode") or "",
                    "rating":         (poi.get("biz_ext") or {}).get("rating") or poi.get("rating") or "",
                    "distance":       poi.get("distance") or "",
                    "tel":            poi.get("tel") or "",
                    "business_area": (
                        poi.get("business_area")
                        or poi.get("businessarea")
                        or (poi.get("biz_ext") or {}).get("business_area")
                        or ""
                    ),
                    "keyword_source": keywords,
                })
        except Exception as exc:
            is_restaurant = poi_type == "050000" or "餐饮" in poi_type
            fallback = self._fallback_from_poi_detail_cache(keywords, is_restaurant=is_restaurant)
            if fallback:
                print(f"[CachedAmapClient] API 不可用，从 poi_detail_cache 返回 {len(fallback)} 条fallback")
            else:
                print(f"[CachedAmapClient][WARN] API 不可用且无fallback匹配, keywords={keywords!r}")
            return fallback

        self._set(key, results)
        return results

    async def poi_detail(self, poi_id: str) -> dict | None:
        key = self._key("detail", poi_id)
        hit = self._get(key, "detail")
        if hit is not None:
            return hit

        result: dict | None = None
        try:
            mcp = await self._get_mcp()
            raw = await mcp.call("maps_search_detail", {"id": poi_id})
            if isinstance(raw, dict):
                result = raw
        except Exception:
            pass

        self._set(key, result)
        return result

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
            if isinstance(raw, dict):
                items = raw.get("results") or []
                if items and isinstance(items[0], dict):
                    r = items[0]
                    result = {
                        "distance_meters":  r.get("distance"),
                        "duration_seconds": r.get("duration"),
                        "eta_minutes":      (
                            round(int(r["duration"]) / 60) if r.get("duration") else None
                        ),
                    }
        except Exception:
            pass

        self._set(key, result)
        return result

    @staticmethod
    def infer_environment(poi: dict) -> str:
        text = " ".join(filter(None, [
            poi.get("name"), poi.get("type"), poi.get("address")
        ])).lower()
        if any(t in text for t in ("室内", "馆", "乐园", "商场", "影院", "水族", "海洋", "展览")):
            return "indoor"
        if any(t in text for t in ("公园", "森林", "广场", "湖", "山", "户外", "草坪", "绿道")):
            return "outdoor"
        if any(t in text for t in ("动物园", "植物园", "游乐园", "景区")):
            return "mixed"
        return "unknown"