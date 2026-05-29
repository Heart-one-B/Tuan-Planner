"""
CachedAmapClient：高德 MCP API 的本地缓存层。

三级架构：
    调用方
      → ① 内存/文件缓存（amap_cache.json，TTL 内直接返回）
      → ② AmapMCPClient（真实高德 API，结果写入缓存）
      → ③ poi_detail_cache.json（断网 fallback，按关键词匹配已缓存 POI）

各操作 TTL：
    geocode   30天（地址不常变）
    weather   6小时（当天有效）
    search    1天（POI 搜索结果）
    detail    7天（POI 详情）
    distance  2小时（路况有变化）
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.tools.amap_mcp_client import AmapMCPClient
from src.utils.path_tool import get_abs_path
from utils.config_handler import tools_conf

_CACHE_FILE        = get_abs_path("data/amap_cache.json")
_POI_DETAIL_CACHE  = get_abs_path("data/poi_detail_cache.json")

_TTL: dict[str, timedelta] = {
    "geocode":  timedelta(days=30),
    "weather":  timedelta(hours=6),
    "search":   timedelta(days=1),
    "detail":   timedelta(days=7),
    "distance": timedelta(hours=2),
}

# 进程内共享的内存缓存，避免同一进程重复读文件
_MEM_CACHE: dict[str, dict] = {}




class CachedAmapClient:
    """高德 MCP API + 本地文件缓存的二级架构客户端。"""

    def __init__(self):
        self._amap = AmapMCPClient()
        # tools.yml: amap_use_cache_only: true  → 只读本地缓存，不调高德 API（省额度）
        self._cache_only: bool = bool(tools_conf.get("amap_use_cache_only", False))

    # ── 缓存基础设施 ──────────────────────────────────────────────────────────

    @staticmethod
    def _key(op: str, *parts: str) -> str:
        """生成缓存键：op 前缀 + 参数 MD5 短码，便于调试。"""
        raw = "|".join(parts)
        short = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{op}:{short}"

    def _load(self) -> dict[str, dict]:
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
        """返回缓存数据；未命中或已过期返回 None。"""
        entry = self._load().get(key)
        if isinstance(entry, dict) and self._fresh(entry, op):
            return entry["data"]
        return None

    def _set(self, key: str, data: Any) -> None:
        cache = self._load()
        cache[key] = {"cached_at": self._now(), "data": data}
        self._save()

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def geocode(self, address: str) -> dict:
        """
        文字地址 → 城市 + 坐标。

        返回：
            {city, coordinates, district, formatted_address}
            city 已去掉"市/省"后缀；coordinates 为"lng,lat"格式。
        """
        key = self._key("geocode", address)
        hit = self._get(key, "geocode")
        if hit is not None:
            return hit

        result: dict = {
            "city": "", "coordinates": "", "district": "", "formatted_address": address
        }
        if self._cache_only:
            print(f"[CachedAmapClient] cache_only=true，跳过 geocode API，返回空结果")
            return result
        try:
            raw = self._amap.maps_geo(address)
            # 高德 MCP 层可能把结果包在 "return" 或 "geocodes" 字段里
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

        # 只缓存有效结果，失败/空结果不写缓存（避免长期缓存错误状态）
        if result.get("city"):
            self._set(key, result)
        return result

    def weather(self, city: str) -> dict:
        """
        查询城市实时天气（调用 src/tools/get_weather.py）。

        返回：
            {city, day_weather, day_temp, day_wind, humidity}
        """
        key = self._key("weather", city)
        hit = self._get(key, "weather")
        if hit is not None:
            return hit

        result: dict = {
            "city": city, "day_weather": "", "day_temp": "",
            "day_wind": "", "humidity": "",
        }
        if self._cache_only:
            print(f"[CachedAmapClient] cache_only=true，跳过 weather API，返回空结果")
            return result
        try:
            from src.tools.get_weather import get_weather
            raw_text: str = get_weather.invoke({"city": city})

            # get_weather 返回格式："{城市}当前天气：{text}，温度 {temp}℃，{wind}，湿度 {humidity}%。"
            # 从文本中解析出结构化字段
            import re
            m = re.search(
                r"(.+?)当前天气：(.+?)，温度\s*(-?\d+)℃，(.+?)，湿度\s*(\d+)%",
                raw_text,
            )
            if m:
                result = {
                    "city":        m.group(1).strip(),
                    "day_weather": m.group(2).strip(),
                    "day_temp":    m.group(3).strip(),
                    "day_wind":    m.group(4).strip(),
                    "humidity":    m.group(5).strip(),
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
        """
        断网 fallback：从 poi_detail_cache.json 中检索已缓存的 POI。

        - 活动搜索：按关键词匹配 POI 名称和类型
        - 餐厅搜索：按 POI 类型含"餐饮"过滤，不依赖关键词
          （因为"亲子餐厅"等搜索词在缓存餐厅名中往往匹配不到）
        """
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
                # 餐厅模式：只要类型含"餐饮"即可，不靠关键词匹配
                if "餐饮" not in poi_type:
                    continue
            else:
                # 活动模式：关键词匹配
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
                "keyword_source": keywords,
                "source":         "poi_detail_cache",
            })

        return results[:10]

    def search_pois(
        self,
        keywords: str,
        *,
        location: str = "",
        city: str = "",
        radius: str = "5000",
        poi_type: str = "",
    ) -> list[dict]:
        """
        POI 搜索。
        有坐标时走 around_search（周边精准搜索）；否则走 text_search（城市范围）。

        返回标准化 POI 列表，每项包含：
            {id, name, address, location, type, rating, distance, tel, keyword_source}
        """
        loc_key = location or city
        key = self._key("search", keywords, loc_key, radius)
        hit = self._get(key, "search")
        if hit is not None:
            return hit

        results: list[dict] = []
        try:
            if self._cache_only:
                raise ValueError("cache_only=true，直接走 poi_detail_cache fallback")
            if not location and not city:
                raise ValueError("both location and city are empty, skipping API call")
            if location:
                raw = self._amap.maps_around_search(
                    keywords=keywords, location=location, radius=radius
                )
            else:
                raw = self._amap.maps_text_search(
                    keywords=keywords, city=city,
                    types=poi_type if poi_type else None
                )

            # 高德 MCP 层可能把结果包在 "return" 字段里
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
                    "keyword_source": keywords,
                })
        except Exception as exc:
            # API 不可用（断网等），尝试从 poi_detail_cache.json 读取
            is_restaurant = poi_type == "050000" or "餐饮" in poi_type
            fallback = self._fallback_from_poi_detail_cache(keywords, is_restaurant=is_restaurant)
            if fallback:
                print(f"[CachedAmapClient] API 不可用，从 poi_detail_cache 返回 {len(fallback)} 条 fallback 数据（{'餐厅' if is_restaurant else '活动'}）")
            else:
                print(f"[CachedAmapClient][WARN] API 不可用且 poi_detail_cache 无匹配，keywords={keywords!r}, is_restaurant={is_restaurant}")
            return fallback

        self._set(key, results)
        return results

    def poi_detail(self, poi_id: str) -> dict | None:
        """
        获取 POI 详情（营业时间、评分、电话等）。
        缓存命中时直接返回，不消耗 API 额度。
        """
        key = self._key("detail", poi_id)
        hit = self._get(key, "detail")
        if hit is not None:
            return hit

        result: dict | None = None
        try:
            raw = self._amap.maps_search_detail(poi_id)
            if isinstance(raw, dict):
                result = raw
        except Exception:
            pass

        self._set(key, result)
        return result

    def distance(self, origin: str, destination: str) -> dict | None:
        """
        计算出发点到目的地的驾车距离和时间。
        返回 {distance_meters, duration_seconds, eta_minutes} 或 None。
        """
        key = self._key("distance", origin, destination)
        hit = self._get(key, "distance")
        if hit is not None:
            return hit

        result: dict | None = None
        try:
            raw = self._amap.maps_distance(origins=origin, destination=destination, type_="1")
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
        """根据 POI 名称和类型推断室内/室外/混合。"""
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