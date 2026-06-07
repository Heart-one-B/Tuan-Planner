# Tool 实现代码展示

本文档用于展示项目中的 Tool 层实现。当前项目的 Tool 分为两类：

- 规划阶段真实事实工具：高德 MCP、天气查询、缓存与 fallback。
- 执行阶段 Mock 工具：用户确认后模拟活动预订、餐厅订座，避免 Demo 阶段真实下单。

调用关系如下：

```text
Fact Gathering Node
  -> CachedAmapClient
      -> AmapMCPClient
          -> maps_geo / maps_around_search / maps_text_search
          -> maps_search_detail / maps_distance
      -> get_weather
      -> local cache / poi_detail_cache fallback

Execution Node
  -> ExecutionAgent
      -> MockBookingAPI
          -> check_availability / reserve_venue
```

## 1. 高德 MCP Client

`AmapMCPClient` 是对 ModelScope 高德 MCP endpoint 的薄封装。它只负责连接 MCP、调用 tool、标准化返回值，并暴露和高德能力一一对应的方法。

```python
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.utils.config_handler import tools_conf


@dataclass
class _ToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None


class AmapMCPClient:
    def __init__(self, url: str | None = None):
        self.url = url or tools_conf.get("amap_mcp_url", "")
        if not self.url:
            raise ValueError("Missing amap_mcp_url in config/tools.yml")

    async def _call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        async with streamable_http_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments or {})
                return self._normalize_result(result)

    @staticmethod
    def _normalize_result(result: Any) -> Any:
        if result is None:
            return None
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                texts = []
                for item in content:
                    text = getattr(item, "text", None)
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                if texts:
                    joined = "\n".join(texts)
                    try:
                        return json.loads(joined)
                    except Exception:
                        return joined
            text = getattr(result, "text", None)
            if isinstance(text, str) and text.strip():
                try:
                    return json.loads(text)
                except Exception:
                    return text
        if isinstance(result, dict):
            return result
        return result

    def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        try:
            return asyncio.run(self._call(tool_name, arguments))
        except BaseException as exc:
            if hasattr(exc, "exceptions") and exc.exceptions:
                raise exc.exceptions[0] from exc
            raise

    def list_tools(self) -> list[_ToolSpec]:
        async def _list():
            async with streamable_http_client(self.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    tools = []
                    for item in getattr(result, "tools", []) or []:
                        tools.append(_ToolSpec(
                            name=getattr(item, "name", ""),
                            description=getattr(item, "description", "") or "",
                            input_schema=getattr(item, "inputSchema", None),
                        ))
                    return tools
        return asyncio.run(_list())

    def maps_geo(self, address: str, city: str | None = None) -> Any:
        payload: dict[str, Any] = {"address": address}
        if city:
            payload["city"] = city
        return self.call("maps_geo", payload)

    def maps_regeocode(self, location: str) -> Any:
        return self.call("maps_regeocode", {"location": location})

    def maps_ip_location(self, ip: str) -> Any:
        return self.call("maps_ip_location", {"ip": ip})

    def maps_weather(self, city: str) -> Any:
        return self.call("maps_weather", {"city": city})

    def maps_search_detail(self, poi_id: str) -> Any:
        return self.call("maps_search_detail", {"id": poi_id})

    def maps_text_search(self, keywords: str, city: str | None = None, types: str | None = None) -> Any:
        payload: dict[str, Any] = {"keywords": keywords}
        if city:
            payload["city"] = city
        if types:
            payload["types"] = types
        return self.call("maps_text_search", payload)

    def maps_around_search(self, keywords: str, location: str, radius: str | None = None) -> Any:
        payload: dict[str, Any] = {"keywords": keywords, "location": location}
        if radius:
            payload["radius"] = radius
        return self.call("maps_around_search", payload)

    def maps_distance(self, origins: str, destination: str, type_: str | None = None) -> Any:
        payload: dict[str, Any] = {"origins": origins, "destination": destination}
        if type_ is not None:
            payload["type"] = type_
        return self.call("maps_distance", payload)

    def maps_direction_walking(self, origin: str, destination: str) -> Any:
        return self.call("maps_direction_walking", {"origin": origin, "destination": destination})

    def maps_direction_driving(self, origin: str, destination: str) -> Any:
        return self.call("maps_direction_driving", {"origin": origin, "destination": destination})

    def maps_direction_transit_integrated(self, origin: str, destination: str, city: str, cityd: str) -> Any:
        return self.call(
            "maps_direction_transit_integrated",
            {"origin": origin, "destination": destination, "city": city, "cityd": cityd},
        )
```

## 2. 天气 Tool

天气工具使用 LangChain `@tool` 封装，供缓存层统一调用。当前实现调用和风天气 API，将结果转为文本，再由 `CachedAmapClient.weather()` 解析为结构化字段。

```python
import requests
from langchain_core.tools import tool

from src.utils.config_handler import tools_conf


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """获取指定城市当前天气。

    Args:
        city: 待查询城市。

    Returns:
        天气文本，包含天气状况、温度、风向、湿度等。
    """
    api_key = tools_conf["weather_api_key"]
    geo_url = (
        "https://pc3tehqmcy.re.qweatherapi.com/geo/v2/city/lookup"
        f"?location={city}&key={api_key}"
    )

    try:
        geo_response = requests.get(geo_url)
        geo_data = geo_response.json()
        if geo_data["code"] != "200":
            return f"找不到城市 {city}，请确认名称是否正确。"

        city_id = geo_data["location"][0]["id"]
        official_name = geo_data["location"][0]["name"]

        weather_url = (
            "https://pc3tehqmcy.re.qweatherapi.com/v7/weather/now"
            f"?location={city_id}&key={api_key}"
        )
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()

        if weather_data["code"] == "200":
            now = weather_data["now"]
            temp = now["temp"]
            text = now["text"]
            wind = now["windDir"]
            humidity = now["humidity"]
            return f"{official_name}当前天气：{text}，温度 {temp}℃，{wind}，湿度 {humidity}%。"
        return "获取天气数据失败。"
    except Exception as exc:
        return f"查询出错: {str(exc)}"
```

## 3. 缓存与 Fallback Tool Client

`CachedAmapClient` 是规划节点真正使用的工具入口。它在 MCP 外层增加 TTL 缓存、结构化标准化、API 失败 fallback 和 POI 环境推断。

```python
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.tools.amap_mcp_client import AmapMCPClient
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


class CachedAmapClient:
    def __init__(self):
        self._amap = AmapMCPClient()
        self._cache_only: bool = bool(tools_conf.get("amap_use_cache_only", False))

    @staticmethod
    def _key(op: str, *parts: str) -> str:
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
        entry = self._load().get(key)
        if isinstance(entry, dict) and self._fresh(entry, op):
            return entry["data"]
        return None

    def _set(self, key: str, data: Any) -> None:
        cache = self._load()
        cache[key] = {"cached_at": self._now(), "data": data}
        self._save()

    def geocode(self, address: str) -> dict:
        key = self._key("geocode", address)
        hit = self._get(key, "geocode")
        if hit is not None:
            return hit

        result: dict = {"city": "", "coordinates": "", "district": "", "formatted_address": address}
        if self._cache_only:
            return result
        try:
            raw = self._amap.maps_geo(address)
            data = raw if isinstance(raw, dict) else {}
            if "return" in data:
                data = {"geocodes": data["return"]}
            geocodes = data.get("geocodes") or []
            if geocodes and isinstance(geocodes[0], dict):
                item = geocodes[0]
                raw_city = item.get("city") or item.get("province") or ""
                result = {
                    "city": raw_city.replace("市", "").replace("省", ""),
                    "coordinates": item.get("location") or "",
                    "district": item.get("district") or "",
                    "formatted_address": item.get("formatted_address") or address,
                }
            else:
                result["error"] = f"geocode no usable data: {raw!r}"
        except Exception as exc:
            result["error"] = str(exc)

        if result.get("city"):
            self._set(key, result)
        return result

    def weather(self, city: str) -> dict:
        key = self._key("weather", city)
        hit = self._get(key, "weather")
        if hit is not None:
            return hit

        result: dict = {"city": city, "day_weather": "", "day_temp": "", "day_wind": "", "humidity": ""}
        if self._cache_only:
            return result
        try:
            from src.tools.get_weather import get_weather

            raw_text: str = get_weather.invoke({"city": city})
            match = re.search(
                r"(.+?)当前天气：(.+?)，温度\s*(-?\d+)℃，(.+?)，湿度\s*(\d+)%",
                raw_text,
            )
            if match:
                result = {
                    "city": match.group(1).strip(),
                    "day_weather": match.group(2).strip(),
                    "day_temp": match.group(3).strip(),
                    "day_wind": match.group(4).strip(),
                    "humidity": match.group(5).strip(),
                }
            else:
                result["raw"] = raw_text
        except Exception as exc:
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

        kw_parts = [item for item in keywords.lower().split() if item]
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
            elif not any(kw in text for kw in kw_parts):
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
                "business_area": detail.get("business_area") or detail.get("businessarea") or "",
                "keyword_source": keywords,
                "source": "poi_detail_cache",
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
        loc_key = location or city
        key = self._key("search", keywords, loc_key, radius, poi_type)
        hit = self._get(key, "search")
        if hit is not None:
            return hit

        results: list[dict] = []
        try:
            if self._cache_only:
                raise ValueError("cache_only=true")
            if not location and not city:
                raise ValueError("both location and city are empty")

            if location:
                raw = self._amap.maps_around_search(keywords=keywords, location=location, radius=radius)
            else:
                raw = self._amap.maps_text_search(
                    keywords=keywords,
                    city=city,
                    types=poi_type if poi_type else None,
                )

            pois: list = []
            if isinstance(raw, dict):
                pois = raw["return"] if "return" in raw else raw.get("pois") or raw.get("results") or []
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
                    "rating": (poi.get("biz_ext") or {}).get("rating") or poi.get("rating") or "",
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
        except Exception:
            is_restaurant = poi_type == "050000" or "餐饮" in poi_type
            return self._fallback_from_poi_detail_cache(keywords, is_restaurant=is_restaurant)

        self._set(key, results)
        return results

    def poi_detail(self, poi_id: str) -> dict | None:
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
                    item = items[0]
                    result = {
                        "distance_meters": item.get("distance"),
                        "duration_seconds": item.get("duration"),
                        "eta_minutes": round(int(item["duration"]) / 60) if item.get("duration") else None,
                    }
        except Exception:
            pass

        self._set(key, result)
        return result

    @staticmethod
    def infer_environment(poi: dict) -> str:
        text = " ".join(filter(None, [poi.get("name"), poi.get("type"), poi.get("address")])).lower()
        if any(token in text for token in ("室内", "馆", "乐园", "商场", "影院", "水族", "海洋", "展览")):
            return "indoor"
        if any(token in text for token in ("公园", "森林", "广场", "湖", "山", "户外", "草坪", "绿道")):
            return "outdoor"
        if any(token in text for token in ("动物园", "植物园", "游乐园", "景区")):
            return "mixed"
        return "unknown"
```

## 4. 规划阶段 Tool 调用示例

`Fact Gathering Node` 中通过 `CachedAmapClient` 串起 MCP 调用。下面是展示版核心调用链：

```python
api = CachedAmapClient()

# 1. 出发地地理编码
geo = api.geocode(origin_area)
city = geo.get("city") or ""
coordinates = geo.get("coordinates") or ""

# 2. 天气
weather = api.weather(city) if city else {}

# 3. 活动 POI 搜索
activities = []
for keyword in activity_keywords:
    pois = api.search_pois(
        keyword,
        location=coordinates,
        city=city,
        radius=radius,
    )
    for poi in pois:
        poi["environment"] = CachedAmapClient.infer_environment(poi)
        activities.append(poi)

# 4. 餐厅 POI 搜索，高德餐饮类型为 050000
restaurants = []
for keyword in restaurant_keywords:
    restaurants.extend(api.search_pois(
        keyword,
        location=coordinates,
        city=city,
        radius=radius,
        poi_type="050000",
    ))

# 5. 补齐详情与 ETA
eta = {}
for poi in activities + restaurants:
    if not poi.get("location") and poi.get("id"):
        detail = api.poi_detail(poi["id"])
        if isinstance(detail, dict) and detail.get("location"):
            poi["location"] = detail["location"]

    if coordinates and poi.get("location") and poi.get("id"):
        distance = api.distance(coordinates, poi["location"])
        if distance:
            eta[poi["id"]] = distance

fact_gathering_result = {
    "weather": weather,
    "activities": activities,
    "restaurants": restaurants,
    "eta": eta,
}
```

## 5. Mock API 实现

`MockBookingAPI` 是当前仍保留的 Mock Tool，负责模拟用户确认后的预订/订座动作。规划阶段不会调用它，只有用户确认方案后才进入执行阶段。

```python
from __future__ import annotations

import time


class MockBookingAPI:
    """预订/订座 Mock，供 execution_node 开发阶段使用。"""

    def check_availability(self, venue_id: str, time_slot: str) -> tuple[bool, str]:
        """检查场馆/餐厅是否有位。"""
        if venue_id == "R1":
            return False, "当前时段已满座，需要排队 60 分钟"
        return True, "余位充足"

    def reserve_venue(self, venue_id: str, user_info: str) -> dict:
        """预订场馆/餐厅，返回订单信息。"""
        time.sleep(0.5)
        return {
            "status": "success",
            "order_id": f"MT{int(time.time())}",
            "venue_id": venue_id,
        }
```

## 6. Mock API 调用代码

执行阶段由 `ExecutionAgent` 调用 Mock 工具。当前实现会为活动和餐厅各生成一个模拟订单；若最终方案字段不完整，也会生成占位订单，保证 Demo 闭环可展示。

```python
import random
import string

from src.tools.mock_api import MockBookingAPI


def _fake_order_id(prefix: str = "MT") -> str:
    suffix = "".join(random.choices(string.digits, k=8))
    return f"{prefix}{suffix}"


class ExecutionAgent:
    def __init__(self):
        self.tools = MockBookingAPI()

    def execute(self, plan: dict):
        print("\n[Execution Agent] 用户已确认，正在执行模拟预订请求...")
        activities = plan.get("activities") or []
        restaurant = plan.get("restaurant") or {}

        orders = []

        if isinstance(activities, list) and activities and isinstance(activities[0], dict):
            act = activities[0]
            act_id = act.get("id") or act.get("name") or "ACT_MOCK"
            act_name = act.get("name") or "活动"
            order_id = _fake_order_id("ACT")
            orders.append({
                "type": "activity",
                "status": "success",
                "order_id": order_id,
                "venue_id": act_id,
                "name": act_name,
                "note": "模拟预订成功，实际接入 API 后将完成真实下单。",
            })
        else:
            orders.append({
                "type": "activity",
                "status": "simulated",
                "order_id": _fake_order_id("ACT"),
                "venue_id": "UNKNOWN",
                "name": "待确认活动",
                "note": "活动信息不完整，已生成模拟订单。",
            })

        if isinstance(restaurant, dict) and restaurant:
            rest_id = restaurant.get("id") or restaurant.get("name") or "REST_MOCK"
            rest_name = restaurant.get("name") or "餐厅"
            order_id = _fake_order_id("RST")
            orders.append({
                "type": "restaurant",
                "status": "success",
                "order_id": order_id,
                "venue_id": rest_id,
                "name": rest_name,
                "note": "模拟订座成功，实际接入 API 后将完成真实订座。",
            })
        else:
            orders.append({
                "type": "restaurant",
                "status": "simulated",
                "order_id": _fake_order_id("RST"),
                "venue_id": "UNKNOWN",
                "name": "待确认餐厅",
                "note": "餐厅信息不完整，已生成模拟订单。",
            })

        return {
            "status": "success",
            "core_plan_changed": False,
            "non_core_failures": [],
            "message": "所有关键预订已模拟完成（当前为 Mock 模式，未进行真实下单）。",
            "orders": orders,
        }
```

`Execution Node` 中的调用方式：

```python
from src.agent.execution_agent import ExecutionAgent


def execution_node(state):
    errors = list(state.get("errors", []))
    user_confirmed = state.get("user_confirmed", False)
    plan = state.get("plan") or {}
    if not plan:
        final_plan_result = state.get("final_plan_result") or {}
        plan = final_plan_result.get("selected_candidate") or {}

    if user_confirmed is True:
        try:
            execution_result = ExecutionAgent().execute(plan)
        except Exception as exc:
            errors.append(f"Execution failed: {exc}")
            execution_result = {"status": "error", "message": str(exc)}
    else:
        execution_result = {
            "status": "cancelled",
            "message": "用户未确认方案，未执行任何预订或下单动作。",
        }

    return {
        "execution_result": execution_result,
        "plan": plan,
        "errors": errors,
    }
```

## 7. 展示要点

这套 Tool 实现的关键点是：

- 规划阶段只使用 MCP/缓存返回的事实，不让模型编造 POI。
- 高德 MCP 调用统一收口在 `AmapMCPClient`，业务节点只接触 `CachedAmapClient`。
- 缓存层负责 TTL、字段标准化、API 失败 fallback、距离 ETA 结构化。
- MockAPI 只用于用户确认后的执行阶段，保证“确认前不下单”。
- 所有 Tool 结果都以结构化 dict/list 返回，便于 Planning、Rule Validation、Scoring 和 Presentation 复用。
