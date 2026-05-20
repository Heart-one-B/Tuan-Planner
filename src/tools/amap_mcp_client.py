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
        return asyncio.run(self._call(tool_name, arguments))

    def list_tools(self) -> list[_ToolSpec]:
        async def _list():
            async with streamable_http_client(self.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    tools = []
                    for item in getattr(result, "tools", []) or []:
                        tools.append(
                            _ToolSpec(
                                name=getattr(item, "name", ""),
                                description=getattr(item, "description", "") or "",
                                input_schema=getattr(item, "inputSchema", None),
                            )
                        )
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
