from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None


class MCPClient:
    """通用 MCP 协议客户端。

    连接任意遵循 MCP 协议的服务器，调用其暴露的任意工具。
    工具名和参数
    在调用时由上层传入，这个类只负责协议层面的通信可靠性。

    职责：
        - 连接生命周期管理（长连接复用，不是每次调用都握手）
        - 失败自动重连
        - 并发限流（避免打爆服务端的隐性QPS限制）
        - 工具调用 / 工具发现

    不负责：
        - 任何具体服务的工具名、参数结构、返回值解析
          （这些是业务层的知识，属于调用方）
    """

    def __init__(self, url: str, max_concurrent: int = 5):
        if not url:
            raise ValueError("MCPClient 需要一个有效的 url")
        self.url = url

        self._session: ClientSession | None = None
        self._client_ctx = None
        self._session_ctx = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ── 连接生命周期 ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """建立连接。幂等：已连接时直接返回。"""
        async with self._lock:
            if self._session is not None:
                return
            await self._do_connect()

    async def _do_connect(self) -> None:
        self._client_ctx = streamable_http_client(self.url)
        read, write, _ = await self._client_ctx.__aenter__()

        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        logger.info(f"[MCPClient] connected to {self.url}")

    async def close(self) -> None:
        async with self._lock:
            await self._do_close()

    async def _do_close(self) -> None:
        if self._session_ctx is not None:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"[MCPClient] session close warning: {e}")
            self._session_ctx = None
            self._session = None

        if self._client_ctx is not None:
            try:
                await self._client_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"[MCPClient] transport close warning: {e}")
            self._client_ctx = None

    async def _reconnect(self) -> None:
        async with self._lock:
            logger.warning(f"[MCPClient] reconnecting to {self.url} ...")
            await self._do_close()
            await self._do_connect()

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ── 工具调用（核心能力） ────────────────────────────────────────────

    async def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """调用服务器上的任意工具。

        tool_name / arguments 完全由调用方决定，这个方法本身
        不假设服务器上有什么工具——这正是MCP协议标准化的价值：
        同一个客户端可以调用任何遵循协议的服务器暴露的任意工具。
        """
        if self._session is None:
            await self.connect()

        async with self._semaphore:
            try:
                return await self._call_once(tool_name, arguments)
            except Exception as e:
                logger.warning(
                    f"[MCPClient] call '{tool_name}' failed ({e})，重连后重试一次"
                )
                await self._reconnect()
                return await self._call_once(tool_name, arguments)

    async def _call_once(self, tool_name: str, arguments: dict[str, Any] | None) -> Any:
        result = await self._session.call_tool(tool_name, arguments or {})
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

    # ── 工具发现（MCP协议原生支持的自省能力） ────────────────────────────

    async def list_tools(self) -> list[ToolSpec]:
        """查询服务器暴露了哪些工具。用于调试或动态发现工具列表。"""
        if self._session is None:
            await self.connect()

        async with self._semaphore:
            result = await self._session.list_tools()
            tools = []
            for item in getattr(result, "tools", []) or []:
                tools.append(ToolSpec(
                    name=getattr(item, "name", ""),
                    description=getattr(item, "description", "") or "",
                    input_schema=getattr(item, "inputSchema", None),
                ))
            return tools