# harness/mcp/stdio_client.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from harness.mcp.mcp_client import ToolSpec

logger = logging.getLogger(__name__)


class StdioMCPClient:
    """连接单个 stdio 传输 MCP Server 的客户端。纯机制,不认识任何
    具体服务器路径——command/args/env 全部由调用方传入。"""

    def __init__(self, command: str, args: list[str] | None = None,
                env: dict[str, str] | None = None):
        self.command = command
        self.args = args or []
        self.env = env
        self._session: ClientSession | None = None
        self._client_ctx = None
        self._session_ctx = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lock:
            if self._session is not None:
                return
            await self._do_connect()

    async def _do_connect(self) -> None:
        server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        self._client_ctx = stdio_client(server_params)
        read, write = await self._client_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        logger.info(f"[StdioMCPClient] connected: {self.command} {' '.join(self.args)}")

    async def close(self) -> None:
        async with self._lock:
            await self._do_close()

    async def _do_close(self) -> None:
        if self._session_ctx is not None:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"[StdioMCPClient] session close warning: {e}")
            self._session_ctx = None
            self._session = None
        if self._client_ctx is not None:
            try:
                await self._client_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"[StdioMCPClient] transport close warning: {e}")
            self._client_ctx = None

    async def _reconnect(self) -> None:
        async with self._lock:
            logger.warning(f"[StdioMCPClient] reconnecting: {self.command} ...")
            await self._do_close()
            await self._do_connect()

    async def __aenter__(self) -> "StdioMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        if self._session is None:
            await self.connect()
        try:
            return await self._call_once(tool_name, arguments)
        except Exception as e:
            logger.warning(f"[StdioMCPClient] call '{tool_name}' failed ({e}),重连后重试一次")
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
                texts = [getattr(item, "text", "") for item in content if getattr(item, "text", None)]
                if texts:
                    joined = "\n".join(texts)
                    try:
                        return json.loads(joined)
                    except Exception:
                        return joined
        return result

    async def list_tools(self) -> list[ToolSpec]:
        if self._session is None:
            await self.connect()
        result = await self._session.list_tools()
        return [
            ToolSpec(
                name=getattr(item, "name", ""),
                description=getattr(item, "description", "") or "",
                input_schema=getattr(item, "inputSchema", None),
            )
            for item in getattr(result, "tools", []) or []
        ]