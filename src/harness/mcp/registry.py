# harness/mcp/registry.py
from __future__ import annotations

import asyncio

from harness.mcp.mcp_client import MCPClient

_clients: dict[str, MCPClient] = {}
_lock = asyncio.Lock()


async def get_mcp_client(url: str, max_concurrent: int = 2) -> MCPClient:
    """按URL获取共享的MCPClient实例。

    同一个URL全局只建立一个连接并复用；不同URL各自独立，
    面向未来接入多个不同MCP服务的场景，不需要为此改任何代码。
    """
    async with _lock:
        if url not in _clients:
            client = MCPClient(url=url, max_concurrent=max_concurrent)
            await client.connect()
            _clients[url] = client
        return _clients[url]


async def close_all() -> None:
    """应用退出时调用，关闭所有MCP连接。"""
    async with _lock:
        for client in _clients.values():
            await client.close()
        _clients.clear()