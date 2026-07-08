# harness/mcp/base.py
from __future__ import annotations

from typing import Any, Protocol

from harness.mcp.mcp_client import ToolSpec


class MCPServerClient(Protocol):
    """连接单个 MCP Server 的客户端协议。

    MCPClient(HTTP 传输)和 StdioMCPClient(stdio 传输)都实现这个接口——
    MCPHost 只认这个协议,不关心底层是哪种传输方式,和 StepExecutor /
    TerminationPolicy / LLMClientBase 是同一族设计:公共接口 + 可替换实现。
    """

    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def list_tools(self) -> list[ToolSpec]: ...
    async def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any: ...