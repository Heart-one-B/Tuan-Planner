# harness/mcp/mcp_host.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from harness.mcp.base import MCPServerClient
from harness.mcp.registry import get_mcp_client
from harness.mcp.stdio_client import StdioMCPClient
from harness.tools.exceptions import DegradedError, RetryableError
from harness.tools.tool_definition import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """连哪个 MCP Server、怎么连——这是业务配置,不是 harness 的知识。"""
    name: str
    transport: Literal["stdio", "http"]
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None


class MCPHost:
    """AI 应用层面的 MCP 连接管理器。不再是单例——不同场景需要不同
    server 组合,单例和"可配置"是矛盾的,改为显式实例化。

    工具命名空间:{server_name}__{tool_name},避免多 server 同名工具冲突。
    """

    def __init__(self, server_configs: list[MCPServerConfig]):
        self._server_configs = server_configs
        self._clients: dict[str, MCPServerClient] = {}
        self._tools: list[ToolDefinition] = []
        self.initialized = False

    async def connect_servers(self) -> None:
        for cfg in self._server_configs:
            client = await self._build_client(cfg)
            await client.connect()
            self._clients[cfg.name] = client
            logger.info(f"🔗 [MCP Host] 已连接 server '{cfg.name}' ({cfg.transport})")
        await self._refresh_tools()
        self.initialized = True

    async def _build_client(self, cfg: MCPServerConfig) -> MCPServerClient:
        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(f"server '{cfg.name}' 是 stdio 传输,必须提供 command")
            return StdioMCPClient(command=cfg.command, args=cfg.args, env=cfg.env)
        elif cfg.transport == "http":
            if not cfg.url:
                raise ValueError(f"server '{cfg.name}' 是 http 传输,必须提供 url")
            return await get_mcp_client(cfg.url)
        raise ValueError(f"未知的 transport: {cfg.transport}")

    async def _refresh_tools(self) -> None:
        self._tools = []
        for server_name, client in self._clients.items():
            specs = await client.list_tools()
            for spec in specs:
                self._tools.append(ToolDefinition(
                    name=f"{server_name}__{spec.name}",
                    description=spec.description,
                    parameters=(spec.input_schema or {}).get("properties", {}),
                    required=(spec.input_schema or {}).get("required", []),
                    func=self._make_caller(server_name, spec.name),
                ))
        logger.info(f"✅ [MCP Host] 已聚合工具: {[t.name for t in self._tools]}")

    def _make_caller(self, server_name: str, tool_name: str):
        async def _async_call(**kwargs) -> str:
            client = self._clients[server_name]
            text = str(await client.call(tool_name, arguments=kwargs))
            if text.startswith("ERROR:RETRYABLE:"):
                raise RetryableError(text.replace("ERROR:RETRYABLE:", ""))
            elif text.startswith("ERROR:DEGRADED:"):
                raise DegradedError(text.replace("ERROR:DEGRADED:", ""))
            return text
        return _async_call

    def get_tools(self) -> list[ToolDefinition]:
        return self._tools

    async def close(self) -> None:
        """只关闭本 Host 独有的 stdio 客户端;http 客户端归全局注册表管。"""
        for cfg in self._server_configs:
            if cfg.transport == "stdio":
                client = self._clients.get(cfg.name)
                if client is not None:
                    await client.close()
        self.initialized = False