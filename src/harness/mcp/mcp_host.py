# agent/mcp/mcp_host.py

import asyncio
import json
import logging
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from harness.tools.exceptions import RetryableError, DegradedError
from harness.tools.tool_executor import ToolDefinition
from utils.path_tool import get_abs_path

logger = logging.getLogger(__name__)


class MCPHost:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: list[ToolDefinition] = []
            cls._instance.initialized = False
        return cls._instance

    async def connect_servers(self):
        import os
        server_script = get_abs_path("agent/mcp/local_mcp_server.py")
        server_params = StdioServerParameters(
            command=sys.executable,    #  确保使用的是当前项目的解释器（虚拟环境的）
            args=["-u", server_script],
            env=os.environ.copy(),
        )

        logger.info("🔗 [MCP Host] 正在连接本地技能服务器...")

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 拿到 MCP server 暴露的工具列表
                response = await session.list_tools()

                self._tools = []
                for t in response.tools:
                    # 把 MCP 工具转成统一的 ToolDefinition
                    tool_def = ToolDefinition(
                        name=t.name,
                        description=t.description or "",
                        parameters=t.inputSchema.get("properties", {}),
                        required=t.inputSchema.get("required", []),
                        func=self._make_caller(server_params, t.name),
                    )
                    self._tools.append(tool_def)

                self.initialized = True
                logger.info(f"✅ [MCP Host] 已挂载工具: {[t.name for t in self._tools]}")

    def _make_caller(self, server_params: StdioServerParameters, tool_name: str):
        async def _async_call(**kwargs) -> str:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=kwargs)
                    text = result.content[0].text if result.content else ""

                    # 解析错误前缀，转成对应异常
                    if text.startswith("ERROR:RETRYABLE:"):
                        raise RetryableError(text.replace("ERROR:RETRYABLE:", ""))
                    elif text.startswith("ERROR:DEGRADED:"):
                        raise DegradedError(text.replace("ERROR:DEGRADED:", ""))

                    return text

        return _async_call

        def _sync_call(**kwargs) -> str:
            return asyncio.run(_async_call(**kwargs))

        return _sync_call

    def get_tools(self) -> list[ToolDefinition]:
        return self._tools


mcp_host = MCPHost()