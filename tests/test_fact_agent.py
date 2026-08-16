# scripts/inspect_amap_tools.py
import asyncio
from harness.mcp.mcp_client import MCPClient
from src.utils.config_handler import tools_conf

async def main():
    client = MCPClient(url=tools_conf.get("amap_mcp_url"))
    await client.connect()
    tools = await client.list_tools()
    for t in tools:
        print(f"\n=== {t.name} ===")
        print(f"description: {t.description}")
        print(f"input_schema: {t.input_schema}")
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())