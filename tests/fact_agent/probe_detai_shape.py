# scripts/probe_detail_shape.py
"""诊断：maps_search_detail 到底返回什么形状。

探针报 cost 覆盖率 0%（50 家餐厅无一有价格），有两种互斥的可能，
处理方式完全不同，必须先分清：

  A. 我们的 bug —— MCP 返回 {"return": {...}} 包装，而
     CachedAmapClient.poi_detail 是三个方法里唯一没拆包的
     （geocode / search_pois 都拆了）。这样 biz_ext 在最外层
     永远找不到，跟真实数据无关。→ 修 poi_detail，价格断言可用。

  B. 高德 MCP 的 maps_search_detail 压根不返回 biz_ext
     （MCP 包装层裁剪了字段，REST 原始接口有、MCP 没有）。
     → 真·硬伤，放弃价格断言。

下面直接打印原始返回的顶层键和完整 JSON，一眼就能分辨。
"""
from __future__ import annotations

import asyncio
import json

from src.tools.cached_amap_client import CachedAmapClient


async def main():
    api = CachedAmapClient()

    # 先搜一批餐厅，拿几个真实 poi_id
    await api.geocode("昆明")
    pois = await api.search_pois("火锅", city="昆明", poi_type="050000")
    if not pois:
        print("没搜到餐厅，先检查 geocode / MCP 配置")
        return

    print(f"搜到 {len(pois)} 家，取前 3 家看详情原始返回\n")

    for poi in pois[:3]:
        pid = poi.get("id")
        name = poi.get("name")
        print("=" * 70)
        print(f"POI: {name}  id={pid}")
        print("=" * 70)

        # 绕开 CachedAmapClient.poi_detail 的加工，直接看 MCP 原始返回
        mcp = await api._get_mcp()
        raw = await mcp.call("maps_search_detail", {"id": pid})

        print(f"类型: {type(raw).__name__}")
        if isinstance(raw, dict):
            print(f"顶层键: {list(raw.keys())}")
            has_return = "return" in raw
            print(f"有 'return' 包装层: {has_return}   ← True 说明是情况 A")

            inner = raw["return"] if has_return else raw
            if isinstance(inner, list) and inner:
                inner = inner[0]
                print("（'return' 是列表，取第 0 项）")
            if isinstance(inner, dict):
                print(f"内层键: {list(inner.keys())}")
                print(f"有 'biz_ext': {'biz_ext' in inner}")
                if "biz_ext" in inner:
                    print(f"biz_ext = {inner['biz_ext']!r}   ← cost 在这里")

        print("\n--- 完整原始返回 ---")
        print(json.dumps(raw, ensure_ascii=False, indent=2)[:3000])
        print()


if __name__ == "__main__":
    asyncio.run(main())