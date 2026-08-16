# scripts/probe_cost.py
"""cost 覆盖率探针 —— 在写死实验断言之前必须先跑这个。

要回答三个问题，缺一个都不该动断言：

  ① 覆盖率：有多少家餐厅真的返回了 cost？
     太低（比如 <50%）→ 价格断言基本是摆设，应该放弃，
     退回"不辣 + 非连锁"两条硬断言。

  ② 区分度：≤100 的比例是多少？
     几乎全部 ≤100 → 约束太松，主组对照组都会通过，实验无区分度。
     几乎全部 >100 → 约束太严，两组都会失败，同样无区分度。
     理想区间大概是 30%~70%，此时"选不选得对"才真正取决于偏好记忆。

  ③ "0.00" 到底存不存在、对应什么商家？
     cost.py 目前把它当未知处理，这是一个**假设**。
     下面会把原始值原样打出来，跑完看一眼就能证实或推翻。

用法：
    python scripts/probe_cost.py
    python scripts/probe_cost.py --origin 陆家嘴 --json probe_result.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics

from agents.fact.tools import FactToolset

DEFAULT_KEYWORDS = ["聚餐", "本帮菜", "川菜", "日料", "烧烤", "火锅", "私房菜"]


async def probe(origin: str, keywords: list[str]) -> dict:
    toolset = FactToolset()
    await toolset.geocode(origin)
    print(f"出发地解析：city={toolset.origin_city} coord={toolset.origin_coordinates}\n")

    rows: list[dict] = []
    for kw in keywords:
        try:
            raw = await toolset.search_pois(kw, is_restaurant=True)
            payload = json.loads(raw)
        except Exception as e:
            print(f"[{kw}] 搜索失败：{e}")
            continue

        for p in payload.get("pois") or []:
            rows.append({
                "keyword": kw,
                "id": p.get("id"),
                "name": p.get("name"),
                "type": p.get("type"),
                "cost": p.get("cost"),
            })
        print(f"[{kw}] 搜到 {payload.get('count', 0)} 家")

    # 按 id 去重——不同关键词会搜到同一家，重复计入会让覆盖率失真
    unique: dict[str, dict] = {}
    for r in rows:
        if r["id"] and r["id"] not in unique:
            unique[r["id"]] = r
    items = list(unique.values())

    known = [r["cost"] for r in items if r["cost"] is not None]
    total = len(items)

    print("\n" + "=" * 60)
    print(f"去重后餐厅总数：{total}")
    print(f"有价格数据：    {len(known)}")
    print(f"覆盖率：        {len(known) / total:.0%}" if total else "覆盖率：N/A")

    if known:
        known_sorted = sorted(known)
        le_100 = sum(1 for c in known if c <= 100)
        print(f"价格区间：      {known_sorted[0]:.0f} ~ {known_sorted[-1]:.0f}")
        print(f"中位数：        {statistics.median(known_sorted):.0f}")
        print(f"≤100 的比例：   {le_100}/{len(known)} = {le_100 / len(known):.0%}")

    print("=" * 60)
    print("\n逐条明细（None = 高德未返回价格）：")
    for r in sorted(items, key=lambda x: (x["cost"] is None, x["cost"] or 0)):
        cost_str = f"{r['cost']:.0f}" if r["cost"] is not None else "—"
        print(f"  {cost_str:>6}  [{r['keyword']}] {r['name']}")

    # ── 给决策用的直接建议 ──
    print("\n" + "-" * 60)
    if not total:
        print("判定：没搜到任何餐厅，先检查 geocode 和高德配置。")
    elif len(known) / total < 0.5:
        print(f"判定：覆盖率 {len(known)/total:.0%} < 50%，**建议放弃价格断言**，")
        print("      只用「不辣 + 非连锁」两条。两条硬断言 + tool_calls_used")
        print("      的量化对比，证据链已经完整；硬凑第三条反而像挑指标。")
    else:
        le_100 = sum(1 for c in known if c <= 100)
        ratio = le_100 / len(known)
        if ratio > 0.8:
            print(f"判定：≤100 占 {ratio:.0%}，阈值太松，两组都会通过 → 无区分度。")
            print(f"      建议把阈值下调到中位数附近：{statistics.median(known):.0f}")
        elif ratio < 0.2:
            print(f"判定：≤100 占 {ratio:.0%}，阈值太严，两组都会失败 → 无区分度。")
            print(f"      建议上调到中位数附近：{statistics.median(known):.0f}")
        else:
            print(f"判定：覆盖率 {len(known)/total:.0%}、≤100 占 {ratio:.0%}，")
            print("      区分度合适，**价格断言可用，阈值 100 保持**。")
    print("-" * 60)

    return {
        "origin": origin,
        "total": total,
        "known": len(known),
        "coverage": (len(known) / total) if total else 0.0,
        "median": statistics.median(known) if known else None,
        "le_100_ratio": (sum(1 for c in known if c <= 100) / len(known)) if known else None,
        "items": items,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="陆家嘴", help="出发地")
    ap.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    ap.add_argument("--json", dest="json_out", default="", help="把结果落成 JSON（进实验报告附录）")
    args = ap.parse_args()

    result = asyncio.run(probe(args.origin, args.keywords))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.json_out}")


if __name__ == "__main__":
    main()