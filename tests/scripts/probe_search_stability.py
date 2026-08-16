# scripts/probe_search_stability.py
"""搜索稳定性探针 —— 开工前的唯一阻塞项。

【要回答的问题】
实测出现：keywords='公园' / 'KTV' 在江安校区周边返回 0 家。这两个词
在该位置不可能没有结果（此前的运行里「江安河公园」「AJ79·KTV」都
出现过）。而同一批并发请求里，别的词是成功的。

这件事必须在写任何新代码之前查清楚，因为：
  · Demo 剧本步骤 1「规划全天」如果随机搜不到餐厅，整个演示断在第一步
  · 任何 evals 的 baseline 都会建在流沙上——同一输入两次结果不同，
    分数就失去意义
  · 它已经造成过一次严重后果：空结果被缓存 1 天，「火锅」被毒了整天，
    表现为"用户点名的火锅附近未搜到"，指向完全错误的方向

【方法：第一性的问法】
同样的参数，重复发 N 次，结果一样吗？

这一个问题就把假设空间砍掉一半，而且只需要**最内层**——直接打 MCP，
不碰 CachedAmapClient 的缓存、不碰 FactToolset 的 refine、不调 LLM。

前面几轮排查的教训：不要用生产链路当调试器。现象出在第 3 层观察、
根因猜第 0 层，隔着缓存、并发、LLM 三层不确定性，只能一直猜。

【三种结论，三种修法】
  串行稳定 + 并发不稳  → 限流。修法：加信号量限制并发数
  串行就不稳          → 高德侧抖动。修法：空结果重试一次
  两者都稳            → 问题在我们的代码。回到第 1 层（缓存/解析）查

用法：
    python scripts/probe_search_stability.py
    python scripts/probe_search_stability.py --repeat 8 --concurrent 10
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 四川大学江安校区。与此前所有诊断同一坐标，结果可横向对比。
ORIGIN = "103.999715,30.557573"
RADIUS = "5000"

# 选词依据：
#   公园 / KTV —— 实测返回过 0 家，是本次要复现的对象
#   火锅       —— 被缓存投毒过，需要确认它本身是稳定的
#   餐厅       —— 通用词，作为"一定有结果"的基准
#   书店 / 电影院 —— 补充样本，判断问题是否与具体词有关
WORDS = ["公园", "KTV", "火锅", "餐厅", "书店", "电影院"]


def parse_pois(raw) -> list:
    """从各种包装形态里取 POI 列表。不假设一种形状——geocode 走
    {"return": [...]}，search 有时是 {"pois": [...]}，有时直接是 list。"""
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return []
    for key in ("pois", "results", "return", "data"):
        v = raw.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


async def search_once(mcp, kw: str) -> dict:
    """一次原始调用。异常就地捕获——我们要统计失败率，
    不能让一次异常中断整轮测量。"""
    t0 = time.monotonic()
    try:
        raw = await mcp.call("maps_around_search", {
            "keywords": kw, "location": ORIGIN, "radius": RADIUS,
        })
        pois = parse_pois(raw)
        return {
            "n": len(pois),
            "ids": frozenset(p.get("id") for p in pois if p.get("id")),
            "ms": int((time.monotonic() - t0) * 1000),
            "error": None,
            # 空结果时保留原始返回：区分"高德返回了空"和"我们没解析出来"
            "raw_head": str(raw)[:200] if not pois else "",
        }
    except Exception as e:
        return {
            "n": -1, "ids": frozenset(),
            "ms": int((time.monotonic() - t0) * 1000),
            "error": f"{type(e).__name__}: {e}",
            "raw_head": "",
        }


def report(kw: str, results: list[dict]) -> dict:
    """判定一组重复调用的稳定性。

    三个维度分开看：
      调用失败  → 抛异常了（网络/限流通常在这里）
      返回空    → 调用成功但 0 条（最隐蔽的一种，会被缓存毒化）
      结果不一致 → 有结果但内容不同（排序抖动或数据源不稳）
    """
    errors = [r for r in results if r["error"]]
    empties = [r for r in results if r["n"] == 0]
    oks = [r for r in results if r["n"] > 0]

    counts = sorted({r["n"] for r in results if r["n"] >= 0})
    id_sets = {r["ids"] for r in oks}
    ms = [r["ms"] for r in results]

    stable = (not errors and not empties and len(id_sets) <= 1)
    mark = "✅" if stable else "⚠️"

    print(f"  {mark} {kw:<8} 数量={counts}  "
          f"耗时 {min(ms)}~{max(ms)}ms  "
          f"失败={len(errors)}  空={len(empties)}  "
          f"结果集变体={len(id_sets)}")

    for r in errors[:2]:
        print(f"       ✗ {r['error'][:150]}")
    for r in empties[:1]:
        if r["raw_head"]:
            print(f"       空结果原始返回: {r['raw_head']}")

    return {"kw": kw, "errors": len(errors), "empties": len(empties),
            "variants": len(id_sets), "stable": stable, "n_total": len(results)}


async def main(repeat: int, concurrent: int) -> int:
    from harness.mcp.registry import close_all, get_mcp_client
    from src.utils.config_handler import tools_conf

    url = tools_conf.get("amap_mcp_url", "")
    if not url:
        print("❌ 配置里没有 amap_mcp_url")
        return 1

    mcp = await get_mcp_client(url)
    print(f"MCP: {url}")
    print(f"坐标: {ORIGIN}  半径: {RADIUS}m\n")

    # ── 阶段一：串行 ──
    # 一次只有一个在途请求，排除并发因素。这一轮不稳 = 高德侧本身抖动。
    print("=" * 72)
    print(f"阶段一：串行 {repeat} 次（排除并发因素）")
    print("=" * 72)
    serial = []
    for kw in WORDS:
        results = []
        for _ in range(repeat):
            results.append(await search_once(mcp, kw))
        serial.append(report(kw, results))

    # ── 阶段二：并发 ──
    # 模拟 FactAgent._run_searches 的 asyncio.gather。
    # 串行稳、这一轮不稳 = 限流。
    print("\n" + "=" * 72)
    print(f"阶段二：并发 {concurrent} 次（模拟 FactAgent 的 gather）")
    print("=" * 72)
    conc = []
    for kw in WORDS:
        results = list(await asyncio.gather(
            *(search_once(mcp, kw) for _ in range(concurrent))
        ))
        conc.append(report(kw, results))

    # ── 阶段三：混合并发 ──
    # 最接近真实：不同关键词同时发出。上面两轮是同词重复，
    # 而 FactAgent 实际发的是 N 个**不同**关键词的并发。
    print("\n" + "=" * 72)
    print(f"阶段三：{len(WORDS)} 个不同关键词同时发出（最接近真实调用）")
    print("=" * 72)
    mixed_results = list(await asyncio.gather(
        *(search_once(mcp, kw) for kw in WORDS)
    ))
    mixed_bad = 0
    for kw, r in zip(WORDS, mixed_results):
        flag = "✅" if r["n"] > 0 else ("💥" if r["error"] else "⚠️空")
        if r["n"] <= 0:
            mixed_bad += 1
        print(f"  {flag} {kw:<8} n={r['n']:<4} {r['ms']}ms "
              f"{r['error'] or ''}")

    # ── 判读 ──
    print("\n" + "=" * 72)
    print("判读")
    print("=" * 72)

    serial_bad = [s for s in serial if not s["stable"]]
    conc_bad = [s for s in conc if not s["stable"]]

    print(f"  串行不稳定的词：{[s['kw'] for s in serial_bad] or '无'}")
    print(f"  并发不稳定的词：{[s['kw'] for s in conc_bad] or '无'}")
    print(f"  混合并发失败数：{mixed_bad}/{len(WORDS)}")

    print()
    if not serial_bad and (conc_bad or mixed_bad):
        print("  🔴 结论：**限流**。串行全稳，并发出问题。")
        print("     修法：给 FactToolset.search_pois 加信号量，限制在途请求数。")
        print("     注意 MCPClient 自己有 max_concurrent，但 registry 默认只传 2；")
        print("     真正的并发发生在 FactAgent._run_searches 的 gather 层。")
        print("     建议先试 Semaphore(3)，跑本探针的阶段三验证。")
    elif serial_bad:
        print("  🔴 结论：**数据源本身不稳定**。串行也会返回空/异常。")
        print("     修法：空结果重试一次（注意：仍然不缓存空结果）。")
        print("     ⚠️ 重试要有上限，且要区分'真的没有'和'这次没拿到'——")
        print("        两者在返回值上一样，只能靠重试后是否仍空来推断。")
    elif not conc_bad and not mixed_bad:
        print("  🟡 结论：**本次未复现**。三个阶段全稳定。")
        print("     可能是当时的临时状况（配额边缘/网络抖动）。")
        print("     不要就此认为问题不存在——空结果不缓存那条修复必须保留，")
        print("     它是这类问题的兜底。建议在不同时段再跑一次本探针。")
    else:
        print("  🟡 结论不明确，看上面各阶段的具体数字。")

    print("\n  ── 与生产代码的差距 ──")
    print("  本探针直接打 MCP，不经过 CachedAmapClient（缓存）和")
    print("  FactToolset（_refine 并发拉详情）。如果本探针全稳而生产")
    print("  仍不稳，问题在那两层——下一步是加壳重测，而不是继续猜。")

    await close_all()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5, help="串行重复次数")
    ap.add_argument("--concurrent", type=int, default=8, help="并发次数")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.repeat, args.concurrent)))