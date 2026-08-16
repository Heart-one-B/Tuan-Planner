# scripts/probe_amap_mcp.py
"""高德 MCP 全量探针：有哪些工具、参数长什么样、真实返回什么。

【为什么需要这个】
当前代码只用了 5 个工具（maps_geo / maps_around_search /
maps_text_search / maps_search_detail / maps_distance），而这 5 个
是当初照着 REST 文档猜着写的。今天已经证明猜错过两次：

  ① cost 字段在顶层，不在 biz_ext 里（按 REST 形状写 → 覆盖率 0%）
  ② poi_type 在主路径上从未被发送（maps_around_search 分支没传
     types），RESTAURANT_TYPE="050000" 是死代码

两次都是同一个错误：**按文档想象的形状写代码，不看真实返回**。
这个脚本把"MCP 到底提供什么、真实返回什么"一次性问清楚，
后面的重构建在观测上，不建在假设上。

三段：
  A. list_tools()  —— 全部工具名 + 参数 schema（含没用上的）
  B. 关键参数验证 —— types 到底认不认（决定偏好能不能进搜索）
  C. 返回结构     —— type/cost 等字段的真实形状与覆盖率

用法（在项目根目录跑）：
    python scripts/probe_amap_mcp.py
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter

# 川大江安校区，与前面几次诊断保持同一坐标，结果可横向对比
ORIGIN = "103.999715,30.557573"
CITY = "成都"

# 高德三级分类里餐饮相关的几个码。用来验证 types 是否生效——
# 050102(川菜) 和 050103(粤菜) 结果应该显著不同；如果两次返回
# 一模一样，说明 types 被 MCP 忽略了。
TYPE_RESTAURANT = "050000"   # 餐饮服务（大类）
TYPE_SICHUAN = "050102"      # 四川菜(川菜)
TYPE_CANTONESE = "050103"    # 广东菜(粤菜)


def hr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def brief(obj, limit: int = 1200) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    return s if len(s) <= limit else s[:limit] + f"\n... (共 {len(s)} 字符，已截断)"


def pois_of(raw) -> list:
    """从各种可能的包装形态里取出 POI 列表。

    不假设一种形状：geocode 走的是 {"return": [...]}，
    search 有时是 {"pois": [...]}，有时直接是 list——
    CachedAmapClient 里已经为此写了三个分支，这里同样不猜。
    """
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return []
    for key in ("pois", "results", "return", "data"):
        v = raw.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


# ── A. 工具清单 ────────────────────────────────────────────────────────

async def stage_a_list_tools(client) -> list:
    hr("A. list_tools() —— MCP 到底提供了什么")
    try:
        specs = await client.list_tools()
    except Exception as e:
        print(f"  ❌ list_tools 失败: {type(e).__name__}: {e}")
        return []

    used = {
        "maps_geo", "maps_around_search", "maps_text_search",
        "maps_search_detail", "maps_distance",
    }
    print(f"  共 {len(specs)} 个工具\n")
    for spec in specs:
        mark = "✅ 已用" if spec.name in used else "🆕 未用"
        print(f"  {mark}  {spec.name}")
        desc = (spec.description or "").strip().replace("\n", " ")
        if desc:
            print(f"          {desc[:110]}")
        schema = spec.input_schema or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        if props:
            for pname, pdef in props.items():
                req = "*" if pname in required else " "
                ptype = (pdef or {}).get("type", "?")
                pdesc = ((pdef or {}).get("description") or "").replace("\n", " ")
                print(f"          {req} {pname} ({ptype}) {pdesc[:70]}")
        print()

    unused = [s.name for s in specs if s.name not in used]
    if unused:
        print(f"  🆕 未使用的工具：{unused}")
        print("     值得逐个看一眼——尤其是能算 POI 之间路程的那种：")
        print("     现在 eta_minutes 全部是「出发地→POI」，而真实行程是")
        print("     「出发地→活动→餐厅」，第二段用的是错的数字。")
    return specs


# ── B. types 参数是否生效 ──────────────────────────────────────────────

async def stage_b_types(client) -> None:
    """这一段决定"正向偏好能不能进搜索"这件事的可行性。

    如果 types 生效：'不吃辣' 可以翻译成 types=050103|050106 正向搜，
                    偏好第一次有了结构化的执行方式。
    如果不生效：    只能退回关键词搜索 + 结果按 type 字段过滤——
                    过滤那一半照样可做（type 在返回里），只是搜索
                    阶段无法收窄。
    """
    hr("B. maps_around_search 认不认 types 参数")

    async def probe(label: str, payload: dict) -> list:
        try:
            raw = await client.call("maps_around_search", payload)
        except Exception as e:
            print(f"  {label}: ❌ {type(e).__name__}: {e}")
            return []
        pois = pois_of(raw)
        names = [p.get("name", "") for p in pois][:6]
        print(f"  {label}: {len(pois)} 家 → {names}")
        return pois

    base = {"location": ORIGIN, "radius": "5000"}

    print("\n  ── 对照三组（同坐标、同半径、同关键词）──")
    no_types = await probe("① 不带 types，keywords=餐厅       ",
                           {**base, "keywords": "餐厅"})
    sichuan = await probe(f"② types={TYPE_SICHUAN}(川菜)，keywords=餐厅",
                          {**base, "keywords": "餐厅", "types": TYPE_SICHUAN})
    canton = await probe(f"③ types={TYPE_CANTONESE}(粤菜)，keywords=餐厅",
                         {**base, "keywords": "餐厅", "types": TYPE_CANTONESE})

    print("\n  ── 无关键词、纯类目（这是""偏好直接变查询""的关键形态）──")
    only_type = await probe(f"④ types={TYPE_RESTAURANT}(餐饮)，无 keywords",
                            {**base, "types": TYPE_RESTAURANT})

    print("\n  ── 判读 ──")
    n_sc = {p.get("id") for p in sichuan}
    n_ct = {p.get("id") for p in canton}

    if not sichuan and not canton and no_types:
        print("  🔴 带 types 的两组都是空 —— MCP 很可能把 types 当成了")
        print("     无法满足的额外过滤条件，或者参数名不对。")
    elif n_sc and n_ct and n_sc == n_ct:
        print("  🔴 川菜组和粤菜组结果**完全相同** —— types 被忽略了。")
        print("     偏好无法进入搜索阶段，只能退回「宽搜 + 按 type 字段过滤」。")
    elif n_sc != n_ct and (n_sc or n_ct):
        print("  🟢 川菜组和粤菜组结果不同 —— types 生效。")
        print("     偏好可以编译成类目码直接进查询，这是最干净的执行方式。")
        if only_type:
            print("  🟢 且支持""纯类目、无关键词""检索 —— 不再依赖模型编关键词。")
    else:
        print("  🟡 结果不足以判定，看上面的原始数量再决定。")


# ── C. 返回结构与字段覆盖率 ────────────────────────────────────────────

async def stage_c_fields(client) -> None:
    """搜索结果里到底带哪些字段，以及 type 字段的真实形态。

    type 是"不吃辣"能否被**可靠过滤**的关键：今天纠结的辛辣词表
    （火锅算不算辣、烧烤要不要收）在店名上确实无解，但如果 type
    字段是 "餐饮服务;中餐厅;四川菜(川菜)" 这种结构化三级分类，
    过滤就是确定性的，不需要猜。
    """
    hr("C. 搜索结果的真实字段 + type 覆盖率")

    try:
        raw = await client.call("maps_around_search", {
            "location": ORIGIN, "radius": "5000", "keywords": "餐厅",
        })
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        return

    pois = pois_of(raw)
    if not pois:
        print(f"  ❌ 没有结果。原始返回：\n{brief(raw)}")
        return

    print(f"  搜到 {len(pois)} 条\n")
    print("  ── 第一条的完整字段 ──")
    print(brief(pois[0], 900))

    keys = Counter()
    for p in pois:
        keys.update(p.keys())
    print(f"\n  ── 字段覆盖率（{len(pois)} 条中出现次数）──")
    for k, n in keys.most_common():
        print(f"    {k:<20} {n}/{len(pois)}")

    typed = [p for p in pois if p.get("type")]
    print(f"\n  ── type 字段（结构化菜系判定的依据）──")
    print(f"    有 type 的：{len(typed)}/{len(pois)}")
    for p in pois[:8]:
        print(f"      {p.get('name','')[:26]:<28} {p.get('type','(空)')}")

    if typed and all(";" in (p.get("type") or "") for p in typed):
        print("\n  🟢 type 是分号分隔的三级分类 —— "
              "'不吃辣'可以靠 `'川菜' in type` 精确过滤，")
        print("     不需要在店名上猜（火锅算不算辣这类歧义就此消失）。")
    else:
        print("\n  🟡 type 不是标准三级分类形态，过滤可靠性要重新评估。")

    # 连锁判定：(XX店) 这个模式够不够
    paren = [p for p in pois if "(" in (p.get("name") or "")
             or "（" in (p.get("name") or "")]
    print(f"\n  ── 连锁判定（名字带括号分店名）──")
    print(f"    带括号的：{len(paren)}/{len(pois)}")
    print("    ⚠️ 这个信号只对'带分店名的连锁'有效。没有括号的独立店"
          "可能仍是连锁，")
    print("       反之带括号的也可能只是标注了位置。判据强度需要人工核对下面几条：")
    for p in pois[:8]:
        flag = "括号" if ("(" in (p.get("name") or "") or "（" in (p.get("name") or "")) else "  —"
        print(f"      [{flag}] {p.get('name')}")


async def main() -> int:
    from harness.mcp.registry import get_mcp_client
    from src.utils.config_handler import tools_conf

    url = tools_conf.get("amap_mcp_url", "")
    if not url:
        print("❌ 配置里没有 amap_mcp_url")
        return 1
    print(f"MCP: {url}")

    client = await get_mcp_client(url)

    await stage_a_list_tools(client)
    await stage_b_types(client)
    await stage_c_fields(client)

    hr("这些结果会决定什么")
    print("  B 段 → 正向偏好（想吃粤菜）能不能直接变成查询")
    print("  C 段 → 负向偏好（不吃辣/不要连锁）能不能可靠过滤")
    print("  A 段 → 有没有工具能算 POI 之间的路程（现在第二段路程是错的）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))