# scripts/exp_e1_memory.py
"""E1：记忆有效性的受控实验。

【实验设计】
单变量：memory ON / OFF，其余（代码、模型、prompt、输入、地点）
完全相同。这是唯一能干净归因的对比——和"有无 harness 的两个版本"
对比不同，那里差异太多，测出差异也说不清是谁的功劳。

  Session 1  用户在**否决**中暴露偏好（不是声明）。
             不说"我不吃辣"，而是先看到方案再拒绝。这是真实对话的
             形状，也让提取器面对真实难度：得从对话里推断，不是抄
             一句现成的自述。
  Session 2  全新 session_id、不带 history、**不提任何偏好**。
             主组能拿到的只有记忆。

【为什么 Session 2 必须是 open 模式】
输入刻意写成"找个地方聚一聚吃个饭"而不是"想吃火锅"。
restaurant_intent=explicit 时，用户点名了品类，系统按点名精确检索
——此时记忆偏好本来就**不该**改写搜索词（本轮原话优先于历史偏好）。
用 explicit 输入做这个实验，测的是分流逻辑，不是记忆。

【断言用 typecode，不用店名】
高德返回的 typecode 是数据源自己维护的三级分类（050102=川菜，
050108=湘菜）。店名匹配在这件事上已经被证伪过：云南的"爱尚菌·
野生菌火锅"是菌汤、"正宗富源酸菜土猪脚火锅"是酸菜，靠"火锅"两个
字判辣必然误判。typecode 让歧义显式локализ到少数几个码上。

【报告形式是比率，不是二元判定】
选中方案通常只有 1 家餐厅，单次通过很可能是撞运气。
主组 5/5 vs 对照组 2/5 是好结果；**对照组 5/5 全过则实验无效**
——说明候选池里本来就没有违规项，断言是白送的。

用法：
    python scripts/exp_e1_memory.py                # n=5，每轮重建记忆
    python scripts/exp_e1_memory.py --n 3
    python scripts/exp_e1_memory.py --reuse-memory # 只建一次记忆，省钱
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import uuid
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path("data/memory")
DB_PATH = Path("data/trace.db")
OUT_DIR = Path("data/experiments")

# ── 剧本（跑前定稿，跑后不改）──────────────────────────────────────
# 剧本先固定、断言后写这个顺序，如实记录在这里：断言是在看到
# Session 1 能提取出什么之后才写的（smoke_memory.py 的产出），
# 属于"从观测到的行为反推指标"，不是"先定指标再编剧本去满足它"。
SESSION_1_TURNS = [
    "这周六下午两点到晚上十点，四个人朋友聚会，"
    "从四川大学江安校区出发，找个地方逛逛然后吃个饭",
    "这个火锅算了，我们几个都不太能吃辣，换个清淡点的口味吧",
    "这家看着像连锁店，想找点本地特色的小馆子",
]

SESSION_2_INPUT = (
    "这周日下午两点到晚上九点，还是四个人，"
    "从四川大学江安校区出发，找个地方聚一聚吃个饭"
)

CLARIFY_ANSWER = "朋友聚会，四个人，下午两点到晚上十点，从四川大学江安校区出发"


def hr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ── 断言 ──────────────────────────────────────────────────────────────

def judge_spicy(restaurants: list[dict]) -> dict:
    """选中的餐厅里有没有确定辛辣的。

    只判 SPICY_TYPES（川菜、湘菜）——存疑类目（火锅店、云贵菜）
    放行。理由见 amap_types 的说明：搜索/过滤阶段的误杀不可逆，
    而两种错误的代价不对称。这里保持和生产代码同一个口径，
    实验用一套判定、生产用另一套是自欺。

    valid=False 表示这次没有餐厅可判（未出方案 / 方案里没餐厅），
    既不算通过也不算失败——把"没测到"记成"通过"是伪造数据。
    """
    from agents.fact.amap_types import label_of, spice_level

    if not restaurants:
        return {"valid": False, "passed": None, "detail": "选中方案里没有餐厅"}

    violations = []
    for r in restaurants:
        code = r.get("type") or ""
        if spice_level(code) == "spicy":
            violations.append(f"{r.get('name')}[{label_of(code)}]")

    return {
        "valid": True,
        "passed": not violations,
        "violations": violations,
        "detail": [f"{r.get('name')}[{label_of(r.get('type') or '')}]"
                   for r in restaurants],
    }


def observe_chain(restaurants: list[dict]) -> dict:
    """连锁：只观测、不断言。

    两个信号都不够强：
      · typecode 品牌码（050301 肯德基等）只覆盖十几个大品牌，
        海底捞、蜜雪冰城这类没有专属码，识别不出来
      · 店名带括号「(XX店)」实测 12/15 命中，但「馨苑餐厅
        (四川大学江安校区店)」显然不是连锁——括号只说明标注了
        位置，不说明是连锁
    判据不成立就不该当断言用。如实记成观测项，报告里说明为什么
    没把它列为硬指标，比硬凑一条经不起追问的断言强。
    """
    from agents.fact.amap_types import is_chain

    known = [r.get("name") for r in restaurants if is_chain(r.get("type") or "")]
    paren = [r.get("name") for r in restaurants
             if "(" in (r.get("name") or "") or "（" in (r.get("name") or "")]
    return {"known_chain_brands": known, "name_has_paren": paren}


# ── 单次运行 ──────────────────────────────────────────────────────────

async def run_once(rep: int, memory_on: bool, build_memory: bool) -> dict:
    """跑一个重复。memory_on=False 时跳过 Session 1（没有记忆可建）。"""
    from src.memory.wiring import build_memory_config, configure_memory, list_memories
    from src.session_runner import Session

    tag = f"{'M' if memory_on else 'C'}{rep}-{uuid.uuid4().hex[:4]}"
    record: dict = {"rep": rep, "group": "main" if memory_on else "control", "tag": tag}

    if memory_on:
        if build_memory:
            if MEMORY_DIR.exists():
                shutil.rmtree(MEMORY_DIR)
        configure_memory(build_memory_config(memory_dir=MEMORY_DIR))
    else:
        configure_memory(None)

    # ── Session 1：埋偏好（仅主组、仅需要重建时）──
    if memory_on and build_memory:
        print(f"  [{tag}] Session 1：三轮埋偏好…")
        s1 = Session(session_id=f"e1a-{tag}", auto_clarify_answer=CLARIFY_ANSWER)
        for text in SESSION_1_TURNS:
            await s1.say(text)
        await s1.close()

    mems = [m.name for m in list_memories()] if memory_on else []
    record["memories"] = mems
    if memory_on:
        print(f"  [{tag}] 记忆库 {len(mems)} 条: {mems}")
        if not mems:
            record["error"] = "记忆库为空，本次主组样本无效"

    # ── Session 2：全新会话，不提偏好 ──
    print(f"  [{tag}] Session 2…")
    s2 = Session(session_id=f"e1b-{tag}", auto_clarify_answer=CLARIFY_ANSWER)
    r2 = await s2.say(SESSION_2_INPUT)

    intent = r2.state.get("intent") or {}
    plan_ctx = r2.state.get("plan_context") or {}
    restaurants = s2.selected_restaurants()

    record.update({
        "restaurant_intent": plan_ctx.get("restaurant_intent"),
        "restaurant_keywords": plan_ctx.get("restaurant_keywords"),
        "preferences": (intent.get("preferences") or {}),
        "recalled": any("memory: 召回 命中" in l for l in r2.task_log),
        "produced_plan": r2.ok,
        "selected_restaurants": [
            {"name": r.get("name"), "typecode": r.get("type"), "cost": r.get("cost")}
            for r in restaurants
        ],
        "spicy": judge_spicy(restaurants),
        "chain_obs": observe_chain(restaurants),
        "task_log": r2.task_log,
        "root_trace_id": r2.root_trace_id,
    })

    # 成本：整棵树的合计，含子 Agent。只看顶层会系统性低估——
    # fact/planning/evaluation 的开销全在子 trace 上。
    try:
        from harness.tracing import SQLiteTraceStorage
        from harness.tracing.analytics import request_cost
        cost = request_cost(SQLiteTraceStorage(db_path=DB_PATH), r2.root_trace_id)
        record["tokens"] = cost.get("total_tokens", 0)
        record["duration_ms"] = cost.get("duration_ms", 0)
    except Exception as e:
        record["tokens"] = 0
        record["cost_error"] = str(e)

    await s2.close()

    sp = record["spicy"]
    mark = "—" if not sp["valid"] else ("✅" if sp["passed"] else "❌")
    print(f"  [{tag}] {mark} 餐厅={sp.get('detail')} tokens={record['tokens']}")
    return record


# ── 汇总 ──────────────────────────────────────────────────────────────

def summarize(records: list[dict], group: str) -> dict:
    rows = [r for r in records if r["group"] == group]
    valid = [r for r in rows if r["spicy"]["valid"]]
    passed = [r for r in valid if r["spicy"]["passed"]]
    tokens = [r["tokens"] for r in rows if r.get("tokens")]
    return {
        "group": group,
        "n": len(rows),
        "valid": len(valid),
        "passed": len(passed),
        "invalid_reason": [r["spicy"]["detail"] for r in rows if not r["spicy"]["valid"]],
        "median_tokens": statistics.median(tokens) if tokens else None,
        "recalled": sum(1 for r in rows if r.get("recalled")),
        "no_plan": sum(1 for r in rows if not r.get("produced_plan")),
    }


async def main(n: int, reuse_memory: bool) -> int:
    from harness.tracing import SQLiteTraceStorage, configure_storage
    configure_storage(SQLiteTraceStorage(db_path=DB_PATH))

    records: list[dict] = []

    hr(f"主组（memory ON）n={n}")
    for i in range(1, n + 1):
        # reuse_memory：只在第一轮建记忆，后续复用。省钱，但代价是
        # 不再覆盖"提取"这一环的非确定性——报告里必须写明这次测的
        # 是"召回→行为"而不是完整链路。
        build = (not reuse_memory) or (i == 1)
        records.append(await run_once(i, memory_on=True, build_memory=build))

    hr(f"对照组（memory OFF）n={n}")
    for i in range(1, n + 1):
        records.append(await run_once(i, memory_on=False, build_memory=False))

    main_s = summarize(records, "main")
    ctrl_s = summarize(records, "control")

    hr("结果")
    for s in (main_s, ctrl_s):
        label = "主组 (ON )" if s["group"] == "main" else "对照 (OFF)"
        rate = f"{s['passed']}/{s['valid']}" if s["valid"] else "无有效样本"
        print(f"  {label}  不辣断言 {rate}   "
              f"有效样本 {s['valid']}/{s['n']}   "
              f"召回命中 {s['recalled']}/{s['n']}   "
              f"中位 tokens {s['median_tokens']}")

    print("\n  ── 有效性判读 ──")
    if ctrl_s["valid"] == 0 or main_s["valid"] == 0:
        print("  🔴 有效样本不足，实验无效。先看是不是大量'方案里没有餐厅'。")
    elif ctrl_s["passed"] == ctrl_s["valid"]:
        print("  🔴 **对照组全部通过 → 实验无效**。")
        print("     说明候选池里本来就没有辛辣餐厅，断言是白送的，")
        print("     主组通过不能归因于记忆。需要换地点或换约束重新设计。")
    elif main_s["passed"] > ctrl_s["passed"]:
        print(f"  🟢 主组 {main_s['passed']}/{main_s['valid']} vs "
              f"对照 {ctrl_s['passed']}/{ctrl_s['valid']} —— 记忆产生了可观测的差异。")
        print(f"     ⚠️ n={n} 只能报比率，不能报率的置信区间，不要写'统计显著'。")
    else:
        print(f"  🟡 主组未优于对照组。这是一个如实的负面结果，"
              f"报告里应当原样呈现。")

    print("\n  ── 连锁（观测项，非断言）──")
    for g in ("main", "control"):
        obs = [r["chain_obs"] for r in records if r["group"] == g]
        brands = [b for o in obs for b in o["known_chain_brands"]]
        paren = [p for o in obs for p in o["name_has_paren"]]
        print(f"    {g:<8} 已知连锁品牌 {len(brands)}  店名带括号 {len(paren)}")
    print("    判据太弱未列为断言：品牌码只覆盖十几个大牌，"
          "括号只说明标注了位置")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"e1_{datetime.now():%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps({
        "config": {"n": n, "reuse_memory": reuse_memory,
                   "session1": SESSION_1_TURNS, "session2": SESSION_2_INPUT},
        "summary": {"main": main_s, "control": ctrl_s},
        "records": records,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  原始数据 → {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--reuse-memory", action="store_true",
                    help="只建一次记忆，后续复用（省钱，但不覆盖提取的非确定性）")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.n, args.reuse_memory)))