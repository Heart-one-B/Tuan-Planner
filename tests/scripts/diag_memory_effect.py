# scripts/diag_memory_effect.py
"""诊断两个红旗，不修改任何 Agent 源码。

红旗一：Session 1 的第 2、3 轮回复与第 1 轮**逐字相同**——用户说
        "火锅算了"、"这家像连锁店"，方案纹丝不动。调整链路可能
        根本没跑通，而 smoke_memory.py 只检查最后一轮，漏掉了。

红旗二：Session 2 搜到 0 家餐厅。用户明确要"吃个饭"、
        plan_mode=activity_plus_meal，餐厅候选池却是空的。
        **可能是记忆造成的负面影响**——must_avoid=['连锁店','重辣']
        进了 fact_task，搜索规划可能生成了"本地特色小馆子"这类
        高德搜不到的关键词。如果属实，这比"记忆有效"更有价值：
        记忆确实改变了搜索行为，但改坏了。

【为什么用运行时包裹而不是加 log】
诊断代码和被诊断的代码必须分开。在 FactAgent 里加 logger 行，
诊断结束后要么忘了删（生产里多一堆噪声），要么删掉之后这次诊断
就不可复现了。运行时包裹只活在这个脚本的进程里，源码一个字不动，
而且脚本本身就是"这个问题是怎么查出来的"的可执行记录。

用法：
    python scripts/diag_memory_effect.py              # 完整跑（含 Session 1）
    python scripts/diag_memory_effect.py --reuse      # 复用已有记忆，省一轮钱
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

MEMORY_DIR = Path("data/memory")
DB_PATH = Path("data/trace.db")

SESSION_1_TURNS = [
    "这周六下午两点到晚上十点，四个人朋友聚会，"
    "从四川大学江安校区出发，找个地方逛逛然后吃个饭",
    "火锅算了，我们几个都不太能吃辣，换个别的口味吧",
    "这家看着像连锁店，想找点本地特色的小馆子",
]

SESSION_2_INPUT = (
    "这周日下午两点到晚上九点，还是四个人，"
    "从四川大学江安校区出发，找个地方聚一聚吃个饭"
)

CLARIFY_ANSWER = "朋友聚会，四个人，下午两点到晚上十点，从四川大学江安校区出发"


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ── 运行时录制器 ───────────────────────────────────────────────────────

@dataclass
class Recorder:
    """记录一次运行里 FactAgent 收到的任务和实际发出的搜索。

    抓的是**实际执行的搜索**而不是模型输出的搜索计划 JSON——
    _flatten_plan 会展开列表、去重，中间可能丢东西，只看模型输出
    会看到一个和真实行为不一致的计划。行为是唯一可信的事实。
    """
    fact_tasks: list = field(default_factory=list)
    searches: list = field(default_factory=list)   # {keywords, is_restaurant, count, error}

    @property
    def restaurant_searches(self) -> list:
        return [s for s in self.searches if s["is_restaurant"]]

    @property
    def restaurant_hits(self) -> int:
        return sum(s["count"] for s in self.restaurant_searches)


@contextmanager
def record(rec: Recorder):
    """在 FactAgent.run 和 FactToolset.search_pois 外面套一层，
    退出时无条件还原（哪怕中途抛异常）——诊断脚本自己留下副作用
    是最没道理的事。"""
    from agents.fact.agent import FactAgent
    from agents.fact.tools import FactToolset

    orig_run = FactAgent.run
    orig_search = FactToolset.search_pois

    async def run_wrapper(self, task, plan_mode="", parent_span=None):
        rec.fact_tasks.append({"task": task, "plan_mode": plan_mode})
        return await orig_run(self, task, plan_mode=plan_mode, parent_span=parent_span)

    async def search_wrapper(self, keywords, is_restaurant=False):
        entry = {"keywords": keywords, "is_restaurant": bool(is_restaurant),
                 "count": 0, "error": None}
        try:
            raw = await orig_search(self, keywords, is_restaurant)
            entry["count"] = len((json.loads(raw) or {}).get("pois") or [])
            return raw
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            rec.searches.append(entry)

    FactAgent.run = run_wrapper
    FactToolset.search_pois = search_wrapper
    try:
        yield rec
    finally:
        FactAgent.run = orig_run
        FactToolset.search_pois = orig_search


def dump_searches(rec: Recorder, indent: str = "    ") -> None:
    if not rec.searches:
        print(f"{indent}(没有发出任何搜索)")
        return
    for s in rec.searches:
        kind = "餐厅" if s["is_restaurant"] else "活动"
        note = f"  ⚠️ {s['error']}" if s["error"] else ""
        flag = "  ← 空结果" if not s["error"] and s["count"] == 0 else ""
        print(f"{indent}[{kind}] {s['keywords']:<24} → {s['count']:>2} 家{flag}{note}")
    print(f"{indent}餐厅搜索 {len(rec.restaurant_searches)} 次，"
          f"累计命中 {rec.restaurant_hits} 家")


def dump_fact_task(rec: Recorder, indent: str = "    ") -> None:
    for t in rec.fact_tasks:
        print(f"{indent}--- fact_task (plan_mode={t['plan_mode']}) ---")
        for line in t["task"].split("\n"):
            print(f"{indent}  {line}")


# ── 红旗一：Session 1 的调整链路 ────────────────────────────────────────

async def probe_adjust_chain(tag: str) -> None:
    from src.session_runner import Session

    hr("红旗一：Session 1 三轮，调整链路有没有真的跑")
    session = Session(session_id=f"diag1-{tag}", auto_clarify_answer=CLARIFY_ANSWER)
    replies: list[str] = []

    for i, text in enumerate(SESSION_1_TURNS, 1):
        print(f"\n  ─── 第 {i} 轮 ───")
        print(f"  用户: {text}")
        r = await session.say(text)
        replies.append(r.reply or "")

        print("  task_log:")
        for line in r.task_log:
            print(f"    {line}")

        route = (r.state.get("feedback_route") or "?")
        has_new = r.state.get("has_new_plan")
        print(f"  feedback_route={route}  has_new_plan={has_new}")

        if i > 1:
            same = replies[i - 1] == replies[i - 2]
            print(f"  与上一轮回复{'完全相同 ❌' if same else '不同 ✅'}")

    print("\n  ── 判读 ──")
    if replies[0] == replies[1] == replies[2]:
        print("  ❌ 三轮回复逐字相同 —— 用户的两次否决没有产生任何效果。")
        print("     看上面每轮的 feedback_route：")
        print("       route=new_plan → routing 没认出这是对当前方案的调整")
        print("       route=adjust 但 has_new_plan=False → orchestrator 跑了")
        print("         但没产出新方案（工具没调 / replan 失败 / evaluate 没跑）")
        print("     后果：Session 1 埋不进正确偏好，E1 实验的前提不成立。")
    else:
        print("  ✅ 调整链路有响应（回复发生了变化）")

    await session.close()


# ── 红旗二：记忆对搜索行为的影响 ──────────────────────────────────────

async def probe_search_effect(tag: str) -> None:
    from src.memory.wiring import build_memory_config, configure_memory, list_memories
    from src.session_runner import Session

    hr("红旗二：同一句输入，记忆开 vs 关，搜索行为差在哪")

    mems = list_memories()
    print(f"  当前记忆库 {len(mems)} 条：{[m.name for m in mems]}")
    if not mems:
        print("  ⚠️  记忆库为空，开/关两组不会有差别。先跑一次不带 --reuse 的完整流程。")

    print(f"\n  输入: {SESSION_2_INPUT}")

    # ── A 组：记忆开 ──
    print("\n  ═══ A 组：记忆 ON ═══")
    configure_memory(build_memory_config(memory_dir=MEMORY_DIR))
    rec_on = Recorder()
    with record(rec_on):
        s_on = Session(session_id=f"diagON-{tag}", auto_clarify_answer=CLARIFY_ANSWER)
        r_on = await s_on.say(SESSION_2_INPUT)

    prefs_on = ((r_on.state.get("intent") or {}).get("preferences")) or {}
    print(f"  intent.preferences = {prefs_on}")
    print("\n  实际发出的搜索：")
    dump_searches(rec_on)

    # ── B 组：记忆关 ──
    print("\n  ═══ B 组：记忆 OFF（对照） ═══")
    configure_memory(None)
    rec_off = Recorder()
    with record(rec_off):
        s_off = Session(session_id=f"diagOFF-{tag}", auto_clarify_answer=CLARIFY_ANSWER)
        r_off = await s_off.say(SESSION_2_INPUT)

    prefs_off = ((r_off.state.get("intent") or {}).get("preferences")) or {}
    print(f"  intent.preferences = {prefs_off}")
    print("\n  实际发出的搜索：")
    dump_searches(rec_off)

    # ── 对照判读 ──
    hr("对照判读")
    print(f"  A(ON )  餐厅搜索 {len(rec_on.restaurant_searches)} 次 → "
          f"命中 {rec_on.restaurant_hits} 家")
    print(f"  B(OFF)  餐厅搜索 {len(rec_off.restaurant_searches)} 次 → "
          f"命中 {rec_off.restaurant_hits} 家")

    if rec_on.restaurant_hits == 0 and rec_off.restaurant_hits > 0:
        print("\n  🔴 因果成立：记忆注入导致餐厅搜不到。")
        print("     记忆把 must_avoid/diet_preference 变成了搜索关键词，")
        print("     而这类描述性词（'本地特色小馆子''非连锁'）不是高德")
        print("     能检索的 POI 类目 —— 偏好被错误地用在了'搜什么'上，")
        print("     它本该只用在'从搜到的里面挑哪个'。")
        print("     这是记忆链路的真实缺陷，比'记忆有效'更值得写进报告。")
    elif rec_on.restaurant_hits == 0 and rec_off.restaurant_hits == 0:
        print("\n  🟡 两组都搜不到餐厅 —— 与记忆无关，是搜索规划本身的问题。")
        print("     看上面 A/B 两组的餐厅关键词是不是都过于具体/描述性。")
    elif rec_on.restaurant_hits > 0:
        print("\n  🟢 A 组餐厅搜得到 —— 上次的 0 家可能是偶发（模型非确定性）。")
        print("     值得多跑几次确认，不要凭单次观察下结论。")

    print("\n  ── A 组的 fact_task（记忆注入后，FactAgent 实际收到的任务）──")
    dump_fact_task(rec_on)

    await s_on.close()


async def main(reuse: bool) -> int:
    from harness.tracing import SQLiteTraceStorage, configure_storage
    from src.memory.wiring import build_memory_config, configure_memory

    configure_storage(SQLiteTraceStorage(db_path=DB_PATH))
    configure_memory(build_memory_config(memory_dir=MEMORY_DIR))

    tag = uuid.uuid4().hex[:6]

    if not reuse:
        await probe_adjust_chain(tag)
    else:
        print("（--reuse：跳过 Session 1，直接用已有记忆做红旗二的对照）")

    await probe_search_effect(tag)

    hr("下一步")
    print("  两个红旗查清之后再决定 E1 怎么跑：")
    print("   · 红旗一坐实 → Session 1 埋不进偏好，剧本或调整链路要先修")
    print("   · 红旗二坐实 → 主组会大面积'无餐厅'，断言全部 invalid，")
    print("                  必须先把偏好从'搜什么'挪回'挑哪个'")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="跳过 Session 1，复用已有记忆")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.reuse)))