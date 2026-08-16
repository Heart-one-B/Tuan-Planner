# scripts/smoke_memory.py
"""A2 验收：记忆的写入与召回是否真的发生。

这不是 E1 实验，是它的前置检查。区别很重要：

  本脚本  只验证"记忆链路通了"——Session 1 写进去了吗，
          Session 2 召回并注入了吗。不判断偏好有没有被遵守。
  E1 实验  验证"记忆有没有改变行为"——n=5 双组，硬断言，比率报告。

先验证链路再跑实验，是因为链路不通时实验会产出"主组和对照组
一样"的结果，而这个结果看起来像"记忆无效"，实际是记忆没跑。
两者在报告里的结论完全相反，必须先排除后者。

用法：
    python scripts/smoke_memory.py
    python scripts/smoke_memory.py --fresh     # 先清空记忆库
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import uuid
from pathlib import Path

MEMORY_DIR = Path("data/memory")
DB_PATH = Path("data/trace.db")

# Session 1：偏好从**否决**中暴露，不从声明中暴露。
# 用户不说"我不吃辣"，而是先看到川菜方案再拒绝——这是真实对话的
# 形状，也让提取器面对真实难度：它得从对话里推断，而不是抄一句
# 现成的自述。剧本先固定、断言后写，顺序如实记录在实验报告里。
SESSION_1_TURNS = [
    "这周六下午两点到晚上十点，四个人朋友聚会，"
    "从四川大学江安校区出发，找个地方逛逛然后吃个饭",
    "火锅算了，我们几个都不太能吃辣，换个别的口味吧",
    "这家看着像连锁店，想找点本地特色的小馆子",
]

# Session 2：全新 session_id，不带任何 history，**不提偏好**。
SESSION_2_INPUT = (
    "这周日下午两点到晚上九点，还是四个人，"
    "从四川大学江安校区出发，找个地方聚一聚吃个饭"
)

CLARIFY_ANSWER = (
    "朋友聚会，四个人，下午两点到晚上十点，从四川大学江安校区出发"
)


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


async def main(fresh: bool) -> int:
    from harness.tracing import SQLiteTraceStorage, configure_storage
    from src.memory.wiring import build_memory_config, configure_memory, list_memories
    from src.session_runner import Session

    configure_storage(SQLiteTraceStorage(db_path=DB_PATH))

    if fresh and MEMORY_DIR.exists():
        shutil.rmtree(MEMORY_DIR)
        print(f"已清空 {MEMORY_DIR}")

    configure_memory(build_memory_config(memory_dir=MEMORY_DIR))

    before = len(list_memories())
    tag = uuid.uuid4().hex[:6]

    # ── Session 1：埋偏好 ──
    hr(f"Session 1：三轮对话，偏好从否决中暴露（记忆库起始 {before} 条）")
    s1 = Session(session_id=f"mem1-{tag}", auto_clarify_answer=CLARIFY_ANSWER)
    for i, text in enumerate(SESSION_1_TURNS, 1):
        print(f"\n  [{i}] 用户: {text}")
        r = await s1.say(text)
        print(f"      回复: {(r.reply or '(空)')[:160]}")
        if not r.ok:
            fatal = [e for e in r.errors if not e.get("recoverable", False)]
            print(f"      ⚠️  本轮未出方案，errors={fatal}")

    print("\n  触发提取…")
    written = await s1.close()

    # ── 检查记忆库 ──
    hr("提取结果")
    memories = list_memories()
    print(f"  记忆库 {before} → {len(memories)} 条（本次新增 {written}）")
    for m in memories:
        print(f"\n  ── {m.name} [{m.type}] {m.updated_at:%Y-%m-%d} ──")
        print(f"     {m.description}")
        body = (m.content or "").strip().replace("\n", "\n     ")
        print(f"     {body[:300]}")

    if not memories:
        print("\n  ❌ 记忆库是空的 —— 提取没有产出任何记忆")
        print("     可能原因：extraction_every_n_runs 没生效 / 提取器判定"
              "无值得记的内容 / 提取子 Agent 调用失败（翻 [memory] 日志）")
        return 1
    print(f"\n  ✅ 提取产出 {len(memories)} 条记忆")

    # ── Session 2：全新会话，不提偏好 ──
    hr("Session 2：全新 session_id，不带 history，不提偏好")
    print(f"  用户: {SESSION_2_INPUT}")
    s2 = Session(session_id=f"mem2-{tag}", auto_clarify_answer=CLARIFY_ANSWER)
    r2 = await s2.say(SESSION_2_INPUT)

    print("\n  --- task_log ---")
    for line in r2.task_log:
        print(f"    {line}")

    # 召回是否发生：task_log 里 intent_node 写的那条
    recalled = any("memory: 召回 命中" in line for line in r2.task_log)
    print(f"\n  {'✅' if recalled else '❌'} 召回"
          f"{'命中并注入' if recalled else '未命中 —— 记忆没有进入本轮上下文'}")
    if not recalled:
        print("     检查：recall_min_memories 是否 ≤ 记忆条数、"
              "recall_top_k 是否非 None、选择器是否判定全部不相关")

    # 偏好是否进了 intent（记忆生效的第一现场）
    prefs = ((r2.state.get("intent") or {}).get("preferences")) or {}
    print(f"\n  intent.preferences = {prefs}")

    restaurants = s2.selected_restaurants()
    if restaurants:
        print(f"\n  --- 选中方案里的餐厅 ---")
        for r in restaurants:
            cost = f"{r['cost']:.0f}" if r.get("cost") is not None else "—"
            print(f"    {cost:>6}  {r.get('name')}  [{r.get('type')}]")
    else:
        print("\n  ⚠️  选中方案里没有餐厅（或未出方案）")

    await s2.close()

    hr("汇总")
    print(f"  Session 1 提取   {'✅' if memories else '❌'}  ({len(memories)} 条)")
    print(f"  Session 2 召回   {'✅' if recalled else '❌'}")
    print(f"  Session 2 出方案 {'✅' if r2.ok else '❌'}")
    ok = bool(memories) and recalled and r2.ok
    if ok:
        print("\n  记忆链路通了，可以跑 E1 实验。")
        print("  注意：本脚本**不判断偏好有没有被遵守**——那是 E1 的事，")
        print("  需要双组对照和 n=5，单次观察说明不了问题。")
    else:
        print("\n  链路不通，先修这里。此时跑 E1 只会得到误导性的结论。")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="先清空记忆库")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.fresh)))