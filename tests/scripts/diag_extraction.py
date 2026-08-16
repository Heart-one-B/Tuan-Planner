# scripts/diag_extraction.py
"""诊断：记忆提取为什么 5 次里只成功 2 次。

【背景】
E1 实验暴露：主组 5 次运行里，3 次结束后记忆库是空的。
那 3 次虽然 memory_config 开着，但没有记忆可召回，行为上和对照组
完全等价——所以那个实验实际不是 5 vs 5，是 2 vs 8。

smoke_memory.py 单次跑成功过，让人以为提取是可靠的。**n=1 掩盖了
40% 的失败率**，这是重复实验存在的全部意义。

【三种互斥的可能，修法完全不同，必须先分清】

  A. 提取器判定"本轮无新增记忆"
     → extraction_started 有，extraction_done count=0
     → 是 prompt 或模型非确定性问题。看对话片段里偏好信号是不是
       太弱，或者 do_not_save 负面清单误伤。

  B. 提取子 Agent 调用失败
     → extraction_failed，带 error
     → 看具体异常。连续失败 3 次还会触发熔断，后续直接跳过。

  C. 压根没触发
     → 两个事件都没有
     → maybe_extract 在门槛检查处就返回了。检查
       extraction_every_n_runs 是否为 None，或 segment 是否为空。

【为什么现在看不到原因】
maybe_extract 内部吞掉异常（返回 bookkeeping，不抛），
extract_session 又包了一层 try/except 返回 0。两层吞咽之后，
调用方只看到"写入 0 条"。这个设计本身是对的——提取是增益不是
主流程，它坏了不该拖垮一次成功的会话——但代价是失败原因不可见。
本脚本把 on_memory_event 回调接出来，让被吞掉的事件重新可见。

用法：
    python scripts/diag_extraction.py            # 跑 5 轮统计成功率
    python scripts/diag_extraction.py --n 3
    python scripts/diag_extraction.py --dialogue-only   # 不跑会话，
        用固定对话直接测提取器（省钱，隔离变量）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path("data/memory_diag")
DB_PATH = Path("data/trace.db")
OUT_DIR = Path("data/experiments")

SESSION_1_TURNS = [
    "这周六下午两点到晚上十点，四个人朋友聚会，"
    "从四川大学江安校区出发，找个地方逛逛然后吃个饭",
    "这个火锅算了，我们几个都不太能吃辣，换个清淡点的口味吧",
    "这家看着像连锁店，想找点本地特色的小馆子",
]

CLARIFY_ANSWER = "朋友聚会，四个人，下午两点到晚上十点，从四川大学江安校区出发"

# --dialogue-only 用的固定对话：把"会话能不能跑通"这个变量摘掉，
# 只测提取器本身。assistant 的回复用真实运行里出现过的形状，
# 不是随手编的——提取器看到的输入形状必须和生产一致，否则测出来
# 的成功率不可迁移。
FIXED_DIALOGUE = [
    {"role": "user", "content": SESSION_1_TURNS[0]},
    {"role": "assistant", "content":
        "# 桌游推理+火锅聚餐\n\n## 行程安排\n"
        "- 14:00-17:00 九号桌游探案馆\n"
        "- 17:30-19:00 巴国妹子毛肚火锅(蓝光·圣菲悦城店)\n\n"
        "这个方案可以吗？需要调整随时告诉我。"},
    {"role": "user", "content": SESSION_1_TURNS[1]},
    {"role": "assistant", "content":
        "# 桌游推理+粤菜聚餐\n\n## 行程安排\n"
        "- 14:00-17:00 九号桌游探案馆\n"
        "- 17:30-19:00 绿茶餐厅(成都双流茂业天地店)\n\n"
        "这个方案可以吗？"},
    {"role": "user", "content": SESSION_1_TURNS[2]},
    {"role": "assistant", "content":
        "# 桌游推理+本地小馆\n\n## 行程安排\n"
        "- 14:00-17:00 九号桌游探案馆\n"
        "- 17:30-19:00 川小厨•猪油面\n\n"
        "这个方案可以吗？"},
]


def hr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


class EventCollector:
    """接住 on_memory_event，让被两层 try/except 吞掉的失败原因重新可见。"""

    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        self.events.append(dict(event))
        etype = event.get("type", "?")
        err = event.get("error")
        line = f"      · {etype} count={event.get('count')}"
        if err:
            line += f"\n        error: {err[:300]}"
        print(line)

    def verdict(self) -> str:
        """把事件序列翻译成三种诊断结论之一。"""
        types = {e.get("type") for e in self.events}
        if "extraction_failed" in types:
            return "B_failed"
        if "extraction_skipped_circuit_open" in types:
            return "B_circuit_open"
        if "extraction_done" in types:
            done = next(e for e in self.events if e.get("type") == "extraction_done")
            return "A_no_new" if not done.get("count") else "ok"
        if "extraction_started" in types:
            return "B_no_done"      # 开始了但没有 done，异常路径
        return "C_not_triggered"


VERDICT_HELP = {
    "ok": "提取成功",
    "A_no_new": "提取器判定「本轮无新增记忆」—— prompt 或模型非确定性问题。"
                "看对话里的偏好信号是否太弱，或被 do_not_save 负面清单误伤"
                "（'一次性的任务细节'这一条可能把'不吃辣'误判成本次对话专属）",
    "B_failed": "提取子 Agent 调用失败 —— 看上面的 error",
    "B_circuit_open": "熔断已打开（连续失败 ≥3 次），后续全部跳过。"
                      "熔断是进程级的，重启即半开",
    "B_no_done": "started 之后没有 done —— 异常路径，翻 [MemoryExtraction] 日志",
    "C_not_triggered": "根本没触发 —— extraction_every_n_runs 为 None？"
                       "或 segment 为空（对话被 _strip_system_messages 滤光了）",
}


async def one_round(rep: int, dialogue_only: bool) -> dict:
    from src.memory.wiring import (
        build_memory_config, configure_memory, extract_session, list_memories,
    )
    from src.model.factory import build_llm_client
    from src.session_runner import Session

    tag = f"D{rep}-{uuid.uuid4().hex[:4]}"
    print(f"\n  ── 第 {rep} 轮 [{tag}] ──")

    # 每轮独立的记忆目录：跨轮共享会让第 2 轮的"去重"逻辑把
    # 第 1 轮已记的内容判成重复而不再写入，那会被误读成"提取失败"。
    if MEMORY_DIR.exists():
        shutil.rmtree(MEMORY_DIR)

    collector = EventCollector()
    cfg = build_memory_config(memory_dir=MEMORY_DIR)
    cfg.on_memory_event = collector
    configure_memory(cfg)

    if dialogue_only:
        dialogue = list(FIXED_DIALOGUE)
        print(f"    使用固定对话（{len(dialogue)} 条），跳过会话")
    else:
        session = Session(session_id=f"diagext-{tag}",
                          auto_clarify_answer=CLARIFY_ANSWER)
        for text in SESSION_1_TURNS:
            r = await session.say(text)
            if not r.ok:
                print(f"    ⚠️ 某轮未出方案")
        dialogue = list(session.dialogue)
        print(f"    会话产生 {len(dialogue)} 条对话")

    print("    提取事件：")
    written = await extract_session(
        session_id=f"diagext-{tag}",
        dialogue=dialogue,
        llm_client=build_llm_client(),
    )

    mems = list_memories()
    verdict = collector.verdict()
    ok = bool(mems)

    print(f"    {'✅' if ok else '❌'} 记忆库 {len(mems)} 条"
          f"（写入报告 {written}）  判定={verdict}")
    for m in mems:
        print(f"       - {m.name} [{m.type}] {m.description}")

    return {
        "rep": rep, "tag": tag, "ok": ok,
        "n_memories": len(mems),
        "memories": [{"name": m.name, "type": m.type,
                      "description": m.description, "content": m.content}
                     for m in mems],
        "verdict": verdict,
        "events": collector.events,
        "dialogue_len": len(dialogue),
        "dialogue": dialogue,
    }


async def main(n: int, dialogue_only: bool, verbose: bool) -> int:
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

    from harness.tracing import SQLiteTraceStorage, configure_storage
    configure_storage(SQLiteTraceStorage(db_path=DB_PATH))

    hr(f"提取成功率诊断 n={n}"
       + ("（固定对话，隔离会话变量）" if dialogue_only else "（完整会话）"))

    records = []
    for i in range(1, n + 1):
        # 熔断是进程级的模块字典。连续失败 3 次之后后面全部跳过，
        # 那会让"失败率"看起来比真实值高。每轮重置，保证各轮独立。
        from harness.memory import reset_circuit
        reset_circuit(f"diagext-{i}")
        records.append(await one_round(i, dialogue_only))

    hr("结果")
    ok = sum(1 for r in records if r["ok"])
    print(f"  成功 {ok}/{n}")

    counts = Counter(r["verdict"] for r in records)
    print("\n  判定分布：")
    for v, c in counts.most_common():
        print(f"    {c}×  {v}")
        print(f"        {VERDICT_HELP.get(v, '?')}")

    # 成功的那几轮记了什么名字——名字不稳定本身也是个问题：
    # 同一段对话每次生成不同的记忆名，会让"同名覆盖更新"这个
    # 去重机制失效，长期使用下记忆库会积累一堆语义重复的条目。
    names = [m["name"] for r in records for m in r["memories"]]
    if names:
        print("\n  记忆命名（跨轮是否稳定）：")
        for name, c in Counter(names).most_common():
            print(f"    {c}×  {name}")
        distinct = len(set(names))
        if distinct > 2:
            print(f"    ⚠️ {distinct} 个不同的名字 —— 命名不稳定。"
                  f"同名覆盖的去重机制会失效，长期会积累语义重复的记忆")

    print("\n  ── 下一步 ──")
    if ok == n:
        print("  提取稳定。E1 里那 3 次失败可能与会话本身有关"
              "（去掉 --dialogue-only 再跑一次对比）")
    elif counts.get("A_no_new"):
        print("  主因是提取器判定「无新增」。这是 prompt 层面的问题：")
        print("   · 检查 MEMORY_DO_NOT_SAVE 里「一次性的任务细节」这一条")
        print("     是否把'不吃辣'误判成了本次对话专属的临时状态")
        print("   · 检查对话片段里偏好信号是否够强（否决式表达比声明式弱）")
    elif counts.get("B_failed") or counts.get("B_circuit_open"):
        print("  主因是子 Agent 调用失败。看上面的 error 内容。")
    elif counts.get("C_not_triggered"):
        print("  主因是根本没触发。检查 build_memory_config 的"
              " extraction_every_n_runs。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"diag_extraction_{datetime.now():%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps({
        "n": n, "dialogue_only": dialogue_only,
        "success": ok, "verdicts": dict(counts), "records": records,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  原始数据 → {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dialogue-only", action="store_true",
                    help="用固定对话直接测提取器，跳过会话（省钱、隔离变量）")
    ap.add_argument("--quiet", action="store_true", help="不开 INFO 日志")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.n, args.dialogue_only, not args.quiet)))