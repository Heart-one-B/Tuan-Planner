# scripts/baseline_d1.py
"""D1 baseline 采集：接入 harness **之前**的运行时特征。

【这份数据为什么不可逆】
它测的不是输出质量（那是 evals/ 67 条的事），是**运行时特征**：
token 随轮次怎么涨、涨在哪个 Agent 身上、约束在第几轮消失、
失败以什么形态出现。Orchestrator 接上 harness 之后，"接入前"
的这条曲线再也造不出来。

【和 evals 的分工】
  evals/run.py    独立用例，每条一个新会话，测断言 passed/failed
  本脚本          一段连续会话，状态跨轮累积，测曲线和失败形态

【期望值纪律】
TURNS 里每一轮的 expect_l1 是**跑之前**写死的。跑完不许反调——
反调就不是观测，是自我确认（同 evals/cases.yaml 开头那条）。

用法：
    python scripts/baseline_d1.py                 # 跑全 12 轮
    python scripts/baseline_d1.py --turns 3       # 只跑前 3 轮（试跑）
    python scripts/baseline_d1.py --dry-run       # 不调 LLM，只验证管线
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

# 脚本在 scripts/，命令行通常在项目根——统一 chdir 到根，
# 相对路径（data/trace.db 等）才有一致含义。这个项目已经为
# "同一份代码两种运行方式下相对路径含义不同"撞过几次。
_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

DB_PATH = Path("data/trace.db")
OUT_DIR = Path("data/baseline")


# ══════════════════════════════════════════════════════════════
# 12 轮脚本（跑之前锁定，跑完不许改）
# ══════════════════════════════════════════════════════════════
#
# expect_l1  这一轮结束后，可代码校验的硬约束集合应该是什么。
#            现在系统里没有 L1，所以这一列是**纯人工预测**，
#            用来和实际观测对照。
# watch      这一轮重点看什么。
#
TURNS: list[dict] = [
    {
        "turn": 1,
        "input": "这周六下午两点到晚上十点，四个人朋友聚会，从四川大学江安校区出发，找个地方逛逛然后吃个饭",
        "expect_l1": {},
        "watch": "基础链路；记下初始 token 作为基准",
    },
    {
        "turn": 2,
        "input": "这个餐厅换一家吧",
        "expect_l1": {},
        "watch": "route 应判 adjust 而非 new_plan",
    },
    {
        "turn": 3,
        "input": "我们不吃辣",
        "expect_l1": {"no_spicy": True},
        "watch": "no_spicy 首次出现；此后每轮都应保持 True（除第 9 轮）",
    },
    {
        "turn": 4,
        "input": "也别太贵，人均一百以内",
        "expect_l1": {"no_spicy": True, "budget_per_person": 100},
        "watch": "【预测静默丢弃】IntentResult schema 无预算字段，"
                 "预期 budget_seen=False。pydantic 丢弃，无告警。",
    },
    {
        "turn": 5,
        "input": "下午的活动换一个",
        "expect_l1": {"no_spicy": True, "budget_per_person": 100},
        "watch": "约束不该因为换活动而变",
    },
    {
        "turn": 6,
        "input": "晚一点开始吧，三点出发",
        "expect_l1": {"no_spicy": True, "budget_per_person": 100},
        "watch": "改时间幅度小，route 仍应是 adjust；判成 new_plan 会重置会话",
    },
    {
        "turn": 7,
        "input": "这个餐厅太远了，换个近点的",
        "expect_l1": {"no_spicy": True, "budget_per_person": 100, "max_eta": "< 上轮实测"},
        "watch": "F-002 现场。看 orchestrator 有没有传 max_eta_minutes，"
                 "以及 action_log 里的 dropped 数是否与实际一致（F-006）",
    },
    {
        "turn": 8,
        "input": "中途想买杯奶茶",
        "expect_l1": {"no_spicy": True, "budget_per_person": 100, "max_eta": "保持"},
        "watch": "【预测静默丢弃】waypoints 在 schema 里声明但 _synthesize "
                 "硬编码 []（L2-13 哨兵）。第二个静默丢弃实例。",
    },
    {
        "turn": 9,
        "input": "算了还是吃火锅吧",
        # ── 本轮期望按方案 A 锁定：后说的覆盖先说的，但**必须告知** ──
        "expect_l1": {"no_spicy": "revoked", "budget_per_person": 100, "max_eta": "保持"},
        "watch": "【核心观测】判定标准（跑前锁定）：\n"
                 "  撤销了 no_spicy 且回复里显式提到先前的不吃辣 → 正确（P1）\n"
                 "  撤销了但只字未提                            → P0-A 静默改写\n"
                 "  未撤销，搜索关键词出现「不辣的火锅」类自相矛盾 → P0-A\n"
                 "  reply_mentions_conflict 需人工标注，机器判不了",
    },
    {
        "turn": 10,
        "input": "行，就这个吧",
        "expect_l1": {"budget_per_person": 100, "max_eta": "保持"},
        "watch": "确认轮，route 应判 chat，不该触发 replan 烧钱（同 L3-06）",
    },
    {
        "turn": 11,
        "input": "再把晚上的餐厅换一家",
        "expect_l1": {"budget_per_person": 100, "max_eta": "保持"},
        "watch": "第 9 轮之后约束还在不在",
    },
    {
        "turn": 12,
        "input": "可以了，就这样",
        "expect_l1": {"budget_per_person": 100, "max_eta": "保持"},
        "watch": "收尾；总 token 与第 1 轮对比",
    },
]


# ══════════════════════════════════════════════════════════════
# trace.db 聚合
# ══════════════════════════════════════════════════════════════
#
# ⚠️ Span.begin(name, task) 落到 start_trace(trace_id, session_id=name,
#    user_input=task)——**traces.session_id 这一列存的其实是 Agent 名**
#    （"plan-request" / "orchestrator" / "evaluation" / "fact" …），
#    不是会话 id。列名和内容不符，但正因如此我们能免费拿到按 Agent
#    的 token 归属。这条本身值得记一条 finding。

_TREE_SQL = """
WITH RECURSIVE tree(trace_id, agent) AS (
    SELECT trace_id, session_id FROM traces WHERE trace_id = ?
    UNION ALL
    SELECT t.trace_id, t.session_id
    FROM traces t JOIN tree ON t.parent_trace_id = tree.trace_id
)
SELECT tree.agent,
       COUNT(c.event_id),
       COALESCE(SUM(c.prompt_tokens), 0),
       COALESCE(SUM(c.completion_tokens), 0),
       COALESCE(SUM(CASE WHEN c.token_source != 'api_usage' THEN 1 ELSE 0 END), 0)
FROM tree LEFT JOIN llm_calls c ON c.trace_id = tree.trace_id
GROUP BY tree.agent
"""


def aggregate_tokens(root_trace_id: str) -> dict:
    """沿 parent_trace_id 递归聚合整棵 span 树的 token，按 Agent 分列。

    分列是必须的：总数只能看出"涨了"，看不出**谁在涨**。而 baseline
    的结论要落到具体 Agent 上才有因果（比如"增长全在 intent，因为
    adjust_history 每轮拼进 prompt"）。
    """
    if not DB_PATH.exists():
        return {"error": f"{DB_PATH} 不存在"}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(_TREE_SQL, (root_trace_id,)).fetchall()
        conn.close()
    except Exception as e:                      # 观测设施失败不该中断采集
        return {"error": f"{type(e).__name__}: {e}"}

    by_agent, tot_p, tot_c, tot_calls, tot_est = {}, 0, 0, 0, 0
    for agent, n_calls, p, c, n_est in rows:
        if not n_calls:
            continue
        by_agent[agent or "?"] = {
            "llm_calls": n_calls, "prompt_tokens": p, "completion_tokens": c,
            "estimated_calls": n_est,
        }
        tot_p += p; tot_c += c; tot_calls += n_calls; tot_est += n_est

    return {
        "by_agent": by_agent,
        "llm_calls": tot_calls,
        "prompt_tokens": tot_p,
        "completion_tokens": tot_c,
        "total_tokens": tot_p + tot_c,
        # estimated 混进来会污染曲线（原则 2：usage 收不到不伪造）。
        # 非 0 时报表要按 token_source 分组，不能直接把两者相加比较。
        "estimated_calls": tot_est,
    }


# ══════════════════════════════════════════════════════════════
# 从 state / task_log 里抠观测项
# ══════════════════════════════════════════════════════════════

_NO_SPICY_RE = re.compile(r"no_spicy=(True|False)")
_ETA_FILTER_RE = re.compile(r"eta≤(\d+)min 过滤掉 (\d+) 个候选")


def observe(result, prev_max_eta: int | None, prev_log_len: int = 0) -> dict:
    """把这一轮的观测项从 state 里挖出来。

    每一项都用 try 兜住：采集脚本自己崩掉会浪费掉前面几轮的真实
    LLM 花费，而这一整段会话是不可逆的。宁可某一列是 None。

    ⚠️ task_log 是**跨轮累积**的（整个 state 带进下一轮）。所以
    必须切出本轮新增的那一段再匹配——否则第 3 轮会读到第 1 轮
    留下的 no_spicy=False。试跑时踩过这个，正是"发送方以为传了、
    接收方读到的是别的东西"的又一个实例。
    """
    st = result.state or {}
    task_log = result.task_log or []
    turn_log = task_log[prev_log_len:]          # 只看本轮新增
    log_text = "\n".join(str(x) for x in turn_log)

    def safe(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    pc = safe(lambda: st.get("plan_context") or {}, {}) or {}
    intent = safe(lambda: st.get("intent") or {}, {}) or {}
    outputs = safe(lambda: st.get("agent_outputs") or {}, {}) or {}
    fact_data = safe(lambda: (outputs.get("fact") or {}).get("data") or {}, {}) or {}
    orch_data = safe(lambda: (outputs.get("orchestrator") or {}).get("data") or {}, {}) or {}

    # no_spicy：fact_node 已经把它写进 task_log，不用改生产代码。
    # 取**最后一个**匹配：一轮内 fact 可能被调用多次。
    # 值为 None 的语义是"本轮没有任何地方计算过 no_spicy"——
    # 这不是缺测，它本身就是观测结果（adjust 路径不走 fact_node）。
    hits = _NO_SPICY_RE.findall(log_text)
    no_spicy = (hits[-1] == "True") if hits else None
    fact_ran = any(str(x).startswith("fact:") for x in turn_log)

    # 预算：IntentResult schema 里没有这个字段，预期整场恒为 False
    blob = json.dumps({"intent": intent, "plan_context": pc}, ensure_ascii=False)
    budget_seen = ("100" in blob) or ("人均" in blob) or ("预算" in blob)

    # eta 过滤：日志声称过滤掉的数量（F-006：声称 ≠ 实际）
    eta_claim = _ETA_FILTER_RE.search(log_text)

    etas = [p.get("eta_minutes") for k in ("activities", "restaurants", "waypoints")
            for p in (fact_data.get(k) or []) if isinstance(p, dict)]
    etas = [e for e in etas if isinstance(e, (int, float))]

    return {
        "route": st.get("feedback_route"),
        "adjust_history_len": len(st.get("adjust_history") or []),
        "adjust_history": list(st.get("adjust_history") or []),

        # fact_ran=False + no_spicy=None → 本轮没有任何辣度执行点。
        # 【试跑实测】adjust 路径不经过 fact_node，且 Orchestrator
        # 的 _tool_search_pois 调 FactAgent 时不传 no_spicy（默认
        # False）——「不吃辣」在整条调整路径上没有执行机构。
        "fact_ran": fact_ran,
        "searched_pois": any("search_pois" in str(a)
                             for a in (orch_data.get("action_log") or [])),
        "no_spicy": no_spicy,
        "budget_seen": budget_seen,
        "waypoints_len": len(fact_data.get("waypoints") or []),
        "restaurant_keywords": pc.get("restaurant_keywords"),
        "diet": (pc.get("preferences") or {}).get("diet"),
        "avoid": (pc.get("preferences") or {}).get("avoid"),
        "max_traffic_minutes": pc.get("max_traffic_minutes"),

        "eta_filter_claimed": (
            {"max_eta": int(eta_claim.group(1)), "dropped_claimed": int(eta_claim.group(2))}
            if eta_claim else None
        ),
        "poi_pool_size": sum(
            len(fact_data.get(k) or []) for k in ("activities", "restaurants", "waypoints")
        ),
        "max_eta_in_pool": max(etas) if etas else None,
        "prev_max_eta": prev_max_eta,

        "orchestrator_actions": orch_data.get("action_log"),
        "origin_city": pc.get("origin_city"),
        "errors": result.errors,
        "fatal_errors": [e for e in (result.errors or [])
                         if not e.get("recoverable", False)],
        "task_log": task_log,
    }


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def git_meta() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return "?"
    return {
        "sha": run("git", "rev-parse", "--short", "HEAD"),
        "dirty": run("git", "status", "--porcelain"),
    }


async def main(n_turns: int, dry_run: bool) -> None:
    meta = git_meta()
    if meta["dirty"]:
        print("⚠️  工作区不干净，baseline 将无法从这个 sha 重建：")
        print(meta["dirty"])
        if input("仍然继续？(yes/N) ").strip().lower() != "yes":
            return

    turns = TURNS[:n_turns]
    print(f"\n{'='*66}\nD1 BASELINE  sha={meta['sha']}  轮数={len(turns)}"
          f"{'  [DRY RUN]' if dry_run else ''}\n{'='*66}\n")

    if dry_run:
        for t in turns:
            print(f"  轮 {t['turn']:>2}  {t['input']}")
        print("\n（dry-run：未调用 LLM）")
        return

    from harness.tracing import SQLiteTraceStorage, configure_storage
    from src.memory.wiring import configure_memory
    from src.session_runner import Session

    configure_storage(SQLiteTraceStorage(db_path=DB_PATH))
    # 记忆关掉：baseline 是对照基准，跨会话记忆注入会引入本轮
    # 不可控的变量。这与 evals 的 memory:false 保持同一口径。
    configure_memory(None)

    session_id = f"baseline-{uuid.uuid4().hex[:8]}"
    session = Session(session_id=session_id, auto_clarify_answer="四个人")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pre_harness_{meta['sha']}.json"
    record = {
        "kind": "pre_harness_baseline",
        "git_sha": meta["sha"],
        "session_id": session_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "memory": False,
        "note": "接入 harness 之前的运行时特征。expect_l1 为跑前锁定的人工预测。",
        "turns": [],
    }

    prev_max_eta = None
    prev_log_len = 0          # task_log 跨轮累积，记住上轮长度好切出本轮
    for spec in turns:
        print(f"\n{'─'*66}\n轮 {spec['turn']}  「{spec['input']}」")
        row: dict = {**spec, "ok": False}

        try:
            result = await session.say(spec["input"])
            obs = observe(result, prev_max_eta, prev_log_len)
            prev_log_len = len(result.task_log or [])
            row.update({
                "ok": True,
                "reply": result.reply,
                "root_trace_id": result.root_trace_id,
                "tokens": aggregate_tokens(result.root_trace_id),
                "observed": obs,
                # ⚠️ 人工标注列：跑完后逐轮填。第 9 轮这一格是
                #    P0-A / P1 分界的唯一依据，机器判不了。
                #    判据（跑前锁定）：回复里必须出现对先前约束的
                #    **显式指涉**；只说"好的，安排火锅"不算。
                "reply_mentions_conflict": None,
            })
            prev_max_eta = obs.get("max_eta_in_pool") or prev_max_eta

            tk = row["tokens"]
            print(f"  route={obs['route']}  adjust={obs['adjust_history_len']}  "
                  f"no_spicy={obs['no_spicy']}  budget_seen={obs['budget_seen']}  "
                  f"waypoints={obs['waypoints_len']}")
            print(f"  fact_ran={obs['fact_ran']}  searched_pois={obs['searched_pois']}  "
                  f"diet={obs['diet']}  avoid={obs['avoid']}")
            if not obs["fact_ran"]:
                print("  ⚠️ 本轮未经过 fact_node —— 辣度约束无执行点")
            print(f"  tokens={tk.get('total_tokens')}  calls={tk.get('llm_calls')}  "
                  f"pool={obs['poi_pool_size']}  max_eta={obs['max_eta_in_pool']}")
            if tk.get("by_agent"):
                parts = [f"{a}:{d['prompt_tokens'] + d['completion_tokens']}"
                         for a, d in sorted(tk["by_agent"].items())]
                print(f"  by_agent  {'  '.join(parts)}")
            if tk.get("estimated_calls"):
                print(f"  ⚠️ {tk['estimated_calls']} 次调用的 token 是估算的，"
                      f"报表需按 token_source 分组")
            if obs["fatal_errors"]:
                print(f"  ⚠️ fatal: {obs['fatal_errors']}")
            print(f"  回复：{(result.reply or '')[:120]}")

        except Exception as e:
            # 中途炸掉也要把前面几轮存下来——那些 token 是真花了的
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"  ❌ {row['error']}")

        record["turns"].append(row)
        # 每轮落盘，不等跑完。整段会话不可逆，进程挂掉不能丢数据。
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                            encoding="utf-8")

        if not row["ok"]:
            print("\n本轮失败，中止（状态已污染，后续轮次不可比）。")
            break

    record["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n{'='*66}\n已写入 {out_path}\n{'='*66}")
    print(f"{'轮':>3} {'route':<12} {'adj':>4} {'fact':>6} {'spicy':>6} "
          f"{'budget':>7} {'wp':>3} {'tokens':>8} {'calls':>6}")
    for r in record["turns"]:
        if not r.get("ok"):
            print(f"{r['turn']:>3}  ❌ {r.get('error','')[:50]}")
            continue
        o, t = r["observed"], r["tokens"]
        print(f"{r['turn']:>3} {str(o['route']):<12} {o['adjust_history_len']:>4} "
              f"{str(o['fact_ran']):>6} {str(o['no_spicy']):>6} "
              f"{str(o['budget_seen']):>7} "
              f"{o['waypoints_len']:>3} {t.get('total_tokens',0):>8} "
              f"{t.get('llm_calls',0):>6}")

    print("\n跑完之后要做的两件事：")
    print("  1. 逐轮填 reply_mentions_conflict（重点是第 9 轮）")
    print("  2. 对照 expect_l1 与 observed，把偏差写进 findings")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=len(TURNS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.turns, a.dry_run))