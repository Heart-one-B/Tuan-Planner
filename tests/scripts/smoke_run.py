# scripts/smoke_run.py
"""阶段 0-1 冒烟：跑通一次完整会话 + 验证 trace 树完整。

分四段，每段独立报告成败——一次改了 8 个文件，失败时必须能立刻
知道是哪一层断的，而不是看着一个 traceback 猜。

  A. 导入      不碰网络，最便宜，先排掉签名/路径错误
  B. 跑会话    routing → ... → presentation，出方案
  C. trace 树  parent_span 接线是否真的形成一棵树
  D. 成本      session_cost() 能否算出真实开销

C 是阶段 1 的硬验收：树散了，后续所有指标都是假的。

【B 失败必须中止】上一版有两个缺陷，都会制造误导：
  ① 汇总里 B 恒打 ✅（写死的），失败也显示通过
  ② B 失败后仍然跑 C/D，而此时 fact/planning/evaluation 一个都没
     执行，树上只有根节点、token 恒为 0——两个必然的 ❌ 看起来像
     "parent_span 没接上"，实际和接线毫无关系。
诊断工具产出假信号比不产出更糟，所以现在 B 一断就停。

用法：
    python scripts/smoke_run.py
    python scripts/smoke_run.py --input "..."
"""
from __future__ import annotations

import argparse
import asyncio
import json
import traceback
import uuid
from pathlib import Path

# 显式写上场景（朋友聚会）：intent 的 scenario 槽位无法从"四个人、
# 逛逛再吃饭"里推断，不写就会触发澄清追问，冒烟白白多烧几轮 LLM。
# 冒烟要验的是主链路能不能出方案，不是澄清逻辑——澄清值得单独测。
DEFAULT_INPUT = (
    "这周六下午两点到晚上十点，四个人朋友聚会，从四川大学江安校区出发，想吃火锅"
)

# 澄清追问时的统一回答：一次性覆盖 intent 可能问到的全部槽位
# （场景/人数/时间/出发地），避免答非所问导致反复追问。
CLARIFY_ANSWER = (
    "朋友聚会，四个人，这周六下午两点到晚上十点，"
    "从四川大学江安校区出发"
)
MAX_CLARIFY_TURNS = 3

DB_PATH = Path("data/trace.db")


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ── A. 导入 ────────────────────────────────────────────────────────────

def stage_a_imports() -> bool:
    """惰性导入，把 ImportError / 签名错误挡在跑真实调用之前。"""
    hr("A. 导入检查")
    mods = [
        ("agents.fact.cost", "agents/fact/cost.py"),
        ("agents.fact.tools", "agents/fact/tools.py"),
        ("agents.fact.agent", "agents/fact/agent.py"),
        ("agents.fact.schema", "agents/fact/schema.py"),
        ("agents.planning.agent", "agents/planning/agent.py"),
        ("agents.evaluation.agent", "agents/evaluation/agent.py"),
        ("agents.orchestrator.agent", "agents/orchestrator/agent.py"),
        ("src.graph.tracing", "src/graph/tracing.py"),
        ("src.graph.workflow", "src/graph/workflow.py"),
    ]
    ok = True
    for mod, path in mods:
        try:
            __import__(mod)
            print(f"  ✅ {path}")
        except Exception as e:
            ok = False
            print(f"  ❌ {path}\n     {type(e).__name__}: {e}")

    # 签名检查：这次改动的核心就是把 trace_id 换成 parent_span，
    # 漏改一个就是运行时 TypeError，不如在这里一次性照出来。
    try:
        import inspect
        from agents.fact.agent import FactAgent
        from agents.planning.agent import PlanningAgent
        from agents.evaluation.agent import EvaluationAgent
        from agents.orchestrator.agent import OrchestratorAgent

        for cls in (FactAgent, PlanningAgent, EvaluationAgent, OrchestratorAgent):
            params = inspect.signature(cls.run).parameters
            if "parent_span" not in params:
                ok = False
                print(f"  ❌ {cls.__name__}.run() 没有 parent_span 参数")
            elif "trace_id" in params:
                ok = False
                print(f"  ❌ {cls.__name__}.run() 仍然保留了 trace_id 参数")
            else:
                print(f"  ✅ {cls.__name__}.run(parent_span=...)")
    except Exception as e:
        ok = False
        print(f"  ❌ 签名检查失败: {e}")

    # POIItem 必须声明 cost，否则 _synthesize 传的 cost 会被
    # pydantic 静默丢弃（默认忽略未声明字段），表现是"所有餐厅
    # 都没有价格"，不报错。
    try:
        from agents.fact.schema import POIItem
        if "cost" in POIItem.model_fields:
            print("  ✅ POIItem.cost 已声明")
        else:
            ok = False
            print("  ❌ POIItem 缺少 cost 字段 —— cost 会被 pydantic 静默丢弃")
    except Exception as e:
        ok = False
        print(f"  ❌ POIItem 检查失败: {e}")

    return ok


# ── B. 跑一次会话 ──────────────────────────────────────────────────────

async def stage_b_run(user_input: str) -> tuple[dict | None, str | None]:
    """返回 (最终 state, root_trace_id)；跑不出方案时 state 为 None。

    用 build_workflow()（不带 checkpointer）而不是模块级的 workflow
    单例——冒烟要的是一次干净的、与应用运行时状态无关的执行。

    澄清追问最多陪 MAX_CLARIFY_TURNS 轮：clarification_node 在达到
    自己的轮次上限后会强制填默认值放行，所以有限轮内一定能收敛。
    超过上限仍在追问，说明 intent 对这段输入的槽位判定有问题，
    那是一个需要单独查的 bug，不该让冒烟无限烧钱陪聊。
    """
    hr("B. 跑一次完整会话")
    from src.graph.workflow import build_workflow
    from src.graph.tracing import ROOT_TRACE_ID_KEY, begin_request_span

    graph = build_workflow()
    session_id = f"smoke-{uuid.uuid4().hex[:8]}"
    span = begin_request_span(user_input)
    print(f"  session_id   = {session_id}")
    print(f"  root_trace_id= {span.trace_id}")
    print(f"  输入          = {user_input}")

    state: dict = {
        "user_input": user_input,
        "session_id": session_id,
        ROOT_TRACE_ID_KEY: span.trace_id,
    }

    try:
        result = await graph.ainvoke(state)

        for turn in range(MAX_CLARIFY_TURNS):
            pending = result.get("pending_clarification") or ""
            if not pending:
                break
            print(f"\n  ⏸  澄清追问 #{turn + 1}：{pending}")
            print(f"     回答：{CLARIFY_ANSWER}")
            result = await graph.ainvoke({
                **result,
                "user_input": CLARIFY_ANSWER,
                ROOT_TRACE_ID_KEY: span.trace_id,
            })

        display = result.get("display_text") or result.get("final_message") or ""
        span.end(display[:200], status="success")

    except Exception as e:
        span.end(str(e), status="error")
        print(f"\n  ❌ 会话抛异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, span.trace_id

    # ── 执行轨迹：哪个节点跑过、有没有错 ──
    print("\n  --- task_log ---")
    for line in result.get("task_log") or []:
        print(f"    {line}")

    errors = result.get("errors") or []
    if errors:
        print("\n  --- errors ---")
        for e in errors:
            print(f"    ❌ [{e.get('node')}] {e.get('error')}")

    if result.get("pending_clarification"):
        print(f"\n  ⚠️  {MAX_CLARIFY_TURNS} 轮之后仍在追问 —— intent 的槽位判定"
              f"可能有问题，值得单独查")

    # ── 候选池抽样：确认 cost 真的进来了 ──
    fact_data = ((result.get("agent_outputs") or {}).get("fact") or {}).get("data") or {}
    restaurants = fact_data.get("restaurants") or []
    if restaurants:
        with_cost = [r for r in restaurants if r.get("cost") is not None]
        print(f"\n  --- 餐厅候选池 {len(restaurants)} 家，"
              f"有价格 {len(with_cost)} 家 ({len(with_cost) / len(restaurants):.0%}) ---")
        for r in restaurants[:5]:
            cost = f"{r['cost']:.0f}" if r.get("cost") is not None else "—"
            print(f"    {cost:>6}  {r.get('name')}")
    else:
        print("\n  ⚠️  餐厅候选池为空")

    print(f"\n  --- 出发地解析 ---")
    print(f"    city={fact_data.get('origin_city') or '(空)'}  "
          f"coord={fact_data.get('origin_coordinates') or '(空)'}")

    display = result.get("display_text") or result.get("final_message") or ""
    print("\n  --- display_text ---")
    print("    " + (display[:600].replace("\n", "\n    ") if display else "(空)"))

    fatal = [e for e in errors if not e.get("recoverable", False)]
    ok = bool(display) and not fatal
    print(f"\n  {'✅ 出方案' if ok else '❌ 未出方案或有不可恢复错误'}")
    return (result if ok else None), span.trace_id


# ── C. trace 树 ────────────────────────────────────────────────────────

def stage_c_trace(root_trace_id: str) -> bool:
    """阶段 1 的硬验收。

    树散了不会有任何报错——只会让 session_cost() 系统性低估，
    而且低估的恰恰是子 Agent 那部分（最容易被忽略、也最花钱的部分）。
    所以必须显式断言，不能靠"看起来跑通了"。
    """
    hr("C. trace 树完整性（阶段 1 硬验收）")
    from harness.tracing import SQLiteTraceStorage

    storage = SQLiteTraceStorage(db_path=DB_PATH)
    tree = storage.get_trace_tree(root_trace_id)
    if tree is None:
        print(f"  ❌ 找不到 root trace {root_trace_id}")
        return False

    def walk(node, depth=0):
        t = node.trace
        print(f"    {'  ' * depth}├─ {t.trace_id[:8]}  {t.session_id:<14} "
              f"llm={t.llm_call_count:<3} tool={t.tool_call_count:<3} "
              f"{t.total_duration_ms}ms  [{t.status}]")
        for c in node.children:
            walk(c, depth + 1)

    print("  trace 树：")
    walk(tree)

    flat = tree.flatten()
    print(f"\n  节点数={len(flat)}  深度={tree.depth()}")

    ok = True
    if len(flat) < 4:
        ok = False
        print(f"  ❌ 节点数 {len(flat)} < 4 —— 树是散的，parent_span 没接上")
        print("     正常一次规划至少有：plan-request / fact / planning / evaluation")
    else:
        print(f"  ✅ 节点数 {len(flat)} ≥ 4")

    # FactAgent 曾经从不 start_trace，导致它的 LLM 调用被
    # record_llm_call 开头的 `if trace is None: return` 静默丢弃。
    # 这里显式验一次那个 bug 确实修好了。
    fact_nodes = [n for n in tree.walk() if n.trace.session_id == "fact"]
    if not fact_nodes:
        ok = False
        print("  ❌ 树里没有 fact 节点 —— FactAgent 的 span 没挂上")
    elif all(n.trace.llm_call_count == 0 for n in fact_nodes):
        ok = False
        print("  ❌ fact 节点的 llm_call_count 全为 0 —— LLM 调用仍在被静默丢弃")
    else:
        total = sum(n.trace.llm_call_count for n in fact_nodes)
        print(f"  ✅ fact 节点记录到 {total} 次 LLM 调用（曾经是 0）")

    return ok


# ── D. 成本 ────────────────────────────────────────────────────────────

def stage_d_cost(root_trace_id: str) -> bool:
    hr("D. 一次请求的真实成本")
    from harness.tracing import SQLiteTraceStorage
    from harness.tracing.analytics import request_cost

    storage = SQLiteTraceStorage(db_path=DB_PATH)
    cost = request_cost(storage, root_trace_id)
    if not cost:
        print("  ❌ request_cost 返回空")
        return False

    print(json.dumps(cost, ensure_ascii=False, indent=2))
    if cost["total_tokens"] <= 0:
        print("\n  ⚠️  total_tokens=0 —— usage 没被记录，"
              "后续 token 相关的校准数据会是空的")
        return False
    print(f"\n  ✅ 本次请求合计 {cost['total_tokens']} tokens，"
          f"跨 {cost['trace_count']} 条 trace")
    return True


# ── main ──────────────────────────────────────────────────────────────

async def main(user_input: str) -> int:
    from harness.tracing import SQLiteTraceStorage, configure_storage
    configure_storage(SQLiteTraceStorage(db_path=DB_PATH))

    if not stage_a_imports():
        print("\n导入或签名有问题，先修这一层，下面全会连带失败。")
        return 1

    result, root_trace_id = await stage_b_run(user_input)
    if result is None:
        hr("汇总")
        print("  A 导入      ✅")
        print("  B 跑通会话  ❌")
        print("  C/D 跳过 —— 主链路没跑完，树上只有根节点、token 恒为 0，")
        print("             此时 C/D 的失败是必然的连带结果，不含任何诊断信息。")
        print("\n  先看 B 段的 errors。")
        return 1

    ok_c = stage_c_trace(root_trace_id)
    ok_d = stage_d_cost(root_trace_id)

    hr("汇总")
    print("  A 导入      ✅")
    print("  B 跑通会话  ✅")
    print(f"  C trace 树  {'✅' if ok_c else '❌'}")
    print(f"  D 成本统计  {'✅' if ok_d else '❌'}")
    if ok_c and ok_d:
        print("\n  阶段 0-1 完成，可以进阶段 2（接记忆）。")
        return 0
    print("\n  会话能跑通，但测量仪器还不准——阶段 3 的指标会是假的，先修 C/D。")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.input)))