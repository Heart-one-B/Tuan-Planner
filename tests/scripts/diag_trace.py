# scripts/diag_trace.py
"""诊断：trace 树为什么是散的。

两个互斥的可能，处理方式完全不同：

  A. root_trace_id 没进 AgentState —— LangGraph 用 TypedDict 定义
     state schema，未声明的键会被丢弃。ainvoke 时传了也没用，
     节点里 state.get("root_trace_id") 恒为 None，于是每个子 Agent
     各开各的根 trace。修法：AgentState 加一行。

  B. root_trace_id 进去了，但某个节点没调 parent_span_of。
     修法：找出那个节点。

下面先查 schema 声明（纯本地），再查数据库里最近的孤儿 trace。
孤儿 trace 的存在本身就证明子 Agent 的 span 确实创建了、
只是没挂上父节点——排除掉"span 压根没开"这种可能。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("data/trace.db")


def check_schema() -> bool:
    print("=" * 70)
    print("1. AgentState 是否声明了 root_trace_id")
    print("=" * 70)
    try:
        from src.graph.state import AgentState
        keys = set(getattr(AgentState, "__annotations__", {}))
    except Exception as e:
        print(f"  ❌ 导入 AgentState 失败: {e}")
        return False

    if "root_trace_id" in keys:
        print("  ✅ 已声明")
        return True

    print("  ❌ 未声明 —— 这就是树散掉的原因")
    print("     LangGraph 会丢弃 state schema 里没有的键，")
    print("     ainvoke 传进去的 root_trace_id 到不了节点。")
    print("\n     修法：src/graph/state.py 的 AgentState 里加一行")
    print("         root_trace_id: str")
    return False


def check_orphans() -> None:
    print("\n" + "=" * 70)
    print("2. 最近 20 条 trace 的父子关系")
    print("=" * 70)
    if not DB_PATH.is_file():
        print(f"  ❌ 找不到 {DB_PATH}（要在项目根目录跑这个脚本）")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT trace_id, session_id, parent_trace_id, llm_call_count, "
        "tool_call_count, status, created_at "
        "FROM traces ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    conn.close()

    if not rows:
        print("  ❌ 表是空的")
        return

    orphans = 0
    for r in rows:
        parent = r["parent_trace_id"]
        mark = "🔗" if parent else "🌱"
        if not parent and r["session_id"] != "plan-request":
            orphans += 1
            mark = "⚠️"
        print(f"  {mark} {r['trace_id'][:8]}  {r['session_id']:<14} "
              f"parent={(parent[:8] if parent else 'NULL'):<10} "
              f"llm={r['llm_call_count']:<3} tool={r['tool_call_count']:<3} "
              f"[{r['status']}]")

    print(f"\n  🌱=根 trace  🔗=有父节点  ⚠️=孤儿（本该有父节点却没有）")
    if orphans:
        print(f"\n  发现 {orphans} 条孤儿 trace —— 子 Agent 的 span 确实创建了，")
        print(f"  只是 parent_trace_id 是 NULL，所以没挂进树里。")
        print(f"  这排除了'span 压根没开'，问题在 parent_span 的传递链上。")
    else:
        print("\n  没有孤儿。如果树仍然是散的，检查 root span 和子 span 的")
        print("  trace_id 是不是来自两次不同的 begin_request_span 调用。")


if __name__ == "__main__":
    ok = check_schema()
    check_orphans()
    if not ok:
        print("\n先加 AgentState 的那一行，再重跑 smoke_run.py。")