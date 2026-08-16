import asyncio
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from langgraph.checkpoint.memory import MemorySaver
from src.graph.workflow import build_workflow
from harness.mcp.registry import close_all


def _print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


async def main():
    app = build_workflow(MemorySaver())
    thread = {"configurable": {"thread_id": "test-001"}}

    user_input = "周六下午两点到晚上九点，我和三个朋友在四川大学江安校区附近聚会，想玩点桌游或者剧本杀，晚饭吃火锅，中途想喝杯奶茶。"

    try:
        _print_section("[输入]")
        print(user_input)

        start = time.time()
        await app.ainvoke({"user_input": user_input}, config=thread)
        elapsed = time.time() - start

        state = app.get_state(thread).values

        _print_section(f"[task_log] (总耗时 {elapsed:.1f}s)")
        for line in state.get("task_log") or []:
            print(f"  {line}")

        errors = state.get("errors") or []
        _print_section("[errors]")
        if errors:
            for e in errors:
                print(f"  {e}")
        else:
            print("  无")

        agent_outputs = state.get("agent_outputs") or {}
        fact_output = agent_outputs.get("fact") or {}
        _print_section("[fact result]")
        print(f"status: {fact_output.get('status')}")
        print(f"summary: {fact_output.get('summary')}")
        fact_data = fact_output.get("data") or {}
        print(f"activities: {len(fact_data.get('activities') or [])}个")
        print(f"restaurants: {len(fact_data.get('restaurants') or [])}个")
        print(f"waypoints: {len(fact_data.get('waypoints') or [])}个")
        print(json.dumps(fact_data, ensure_ascii=False, indent=2))

        planning_output = agent_outputs.get("planning") or {}
        _print_section("[planning result]")
        print(f"status: {planning_output.get('status')}")
        print(f"summary: {planning_output.get('summary')}")
        candidates = (planning_output.get("data") or {}).get("candidates") or []
        for c in candidates:
            print(f"  {c.get('id')} - {c.get('title')}")

        eval_output = agent_outputs.get("evaluation") or {}
        _print_section("[evaluation result]")
        print(f"status: {eval_output.get('status')}")
        print(f"summary: {eval_output.get('summary')}")
        print(json.dumps(eval_output.get("data") or {}, ensure_ascii=False, indent=2))

        _print_section("[display_text]")
        print(state.get("display_text") or "（无）")

        pending = state.get("pending_clarification")
        if pending:
            _print_section("[触发了追问]")
            print(pending)

    finally:
        # 显式关闭所有MCP连接，避免进程退出时的跨Task cancel scope错误
        await close_all()


if __name__ == "__main__":
    asyncio.run(main())