import asyncio
import json
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from langgraph.checkpoint.memory import MemorySaver

from src.graph.workflow import build_workflow
from harness.mcp.registry import close_all

_DEFAULT_USER_INPUT = "今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。"
_THREAD_ID = "cli-session"


def _print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _dump_agent_outputs(state: dict):
    """完整打印每个Agent的输出，排错用。"""
    agent_outputs = state.get("agent_outputs") or {}

    for name in ("fact", "planning", "evaluation"):
        output = agent_outputs.get(name) or {}
        _print_section(f"[{name} result]")
        print(f"status: {output.get('status')}")
        print(f"summary: {output.get('summary')}")
        data = output.get("data") or {}
        if data:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print("(空)")

    _print_section("[task_log]")
    for line in state.get("task_log") or []:
        print(f"  {line}")

    errors = state.get("errors") or []
    _print_section("[errors]")
    if errors:
        for e in errors:
            print(f"  {e}")
    else:
        print("  无")


async def run_turn(app, user_input: str, thread: dict) -> None:
    await app.ainvoke({"user_input": user_input}, config=thread)

    state = app.get_state(thread).values
    _dump_agent_outputs(state)

    while state.get("pending_clarification"):
        print(f"\n{state['pending_clarification']}")
        user_reply = input("> ").strip()
        while not user_reply:
            user_reply = input("> ").strip()

        await app.ainvoke({"user_input": user_reply}, config=thread)
        state = app.get_state(thread).values
        _dump_agent_outputs(state)

    _print_section("[display_text]")
    display_text = state.get("display_text")
    if isinstance(display_text, str) and display_text.strip():
        print(display_text)
        return

    final_message = state.get("final_message")
    if isinstance(final_message, str) and final_message.strip():
        print(final_message)
        return

    print("（没有生成可展示的内容）")


async def main():
    print("=" * 60)
    print("AI Hackathon - 本地行程规划助手 CLI Demo（完整日志模式）")
    print("输入 'q' 或 'exit' 退出，直接回车使用默认场景")
    print("=" * 60 + "\n")

    app = build_workflow(MemorySaver())
    thread = {"configurable": {"thread_id": _THREAD_ID}}

    try:
        while True:
            try:
                user_input = input("你：\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再见！")
                break

            if user_input.lower() in ("q", "exit", "quit", "退出"):
                print("\n再见！")
                break

            if not user_input:
                user_input = _DEFAULT_USER_INPUT
                print(f"（使用默认场景：{user_input}）")

            print()
            try:
                await run_turn(app, user_input, thread)
            except KeyboardInterrupt:
                print("\n\n（当前规划已中断）")
                continue
            except Exception as e:
                print(f"\n[ERROR] 出现异常：{e}")
                import traceback
                traceback.print_exc()
                continue

            print("\n" + "-" * 60 + "\n")
    finally:
        await close_all()


if __name__ == "__main__":
    asyncio.run(main())