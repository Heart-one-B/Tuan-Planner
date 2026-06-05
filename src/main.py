import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph.workflow import build_workflow

_DEFAULT_USER_INPUT = "今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。"


def run_once(app, user_input: str, thread_id: str) -> None:
    """单轮规划：从用户输入到最终输出，含 clarification 中断恢复。"""
    thread = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_input": user_input,
        "runtime_origin_area": "",
        "conversation_turns": [user_input],
        "clarification_round": 0,
        "errors": [],
    }

    app.invoke(initial_state, config=thread)

    # clarification / confirmation 中断恢复循环
    while True:
        snapshot = app.get_state(thread)
        if not snapshot.next:
            break

        pending = snapshot.values.get("pending_clarification", "")
        if pending:
            print(f"\n{pending}")
            user_reply = input("> ").strip()
            while not user_reply:
                user_reply = input("> ").strip()
            app.invoke(
                Command(resume=None, update={"user_reply": user_reply}),
                config=thread,
            )
            continue

        # confirmation 中断
        confirm_prompt = snapshot.values.get("confirmation_prompt", "")
        if confirm_prompt:
            print(f"\n{confirm_prompt}")
        user_reply = input("> ").strip()
        while not user_reply:
            user_reply = input("> ").strip()
        app.invoke(
            Command(resume=None, update={"user_confirmation": user_reply}),
            config=thread,
        )

    # 输出最终结果
    final_state = app.get_state(thread).values

    llm_answer = final_state.get("llm_answer")
    if isinstance(llm_answer, str) and llm_answer.strip():
        print(f"\n{llm_answer}")
        return

    final_message = final_state.get("final_message")
    if isinstance(final_message, str) and final_message.strip():
        print(f"\n{final_message}")
    elif final_state.get("user_confirmed"):
        print("\n[DONE] 搞定了！所有订单已处理完成。")
        print("[MOCK] 详细凭证已发送至您的手机（模拟），您可以随时出发！")
    else:
        print("\n好的，您可以告诉我需要调整的地方。")

    if final_state.get("errors"):
        print("\n[错误信息]")
        for error in final_state["errors"]:
            print(f"  - {error}")


def main():
    print("=" * 60)
    print("Meituan Local Life Execution Agent - Hackathon Demo")
    print("输入 'q' 或 'exit' 退出，直接回车使用默认场景")
    print("=" * 60 + "\n")

    session_count = 0

    while True:
        # 每轮对话用独立的 thread_id 和 MemorySaver，互不干扰
        session_count += 1
        thread_id = f"cli-session-{session_count}"
        app = build_workflow(MemorySaver())

        try:
            user_input = input("请输入您的需求:\n> ").strip()
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
            run_once(app, user_input, thread_id)
        except KeyboardInterrupt:
            print("\n\n（当前规划已中断）")
            continue
        except Exception as e:
            print(f"\n[ERROR] 出现异常：{e}")
            continue

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()