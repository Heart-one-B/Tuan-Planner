import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph.workflow import build_workflow

_DEFAULT_USER_INPUT = "今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。"

# 多轮对话复用同一个 app 和 thread_id，state 在 checkpointer 里持久化
_THREAD_ID = "cli-session"


def run_turn(app, user_input: str, thread: dict) -> None:
    """
    单次对话轮次：注入 user_input，恢复图执行，处理中断。
    """
    # 注入新的用户输入，图从 feedback_router（入口）开始执行
    app.invoke(
        {"user_input": user_input},
        config=thread,
    )

    # clarification / confirmation 中断恢复循环
    while True:
        snapshot = app.get_state(thread)
        if not snapshot.next:
            break

        # clarification 中断：等待用户回答追问
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

        # confirmation 中断：等待用户确认方案
        confirmation_prompt = snapshot.values.get("pending_confirmation", {})
        if confirmation_prompt:
            prompt_text = confirmation_prompt.get("message", "[系统提示] 确定按照此方案执行一键下单吗？(y/n): ")
            print(f"\n{prompt_text}", end="")
        else:
            print("\n[系统提示] 确定按照此方案执行一键下单吗？(y/n): ", end="")

        user_reply = input().strip()
        while not user_reply:
            user_reply = input().strip()
        app.invoke(
            Command(resume=None, update={"user_confirmation": user_reply}),
            config=thread,
        )

    # 输出本轮结果
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

    if final_state.get("errors"):
        print("\n[错误信息]")
        for error in final_state["errors"]:
            print(f"  - {error}")


def main():
    print("=" * 60)
    print("Meituan Local Life Execution Agent - Hackathon Demo")
    print("输入 'q' 或 'exit' 退出，直接回车使用默认场景")
    print("=" * 60 + "\n")

    # 多轮对话复用同一个 app 和 thread，state 在 checkpointer 里持久化
    app = build_workflow(MemorySaver())
    thread = {"configurable": {"thread_id": _THREAD_ID}}

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
            run_turn(app, user_input, thread)
        except KeyboardInterrupt:
            print("\n\n（当前规划已中断）")
            continue
        except Exception as e:
            print(f"\n[ERROR] 出现异常：{e}")
            continue

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()