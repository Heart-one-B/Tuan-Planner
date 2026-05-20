import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.workflow import app


_DEFAULT_USER_INPUT = "今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。"


def main():
    print("=" * 60)
    print("Meituan Local Life Execution Agent - Hackathon Demo")
    print("=" * 60 + "\n")

    user_input = input("请输入您的需求(回车使用默认场景):\n> ")
    if not user_input.strip():
        user_input = _DEFAULT_USER_INPUT

    runtime_origin_area = "area_central"
    final_state = app.invoke(
        {
            "user_input": user_input,
            "runtime_origin_area": runtime_origin_area,
            "conversation_turns": [user_input],
            "clarification_round": 0,
            "errors": [],
        }
    )

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
        print("\n好的，您可以告诉我需要调整的地方，我重新为您规划。")

    if final_state.get("errors"):
        print("\n[错误信息]")
        for error in final_state["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
