import sys
import os

# 确保路径正确
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.workflow import app


_DEFAULT_USER_INPUT = "今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。"
_MAX_CLARIFICATION_ROUNDS = 5


def _build_user_input(turns: list[str]) -> str:
    if not turns:
        return ""
    first_turn, *follow_ups = turns
    if not follow_ups:
        return first_turn
    extra_lines = [f"补充信息：{item}" for item in follow_ups]
    return "\n".join([first_turn, *extra_lines])


def main():
    print("=" * 60)
    print("Meituan Local Life Execution Agent - Hackathon Demo")
    print("=" * 60 + "\n")

    # 1. 收集初始需求
    user_input = input("请输入您的需求 (回车使用默认场景):\n> ")
    if not user_input.strip():
        user_input = _DEFAULT_USER_INPUT

    # MVP 运行时位置注入：当前 CLI 没有真实定位能力，先用稳定的默认区域，
    # 避免“离家近/附近”类需求在手动测试时频繁卡在位置澄清。
    runtime_origin_area = "area_central"
    conversation_turns = [user_input]
    clarification_round = 0

    while True:
        final_state = app.invoke(
            {
                "user_input": _build_user_input(conversation_turns),
                "runtime_origin_area": runtime_origin_area,
                "errors": [],
            }
        )

        if final_state.get("clarification_needed") is True:
            clarification_round += 1
            if clarification_round > _MAX_CLARIFICATION_ROUNDS:
                print("\n[系统提示] 澄清轮次过多，暂时无法继续规划，请重新描述一次需求。")
                break

            follow_up = final_state.get("follow_up_message") or final_state.get("llm_answer")
            if isinstance(follow_up, str) and follow_up.strip():
                print(f"\n[系统追问] {follow_up}")
            else:
                print("\n[系统追问] 还缺少继续规划所需的信息，请再补充一点。")

            extra_input = input("> ")
            while not extra_input.strip():
                extra_input = input("> ")
            conversation_turns.append(extra_input.strip())
            continue

        llm_answer = final_state.get("llm_answer")
        if isinstance(llm_answer, str) and llm_answer.strip():
            print(f"\n{llm_answer}")
            break

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
        break


if __name__ == "__main__":
    main()
