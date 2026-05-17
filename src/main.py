import sys
import os

# 确保路径正确
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.workflow import app


def main():
    print("=" * 60)
    print("Meituan Local Life Execution Agent - Hackathon Demo")
    print("=" * 60 + "\n")

    # 1. 意图解析
    user_input = input("请输入您的需求 (回车使用默认场景):\n> ")
    if not user_input.strip():
        user_input = "今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。"

    # MVP 运行时位置注入：当前 CLI 没有真实定位能力，先用稳定的默认区域，
    # 避免“离家近/附近”类需求在手动测试时频繁卡在位置澄清。
    runtime_origin_area = "area_central"

    final_state = app.invoke(
        {
            "user_input": user_input,
            "runtime_origin_area": runtime_origin_area,
            "errors": [],
        }
    )
    if final_state.get("user_confirmed"):
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
