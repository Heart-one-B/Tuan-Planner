import sys
import os

# 确保路径正确
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.intent_agent import IntentAgent
from src.agent.planning_agent import PlanningAgent
from src.agent.execution_agent import ExecutionAgent
from src.agent.presentation_agent import PresentationAgent


def main():
    print("=" * 60)
    print("🐿️  美团本地生活执行 Agent - Hackathon Demo")
    print("=" * 60 + "\n")

    # 1. 意图解析
    user_input = input("请输入您的需求 (回车使用默认场景):\n> ")
    if not user_input.strip():
        user_input = "今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。"

    intent_agent = IntentAgent()
    intent = intent_agent.parse(user_input)

    # 2. 规划方案 (包含工具调用和救场逻辑)
    planning_agent = PlanningAgent()
    plan = planning_agent.plan(intent)

    # 3. 展示方案 (生成表格 + 解释 + 询问)
    pres_agent = PresentationAgent()
    display_text = pres_agent.generate_plan_display(plan, intent)

    print("\n" + "=" * 20 + " 方案详情 " + "=" * 20)
    print(display_text)
    print("=" * 50)

    # 4. 用户交互与确认
    confirm = input("\n[系统提示] 确定按照此方案执行一键下单吗？(y/n): ")

    if confirm.lower() == 'y':
        # 5. 执行下单逻辑
        exe_agent = ExecutionAgent()
        exe_agent.execute(plan)

        print("\n🎉 搞定了！所有订单已处理完成。")
        print("📱 详细凭证已发送至您的手机（模拟），您可以随时出发！")
    else:
        print("\n好的，您可以告诉我需要调整的地方，我重新为您规划。")


if __name__ == "__main__":
    main()