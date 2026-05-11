from langchain_core.messages import HumanMessage, SystemMessage
from src.model.factory import chat_model


class PresentationAgent:
    def generate_plan_display(self, plan: dict, intent: dict) -> str:
        """
        根据规划结果，生成带有表格、详细说明和满意度询问的文本
        """
        print("📱 [Presentation Agent] 正在排版最终方案...")

        # 整理干预信息，让模型知道“救场”的背景
        intervention_context = " ".join(plan.get("exceptions_handled", []))

        prompt = f"""
        你是一个专业的美团生活助理。请根据以下规划数据，为用户生成一份精美的下午行程建议。

        用户背景: {intent.get('scenario')} 场景 (备注: {intent.get('diet_preference')}需求)
        规划数据:
        - 玩乐活动: {plan['activities'][0]['name']} (类型: {plan['activities'][0]['type']})
        - 晚餐餐厅: {plan['restaurant']['name']} (备注: 已匹配{intent.get('diet_preference')}需求)
        - 异常救场信息: {intervention_context}

        请严格按以下三个部分进行回复：

        第一部分：行程时间表
        以 Markdown 表格形式展示，包含三列：【时间段】、【活动】、【备注】。
        注意：下午行程从14:00开始，晚餐安排在18:00-19:30。
        例子：18:00-19:30 | 晚餐 | {plan['restaurant']['name']} (已预留4人位)

        第二部分：方案亮点说明
        分点说明为什么这么安排。必须提到：
        1. 针对天气的调整（如果是室内活动，解释原因）。
        2. 针对减脂/亲子需求的适配。
        3. 针对餐厅排队情况的优化（提到已避开满座餐厅）。

        第三部分：结束语
        询问用户是否满意该安排。例如：“您对这个安排满意吗？如果没问题，我可以为您一键完成预订。”
        """

        try:
            response = chat_model.invoke([
                SystemMessage(content="你是一个贴心的本地生活规划专家，擅长使用Markdown表格。"),
                HumanMessage(content=prompt)
            ])
            return response.content
        except Exception as e:
            return f"方案生成失败，请稍后重试。错误: {e}"