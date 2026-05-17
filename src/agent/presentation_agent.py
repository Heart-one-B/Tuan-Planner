from langchain_core.messages import HumanMessage, SystemMessage
from src.model.factory import chat_model


class PresentationAgent:
    def generate_plan_display(self, plan: dict, intent: dict) -> str:
        """
        根据规划结果，生成带有表格、详细说明和满意度询问的文本
        """
        print("[Presentation Agent] 正在排版最终方案...")

        # 整理干预信息，让模型知道“救场”的背景
        intervention_context = " ".join(plan.get("exceptions_handled", []))
        activities = plan.get("activities") or [{}]
        activity = activities[0] if isinstance(activities, list) and activities else {}
        restaurant = plan.get("restaurant") or {}
        activity_name = activity.get("name", "待确认活动")
        activity_type = activity.get("type", "未知")
        restaurant_name = restaurant.get("name", "待确认餐厅")
        diet_pref = intent.get("diet_preference")
        if isinstance(diet_pref, list):
            diet_pref_text = "、".join(str(item) for item in diet_pref) if diet_pref else "无"
        else:
            diet_pref_text = diet_pref or "无"

        prompt = f"""
        你是一个专业的美团生活助理。请根据以下规划数据，为用户生成一份精美的下午行程建议。

        用户背景: {intent.get('scenario')} 场景 (备注: {diet_pref_text}需求)
        规划数据:
        - 玩乐活动: {activity_name} (类型: {activity_type})
        - 晚餐餐厅: {restaurant_name} (备注: 已匹配{diet_pref_text}需求)
        - 异常救场信息: {intervention_context}

        请严格按以下三个部分进行回复：

        第一部分：行程时间表
        以 Markdown 表格形式展示，包含三列：【时间段】、【活动】、【备注】。
        注意：下午行程从14:00开始，晚餐安排在18:00-19:30。
        例子：18:00-19:30 | 晚餐 | {restaurant_name} (待确认后执行订座)

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
            return self._generate_fallback_display(plan, intent, str(e))

    def _generate_fallback_display(self, plan: dict, intent: dict, error: str) -> str:
        activities = plan.get("activities") or [{}]
        activity = activities[0]
        restaurant = plan.get("restaurant") or {}
        exceptions = plan.get("exceptions_handled") or []
        exception_text = "\n".join(f"- {item}" for item in exceptions) or "- 暂无异常，当前方案可执行。"

        return f"""# 下午行程建议（{intent.get('scenario', 'family')} 场景）

## 第一部分：行程时间表

| 时间段 | 活动 | 备注 |
| --- | --- | --- |
| 14:00-17:00 | {activity.get('name', '待确认活动')} | 类型：{activity.get('type', '未知')} |
| 17:00-18:00 | 休息/转场 | 预留交通和休息时间 |
| 18:00-19:30 | {restaurant.get('name', '待确认餐厅')} | 已按当前约束匹配 |

## 第二部分：方案亮点说明

{exception_text}

- 已结合用户场景、天气风险和餐厅可用性生成方案。
- 模型文案生成不可用，当前使用规则化 fallback 展示，不影响 Mock 规划与执行。

## 第三部分：结束语

您对这个安排满意吗？如果没问题，我可以为您一键完成预订。

> fallback reason: {error}
"""
