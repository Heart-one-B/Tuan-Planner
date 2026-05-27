from langchain_core.messages import HumanMessage, SystemMessage

from src.model.factory import get_chat_model


class PresentationAgent:
    @staticmethod
    def _format_timeline(timeline, date_label: str = "") -> str:
        if not isinstance(timeline, list) or not timeline:
            return "待确认时间：先活动，再用餐"
        lines = ["| 时间 | 地点 | 类型 |", "| :--- | :--- | :--- |"]
        last_time = ""
        for item in timeline:
            if not isinstance(item, dict):
                continue
            time_text = item.get("time") or "待确认"
            place_text = item.get("item") or "待确认"
            type_text = item.get("type") or "未知"
            if type_text == "departure" and date_label != "今天":
                continue
            if time_text == last_time and type_text == "restaurant":
                try:
                    hour, minute = [int(x) for x in time_text.split(":", 1)]
                    minute += 1
                    if minute >= 60:
                        hour += 1
                        minute -= 60
                    time_text = f"{hour:02d}:{minute:02d}"
                except Exception:
                    pass
            lines.append(f"| {time_text} | {place_text} | {type_text} |")
            last_time = time_text
        return "\n".join(lines)

    def generate_plan_display(self, plan: dict, intent: dict) -> str:
        print("[Presentation Agent] 正在排版最终方案...")

        intervention_context = " ".join(plan.get("exceptions_handled", []))
        activities = plan.get("activities") or [{}]
        activity = activities[0] if isinstance(activities, list) and activities else {}
        restaurant = plan.get("restaurant") or {}
        activity_name = activity.get("name", "待确认活动")
        activity_type = activity.get("type", "未知")
        restaurant_name = restaurant.get("name", "待确认餐厅")
        schedule_result = plan.get("schedule_timing_result") if isinstance(plan.get("schedule_timing_result"), dict) else {}
        date_label = schedule_result.get("normalized_date_label") if isinstance(schedule_result.get("normalized_date_label"), str) else ""
        if not date_label:
            time_info = intent.get("time") if isinstance(intent.get("time"), dict) else {}
            date_label = time_info.get("date_label") if isinstance(time_info.get("date_label"), str) else ""
        timeline = self._format_timeline(plan.get("timeline"), date_label)
        time_phrase = plan.get("time_phrase") or "待确认时间"
        final_score = plan.get("final_score")
        selected_candidate_id = plan.get("selected_candidate_id")

        diet_pref = intent.get("diet_preference")
        if isinstance(diet_pref, list):
            diet_pref_text = "、".join(str(item) for item in diet_pref) if diet_pref else "无"
        else:
            diet_pref_text = diet_pref or "无"

        if isinstance(schedule_result, dict) and isinstance(schedule_result.get("timeline"), list) and schedule_result.get("timeline"):
            timeline_text = self._format_timeline(schedule_result.get("timeline"), date_label)
            return self._generate_direct_display(
                intent=intent,
                timeline_text=timeline_text,
                activity_name=activity_name,
                activity_type=activity_type,
                restaurant_name=restaurant_name,
                final_score=final_score,
                selected_candidate_id=selected_candidate_id,
                intervention_context=intervention_context,
                time_phrase=time_phrase,
            )

        prompt = f"""
        你是一个专业的本地生活规划助手。
        请根据以下最终计划数据，为用户生成一份可确认但不夸大执行状态的方案说明。
        这是确认前展示，不是执行完成通知。不要虚构已经预订成功。

        用户背景: {intent.get('scenario')} 场景 (备注: {diet_pref_text}需求)
        最终计划ID: {selected_candidate_id}
        最终评分: {final_score}
        规划数据:
        - 时间安排: {timeline}
        - 活动: {activity_name} (类型: {activity_type})
        - 餐厅: {restaurant_name}
        - 自动调整信息: {intervention_context or "无"}

        请严格按以下三个部分输出:
        第一部分：行程安排
        用简洁列表或表格展示，不要擅自补具体小时分钟。
        只能使用已给出的时间表达，例如"{time_phrase}" 或"{timeline}"。
        第二部分：方案说明
        说明为什么这样安排，重点提到:
        1. 时间窗口匹配
        2. 饮食/亲子适配
        3. 自动避让排队、天气或拥挤度风险
        第三部分：确认提示
        明确告诉用户：当前只是方案展示，确认后才会执行预约或下单。
        """

        try:
            response = get_chat_model().invoke(
                [
                    SystemMessage(content="你是一个谨慎的本地生活规划助手，确认前不要写成已经执行完成。"),
                    HumanMessage(content=prompt),
                ]
            )
            return response.content
        except Exception as e:
            return self._generate_fallback_display(plan, intent, str(e))

    def _generate_direct_display(
        self,
        *,
        intent: dict,
        timeline_text: str,
        activity_name: str,
        activity_type: str,
        restaurant_name: str,
        final_score,
        selected_candidate_id,
        intervention_context: str,
        time_phrase: str,
    ) -> str:
        return f"""# 方案建议（{intent.get('scenario', 'family')} 场景）

## 第一部分：行程安排
{timeline_text}

## 第二部分：方案说明
- 当前方案已根据时间窗口和通勤时间生成。
- 已结合用户场景、天气风险和餐厅可用性做初步匹配。
- 自动调整信息：{intervention_context or '无'}

## 第三部分：确认提示
当前还是方案展示阶段，确认后才会继续执行预约或下单。
"""

    def _generate_fallback_display(self, plan: dict, intent: dict, error: str) -> str:
        activities = plan.get("activities") or [{}]
        activity = activities[0]
        restaurant = plan.get("restaurant") or {}
        timeline = plan.get("timeline") or "待确认时间：先活动，再用餐"
        exceptions = plan.get("exceptions_handled") or []
        exception_text = "\n".join(f"- {item}" for item in exceptions) or "- 当前方案未触发额外自动调整。"
        final_score = plan.get("final_score", "N/A")
        selected_candidate_id = plan.get("selected_candidate_id", "")

        return f"""# 方案建议（{intent.get('scenario', 'family')} 场景）

## 第一部分：行程安排
- 时间安排：{timeline}
- 活动：{activity.get('name', '待确认活动')}（类型：{activity.get('type', '未知')}）
- 餐厅：{restaurant.get('name', '待确认餐厅')}
- 选中的候选：{selected_candidate_id}
- 最终评分：{final_score}

## 第二部分：方案说明
{exception_text}
- 当前方案基于已确认的时间表达生成，没有再默认补下午时间。
- 已结合用户场景、天气风险和餐厅可用性做初步匹配。

## 第三部分：确认提示
当前还是方案展示阶段，确认后才会继续执行预约或下单。
> fallback reason: {error}
"""
