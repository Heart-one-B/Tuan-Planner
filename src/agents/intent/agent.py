from __future__ import annotations

import json
import re

from pydantic import ValidationError

from agents.intent.schema import IntentResult
from agents.intent.prompt import INTENT_SYSTEM_PROMPT
from harness.llm.base import LLMClientBase


class IntentAgent:
    """意图解析 Agent。

    不需要工具调用和 ReAct loop——意图解析是单次推理任务。
    直接用 LLMClientBase 调用,Pydantic 校验输出,失败自动重试。
    和 StructuredAgent 共用同一个 LLMClientBase,遵循 Harness 注入原则。
    """

    def __init__(self, llm_client: LLMClientBase, max_retries: int = 2):
        self._llm = llm_client
        self._max_retries = max_retries

    async def parse(
        self,
        user_input: str,
        preference_context: str = "",
        trace_id: str | None = None,
    ) -> IntentResult:
        schema_str = json.dumps(
            IntentResult.model_json_schema(), ensure_ascii=False, indent=2
        )
        system_prompt = INTENT_SYSTEM_PROMPT.format(schema=schema_str)

        # 有记忆上下文时注入,本轮原话永远最高优先级
        if preference_context.strip():
            user_content = (
                f"# 历史偏好档案（软参考，不是硬约束）\n"
                f"{preference_context.strip()}\n\n"
                f"## 使用规则\n"
                f"- 本轮用户原话永远优先于历史偏好\n"
                f"- 历史偏好只用于补充软偏好和关键词倾向\n"
                f"- 不要用历史偏好填充日期、时间、出发地、人数等硬槽位\n\n"
                f"# 本轮用户原话（最高优先级）\n{user_input}"
            )
        else:
            user_content = user_input

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_content = ""

        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._llm.call(
                    trace_id=trace_id or "intent",
                    messages=messages,
                )
                last_content = resp.choices[0].message.content or ""

                match = re.search(r"\{.*\}", last_content, re.DOTALL)
                raw = match.group() if match else last_content
                result = IntentResult.model_validate_json(raw)

                # 框架层统一修正 clarification 相关字段
                if result.missing_slots:
                    result.current_asking_slot = result.missing_slots[0]
                    result.clarification_needed = True
                else:
                    result.current_asking_slot = None
                    result.clarification_needed = False
                    result.follow_up_message = None

                result.raw_query = user_input
                return result

            except (ValidationError, json.JSONDecodeError) as e:
                if attempt == self._max_retries:
                    raise
                # 把错误回灌给模型,让它自己修正
                messages.append({"role": "assistant", "content": last_content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"输出格式不正确：{e}。"
                        "请严格按照 JSON Schema 重新输出，"
                        "missing_slots 必须是字符串列表如 [\"scenario\", \"time_day\"]。"
                    ),
                })