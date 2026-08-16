# agents/intent/agent.py
from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from agents.intent.schema import IntentResult
from agents.intent.prompt import INTENT_SYSTEM_PROMPT
from harness.llm.base import LLMClientBase

logger = logging.getLogger(__name__)

# 明确辛辣的菜系词。只收录**无歧义**的：
#   收录：川菜/湘菜/麻辣/香锅/水煮/冒菜/串串/江湖菜/毛血旺/口味虾
#   不收录：火锅 —— 实测证据，云南的"爱尚菌·野生菌火锅"是菌汤、
#           "正宗富源酸菜土猪脚火锅"是酸菜，都不辣；而成都的火锅
#           通常是辣的。同一个词在不同地区含义相反，硬过滤会在
#           一半地区产生误判。
#   不收录：烧烤 —— 辣度取决于点单，不是菜系属性。
#
# 宁可漏过（有歧义的词放行，交给后面的挑选环节按 must_avoid 判断），
# 也不误杀（把用户可能想要的整类餐厅从搜索阶段就抹掉）。
# 搜索阶段的误杀是不可逆的：没搜到就不在候选池里，后面再也救不回来。
_SPICY_CUISINES = (
    "川菜", "川味", "湘菜", "麻辣", "香锅", "水煮", "冒菜",
    "串串", "江湖菜", "毛血旺", "口味虾", "重庆菜",
)

# 用户表达"不吃辣"的说法。命中任一才启用上面的过滤——
# 用户没说过不吃辣的时候，川菜是完全正当的推荐。
_NO_SPICY_SIGNALS = ("不辣", "不吃辣", "不能吃辣", "清淡", "微辣", "少辣", "忌辣", "重辣")


class IntentAgent:
    """意图解析 Agent。

    不需要工具调用和 ReAct loop——意图解析是单次推理任务。
    直接用 LLMClientBase 调用，Pydantic 校验输出，失败自动重试。

    【重试只覆盖格式错误，这是刻意的】
    except 只捕 ValidationError / JSONDecodeError。网络超时、429、
    5xx 不在这里重试——OpenAIClient 内部的 retry_with_backoff 已经
    覆盖了那一层，在这里再重试一次会变成"重试的重试"，指数退避的
    节奏被打乱，对 429 尤其糟糕（本该等待的时候反而加压）。
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

        # 有记忆上下文时注入，本轮原话永远最高优先级
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
                return self._enforce_consistency(result)

            except (ValidationError, json.JSONDecodeError) as e:
                if attempt == self._max_retries:
                    raise
                # 把错误回灌给模型，让它自己修正
                messages.append({"role": "assistant", "content": last_content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"输出格式不正确：{e}。"
                        "请严格按照 JSON Schema 重新输出，"
                        "missing_slots 必须是字符串列表如 [\"scenario\", \"time_day\"]。"
                    ),
                })

        # 结构性兜底：上面的循环在最后一轮要么 return 要么 raise，
        # 理论上到不了这里。但"到不了"是靠推理保证的，不是靠结构——
        # 万一 max_retries 被改成负数，函数会隐式返回 None，而调用方
        # 拿到 None 之后是在 .model_dump() 那一行才炸，错误现场离
        # 真正的原因隔了好几层。显式失败比隐式 None 好。
        raise RuntimeError(
            f"IntentAgent 重试循环异常退出（max_retries={self._max_retries}），未产出结果"
        )

    # ── 字段间一致性 ──────────────────────────────────────────────────

    @staticmethod
    def _enforce_consistency(result: IntentResult) -> IntentResult:
        """校验并修正**同一次输出内部**的自相矛盾。

        【这修的是一个实测到的真实矛盾】
        记忆注入"用户不太能吃辣"之后，模型的输出是：

            diet_preference = ['不辣', '清淡']
            must_avoid      = ['连锁店', '重辣']
            restaurant_keywords 里却包含 '川菜馆'

        偏好解析对了，搜索关键词却和它直接冲突。而 restaurant_keywords
        会一路走到 FactAgent 的搜索里——川菜馆搜回来的都是辣菜，
        整个候选池被污染，后面再怎么挑也挑不出符合偏好的。

        这和 PlanningAgent 已有的 validate_meal_phase_consistency 是
        同一类做法：**结构化输出的字段之间可能自相矛盾，代码要校验，
        不能指望模型自己保持一致**。单个字段的合法性 pydantic 管了，
        字段之间的关系没人管——这是 schema 校验天然的盲区。

        选"过滤冲突关键词"而不是"回灌让模型重来"：过滤是确定性的、
        零成本的；重来要多一个往返，而且模型刚刚已经在这件事上错过
        一次，没有理由相信第二次就对。

        过滤后关键词可能为空——不补默认值，交给下游
        _build_plan_context 的既有兜底（它本来就有一套
        `or diet_preference or ["简餐","聚餐","特色餐厅"]` 的链）。
        在这里再补一层会变成两处兜底、口径可能不一致。
        """
        prefs = result.preferences
        if prefs is None:
            return result

        diet = _as_text_list(getattr(prefs, "diet_preference", None))
        avoid = _as_text_list(getattr(prefs, "must_avoid", None))
        signals = diet + avoid
        wants_no_spicy = any(
            sig in item for item in signals for sig in _NO_SPICY_SIGNALS
        )
        if not wants_no_spicy:
            return result

        for field in ("restaurant_keywords", "restaurant_explicit_types"):
            values = _as_text_list(getattr(result, field, None))
            if not values:
                continue
            kept = [v for v in values
                    if not any(sp in v for sp in _SPICY_CUISINES)]
            dropped = [v for v in values if v not in kept]
            if dropped:
                logger.warning(
                    f"[IntentAgent] 字段间矛盾：用户表达了不吃辣"
                    f"（diet={diet} avoid={avoid}），但 {field} 含辛辣菜系 "
                    f"{dropped}，已过滤。剩余：{kept}"
                )
                try:
                    setattr(result, field, kept)
                except Exception as e:
                    # pydantic 模型字段可能是只读或类型不符——过滤失败
                    # 只记日志不抛异常：一个不完美的关键词列表远好过
                    # 让整次意图解析失败。
                    logger.error(f"[IntentAgent] 写回 {field} 失败: {e}")

        return result


def _as_text_list(v) -> list[str]:
    """把 str / list / None 统一成字符串列表。
    IntentResult 里这几个字段的实际类型随模型输出浮动
    （schema 说是 list，模型偶尔给 str），统一处理不做假设。"""
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, (list, tuple)):
        return [x.strip() for x in v if isinstance(x, str) and x.strip()]
    return []