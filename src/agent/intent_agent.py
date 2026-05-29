from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.model.factory import get_chat_model


# ── 子模型 ────────────────────────────────────────────────────────────────────

class ParticipantsInfo(BaseModel):
    people_count: int | None = Field(None, description="总人数，用户未提及时为 null")
    has_child: bool = Field(False, description="是否有儿童随行")
    child_age: int | None = Field(None, description="儿童年龄（岁），未提及时为 null")


class TimeInfo(BaseModel):
    date_label: str | None = Field(None, description="日期语义标签，如 '今天' '明天' '周末'，未提及时为 null")
    start_time: str | None = Field(None, description="活动开始时间，24小时制，如 '15:00'，未提及时为 null")
    end_time: str | None = Field(None, description="活动结束时间，24小时制，如 '20:00'，未提及时为 null")
    time_phrase: str | None = Field(None, description="原文时间描述，如 '下午三点到晚上八点'，供展示用")


class LocationInfo(BaseModel):
    origin_area_hint: str | None = Field(None, description="出发/所在区域，如 '国贸'，未提及时为 null")
    location_text: str | None = Field(None, description="原文中的位置描述，未提及时为 null")


class PreferencesInfo(BaseModel):
    distance_preference: str | None = Field(None, description="距离偏好，如 '别太远'，未提及时为 null")
    diet_preference: list[str] = Field(default_factory=list, description="饮食偏好关键词，如 ['火锅', '减脂']")
    activity_style: list[str] = Field(default_factory=list, description="活动风格偏好，如 ['亲子', '拍照']")
    must_avoid: list[str] = Field(default_factory=list, description="明确排除的内容，如 ['烧烤']")


# ── 主输出模型 ─────────────────────────────────────────────────────────────────

class IntentResult(BaseModel):
    is_leisure_planning: bool = Field(description="是否为休闲/出行规划类请求")
    scenario: Literal["family", "friends", "couple", "team", "unknown", "none"] = Field(
        description=(
            "场景类型："
            "family=家庭/亲子, friends=朋友聚餐, couple=情侣约会, "
            "team=公司团建, unknown=有规划意图但场景不明, none=非规划需求"
        ),
    )
    participants: ParticipantsInfo = Field(default_factory=ParticipantsInfo)
    time: TimeInfo = Field(default_factory=TimeInfo)
    location: LocationInfo = Field(default_factory=LocationInfo)
    preferences: PreferencesInfo = Field(default_factory=PreferencesInfo)

    restaurant_keywords: list[str] = Field(
        default_factory=list,
        description="可用于 POI 搜索的餐厅类型词，3-5 个，如 '火锅' '日料' '亲子餐厅'",
    )
    restaurant_explicit_types: list[str] = Field(
        default_factory=list,
        description="用户明确说出的餐厅类型，没有则空列表",
    )
    activity_keywords: list[str] = Field(
        default_factory=list,
        description="可用于 POI 搜索的活动场地类型词，3-5 个，如 '公园' '美术馆' '儿童乐园'",
    )
    activity_explicit_types: list[str] = Field(
        default_factory=list,
        description="用户明确说出的活动类型，没有则空列表",
    )

    need_retrieval: bool = Field(description="是否需要调用外部 POI/餐厅检索")
    clarification_needed: bool = Field(description="是否需要向用户追问缺失信息")

    # 按优先级排列的缺口列表，LLM 只填实际缺失的槽位
    missing_slots: list[str] = Field(
        default_factory=list,
        description=(
            "缺失的关键槽位，按优先级顺序排列，"
            "可选值：scenario / time_day / time_window / origin_area / people_count"
        ),
    )
    current_asking_slot: str | None = Field(
        None,
        description="当前正在追问的槽位名（missing_slots[0]），clarification_needed=false 时为 null",
    )
    follow_up_message: str | None = Field(
        None,
        description="只针对 missing_slots[0] 生成的一句自然追问话术，clarification_needed=false 时为 null",
    )

    raw_query: str = Field(default="", description="原样返回用户输入，不做任何修改")


# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个本地休闲行程规划助手的意图解析模块。
你的唯一职责：理解用户输入，提取结构化意图，严格按照给定 schema 的 JSON 格式返回，不添加任何多余解释。

【必须严格遵循的 JSON Schema】
{schema}

## 场景判断
- family：提到家人、亲子、老婆/老公、孩子、全家
- friends：提到朋友、同学
- couple：提到情侣、约会、两个人、男/女朋友
- team：提到公司、同事、团建
- unknown：有休闲规划意图但场景不明确
- none：闲聊、问答、非规划需求

## 时间解析
直接理解用户原文，转换为 24 小时制填入 start_time / end_time：
- "下午三点到晚上八点" → start_time: "15:00", end_time: "20:00"
- "上午" → start_time: "09:00", end_time: "12:00"
- "下午" → start_time: "13:00", end_time: "18:00"
- "晚上" → start_time: "18:00", end_time: "21:00"
用户完全没有提到时间段时，start_time / end_time 均填 null。

## 关键词生成规则
### 餐厅关键词
- ✅ 合法：火锅、日料、亲子餐厅、烤肉、轻食
- ❌ 非法：不辣的餐厅、适合聚会的地方
- 用户明确提到的类型必须保留，额外补充 1-2 个互补类型
- 排除项放入 preferences.must_avoid，不放入 keywords

### 活动关键词
- ✅ 合法：商场、公园、美术馆、儿童乐园、博物馆
- ❌ 非法：逛街、散步、室内、户外

## people_count 推断规则
不要轻易把 people_count 列为缺口，优先从上下文推断：
- "我和老婆孩子" / "我们一家三口" → people_count=3
- "带老婆" / "和女朋友" / "两个人" → people_count=2
- "我和朋友们" / "几个同事" → people_count 仍未知，可列为缺口
只有真正无法推断时，才将 people_count 列入 missing_slots。

## start_time / end_time 严格规则
**用户在整个对话中从未提到过具体时间段时，start_time 和 end_time 必须填 null，禁止脑补任何默认值。**
即使是多轮对话，只要用户没有明确说过时间，就保持 null。

## missing_slots 与追问规则
is_leisure_planning=true 时，按以下优先级检查缺口，所有缺失的都要填入 missing_slots：
1. scenario 为 unknown → 加入 "scenario"
2. date_label 为 null → 加入 "time_day"
3. date_label 不为 null 但 start_time 为 null → 加入 "time_window"
4. origin_area_hint 为 null → 加入 "origin_area"
5. people_count 经过推断后仍为 null → 加入 "people_count"

clarification_needed = (missing_slots 非空)
current_asking_slot = missing_slots[0]
follow_up_message = 只针对 current_asking_slot 的一句自然口语追问，例如：
  - scenario    → "这次打算和谁一起出去？家人、朋友还是另一半？"
  - time_day    → "你想安排在哪天呀？"
  - time_window → "请告知我你们计划的时间段"
  - origin_area → "你们大概从哪个区域出发？"
  - people_count→ "这次一共几个人一起去？"

is_leisure_planning=false 时：clarification_needed=false，missing_slots=[]，follow_up_message=null
"""

#    - origin_area → "你们大概从哪个区域出发？" 位置信息默认
# ── Agent ─────────────────────────────────────────────────────────────────────

class IntentAgent:
    def __init__(self, max_retries: int = 2):
        self._model = get_chat_model()
        self._max_retries = max_retries

    def parse(self, user_input: str) -> IntentResult:
        print(f"[IntentAgent] 解析中：{user_input!r}")

        schema_str = json.dumps(IntentResult.model_json_schema(), ensure_ascii=False, indent=2)
        system_content = _SYSTEM_PROMPT.format(schema=schema_str)

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_input),
        ]

        last_response_content = ""

        for attempt in range(self._max_retries + 1):
            try:
                response = self._model.invoke(messages)
                last_response_content = response.content

                match = re.search(r"\{.*\}", response.content, re.DOTALL)
                raw = match.group() if match else response.content

                result = IntentResult.model_validate_json(raw)

                # 强制同步：确保 current_asking_slot 与 missing_slots 一致，不依赖 LLM 自觉
                if result.missing_slots:
                    result.current_asking_slot = result.missing_slots[0]
                    result.clarification_needed = True
                else:
                    result.current_asking_slot = None
                    result.clarification_needed = False
                    result.follow_up_message = None

                result.raw_query = user_input

                print(
                    f"[IntentAgent] 完成：scenario={result.scenario}, "
                    f"missing_slots={result.missing_slots}"
                )
                print(result.model_dump_json(indent=2))
                return result

            except (ValidationError, json.JSONDecodeError) as e:
                if attempt == self._max_retries:
                    raise
                print(f"[IntentAgent] 第 {attempt + 1} 次解析失败，重试：{e}")
                messages.append(AIMessage(content=last_response_content))
                messages.append(HumanMessage(
                    content=(
                        f"输出格式不正确：{e}。"
                        "请严格按照 JSON Schema 重新输出，确保包含所有必填字段，"
                        "missing_slots 必须是字符串列表，如 [\"scenario\", \"time_day\"]。"
                    )
                ))