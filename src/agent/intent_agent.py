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


class WaypointRequest(BaseModel):
    raw_text: str = Field(description="用户原文描述，如'吃DQ冰淇淋'")
    keyword: str = Field(description="直接用于高德搜索的关键词，如'DQ''星巴克''冰淇淋'")
    time_hint: str | None = Field(None, description="用户指定的时间，如'中途''下午'，没有则null")


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
        description="可用于 POI 搜索的餐厅类型词，3-5 个，如 '川菜' '火锅' '聚会餐厅'",
    )
    restaurant_explicit_types: list[str] = Field(
        default_factory=list,
        description="用户明确说出的餐厅类型，没有则空列表",
    )
    activity_keywords: list[str] = Field(
        default_factory=list,
        description="可用于 POI 搜索的活动场地类型词，3-5 个，覆盖该场景下多样的活动可能",
    )
    activity_explicit_types: list[str] = Field(
        default_factory=list,
        description="用户明确说出的活动类型，没有则空列表",
    )
    waypoint_requests: list[WaypointRequest] = Field(
        default_factory=list,
        description=(
            "用户明确点名的途径需求，包括品牌/小吃/饮品等不属于主要活动或正餐的需求。"
            "例如'吃DQ冰淇淋'→ keyword='DQ'，'顺路买杯星巴克'→ keyword='星巴克'，"
            "'中途想吃个甜品'→ keyword='甜品'。没有则空列表。"
        ),
    )

    need_retrieval: bool = Field(description="是否需要调用外部 POI/餐厅检索")
    clarification_needed: bool = Field(description="是否需要向用户追问缺失信息")

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
你的职责：理解用户输入，判断意图与场景、识别信息缺口、并为后续的 POI 检索准备一批多样的搜索关键词。为了形成准确的关键词，必须要进行命名实体识别，识别出诸如"DQ冰淇淋"，"老地方火锅"等特定名称，并将这些结果根据你的理解放入活动或者餐厅关键词。
严格按照给定 schema 的 JSON 格式返回，不添加任何多余解释。

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
- "下午三点到晚上十点" → start_time: "15:00", end_time: "22:00"
- "上午" → start_time: "09:00", end_time: "12:00"
- "下午" → start_time: "13:00", end_time: "18:00"
- "晚上" → start_time: "18:00", end_time: "21:00"
**用户完全没有提到时间段时，start_time / end_time 均填 null，禁止脑补默认值。**

## waypoint_requests（途径需求，重要）
用户有时会提到非主要活动、非正餐的小需求，如：
- "中途想吃个DQ冰淇淋" → keyword="DQ"
- "顺路买杯星巴克" → keyword="星巴克"
- "想喝个奶茶" → keyword="奶茶"
- "去买点甜品" → keyword="甜品"

这类需求的特征：不是主要活动（不占用完整时段），不是正餐（不是午饭/晚饭），
而是行程中的"顺路小事"。必须识别并填入 waypoint_requests，keyword 直接用于高德搜索。
**用户需求至上，明确的品牌名（DQ/星巴克/麦当劳）或品类（奶茶/甜品/冰淇淋）都要识别。**

## 关键词生成（核心理念）
关键词只是用来去地图 API 检索 POI、把一批多样的候选场所捞进池子，供后续规划模块挑选。

1. **按场景发挥推荐能力，生成多样的活动类型。**（3-5 个，尽量覆盖不同类型）

2. **用户明确点名的活动，只占其中一项，绑定到它出现的时段，不扩散污染整个列表。**

3. **餐厅关键词对应"吃饭那顿"（川菜/火锅/日料/轻食等），不被酒类或小吃主导。**
   - waypoint_requests 里的小需求（奶茶/冰淇淋）不要混入 restaurant_keywords

4. **关键词必须是可被地图按类型搜索的名词**，不要写描述性语句或场景标签：
   - ❌ 非法："情侣约会""朋友聚餐""家庭出游""适合情侣""休闲活动"——这些是场景描述，高德搜不出有效场地
   - ✅ 合法："咖啡馆""美术馆""书店""花市""文创园""公园""电影院""桌游""台球""KTV"
   - couple 场景典型示例：咖啡馆、美术馆、公园、电影院、书店（选3-5个，绝对不要出现"情侣约会"）
   - friends 场景典型示例：桌游、剧本杀、台球、KTV、咖啡馆

5. **关键词必须包含命名实体识别的结果，用户特别指定的内容需要根据你的理解加入到活动或者餐厅关键词中用于搜索**
   - 合法示例：DQ冰淇淋、银河九天KTV、老地方火锅...
   - 类似"中途想吃DQ冰淇淋"这种输入就需要把'DQ冰淇淋'放到活动关键词中，类似"中午/晚上想吃DQ冰淇淋"这种输入就需要把'DQ冰淇淋'放到餐厅关键词中

## people_count 推断规则
- "我和老婆孩子" / "一家三口" → 3；"带老婆" / "两个人" → 2；"和三个朋友" → 4
- 真正无法推断时才列入 missing_slots

## missing_slots 与追问规则
is_leisure_planning=true 时，按优先级检查：
1. scenario 为 unknown → "scenario"
2. date_label 为 null → "time_day"
3. date_label 不为 null 但 start_time 为 null → "time_window"
4. origin_area_hint 为 null → "origin_area"
5. people_count 经推断后仍为 null → "people_count"

clarification_needed = (missing_slots 非空)
current_asking_slot = missing_slots[0]
follow_up_message = 只针对 current_asking_slot 的一句自然口语追问

is_leisure_planning=false 时：clarification_needed=false，missing_slots=[]，follow_up_message=null
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class IntentAgent:
    def __init__(self, max_retries: int = 2):
        self._model = get_chat_model()
        self._max_retries = max_retries

    def parse(
        self,
        user_input: str,
        runtime_origin_area: str = "",
        preference_context: str = "",
    ) -> IntentResult:
        print(f"[IntentAgent] 解析中：{user_input!r}")

        schema_str = json.dumps(IntentResult.model_json_schema(), ensure_ascii=False, indent=2)
        system_content = _SYSTEM_PROMPT.format(schema=schema_str)
        user_content = user_input
        if preference_context.strip():
            user_content = f"""\
# 历史偏好档案（软参考，不是硬约束）
{preference_context.strip()}

## 使用规则
- 本轮用户原话永远优先于历史偏好。
- 历史偏好只用于补充软偏好、活动/餐饮关键词和风格倾向。
- 不要用历史偏好填充日期、时间、出发地、人数等硬槽位，除非用户明确说"照旧""和上次一样""老地方"。
- 如果本轮需求与历史偏好冲突，以本轮需求为准。

# 本轮用户原话（最高优先级）
{user_input}
"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]

        last_response_content = ""

        for attempt in range(self._max_retries + 1):
            try:
                response = self._model.invoke(messages)
                last_response_content = response.content

                match = re.search(r"\{.*\}", response.content, re.DOTALL)
                raw = match.group() if match else response.content

                result = IntentResult.model_validate_json(raw)

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
                    f"missing_slots={result.missing_slots}, "
                    f"waypoints={[w.keyword for w in result.waypoint_requests]}"
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