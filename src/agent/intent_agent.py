from __future__ import annotations

import json
import re
from typing import Literal
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
#
# 重新定位：意图模块是"分诊台 + 检索调度"，不是"语义翻译官"。
#   - 路由判断（是否规划 / 场景 / 缺口）必须结构化，给控制流用。
#   - 关键词只是"喂给高德检索"的输入，用来把一批多样的 POI 捞进池子，
#     不是替规划节点预判行程。规划由下游模型读 raw_query 自己做。
#   - 因此关键词生成的目标是【多样、覆盖场景】，不是【精确复述用户某个词】。
#     用户点名的活动（如"喝酒"）只占其中一项、绑定到它出现的时段，绝不扩散到所有关键词。
#
_SYSTEM_PROMPT = """\
你是一个本地休闲行程规划助手的意图解析模块。
你的职责：理解用户输入，判断意图与场景、识别信息缺口、并为后续的 POI 检索准备一批多样的搜索关键词。为了形成准确的关键词，必须要进行命名实体识别，识别出诸如“DQ冰淇淋”，“老地方火锅”等特定名称，并将这些结果根据你的理解放入活动或者餐厅关键词
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
**用户完全没有提到时间段时，start_time / end_time 均填 null，禁止脑补默认值。** 即使多轮对话，只要用户没明确说过时间，就保持 null。

## 关键词生成（核心理念，请认真理解）
关键词只是用来去地图 API 检索 POI、把一批**多样**的候选场所捞进池子，供后续规划模块挑选。
它**不是**让你替用户决定行程，也**不是**精确复述用户说的某个词。所以：

1. **按场景发挥推荐能力，生成多样的活动类型。**
   就像有人问你"我和朋友周末想聚聚，有什么推荐"，你会自然想到桌游、剧本杀、台球、KTV、咖啡馆、livehouse、电影、密室、运动馆等多种可能——请把这种多样性体现在 activity_keywords 里（3-5 个，尽量覆盖不同类型，不要高度同质）。
   - 不同场景的典型活动各不相同，请结合场景自行判断，不要套用固定模板。

2. **用户明确点名的活动，只占其中一项，并绑定到它出现的时段，绝不让它扩散污染整个列表。**
   例如"晚上想喝点酒"——"喝酒"是**晚间的一项活动**，对应 activity_keywords 里**一个**词（如"清吧"或"小酒馆"），不要因此让 activity_keywords 变成"酒吧/小酒馆/夜市/餐吧"一整排都围着酒转，那样会把"和朋友聚聚"本该有的下午活动（桌游、台球等）全挤掉。
   - 判断要点：用户"在某个时段想做某事"≠"整天的主题就是这件事"。点缀性的（喝点、顺便、随便）尤其不要放大成主题。

3. **餐厅关键词对应"吃饭那顿"，不要被酒类场所主导。**
   - 合法示例：川菜、火锅、烤肉、日料、轻食、聚会餐厅、特色餐厅
   - "晚上喝点酒"不应让 restaurant_keywords 变成"酒吧/居酒屋/清吧"——喝酒归活动（晚间），吃饭那顿仍应是正经餐厅类型（结合场景与地域，如成都朋友聚会可给"川菜/火锅/串串/特色餐厅"）。
   - 用户明确点名的餐厅类型放入 restaurant_explicit_types 并保留在 restaurant_keywords 中。
   - 明确排除项放入 preferences.must_avoid，不放入 keywords。

4. **关键词必须是可被地图按"类型/品类"搜索的名词**（如 火锅、桌游、公园、美术馆），
   不要写"不辣的餐厅""适合聚会的地方""逛街""散步"这类无法直接检索的描述。
   
5. **关键词必须包含命名实体识别的结果，用户特别指定的内容需要根据你的理解加入到活动或者餐厅关键词中用于搜索**
   - 合法示例：DQ冰淇淋、银河九天KTV、老地方火锅...
   - 类似“中途想吃DQ冰淇淋”这种输入就需要把‘DQ冰淇淋’放到活动关键词中，类似“中午/晚上想吃DQ冰淇淋”这种输入就需要把‘DQ冰淇淋’放到餐厅关键词中

## people_count 推断规则
优先从上下文推断，不要轻易列为缺口：
- "我和老婆孩子" / "一家三口" → 3
- "带老婆" / "和女朋友" / "两个人" → 2
- "和三个朋友" → 4（用户 + 3）
- "我和朋友们" / "几个同事" → 仍未知，可列为缺口
只有真正无法推断时，才将 people_count 列入 missing_slots。

## missing_slots 与追问规则
is_leisure_planning=true 时，按以下优先级检查缺口，所有缺失的都填入 missing_slots：
1. scenario 为 unknown → "scenario"
2. date_label 为 null → "time_day"
3. date_label 不为 null 但 start_time 为 null → "time_window"
4. origin_area_hint 为 null → "origin_area"
5. people_count 经推断后仍为 null → "people_count"

clarification_needed = (missing_slots 非空)
current_asking_slot = missing_slots[0]
follow_up_message = 只针对 current_asking_slot 的一句自然口语追问，例如：
  - scenario    → "这次打算和谁一起出去？家人、朋友还是另一半？"
  - time_day    → "你想安排在哪天呀？"
  - time_window → "大概几点到几点呢？"
  - origin_area → "你们大概从哪个区域出发？"
  - people_count→ "这次一共几个人一起去？"

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
- 不要用历史偏好填充日期、时间、出发地、人数等硬槽位，除非用户明确说“照旧”“和上次一样”“老地方”。
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

                # 强制同步：current_asking_slot 与 missing_slots 一致，不依赖 LLM 自觉
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
