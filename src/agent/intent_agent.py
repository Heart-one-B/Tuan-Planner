from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.model.factory import get_chat_model
import json
import re
from pydantic import ValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ── 子模型 ────────────────────────────────────────────────────────────────────

class ParticipantsInfo(BaseModel):
    people_count: int | None = Field(
        None,
        description="总人数，用户未提及时为 null",
    )
    has_child: bool = Field(
        False,
        description="是否有儿童随行",
    )
    child_age: int | None = Field(
        None,
        description="儿童年龄（岁），未提及时为 null",
    )


class TimeInfo(BaseModel):
    date_label: str | None = Field(
        None,
        description="日期语义标签，如 '今天' '明天' '周末' '下周六'，未提及时为 null",
    )
    start_time: str | None = Field(
        None,
        description="活动开始时间，24小时制字符串，如 '14:00'，未提及时为 null",
    )
    end_time: str | None = Field(
        None,
        description="活动结束时间，24小时制字符串，如 '17:00'，未提及时为 null",
    )
    duration_hours: float | None = Field(
        None,
        ge=0.5,
        description="预计活动时长（小时），未提及时为 null",
    )
    time_phrase: str | None = Field(
        None,
        description="原文时间描述短语，如 '周末下午三点到五点'，供展示用",
    )

    @model_validator(mode="after")
    def infer_missing_times(self) -> TimeInfo:
        """start / end / duration 三者知二推一。"""
        try:
            if self.start_time and self.end_time and self.duration_hours is None:
                sh, sm = map(int, self.start_time.split(":"))
                eh, em = map(int, self.end_time.split(":"))
                self.duration_hours = round((eh * 60 + em - sh * 60 - sm) / 60, 1)
            elif self.start_time and self.duration_hours and self.end_time is None:
                sh, sm = map(int, self.start_time.split(":"))
                total = sh * 60 + sm + int(self.duration_hours * 60)
                self.end_time = f"{total // 60:02d}:{total % 60:02d}"
        except (ValueError, AttributeError):
            pass
        return self


class LocationInfo(BaseModel):
    origin_area_hint: str | None = Field(
        None,
        description="用户提到的出发/所在区域，如 '国贸' '家附近'，未提及时为 null",
    )
    location_text: str | None = Field(
        None,
        description="原文中的位置描述，未提及时为 null",
    )


class PreferencesInfo(BaseModel):
    distance_preference: str | None = Field(
        None,
        description="距离偏好原文描述，如 '别太远' '可以跑远一点'，未提及时为 null",
    )
    diet_preference: list[str] = Field(
        default_factory=list,
        description="饮食偏好关键词列表，如 ['火锅', '减脂']",
    )
    activity_style: list[str] = Field(
        default_factory=list,
        description="活动风格偏好，如 ['亲子', '拍照']",
    )
    must_avoid: list[str] = Field(
        default_factory=list,
        description="明确排除的内容，如 ['烧烤', '自助']",
    )


class MissingSlots(BaseModel):
    fields: list[str] = Field(
        default_factory=list,
        description=(
            "缺失的关键槽位名称列表，可选值："
            "scenario / time_day / time_window / origin_area / people_count"
        ),
    )


# ── 主输出模型 ─────────────────────────────────────────────────────────────────

class IntentResult(BaseModel):
    """意图分析 Agent 的结构化输出。"""

    is_leisure_planning: bool = Field(
        description="是否为休闲/出行规划类请求",
    )
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
        description=(
            "可直接用于高德/大众点评搜索的餐厅候选关键词，3-5 个。"
            "必须是 POI 类型词（如 '火锅' '日料' '亲子餐厅'），"
            "禁止出现 '不辣餐厅' '菜系' 这类无法被 API 检索的描述词。"
        ),
    )
    restaurant_explicit_types: list[str] = Field(
        default_factory=list,
        description="用户明确说出的餐厅类型，没有则空列表",
    )
    activity_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "可直接用于高德搜索的活动场地关键词，3-5 个。"
            "必须是 POI 场所类型词（如 '商场' '公园' '美术馆' '儿童乐园'），"
            "禁止出现 '逛街' '散步' '室内' '户外' 这类泛词。"
        ),
    )
    activity_explicit_types: list[str] = Field(
        default_factory=list,
        description="用户明确说出的活动类型，没有则空列表",
    )

    need_retrieval: bool = Field(
        description="是否需要调用外部 POI/餐厅检索",
    )
    clarification_needed: bool = Field(
        description="是否需要向用户追问缺失信息",
    )
    missing_slots: MissingSlots = Field(default_factory=MissingSlots)
    follow_up_message: str | None = Field(
        None,
        description="clarification_needed=true 时向用户提问的话术，否则为 null",
    )
    raw_query: str = Field(
        description="原样返回用户输入，不做任何修改",
    )


# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个本地休闲行程规划助手的意图解析模块。
你的唯一职责：理解用户输入，提取结构化意图，严格按照给定 schema 返回，不添加任何多余解释。

## 场景判断
- family：提到家人、亲子、老婆/老公、孩子、全家
- friends：提到朋友、同学
- couple：提到情侣、约会、两个人、男/女朋友
- team：提到公司、同事、团建
- unknown：有休闲规划意图但场景不明确
- none：闲聊、问答、非规划需求

## 时间解析
- 用户说"下午"推断为 start_time: "13:00", end_time: "18:00"
- 用户说"晚上吃饭"推断为 start_time: "18:00", end_time: "21:00"
- 用户说"上午"推断为 start_time: "09:00", end_time: "12:00"
- 用户明确说"三点到五点"则直接填 start_time: "15:00", end_time: "17:00"
- 有 start 和 end 时不需要填 duration_hours，validator 会自动推算

## 关键词生成规则
### 餐厅关键词
- ✅ 合法：火锅、日料、亲子餐厅、烤肉、轻食
- ❌ 非法：不辣的餐厅、适合聚会的地方、菜系
- 用户明确提到的类型必须保留，额外补充 1-2 个互补类型
- 排除项放入 preferences.must_avoid，不放入 keywords

### 活动关键词
- ✅ 合法：商场、公园、美术馆、儿童乐园、博物馆、购物中心
- ❌ 非法：逛街、散步、室内、户外、休闲
- 尽量同时包含室内和户外两类场所

## 澄清规则
is_leisure_planning=true 且以下槽位缺失时，设置 clarification_needed=true：
- scenario 不明确 → 追问"这次是和家人、朋友还是情侣一起？"
- time_day 缺失 → 追问"你想安排在哪天？"
- time_window 缺失（有日期但无时间段）→ 追问"大概几点开始？"
每次只问一个问题，优先级：scenario > time_day > time_window
is_leisure_planning=false 时 clarification_needed 必须为 false。
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class IntentAgent:
    def __init__(self, max_retries: int = 2):
        self._model = get_chat_model()
        self._max_retries = max_retries

    def parse(self, user_input: str) -> IntentResult:
        print(f"[IntentAgent] 解析中：{user_input!r}")

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]

        for attempt in range(self._max_retries + 1):
            try:
                response = self._model.invoke(messages)
                # 提取 JSON 块
                text = response.content
                match = re.search(r"\{.*\}", text, re.DOTALL)
                raw = match.group() if match else text
                # 直接用 Pydantic 解析，字段不合法会抛 ValidationError
                result = IntentResult.model_validate_json(raw)
                result.raw_query = user_input
                print(
                    f"[IntentAgent] 完成：scenario={result.scenario}, "
                    f"clarification_needed={result.clarification_needed}"
                )
                print(f"[IntentAgent] 原始结果: {result.model_dump_json(indent=2)}")
                return result
            except (ValidationError, json.JSONDecodeError) as e:
                if attempt == self._max_retries:
                    raise
                print(f"[IntentAgent] 第 {attempt + 1} 次解析失败，重试：{e}")
                # 把错误反馈给 LLM，引导它修正
                messages.append(AIMessage(content=response.content))
                messages.append(HumanMessage(content=f"输出格式不正确：{e}，请严格按照 JSON schema 重新输出。"))

if __name__ == "__main__":
    agent = IntentAgent()

    test_cases = [
        "周末想带老婆孩子去吃火锅然后逛商场",
    ]

    for i, query in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"[Test Case {i}] 用户输入: {query}")

        try:
            result = agent.parse(query)

            print("\n[结构化结果]")
            print(result.model_dump_json(indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")