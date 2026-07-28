# harness/message_id.py
from __future__ import annotations

import uuid

# 字段名故意以下划线开头 + 简短——它是 harness 内部记账用的元数据,
# 混在消息 dict 里跟随消息本体一起流转(不用旁路映射表,理由见设计
# 讨论:两本账必然漂移,这个仓库已经为此付过学费)。命名要足够独特,
# 不会和 provider 的字段、业务字段撞车,又不至于太长影响可读性。
HID_KEY = "_hid"


def new_hid() -> str:
    return uuid.uuid4().hex


def tag_message(message: dict) -> dict:
    """给一条 harness 自己创建的 dict 消息打上稳定 ID,返回新 dict
    (不原地修改入参)。

    只用于 harness 自己 new 出来的 dict 消息(目前两处:Agent._messages
    里的 task 消息、recall.py 里的召回注入消息)。绝不用于 assistant/
    tool 消息——它们在真实运行中经常是 provider SDK 返回的 pydantic/
    dataclass 对象,不受控地 setattr 任意字段是脆弱依赖(这类对象的
    字段集合由 SDK 版本决定,不是我们能保证稳定的接口)。这是"只给
    需要被锚定的消息打 ID"这个最小化原则的直接体现:需要被水位线/
    surfaced 机制引用的,只有这两类消息。
    """
    return {**message, HID_KEY: new_hid()}


def get_hid(message) -> str | None:
    """兼容 dict 和 SDK 消息对象两种形态(与 token_counter.py 处理
    tool_calls 时的兼容写法同构)。SDK 对象上不会有这个属性,统一
    返回 None——调用方据此天然知道"这条消息没有稳定 ID,不能被
    锚定引用",不需要额外判断消息类型。"""
    if isinstance(message, dict):
        return message.get(HID_KEY)
    return None


def find_index_by_hid(messages: list, hid: str | None) -> int | None:
    """在消息列表里按 hid 定位下标。hid 为 None 时直接返回 None
    (调用方不该拿 None 来查找,这里防御性处理而不是让它意外匹配到
    某条同样没有 _hid、get_hid 也返回 None 的消息——那会是一个
    静默的错误匹配,比抛异常更危险)。"""
    if hid is None:
        return None
    for i, m in enumerate(messages):
        if get_hid(m) == hid:
            return i
    return None


def strip_hid(message: dict) -> dict:
    """出网前剥离。只在确认有这个 key 时才拷贝一份新 dict,没有则
    原样返回入参——避免对绝大多数(本来就没有 _hid 的 assistant/tool
    消息)做无意义的拷贝。"""
    if HID_KEY not in message:
        return message
    return {k: v for k, v in message.items() if k != HID_KEY}