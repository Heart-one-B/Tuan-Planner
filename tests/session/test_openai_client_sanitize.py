# tests/test_openai_client_sanitize.py
from harness.llm.openai_client import OpenAIClient
from harness.message_id import HID_KEY, tag_message


def test_sanitize_strips_hid_from_dict_messages():
    messages = [
        {"role": "system", "content": "系统提示"},
        tag_message({"role": "user", "content": "你好"}),
    ]
    assert HID_KEY in messages[1]  # 打标动作本身生效,先确认前提成立

    cleaned = OpenAIClient._sanitize_for_wire(messages)

    for m in cleaned:
        assert HID_KEY not in m  # 出网前必须被剥干净,一个都不能漏


def test_sanitize_strips_reasoning_content_and_hid_together():
    messages = [
        {"role": "assistant", "content": "回复", "reasoning_content": "内部推理过程"},
        tag_message({"role": "user", "content": "继续"}),
    ]
    cleaned = OpenAIClient._sanitize_for_wire(messages)
    assert "reasoning_content" not in cleaned[0]
    assert HID_KEY not in cleaned[1]


def test_sanitize_does_not_mutate_original_messages():
    """净化不该改动调用方原始的消息对象(那些消息可能还要被拿去做
    快照持久化、水位线/召回的 hid 查找——如果被原地改没了 _hid,
    这些后续用途全部失效)。"""
    original = tag_message({"role": "user", "content": "你好"})
    messages = [original]
    OpenAIClient._sanitize_for_wire(messages)
    assert HID_KEY in original  # 原始 dict 应该完好无损


def test_sanitize_passes_through_messages_without_hid_unchanged():
    """没有 _hid 的消息(比如 assistant/tool 消息)应该原样通过,
    不做无意义的拷贝。"""
    plain = {"role": "assistant", "content": "普通回复"}
    messages = [plain]
    cleaned = OpenAIClient._sanitize_for_wire(messages)
    assert cleaned[0] is plain  # 同一个对象,没有被拷贝


def test_sanitize_leaves_non_dict_messages_untouched():
    """SDK 消息对象(非 dict)不会有 _hid,也不该被这个函数动到。"""
    from types import SimpleNamespace
    sdk_message = SimpleNamespace(role="assistant", content="回复")
    messages = [sdk_message]
    cleaned = OpenAIClient._sanitize_for_wire(messages)
    assert cleaned[0] is sdk_message
