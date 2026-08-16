# tests/test_message_id.py
from types import SimpleNamespace

from harness.message_id import find_index_by_hid, get_hid, new_hid, strip_hid, tag_message


def test_tag_message_adds_unique_hid_without_mutating_input():
    original = {"role": "user", "content": "你好"}
    tagged = tag_message(original)
    assert "_hid" not in original  # 不原地修改入参
    assert "_hid" in tagged
    assert tagged["role"] == "user"
    assert tagged["content"] == "你好"


def test_new_hid_is_unique():
    assert new_hid() != new_hid()


def test_get_hid_dict_form():
    tagged = tag_message({"role": "user", "content": "x"})
    assert get_hid(tagged) == tagged["_hid"]


def test_get_hid_dict_without_hid_returns_none():
    assert get_hid({"role": "assistant", "content": "x"}) is None


def test_get_hid_sdk_object_returns_none():
    obj = SimpleNamespace(role="assistant", content="x")
    assert get_hid(obj) is None


def test_find_index_by_hid_locates_correct_message():
    m1 = tag_message({"role": "user", "content": "第一条"})
    m2 = {"role": "assistant", "content": "回复"}
    m3 = tag_message({"role": "user", "content": "第三条"})
    messages = [m1, m2, m3]
    assert find_index_by_hid(messages, m1["_hid"]) == 0
    assert find_index_by_hid(messages, m3["_hid"]) == 2


def test_find_index_by_hid_not_found_returns_none():
    messages = [tag_message({"role": "user", "content": "x"})]
    assert find_index_by_hid(messages, "不存在的hid") is None


def test_find_index_by_hid_none_input_returns_none():
    """防御性处理:hid=None 不该意外匹配到某条同样没有 _hid 的消息
    (get_hid 对那类消息也返回 None)——那会是一次静默的错误匹配。"""
    messages = [{"role": "assistant", "content": "没有 hid 的消息"}]
    assert find_index_by_hid(messages, None) is None


def test_strip_hid_removes_key():
    tagged = tag_message({"role": "user", "content": "x"})
    stripped = strip_hid(tagged)
    assert "_hid" not in stripped
    assert stripped["content"] == "x"
    assert "_hid" in tagged  # 不原地修改入参


def test_strip_hid_passthrough_when_absent():
    plain = {"role": "assistant", "content": "x"}
    assert strip_hid(plain) is plain  # 没有 _hid 时原样返回,不做多余拷贝
