# tests/agent/test_openai_stream_shaping.py
"""OpenAIClient.stream() 的 chunk → StreamDelta 转换逻辑测试。

为什么单独一个文件:这段逻辑处在"真实 SDK 对象形状"和"harness 归一化
契约"的交界处,test_streaming.py 用的 StreamingFakeLLMClient 是直接
构造 StreamDelta 的(它测的是 StreamDelta 之后的链路),压根不经过
这段转换代码——也就是说,113 条测试全绿的时候,这段代码的覆盖率是 0。

这里用假的 SDK chunk 对象(SimpleNamespace 模拟
chunk.choices[0].delta.tool_calls 的形状)驱动真实的
OpenAIClient.stream(),不需要网络。覆盖不到的只剩"真实 provider 到底
发什么形状的 chunk"——那个只能靠真实冒烟,但至少"给定这个形状,
转换得对不对"现在有测试了。

核心场景:一个 chunk 里携带**多个** tool_call 分片。原实现写的是
`tc = tcs[0]`,注释说"每个 chunk 通常只带一个"——并行工具调用时这个
"通常"不成立,取 [0] 会静默丢掉其余分片。这条测试就是钉死这个修复。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.llm.base import StreamDelta
from harness.llm.streaming import StreamAccumulator


def _make_client():
    """构造一个 OpenAIClient,但把底层 SDK 调用替换掉——我们只想测
    chunk → StreamDelta 的转换,不想碰网络。"""
    from harness.llm.openai_client import OpenAIClient
    return OpenAIClient(api_key="fake", base_url="https://example.invalid",
                        model_name="fake-model")


def _tc_fragment(index: int, tc_id=None, name=None, arguments=None):
    """模拟 SDK 的 ChoiceDeltaToolCall 对象形状。"""
    return SimpleNamespace(
        index=index, id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


def _usage_only_chunk(prompt_tokens: int, completion_tokens: int):
    """模拟 stream_options={"include_usage":True} 的收尾 chunk:
    choices 为空,只带 usage。"""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens),
    )


async def _drive(client, chunks) -> list[StreamDelta]:
    """把一串假 chunk 喂给 client.stream(),收集产出的 StreamDelta。
    通过替换 retry_with_backoff 返回的对象绕开网络。"""
    class _FakeStream:
        def __aiter__(self):
            async def gen():
                for c in chunks:
                    yield c
            return gen()

    import harness.llm.openai_client as mod
    original = mod.retry_with_backoff

    async def _fake_retry(fn, **kwargs):
        return _FakeStream()

    mod.retry_with_backoff = _fake_retry
    try:
        return [d async for d in client.stream(trace_id="t1", messages=[])]
    finally:
        mod.retry_with_backoff = original


# ── 核心场景:一个 chunk 多个 tool_call 分片(修复前会丢数据) ─────────────

async def test_multiple_tool_call_fragments_in_one_chunk_all_survive():
    client = _make_client()
    chunks = [
        _chunk(tool_calls=[
            _tc_fragment(0, tc_id="c0", name="weather"),
            _tc_fragment(1, tc_id="c1", name="stock"),   # 修复前这个会被丢掉
        ]),
        _chunk(tool_calls=[
            _tc_fragment(0, arguments='{"city":"北京"}'),
            _tc_fragment(1, arguments='{"code":"600519"}'),
        ], finish_reason="tool_calls"),
    ]

    deltas = await _drive(client, chunks)

    acc = StreamAccumulator()
    for d in deltas:
        acc.feed(d)
    msg = acc.build_message()

    assert len(msg.tool_calls) == 2, "并行工具调用的第二个不该被静默丢掉"
    assert msg.tool_calls[0].id == "c0"
    assert msg.tool_calls[0].function.name == "weather"
    assert msg.tool_calls[0].function.arguments == '{"city":"北京"}'
    assert msg.tool_calls[1].id == "c1"
    assert msg.tool_calls[1].function.name == "stock"
    assert msg.tool_calls[1].function.arguments == '{"code":"600519"}'
    assert msg.finish_reason == "tool_calls"


async def test_finish_reason_not_duplicated_across_fragments():
    """一个 chunk 里有 N 个分片时,finish_reason 只该被带一次
    (挂在最后一个分片上),不能每个分片都带一遍——虽然
    StreamAccumulator 是覆盖赋值、重复了也不会错,但"每样信息恰好
    被消费一次"是这段转换代码该守的契约,不能靠下游宽容来兜底。"""
    client = _make_client()
    chunks = [_chunk(
        tool_calls=[_tc_fragment(0, tc_id="c0", name="a"),
                    _tc_fragment(1, tc_id="c1", name="b")],
        finish_reason="tool_calls",
    )]

    deltas = await _drive(client, chunks)

    with_finish = [d for d in deltas if d.finish_reason is not None]
    assert len(with_finish) == 1
    assert with_finish[-1].tool_call_delta["index"] == 1   # 挂在最后一个上


async def test_content_not_duplicated_across_fragments():
    """同理:content 只该挂在第一个分片上,不能每个分片重复一遍——
    重复的话 StreamAccumulator 会把它 join 两次,文本直接翻倍。"""
    client = _make_client()
    chunks = [_chunk(
        content="我来查一下",
        tool_calls=[_tc_fragment(0, tc_id="c0", name="a"),
                    _tc_fragment(1, tc_id="c1", name="b")],
    )]

    deltas = await _drive(client, chunks)

    acc = StreamAccumulator()
    for d in deltas:
        acc.feed(d)
    assert acc.build_message().content == "我来查一下"   # 不是"我来查一下我来查一下"


# ── 常规场景 ─────────────────────────────────────────────────────────────

async def test_single_tool_call_split_across_chunks():
    client = _make_client()
    chunks = [
        _chunk(tool_calls=[_tc_fragment(0, tc_id="c0", name="search")]),
        _chunk(tool_calls=[_tc_fragment(0, arguments='{"q":')]),
        _chunk(tool_calls=[_tc_fragment(0, arguments='"x"}')], finish_reason="tool_calls"),
    ]

    deltas = await _drive(client, chunks)
    acc = StreamAccumulator()
    for d in deltas:
        acc.feed(d)
    msg = acc.build_message()

    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.arguments == '{"q":"x"}'


async def test_pure_text_stream_has_no_tool_calls():
    client = _make_client()
    chunks = [
        _chunk(content="第一段"),
        _chunk(content="第二段", finish_reason="stop"),
    ]

    deltas = await _drive(client, chunks)
    acc = StreamAccumulator()
    for d in deltas:
        acc.feed(d)
    msg = acc.build_message()

    assert msg.content == "第一段第二段"
    assert msg.tool_calls is None
    assert msg.finish_reason == "stop"


async def test_usage_only_trailing_chunk_is_captured_not_skipped():
    """Phase 1 真实踩过的坑在流式路径上的对应版本:最后一个 chunk
    choices 为空,不能因为"没有 choices"就整个跳过——它带着 usage,
    而 usage 直接决定 TokenCounter 走真实读数还是估算(进而决定
    压缩触发时机是否系统性偏早)。"""
    client = _make_client()
    chunks = [
        _chunk(content="内容", finish_reason="stop"),
        _usage_only_chunk(prompt_tokens=123, completion_tokens=45),
    ]

    deltas = await _drive(client, chunks)
    acc = StreamAccumulator()
    for d in deltas:
        acc.feed(d)
    msg = acc.build_message()

    assert msg.usage is not None
    assert msg.usage.prompt_tokens == 123
    assert msg.usage.source == "api_usage"