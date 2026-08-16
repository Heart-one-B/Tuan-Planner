# tests/agent/test_streaming.py
"""对应待做计划表(修订版)任务 5.1-5.9 的核心验证。

第一性问题:流式改造真正的风险不在"能不能吐 token",而在
①归并逻辑对不对(tool_calls 分片乱序/交错到达能不能正确拼回去)
②零回归是不是真的(旧测试一行不改还能不能过,不能只凭推理相信)
③静默升档在流式下的缓冲/补发时机对不对(这是本轮设计复盘时改了
两次才定下来的逻辑,必须专门测,不能只靠 code review 自信)。
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness.agent.abort import AbortSignal
from harness.agent.finish_tool import build_finish_tool
from harness.agent.loop import Budget, wrap_store, _call_model
from harness.agent.permission import AllowAllPolicy
from harness.agent.query import run_to_outcome
from harness.agent.run_context import RunContext
from harness.agent.state import LoopConfig, LoopState
from harness.llm.base import NormalizedUsage, StreamDelta
from harness.llm.streaming import StreamAccumulator
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor

from tests.loop.fakes import FakeLLMClient, RaisingLLMClient, StreamingFakeLLMClient


def _role(m) -> str | None:
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)


def _run_ctx() -> RunContext:
    return RunContext.begin("test-agent", "测试任务")


def _make_cfg(llm, executor, **overrides) -> LoopConfig:
    defaults = dict(
        llm=llm, tool_executor=executor, budget=Budget(),
        permission_policy=AllowAllPolicy(), tools=executor.schemas,
    )
    defaults.update(overrides)
    return LoopConfig(**defaults)


def _make_state(messages, run_ctx, **overrides) -> LoopState:
    return LoopState(store=wrap_store(messages, run_ctx), **overrides)


class _DummyData(BaseModel):
    answer: str


async def _search(query: str) -> str:
    return f"搜索结果:{query}"


def _executor_with_search_and_finish() -> ToolExecutor:
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="search", func=_search, description="", parameters={}))
    executor.register(build_finish_tool(_DummyData))
    return executor


# ── StreamAccumulator:核心归并逻辑 ───────────────────────────────────────

def test_accumulator_joins_content_across_multiple_deltas():
    acc = StreamAccumulator()
    for piece in ["这是", "一段", "被拆开的", "文本"]:
        acc.feed(StreamDelta(content=piece))
    msg = acc.build_message()
    assert msg.content == "这是一段被拆开的文本"


def test_accumulator_joins_reasoning_separately_from_content():
    acc = StreamAccumulator()
    acc.feed(StreamDelta(reasoning="思考A"))
    acc.feed(StreamDelta(content="输出B"))
    acc.feed(StreamDelta(reasoning="思考C"))
    msg = acc.build_message()
    assert msg.reasoning == "思考A思考C"
    assert msg.content == "输出B"


def test_accumulator_merges_single_tool_call_split_across_chunks():
    """真实场景:第一个 chunk 只带 id+name 的前几个字符,后续 chunk
    才逐步补全 arguments——这是任务 5.4 要测的核心场景。"""
    acc = StreamAccumulator()
    acc.feed(StreamDelta(tool_call_delta={"index": 0, "id": "call_1", "name": "search", "arguments": None}))
    acc.feed(StreamDelta(tool_call_delta={"index": 0, "id": None, "name": None, "arguments": '{"qu'}))
    acc.feed(StreamDelta(tool_call_delta={"index": 0, "id": None, "name": None, "arguments": 'ery":"x"}'}))
    msg = acc.build_message()

    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_1"
    assert msg.tool_calls[0].function.name == "search"
    assert msg.tool_calls[0].function.arguments == '{"query":"x"}'


def test_accumulator_merges_interleaved_multiple_tool_calls_by_index():
    """两个 tool_call 的分片交错到达(不是先把第一个发完再发第二个),
    必须按 index 分别归并,不能串到一起。"""
    acc = StreamAccumulator()
    acc.feed(StreamDelta(tool_call_delta={"index": 0, "id": "c0", "name": "a", "arguments": None}))
    acc.feed(StreamDelta(tool_call_delta={"index": 1, "id": "c1", "name": "b", "arguments": None}))
    acc.feed(StreamDelta(tool_call_delta={"index": 0, "arguments": '{"x":1}'}))
    acc.feed(StreamDelta(tool_call_delta={"index": 1, "arguments": '{"y":2}'}))
    msg = acc.build_message()

    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0].id == "c0"
    assert msg.tool_calls[0].function.arguments == '{"x":1}'
    assert msg.tool_calls[1].id == "c1"
    assert msg.tool_calls[1].function.arguments == '{"y":2}'


def test_accumulator_finish_reason_on_last_content_chunk():
    """真实 provider 形态一:finish_reason 和最后一条内容 chunk 同时到达。"""
    acc = StreamAccumulator()
    acc.feed(StreamDelta(content="答案"))
    acc.feed(StreamDelta(content="完毕", finish_reason="stop"))
    msg = acc.build_message()
    assert msg.finish_reason == "stop"
    assert msg.content == "答案完毕"


def test_accumulator_finish_reason_on_separate_trailing_chunk():
    """真实 provider 形态二:finish_reason 单独出现在一个 content/
    tool_call_delta 都是 None 的收尾 chunk 上。归并逻辑两种都要扛得住
    (任务 5.5 的要求)。"""
    acc = StreamAccumulator()
    acc.feed(StreamDelta(content="答案完毕"))
    acc.feed(StreamDelta(finish_reason="stop"))   # 单独的收尾 chunk
    msg = acc.build_message()
    assert msg.finish_reason == "stop"
    assert msg.content == "答案完毕"


def test_accumulator_usage_on_trailing_chunk_without_choices():
    """usage 常见于一个 choices 为空、只带 usage 的收尾 chunk——
    任务 5.5 的要求:跳过 choices 为空的 chunk,但仍要读它的 usage。"""
    acc = StreamAccumulator()
    acc.feed(StreamDelta(content="内容"))
    acc.feed(StreamDelta(usage=NormalizedUsage(prompt_tokens=42, completion_tokens=7, source="api_usage")))
    msg = acc.build_message()
    assert msg.usage.prompt_tokens == 42
    assert msg.usage.source == "api_usage"


def test_accumulator_usage_stays_none_when_never_fed():
    """从未收到过 usage 时,不伪造一个全零的 NormalizedUsage——
    None 让下游(TokenCounter)保持估算模式,而不是被一个假的 0
    污染成"看似有真实读数"。"""
    acc = StreamAccumulator()
    acc.feed(StreamDelta(content="内容"))
    msg = acc.build_message()
    assert msg.usage is None


# ── 零回归:默认退化实现的机制性证据 ──────────────────────────────────────

def test_fake_llm_client_does_not_override_stream():
    """结构性确认(不只是跑一遍旧测试全绿这种经验性证据):
    FakeLLMClient 类自己没有定义 stream 方法——它必然继承
    LLMClientBase.stream() 的默认退化实现。这条断言失败,意味着
    "旧测试零改动通过"这件事随时可能被一次不经意的改动破坏而不
    被立刻发现。"""
    assert "stream" not in FakeLLMClient.__dict__


def test_raising_llm_client_also_does_not_override_stream():
    """同理:RaisingLLMClient 只重写了 call(),没有重写 stream()——
    异常能在流式路径下原样抛出,靠的是默认实现内部调用 self.call()
    这条链路,不是巧合。"""
    assert "stream" not in RaisingLLMClient.__dict__


async def test_raising_llm_client_raises_through_default_stream_fallback():
    """真正走一遍 run_query_loop,确认异常确实会在流式路径下传播。"""
    llm = RaisingLLMClient(RuntimeError("boom"))
    cfg = _make_cfg(llm, ToolExecutor())
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    with pytest.raises(RuntimeError, match="boom"):
        await run_to_outcome(cfg, state, run_ctx)


# ── run_query_loop 真的实时转发,不是攒够了再一次性吐 ─────────────────────

async def test_plain_completion_streams_tokens_in_real_time_order():
    llm = StreamingFakeLLMClient([{"stream_deltas": [
        {"content": "第一"}, {"content": "第二"}, {"content": "第三", "finish_reason": "stop"},
    ]}])
    cfg = _make_cfg(llm, ToolExecutor())
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    events = []
    from harness.agent.query import query
    async for ev in query(cfg, state, run_ctx):
        events.append(ev)

    token_events = [e for e in events if e["type"] == "token"]
    assert [e["content"] for e in token_events] == ["第一", "第二", "第三"]
    outcome = next(e["outcome"] for e in events if e["type"] == "outcome")
    assert outcome.final_text == "第一第二第三"


async def test_content_alongside_tool_calls_now_streams_as_token_not_thinking():
    """第五刀的行为变更(如实测出来,不是只在注释里说):模型同时带
    content 和 tool_calls 时,content 现在走 token 事件,不再被归类
    成 thinking(round1-4 的启发式在流式场景下无法成立,见
    _delta_to_events 的说明)。"""
    llm = StreamingFakeLLMClient([
        {"stream_deltas": [
            {"content": "我先查一下"},
            {"tool_call_delta": {"index": 0, "id": "c1", "name": "search", "arguments": None}},
            {"tool_call_delta": {"index": 0, "arguments": '{"query":"x"}'}, "finish_reason": "tool_calls"},
        ]},
        {"stream_deltas": [{"content": "最终答案", "finish_reason": "stop"}]},
    ])
    executor = _executor_with_search_and_finish()
    cfg = _make_cfg(llm, executor)
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    events = []
    from harness.agent.query import query
    async for ev in query(cfg, state, run_ctx):
        events.append(ev)

    token_events = [e for e in events if e["type"] == "token"]
    thinking_events = [e for e in events if e["type"] == "thinking"]
    assert any(e["content"] == "我先查一下" for e in token_events)
    assert thinking_events == []


# ── 静默升档在流式下的缓冲/补发时机 ───────────────────────────────────────

async def test_streaming_silent_upgrade_buffers_first_call_and_forwards_second():
    """第一次调用被截断:第一次的 delta 不该被实时转发(缓冲丢弃),
    只有升档重试那次的 delta 才应该出现在事件流里。"""
    llm = StreamingFakeLLMClient([
        {"stream_deltas": [{"content": "被截断的内容"}, {"finish_reason": "length"}]},
        {"stream_deltas": [{"content": "升档后完整了"}, {"finish_reason": "stop"}]},
    ])
    cfg = _make_cfg(llm, ToolExecutor(),
                    budget=Budget(max_output_tokens=100, max_output_tokens_upgraded=1000))
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    events = []
    from harness.agent.query import query
    async for ev in query(cfg, state, run_ctx):
        events.append(ev)

    token_events = [e for e in events if e["type"] == "token"]
    # 第一次的"被截断的内容"完全没有出现在事件流里——被丢弃的缓冲
    assert [e["content"] for e in token_events] == ["升档后完整了"]
    assert llm.calls[0].get("max_tokens") == 100
    assert llm.calls[1].get("max_tokens") == 1000

    outcome = next(e["outcome"] for e in events if e["type"] == "outcome")
    assert outcome.final_text == "升档后完整了"
    assert outcome.output_truncation_count == 0
    assert outcome.output_upgraded is True


async def test_streaming_no_truncation_forwards_buffered_content_normally():
    """配置了升档,但第一次调用根本没被截断:缓冲的内容要原样补发
    出去,不能因为走了缓冲分支就把内容弄丢。"""
    llm = StreamingFakeLLMClient([
        {"stream_deltas": [{"content": "一次成功"}, {"finish_reason": "stop"}]},
    ])
    cfg = _make_cfg(llm, ToolExecutor(),
                    budget=Budget(max_output_tokens=100, max_output_tokens_upgraded=1000))
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    events = []
    from harness.agent.query import query
    async for ev in query(cfg, state, run_ctx):
        events.append(ev)

    token_events = [e for e in events if e["type"] == "token"]
    assert [e["content"] for e in token_events] == ["一次成功"]
    assert llm.call_count == 1   # 没有升档重试
    outcome = next(e["outcome"] for e in events if e["type"] == "outcome")
    assert outcome.output_upgraded is False


async def test_no_upgrade_configured_streams_in_real_time_no_buffering():
    """没配置升档(默认):完全不走缓冲分支,和普通流式路径一样实时
    转发——用一个只在"缓冲态"下才会出现次序偏差的场景反向验证。"""
    llm = StreamingFakeLLMClient([{"stream_deltas": [
        {"content": "A"}, {"content": "B"}, {"content": "C", "finish_reason": "length"},
    ]}, {"stream_deltas": [{"content": "D", "finish_reason": "stop"}]}])
    cfg = _make_cfg(llm, ToolExecutor())   # 默认 Budget,不配置升档
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    events = []
    from harness.agent.query import query
    async for ev in query(cfg, state, run_ctx):
        events.append(ev)

    token_events = [e for e in events if e["type"] == "token"]
    # 第一次调用的 A/B/C 全部实时转发了(不是升档场景,不缓冲),
    # 截断后走的是普通 nudge 恢复路径,不是升档重试
    assert [e["content"] for e in token_events] == ["A", "B", "C", "D"]
    outcome = next(e["outcome"] for e in events if e["type"] == "outcome")
    assert outcome.output_truncation_count == 1
    assert outcome.output_upgraded is False


# ── _exhaust 的 force_answer/force_finish 统一走新路径 ────────────────────

async def test_force_answer_streams_via_new_path_without_tools_kwarg():
    llm = StreamingFakeLLMClient([
        {"stream_deltas": [{"tool_call_delta": {"index": 0, "id": "c1", "name": "search", "arguments": '{"query":"q"}'}}, {"finish_reason": "tool_calls"}]},
        {"stream_deltas": [{"content": "强制作答的内容"}, {"finish_reason": "stop"}]},
    ])
    executor = _executor_with_search_and_finish()
    cfg = _make_cfg(llm, executor, budget=Budget(max_rounds=1, exhausted_action="force_answer"))
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    events = []
    from harness.agent.query import query
    async for ev in query(cfg, state, run_ctx):
        events.append(ev)

    token_events = [e for e in events if e["type"] == "token"]
    assert [e["content"] for e in token_events] == ["强制作答的内容"]
    assert "tools" not in llm.calls[1]   # force_answer 不该带 tools

    outcome = next(e["outcome"] for e in events if e["type"] == "outcome")
    assert outcome.status == "exhausted"
    assert outcome.final_text == "强制作答的内容"


async def test_force_finish_does_not_leak_intermediate_events():
    """force_finish 的模型调用中间事件(token/thinking)不该吐给
    宿主——延续第一刀就定下的"exhaust 场景保持事件面简洁"这条界限,
    第五刀换了内部实现方式,但这条界限不变。"""
    llm = StreamingFakeLLMClient([
        {"stream_deltas": [{"tool_call_delta": {"index": 0, "id": "c1", "name": "search", "arguments": '{"query":"q"}'}}, {"finish_reason": "tool_calls"}]},
        {"stream_deltas": [
            {"content": "不该被外部看到的中间文本"},
            {"tool_call_delta": {"index": 0, "id": "c2", "name": "finish",
                                 "arguments": '{"status":"ok","summary":"done","data":{"answer":"1"}}'}},
            {"finish_reason": "tool_calls"},
        ]},
    ])
    executor = _executor_with_search_and_finish()
    cfg = _make_cfg(llm, executor, budget=Budget(max_rounds=1, exhausted_action="force_finish"))
    run_ctx = _run_ctx()
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    events = []
    from harness.agent.query import query
    async for ev in query(cfg, state, run_ctx):
        events.append(ev)

    token_events = [e for e in events if e["type"] == "token"]
    assert not any("不该被外部看到" in e["content"] for e in token_events)
    outcome = next(e["outcome"] for e in events if e["type"] == "outcome")
    assert outcome.status == "exhausted"
    assert outcome.result is not None
    assert outcome.result.data == {"answer": "1"}


# ── 中断检查点②:流式消费循环内(第四刀承诺补上,第五刀落地) ──────────────

async def test_abort_checkpoint_mid_stream_in_main_loop():
    """中断信号在流式消费期间(还没收完这次响应)才被设置——检查点②
    要能在这里截获,产出 aborted_streaming,而不是等这一轮彻底结束
    (走到工具批次)才有机会检查。"""
    from harness.agent.abort import AbortSignal

    sig = AbortSignal()

    class _AbortAfterFirstDelta(StreamingFakeLLMClient):
        async def stream(self, trace_id, **kwargs):
            self.calls.append(kwargs)
            spec = self.responses[self.call_count]
            self.call_count += 1
            for i, d in enumerate(spec["stream_deltas"]):
                yield StreamDelta(**d)
                if i == 0:
                    # 在第一个 delta 被 _call_model 消费之后才设置中断——
                    # 这样才能正确模拟"处理完第一个、第二个被检查点拦下",
                    # 而不是"中断信号在第一个都还没处理时就已经生效"。
                    sig.abort(reason="流式消费到一半取消")

    llm = _AbortAfterFirstDelta([{"stream_deltas": [
        {"content": "第一段"}, {"content": "第二段,不该被看到"}, {"finish_reason": "stop"},
    ]}])
    cfg = _make_cfg(llm, ToolExecutor())
    run_ctx = RunContext.begin("test-agent", "测试任务", abort=sig)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    events = []
    from harness.agent.query import query
    async for ev in query(cfg, state, run_ctx):
        events.append(ev)

    token_events = [e for e in events if e["type"] == "token"]
    # 只有第一个 delta("第一段")在检查点发现中断之前被 yield 过,
    # 后续的都被检查点拦下——检查点在"处理这个 delta 之前"检查,
    # 所以第一个 delta 本身(信号是它自己设置的,设置之后才轮到
    # 下一次循环迭代去检查)仍然会被正常处理一次。
    assert token_events == [{"type": "token", "content": "第一段"}]
    outcome = next(e["outcome"] for e in events if e["type"] == "outcome")
    assert outcome.status == "aborted"
    assert outcome.reason == "aborted_streaming"
    assert not any(_role(m) == "tool" for m in outcome.messages)  # 干净,不需要修复


async def test_abort_checkpoint_mid_stream_in_force_answer():
    """同样的检查点在 _exhaust 的 force_answer 路径里也要生效——
    它复用的是同一个 _call_model,理应免费获得这个能力。"""
    from harness.agent.abort import AbortSignal

    sig = AbortSignal()

    class _AbortMidStream(StreamingFakeLLMClient):
        async def stream(self, trace_id, **kwargs):
            self.calls.append(kwargs)
            spec = self.responses[self.call_count]
            self.call_count += 1
            for i, d in enumerate(spec["stream_deltas"]):
                if i == 1:
                    sig.abort(reason="强制作答期间取消")
                yield StreamDelta(**d)

    llm = _AbortMidStream([
        {"stream_deltas": [{"tool_call_delta": {"index": 0, "id": "c1", "name": "search", "arguments": '{"query":"q"}'}}, {"finish_reason": "tool_calls"}]},
        {"stream_deltas": [{"content": "A"}, {"content": "B"}, {"content": "C", "finish_reason": "stop"}]},
    ])
    executor = _executor_with_search_and_finish()
    cfg = _make_cfg(llm, executor, budget=Budget(max_rounds=1, exhausted_action="force_answer"))
    run_ctx = RunContext.begin("t", "task", abort=sig)
    state = _make_state([{"role": "user", "content": "问题"}], run_ctx)

    events = []
    from harness.agent.query import query
    async for ev in query(cfg, state, run_ctx):
        events.append(ev)

    outcome = next(e["outcome"] for e in events if e["type"] == "outcome")
    assert outcome.status == "aborted"
    assert outcome.reason == "aborted_streaming"