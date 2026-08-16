# tests/agent/fakes.py
from __future__ import annotations

import json
from types import SimpleNamespace

from harness.llm.base import LLMClientBase, NormalizedUsage, StreamDelta


def _build_tool_call(spec: dict):
    args = spec.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return SimpleNamespace(
        id=spec.get("id", "call_0"),
        function=SimpleNamespace(name=spec["name"], arguments=args),
    )


def _build_message(spec: dict):
    tool_calls_spec = spec.get("tool_calls") or []
    tool_calls = [_build_tool_call(tc) for tc in tool_calls_spec] or None
    finish_reason = spec.get("finish_reason")
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"
    msg = SimpleNamespace(
        role="assistant",   # 真实 SDK 消息对象总有 .role="assistant";
                            # _find_pending_call_context 依赖它定位挂起点
        content=spec.get("content"),
        tool_calls=tool_calls,
        reasoning=spec.get("reasoning"),
        finish_reason=finish_reason,
        usage=NormalizedUsage(
            prompt_tokens=spec.get("prompt_tokens", 10),
            completion_tokens=spec.get("completion_tokens", 5),
            source="api_usage",
        ),
    )
    return msg


class FakeLLMClient(LLMClientBase):
    """按脚本顺序回放响应。responses 用完后再调用会抛异常
    (超出脚本长度的调用是测试设计错误的信号,不该静默返回垃圾数据)。

    stream=True 时(仅 _exhaust 的 force_answer 路径会用到)按同一条
    脚本的 content 字段逐字符切成 chunk 吐出,模拟真实流式分片,
    包含"最后一个 chunk choices 为空"的边界情况(Phase 1 的真实坑)。
    """

    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.call_count = 0
        self.calls: list[dict] = []  # 记录每次调用的 messages/tools,供测试断言

    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        self.calls.append(kwargs)
        if self.call_count >= len(self.responses):
            raise AssertionError(
                f"FakeLLMClient 脚本已耗尽(第 {self.call_count + 1} 次调用,"
                f"只准备了 {len(self.responses)} 条响应)——多半是被测代码"
                f"多调用了模型一次,检查终止判定逻辑。"
            )
        spec = self.responses[self.call_count]
        self.call_count += 1

        if stream:
            return self._fake_stream(spec)

        msg = _build_message(spec)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=msg.finish_reason)])

    @staticmethod
    async def _fake_stream(spec: dict):
        text = spec.get("content") or ""
        for ch in text:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))])
        # 模拟真实 API 常见的"最后一个 chunk choices 为空"边界
        yield SimpleNamespace(choices=[])


class RaisingLLMClient(FakeLLMClient):
    """第三刀新增:每次 call() 都直接抛出指定异常,用于测试第三层
    (query.py)的异常收尾路径、Agent 的 on_error_snapshot 钩子。

    第五刀说明:这个类没有单独的"流式变体"——它只重写了 call(),
    没有重写 stream()。循环改走 stream() 之后,它会自动继承
    LLMClientBase.stream() 的默认实现,而那个默认实现内部就是
    调用 self.call(),所以异常照样会在 stream() 路径下原样抛出,
    不需要额外代码。这不是巧合,是默认退化实现设计对了的直接证据——
    test_abort.py/test_state_and_layers.py 里用它测的异常路径,
    第五刀之后依然一行不改地成立(见 test_streaming.py 的验证)。
    """

    def __init__(self, exc: Exception):
        super().__init__([])
        self._exc = exc

    async def call(self, trace_id: str, stream: bool = False, **kwargs) -> object:
        self.calls.append(kwargs)
        raise self._exc


class StreamingFakeLLMClient(FakeLLMClient):
    """第五刀新增(任务 5.3 的反面):FakeLLMClient 的流式变体,
    只有测试真正需要模拟"边收边吐"行为(逐 chunk、tool_calls 按
    index 分片到达、finish_reason 落在不同位置)时才用这个类。

    刻意不让 FakeLLMClient 本身重写 stream()——第一到第四刀写的
    全部 93 条测试用的都是普通 FakeLLMClient,循环改走 stream()
    之后,它们会自动落进 LLMClientBase.stream() 的默认退化实现,
    不需要改一行就能继续通过。这是"新能力可选、不配置即无感"
    这条老规矩在测试基础设施层面的验证方式,不是偷懒少写一个类。

    responses 里每条 spec 必须显式提供 "stream_deltas": list[dict],
    每个 dict 的 key 对应 StreamDelta 的字段名,原样构造成
    StreamDelta 逐个 yield——故意不做"自动从 content/tool_calls
    生成 chunk 序列"这种便利封装,因为这里恰恰是要精确控制切片
    方式本身(任务 5.4 的按 index 归并逻辑要测的就是这个),自动
    生成会掩盖真实的分片行为,测不出归并代码的 bug。
    """

    async def stream(self, trace_id: str, **kwargs):
        self.calls.append(kwargs)
        if self.call_count >= len(self.responses):
            raise AssertionError(
                f"StreamingFakeLLMClient 脚本已耗尽(第 {self.call_count + 1} 次调用,"
                f"只准备了 {len(self.responses)} 条响应)"
            )
        spec = self.responses[self.call_count]
        self.call_count += 1
        if "stream_deltas" not in spec:
            raise AssertionError(
                f"StreamingFakeLLMClient 的第 {self.call_count} 条响应缺少 "
                f"'stream_deltas' 字段——这个类不做自动分片,必须显式提供。"
            )
        for d in spec["stream_deltas"]:
            yield StreamDelta(**d)