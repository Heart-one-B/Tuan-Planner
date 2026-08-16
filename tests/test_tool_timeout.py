# tests/test_tool_timeout.py
"""
验证 Phase 0 第三项修复:工具执行超时。

覆盖:
  1. 异步工具超时 → 转成【】前缀消息,不无限挂起
  2. 同步(阻塞)工具超时 → execute() 按时返回(即便底层线程仍在跑,
     这是已知的物理限制,测试验证的是"调用方不会被卡住"而非"线程被杀死")
  3. 单个工具的 timeout 覆盖全局默认值
  4. 未超时的正常工具不受影响
  5. tracer 记录的 status 为 "timeout"
"""
from __future__ import annotations

import asyncio
import sys
import time

from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor
from harness.tracing import tracer as tracer_module


class _FakeToolCall:
    """模拟 LLM 返回的 tool_call 对象。"""
    def __init__(self, name: str, arguments: dict):
        self.function = type("F", (), {"name": name, "arguments": __import__("json").dumps(arguments)})()
        self.id = "fake_call_id"


def _reset_tracer():
    tracer_module._active.clear()
    tracer_module._start_times.clear()
    tracer_module._storage = None


async def test_async_tool_timeout_returns_promptly():
    async def slow_async_tool(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "不该跑到这里"

    ex = ToolExecutor(default_timeout=0.2)
    ex.register(ToolDefinition(
        name="slow", description="慢工具",
        parameters={"seconds": {"type": "number"}}, required=["seconds"],
        func=slow_async_tool,
    ))

    start = time.time()
    result = await ex.execute(_FakeToolCall("slow", {"seconds": 5.0}))
    elapsed = time.time() - start

    assert "执行超时" in result, f"超时提示词不对: {result}"
    assert elapsed < 1.0, f"execute() 应该在超时阈值附近返回,实际耗时 {elapsed:.2f}s(卡住了)"
    print(f"✅ test_async_tool_timeout_returns_promptly 通过(耗时 {elapsed:.2f}s,未被 5s 的任务卡住)")


async def test_sync_blocking_tool_timeout_returns_promptly():
    def slow_sync_tool(seconds: float) -> str:
        time.sleep(seconds)   # 真阻塞,不是 await
        return "不该跑到这里"

    ex = ToolExecutor(default_timeout=0.2)
    ex.register(ToolDefinition(
        name="slow_sync", description="慢同步工具",
        parameters={"seconds": {"type": "number"}}, required=["seconds"],
        func=slow_sync_tool,
    ))

    start = time.time()
    result = await ex.execute(_FakeToolCall("slow_sync", {"seconds": 3.0}))
    elapsed = time.time() - start

    assert "执行超时" in result, f"超时提示词不对: {result}"
    # 关键断言:调用方(execute)必须按时返回,不能被同步阻塞拖住——
    # 即便底层线程池里那个 time.sleep(3.0) 还在后台悄悄跑完,这是已知限制
    assert elapsed < 1.0, f"execute() 被同步阻塞拖住了,耗时 {elapsed:.2f}s"
    print(f"✅ test_sync_blocking_tool_timeout_returns_promptly 通过"
          f"(耗时 {elapsed:.2f}s,调用方未被卡住;后台线程仍会跑完 3s,这是已知限制)")


async def test_per_tool_timeout_overrides_default():
    async def slow_async_tool(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "完成"

    ex = ToolExecutor(default_timeout=10.0)   # 全局默认给得很宽松
    ex.register(ToolDefinition(
        name="strict", description="有自己更严格超时的工具",
        parameters={"seconds": {"type": "number"}}, required=["seconds"],
        func=slow_async_tool,
        timeout=0.2,   # 单工具覆盖为更短
    ))

    start = time.time()
    result = await ex.execute(_FakeToolCall("strict", {"seconds": 2.0}))
    elapsed = time.time() - start

    assert "执行超时" in result, "单工具的 timeout 覆盖应该生效,但没有超时"
    assert elapsed < 1.0, f"应该按单工具的 0.2s 超时,而不是全局的 10s,实际耗时 {elapsed:.2f}s"
    print(f"✅ test_per_tool_timeout_overrides_default 通过(耗时 {elapsed:.2f}s,单工具覆盖生效)")


async def test_normal_tool_not_affected():
    async def fast_tool(x: str) -> str:
        return f"echo:{x}"

    ex = ToolExecutor(default_timeout=1.0)
    ex.register(ToolDefinition(
        name="fast", description="正常工具",
        parameters={"x": {"type": "string"}}, required=["x"], func=fast_tool,
    ))

    result = await ex.execute(_FakeToolCall("fast", {"x": "hi"}))
    assert result == "echo:hi", f"正常工具不该受超时机制影响,结果: {result}"
    print("✅ test_normal_tool_not_affected 通过")


async def test_timeout_recorded_in_tracer():
    _reset_tracer()

    async def slow_async_tool() -> str:
        await asyncio.sleep(5.0)
        return "x"

    ex = ToolExecutor(default_timeout=0.2)
    ex.register(ToolDefinition(
        name="slow", description="慢工具", parameters={}, required=[], func=slow_async_tool,
    ))

    tracer_module.start_trace("t1", "session1", "task1")
    await ex.execute(_FakeToolCall("slow", {}), trace_id="t1")

    # trace 已经 end 之前查不到(还在 _active 里),直接看 tool_events
    trace = tracer_module._active.get("t1")
    assert trace is not None
    assert len(trace.tool_events) == 1
    assert trace.tool_events[0].status == "timeout", \
        f"应该记录 status=timeout,实际 {trace.tool_events[0].status}"
    print("✅ test_timeout_recorded_in_tracer 通过")


async def main():
    tests = [
        test_async_tool_timeout_returns_promptly,
        test_sync_blocking_tool_timeout_returns_promptly,
        test_per_tool_timeout_overrides_default,
        test_normal_tool_not_affected,
        test_timeout_recorded_in_tracer,
    ]
    failed = 0
    for t in tests:
        try:
            await t()
        except AssertionError as e:
            failed += 1
            print(f"❌ {t.__name__} 失败: {e}")
        except Exception as e:
            failed += 1
            print(f"💥 {t.__name__} 抛出意外异常: {type(e).__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"总计 {len(tests)} 项, 失败 {failed} 项")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())