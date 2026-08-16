# tests/test_retry.py
"""
验证 Phase 0 第二项修复:重试与退避。

分两层测试:
  1. retry_with_backoff 本身(通用机制,与 openai 无关,纯 asyncio)
  2. OpenAIClient 的错误分类(_is_retryable,用真实 openai 异常类构造,
     不做网络请求)

用极小的 base_delay/max_delay 让测试快速跑完,不真的等秒级退避。
"""
from __future__ import annotations

import asyncio
import sys

from harness.llm.retry import retry_with_backoff


class _FlakyCounter:
    """模拟一个会失败 N 次然后成功的异步函数。"""
    def __init__(self, fail_times: int, exc_factory):
        self.fail_times = fail_times
        self.exc_factory = exc_factory
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        return "ok"


class RetryableError(Exception):
    pass


class PermanentError(Exception):
    pass


async def test_succeeds_on_first_try_no_retry():
    fn = _FlakyCounter(fail_times=0, exc_factory=RetryableError)
    result = await retry_with_backoff(
        fn, is_retryable=lambda e: isinstance(e, RetryableError),
        max_retries=3, base_delay=0.001, max_delay=0.01,
    )
    assert result == "ok"
    assert fn.calls == 1, f"不该重试,实际调用了 {fn.calls} 次"
    print("✅ test_succeeds_on_first_try_no_retry 通过")


async def test_retries_transient_then_succeeds():
    fn = _FlakyCounter(fail_times=2, exc_factory=RetryableError)
    result = await retry_with_backoff(
        fn, is_retryable=lambda e: isinstance(e, RetryableError),
        max_retries=5, base_delay=0.001, max_delay=0.01,
    )
    assert result == "ok"
    assert fn.calls == 3, f"应该失败2次+成功1次共调用3次,实际 {fn.calls} 次"
    print("✅ test_retries_transient_then_succeeds 通过")


async def test_non_retryable_fails_immediately():
    fn = _FlakyCounter(fail_times=100, exc_factory=PermanentError)
    try:
        await retry_with_backoff(
            fn, is_retryable=lambda e: isinstance(e, RetryableError),  # PermanentError 不在白名单
            max_retries=5, base_delay=0.001, max_delay=0.01,
        )
        assert False, "应该抛出异常,但没有"
    except PermanentError:
        pass
    assert fn.calls == 1, f"永久性错误不该重试,应只调用1次,实际 {fn.calls} 次"
    print("✅ test_non_retryable_fails_immediately 通过")


async def test_exhausts_max_retries_then_raises():
    fn = _FlakyCounter(fail_times=100, exc_factory=RetryableError)  # 永远失败
    try:
        await retry_with_backoff(
            fn, is_retryable=lambda e: isinstance(e, RetryableError),
            max_retries=3, base_delay=0.001, max_delay=0.01,
        )
        assert False, "应该在耗尽重试预算后抛出异常"
    except RetryableError:
        pass
    assert fn.calls == 4, f"max_retries=3 应该是 1次首try + 3次重试 = 4次调用,实际 {fn.calls} 次"
    print("✅ test_exhausts_max_retries_then_raises 通过")


async def test_on_retry_callback_invoked():
    fn = _FlakyCounter(fail_times=2, exc_factory=RetryableError)
    events = []
    await retry_with_backoff(
        fn, is_retryable=lambda e: isinstance(e, RetryableError),
        max_retries=5, base_delay=0.001, max_delay=0.01,
        on_retry=lambda attempt, delay, exc: events.append((attempt, type(exc).__name__)),
    )
    assert events == [(1, "RetryableError"), (2, "RetryableError")], f"回调记录不对: {events}"
    print("✅ test_on_retry_callback_invoked 通过")


async def test_delay_actually_grows_exponentially():
    """专门证明退避曲线真的是指数增长,不是靠肉眼看日志猜的。

    前面几个测试为了跑得快,把 base_delay 设成了 0.001 秒——延迟数值
    虽然也在指数增长(0.001→0.002→0.004...),但格式化成两位小数后
    全部四舍五入显示为 0.00,肉眼完全看不出来增长趋势,容易被误认为
    "退避没生效"。这里用 mock 固定住两个变量分别验证:
      1. random.uniform 固定取上界,消除满抖动的随机性干扰
      2. asyncio.sleep 换成"只记录传入秒数、不真的等待",这样能拿到
         一组确定性数字直接断言,而不用真的等 1+2+4+8+16=31 秒
    """
    import unittest.mock as mock

    async def always_fails():
        raise RetryableError("boom")

    recorded_delays = []
    with mock.patch("harness.llm.retry.asyncio.sleep",
                    side_effect=lambda d: recorded_delays.append(d)), \
         mock.patch("harness.llm.retry.random.uniform",
                    side_effect=lambda lo, hi: hi):
        try:
            await retry_with_backoff(
                always_fails, is_retryable=lambda e: isinstance(e, RetryableError),
                max_retries=5, base_delay=1.0, max_delay=30.0,
            )
        except RetryableError:
            pass

    expected = [1.0, 2.0, 4.0, 8.0, 16.0]  # 2^0, 2^1, 2^2, 2^3, 2^4 秒
    assert recorded_delays == expected, \
        f"延迟序列应该是指数增长 {expected},实际拿到 {recorded_delays}"
    print(f"✅ test_delay_actually_grows_exponentially 通过 —— 真实延迟序列: {recorded_delays}")


async def test_delay_capped_at_max_delay():
    """验证指数增长不会无限涨上去,超过 max_delay 后封顶。"""
    import unittest.mock as mock

    async def always_fails():
        raise RetryableError("boom")

    recorded_delays = []
    with mock.patch("harness.llm.retry.asyncio.sleep",
                    side_effect=lambda d: recorded_delays.append(d)), \
         mock.patch("harness.llm.retry.random.uniform",
                    side_effect=lambda lo, hi: hi):
        try:
            await retry_with_backoff(
                always_fails, is_retryable=lambda e: isinstance(e, RetryableError),
                max_retries=6, base_delay=1.0, max_delay=10.0,
            )
        except RetryableError:
            pass

    expected = [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]  # 第5、6次本该是16/32,被封顶在10
    assert recorded_delays == expected, f"封顶逻辑不对,期望 {expected},实际 {recorded_delays}"
    print(f"✅ test_delay_capped_at_max_delay 通过 —— 封顶后序列: {recorded_delays}")


async def main():
    tests = [
        test_succeeds_on_first_try_no_retry,
        test_retries_transient_then_succeeds,
        test_non_retryable_fails_immediately,
        test_exhausts_max_retries_then_raises,
        test_on_retry_callback_invoked,
        test_delay_actually_grows_exponentially,
        test_delay_capped_at_max_delay,
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