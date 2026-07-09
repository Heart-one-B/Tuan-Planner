# harness/llm/retry.py
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    is_retryable: Callable[[Exception], bool],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    """通用指数退避重试(满抖动)。

    与具体 provider 无关——"怎么退避"是不变的机制,"什么错误值得重试"
    是随 provider 变化的策略(is_retryable 参数注入)。这和 TerminationPolicy /
    DagScheduler 的 gate 是同一个设计原则的又一次应用:机制与策略分离。

    满抖动(uniform(0, delay))而非固定指数退避,避免多个并发调用方
    在同一个时间点集中重试造成惊群效应。

    耗尽 max_retries 后原样抛出最后一次的异常,不吞、不转换——
    调用方(AgentLoop)已有统一的异常处理路径(捕获→关闭 span→re-raise),
    这里不需要,也不应该重新设计一套错误处理。
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as e:
            if not is_retryable(e) or attempt >= max_retries:
                if attempt > 0:
                    logger.error(
                        f"[retry] 重试 {attempt} 次后仍失败,放弃: {type(e).__name__}: {e}"
                    )
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            delay = random.uniform(0, delay)
            attempt += 1
            if on_retry:
                on_retry(attempt, delay, e)
            else:
                logger.warning(
                    f"[retry] 第 {attempt}/{max_retries} 次重试,"
                    f"{delay:.2f}s 后重试,原因: {type(e).__name__}: {e}"
                )
            await asyncio.sleep(delay)