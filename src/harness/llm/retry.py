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