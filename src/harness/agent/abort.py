# harness/agent/abort.py
from __future__ import annotations

import asyncio


class AbortSignal:
    """纯机制,不认识 Ctrl+C——宿主自己决定什么触发它(信号处理器、
    HTTP 取消、超时定时器都行)。与 PermissionPolicy 同族:harness
    定义机制,宿主提供策略。

    宿主构造一个实例、传给 Agent.run()/events()/resume()/
    resume_events() 的 abort 参数,需要中断时调用 abort()。循环在
    几个检查点读它的状态(见 loop.py),发现已中断就走
    status="aborted" 的收口路径。

    不传(默认 None,见 RunContext.abort)= 永不中断,"不配置就不该
    有感"——这是本刀唯一新增的可选能力,不影响任何现有调用方。
    """

    def __init__(self):
        self._event = asyncio.Event()
        self._reason: str | None = None

    def abort(self, reason: str = "user_requested") -> None:
        """幂等:重复调用不报错,以最新一次给出的 reason 为准
        (正常使用只会调用一次,允许重复调用是为了不强迫调用方自己
        先查一遍"是不是已经中断过了")。"""
        self._reason = reason
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason