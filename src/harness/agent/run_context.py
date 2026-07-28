# harness/agent/run_context.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.agent.abort import AbortSignal
from harness.tracing.span import Span


@dataclass
class RunContext:
    """一次 Agent 运行的全部运行期状态的唯一容器。

    【第四刀改动,任务 4.6】abort 字段落在这里,不在 LoopState——
    第二/三刀曾经在 LoopState 上留过一个同名占位字段(类型 Any,
    没有任何代码消费),本刀实现真正的检查点时发现放错了地方:
    RunContext 是工具执行(ToolExecutor.execute(..., run_ctx=run_ctx))
    和权限检查(PermissionPolicy.check(..., run_ctx))都能摸到的对象,
    "这次运行有没有被要求中断"是这两处都可能需要感知的信息,不该
    锁在只有 loop.py 内部流转的 LoopState 里。已把 LoopState 那个
    占位字段删掉。

    默认 None = 永不中断,"不配置就不该有感"。
    """

    span: Span
    state: dict[str, Any] = field(default_factory=dict)
    context_manager: "ContextManager | None" = None
    abort: AbortSignal | None = None

    @classmethod
    def begin(cls, name: str, task: str, parent: Span | None = None,
             abort: AbortSignal | None = None) -> "RunContext":
        return cls(span=Span.begin(name, task, parent=parent), abort=abort)

    def child(self, name: str, task: str) -> "RunContext":
        """派生子运行:新 span 挂在当前 span 下,state 全新(子运行状态隔离)。
        abort 不自动传给子运行——子 Agent(如提取 Agent)是否该跟着
        主运行一起中断是一个策略问题,不是机制问题,这里不替调用方
        决定,需要的话调用方自己在 child() 之后手动设置。"""
        return RunContext(span=self.span.child(name, task))