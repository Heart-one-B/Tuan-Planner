# harness/agent/run_context.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.tracing.span import Span


@dataclass
class RunContext:
    """一次 Agent 运行的全部运行期状态的唯一容器。

    与 Claude Code 的模块划分同构:其 Rust 重写版把 session(运行状态)
    和 loop、permissions、config、prompt 放在同一个 runtime/ 模块——
    循环本体和循环直接读写的运行期状态天然是一伙的,不应拆开。

    解决两个问题:
      1. 并发安全 —— 状态不再挂在 Agent 实例的 self 上,每次 run() 一个
         RunContext,同一 Agent 实例并发跑互不串台。
      2. 状态有处安放 —— 编排型工具(读写运行期状态的工具)通过
         run_ctx 参数拿到 state,依赖显式化,可以搬出 Agent 类独立成文件。

    Phase 2 的 ContextManager、Phase 3 的快照(持久化的就是整个RunContext)都挂载在这里,这是它们的预留接口。
    """

    span: Span
    state: dict[str, Any] = field(default_factory=dict)
    # context_manager: "ContextManager"   # ← Phase 2 挂载点(预留,勿删注释)

    @classmethod
    def begin(cls, name: str, task: str, parent: Span | None = None) -> "RunContext":
        return cls(span=Span.begin(name, task, parent=parent))

    def child(self, name: str, task: str) -> "RunContext":
        """派生子运行:新 span 挂在当前 span 下,state 全新(子运行状态隔离)。"""
        return RunContext(span=self.span.child(name, task))