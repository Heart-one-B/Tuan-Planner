# harness/tracing/span.py
from __future__ import annotations

import uuid

from harness.tracing import tracer


class Span:
    """span 树的最小实现:每个 span 落库为一条独立 trace,用 parent_trace_id 串联。
    职责边界:Span 全部工作是"上报",不参与循环的执行逻辑——
    去掉它循环依然能转,只是没人知道发生了什么。

    这是对扁平 tracer 的止血方案——修复"嵌套 Agent 复用同一 trace_id
    导致父 trace 被覆盖丢失"的问题。接口按 OTel 父子语义设计
    (begin / child / end)
    """

    def __init__(self, trace_id: str, name: str):
        self.trace_id = trace_id
        self.name = name
        self._ended = False

    @classmethod
    def begin(cls, name: str, task: str, parent: "Span | None" = None) -> "Span":
        trace_id = str(uuid.uuid4())
        tracer.start_trace(
            trace_id,
            name,
            task,
            parent_trace_id=parent.trace_id if parent else None,
        )
        return cls(trace_id, name)

    def child(self, name: str, task: str) -> "Span":
        """派生子 span:子 Agent / 子任务用这个,而不是复用父 trace_id。"""
        return Span.begin(name, task, parent=self)

    def end(self, summary: str, status: str = "success") -> None:
        # 幂等:重复 end 不报错,方便 finally 兜底
        if self._ended:
            return
        self._ended = True
        tracer.end_trace(self.trace_id, summary, status=status)