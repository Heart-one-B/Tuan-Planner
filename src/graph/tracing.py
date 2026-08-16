# src/graph/tracing.py
"""graph 层的 span 辅助。

【为什么 state 里存字符串而不是 Span 对象】
LangGraph 的 state 在配了 checkpointer 时要能序列化，塞一个 Span
实例进去会在落盘那一步炸掉——而且是"平时开发不配 checkpointer 都
好好的，一上恢复场景就崩"的那种延迟暴露。存 trace_id 字符串则
两种配置下都安全。

Span 本身是个很薄的对象（__init__ 只存 trace_id + name，真正的
副作用在 Span.begin() 里的 tracer.start_trace），所以从一个
trace_id 重建"父 span 的壳"是完全安全的：它只被用来给子 span 提供
parent_trace_id，不会重复注册一条 trace。

【为什么需要根 span】
每个节点各开各的 root span 也能跑，但那样 trace 是一片互不相连的
平地：session_cost() 靠 parent_trace_id 递归聚合整棵树，树散了，
"一次用户请求到底花了多少 token"就永远算不出来——而这正是子 Agent
开销最容易被低估的地方（提取、召回、压缩都不出现在用户可见的
对话里，但都是真金白银）。
"""
from __future__ import annotations

from harness.tracing.span import Span

ROOT_TRACE_ID_KEY = "root_trace_id"


def begin_request_span(user_input: str) -> Span:
    """在调用 workflow.ainvoke() 之前开一次，一次用户请求一棵树。

    刻意不放在 routing_node 里：LangGraph 没有"整图结束"的钩子，
    span 在节点里开就没有可靠的地方关，容易漏掉某条终止路径导致
    trace 永远停在 running 状态（Tracer.active_count 会一直涨，
    那是泄漏）。放在 invoke 的外面，开和关天然对齐一次请求。
    """
    return Span.begin("plan-request", user_input, parent=None)


def parent_span_of(state) -> Span | None:
    """节点里取父 span。state 里没有 root_trace_id 时返回 None——
    此时子 Agent 各自开独立的根 trace，功能完全正常，只是 trace 树
    是散的。降级而不是报错：tracing 是观测设施，缺了它业务照跑，
    这和 Tracer._safe_call 只记日志不上抛是同一条原则。
    """
    trace_id = (state or {}).get(ROOT_TRACE_ID_KEY)
    if not trace_id:
        return None
    return Span(trace_id=trace_id, name="plan-request")