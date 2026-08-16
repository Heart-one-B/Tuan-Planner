# harness/tracing/tracer.py
"""
【Phase 4:去全局单例】(见模块内 Tracer 类的说明)

【本轮:故障隔离 + 持久化改造】

背景:实测发现两个真实问题。

  ① 崩溃 = 审计记录全丢。save_trace() 只在 run 结束时调用一次,
     进程如果在结束前崩溃(付款类工具已经真实执行,但没来得及记录),
     一条痕迹都不会落盘。修法见 durable 参数 —— 拆成
     begin_trace/record_event/finish_trace 三个时间点分别持久化。

  ② 观测层故障会拖垮主流程。实测:模型正常回答、业务逻辑完全成功,
     仅因为写 trace 失败(模拟磁盘满),整个 run 抛异常挂掉。这是
     "记忆系统的 on_memory_event 要包 try/except,不能让它拖垮提取
     流程"同一条原则,只是之前没有用在 tracing 上。

     这一条**必须**在 ① 之前生效——durable 模式让每次工具调用都多一次
     写盘,如果不先做故障隔离,这次改造会让"tracing 后端偶发故障"
     从"影响观测质量"升级成"影响业务能否跑完",是净负面。
     所以下面每一处调 storage 的地方都经过 _safe_storage_call。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

from harness.tracing.models import Trace, ToolEvent, LLMCall
from harness.tracing.storage_base import TraceStorageBase

logger = logging.getLogger(__name__)


class Tracer:
    """一个独立的 trace 收集器。所有原本的模块级状态现在是实例状态。

    正常使用不需要直接构造它——模块级函数会落到默认实例上。需要
    隔离时(测试、多租户)才显式造一个,配合 use_tracer() 使用。

    Args:
        storage: 持久化后端,None 表示只在内存里跑,不落盘。
        durable: True(默认)时,每个事件产生的当下就调用 storage 的
                begin_trace/record_event/finish_trace 立即持久化;
                False 时退回改造前的行为——只在 end_trace 时调用一次
                save_trace,把整条 trace 一次性写完。

                默认 True 的理由:False 时的行为是**静默丢数据**——
                进程崩溃时,已经执行过的工具调用不会留下任何痕迹,
                而调用方通常不会意识到这一点,直到真的需要审计追溯
                才发现记录是空的。静默的正确性问题比一次可见的性能
                变化(实测约 2.2×,单次操作仍在毫秒级)更值得默认避免。

                这个默认值改变了配置了 storage 的现有部署的写入量,
                如果你的场景对这点开销敏感、或者后端本身已经有自己
                的持久化保证,显式传 durable=False 退回旧行为。

        redact: 可选的脱敏钩子,签名 (obj) -> obj | None。在事件/trace
                交给 storage 持久化**之前**调用,返回修改后的对象,
                或返回 None 表示"这条不持久化"。obj 的类型是
                Trace / ToolEvent / LLMCall 之一,按需自己 isinstance
                判断。默认 None = 不脱敏,与现状一致——没有加密,
                tool_events.args/result 这类字段默认明文落盘(和
                snapshots/offload 的既有选择一致,不是这次引入的
                新问题,但这次让持久化更完整,值得在这里提供一个
                收窄暴露面的口子)。

                redact 钩子自身抛异常时:记日志,**跳过这条事件的
                持久化**(不是"当作没配置 redact、原样存下去")——
                既然显式配置了脱敏,钩子出 bug 时选择"少存一条"而不是
                "存一条本该被脱敏但没脱敏成功的数据",是更安全的
                失败方向。
    """

    def __init__(self, storage: TraceStorageBase | None = None,
                durable: bool = True,
                redact: Callable[[object], object | None] | None = None):
        self._active: dict[str, Trace] = {}
        self._start_times: dict[str, float] = {}
        self._storage: TraceStorageBase | None = storage
        self.durable = durable
        self.redact = redact

    # ── 配置 ──────────────────────────────────────────────────────────

    def configure_storage(self, storage: TraceStorageBase) -> None:
        self._storage = storage
        logger.info(f"[Tracer] storage backend: {type(storage).__name__}")

    @property
    def storage(self) -> TraceStorageBase | None:
        return self._storage

    # ── 内部:故障隔离 + 脱敏,所有落盘操作的唯一出口 ───────────────────

    def _safe_call(self, fn, *args, **kwargs) -> None:
        """任何一次 storage 调用失败,只记日志、不向上抛。

        这是"观测层故障不该拖垮主流程"这条原则的唯一落点——不是
        分散在每个方法里各自 try/except,是收口到这一个函数,保证
        新增的持久化调用点(以后如果还要加)天然继承这条保护,不需要
        每次新增都记得再包一层。
        """
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"[Tracer] storage 操作 {getattr(fn, '__name__', fn)} 失败"
                f"(trace 在内存里仍然完整,只是这次没能持久化): {e}",
                exc_info=True,
            )

    def _redacted(self, obj):
        """返回脱敏后的对象,或 None 表示"别存这条"。没配置 redact
        时原样返回(几乎零开销的直通)。"""
        if self.redact is None:
            return obj
        try:
            return self.redact(obj)
        except Exception as e:
            logger.error(f"[Tracer] redact 钩子自身抛异常,这条事件将不会"
                        f"被持久化(宁可少存,不存脱敏失败的原始数据): {e}",
                        exc_info=True)
            return None

    # ── 生命周期 ──────────────────────────────────────────────────────

    def start_trace(self, trace_id: str, session_id: str, user_input: str,
                    parent_trace_id: str | None = None) -> None:
        trace = Trace(
            trace_id=trace_id, session_id=session_id, user_input=user_input,
            final_reply="", total_duration_ms=0, tool_call_count=0,
            llm_call_count=0, status="running", parent_trace_id=parent_trace_id,
        )
        self._active[trace_id] = trace
        self._start_times[trace_id] = time.time()
        logger.info(f"[Tracer] start  trace_id={trace_id}  parent={parent_trace_id}")

        if self.durable and self._storage is not None:
            persisted = self._redacted(trace)
            if persisted is not None:
                self._safe_call(self._storage.begin_trace, persisted)

    def end_trace(self, trace_id: str, final_reply: str, status: str = "success") -> None:
        trace = self._active.pop(trace_id, None)
        start = self._start_times.pop(trace_id, None)
        if trace is None:
            return

        trace.final_reply = final_reply
        trace.status = status
        trace.total_duration_ms = int((time.time() - start) * 1000) if start else 0

        logger.info(
            f"[Tracer] end  trace_id={trace_id}  status={status}  "
            f"duration={trace.total_duration_ms}ms  "
            f"tools={trace.tool_call_count}  llm_calls={trace.llm_call_count}"
        )

        if self._storage is None:
            return
        persisted = self._redacted(trace)
        if persisted is None:
            return
        if self.durable:
            self._safe_call(self._storage.finish_trace, persisted)
        else:
            self._safe_call(self._storage.save_trace, persisted)

    # ── 事件记录 ──────────────────────────────────────────────────────

    def record_tool_event(self, trace_id: str, tool_name: str, args: dict,
                          result: str, duration_ms: int,
                          status: str = "success") -> None:
        trace = self._active.get(trace_id)
        if trace is None:
            return
        event = ToolEvent(
            event_id=str(uuid.uuid4()), trace_id=trace_id, tool_name=tool_name,
            args=json.dumps(args, ensure_ascii=False), result=result[:500],
            status=status, duration_ms=duration_ms,
        )
        trace.tool_events.append(event)
        trace.tool_call_count += 1

        if self.durable and self._storage is not None:
            persisted = self._redacted(event)
            if persisted is not None:
                self._safe_call(self._storage.record_event, trace_id, persisted)

    def record_llm_call(self, trace_id: str, prompt_tokens: int,
                        completion_tokens: int, output: str, has_tool_calls: bool,
                        duration_ms: int, token_source: str = "estimated",
                        reasoning: str | None = None) -> None:
        trace = self._active.get(trace_id)
        if trace is None:
            return
        call = LLMCall(
            event_id=str(uuid.uuid4()), trace_id=trace_id,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            token_source=token_source, output=output[:500],
            has_tool_calls=has_tool_calls, duration_ms=duration_ms,
            reasoning=(reasoning or "")[:500] or None,
        )
        trace.llm_calls.append(call)
        trace.llm_call_count += 1

        if self.durable and self._storage is not None:
            persisted = self._redacted(call)
            if persisted is not None:
                self._safe_call(self._storage.record_event, trace_id, persisted)

    # ── 运维:泄漏可见 + 可回收 ──────────────────────────────────────────

    @property
    def active_count(self) -> int:
        """当前有多少条 trace 还没结束。常驻服务应该监控这个数字——
        它只涨不落就说明有 trace 在泄漏。"""
        return len(self._active)

    def active_trace_ids(self) -> list[str]:
        return list(self._active)

    def discard_trace(self, trace_id: str) -> bool:
        """丢弃一条未结束的 trace,不当作正常完成来记录。

        与 end_trace 的区别是刻意的:end_trace 表示"这次运行结束了,
        记下来",discard 表示"这条不作数"。durable 模式下 begin_trace
        可能已经把这条 trace 写进库里了,所以这里也要调用
        storage.delete_trace——不删的话,discard 的语义只在内存里
        成立,磁盘上还留着一条本不该存在的记录。
        """
        self._start_times.pop(trace_id, None)
        found = self._active.pop(trace_id, None) is not None
        if self.durable and self._storage is not None:
            self._safe_call(self._storage.delete_trace, trace_id)
        return found

    def reset(self) -> None:
        """清空全部在途状态(不动 storage 配置)。测试专用。"""
        self._active.clear()
        self._start_times.clear()


# ── 默认实例 + 上下文作用域 ────────────────────────────────────────────

_default_tracer = Tracer()
_current: ContextVar[Tracer | None] = ContextVar("harness_current_tracer", default=None)


def current_tracer() -> Tracer:
    return _current.get() or _default_tracer


def default_tracer() -> Tracer:
    return _default_tracer


@contextmanager
def use_tracer(tracer: Tracer):
    """在这个作用域内把 tracer 设为当前实例。

    asyncio 语义:每个 Task 创建时复制一份当前 context,所以
    asyncio.gather(run_a(), run_b()) 里两个 Task 各自 set 的 tracer
    互不影响。同一个 Task 内是普通的作用域嵌套,退出时靠
    reset(token) 精确还原成进入前的那一个,不是简单设回 None
    (那样会破坏嵌套)。
    """
    token = _current.set(tracer)
    try:
        yield tracer
    finally:
        _current.reset(token)


# ── 模块级 API:签名与改造前完全一致,现有调用方零改动 ────────────────

def configure_storage(storage: TraceStorageBase) -> None:
    """Set the storage backend on the current tracer.

    Example:
        from harness.tracing import configure_storage, SQLiteTraceStorage
        configure_storage(SQLiteTraceStorage(db_path="data/trace.db"))
    """
    current_tracer().configure_storage(storage)


def start_trace(trace_id: str, session_id: str, user_input: str,
                parent_trace_id: str | None = None) -> None:
    current_tracer().start_trace(trace_id, session_id, user_input, parent_trace_id)


def end_trace(trace_id: str, final_reply: str, status: str = "success") -> None:
    current_tracer().end_trace(trace_id, final_reply, status)


def record_tool_event(trace_id: str, tool_name: str, args: dict, result: str,
                      duration_ms: int, status: str = "success") -> None:
    current_tracer().record_tool_event(trace_id, tool_name, args, result,
                                       duration_ms, status)


def record_llm_call(trace_id: str, prompt_tokens: int, completion_tokens: int,
                    output: str, has_tool_calls: bool, duration_ms: int,
                    token_source: str = "estimated",
                    reasoning: str | None = None) -> None:
    current_tracer().record_llm_call(trace_id, prompt_tokens, completion_tokens,
                                     output, has_tool_calls, duration_ms,
                                     token_source, reasoning)