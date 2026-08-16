# tests/tracing/test_durability.py
"""tracing 持久化改造的验证。

对应的问题(按严重程度):
① 观测层故障不该拖垮主流程 —— 这是必须先成立的前提,没有它,
   逐事件写入这次改造是净负面(把偶发故障从"影响观测"升级成
   "影响业务能否跑完")
② 崩溃后审计记录要能查到 —— 这是本轮改造真正要解决的问题
③ 不该破坏任何既有行为 —— durable=False、旧后端、discard 语义
"""
from __future__ import annotations

import asyncio

import pytest

from harness.tracing import Tracer, SQLiteTraceStorage, use_tracer
from harness.tracing.storage_base import TraceStorageBase
from harness.tracing.models import LLMCall, ToolEvent, Trace


class _AlwaysBrokenStorage(TraceStorageBase):
    """每一个持久化方法都抛异常,模拟磁盘满/DB锁超时/远程后端故障。"""

    def save_trace(self, trace):
        raise OSError("database or disk is full")

    def get_traces_by_session(self, session_id):
        return []

    def begin_trace(self, trace):
        raise OSError("database or disk is full")

    def record_event(self, trace_id, event):
        raise OSError("database or disk is full")

    def finish_trace(self, trace):
        raise OSError("database or disk is full")

    def delete_trace(self, trace_id):
        raise OSError("database or disk is full")


class _RecordingStorage(TraceStorageBase):
    def __init__(self):
        self.begun: list[Trace] = []
        self.events: list[tuple[str, object]] = []
        self.finished: list[Trace] = []
        self.saved: list[Trace] = []
        self.deleted: list[str] = []

    def save_trace(self, trace):
        self.saved.append(trace)

    def get_traces_by_session(self, session_id):
        return []

    def begin_trace(self, trace):
        self.begun.append(trace)

    def record_event(self, trace_id, event):
        self.events.append((trace_id, event))

    def finish_trace(self, trace):
        self.finished.append(trace)

    def delete_trace(self, trace_id):
        self.deleted.append(trace_id)
        return True


# ══ ① 故障隔离:最优先的一条,必须先成立 ═══════════════════════════════

async def test_broken_storage_does_not_crash_the_agent():
    """核心验证:模型正常回答、业务逻辑完全成功,即使 storage 每一次
    调用都抛异常,agent.run() 也必须正常返回,不能把观测层的故障
    传染给主流程。"""
    import sys
    sys.path.insert(0, "tests")
    from tests.loop.fakes import FakeLLMClient
    from harness.agent.agent import Agent
    from harness.tools.tool_executor import ToolExecutor

    t = Tracer(storage=_AlwaysBrokenStorage())
    with use_tracer(t):
        agent = Agent(FakeLLMClient([{"content": "正常答案", "tool_calls": []}]),
                      ToolExecutor(), "你是助手")
        outcome = await agent.run("问题")

    assert outcome.status == "completed"
    assert outcome.final_text == "正常答案"


def test_begin_trace_failure_does_not_raise():
    t = Tracer(storage=_AlwaysBrokenStorage())
    t.start_trace("x", "s", "task")   # 不该抛
    assert t.active_count == 1        # 内存记账仍然正常


def test_record_event_failure_does_not_raise():
    t = Tracer(storage=_AlwaysBrokenStorage())
    t.start_trace("x", "s", "task")
    t.record_tool_event("x", "tool", {}, "result", 10)   # 不该抛
    t.record_llm_call("x", 10, 5, "out", False, 20)       # 不该抛
    # 内存记账不受影响
    trace = t._active["x"]
    assert len(trace.tool_events) == 1
    assert len(trace.llm_calls) == 1


def test_finish_trace_failure_does_not_raise():
    t = Tracer(storage=_AlwaysBrokenStorage())
    t.start_trace("x", "s", "task")
    t.end_trace("x", "done")   # 不该抛
    assert t.active_count == 0   # 内存记账(移出在途)仍然发生


def test_discard_failure_does_not_raise_and_still_frees_memory():
    t = Tracer(storage=_AlwaysBrokenStorage())
    t.start_trace("x", "s", "task")
    found = t.discard_trace("x")   # storage.delete_trace 抛异常,不该向上传播
    assert found is True
    assert t.active_count == 0


# ══ ② 崩溃后可审计:本轮改造真正要解决的问题 ═══════════════════════════

def test_crash_before_end_trace_still_leaves_audit_trail(tmp_path):
    """核心场景:开始一个 run、记录一次工具调用(内含"转账"这类关键
    证据),但**不调用 end_trace**——模拟进程在这里被杀。用一个全新的
    storage 实例重新打开数据库(模拟'进程重启后来查'),必须能查到
    这条 trace 和它已经发生过的事件。"""
    storage = SQLiteTraceStorage(tmp_path / "t.db")
    t = Tracer(storage=storage)

    t.start_trace("run-1", "sess", "给张三转账100元")
    t.record_tool_event("run-1", "transfer", {"to": "张三", "amount": 100},
                        "已转账 TX-001", 50)
    t.record_llm_call("run-1", 1000, 200, "好的,已完成转账", False, 300)
    # 故意不调用 t.end_trace(...)

    reopened = SQLiteTraceStorage(tmp_path / "t.db")
    got = reopened.get_trace("run-1")

    assert got is not None
    assert got.status == "running"
    assert len(got.tool_events) == 1
    assert got.tool_events[0].tool_name == "transfer"
    assert "张三" in got.tool_events[0].args
    assert got.tool_events[0].result == "已转账 TX-001"
    assert len(got.llm_calls) == 1
    assert got.llm_calls[0].prompt_tokens == 1000


def test_unfinished_trace_is_queryable_after_crash(tmp_path):
    """崩溃留下的孤儿要能被主动查出来,不需要知道具体 trace_id。"""
    storage = SQLiteTraceStorage(tmp_path / "t.db")
    t = Tracer(storage=storage)
    t.start_trace("orphan-1", "s", "task")
    t.start_trace("orphan-2", "s", "task")
    t.record_tool_event("orphan-1", "tool", {}, "r", 1)

    unfinished = storage.get_unfinished_traces()

    assert {tr.trace_id for tr in unfinished} == {"orphan-1", "orphan-2"}


def test_finished_trace_does_not_appear_in_unfinished(tmp_path):
    storage = SQLiteTraceStorage(tmp_path / "t.db")
    t = Tracer(storage=storage)
    t.start_trace("r1", "s", "task")
    t.end_trace("r1", "done")

    assert storage.get_unfinished_traces() == []


def test_finish_trace_updates_not_duplicates(tmp_path):
    """durable 模式下 begin_trace 已经写过一次,finish_trace 用
    UPDATE——确认最终状态正确覆盖,而不是变成两行或者丢字段。"""
    storage = SQLiteTraceStorage(tmp_path / "t.db")
    t = Tracer(storage=storage)
    t.start_trace("r1", "sess-x", "原始任务")
    t.record_tool_event("r1", "tool", {}, "r", 1)
    t.end_trace("r1", "最终答案", status="success")

    got = storage.get_trace("r1")
    assert got.status == "success"
    assert got.final_reply == "最终答案"
    assert got.session_id == "sess-x"          # begin_trace 时写的字段还在
    assert len(got.tool_events) == 1            # 事件没有因为 UPDATE 被清空

    all_rows = storage.get_traces_by_session("sess-x")
    assert len(all_rows) == 1                    # 不是两行


# ══ ③ 不破坏既有行为 ═══════════════════════════════════════════════════

def test_durable_false_only_writes_at_end(tmp_path):
    """durable=False 退回改造前的行为:结束前数据库里完全没有这条
    trace,只有 end_trace 时才一次性写完整条记录。"""
    storage = SQLiteTraceStorage(tmp_path / "t.db")
    t = Tracer(storage=storage, durable=False)

    t.start_trace("r1", "s", "task")
    t.record_tool_event("r1", "tool", {}, "result", 10)
    assert storage.get_trace("r1") is None       # 结束前:数据库里没有

    t.end_trace("r1", "done")
    got = storage.get_trace("r1")
    assert got is not None
    assert len(got.tool_events) == 1              # 结束时一次性写完整


def test_default_durable_is_true():
    assert Tracer().durable is True


def test_discard_removes_persisted_trace(tmp_path):
    """discard 和 end 的区别:durable 模式下 begin_trace 已经把行
    写进库里了,discard 必须真的删掉,不能只清内存。"""
    storage = SQLiteTraceStorage(tmp_path / "t.db")
    t = Tracer(storage=storage)
    t.start_trace("r1", "s", "task")
    assert storage.get_trace("r1") is not None

    t.discard_trace("r1")

    assert storage.get_trace("r1") is None


def test_old_backend_without_new_methods_behaves_unchanged():
    """没实现 begin_trace/record_event 的旧后端(只实现了两个抽象
    方法):行为必须和改造前逐字节一致——begin/record 阶段是 no-op,
    只在 end_trace 时通过默认的 finish_trace→save_trace 写一次。"""
    class OldStyle(TraceStorageBase):
        def __init__(self):
            self.saved = []
        def save_trace(self, trace):
            self.saved.append(trace)
        def get_traces_by_session(self, s):
            return []

    backend = OldStyle()
    t = Tracer(storage=backend)   # durable=True(默认),但后端没实现新方法

    t.start_trace("r1", "s", "task")
    t.record_tool_event("r1", "tool", {}, "r", 1)
    assert backend.saved == []     # begin/record 是 no-op,还没写

    t.end_trace("r1", "done")
    assert len(backend.saved) == 1   # 退回 save_trace,一次性写完
    assert len(backend.saved[0].tool_events) == 1


# ══ redact 钩子 ═══════════════════════════════════════════════════════

def test_redact_hook_can_scrub_sensitive_fields():
    storage = _RecordingStorage()

    def redact(obj):
        if isinstance(obj, ToolEvent):
            return ToolEvent(**{**obj.__dict__, "args": "[已脱敏]", "result": "[已脱敏]"})
        return obj

    t = Tracer(storage=storage, redact=redact)
    t.start_trace("r1", "s", "task")
    t.record_tool_event("r1", "transfer", {"to": "张三", "amount": 100}, "TX-001", 1)

    persisted_event = storage.events[0][1]
    assert persisted_event.args == "[已脱敏]"
    assert persisted_event.result == "[已脱敏]"
    # 内存里的原始事件不受影响 —— redact 只影响持久化的副本
    assert "张三" in t._active["r1"].tool_events[0].args


def test_redact_returning_none_skips_persistence():
    storage = _RecordingStorage()
    t = Tracer(storage=storage, redact=lambda obj: None)   # 全部丢弃

    t.start_trace("r1", "s", "task")
    t.record_tool_event("r1", "tool", {}, "r", 1)

    assert storage.begun == []
    assert storage.events == []


def test_redact_hook_exception_skips_persistence_not_crashes():
    """redact 自身有 bug 时:跳过这条持久化,不是原样存下去
    (显式配置了脱敏,失败时选择'少存'比'存脱敏失败的原始数据'更安全),
    也不该让调用崩溃。"""
    storage = _RecordingStorage()

    def broken_redact(obj):
        raise ValueError("redact 自己的 bug")

    t = Tracer(storage=storage, redact=broken_redact)
    t.start_trace("r1", "s", "task")   # 不该抛

    assert storage.begun == []          # 没有存下未脱敏的原始数据
    assert t.active_count == 1          # 内存记账不受影响


# ══ 并发 ═══════════════════════════════════════════════════════════════

async def test_concurrent_runs_do_not_lose_events(tmp_path):
    """长连接 + 锁的并发安全性:多个 run 并发写同一个 SQLite 实例,
    不该因为锁竞争丢事件或抛异常。"""
    storage = SQLiteTraceStorage(tmp_path / "t.db")

    async def one_run(i: int):
        with use_tracer(Tracer(storage=storage)):
            from harness.tracing.tracer import (
                start_trace, record_tool_event, end_trace,
            )
            start_trace(f"run-{i}", "s", "task")
            for j in range(5):
                record_tool_event(f"run-{i}", f"tool{j}", {}, "r", 1)
            end_trace(f"run-{i}", "done")

    await asyncio.gather(*[one_run(i) for i in range(20)])

    for i in range(20):
        t = storage.get_trace(f"run-{i}")
        assert t is not None, f"run-{i} 丢失"
        assert len(t.tool_events) == 5, f"run-{i} 事件数不对: {len(t.tool_events)}"


# ══ SQLite 长连接管理 ══════════════════════════════════════════════════

def test_close_releases_connection(tmp_path):
    storage = SQLiteTraceStorage(tmp_path / "t.db")
    assert storage._conn is not None
    storage.close()
    assert storage._conn is None


def test_journal_mode_is_wal(tmp_path):
    """WAL 只有配长连接才有意义(实测:短连接+WAL 反而比短连接
    不加 WAL 慢 —— 连接建立要额外读 -wal 文件)。这里确认它真的被
    设置上了,不只是注释里说说。"""
    import sqlite3
    storage = SQLiteTraceStorage(tmp_path / "t.db")
    conn = sqlite3.connect(str(tmp_path / "t.db"))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode.lower() == "wal"