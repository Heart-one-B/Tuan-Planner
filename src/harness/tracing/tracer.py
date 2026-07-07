# harness/tracing/tracer.py
import json
import logging
import time
import uuid

from harness.tracing.models import Trace, ToolEvent, LLMCall
from harness.tracing.storage_base import TraceStorageBase

logger = logging.getLogger(__name__)

# In-flight traces keyed by trace_id
_active: dict[str, Trace] = {}
_start_times: dict[str, float] = {}

# Storage backend — set via configure_storage() at startup
_storage: TraceStorageBase | None = None


def configure_storage(storage: TraceStorageBase) -> None:
    """Set the storage backend. Call once at application startup.

    If never called the tracer still works in memory but nothing is persisted.

    Example:
        from harness.tracing import configure_storage, SQLiteTraceStorage
        configure_storage(SQLiteTraceStorage(db_path="data/trace.db"))
    """
    global _storage
    _storage = storage
    logger.info(f"[Tracer] storage backend: {type(storage).__name__}")


# ── lifecycle ─────────────────────────────────────────────────────────────────

def start_trace(
    trace_id: str,
    session_id: str,
    user_input: str,
    parent_trace_id: str | None = None,
) -> None:
    _active[trace_id] = Trace(
        trace_id=trace_id,
        session_id=session_id,
        user_input=user_input,
        final_reply="",
        total_duration_ms=0,
        tool_call_count=0,
        llm_call_count=0,
        status="running",
        parent_trace_id=parent_trace_id,
    )
    _start_times[trace_id] = time.time()
    logger.info(f"[Tracer] start  trace_id={trace_id}  parent={parent_trace_id}")


def end_trace(trace_id: str, final_reply: str, status: str = "success") -> None:
    trace = _active.pop(trace_id, None)
    start = _start_times.pop(trace_id, None)
    if trace is None:
        return

    trace.final_reply = final_reply
    trace.status = status
    trace.total_duration_ms = int((time.time() - start) * 1000)

    logger.info(
        f"[Tracer] end  trace_id={trace_id}  status={status}  "
        f"duration={trace.total_duration_ms}ms  "
        f"tools={trace.tool_call_count}  llm_calls={trace.llm_call_count}"
    )

    if _storage:
        _storage.save_trace(trace)


# ── event recording ───────────────────────────────────────────────────────────

def record_tool_event(
    trace_id: str,
    tool_name: str,
    args: dict,
    result: str,
    duration_ms: int,
    status: str = "success",
) -> None:
    trace = _active.get(trace_id)
    if trace is None:
        return

    trace.tool_events.append(ToolEvent(
        event_id=str(uuid.uuid4()),
        trace_id=trace_id,
        tool_name=tool_name,
        args=json.dumps(args, ensure_ascii=False),
        result=result[:500],
        status=status,
        duration_ms=duration_ms,
    ))
    trace.tool_call_count += 1


def record_llm_call(
    trace_id: str,
    input_messages: list,
    output: str,
    has_tool_calls: bool,
    duration_ms: int,
    reasoning: str | None = None,
) -> None:
    trace = _active.get(trace_id)
    if trace is None:
        return

    input_token_count = sum(
        len(m["content"])
        if isinstance(m, dict) and isinstance(m.get("content"), str)
        else len(m.content or "")
        if hasattr(m, "content") and isinstance(m.content, str)
        else 0
        for m in input_messages
    )

    trace.llm_calls.append(LLMCall(
        event_id=str(uuid.uuid4()),
        trace_id=trace_id,
        input_token_count=input_token_count,
        output=output[:500],
        has_tool_calls=has_tool_calls,
        duration_ms=duration_ms,
        reasoning=(reasoning or "")[:500] or None,
    ))
    trace.llm_call_count += 1