# harness/tracing/models.py
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolEvent:
    event_id: str
    trace_id: str
    tool_name: str
    args: str           # JSON string
    result: str
    status: str         # success | param_error | retryable | missing_info |
                        # not_found | degraded | unavailable | partial | error
    duration_ms: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMCall:
    event_id: str
    trace_id: str
    input_token_count: int
    output: str
    has_tool_calls: bool
    duration_ms: int
    reasoning: str | None = None            # 新增:归一化后的推理内容(截断存储)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Trace:
    trace_id: str
    session_id: str
    user_input: str
    final_reply: str
    total_duration_ms: int
    tool_call_count: int
    llm_call_count: int
    status: str         # success | timeout | error
    parent_trace_id: str | None = None      # 新增:父 trace 串联(span 树)
    created_at: datetime = field(default_factory=datetime.now)
    tool_events: list[ToolEvent] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)