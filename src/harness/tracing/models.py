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
    created_at: datetime = field(default_factory=datetime.now)
    tool_events: list[ToolEvent] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)