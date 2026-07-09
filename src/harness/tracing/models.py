# harness/tracing/models.py
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolEvent:
    event_id: str
    trace_id: str
    tool_name: str
    args: str
    result: str
    status: str
    duration_ms: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMCall:
    event_id: str
    trace_id: str
    prompt_tokens: int
    completion_tokens: int
    token_source: str        # "api_usage" | "estimated" —— 标注数字可信度
    output: str
    has_tool_calls: bool
    duration_ms: int
    reasoning: str | None = None
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
    status: str
    parent_trace_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    tool_events: list[ToolEvent] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)