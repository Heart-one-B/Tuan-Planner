import json
import logging
import uuid
from typing import AsyncGenerator

from harness.llm.base import LLMClientBase
from harness.tools.tool_executor import ToolExecutor
from harness.tracing import tracer

logger = logging.getLogger(__name__)


class ReactAgent:
    """ReAct loop agent.

    Wires together an LLMClient, a ToolExecutor, and the tracing layer.
    All three are injected — no global singletons anywhere.

    Yields a stream of typed event dicts:
        {"type": "tool_start", "tool": str,  "args": dict}
        {"type": "tool_end",   "tool": str,  "result": str}
        {"type": "token",      "content": str}
        {"type": "done"}
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        tool_executor: ToolExecutor,
        system_prompt: str,
        max_tool_calls: int = 10,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.max_tool_calls = max_tool_calls

    async def run(
        self,
        user_input: str,
        session_id: str = None,
        history: list[dict] = None,
    ) -> AsyncGenerator[dict, None]:

        trace_id = str(uuid.uuid4())
        tracer.start_trace(trace_id, session_id or "unknown", user_input)

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + (history or [])
            + [{"role": "user", "content": user_input}]
        )

        tool_call_count = 0
        final_reply = ""

        try:
            while True:
                resp = await self.llm_client.call(
                    trace_id=trace_id,
                    messages=messages,
                    tools=self.tool_executor.schemas,
                )

                msg = resp.choices[0].message

                # ── no tool calls: stream the final answer ────────────────────
                if not msg.tool_calls:
                    for char in msg.content:
                        final_reply += char
                        yield {"type": "token", "content": char}
                    yield {"type": "done"}
                    tracer.end_trace(trace_id, final_reply, status="success")
                    return

                # ── tool calls: execute each and append results ───────────────
                messages.append(msg)

                for tc in msg.tool_calls:
                    tool_call_count += 1
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    logger.info(
                        f"[ReactAgent] tool_call #{tool_call_count}: "
                        f"{tool_name}  args={tool_args}"
                    )
                    yield {"type": "tool_start", "tool": tool_name, "args": tool_args}

                    result = await self.tool_executor.execute(tc, trace_id=trace_id)

                    logger.info(f"[ReactAgent] tool_end: {result[:120]}")
                    yield {"type": "tool_end", "tool": tool_name, "result": result}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })

                # ── budget exhausted: force a final answer ────────────────────
                if tool_call_count >= self.max_tool_calls:
                    logger.warning(
                        f"[ReactAgent] max_tool_calls={self.max_tool_calls} reached"
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "你已用完工具调用次数，请根据已有信息尽力回答，"
                            "信息不足请告知用户缺少什么。"
                        ),
                    })
                    stream = await self.llm_client.call(
                        trace_id=trace_id,
                        messages=messages,
                        stream=True,
                    )
                    async for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            final_reply += content
                            yield {"type": "token", "content": content}
                    yield {"type": "done"}
                    tracer.end_trace(trace_id, final_reply, status="timeout")
                    return

        except Exception as e:
            logger.error(f"[ReactAgent] error: {e}", exc_info=True)
            tracer.end_trace(trace_id, "", status="error")
            raise