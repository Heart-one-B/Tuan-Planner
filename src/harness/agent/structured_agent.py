from __future__ import annotations

import json
import uuid
from typing import Type

from pydantic import BaseModel, ValidationError

from harness.agent.result import AgentResult
from harness.llm.base import LLMClientBase
from harness.tools.tool_executor import ToolExecutor
from harness.tracing import tracer


class StructuredAgent:
    """子 Agent 内核:ReAct 自由探索,通过 finish 工具收口为结构化结果。

    多 Agent 协作的通用基类。所有子 Agent 返回统一信封 AgentResult:
        status   框架统一(ok/partial/empty/error)—— 上层据此决策
        summary  框架统一(一句话结论)
        data     各 Agent 自己的 output_schema(业务字段,各不相同)

    finish 工具的参数 = 框架信封字段 + 嵌套的 data(业务 schema)。
    框架管信封,业务管 data,职责清晰。
    """

    FINISH_TOOL = "finish"

    def __init__(
        self,
        llm_client: LLMClientBase,
        tool_executor: ToolExecutor,
        system_prompt: str,
        output_schema: Type[BaseModel],   # 只描述 data 的形状,不含 status/summary
        max_tool_calls: int = 10,
        name: str = "agent",
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.output_schema = output_schema
        self.max_tool_calls = max_tool_calls
        self.name = name

    # ── finish schema:框架信封 包住 业务 data ──────────────────────────────
    def _finish_schema(self) -> dict:
        data_schema = self.output_schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.FINISH_TOOL,
                "description": (
                    "当你已收集到足够信息、可以给出最终结论时调用此工具结束任务"
                    "(这是结束的唯一方式,不要用自然语言回答)。\n"
                    "status 如实填写:ok=拿到理想结果, partial=只拿到部分, "
                    "empty=确实查不到, error=持续失败无法完成。\n"
                    "summary 用一句话概括结论。\n"
                    "data 按其 schema 填写本次任务的结构化结果。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["ok", "partial", "empty", "error"],
                            "description": "任务完成状态",
                        },
                        "summary": {
                            "type": "string",
                            "description": "一句话结论",
                        },
                        "data": data_schema,   # ← 各 Agent 的业务 schema 注入这里
                    },
                    "required": ["status", "summary", "data"],
                },
            },
        }

    # ── 校验 finish 参数 → AgentResult(信封 + 校验过的 data)───────────────
    def _finish_to_result(self, raw_args: dict, default_status: str) -> AgentResult:
        status = raw_args.get("status") or default_status
        summary = raw_args.get("summary") or ""
        # 只用 output_schema 校验 data 部分(业务字段)
        validated_data = self.output_schema.model_validate(raw_args.get("data") or {})
        return AgentResult(
            status=status,
            summary=summary,
            data=validated_data.model_dump(),
        )

    async def run(self, task: str, trace_id: str | None = None) -> AgentResult:
        trace_id = trace_id or str(uuid.uuid4())
        tracer.start_trace(trace_id, self.name, task)

        tools = self.tool_executor.schemas + [self._finish_schema()]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        tool_call_count = 0

        try:
            while True:
                resp = await self.llm_client.call(
                    trace_id=trace_id, messages=messages, tools=tools,
                )
                msg = resp.choices[0].message

                if not msg.tool_calls:
                    messages.append({
                        "role": "user",
                        "content": "请调用 finish 工具给出结构化结论,不要用自然语言回答。",
                    })
                    continue

                messages.append(msg)
                hit_finish = False

                for tc in msg.tool_calls:
                    if tc.function.name == self.FINISH_TOOL:
                        hit_finish = True
                        try:
                            raw = json.loads(tc.function.arguments)
                            result = self._finish_to_result(raw, default_status="ok")
                            tracer.end_trace(trace_id, result.summary, status="success")
                            return result
                        except (ValidationError, json.JSONDecodeError) as e:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"finish 参数格式错误: {e}。请严格按 schema 重新调用 finish。",
                            })
                            break

                    tool_call_count += 1
                    result_str = await self.tool_executor.execute(tc, trace_id=trace_id)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result_str),
                    })

                if hit_finish:
                    continue

                if tool_call_count >= self.max_tool_calls:
                    return await self._force_finish(messages, tools, trace_id)

        except Exception as e:
            tracer.end_trace(trace_id, str(e), status="error")
            return AgentResult(status="error", summary=f"运行异常: {e}", data={})

    async def _force_finish(self, messages: list, tools: list, trace_id: str) -> AgentResult:
        messages.append({
            "role": "user",
            "content": (
                "已达工具调用上限。立即调用 finish 工具,如实汇报已拿到的信息,"
                "status 据实填写(partial / empty / error)。"
            ),
        })
        resp = await self.llm_client.call(trace_id=trace_id, messages=messages, tools=tools)
        msg = resp.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == self.FINISH_TOOL:
                    try:
                        raw = json.loads(tc.function.arguments)
                        result = self._finish_to_result(raw, default_status="partial")
                        tracer.end_trace(trace_id, result.summary, status="timeout")
                        return result
                    except (ValidationError, json.JSONDecodeError):
                        break
        fallback = AgentResult(status="error", summary="未能在限定轮次内收口", data={})
        tracer.end_trace(trace_id, fallback.summary, status="timeout")
        return fallback