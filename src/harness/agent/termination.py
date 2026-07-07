# harness/agent/termination.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Type

from pydantic import BaseModel, ValidationError

from harness.agent.result import AgentResult


# ── 终止决策:策略返回给循环内核的三种指令 ──────────────────────────────
@dataclass
class Finish:
    """本次运行结束。final_text 是面向用户的文本,result 是结构化载荷
    (如 AgentResult)——两者按策略需要填,循环内核不解读 result 的语义。"""
    final_text: str = ""
    result: Any = None


@dataclass
class Nudge:
    """还不能结束,把这条消息以 user 角色追加回去,催模型继续。"""
    message: str


@dataclass
class Reject:
    """终止工具的参数不合格:以 tool 角色回填错误信息,让模型重试。"""
    tool_call_id: str
    message: str


# ── 策略接口 ───────────────────────────────────────────────────────────
class TerminationPolicy(Protocol):
    """终止策略:回答"这次运行怎样才算结束"。

    与 DagScheduler 的 gate、AgentLoop 的 Budget 同属一族思想——
    可变的判断逻辑策略化注入,循环内核只认接口。
    """

    def extra_tools(self) -> list[dict]:
        """需要额外注入给模型的工具 schema(如 finish)。没有就返回 []。"""
        ...

    def owns(self, tool_call) -> bool:
        """这个 tool_call 是不是终止信号(而非真工具)?
        循环内核用它把终止调用从普通调用里分拣出来:
        普通调用先全部执行完,终止调用最后处理——
        这在结构上杜绝了"同轮混发普通工具+finish 时普通工具结果被丢"的 bug。"""
        ...

    def intercept(self, tool_call, *, forced: bool) -> Finish | Reject:
        """处理被 owns() 认领的调用。forced=True 表示预算耗尽后的强制收口,
        策略可据此调整默认值(如 status 从 ok 降为 partial)。"""
        ...

    def on_plain_message(self, msg) -> Finish | Nudge:
        """模型没调任何工具、只回了文本时,算结束还是要催?"""
        ...


# ── 内置策略一:文本即答案(原 ReactAgent 语义)──────────────────────────
class AnswerTermination:
    """模型停止调工具、输出纯文本 = 最终答案。"""

    def extra_tools(self) -> list[dict]:
        return []

    def owns(self, tool_call) -> bool:
        return False

    def intercept(self, tool_call, *, forced: bool) -> Finish | Reject:
        raise RuntimeError("AnswerTermination 不认领任何 tool_call")

    def on_plain_message(self, msg) -> Finish:
        return Finish(final_text=msg.content or "")


# ── 内置策略二:finish 工具收口(原 StructuredAgent 语义)─────────────────
class FinishToolTermination:
    """唯一的结束方式是调用 finish 工具,产出结构化 AgentResult。

    schema 结构:框架信封(status/summary)包住业务 data(output_schema),
    与原 StructuredAgent._finish_schema 完全一致。
    """

    def __init__(self, output_schema: Type[BaseModel], tool_name: str = "finish"):
        self.output_schema = output_schema
        self.tool_name = tool_name

    def extra_tools(self) -> list[dict]:
        return [self._finish_schema()]

    def owns(self, tool_call) -> bool:
        return tool_call.function.name == self.tool_name

    def intercept(self, tool_call, *, forced: bool) -> Finish | Reject:
        try:
            raw = json.loads(tool_call.function.arguments)
            result = self._to_result(raw, default_status="partial" if forced else "ok")
            return Finish(final_text=result.summary, result=result)
        except (ValidationError, json.JSONDecodeError) as e:
            return Reject(
                tool_call_id=tool_call.id,
                message=f"finish 参数格式错误: {e}。请严格按 schema 重新调用 finish。",
            )

    def on_plain_message(self, msg) -> Nudge:
        return Nudge("请调用 finish 工具给出结构化结论,不要用自然语言回答。")

    # ── 内部 ──
    def _to_result(self, raw_args: dict, default_status: str) -> AgentResult:
        status = raw_args.get("status") or default_status
        summary = raw_args.get("summary") or ""
        validated = self.output_schema.model_validate(raw_args.get("data") or {})
        return AgentResult(status=status, summary=summary, data=validated.model_dump())

    def _finish_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
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
                        "summary": {"type": "string", "description": "一句话结论"},
                        "data": self.output_schema.model_json_schema(),
                    },
                    "required": ["status", "summary", "data"],
                },
            },
        }