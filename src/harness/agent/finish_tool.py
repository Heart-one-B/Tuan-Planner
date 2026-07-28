# harness/agent/finish_tool.py
from __future__ import annotations

from typing import Type

from pydantic import BaseModel, ValidationError

from harness.agent.result import AgentResult
from harness.tools.tool_definition import ToolDefinition

# 设计方案 §3.3:结构化收口是"Agent 即工具"场景的真实需求(上层调用方
# 是程序,需要 schema 校验过的 dict,不是自由文本——CC 没有这个需求,
# 不是它比这里先进,是场景不同,见 memory/recall.py 的 _parse_selection
# 已经踩过的"自由文本让模型只输出 JSON 仍会包 markdown 围栏"的坑)。
#
# 但"怎么收口"不该是一层独立的 TerminationPolicy 抽象——它就是一个
# 普通工具,和 memory_write、read_offloaded_result 走同一条注册、
# 同一条执行、同一套【】前缀错误反馈路径。它和别的工具唯一的区别是
# ToolDefinition.terminal=True 这一个字段:AgentLoop 处理完一批工具
# 调用后,据此检查 run_ctx.state["result"] 是否被写入,写入了就是
# "模型已经明确表达完成、且参数通过了 schema 校验",立即收口,不需要
# TerminationPolicy.owns()/intercept() 这层单独的分拣接口。
#
# 参数格式错误时的反馈复用 ToolExecutor 的【参数错误,请调整后重试】
# 前缀约定——模型看得懂、会重试,不需要一个专门的 Reject 通道
# (旧 termination.py 的 Reject 就此消失,连带消失的是它"没有次数上限"
# 那个 bug:现在参数错误就是普通工具失败,天然计入 tool_calls_used 预算,
# 不需要额外的计数器)。


def build_finish_tool(output_schema: Type[BaseModel], name: str = "finish") -> ToolDefinition:
    """构造一个"结构化收口"工具。

    output_schema:业务方定义的结果 data 部分的 pydantic 模型。
    name:注册进 ToolExecutor 的工具名,同时也是
        LoopConfig/Agent 的 require_terminal_tool 要填的值。

    产出的 ToolDefinition.terminal=True——这是它和普通工具唯一的
    结构性区别,AgentLoop 靠这一个字段识别它,不靠名字匹配。
    """

    async def finish(status: str, summary: str, data: dict, run_ctx) -> str:
        # 框架信封(status/summary)和业务 data 一起校验、一起失败——
        # status 不在 output_schema.model_validate() 的校验范围内
        # (它是 AgentResult 自己的 Literal 字段),放进同一个 try 里,
        # 让 AgentResult(...) 的构造顺带校验 status 是否是合法枚举值,
        # 一次失败一次反馈,不需要为 status 单独写一段校验逻辑。
        try:
            validated_data = output_schema.model_validate(data or {})
            result = AgentResult(
                status=status, summary=summary, data=validated_data.model_dump(),
            )
        except ValidationError as e:
            return f"【参数错误，请调整后重试】{name} 参数不符合要求：{e}"

        # 写入 run_ctx.state 而不是直接 return 给调用方——ToolExecutor.execute()
        # 的返回值必须是字符串(它要作为 tool 消息的 content 塞回上下文,
        # 模型也要看到"已收口"这件事发生了),结构化结果走 run_ctx 这条
        # 编排型工具早就铺好的旁路(见 RunContext 的 state 字段设计初衷)。
        run_ctx.state["result"] = result
        return "已记录最终结论。"

    return ToolDefinition(
        name=name,
        terminal=True,
        func=finish,
        description=(
            "当你已收集到足够信息、可以给出最终结论时调用此工具结束任务"
            "(这是结束的唯一方式,不要用自然语言回答)。\n"
            "status 如实填写:ok=拿到理想结果, partial=只拿到部分, "
            "empty=确实查不到, error=持续失败无法完成。\n"
            "summary 用一句话概括结论。\n"
            "data 按其 schema 填写本次任务的结构化结果。"
        ),
        parameters={
            "status": {
                "type": "string",
                "enum": ["ok", "partial", "empty", "error"],
                "description": "任务完成状态",
            },
            "summary": {"type": "string", "description": "一句话结论"},
            "data": output_schema.model_json_schema(),
        },
        required=["status", "summary", "data"],
    )