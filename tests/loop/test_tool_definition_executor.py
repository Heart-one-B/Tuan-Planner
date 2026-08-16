# tests/agent/test_tool_definition_executor.py
"""对应待做计划表任务 1.1 / 1.2:ToolDefinition.terminal 字段、
ToolExecutor.is_terminal()。"""
from __future__ import annotations

from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor


async def _noop(**kwargs) -> str:
    return "ok"


def test_terminal_defaults_to_false():
    td = ToolDefinition(name="search", func=_noop, description="", parameters={})
    assert td.terminal is False


def test_terminal_can_be_set_true():
    td = ToolDefinition(name="finish", func=_noop, description="", parameters={}, terminal=True)
    assert td.terminal is True


def test_read_only_field_exists_and_defaults_false():
    """本次只加字段不消费(见设计方案"明确不做的事"),fail-closed 默认。"""
    td = ToolDefinition(name="search", func=_noop, description="", parameters={})
    assert td.read_only is False


def test_executor_is_terminal_reflects_registered_tool():
    executor = ToolExecutor()
    executor.register(ToolDefinition(name="search", func=_noop, description="", parameters={}, terminal=False))
    executor.register(ToolDefinition(name="finish", func=_noop, description="", parameters={}, terminal=True))

    assert executor.is_terminal("search") is False
    assert executor.is_terminal("finish") is True


def test_executor_is_terminal_unregistered_name_returns_false():
    """未注册的工具名不该抛异常,也不该被误判为 terminal
    (照 is_exempt_from_offload 同样的防御写法)。"""
    executor = ToolExecutor()
    assert executor.is_terminal("does_not_exist") is False


def test_to_openai_schema_does_not_leak_terminal_field():
    """terminal 是 harness 内部记账字段,绝不能出现在发给模型的
    工具 schema 里——模型不需要、也不该知道这件事。"""
    td = ToolDefinition(
        name="finish", func=_noop, terminal=True,
        description="收口", parameters={"x": {"type": "string"}}, required=["x"],
    )
    schema = td.to_openai_schema()
    assert "terminal" not in schema["function"]
    assert "terminal" not in schema
    assert schema["function"]["name"] == "finish"
