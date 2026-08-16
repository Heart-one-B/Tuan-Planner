# tests/agent/test_finish_tool.py
"""对应待做计划表任务 1.3:build_finish_tool。

第一性验证点:
  ① terminal=True(AgentLoop 识别它靠这一个字段,不靠名字)
  ② 校验通过 → 写 run_ctx.state["result"],返回值是给模型看的
    确认文案,不是结构化数据本身(结构化数据走 run_ctx 旁路)
  ③ 校验失败(data 不合 schema / status 不是合法枚举)→ 走
    ToolExecutor 统一的【参数错误,请调整后重试】前缀,不写 state
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from harness.agent.finish_tool import build_finish_tool
from harness.agent.result import AgentResult


class _DummyData(BaseModel):
    answer: str
    confidence: float


def _fake_run_ctx() -> SimpleNamespace:
    return SimpleNamespace(state={})


def test_finish_tool_is_terminal():
    td = build_finish_tool(_DummyData)
    assert td.terminal is True
    assert td.name == "finish"


def test_finish_tool_custom_name():
    td = build_finish_tool(_DummyData, name="submit_answer")
    assert td.name == "submit_answer"


async def test_finish_success_writes_result_to_run_ctx_state():
    td = build_finish_tool(_DummyData)
    run_ctx = _fake_run_ctx()

    msg = await td.func(
        status="ok", summary="拿到答案了",
        data={"answer": "42", "confidence": 0.9},
        run_ctx=run_ctx,
    )

    assert "参数错误" not in msg
    result = run_ctx.state["result"]
    assert isinstance(result, AgentResult)
    assert result.status == "ok"
    assert result.summary == "拿到答案了"
    assert result.data == {"answer": "42", "confidence": 0.9}


async def test_finish_invalid_data_does_not_write_state():
    td = build_finish_tool(_DummyData)
    run_ctx = _fake_run_ctx()

    msg = await td.func(
        status="ok", summary="残缺数据",
        data={"answer": "42"},  # 缺 confidence
        run_ctx=run_ctx,
    )

    assert "参数错误" in msg
    assert "result" not in run_ctx.state


async def test_finish_invalid_status_does_not_write_state():
    """status 不在 AgentResult 的合法枚举里(ok/partial/empty/error),
    应该和 data 校验失败走同一条反馈路径。"""
    td = build_finish_tool(_DummyData)
    run_ctx = _fake_run_ctx()

    msg = await td.func(
        status="definitely_not_valid", summary="坏状态",
        data={"answer": "42", "confidence": 0.5},
        run_ctx=run_ctx,
    )

    assert "参数错误" in msg
    assert "result" not in run_ctx.state


def test_finish_schema_embeds_output_schema_under_data():
    td = build_finish_tool(_DummyData)
    schema = td.to_openai_schema()
    props = schema["function"]["parameters"]["properties"]
    assert set(["status", "summary", "data"]) <= set(props.keys())
    assert "answer" in props["data"]["properties"]
    assert "confidence" in props["data"]["properties"]
