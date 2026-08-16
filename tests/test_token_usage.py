# tests/test_token_usage.py
"""
验证 Phase 0 第一项修复:真实 token 计数。

覆盖:
  1. response.usage 存在时,优先读取真实值,token_source 标 api_usage
  2. usage 缺失时,退回字符数估算,token_source 如实标 estimated
  3. completion_tokens 此前从未被记录过(旧代码只数输入),现在补上并可验证非零
  4. tracer + SQLiteTraceStorage 全链路落库不报错、字段完整可读回
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

from harness.llm.openai_client import OpenAIClient
from harness.tracing import tracer as tracer_module
from harness.tracing.storage_sqlite import SQLiteTraceStorage


def _reset_tracer():
    tracer_module._active.clear()
    tracer_module._start_times.clear()
    tracer_module._storage = None


def test_extract_usage_prefers_real_api_usage():
    # 构造一个带真实 usage 的假 response,直接测静态方法,不碰网络
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=123, completion_tokens=45))
    prompt, completion, source = OpenAIClient._extract_usage(
        response, input_messages=[{"role": "user", "content": "x" * 999}], output_text="y" * 999,
    )
    assert (prompt, completion, source) == (123, 45, "api_usage"), \
        f"应优先使用真实 usage,实际得到 {(prompt, completion, source)}"
    print("✅ test_extract_usage_prefers_real_api_usage 通过")


def test_extract_usage_falls_back_when_usage_missing():
    response = SimpleNamespace(usage=None)   # 模拟某些兼容端点不返回 usage
    prompt, completion, source = OpenAIClient._extract_usage(
        response,
        input_messages=[{"role": "user", "content": "1234567890"}],  # 10 字符
        output_text="12345",  # 5 字符
    )
    assert source == "estimated", f"usage 缺失时应标注 estimated,实际 {source}"
    assert prompt == 10 and completion == 5, f"估算数值不对: {(prompt, completion)}"
    print("✅ test_extract_usage_falls_back_when_usage_missing 通过"
          "(旧代码从未记录过 completion_tokens,这里首次验证输出侧用量可见)")


def test_extract_usage_falls_back_when_fields_partially_missing():
    # usage 对象存在,但字段不全(某些边缘 provider 可能出现)
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=None))
    prompt, completion, source = OpenAIClient._extract_usage(
        response, input_messages=[{"role": "user", "content": "abc"}], output_text="de",
    )
    assert source == "estimated", "usage 字段不全时也应该退回估算,不能只用半份真实数据拼凑"
    print("✅ test_extract_usage_falls_back_when_fields_partially_missing 通过")


def test_tracer_and_sqlite_round_trip():
    _reset_tracer()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "trace.db")
        storage = SQLiteTraceStorage(db_path=db_path)
        tracer_module.configure_storage(storage)

        tracer_module.start_trace("t1", "session1", "task1")
        tracer_module.record_llm_call(
            trace_id="t1", prompt_tokens=200, completion_tokens=50,
            token_source="api_usage", output="final answer",
            has_tool_calls=False, duration_ms=800, reasoning="想了一下",
        )
        tracer_module.end_trace("t1", "final answer", status="success")

        rows = storage.get_traces_by_session("session1")
        assert len(rows) == 1, f"应该落库 1 条 trace,实际 {len(rows)}"
        assert rows[0]["llm_call_count"] == 1

        # 直接查 llm_calls 表,验证字段完整落库、可读回
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        call_row = conn.execute("SELECT * FROM llm_calls WHERE trace_id='t1'").fetchone()
        conn.close()
        assert call_row is not None, "llm_calls 表里没查到落库记录"
        assert call_row["prompt_tokens"] == 200
        assert call_row["completion_tokens"] == 50
        assert call_row["token_source"] == "api_usage"
        assert call_row["reasoning"] == "想了一下"

    print("✅ test_tracer_and_sqlite_round_trip 通过")


def main():
    tests = [
        test_extract_usage_prefers_real_api_usage,
        test_extract_usage_falls_back_when_usage_missing,
        test_extract_usage_falls_back_when_fields_partially_missing,
        test_tracer_and_sqlite_round_trip,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"❌ {t.__name__} 失败: {e}")
        except Exception as e:
            failed += 1
            print(f"💥 {t.__name__} 抛出意外异常: {type(e).__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"总计 {len(tests)} 项, 失败 {failed} 项")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()