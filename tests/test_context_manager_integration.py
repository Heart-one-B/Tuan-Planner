# tests/test_context_manager_integration.py
"""
Phase 2 收官批:验证"装上去之后真的转",不是零件各自合格。

覆盖:
  1. 不配置 context_config 时,Agent 行为与 Phase 1 完全一致(向后兼容)
  2. 配置后:长会话中途自动触发阈值压缩,循环不用管
  3. 超大工具结果在阶段⑤被自动卸载
  4. 溢出反应:LLM 报溢出 → 紧急压缩 → 重试成功
  5. 溢出反应耗尽:连续两次溢出 → status="overflow" 终局
  6. finish_reason=="length":不当作完成,追加续写提示后继续
  7. ContextManager 快照往返
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

from harness.agent import Agent, AnswerTermination, Budget, FinishToolTermination
from harness.context.budget import ContextBudget
from harness.context.compactor import Compactor
from harness.context.context_manager import ContextManagerConfig
from harness.context.offload import OffloadStore
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor

from tests.fake_llm import FakeLLMClient, Overflow, text_message, tool_call, tool_call_message


def _make_context_config(tmpdir: Path, **budget_overrides) -> ContextManagerConfig:
    budget = ContextBudget(
        max_tokens=100_000, output_reserve=4_000, compaction_reserve=16_000,
        compaction_threshold=0.85, min_compact_tokens=50,   # 故意调小,方便测试触发
        default_tool_result_max_chars=200,
        offload_dir=tmpdir,
        **budget_overrides,
    )
    return ContextManagerConfig(
        budget=budget, offload_store=OffloadStore(tmpdir), compactor=Compactor(max_attempts=1),
    )


async def test_without_context_config_behaves_like_phase1():
    executor = ToolExecutor()
    llm = FakeLLMClient(script=[text_message("答案")])
    agent = Agent(llm_client=llm, tool_executor=executor, system_prompt="sys")   # 不传 context_config

    outcome = await agent.run("task")
    assert outcome.status == "completed" and outcome.final_text == "答案"
    print("✅ test_without_context_config_behaves_like_phase1 通过(零配置零改动)")


async def test_threshold_compaction_triggers_mid_loop(tmpdir: Path):
    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="echo", description="回显", parameters={"x": {"type": "string"}},
        required=["x"], func=lambda x: f"echo:{x}",
    ))
    # 压缩器用独立的 FakeLLMClient,不与主循环共享脚本队列——
    # 否则压缩触发时会误取走主循环下一轮的脚本项(如 tool_call_message),
    # 导致压缩误入熔断降级路径,测不到"真实 LLM 摘要成功"这条主路径。
    compaction_llm = FakeLLMClient(script=[
        text_message("## 任务目标\n压缩摘要1") for _ in range(5)
    ])
    budget = ContextBudget(
        max_tokens=100_000, output_reserve=4_000, compaction_reserve=16_000,
        compaction_threshold=0.85, min_compact_tokens=50,
        default_tool_result_max_chars=200, offload_dir=tmpdir,
    )
    cfg = ContextManagerConfig(
        budget=budget, offload_store=OffloadStore(tmpdir),
        compactor=Compactor(llm_client=compaction_llm, max_attempts=1),
    )

    script = []
    for i in range(4):
        script.append(tool_call_message(
            [tool_call(f"c{i}", "echo", {"x": "x" * 50})], usage=(75_000, 100),
        ))
    script.append(text_message("完成", usage=(75_000, 50)))

    llm = FakeLLMClient(script=script)
    agent = Agent(llm_client=llm, tool_executor=executor, system_prompt="sys",
                 context_config=cfg, budget=Budget(max_tool_calls=10))

    events = []
    async for ev in agent.events("task"):
        events.append(ev)

    compacted = [e for e in events if e["type"] == "context_compacted"]
    assert len(compacted) >= 1, "占用早已超阈值,应该在循环中途自动触发过至少一次压缩"
    assert compacted[0]["trigger"] == "threshold"
    assert compacted[0]["degraded"] is False, \
        "压缩器有独立可用的客户端,应该走真实 LLM 摘要成功路径,不该降级"
    print(f"✅ test_threshold_compaction_triggers_mid_loop 通过"
          f"(触发了 {len(compacted)} 次压缩,真实摘要路径,循环全程无需外部干预)")


async def test_oversized_tool_result_offloaded_at_stage5(tmpdir: Path):
    executor = ToolExecutor()
    big_result = "X" * 5_000   # 远超 budget 里配的 200 字符阈值
    executor.register(ToolDefinition(
        name="big_tool", description="返回大结果", parameters={},
        required=[], func=lambda: big_result,
    ))
    cfg = _make_context_config(tmpdir)
    llm = FakeLLMClient(script=[
        tool_call_message([tool_call("c1", "big_tool", {})], usage=(100, 50)),
        text_message("好的", usage=(200, 20)),
    ])
    agent = Agent(llm_client=llm, tool_executor=executor, system_prompt="sys", context_config=cfg)

    outcome = await agent.run("task")
    all_text = str(outcome.messages)
    assert "结果过大已卸载" in all_text, "超大结果应在阶段⑤被自动卸载"
    assert big_result not in all_text, "原始大结果不该留在消息历史里"
    # 第二轮发给模型的 messages 里应该是卸载后的预览,不是原文
    second_call_msgs = str(llm.calls[1]["messages"])
    assert big_result not in second_call_msgs
    print("✅ test_oversized_tool_result_offloaded_at_stage5 通过(工具结果自动卸载,无需手动干预)")


async def test_overflow_recovers_after_one_emergency_compaction(tmpdir: Path):
    cfg = _make_context_config(tmpdir)
    executor = ToolExecutor()
    llm = FakeLLMClient(script=[
        Overflow(),                       # 第一次调用直接报溢出
        text_message("恢复后的答案", usage=(500, 30)),   # 紧急压缩后重试,成功
    ])
    agent = Agent(llm_client=llm, tool_executor=executor, system_prompt="sys", context_config=cfg)

    events = []
    async for ev in agent.events("task"):
        events.append(ev)

    notices = [e for e in events if e["type"] == "notice"]
    compacted = [e for e in events if e["type"] == "context_compacted" and e["trigger"] == "overflow"]
    outcomes = [e["outcome"] for e in events if e["type"] == "outcome"]

    assert len(compacted) == 1, "应该恰好触发一次紧急压缩"
    assert outcomes[0].status == "completed", "紧急压缩后重试应该成功完成"
    assert outcomes[0].final_text == "恢复后的答案"
    print(f"✅ test_overflow_recovers_after_one_emergency_compaction 通过"
          f"(溢出→紧急压缩→重试成功,notice数={len(notices)})")


async def test_overflow_terminates_after_second_failure(tmpdir: Path):
    cfg = _make_context_config(tmpdir)
    executor = ToolExecutor()
    llm = FakeLLMClient(script=[
        Overflow(),   # 第一次溢出 → 触发紧急压缩,重试
        Overflow(),   # 重试后依然溢出 → 不再给第二次机会,终止
    ])
    agent = Agent(llm_client=llm, tool_executor=executor, system_prompt="sys", context_config=cfg)

    outcome = await agent.run("task")
    assert outcome.status == "overflow", \
        f"连续两次溢出应该终止为 overflow status,实际 {outcome.status}"
    print(f"✅ test_overflow_terminates_after_second_failure 通过"
          f"(status 如实标注为 overflow,不与 exhausted 混淆)")


async def test_run_result_maps_overflow_to_error():
    executor = ToolExecutor()
    llm = FakeLLMClient(script=[Overflow(), Overflow()])
    agent = Agent(llm_client=llm, tool_executor=executor, system_prompt="sys")  # 无 context_config

    result = await agent.run_result("task")
    assert result.status == "error"
    assert "上下文窗口" in result.summary or "拆分" in result.summary
    print("✅ test_run_result_maps_overflow_to_error 通过"
          "(无 ContextManager 时溢出无法自救,run_result 正确映射为 error)")


async def test_length_truncation_not_treated_as_finish():
    executor = ToolExecutor()
    llm = FakeLLMClient(script=[
        text_message("这段话说到一半就被截断了", finish_reason="length"),
        text_message("续写完成的完整答案", finish_reason="stop"),
    ])
    agent = Agent(llm_client=llm, tool_executor=executor, system_prompt="sys")

    outcome = await agent.run("task")
    assert outcome.status == "completed"
    assert outcome.final_text == "续写完成的完整答案", \
        "被截断的响应不该被当作最终答案,应该催续写并等到真正 stop 的那次"
    assert len(llm.calls) == 2, f"应该发起了续写调用,实际调用次数 {len(llm.calls)}"
    nudge_text = str(llm.calls[1]["messages"][-1])
    assert "被截断" in nudge_text or "继续" in nudge_text
    print("✅ test_length_truncation_not_treated_as_finish 通过(截断响应被正确识别并催续写)")


async def test_snapshot_roundtrip(tmpdir: Path):
    from harness.context.context_manager import ContextManager

    cfg = _make_context_config(tmpdir)
    cm = cfg.build()
    cm.init([{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}], prefix_len=2)
    cm.append({"role": "assistant", "content": "hi"})
    cm.note_api_usage(prompt_tokens=1234)

    snap = cm.to_snapshot()
    assert snap.total_tokens == 1234 and snap.token_source == "api_usage"

    cm2 = ContextManager.from_snapshot(
        snap, cfg.budget, cfg.offload_store, cfg.compactor, prefix_len=2,
    )
    assert cm2.messages == cm.messages
    assert cm2.counter.source == "estimated", \
        "跨进程恢复后 api_usage 的连续性无法保证,应重新估算(如实,不是缺陷)"
    print("✅ test_snapshot_roundtrip 通过")


async def main():
    failed = 0
    plain_tests = [test_without_context_config_behaves_like_phase1,
                   test_run_result_maps_overflow_to_error,
                   test_length_truncation_not_treated_as_finish]
    for t in plain_tests:
        try:
            await t()
        except AssertionError as e:
            failed += 1; print(f"❌ {t.__name__} 失败: {e}")
        except Exception as e:
            failed += 1; print(f"💥 {t.__name__}: {type(e).__name__}: {e}")

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        dir_tests = [
            test_threshold_compaction_triggers_mid_loop,
            test_oversized_tool_result_offloaded_at_stage5,
            test_overflow_recovers_after_one_emergency_compaction,
            test_overflow_terminates_after_second_failure,
            test_snapshot_roundtrip,
        ]
        for t in dir_tests:
            try:
                await t(tmpdir / t.__name__)
            except AssertionError as e:
                failed += 1; print(f"❌ {t.__name__} 失败: {e}")
            except Exception as e:
                failed += 1; print(f"💥 {t.__name__}: {type(e).__name__}: {e}")

    total = len(plain_tests) + 5
    print(f"\n{'='*50}\n总计 {total} 项, 失败 {failed} 项")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())