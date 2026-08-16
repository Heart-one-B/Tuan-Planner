# tests/test_compactor.py
"""
Phase 2 第二批:压缩契约 + 带熔断的 Compactor。

验证的声称:
  1. 边界三规则:system+首条user永不动;最近N轮原文保留;
     切口不拆散工具调用配对
  2. 正常压缩:中间段被摘要替换,摘要保留卸载引用
  3. 熔断降级:LLM 连续失败 max_attempts 次后不再重试,
     走规则清理,degraded 如实标注
  4. 无可压中间段时如实跳过,不制造假审计
  5. focus 拼进契约而非取代契约(章节要求仍在 system prompt 里)
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from harness.context.compaction_prompt import build_compaction_messages
from harness.context.compactor import Compactor
from harness.context.offload import CLEARED_MARKER, OFFLOAD_MARKER

from tests.fake_llm import FakeLLMClient, text_message


def _asst(cid):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "function": {"name": "t", "arguments": "{}"}}]}


def _tool(cid, content="工具结果原文"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _build_history(rounds: int) -> list:
    """system + user任务 + N个工具轮次 的标准历史。"""
    msgs = [
        {"role": "system", "content": "SYSTEM_PROMPT"},
        {"role": "user", "content": "TASK_ORIGINAL"},
    ]
    for i in range(rounds):
        msgs += [_asst(f"c{i}"), _tool(f"c{i}", f"第{i}轮结果")]
    return msgs


class _AlwaysFailClient:
    def __init__(self):
        self.calls = 0

    async def call(self, trace_id, stream=False, **kwargs):
        self.calls += 1
        raise RuntimeError("模拟压缩调用失败")


async def test_boundary_prefix_and_tail_preserved():
    history = _build_history(rounds=5)
    fake = FakeLLMClient(script=[text_message("## 任务目标\n...摘要正文...")])
    compactor = Compactor()

    outcome = await compactor.compact(
        history, fallback_client=fake, trace_id="t1", keep_recent_rounds=2,
    )
    new = outcome.new_messages

    # 规则1:头部前缀原样在最前
    assert new[0]["content"] == "SYSTEM_PROMPT" and new[1]["content"] == "TASK_ORIGINAL"
    # 规则2:最近2轮(c3,c4)原文保留在尾部
    tail_text = str(new[-4:])
    assert "第3轮结果" in tail_text and "第4轮结果" in tail_text
    # 中间段(c0,c1,c2)被摘要替换,原文消失
    all_text = str(new)
    assert "第0轮结果" not in all_text and "摘要正文" in all_text
    # 规则3:工具配对不被拆散——每个存活的 tool_call_id 都有配对
    asst_ids = {tc["id"] for m in new if isinstance(m, dict) and m.get("tool_calls")
                for tc in m["tool_calls"]}
    tool_ids = {m["tool_call_id"] for m in new
                if isinstance(m, dict) and m.get("role") == "tool"}
    assert asst_ids == tool_ids == {"c3", "c4"}, \
        f"配对被切开: asst={asst_ids} tool={tool_ids}"
    assert not outcome.degraded
    print("✅ test_boundary_prefix_and_tail_preserved 通过(三条边界规则全部成立)")


async def test_summary_flows_offload_refs():
    """摘要保留卸载引用——这里验证的是数据通路:中间段里的卸载预览
    (含ref)被完整渲染进压缩器的输入。契约prompt要求保留它,
    模型是否照做属于模型行为,通路正确是我们这层能保证的部分。"""
    history = _build_history(rounds=4)
    history[5] = _tool("c1", f"{OFFLOAD_MARKER} 引用: t1/call_x.txt\n预览...")

    fake = FakeLLMClient(script=[text_message("## 产物与引用\nt1/call_x.txt: 搜索结果")])
    compactor = Compactor()
    outcome = await compactor.compact(history, fallback_client=fake,
                                      trace_id="t1", keep_recent_rounds=2)

    compaction_call = fake.calls[0]
    rendered_input = str(compaction_call["messages"])
    assert "t1/call_x.txt" in rendered_input, "卸载 ref 必须进入压缩器的输入"
    assert "t1/call_x.txt" in str(outcome.new_messages), "摘要消息里应有 ref"
    print("✅ test_summary_flows_offload_refs 通过(卸载引用的数据通路完整)")


async def test_circuit_breaker_degrades_after_max_attempts():
    history = _build_history(rounds=6)
    failing = _AlwaysFailClient()
    compactor = Compactor(max_attempts=2)

    outcome = await compactor.compact(history, fallback_client=failing,
                                      trace_id="t1", keep_recent_rounds=2)

    assert failing.calls == 2, f"熔断:应恰好尝试 max_attempts=2 次,实际 {failing.calls}"
    assert outcome.degraded is True, "降级必须如实标注"
    # 降级效果:中间段工具结果被规则清空,尾部原文完好
    all_text = str(outcome.new_messages)
    assert CLEARED_MARKER in all_text, "降级应清空中间段工具结果"
    assert "第5轮结果" in all_text and "第4轮结果" in all_text, "尾部保留区不许动"
    assert "第0轮结果" not in all_text
    # 消息结构完好:配对依然成立(assistant 骨架还在)
    asst_ids = {tc["id"] for m in outcome.new_messages
                if isinstance(m, dict) and m.get("tool_calls") for tc in m["tool_calls"]}
    tool_ids = {m["tool_call_id"] for m in outcome.new_messages
                if isinstance(m, dict) and m.get("role") == "tool"}
    assert asst_ids == tool_ids, "降级路径也不许破坏配对"
    print(f"✅ test_circuit_breaker_degrades_after_max_attempts 通过"
          f"(恰好尝试{failing.calls}次后熔断,降级清理生效且结构完好)")


async def test_no_middle_section_skips_honestly():
    history = _build_history(rounds=2)   # 全在保留区,无中间段
    failing = _AlwaysFailClient()        # 如果它被调用,说明没有正确跳过
    compactor = Compactor()

    outcome = await compactor.compact(history, fallback_client=failing,
                                      trace_id="t1", keep_recent_rounds=2)

    assert failing.calls == 0, "无可压中间段时不该发起任何 LLM 调用"
    assert outcome.new_messages == history, "应原样返回"
    assert outcome.result.dropped_message_count == 0
    print("✅ test_no_middle_section_skips_honestly 通过")


async def test_tier1_shrinks_long_conversation_text():
    """对话密集型会话的降级覆盖——被指出的缺口:旧降级只清 tool 消息,
    纯会话内容(长 assistant 推理)完全不动,降级形同虚设。"""
    long_reasoning = "推理" * 1000   # 2000字符的 assistant 长文本
    history = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "TASK"},
        {"role": "assistant", "content": long_reasoning},
        {"role": "user", "content": "继续"},
        {"role": "assistant", "content": long_reasoning},
        {"role": "user", "content": "短消息不该动"},
    ] + [_asst("c0"), _tool("c0"), _asst("c1"), _tool("c1"),
         _asst("c2"), _tool("c2")]

    failing = _AlwaysFailClient()
    compactor = Compactor(max_attempts=1)
    outcome = await compactor.compact(history, fallback_client=failing,
                                      trace_id="t1", trigger="threshold",
                                      keep_recent_rounds=2)

    assert outcome.degraded and outcome.user_notice is None, \
        "threshold 触发只走一档,不该产生用户告知"
    middle_text = str(outcome.new_messages)
    assert "压缩降级:原文 2000 字符被截断" in middle_text, "超长纯文本应被截断"
    assert "短消息不该动" in middle_text, "短消息不许碰"
    # 释放量断言:降级后确实变小了(旧版对这种历史释放量≈0)
    assert outcome.result.tokens_after < outcome.result.tokens_before * 0.6, \
        f"对话密集型历史降级应显著缩小: {outcome.result.tokens_before}->{outcome.result.tokens_after}"
    print("✅ test_tier1_shrinks_long_conversation_text 通过(缺口已补:纯会话内容也能降级)")


async def test_tier2_overflow_drops_middle_and_notifies_user():
    """overflow 触发+熔断:第二档降级——整段丢弃+双向告知(用户方案)。"""
    history = _build_history(rounds=6)
    failing = _AlwaysFailClient()
    compactor = Compactor(max_attempts=1)

    outcome = await compactor.compact(history, fallback_client=failing,
                                      trace_id="t1", trigger="overflow",
                                      keep_recent_rounds=2)

    assert outcome.degraded is True
    # 给用户的告知
    assert outcome.user_notice is not None, "第二档必须产生用户告知"
    assert "另起新会话" in outcome.user_notice and "移出上下文" in outcome.user_notice
    # 给模型的告知(历史里的占位消息,防幻觉连续性)
    all_text = str(outcome.new_messages)
    assert "已移出上下文" in all_text and "不可恢复" in all_text, "历史里必须有告知模型的占位消息"
    # 头尾完好,中间真的没了
    assert "TASK_ORIGINAL" in all_text and "第5轮结果" in all_text
    assert "第0轮结果" not in all_text and "第1轮结果" not in all_text
    # 中间段整体替换为一条消息:总消息数 = prefix2 + 占位1 + tail4
    assert len(outcome.new_messages) == 7, f"实际 {len(outcome.new_messages)} 条"
    print("✅ test_tier2_overflow_drops_middle_and_notifies_user 通过(双向告知齐全)")


async def test_explicit_prefix_len_overrides_heuristic():
    """带 history 的场景:首条 user 是旧历史不是任务,启发式会钉错;
    显式 prefix_len 由构造方声明,钉住真正该保护的前缀。"""
    history = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "旧历史的第一句"},        # 启发式会错钉这条
        {"role": "assistant", "content": "旧历史的回复"},
        {"role": "user", "content": "REAL_TASK真正的任务"},   # 真任务在这
    ] + [_asst(f"c{i}") for i in range(0)] + \
        [x for i in range(5) for x in (_asst(f"c{i}"), _tool(f"c{i}", f"第{i}轮"))]

    fake = FakeLLMClient(script=[text_message("摘要")])
    compactor = Compactor()
    outcome = await compactor.compact(
        history, fallback_client=fake, trace_id="t1",
        keep_recent_rounds=2, prefix_len=4,     # 显式钉住前4条(含真任务)
    )
    all_text = str(outcome.new_messages)
    assert "REAL_TASK" in all_text, "显式 prefix_len 必须保住真任务原文"
    assert "旧历史的第一句" in all_text, "prefix_len=4 范围内的都保留"
    assert "第0轮" not in all_text, "中间段照常被压"
    print("✅ test_explicit_prefix_len_overrides_heuristic 通过")


def test_focus_appended_not_replacing_contract():
    msgs = build_compaction_messages("对话文本", focus="重点保留API决策")
    system = msgs[0]["content"]
    assert "## 产物与引用" in system and "## 未完成的事项" in system, \
        "focus 不豁免任何固定章节"
    assert "重点保留API决策" in system, "focus 要拼进契约"
    assert system.index("## 产物与引用") < system.index("重点保留API决策"), \
        "focus 应追加在契约之后,不是插在前面稀释章节要求"
    print("✅ test_focus_appended_not_replacing_contract 通过")


async def main():
    failed = 0
    for t in [test_boundary_prefix_and_tail_preserved,
              test_summary_flows_offload_refs,
              test_circuit_breaker_degrades_after_max_attempts,
              test_no_middle_section_skips_honestly,
              test_tier1_shrinks_long_conversation_text,
              test_tier2_overflow_drops_middle_and_notifies_user,
              test_explicit_prefix_len_overrides_heuristic]:
        try:
            await t()
        except AssertionError as e:
            failed += 1; print(f"❌ {t.__name__} 失败: {e}")
        except Exception as e:
            failed += 1; print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
    try:
        test_focus_appended_not_replacing_contract()
    except AssertionError as e:
        failed += 1; print(f"❌ test_focus_appended_not_replacing_contract 失败: {e}")

    print(f"\n{'='*50}\n总计 8 项, 失败 {failed} 项")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())