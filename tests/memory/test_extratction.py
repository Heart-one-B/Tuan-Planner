# tests/test_extraction_core.py
import pytest

from harness.memory.extraction import (
    ExtractionBookkeeping,
    _failure_streaks,
    circuit_failure_count,
    maybe_extract,
    reset_circuit,
)
from harness.memory.instructions import MemoryConfig
from harness.memory.store import FileMemoryStore
from tests.memory.fakes import FakeLLMClient, RaisingLLMClient, text_finish, tool_call_then


@pytest.fixture(autouse=True)
def _clean_circuit_state():
    """熔断计数是模块级字典,测试之间必须手动清场,否则会串台。"""
    _failure_streaks.clear()
    yield
    _failure_streaks.clear()


@pytest.fixture
def memory_store(tmp_path):
    return FileMemoryStore(tmp_path / "memories")


@pytest.fixture
def memory_config(tmp_path):
    return MemoryConfig(memory_dir=tmp_path / "memories", extraction_every_n_runs=3)


def _messages(n: int) -> list[dict]:
    """构造 n 条简单的 user/assistant 交替消息,模拟已经跑过的对话历史。"""
    out = [{"role": "system", "content": "系统提示"}]
    for i in range(n):
        out.append({"role": "user", "content": f"用户第{i}句"})
        out.append({"role": "assistant", "content": f"回复第{i}句"})
    return out


async def test_disabled_when_every_n_runs_is_none(memory_store, memory_config):
    memory_config.extraction_every_n_runs = None
    bk = ExtractionBookkeeping(watermark=0, runs_since=5)
    result = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=FakeLLMClient([]),
    )
    assert result is bk  # 原样传回,未开启时不做任何事


async def test_below_threshold_just_increments_counter(memory_store, memory_config):
    bk = ExtractionBookkeeping(watermark=0, runs_since=0)
    llm = FakeLLMClient([])  # 不该被调用
    result = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert result.runs_since == 1
    assert result.watermark == 0
    assert llm.calls == []  # 没达到阈值(3),不应该真的调用模型


async def test_reaching_threshold_triggers_extraction_and_resets(memory_store, memory_config):
    # extraction_every_n_runs=3,runs_since 从 2 开始,这次 +1=3 达到阈值
    bk = ExtractionBookkeeping(watermark=0, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    all_msgs = _messages(3)
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert len(llm.calls) == 1  # 真的调用了一次
    assert result.runs_since == 0  # 提取成功后计数清零
    assert result.watermark == len(all_msgs)  # 水位线推进到当前末尾


async def test_extraction_writes_memory_via_tool_call(memory_store, memory_config, tmp_path):
    bk = ExtractionBookkeeping(watermark=0, runs_since=2)
    llm = FakeLLMClient([
        tool_call_then("memory_write", {
            "name": "user_prefers_window_seat", "type": "user",
            "description": "用户偏好靠窗座位",
            "content": "用户明确表示订机票/高铁时偏好靠窗座位。",
        }),
        text_finish("已记录"),
    ])
    all_msgs = _messages(3)
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert result.runs_since == 0
    assert (tmp_path / "memories" / "user_prefers_window_seat.md").is_file()


async def test_extraction_failure_does_not_advance_watermark(memory_store, memory_config):
    bk = ExtractionBookkeeping(watermark=0, runs_since=2)
    raising = RaisingLLMClient()
    all_msgs = _messages(3)
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert result.watermark == bk.watermark  # 水位线不推进
    assert result.runs_since >= memory_config.extraction_every_n_runs  # 下次立刻重试
    assert circuit_failure_count("s1") == 1


async def test_circuit_breaker_opens_after_max_failures(memory_store, memory_config):
    memory_config.extraction_max_failures = 2
    raising = RaisingLLMClient()
    bk = ExtractionBookkeeping(watermark=0, runs_since=2)

    # 第一次失败
    bk = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert circuit_failure_count("s1") == 1

    # 第二次失败,达到 max_failures=2,熔断开启
    bk = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert circuit_failure_count("s1") == 2
    calls_before = raising.call_count

    # 第三次:熔断已开,不应该再真的调用模型
    bk = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert raising.call_count == calls_before  # 没有新的调用发生,被熔断拦住了


async def test_reset_circuit_allows_retry(memory_store, memory_config):
    memory_config.extraction_max_failures = 1
    raising = RaisingLLMClient()
    bk = ExtractionBookkeeping(watermark=0, runs_since=2)
    bk = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert circuit_failure_count("s1") == 1

    reset_circuit("s1")
    assert circuit_failure_count("s1") == 0

    # 复位后,换一个正常工作的 llm,应该能重新真正尝试提取
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    bk = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert len(llm.calls) == 1


async def test_empty_segment_advances_watermark_without_calling_model(memory_store, memory_config):
    all_msgs = _messages(3)
    # watermark 已经指到末尾,没有新内容
    bk = ExtractionBookkeeping(watermark=len(all_msgs), runs_since=2)
    llm = FakeLLMClient([])
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert llm.calls == []
    assert result.watermark == len(all_msgs)
    assert result.runs_since == 0


async def test_watermark_out_of_bounds_degrades_to_full_window(memory_store, memory_config):
    """模拟压缩把消息列表缩短后,旧水位线越界的情形——应该退化为
    提取整个当前窗口,而不是抛异常或跳过。"""
    all_msgs = _messages(2)  # 压缩后只剩 5 条(1 system + 2*2)
    bk = ExtractionBookkeeping(watermark=999, runs_since=2)  # 越界的旧水位线
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert len(llm.calls) == 1  # 没有因越界而跳过,退化后依然真的提取了
    assert result.watermark == len(all_msgs)


async def test_recall_injection_message_excluded_from_extraction_segment(
    memory_store, memory_config,
):
    """第二期和第三期交互场景:召回(第三期)往 history 里塞了一条
    "记忆全文"的注入消息,提取(第二期)不该把它当成新对话内容重新
    扫一遍——这条消息应该被过滤掉,不出现在喂给提取 Agent 的 segment
    里(体现在:提取 Agent 收到的最后一条消息不含召回标记)。"""
    from harness.memory.instructions import MEMORY_RECALL_MARKER

    all_msgs = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "用户第一句"},
        {"role": "user", "content": f"{MEMORY_RECALL_MARKER} 这里是召回的记忆全文,"
                                    f"包含用户过敏信息等敏感细节"},
        {"role": "assistant", "content": "回复"},
    ]
    bk = ExtractionBookkeeping(watermark=0, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    sent_messages = llm.calls[0]
    joined = "\n".join(
        (m.get("content") or "") if isinstance(m, dict) else ""
        for m in sent_messages
    )
    assert MEMORY_RECALL_MARKER not in joined
    assert "用户第一句" in joined  # 其余正常对话内容不受影响


async def test_extraction_task_contains_do_not_save_and_current_index(
    memory_store, memory_config,
):
    """去重指令拼装:提取任务里应该带上负面清单,以及当前记忆索引
    (哪怕索引为空也要有对应文案,不能悄悄漏掉)。"""
    from harness.memory.models import MemoryRecord
    memory_store.write(MemoryRecord(
        name="existing_pref", type="user",
        description="已有的一条用户偏好记忆", content="正文",
    ))

    bk = ExtractionBookkeeping(watermark=0, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert len(llm.calls) == 1
    sent_messages = llm.calls[0]
    task_text = sent_messages[-1]["content"]
    assert "不要写入以下内容" in task_text
    assert "existing_pref" in task_text  # 现有索引被拼进了提取指令


async def test_extraction_llm_override_used_instead_of_fallback(memory_store, memory_config):
    """MemoryConfig.extraction_llm 配置了专门的便宜模型时,应该用它,
    不用调用方传入的 fallback_llm_client。"""
    dedicated = FakeLLMClient([text_finish("本轮无新增记忆")])
    fallback = FakeLLMClient([])  # 不该被用到
    memory_config.extraction_llm = dedicated

    bk = ExtractionBookkeeping(watermark=0, runs_since=2)
    await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=fallback,
    )
    assert len(dedicated.calls) == 1
    assert fallback.calls == []


async def test_extraction_event_callback_fires_on_success_and_failure(
    memory_store, memory_config,
):
    events = []
    memory_config.on_memory_event = lambda e: events.append(e)

    bk = ExtractionBookkeeping(watermark=0, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    types = [e["type"] for e in events]
    assert "extraction_started" in types
    assert "extraction_done" in types

    events.clear()
    bk2 = ExtractionBookkeeping(watermark=0, runs_since=2)
    await maybe_extract(
        session_id="s2", all_messages=_messages(3), bookkeeping=bk2,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=RaisingLLMClient(),
    )
    types2 = [e["type"] for e in events]
    assert "extraction_failed" in types2


async def test_buggy_callback_does_not_break_extraction(memory_store, memory_config):
    """回调自己抛异常,不能拖垮提取流程本身。"""
    def bad_callback(event):
        raise ValueError("宿主回调写崩了")

    memory_config.on_memory_event = bad_callback
    bk = ExtractionBookkeeping(watermark=0, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    result = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert result.runs_since == 0  # 提取本身依然正常完成