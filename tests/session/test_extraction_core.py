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
from harness.message_id import tag_message
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
    """构造 n 条简单的 user/assistant 交替消息,模拟已经跑过的对话历史。
    每条 user 消息都打上稳定 ID(模拟真实 Agent._messages() 对 task
    消息的打标行为),第一条 user 消息的 hid 作为"提取起点"供测试使用。"""
    out = [{"role": "system", "content": "系统提示"}]
    for i in range(n):
        out.append(tag_message({"role": "user", "content": f"用户第{i}句"}))
        out.append({"role": "assistant", "content": f"回复第{i}句"})
    return out


def _first_user_hid(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return m["_hid"]
    raise AssertionError("没有找到任何 user 消息")


async def test_disabled_when_every_n_runs_is_none(memory_store, memory_config):
    memory_config.extraction_every_n_runs = None
    bk = ExtractionBookkeeping(boundary_hid=None, runs_since=5)
    result = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=FakeLLMClient([]),
    )
    assert result is bk  # 原样传回,未开启时不做任何事


async def test_below_threshold_just_increments_counter(memory_store, memory_config):
    bk = ExtractionBookkeeping(boundary_hid=None, runs_since=0)
    llm = FakeLLMClient([])  # 不该被调用
    result = await maybe_extract(
        session_id="s1", all_messages=_messages(3), bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert result.runs_since == 1
    assert result.boundary_hid is None
    assert llm.calls == []  # 没达到阈值(3),不应该真的调用模型


async def test_reaching_threshold_triggers_extraction_and_resets(memory_store, memory_config):
    # extraction_every_n_runs=3,runs_since 从 2 开始,这次 +1=3 达到阈值
    all_msgs = _messages(3)
    boundary = _first_user_hid(all_msgs)
    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert len(llm.calls) == 1  # 真的调用了一次
    assert result.runs_since == 0  # 提取成功后计数清零
    assert result.boundary_hid is None  # 起点重置,等待下次 run 重新锚定


async def test_extraction_writes_memory_via_tool_call(memory_store, memory_config, tmp_path):
    all_msgs = _messages(3)
    boundary = _first_user_hid(all_msgs)
    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)
    llm = FakeLLMClient([
        tool_call_then("memory_write", {
            "name": "user_prefers_window_seat", "type": "user",
            "description": "用户偏好靠窗座位",
            "content": "用户明确表示订机票/高铁时偏好靠窗座位。",
        }),
        text_finish("已记录"),
    ])
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert result.runs_since == 0
    assert (tmp_path / "memories" / "user_prefers_window_seat.md").is_file()


async def test_extraction_failure_does_not_advance_boundary(memory_store, memory_config):
    all_msgs = _messages(3)
    boundary = _first_user_hid(all_msgs)
    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)
    raising = RaisingLLMClient()
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert result.boundary_hid == boundary  # 起点不推进
    assert result.runs_since >= memory_config.extraction_every_n_runs  # 下次立刻重试
    assert circuit_failure_count("s1") == 1


async def test_circuit_breaker_opens_after_max_failures(memory_store, memory_config):
    memory_config.extraction_max_failures = 2
    raising = RaisingLLMClient()
    all_msgs = _messages(3)
    boundary = _first_user_hid(all_msgs)
    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)

    bk = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert circuit_failure_count("s1") == 1

    bk = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert circuit_failure_count("s1") == 2
    calls_before = raising.call_count

    bk = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert raising.call_count == calls_before  # 熔断拦住了,没有新调用


async def test_reset_circuit_allows_retry(memory_store, memory_config):
    memory_config.extraction_max_failures = 1
    raising = RaisingLLMClient()
    all_msgs = _messages(3)
    boundary = _first_user_hid(all_msgs)
    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)
    bk = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=raising,
    )
    assert circuit_failure_count("s1") == 1

    reset_circuit("s1")
    assert circuit_failure_count("s1") == 0

    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    bk = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert len(llm.calls) == 1


async def test_empty_segment_resets_boundary_without_calling_model(memory_store, memory_config):
    # 空消息列表模拟"没有任何新内容"的场景,不必为空内容真调一次模型。
    bk = ExtractionBookkeeping(boundary_hid=None, runs_since=2)
    llm = FakeLLMClient([])
    result = await maybe_extract(
        session_id="s1", all_messages=[], bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert llm.calls == []
    assert result.boundary_hid is None
    assert result.runs_since == 0


async def test_boundary_lost_degrades_to_full_window(memory_store, memory_config):
    """模拟压缩把锚点所在的消息段吞掉后,boundary_hid 在当前消息列表
    里再也找不到的情形——应该退化为提取整个当前窗口,而不是抛异常
    或跳过。"""
    all_msgs = _messages(2)  # 压缩后只剩这些,原锚点已经不在里面了
    bk = ExtractionBookkeeping(boundary_hid="hid-that-no-longer-exists", runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert len(llm.calls) == 1  # 没有因锚点丢失而跳过,退化后依然真的提取了
    assert result.boundary_hid is None


async def test_off_by_one_regression_index_message_shifts_list(memory_store, memory_config):
    """审查发现的真实 bug 回归测试:如果提取用下标定位(旧方案),
    两次 run 之间因为记忆条数从 0 变为非 0、索引消息出现,会导致
    整个消息列表平移一位,旧下标失真、可能漏提取一条。hid 方案下
    这个场景应该完全不受影响——不管列表有没有平移,只要锚点那条
    消息还在,就能正确定位。"""
    # 模拟"上一轮"结束时的消息列表(没有索引消息,因为当时还没有
    # 任何记忆),第一条 user 消息是本次提取应该覆盖的起点。
    without_index = [
        {"role": "system", "content": "系统提示"},
        tag_message({"role": "user", "content": "第一句(提取起点)"}),
        {"role": "assistant", "content": "回复1"},
    ]
    boundary = without_index[1]["_hid"]

    # 模拟"这一轮"结束时:期间产生了第一条记忆,索引消息被插入到了
    # system 之后、第一句用户消息之前——整个列表相对上一轮向后平移
    # 了一位。用下标方案存的旧水位线(比如"1")现在会指向索引消息
    # 本身而不是那条 user 消息,这正是审查发现的 bug。
    with_index_inserted = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "[记忆索引] 假设这是新出现的索引消息"},
        without_index[1],  # 同一条 user 消息对象,hid 不变
        without_index[2],
        tag_message({"role": "user", "content": "第二句"}),
        {"role": "assistant", "content": "回复2"},
    ]

    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    await maybe_extract(
        session_id="s1", all_messages=with_index_inserted, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    sent_messages = llm.calls[0]
    joined = "\n".join(
        (m.get("content") or "") if isinstance(m, dict) else "" for m in sent_messages
    )
    # 正确行为:segment 应该从"第一句"开始(它是锚点本身),索引消息
    # (在锚点之前)不该被包含在提取内容里。
    assert "第一句" in joined
    assert "第二句" in joined
    assert "[记忆索引]" not in joined


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

    all_msgs = _messages(3)
    boundary = _first_user_hid(all_msgs)
    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
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

    all_msgs = _messages(3)
    boundary = _first_user_hid(all_msgs)
    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)
    await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
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

    all_msgs = _messages(3)
    boundary = _first_user_hid(all_msgs)
    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    types = [e["type"] for e in events]
    assert "extraction_started" in types
    assert "extraction_done" in types

    events.clear()
    all_msgs2 = _messages(3)
    boundary2 = _first_user_hid(all_msgs2)
    bk2 = ExtractionBookkeeping(boundary_hid=boundary2, runs_since=2)
    await maybe_extract(
        session_id="s2", all_messages=all_msgs2, bookkeeping=bk2,
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
    all_msgs = _messages(3)
    boundary = _first_user_hid(all_msgs)
    bk = ExtractionBookkeeping(boundary_hid=boundary, runs_since=2)
    llm = FakeLLMClient([text_finish("本轮无新增记忆")])
    result = await maybe_extract(
        session_id="s1", all_messages=all_msgs, bookkeeping=bk,
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=llm,
    )
    assert result.runs_since == 0  # 提取本身依然正常完成


async def test_recall_injection_message_excluded_from_extraction_segment(
    memory_store, memory_config,
):
    """第二期和第三期交互场景:召回(第三期)往 history 里塞了一条
    "记忆全文"的注入消息,提取(第二期)不该把它当成新对话内容重新
    扫一遍——这条消息应该被过滤掉,不出现在喂给提取 Agent 的 segment
    里(体现在:提取 Agent 收到的最后一条消息不含召回标记)。"""
    from harness.memory.instructions import MEMORY_RECALL_MARKER

    first_user = tag_message({"role": "user", "content": "用户第一句"})
    all_msgs = [
        {"role": "system", "content": "系统提示"},
        first_user,
        {"role": "user", "content": f"{MEMORY_RECALL_MARKER} 这里是召回的记忆全文,"
                                    f"包含用户过敏信息等敏感细节"},
        {"role": "assistant", "content": "回复"},
    ]
    bk = ExtractionBookkeeping(boundary_hid=first_user["_hid"], runs_since=2)
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
