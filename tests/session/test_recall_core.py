# tests/test_recall_core.py
import pytest

from harness.memory.instructions import MemoryConfig
from harness.memory.models import MemoryRecord
from harness.memory.recall import (
    SurfacedMemory,
    _recall_failure_streaks,
    maybe_recall,
    recall_circuit_failure_count,
    reset_recall_circuit,
)
from harness.memory.store import FileMemoryStore
from harness.message_id import tag_message
from tests.memory.fakes import FakeLLMClient, RaisingLLMClient, text_finish


@pytest.fixture(autouse=True)
def _clean_circuit_state():
    _recall_failure_streaks.clear()
    yield
    _recall_failure_streaks.clear()


@pytest.fixture
def memory_store(tmp_path):
    return FileMemoryStore(tmp_path / "memories")


@pytest.fixture
def memory_config(tmp_path):
    return MemoryConfig(
        memory_dir=tmp_path / "memories", recall_top_k=3, recall_min_memories=2,
    )


def _seed(store: FileMemoryStore, n: int, prefix: str = "mem") -> list[str]:
    names = []
    for i in range(n):
        name = f"{prefix}_{i}"
        store.write(MemoryRecord(
            name=name, type="user", description=f"第{i}条记忆的描述", content=f"正文{i}",
        ))
        names.append(name)
    return names


async def test_disabled_when_top_k_is_none(memory_store, memory_config):
    memory_config.recall_top_k = None
    _seed(memory_store, 5)
    llm = FakeLLMClient([])
    outcome = await maybe_recall(
        session_id="s1", task="帮我订机票", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert llm.calls == []


async def test_below_min_memories_skips(memory_store, memory_config):
    _seed(memory_store, 1)  # 少于 recall_min_memories=2
    llm = FakeLLMClient([])
    outcome = await maybe_recall(
        session_id="s1", task="帮我订机票", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert llm.calls == []


async def test_all_candidates_already_surfaced_and_live_skips(memory_store, memory_config):
    names = _seed(memory_store, 3)
    # 模拟这些记忆之前已经推送过,且承载它们的注入消息(hid=inj_hid)
    # 仍然在当前 history 里——应该被排除,没有新东西可召回。
    injection = tag_message({"role": "user", "content": "[记忆召回] ..."})
    already = [SurfacedMemory(name=n, hid=injection["_hid"]) for n in names]
    llm = FakeLLMClient([])
    outcome = await maybe_recall(
        session_id="s1", task="帮我订机票", trace_id="t1",
        history=[injection], already_surfaced=already,
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert llm.calls == []


async def test_successful_selection_builds_injection(memory_store, memory_config):
    names = _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}", "{names[2]}"]}}')])
    outcome = await maybe_recall(
        session_id="s1", task="帮我订机票", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is not None
    assert outcome.injection_message["role"] == "user"
    assert "_hid" in outcome.injection_message  # 注入消息必须带稳定 ID
    content = outcome.injection_message["content"]
    assert names[0] in content
    assert names[2] in content
    assert names[1] not in content  # 没被选中的不该出现

    surfaced_names = {m.name for m in outcome.newly_surfaced}
    assert surfaced_names == {names[0], names[2]}
    # 所有新推送的记忆应该共享同一个 hid(它们是同一条注入消息的一部分)
    hids = {m.hid for m in outcome.newly_surfaced}
    assert hids == {outcome.injection_message["_hid"]}


async def test_already_surfaced_excluded_from_candidates(memory_store, memory_config):
    names = _seed(memory_store, 3)
    injection = tag_message({"role": "user", "content": "[记忆召回] ..."})
    already = [SurfacedMemory(name=names[0], hid=injection["_hid"])]
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[1]}"]}}')])
    await maybe_recall(
        session_id="s1", task="帮我订机票", trace_id="t1",
        history=[injection], already_surfaced=already,
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    sent = llm.calls[0]
    candidates_text = sent[-1]["content"]
    assert names[0] not in candidates_text  # 已推送且仍在场的不该出现在候选清单里
    assert names[1] in candidates_text


async def test_stale_surfaced_hid_becomes_recallable_again(memory_store, memory_config):
    """压缩联动的核心场景:某条记忆之前推送过,但承载它的注入消息的
    hid 在当前 history 里已经找不到了(被压缩吃掉)——这条记忆应该
    重新出现在候选清单里,而不是被永久排除。"""
    names = _seed(memory_store, 3)
    already = [SurfacedMemory(name=names[0], hid="hid-that-was-compacted-away")]
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}"]}}')])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1",
        history=[],  # 当前 history 里已经没有那个 hid 了
        already_surfaced=already,
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    sent = llm.calls[0]
    candidates_text = sent[-1]["content"]
    assert names[0] in candidates_text  # 重新变为候选
    assert outcome.newly_surfaced[0].name == names[0]
    assert outcome.newly_surfaced[0].hid != "hid-that-was-compacted-away"  # 拿到了新 hid


async def test_legacy_none_hid_conservatively_excluded(memory_store, memory_config):
    """v3 迁移过来的旧数据(hid=None)应该保守地当作"仍然有效",
    不允许重新召回——宁可少推一次,不引入误判重推的风险。"""
    names = _seed(memory_store, 3)
    already = [SurfacedMemory(name=names[0], hid=None)]
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[1]}"]}}')])
    await maybe_recall(
        session_id="s1", task="任务", trace_id="t1",
        history=[], already_surfaced=already,
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    sent = llm.calls[0]
    candidates_text = sent[-1]["content"]
    assert names[0] not in candidates_text  # legacy 条目仍然被排除


async def test_hallucinated_name_is_dropped(memory_store, memory_config):
    names = _seed(memory_store, 2)
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}", "不存在的记忆名"]}}')])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert [m.name for m in outcome.newly_surfaced] == [names[0]]
    assert "不存在的记忆名" not in (outcome.injection_message or {}).get("content", "")


async def test_top_k_defensive_truncation(memory_store, memory_config):
    memory_config.recall_top_k = 1
    names = _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish(
        f'{{"selected": ["{names[0]}", "{names[1]}", "{names[2]}"]}}'
    )])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert len(outcome.newly_surfaced) == 1  # 即使模型返回超过 top_k,也截断


async def test_empty_selection_is_success_not_failure(memory_store, memory_config):
    _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish('{"selected": []}')])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert outcome.newly_surfaced == []
    assert recall_circuit_failure_count("s1") == 0  # 判定为无相关项,不是失败


async def test_markdown_fenced_json_still_parses(memory_store, memory_config):
    names = _seed(memory_store, 2)
    llm = FakeLLMClient([text_finish(f'```json\n{{"selected": ["{names[0]}"]}}\n```')])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert [m.name for m in outcome.newly_surfaced] == [names[0]]


async def test_malformed_json_fails_open(memory_store, memory_config):
    _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish("这不是 JSON,我随便说点什么")])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert recall_circuit_failure_count("s1") == 1


async def test_truncated_output_treated_as_failure(memory_store, memory_config):
    _seed(memory_store, 3)
    llm = FakeLLMClient([{
        "content": '{"selected": ["x', "tool_calls": None, "finish_reason": "length",
    }])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert recall_circuit_failure_count("s1") == 1


async def test_circuit_breaker_independent_from_extraction(memory_store, memory_config):
    from harness.memory.extraction import _failure_streaks as extraction_streaks
    extraction_streaks.clear()
    memory_config.recall_max_failures = 1
    _seed(memory_store, 3)
    raising = RaisingLLMClient()
    await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=raising,
    )
    assert recall_circuit_failure_count("s1") == 1
    assert extraction_streaks.get("s1", 0) == 0  # 提取的熔断不受影响

    calls_before = raising.call_count
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=raising,
    )
    assert raising.call_count == calls_before  # 熔断后不再真的调用
    assert outcome.injection_message is None
    extraction_streaks.clear()


async def test_reset_recall_circuit(memory_store, memory_config):
    memory_config.recall_max_failures = 1
    _seed(memory_store, 3)
    raising = RaisingLLMClient()
    await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=raising,
    )
    assert recall_circuit_failure_count("s1") == 1
    reset_recall_circuit("s1")
    assert recall_circuit_failure_count("s1") == 0


async def test_staleness_warning_included_for_old_memory(memory_store, memory_config, tmp_path):
    import os
    import time
    names = _seed(memory_store, 2)
    old_path = tmp_path / "memories" / f"{names[0]}.md"
    old_time = time.time() - 10 * 86400  # 10 天前
    os.utime(old_path, (old_time, old_time))

    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}"]}}')])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert "过期警告" in outcome.injection_message["content"]


async def test_recall_llm_override_used_instead_of_fallback(memory_store, memory_config):
    names = _seed(memory_store, 3)
    dedicated = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}"]}}')])
    fallback = FakeLLMClient([])
    memory_config.recall_llm = dedicated

    await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=fallback,
    )
    assert len(dedicated.calls) == 1
    assert fallback.calls == []


async def test_event_callback_fires_on_success_and_failure(memory_store, memory_config):
    events = []
    memory_config.on_memory_event = lambda e: events.append(e)
    names = _seed(memory_store, 3)

    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}"]}}')])
    await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert any(e["type"] == "recall_done" for e in events)

    events.clear()
    await maybe_recall(
        session_id="s2", task="任务", trace_id="t1", history=[], already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=RaisingLLMClient(),
    )
    assert any(e["type"] == "recall_failed" for e in events)


async def test_trace_id_is_reused_not_a_new_span(memory_store, memory_config):
    """选择器调用应该复用调用方传入的 trace_id,不应该自己另开一个。"""
    names = _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}"]}}')])
    await maybe_recall(
        session_id="s1", task="任务", trace_id="shared-trace-id", history=[],
        already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert llm.trace_ids == ["shared-trace-id"]
