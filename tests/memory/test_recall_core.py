# tests/test_recall_core.py
import pytest

from harness.memory.instructions import MemoryConfig
from harness.memory.models import MemoryRecord
from harness.memory.recall import (
    _recall_failure_streaks,
    maybe_recall,
    recall_circuit_failure_count,
    reset_recall_circuit,
)
from harness.memory.store import FileMemoryStore
from tests.memory.fakes import FakeLLMClient, text_finish,RaisingLLMClient


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
        session_id="s1", task="帮我订机票", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert llm.calls == []


async def test_below_min_memories_skips(memory_store, memory_config):
    _seed(memory_store, 1)  # 少于 recall_min_memories=2
    llm = FakeLLMClient([])
    outcome = await maybe_recall(
        session_id="s1", task="帮我订机票", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert llm.calls == []


async def test_all_candidates_already_surfaced_skips(memory_store, memory_config):
    names = _seed(memory_store, 3)
    llm = FakeLLMClient([])
    outcome = await maybe_recall(
        session_id="s1", task="帮我订机票", trace_id="t1", already_surfaced=names,
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert llm.calls == []


async def test_successful_selection_builds_injection(memory_store, memory_config):
    names = _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}", "{names[2]}"]}}')])
    outcome = await maybe_recall(
        session_id="s1", task="帮我订机票", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is not None
    assert outcome.injection_message["role"] == "user"
    content = outcome.injection_message["content"]
    assert names[0] in content
    assert names[2] in content
    assert names[1] not in content  # 没被选中的不该出现
    assert set(outcome.newly_surfaced) == {names[0], names[2]}


async def test_already_surfaced_excluded_from_candidates(memory_store, memory_config):
    names = _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[1]}"]}}')])
    await maybe_recall(
        session_id="s1", task="帮我订机票", trace_id="t1", already_surfaced=[names[0]],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    sent = llm.calls[0]
    candidates_text = sent[-1]["content"]
    assert names[0] not in candidates_text  # 已推送过的不该出现在候选清单里
    assert names[1] in candidates_text


async def test_hallucinated_name_is_dropped(memory_store, memory_config):
    names = _seed(memory_store, 2)
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}", "不存在的记忆名"]}}')])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.newly_surfaced == [names[0]]
    assert "不存在的记忆名" not in (outcome.injection_message or {}).get("content", "")


async def test_top_k_defensive_truncation(memory_store, memory_config):
    memory_config.recall_top_k = 1
    names = _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish(
        f'{{"selected": ["{names[0]}", "{names[1]}", "{names[2]}"]}}'
    )])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert len(outcome.newly_surfaced) == 1  # 即使模型返回超过 top_k,也截断


async def test_empty_selection_is_success_not_failure(memory_store, memory_config):
    _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish('{"selected": []}')])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.injection_message is None
    assert outcome.newly_surfaced == []
    assert recall_circuit_failure_count("s1") == 0  # 判定为无相关项,不是失败


async def test_markdown_fenced_json_still_parses(memory_store, memory_config):
    names = _seed(memory_store, 2)
    llm = FakeLLMClient([text_finish(f'```json\n{{"selected": ["{names[0]}"]}}\n```')])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert outcome.newly_surfaced == [names[0]]


async def test_malformed_json_fails_open(memory_store, memory_config):
    _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish("这不是 JSON,我随便说点什么")])
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
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
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
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
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=raising,
    )
    assert recall_circuit_failure_count("s1") == 1
    assert extraction_streaks.get("s1", 0) == 0  # 提取的熔断不受影响

    calls_before = raising.call_count
    outcome = await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
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
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
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
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert "过期警告" in outcome.injection_message["content"]


async def test_recall_llm_override_used_instead_of_fallback(memory_store, memory_config):
    names = _seed(memory_store, 3)
    dedicated = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}"]}}')])
    fallback = FakeLLMClient([])
    memory_config.recall_llm = dedicated

    await maybe_recall(
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
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
        session_id="s1", task="任务", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert any(e["type"] == "recall_done" for e in events)

    events.clear()
    await maybe_recall(
        session_id="s2", task="任务", trace_id="t1", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config,
        fallback_llm_client=RaisingLLMClient(),
    )
    assert any(e["type"] == "recall_failed" for e in events)


async def test_trace_id_is_reused_not_a_new_span(memory_store, memory_config):
    """选择器调用应该复用调用方传入的 trace_id,不应该自己另开一个。"""
    names = _seed(memory_store, 3)
    llm = FakeLLMClient([text_finish(f'{{"selected": ["{names[0]}"]}}')])
    await maybe_recall(
        session_id="s1", task="任务", trace_id="shared-trace-id", already_surfaced=[],
        memory_store=memory_store, memory_config=memory_config, fallback_llm_client=llm,
    )
    assert llm.trace_ids == ["shared-trace-id"]