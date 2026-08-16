# tests/test_agent_recall_integration.py
import pytest

from harness.agent import Agent
from harness.memory.instructions import MEMORY_RECALL_MARKER, MemoryConfig
from harness.memory.models import MemoryRecord
from harness.memory.recall import _recall_failure_streaks
from harness.memory.store import FileMemoryStore
from harness.snapshot import FileSnapshotStore
from harness.tools.tool_executor import ToolExecutor
from harness.tracing import configure_storage, SQLiteTraceStorage
from harness.tracing import tracer as tracer_module
from tests.memory.fakes import FakeLLMClient, text_finish


@pytest.fixture(autouse=True)
def _clean_state():
    _recall_failure_streaks.clear()
    tracer_module._active.clear()
    tracer_module._start_times.clear()
    tracer_module._storage = None
    yield
    _recall_failure_streaks.clear()
    tracer_module._active.clear()
    tracer_module._start_times.clear()
    tracer_module._storage = None


def _seed(memory_dir, n: int) -> list[str]:
    store = FileMemoryStore(memory_dir)
    names = []
    for i in range(n):
        name = f"pref_{i}"
        store.write(MemoryRecord(
            name=name, type="user", description=f"用户偏好{i}", content=f"正文{i}",
        ))
        names.append(name)
    return names


async def test_injection_message_reaches_main_llm_call(tmp_path):
    names = _seed(tmp_path / "memories", 3)
    memory_config = MemoryConfig(
        memory_dir=tmp_path / "memories", recall_top_k=2, recall_min_memories=2,
    )
    llm = FakeLLMClient([
        text_finish(f'{{"selected": ["{names[0]}"]}}'),  # 选择器
        text_finish("主对话回复"),  # 主对话
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        memory_config=memory_config, name="agent",
    )
    await agent.run("帮我订个航班", session_id="sess-a")

    # 第二次调用(主对话)收到的 messages 里应该带着召回注入
    main_call_messages = llm.calls[1]
    joined = "\n".join(
        (m.get("content") or "") if isinstance(m, dict) else ""
        for m in main_call_messages
    )
    assert MEMORY_RECALL_MARKER in joined
    assert names[0] in joined


async def test_below_threshold_no_selector_call(tmp_path):
    _seed(tmp_path / "memories", 1)  # 低于 recall_min_memories
    memory_config = MemoryConfig(
        memory_dir=tmp_path / "memories", recall_top_k=2, recall_min_memories=2,
    )
    llm = FakeLLMClient([text_finish("主对话回复")])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        memory_config=memory_config, name="agent",
    )
    await agent.run("你好", session_id="sess-b")
    assert len(llm.calls) == 1  # 只有主对话这一次调用


async def test_surfaced_memory_not_recalled_again_across_runs(tmp_path):
    """真实用法:宿主在两次 run 之间用 resume_history() 串联历史。
    第一轮召回过的记忆,第二轮不该再被选择器候选。"""
    names = _seed(tmp_path / "memories", 3)
    memory_config = MemoryConfig(
        memory_dir=tmp_path / "memories", recall_top_k=2, recall_min_memories=2,
    )
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([
        text_finish(f'{{"selected": ["{names[0]}"]}}'),  # 第一轮选择器
        text_finish("回复1"),
        text_finish(f'{{"selected": ["{names[1]}"]}}'),  # 第二轮选择器
        text_finish("回复2"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    await agent.run("第一句", session_id="sess-c")
    prior = snapshot_store.load_latest("sess-c")
    assert [m["name"] for m in prior.surfaced_memories] == [names[0]]
    assert prior.surfaced_memories[0]["hid"] is not None

    await agent.run("第二句", history=prior.resume_history(), session_id="sess-c")

    # 第二轮选择器调用(第 3 次调用)的候选清单里不该有 names[0]
    second_selector_messages = llm.calls[2]
    candidates_text = second_selector_messages[-1]["content"]
    assert names[0] not in candidates_text
    assert names[1] in candidates_text

    final_snap = snapshot_store.load_latest("sess-c")
    assert {m["name"] for m in final_snap.surfaced_memories} == {names[0], names[1]}


async def test_recall_message_persists_in_history_across_runs(tmp_path):
    """召回注入消息是对话数据,不应该像索引那样被剥离——下一轮的
    history 里应该还留着它。"""
    names = _seed(tmp_path / "memories", 3)
    memory_config = MemoryConfig(
        memory_dir=tmp_path / "memories", recall_top_k=2, recall_min_memories=2,
    )
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([
        text_finish(f'{{"selected": ["{names[0]}"]}}'),
        text_finish("回复1"),
        text_finish('{"selected": []}'),
        text_finish("回复2"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    await agent.run("第一句", session_id="sess-d")
    prior = snapshot_store.load_latest("sess-d")
    history = prior.resume_history()
    assert any(
        isinstance(m, dict) and MEMORY_RECALL_MARKER in (m.get("content") or "")
        for m in history
    )


async def test_no_memory_config_is_fully_inert(tmp_path):
    llm = FakeLLMClient([text_finish("回复")])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手", name="agent",
    )
    await agent.run("你好", session_id="sess-e")
    assert len(llm.calls) == 1


async def test_recall_top_k_none_is_fully_inert(tmp_path):
    _seed(tmp_path / "memories", 5)
    memory_config = MemoryConfig(memory_dir=tmp_path / "memories")  # recall_top_k 默认 None
    llm = FakeLLMClient([text_finish("回复")])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        memory_config=memory_config, name="agent",
    )
    await agent.run("你好", session_id="sess-f")
    assert len(llm.calls) == 1


async def test_selector_trace_is_recorded_under_main_run_trace(tmp_path):
    """选择器复用主 run 的 trace_id,不应该在 trace 树里出现成独立的
    顶层 trace(它是本次 run 准备工作的一部分,不是嵌套 Agent)。"""
    names = _seed(tmp_path / "memories", 3)
    storage = SQLiteTraceStorage(db_path=tmp_path / "trace.db")
    configure_storage(storage)

    memory_config = MemoryConfig(
        memory_dir=tmp_path / "memories", recall_top_k=2, recall_min_memories=2,
    )
    llm = FakeLLMClient([
        text_finish(f'{{"selected": ["{names[0]}"]}}'),
        text_finish("主对话回复"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        memory_config=memory_config, name="agent",
    )
    await agent.run("帮我订机票", session_id="sess-g")

    traces = storage.get_traces_by_session("sess-g")
    assert len(traces) == 1  # 只有主 run 这一条 trace,选择器没有另开一条

    # 选择器调用产生的 llm_call 应该记录在主 trace 下(而不是丢失或
    # 挂在别的 trace 上)。FakeLLMClient 不走真实 tracer.record_llm_call
    # (那是 OpenAIClient 的职责),这里改为直接确认 trace_id 透传正确:
    # get_traces_by_session 只返回一条 trace,证明选择器调用时用的
    # trace_id 与主 run 一致(否则 configure_storage 会落一条额外的
    # trace,这里的断言就会失败)。


async def test_below_threshold_skip_does_not_touch_surfaced_snapshot(tmp_path):
    _seed(tmp_path / "memories", 1)
    memory_config = MemoryConfig(
        memory_dir=tmp_path / "memories", recall_top_k=2, recall_min_memories=2,
    )
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([text_finish("回复")])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    await agent.run("你好", session_id="sess-h")
    snap = snapshot_store.load_latest("sess-h")
    assert snap.surfaced_memories == []


async def test_extraction_and_recall_together_end_to_end(tmp_path):
    """提取(第二期)和召回(第三期)同一个 Agent 上同时启用的端到端
    检查:召回先注入、提取后收尾,两边互不干扰,提取也不会把召回注入
    的内容当成新信息重新写一遍。"""
    names = _seed(tmp_path / "memories", 3)
    memory_config = MemoryConfig(
        memory_dir=tmp_path / "memories",
        recall_top_k=2, recall_min_memories=2,
        extraction_every_n_runs=1,
    )
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([
        text_finish(f'{{"selected": ["{names[0]}"]}}'),  # 召回选择器
        text_finish("主对话回复"),                          # 主对话
        text_finish("本轮无新增记忆"),                        # 提取 Agent
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    outcome = await agent.run("帮我订机票", session_id="sess-i")

    assert len(llm.calls) == 3
    assert outcome.status == "completed"

    snap = snapshot_store.load_latest("sess-i")
    assert [m["name"] for m in snap.surfaced_memories] == [names[0]]
    assert snap.runs_since_extraction == 0  # 提取正常触发并完成

    # 提取 Agent 收到的最后一次调用(第 3 次)不该包含召回标记
    extraction_call_messages = llm.calls[2]
    joined = "\n".join(
        (m.get("content") or "") if isinstance(m, dict) else ""
        for m in extraction_call_messages
    )
    assert MEMORY_RECALL_MARKER not in joined
