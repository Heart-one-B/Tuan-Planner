# tests/test_agent_extraction_integration.py
import pytest

from harness.agent import Agent
from harness.memory.extraction import _failure_streaks
from harness.memory.instructions import MemoryConfig
from harness.snapshot import FileSnapshotStore
from harness.tools.tool_executor import ToolExecutor
from tests.memory.fakes import FakeLLMClient, text_finish, tool_call_then


@pytest.fixture(autouse=True)
def _clean_circuit_state():
    _failure_streaks.clear()
    yield
    _failure_streaks.clear()


def _memory_config(tmp_path, **overrides):
    kwargs = dict(memory_dir=tmp_path / "memories", extraction_every_n_runs=2)
    kwargs.update(overrides)
    return MemoryConfig(**kwargs)


async def test_first_run_anchors_boundary_without_extracting(tmp_path):
    """extraction_every_n_runs=2:第一次 run 结束不该触发提取(只占用
    主对话自己的那一次 LLM 调用),但应该已经把提取起点锚定到这次的
    task 消息上——这是稳定 ID 方案的关键行为:锚定和"真的去提取"是
    两件独立的事,锚定必须尽早发生,不然到真正触发提取时就不知道
    该从哪条消息开始算了。"""
    memory_config = _memory_config(tmp_path)
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([text_finish("主对话回复1")])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    await agent.run("你好", session_id="sess-a")
    assert len(llm.calls) == 1  # 只有主对话这一次调用,没有额外的提取调用

    snap = snapshot_store.load_latest("sess-a")
    assert snap.runs_since_extraction == 1
    assert snap.extraction_boundary_hid is not None  # 已经锚定,但还没真的提取


async def test_auto_mode_extracts_after_n_runs(tmp_path):
    memory_config = _memory_config(tmp_path)  # every_n_runs=2
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([
        text_finish("主对话回复1"),
        text_finish("主对话回复2"),
        text_finish("本轮无新增记忆"),  # 提取 Agent 的响应
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    await agent.run("第一句", session_id="sess-b")
    await agent.run("第二句", session_id="sess-b")

    assert len(llm.calls) == 3  # 两次主对话 + 一次提取

    snap = snapshot_store.load_latest("sess-b")
    assert snap.runs_since_extraction == 0  # 提取成功后清零
    assert snap.extraction_boundary_hid is None  # 重置,等待下次 run 重新锚定


async def test_auto_mode_extraction_writes_real_memory(tmp_path):
    memory_config = _memory_config(tmp_path, extraction_every_n_runs=1)
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([
        text_finish("主对话回复"),
        tool_call_then("memory_write", {
            "name": "user_role", "type": "user",
            "description": "用户是十年 Go 后端",
            "content": "用户自我介绍是十年 Go 后端工程师,刚接触前端。",
        }),
        text_finish("已记录"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    await agent.run("我是十年 Go 后端,刚接触前端", session_id="sess-c")

    assert (tmp_path / "memories" / "user_role.md").is_file()


async def test_boundary_segment_spans_multiple_threaded_runs(tmp_path):
    """真实用法:宿主在每次 run() 之间用 snapshot.resume_history() 把
    历史串起来(这是本仓库一贯的显式约定,Agent 自己不会偷偷帮你做)。
    这个测试要验证的不是"触发计数对不对"(上面几个测试已经够了),
    而是提取起点锚定后切出来的 segment 是否真的覆盖了两轮对话的
    内容——如果宿主忘了串联历史,extraction_every_n_runs 依然会照常
    触发,但每次看到的都只是"这一轮"的孤立内容,起点锚定机制就形同
    虚设。
    """
    memory_config = _memory_config(tmp_path, extraction_every_n_runs=2)
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([
        text_finish("主对话回复1"),
        text_finish("主对话回复2"),
        text_finish("本轮无新增记忆"),  # 提取 Agent 的响应
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )

    await agent.run("第一句你好", session_id="sess-h")
    prior = snapshot_store.load_latest("sess-h")
    # 真实宿主的正确用法:把上一轮的 history 接着传给下一轮
    await agent.run("第二句你好", history=prior.resume_history(), session_id="sess-h")

    assert len(llm.calls) == 3  # 两次主对话 + 一次提取

    # 提取 Agent 收到的最后一次调用里,history 部分应该同时包含
    # 第一轮和第二轮的用户发言——不是只有最后一轮。
    extraction_call_messages = llm.calls[-1]
    joined = "\n".join(
        (m.get("content") or "") if isinstance(m, dict) else (getattr(m, "content", None) or "")
        for m in extraction_call_messages
    )
    assert "第一句你好" in joined
    assert "第二句你好" in joined


async def test_manual_mode_anchors_but_does_not_auto_trigger(tmp_path):
    """manual 模式下起点锚定依然要发生(不然宿主以后调 extract_now()
    时不知道该从哪开始),但不会自动触发真正的提取判断。"""
    memory_config = _memory_config(tmp_path, extraction_every_n_runs=1, extraction_mode="manual")
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([text_finish("主对话回复")])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    outcome = await agent.run("你好", session_id="sess-d")
    assert len(llm.calls) == 1  # manual 模式下 run() 不会自动触发提取判断

    snap = snapshot_store.load_latest("sess-d")
    assert snap.runs_since_extraction == 0  # 从未被 maybe_extract 真正判定过
    assert snap.extraction_boundary_hid is not None  # 但起点已经锚定
    return outcome


async def test_manual_mode_extract_now_writes_back_to_snapshot(tmp_path):
    memory_config = _memory_config(tmp_path, extraction_every_n_runs=1, extraction_mode="manual")
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([
        text_finish("主对话回复"),
        text_finish("本轮无新增记忆"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    outcome = await agent.run("你好", session_id="sess-e")

    # 宿主自己决定何时触发,这里模拟"拿到结果后手动调用"
    bookkeeping = await agent.extract_now("sess-e", outcome)
    assert bookkeeping.runs_since == 0
    assert bookkeeping.boundary_hid is None  # 提取成功,起点重置

    snap = snapshot_store.load_latest("sess-e")
    assert snap.runs_since_extraction == 0
    assert snap.extraction_boundary_hid is None
    # 主对话内容本身应该完好无损(extract_now 只应该动提取相关字段)
    assert snap.status == outcome.status


async def test_no_memory_config_is_fully_inert(tmp_path):
    """没配置 memory_config 时,提取相关代码路径完全不触碰
    (第一期就已经确立的老规矩,第二期不能破坏它)。"""
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([text_finish("回复")])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, name="agent",
    )
    await agent.run("你好", session_id="sess-f")
    assert len(llm.calls) == 1
    snap = snapshot_store.load_latest("sess-f")
    assert snap.extraction_boundary_hid is None
    assert snap.runs_since_extraction == 0


async def test_events_path_also_triggers_extraction(tmp_path):
    """events() 是流式入口,应该和 run() 一样接入提取,不能只接一半。"""
    memory_config = _memory_config(tmp_path, extraction_every_n_runs=1)
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    llm = FakeLLMClient([
        text_finish("主对话回复"),
        text_finish("本轮无新增记忆"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=ToolExecutor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, memory_config=memory_config, name="agent",
    )
    events = []
    async for ev in agent.events("你好", session_id="sess-g"):
        events.append(ev)

    assert len(llm.calls) == 2  # 主对话 + 提取
    snap = snapshot_store.load_latest("sess-g")
    assert snap.runs_since_extraction == 0
