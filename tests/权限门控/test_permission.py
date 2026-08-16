# tests/test_permission.py
import pytest

from harness.agent import Agent
from harness.agent.permission import Allow, AllowAllPolicy, Defer, Deny
from harness.snapshot import FileSnapshotStore
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor
from tests.memory.fakes import FakeLLMClient, text_finish, tool_call_then


async def _pay(amount: int) -> str:
    return f"已支付 {amount} 元"


def _make_executor() -> ToolExecutor:
    ex = ToolExecutor()
    ex.register(ToolDefinition(
        name="pay", description="支付一笔款项",
        parameters={"amount": {"type": "integer"}}, required=["amount"], func=_pay,
    ))
    return ex


class _AllowPolicy:
    async def check(self, tool_name, args, run_ctx):
        return Allow()


class _DenyPolicy:
    def __init__(self, reason="风控拦截"):
        self.reason = reason

    async def check(self, tool_name, args, run_ctx):
        return Deny(reason=self.reason)


class _DeferOncePolicy:
    """第一次调用返回 Defer,记录看到的参数,供断言用。"""
    def __init__(self, approval_id="appr-1"):
        self.approval_id = approval_id
        self.seen_args: list[dict] = []

    async def check(self, tool_name, args, run_ctx):
        self.seen_args.append(args)
        return Defer(approval_id=self.approval_id, reason="金额超过阈值,需人工审批")


class _AlwaysDeferPolicy:
    """每次都挂起,用不同的 approval_id 区分,测链式挂起。"""
    def __init__(self):
        self.n = 0

    async def check(self, tool_name, args, run_ctx):
        self.n += 1
        return Defer(approval_id=f"appr-{self.n}")


async def test_no_policy_configured_is_fully_inert(tmp_path):
    """未配置 permission_policy 时行为与门控机制加入之前完全一致
    (AllowAllPolicy 是默认值)——"不配置即无感"的回归。"""
    llm = FakeLLMClient([
        tool_call_then("pay", {"amount": 100}),
        text_finish("已完成"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        name="agent",
    )
    outcome = await agent.run("帮我付 100 元", session_id="sess-a")
    assert outcome.status == "completed"
    assert "已支付 100 元" in str(outcome.messages)


async def test_allow_executes_tool_normally(tmp_path):
    llm = FakeLLMClient([
        tool_call_then("pay", {"amount": 100}),
        text_finish("已完成"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        permission_policy=_AllowPolicy(), name="agent",
    )
    outcome = await agent.run("帮我付 100 元", session_id="sess-b")
    assert outcome.status == "completed"
    assert "已支付 100 元" in str(outcome.messages)


async def test_deny_blocks_tool_and_feeds_reason_back_to_model(tmp_path):
    llm = FakeLLMClient([
        tool_call_then("pay", {"amount": 100}),
        text_finish("好的,已了解无法支付"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        permission_policy=_DenyPolicy(reason="超出预算上限"), name="agent",
    )
    outcome = await agent.run("帮我付 100 元", session_id="sess-c")
    assert outcome.status == "completed"
    assert "已支付 100 元" not in str(outcome.messages)  # 工具从未真正执行
    assert "超出预算上限" in str(outcome.messages)  # 拒绝理由回填给了模型
    second_call_text = str(llm.calls[1])
    assert "权限拒绝" in second_call_text and "超出预算上限" in second_call_text


async def test_defer_suspends_run_and_persists_pending_state(tmp_path):
    policy = _DeferOncePolicy()
    llm = FakeLLMClient([tool_call_then("pay", {"amount": 5000})])
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        permission_policy=policy, snapshot_store=snapshot_store, name="agent",
    )
    outcome = await agent.run("帮我付 5000 元", session_id="sess-d")

    assert outcome.status == "awaiting_approval"
    assert outcome.pending_approval_id == "appr-1"
    assert outcome.pending_tool_call_id is not None
    assert policy.seen_args == [{"amount": 5000}]

    snap = snapshot_store.load_latest("sess-d")
    assert snap.pending_approval_id == "appr-1"
    assert snap.pending_tool_call_id == outcome.pending_tool_call_id
    assert "已支付" not in str(outcome.messages)


async def test_resume_with_allow_executes_tool_and_completes(tmp_path):
    policy = _DeferOncePolicy()
    llm = FakeLLMClient([
        tool_call_then("pay", {"amount": 5000}),
        text_finish("已完成大额支付"),
    ])
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        permission_policy=policy, snapshot_store=snapshot_store, name="agent",
    )
    first = await agent.run("帮我付 5000 元", session_id="sess-e")
    assert first.status == "awaiting_approval"

    resumed = await agent.resume("sess-e", Allow())
    assert resumed.status == "completed"
    assert "已支付 5000 元" in str(resumed.messages)

    snap = snapshot_store.load_latest("sess-e")
    assert snap.pending_approval_id is None
    assert snap.pending_tool_call_id is None


async def test_resume_with_deny_synthesizes_denial_and_completes(tmp_path):
    policy = _DeferOncePolicy()
    llm = FakeLLMClient([
        tool_call_then("pay", {"amount": 5000}),
        text_finish("好的,取消这笔支付"),
    ])
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        permission_policy=policy, snapshot_store=snapshot_store, name="agent",
    )
    first = await agent.run("帮我付 5000 元", session_id="sess-f")
    assert first.status == "awaiting_approval"

    resumed = await agent.resume("sess-f", Deny(reason="人工审核不通过"))
    assert resumed.status == "completed"
    assert "已支付 5000 元" not in str(resumed.messages)
    assert "人工审核不通过" in str(resumed.messages)


async def test_resume_without_pending_approval_raises(tmp_path):
    llm = FakeLLMClient([text_finish("你好")])
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        snapshot_store=snapshot_store, name="agent",
    )
    await agent.run("你好", session_id="sess-g")
    with pytest.raises(ValueError, match="没有待处理的审批"):
        await agent.resume("sess-g", Allow())


async def test_resume_without_snapshot_store_raises():
    llm = FakeLLMClient([])
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        name="agent",
    )
    with pytest.raises(ValueError, match="snapshot_store"):
        await agent.resume("sess-h", Allow())


async def test_chained_defer_across_multiple_tool_calls_in_one_batch(tmp_path):
    """一批工具调用里有两个都需要审批:第一个批准后,第二个应该
    重新过一遍权限检查、再次挂起——不是"批准一个就等于批准这一整批"。
    """
    policy = _AlwaysDeferPolicy()
    llm = FakeLLMClient([{
        "content": None,
        "tool_calls": [("pay", {"amount": 100}), ("pay", {"amount": 200})],
        "finish_reason": "tool_calls",
    }])
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        permission_policy=policy, snapshot_store=snapshot_store, name="agent",
    )
    first = await agent.run("帮我付两笔钱", session_id="sess-i")
    assert first.status == "awaiting_approval"
    first_approval_id = first.pending_approval_id

    second = await agent.resume("sess-i", Allow())
    assert second.status == "awaiting_approval"
    assert second.pending_approval_id != first_approval_id
    assert "已支付 100 元" in str(second.messages)
    assert "已支付 200 元" not in str(second.messages)


async def test_no_double_round_processing_regression(tmp_path):
    """审查时推演出的真实 bug 回归:_round_loop 处理完一轮之后,不该
    因为收尾逻辑的递归调用而把同一轮或后续轮次重复处理一遍——用
    "LLM 恰好被调用了预期次数、脚本恰好耗尽"来断言,如果哪一轮被
    多算一次,脚本会提前耗尽并抛异常,这个断言本身就是回归探测器。
    """
    llm = FakeLLMClient([
        tool_call_then("pay", {"amount": 10}),
        tool_call_then("pay", {"amount": 20}),
        tool_call_then("pay", {"amount": 30}),
        text_finish("三笔都已完成"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        permission_policy=AllowAllPolicy(), name="agent",
    )
    outcome = await agent.run("依次付 10、20、30 元", session_id="sess-j")
    assert outcome.status == "completed"
    assert outcome.rounds == 4
    assert len(llm.calls) == 4
    text = str(outcome.messages)
    assert text.count("已支付 10 元") == 1
    assert text.count("已支付 20 元") == 1
    assert text.count("已支付 30 元") == 1


async def test_malformed_tool_arguments_do_not_crash_permission_check(tmp_path):
    """模型偶尔会吐出格式不对的 JSON 参数——权限检查不该因此崩掉,
    策略拿到的是空 dict,自己决定怎么处理(这里策略选择放行)。"""
    seen = []

    class _RecordingPolicy:
        async def check(self, tool_name, args, run_ctx):
            seen.append(args)
            return Allow()

    llm = FakeLLMClient([
        {"content": None, "tool_calls": [("pay", {"amount": 1})],
         "finish_reason": "tool_calls"},
        text_finish("完成"),
    ])
    agent = Agent(
        llm_client=llm, tool_executor=_make_executor(), system_prompt="你是助手",
        permission_policy=_RecordingPolicy(), name="agent",
    )
    outcome = await agent.run("付款", session_id="sess-k")
    assert outcome.status == "completed"
    assert seen == [{"amount": 1}]
