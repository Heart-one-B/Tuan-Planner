# tests/agent/test_repair.py
"""对应待做计划表任务 4.1/4.4 的核心验证。

第一性问题:repair_orphan_tool_calls 是"先认错,再继续"哲学的落地——
测试要证明两件事:①它确实能找到所有没配对的 tool_call,不多不少;
②repaired_history() 和 resume_history() 的边界真的如设计所说,
一个会修、一个绝不会修,混用的后果在这里被实际验证出来,不是只在
注释里写着吓唬人。
"""
from __future__ import annotations

from datetime import datetime

from harness.agent.repair import ORPHAN_MARKER, repair_orphan_tool_calls, repaired_history
from harness.snapshot.models import RunSnapshot


def _assistant_with_calls(*call_ids: str) -> dict:
    return {
        "role": "assistant", "content": None,
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "t", "arguments": "{}"}}
            for cid in call_ids
        ],
    }


def _tool_result(call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# ── 核心算法:通用扫描 ─────────────────────────────────────────────────────

def test_repair_fills_single_missing_tool_result():
    messages = [
        {"role": "user", "content": "问题"},
        _assistant_with_calls("c1"),
    ]
    count = repair_orphan_tool_calls(messages, reason="测试")

    assert count == 1
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "c1"
    assert ORPHAN_MARKER in tool_msgs[0]["content"]
    assert "测试" in tool_msgs[0]["content"]


def test_repair_leaves_already_paired_calls_untouched():
    """已经有真实结果的 tool_call 不该被"顺手"再插一条——修复只填
    真正缺失的洞,不重复。"""
    messages = [
        {"role": "user", "content": "问题"},
        _assistant_with_calls("c1"),
        _tool_result("c1", "真实结果"),
    ]
    count = repair_orphan_tool_calls(messages, reason="测试")

    assert count == 0
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "真实结果"


def test_repair_fills_only_the_missing_one_in_mixed_batch():
    """同一批里 c1 有真实结果、c2 没有——只补 c2,c1 保持原样,且
    插入位置在 c1 结果之后(不打乱已有顺序)。"""
    messages = [
        {"role": "user", "content": "问题"},
        _assistant_with_calls("c1", "c2"),
        _tool_result("c1", "真实结果"),
    ]
    count = repair_orphan_tool_calls(messages, reason="用户中断")

    assert count == 1
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0] == {"role": "tool", "tool_call_id": "c1", "content": "真实结果"}
    assert tool_msgs[1]["tool_call_id"] == "c2"
    assert ORPHAN_MARKER in tool_msgs[1]["content"]


def test_repair_handles_multiple_assistant_messages_independently():
    messages = [
        {"role": "user", "content": "问题"},
        _assistant_with_calls("c1"),
        _tool_result("c1"),
        {"role": "user", "content": "追问"},
        _assistant_with_calls("c2", "c3"),
        _tool_result("c2"),
        # c3 缺失
    ]
    count = repair_orphan_tool_calls(messages, reason="执行异常")

    assert count == 1
    ids_with_results = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
    assert set(ids_with_results) == {"c1", "c2", "c3"}


def test_repair_is_a_no_op_when_nothing_is_orphaned():
    messages = [
        {"role": "user", "content": "问题"},
        _assistant_with_calls("c1"),
        _tool_result("c1"),
        {"role": "assistant", "content": "纯文本回复,没有工具调用"},
    ]
    count = repair_orphan_tool_calls(messages, reason="不该被用到")

    assert count == 0
    assert len(messages) == 4   # 一条都没多


def test_repair_handles_sdk_object_style_messages():
    """孤儿修复不能假设消息全是 dict——assistant 消息在真实运行时
    经常是 SDK 风格对象(见 fakes.py 的约定)。"""
    from types import SimpleNamespace

    assistant_msg = SimpleNamespace(
        role="assistant", content=None,
        tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="t", arguments="{}"))],
    )
    messages = [{"role": "user", "content": "问题"}, assistant_msg]

    count = repair_orphan_tool_calls(messages, reason="测试")

    assert count == 1
    assert messages[-1]["tool_call_id"] == "c1"


# ── 关键边界:repaired_history() vs resume_history() ─────────────────────

def _pending_snapshot() -> RunSnapshot:
    return RunSnapshot(
        session_id="s1", task="问题", trace_id="t1", status="awaiting_approval",
        rounds=1, tool_calls_used=1,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "问题"},
            _assistant_with_calls("c1"),
        ],
        created_at=datetime.now().isoformat(),
        pending_tool_call_id="c1", pending_approval_id="a1",
    )


def test_resume_history_keeps_orphan_for_resume_to_find():
    """resume_history() 绝不修复——Agent.resume() 依赖这个孤儿
    tool_call 本身来定位挂起点,修复了反而会让 resume() 失败。"""
    snap = _pending_snapshot()
    history = snap.resume_history()

    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert tool_msgs == []   # c1 仍然是孤儿,没有被填任何占位


def test_repaired_history_fills_orphan_for_standalone_continuation():
    """repaired_history() 给"放弃 resume、当作全新历史续跑"的场景用,
    必须修复,否则送出去会被 provider 拒收(400)。"""
    snap = _pending_snapshot()
    history = repaired_history(snap)

    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "c1"
    assert ORPHAN_MARKER in tool_msgs[0]["content"]
    assert "审批未完成" in tool_msgs[0]["content"]


def test_repaired_history_and_resume_history_do_not_mutate_snapshot_messages():
    """两者都不该改动 snapshot.messages 本身(那是落盘的历史事实,
    不该被这两个只是"生成一份可用副本"的方法悄悄改写)。"""
    snap = _pending_snapshot()
    original_len = len(snap.messages)

    snap.resume_history()
    repaired_history(snap)

    assert len(snap.messages) == original_len
    assert not any(m.get("role") == "tool" for m in snap.messages)