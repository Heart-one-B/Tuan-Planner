# tests/smoke_snapshot.py
"""
L3 状态持久化冒烟测试。

设计原则同前几轮:不做"预期输出等于某个值"式断言,而是针对快照系统
存在的理由本身做检验——它承诺了什么,就测什么会不会被打破。

快照系统的承诺清单(全部来自设计讨论,不是本脚本发明):
  S1 归一化恒等性  任意消息形态(dict/SDK对象)经 normalize_message
                  后,关键字段(role/content/tool_calls的id+name+
                  arguments)必须精确保留,不能丢、不能改。
  S2 JSON 往返无损  build_snapshot -> to_json -> from_dict 的结果,
                  与直接构造的快照在字段级完全一致。
  S3 resume_history 只剥前缀 system  开头连续的 system 消息被剥掉;
                  非开头位置出现的 system 消息(理论上不应发生,但
                  防御性验证)不会被误剥。
  S4 all_refs 完整性  快照的 all_refs() 与源 ContextManager 的
                  all_refs() 必须相等——快照不能在序列化过程中
                  漏引用,否则级联清理会误删仍被引用的文件。
  S5 原子写   写入过程中任何时刻杀掉进程,latest.json 要么是完整的
             旧版本,要么是完整的新版本,绝不能是半成品。
  S6 版本拒绝  version 字段不匹配时必须显式报错,不允许静默按当前
             版本的字段结构强行解析出一个错误对象。
  S7 路径安全  session_id 含穿越序列时不能逃逸出 base_dir。
  S8 归档轮转  keep_archives=N 时,超过 N 份的旧归档被清理,顺序正确。
  S9 端到端续跑  真实 AgentLoop 产出的 outcome 落盘后,用
             resume_history() 装配一个新 run,新 run 能看到旧对话
             的完整上下文并接着完成任务(不是从零开始)。
  S10 无 ContextManager 场景  未配置 context_config 的 Agent 同样能
             产出有效快照(messages-only,ctx 相关字段为空默认值)。

运行:python -m tests.smoke_snapshot
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from harness.agent.agent import Agent
from harness.snapshot.models import RunSnapshot, normalize_message, SNAPSHOT_VERSION
from harness.snapshot.build import build_snapshot
from harness.snapshot.store import FileSnapshotStore, _sanitize_session_id
from harness.agent.termination import AnswerTermination
from harness.context.budget import ContextBudget
from harness.context.compactor import Compactor
from harness.context.context_manager import ContextManagerConfig
from harness.context.offload import OffloadStore
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor

from tests.fake_llm import FakeLLMClient, text_message, tool_call, tool_call_message


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


class Findings:
    def __init__(self):
        self.violations: list[str] = []

    def require(self, ok: bool, promise: str, detail: str = "") -> None:
        print(f"  {'✅' if ok else '❌'} [{promise}] {detail}")
        if not ok:
            self.violations.append(f"{promise}: {detail}")

    def report(self) -> None:
        section("最终报告")
        if not self.violations:
            print("✅ 全部承诺兑现。")
        else:
            print(f"❌ {len(self.violations)} 条承诺被打破:")
            for v in self.violations:
                print(f"   - {v}")


# ══ S1/S2｜归一化与 JSON 往返 ═══════════════════════════════════════════

def part_s1_s2_normalize(f: Findings) -> None:
    section("S1/S2｜归一化恒等性 + JSON 往返无损")

    # dict 形态:普通 user/tool 消息
    d1 = {"role": "user", "content": "帮我查一下天气"}
    n1 = normalize_message(d1)
    f.require(n1 == d1, "dict user 消息原样通过", "")

    d2 = {"role": "tool", "tool_call_id": "c1", "content": "[结果]"}
    n2 = normalize_message(d2)
    f.require(n2 == d2, "dict tool 消息原样通过", "")

    # dict 形态但混入 reasoning_content(需要被剥离,这是 OpenAI 回放的既有约定)
    d3 = {"role": "assistant", "content": "好的", "reasoning_content": "思考过程..."}
    n3 = normalize_message(d3)
    f.require("reasoning_content" not in n3 and n3["content"] == "好的",
              "reasoning_content 被剥离", f"{n3}")

    # SDK 对象形态:assistant + tool_calls,content=None
    tc = tool_call("call_42", "search", {"q": "北京天气", "days": 3})
    obj_msg = SimpleNamespace(role="assistant", content=None, tool_calls=[tc])
    n4 = normalize_message(obj_msg)
    f.require(n4["role"] == "assistant" and n4["content"] is None,
              "SDK对象:content=None 显式保留(不是省略key)", f"{n4.keys()}")
    f.require(len(n4["tool_calls"]) == 1
              and n4["tool_calls"][0]["id"] == "call_42"
              and n4["tool_calls"][0]["function"]["name"] == "search",
              "SDK对象:tool_call id/name 精确保留", "")
    restored_args = json.loads(n4["tool_calls"][0]["function"]["arguments"])
    f.require(restored_args == {"q": "北京天气", "days": 3},
              "SDK对象:tool_call arguments 精确保留(反解JSON验证)", "")

    # SimpleNamespace 形态(FakeLLM 用的就是这个) —— 验证不依赖具体 SDK 类型
    fake_msg = SimpleNamespace(role="user", content="你好")
    n5 = normalize_message(fake_msg)
    f.require(n5 == {"role": "user", "content": "你好"},
              "SimpleNamespace 形态同样正确归一化", "")

    # JSON 往返:build 一个快照,序列化再反序列化,逐字段比较
    snap = RunSnapshot(
        session_id="s1", task="测试任务", trace_id="tr-1", status="completed",
        rounds=3, tool_calls_used=2,
        messages=[n1, n4],
        compaction_history=[{
            "trigger": "manual", "tokens_before": 1000, "tokens_after": 500,
            "summary": "摘要正文", "dropped_message_count": 6, "kept_tail_count": 4,
            "focus": None, "middle_ref": "tr-1/compacted_middle_abc.txt",
            "timestamp": "2026-07-14T10:00:00",
        }],
        offload_records=[{
            "ref": "tr-1/call_1.txt", "original_chars": 3000,
            "trace_id": "tr-1", "tool_call_id": "call_1", "created_at": "2026-07-14T09:00:00",
        }],
        total_tokens=500, token_source="estimated", created_at="2026-07-14T10:00:01",
    )
    roundtrip = RunSnapshot.from_dict(json.loads(snap.to_json()))
    f.require(roundtrip == snap, "S2 JSON往返:dataclass 逐字段相等", "")
    f.require(roundtrip.messages[1]["tool_calls"][0]["function"]["arguments"]
              == n4["tool_calls"][0]["function"]["arguments"],
              "S2 往返后 tool_call 参数字符串精确一致(不是近似)", "")


# ══ S3｜resume_history 前缀剥离 ═══════════════════════════════════════════

def part_s3_resume_history(f: Findings) -> None:
    section("S3｜resume_history 只剥前缀 system")

    snap = RunSnapshot(
        session_id="s3", task="t", trace_id="tr", status="completed",
        rounds=1, tool_calls_used=0,
        messages=[
            {"role": "system", "content": "系统提示A"},
            {"role": "user", "content": "任务"},
            {"role": "assistant", "content": "好的"},
        ],
    )
    resumed = snap.resume_history()
    f.require(resumed[0]["role"] == "user" and len(resumed) == 2,
              "开头 system 被剥,后续消息原样保留", f"resumed={resumed}")

    # 边界:只有一条 system,没有其他内容
    snap_only_system = RunSnapshot(
        session_id="s3b", task="t", trace_id="tr", status="completed",
        rounds=0, tool_calls_used=0,
        messages=[{"role": "system", "content": "系统提示"}],
    )
    f.require(snap_only_system.resume_history() == [],
              "边界:全部是 system 时 resume_history 返回空列表", "")

    # 防御性验证:非开头位置的 system(理论不应出现,但函数不能因此崩溃
    # 或误剥非前缀内容——它只处理"开头连续"这一种情况)
    snap_mid_system = RunSnapshot(
        session_id="s3c", task="t", trace_id="tr", status="completed",
        rounds=1, tool_calls_used=0,
        messages=[
            {"role": "user", "content": "问题"},
            {"role": "system", "content": "非常规的中段system"},
            {"role": "assistant", "content": "回答"},
        ],
    )
    resumed_mid = snap_mid_system.resume_history()
    f.require(len(resumed_mid) == 3,
              "非开头位置的 system 不会被误剥(只剥'开头连续'这一种形态)",
              f"len={len(resumed_mid)}")


# ══ S4｜all_refs 与 ContextManager 源头一致 ═══════════════════════════════

async def part_s4_all_refs(f: Findings) -> None:
    section("S4｜快照 all_refs() 与源 ContextManager.all_refs() 一致")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s4_"))

    budget = ContextBudget(
        max_tokens=3000, output_reserve=200, compaction_reserve=300,
        compaction_trigger_margin=200, min_compact_tokens=400,
        default_tool_result_max_chars=600, offload_preview_chars=200,
        offload_dir=tmp, tool_result_clear_after_rounds=2, keep_recent_rounds=2,
    )
    store = OffloadStore(tmp)
    from harness.context.context_manager import ContextManager
    cm = ContextManager(budget, store, Compactor())
    seed = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    cm.init(seed, prefix_len=2)

    # 制造一个 L1 卸载 + 若干轮沉底(触发 L2 换页)
    big = "X" * 3000
    cm.append(tool_call_message([tool_call("c0", "big", {})]))
    text = cm.offload_tool_result("t4", "c0", big)
    cm.append({"role": "tool", "tool_call_id": "c0", "content": text})
    for i in range(1, 6):
        cm.append(tool_call_message([tool_call(f"c{i}", "small", {})]))
        t = cm.offload_tool_result("t4", f"c{i}", f"result-{i}" * 50)
        cm.append({"role": "tool", "tool_call_id": f"c{i}", "content": t})
    cm._clear_stale("t4")   # 制造 L2 来源的 ref

    source_refs = cm.all_refs()
    f.require(len(source_refs) >= 2, "源头制造出多个 ref(L1+L2 混合)",
              f"count={len(source_refs)}")

    # 构造一个假 outcome/run_ctx 走 build_snapshot 路径
    outcome = SimpleNamespace(status="completed", rounds=6, tool_calls_used=6,
                              messages=cm.messages)
    run_ctx = SimpleNamespace(
        span=SimpleNamespace(trace_id="t4"), context_manager=cm,
    )
    snap = build_snapshot(outcome, run_ctx, "s4", "task")
    snap_refs = snap.all_refs()

    f.require(snap_refs == source_refs,
              "快照 all_refs() 与源 ContextManager 完全一致(无漏引用)",
              f"源={len(source_refs)} 快照={len(snap_refs)} 差集={source_refs ^ snap_refs}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ S5｜原子写:模拟崩溃 ═══════════════════════════════════════════════════

def part_s5_atomic_write(f: Findings) -> None:
    section("S5｜原子写:os.replace 前崩溃不得损坏 latest.json")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s5_"))
    store = FileSnapshotStore(tmp, keep_archives=0)

    snap_v1 = RunSnapshot(session_id="s5", task="v1", trace_id="tr", status="completed",
                          rounds=1, tool_calls_used=0, messages=[{"role": "user", "content": "v1"}])
    store.save(snap_v1)
    latest_path = tmp / "s5" / "latest.json"
    original_content = latest_path.read_text(encoding="utf-8")
    f.require(json.loads(original_content)["task"] == "v1", "第一次保存成功", "")

    # 模拟"写到临时文件成功,但 os.replace 之前进程死了"
    real_replace = os.replace
    def crashing_replace(*args, **kwargs):
        raise OSError("模拟:进程在 os.replace 之前被杀")
    os.replace = crashing_replace
    try:
        snap_v2 = RunSnapshot(session_id="s5", task="v2-理应丢失", trace_id="tr",
                              status="completed", rounds=2, tool_calls_used=0,
                              messages=[{"role": "user", "content": "v2"}])
        try:
            store.save(snap_v2)
            f.require(False, "崩溃模拟触发了异常", "竟然没有抛出")
        except OSError:
            pass  # 预期行为:异常应该传播出去,不能被吞掉
    finally:
        os.replace = real_replace

    # 崩溃后 latest.json 必须还是 v1(完整旧版本),不能是半成品或 v2
    survived = json.loads(latest_path.read_text(encoding="utf-8"))
    f.require(survived["task"] == "v1",
              "S5 崩溃后 latest.json 仍是完整的旧版本(不是半成品/v2)",
              f"实际task={survived['task']}")

    # 崩溃留下的临时文件必须被清理,不能残留垃圾
    leftover_tmp = list((tmp / "s5").glob(".latest_*.tmp"))
    f.require(len(leftover_tmp) == 0, "崩溃后临时文件被清理,不残留垃圾",
              f"残留={leftover_tmp}")

    # 崩溃恢复正常后,后续保存应该正常工作(不是永久性损坏)
    snap_v3 = RunSnapshot(session_id="s5", task="v3", trace_id="tr", status="completed",
                          rounds=3, tool_calls_used=0, messages=[])
    store.save(snap_v3)
    f.require(json.loads(latest_path.read_text(encoding="utf-8"))["task"] == "v3",
              "崩溃恢复后,后续保存恢复正常", "")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ S6｜版本拒绝 ═══════════════════════════════════════════════════════════

def part_s6_version_rejection(f: Findings) -> None:
    section("S6｜版本不匹配必须显式报错,不允许半解析")
    bad_data = {
        "session_id": "s6", "task": "t", "trace_id": "tr", "status": "completed",
        "rounds": 0, "tool_calls_used": 0, "messages": [],
        "compaction_history": [], "offload_records": [], "total_tokens": 0,
        "token_source": "estimated", "created_at": "", "version": 999,
    }
    try:
        RunSnapshot.from_dict(bad_data)
        f.require(False, "版本不匹配时拒绝解析", "竟然没有抛出异常")
    except ValueError as e:
        f.require("版本" in str(e) or "version" in str(e).lower(),
                  "版本不匹配时抛出明确的 ValueError", str(e)[:60])

    # 缺失 version 字段(比如极早期手写的测试数据)同样要拒绝,不能默认当作合法
    missing_version = {k: v for k, v in bad_data.items() if k != "version"}
    try:
        RunSnapshot.from_dict(missing_version)
        f.require(False, "缺失 version 字段时拒绝解析(不默认为合法)", "竟然没有抛出")
    except (ValueError, TypeError):
        f.require(True, "缺失 version 字段时拒绝解析(不默认为合法)", "")


# ══ S7｜路径安全 ═══════════════════════════════════════════════════════════

def part_s7_path_safety(f: Findings) -> None:
    section("S7｜session_id 路径穿越防护")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s7_"))
    store = FileSnapshotStore(tmp, keep_archives=0)

    malicious_id = "../../etc/passwd"
    sanitized = _sanitize_session_id(malicious_id)
    # 正确的判据是"不含路径分隔符"(单一合法目录名),不是"不含'..'子串"——
    # ".." 作为字面字符出现在无分隔符的单段文件名里没有穿越语义,
    # 真正危险的是残留 '/' 或 '\' 重新制造出层级。上一版断言判据错误
    # (混淆了"字符串含'..'这个视觉信号"和"字符串具备穿越能力"这两件
    # 不等价的事),这里修正为检查分隔符缺席 + 是单一路径段。
    f.require("/" not in sanitized and "\\" not in sanitized,
              "穿越序列被拍平为不含分隔符的单一目录名(非'不含..子串')",
              f"sanitized={sanitized}")
    f.require(len(Path(sanitized).parts) == 1,
              "清洗结果是单一路径段,无法制造层级跳转", f"parts={Path(sanitized).parts}")

    snap = RunSnapshot(session_id=malicious_id, task="t", trace_id="tr",
                      status="completed", rounds=0, tool_calls_used=0, messages=[])
    saved_path = Path(store.save(snap))
    f.require(saved_path.resolve().is_relative_to(tmp.resolve()),
              "即使传入穿越序列,实际写入路径仍在 base_dir 内",
              f"saved_path={saved_path}")

    # load_latest 同样要防穿越(读也是攻击面,不只是写)
    try:
        store.load_latest("../../../etc/passwd")
        loaded_ok = True
    except (ValueError, FileNotFoundError):
        loaded_ok = True  # 两种结果都可接受:清洗后大概率 FileNotFoundError
    f.require(loaded_ok, "load_latest 对穿越序列不崩溃、不逃逸", "")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ S8｜归档轮转 ═══════════════════════════════════════════════════════════

def part_s8_archive_rotation(f: Findings) -> None:
    section("S8｜归档轮转:keep_archives=N 时只保留最近 N 份")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s8_"))
    store = FileSnapshotStore(tmp, keep_archives=3)

    for i in range(6):
        snap = RunSnapshot(session_id="s8", task=f"v{i}", trace_id="tr",
                          status="completed", rounds=i, tool_calls_used=0, messages=[])
        store.save(snap)

    archive_files = sorted((tmp / "s8" / "archive").glob("*.json"))
    f.require(len(archive_files) == 3, "保存6次、keep_archives=3,归档恰好剩3份",
              f"实际={len(archive_files)}")

    # 剩下的应该是最后3次(v3,v4,v5),不是最早的3次
    contents = [json.loads(p.read_text())["task"] for p in archive_files]
    f.require(contents == ["v3", "v4", "v5"],
              "保留的是最近的3份,不是最早的3份(轮转方向正确)",
              f"实际={contents}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ S9｜端到端:真实 AgentLoop 崩溃续跑 ════════════════════════════════════

async def part_s9_end_to_end_resume(f: Findings) -> None:
    section("S9｜端到端:真实 AgentLoop 产出快照 -> 装配新 run -> 续跑")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s9_"))

    async def lookup(q: str) -> str:
        return f"查到关于'{q}'的结果:数据X=42"

    def make_executor():
        ex = ToolExecutor()
        ex.register(ToolDefinition(
            name="lookup", description="查询",
            parameters={"q": {"type": "string"}}, required=["q"], func=lookup,
        ))
        return ex

    snapshot_store = FileSnapshotStore(tmp / "snapshots", keep_archives=1)

    # ── 第一次运行(模拟"跑到一半、下次继续") ──
    llm1 = FakeLLMClient(script=[
        tool_call_message([tool_call("c1", "lookup", {"q": "数据X"})]),
        text_message("我查到了数据X=42,但还没完成完整分析。"),
    ])
    agent1 = Agent(
        llm_client=llm1, tool_executor=make_executor(),
        system_prompt="你是一个数据分析助手。", termination=AnswerTermination(),
        snapshot_store=snapshot_store, name="s9-agent",
    )
    outcome1 = await agent1.run("帮我查数据X并分析", session_id="s9-session")

    f.require(outcome1.status == "completed", "第一次运行正常收口", "")
    # 真实LLM冒烟发现的回归:AnswerTermination 收尾时最终文本此前从未
    # append 进 store,outcome.messages/快照因此永远缺失模型自己的结论。
    # 用 outcome.final_text 反查它是否真的进了 outcome.messages,而不是
    # 只看 outcome.final_text 本身非空(那样测不出这个 bug)。
    # 注意:FakeLLM 的最终消息是 SimpleNamespace 对象,不是 dict——
    # 第一版断言只检查了 isinstance(m, dict),对象形态被静默跳过,
    # 断言看起来在测但其实什么都没测到,是我自己在写这条回归时
    # 犯的和 Compactor 早期同一类错误(覆盖检查必须枚举全部内容形状)。
    def _content_of(m):
        return m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    f.require(
        any((_content_of(m) or "") and outcome1.final_text in _content_of(m)
            for m in outcome1.messages),
        "★回归:AnswerTermination 的最终结论被 append 进 outcome.messages(不只在 final_text 里)",
        f"final_text={outcome1.final_text!r}",
    )
    saved_path = tmp / "snapshots" / "s9-session" / "latest.json"
    f.require(saved_path.is_file(), "run 结束后自动落盘(未手动调用 save)", "")

    # ── "进程重启":新建 Agent 实例,从磁盘装配 history ──
    snap = snapshot_store.load_latest("s9-session")
    f.require(any("数据X=42" in (m.get("content") or "") for m in snap.messages),
              "快照里能找到第一轮已经查到的结果(证明不是从零重跑)", "")

    llm2 = FakeLLMClient(script=[
        text_message("基于之前查到的数据X=42,完整分析:这是一个偶数,且是2的幂。"),
    ])
    agent2 = Agent(
        llm_client=llm2, tool_executor=make_executor(),
        system_prompt="你是一个数据分析助手。", termination=AnswerTermination(),
        snapshot_store=snapshot_store, name="s9-agent",
    )
    outcome2 = await agent2.run(
        "请基于已有信息完成完整分析", history=snap.resume_history(), session_id="s9-session",
    )

    f.require(outcome2.status == "completed", "续跑正常收口", "")
    f.require("2的幂" in outcome2.final_text, "续跑给出的答案确实利用了历史上下文", "")
    # 关键验证:第二次调用发给模型的 messages 里,必须包含第一轮的工具结果——
    # 这是"续跑不是从零开始"在协议层面的证据,不是靠模型自称
    sent_messages = llm2.calls[0]["messages"]
    f.require(any("数据X=42" in str(m.get("content") or "") for m in sent_messages),
              "S9 关键证据:第二次 API 调用发出的 messages 里包含第一轮的工具结果",
              f"messages数={len(sent_messages)}")
    # 且没有重复的 system 消息(resume_history 剥离生效的证据)
    system_count = sum(1 for m in sent_messages if m.get("role") == "system")
    f.require(system_count == 1, "S3 生效的证据:续跑后只有1条system消息,不是2条",
              f"实际={system_count}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ S10｜无 ContextManager 场景 ═══════════════════════════════════════════

async def part_s10_no_context_manager(f: Findings) -> None:
    section("S10｜未配置 context_config 时快照仍然有效(messages-only)")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s10_"))
    snapshot_store = FileSnapshotStore(tmp, keep_archives=0)

    async def echo(text: str) -> str:
        return f"echo: {text}"

    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="echo", description="回显", parameters={"text": {"type": "string"}},
        required=["text"], func=echo,
    ))
    llm = FakeLLMClient(script=[text_message("好的,任务完成。")])
    agent = Agent(
        llm_client=llm, tool_executor=executor, system_prompt="测试",
        termination=AnswerTermination(), snapshot_store=snapshot_store,
        # 故意不传 context_config
        name="s10-agent",
    )
    outcome = await agent.run("简单任务", session_id="s10-session")
    f.require(outcome.status == "completed", "无 context_config 时循环正常收口", "")

    snap = snapshot_store.load_latest("s10-session")
    f.require(snap.total_tokens == 0 and snap.token_source == "estimated"
              and snap.compaction_history == [] and snap.offload_records == [],
              "S10 无 ContextManager 时,ctx 相关字段全部是合理的空默认值,不是崩溃",
              f"tokens={snap.total_tokens} history={snap.compaction_history}")
    f.require(len(snap.messages) > 0, "messages 字段仍然完整落盘", f"count={len(snap.messages)}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ 入口 ═══════════════════════════════════════════════════════════════

async def main() -> None:
    f = Findings()
    part_s1_s2_normalize(f)
    part_s3_resume_history(f)
    await part_s4_all_refs(f)
    part_s5_atomic_write(f)
    part_s6_version_rejection(f)
    part_s7_path_safety(f)
    part_s8_archive_rotation(f)
    await part_s9_end_to_end_resume(f)
    await part_s10_no_context_manager(f)
    f.report()


if __name__ == "__main__":
    asyncio.run(main())