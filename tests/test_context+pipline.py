# tests/smoke_context_full.py
"""
上下文压缩管线完整冒烟测试(第三版,最终定稿)。

版本历史:
  v1  用"调小预算"让各级自然触发 -> 实测证明是概率游戏,L2 稳态震荡
      导致 L3 永远够不着。
  v2  每级用自己的触发入口确定性驱动,逐级隔离 + 组合观察。
      实测发现:offload_records 账本漏记(已修)、P5 记账口径不一致(已修)、
      一档降级不给 ref 线索(已拍板:给)。
  v3  本版:三项修复/拍板全部转为硬断言回归;修正 v2 自身两处断言 bug
      (Part A 字面量 3000 vs 实际 len(big)=3018;Part D1 前缀断言自己
      比自己恒真)。

覆盖矩阵:
  Part A  L1 卸载(阈值触发)+ 取回工具完整分页 + 路径穿越防护
  Part B  L2 沉底换页 + 账本回传(offload_records 修复的回归)
  Part C  L3 摘要压缩(trigger="manual" 确定性驱动)
  Part D  熔断一档(含 ref 线索拍板的回归)与二档(overflow 强制构造)
  Part E  组合涌现:三级共存长跑,只观察时间线
  Part F  AgentLoop 溢出接线:真循环 ContextOverflowError -> 紧急压缩 -> 收口
  Part G  P5 记账口径修复的回归(大参数 tool_calls)

持续检验的承诺(源自代码 docstring,非本脚本发明):
  P1 头部前缀永不动(is 恒等,非内容相等)
  P2 切口不拆散工具调用配对
  P3 换页可还原且内容一致
  P4 压缩事件与审计记录一一对应
  账本完整性:磁盘文件 ⊆ all_refs()
  全局不变量(本轮拍板后无例外):任何内容离开窗口,窗口内必留可见指针

运行:python -m tests.smoke_context_full
预期:0 违约、0 flag。任何 ❌ 都意味着实现与文档承诺脱节。
"""
from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from pathlib import Path

from harness.agent.agent import Agent
from harness.agent.termination import AnswerTermination
from harness.context.budget import ContextBudget
from harness.context.compactor import Compactor
from harness.context.context_manager import ContextManager, ContextManagerConfig
from harness.context.offload import (
    CLEARED_MARKER, OFFLOAD_MARKER, RETRIEVAL_TOOL_NAME,
    OffloadStore, build_retrieval_tool,
)
from harness.context.token_counter import estimate_messages_tokens
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor

from tests.fake_llm import (
    FakeLLMClient, Overflow, text_message, tool_call, tool_call_message,
)

# ══ 基础设施 ═══════════════════════════════════════════════════════════

def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def make_content(marker: str, length: int) -> str:
    """length 是填充部分的长度;总长 = length + 头尾标记(len 以实测为准,
    断言一律引用 len() 而非字面量——v2 的教训)。"""
    pad = ("契约验证填充字符。" * ((length // 9) + 1))[:length]
    return f"<<{marker}>>{pad}<<{marker}_END>>"


GOOD_SUMMARY = (
    "<analysis>草稿区,应被剥离</analysis>\n<summary>\n"
    "## 任务目标\n多轮工具查询任务\n"
    "## 用户消息全录\n- 请多次调用工具完成任务\n"
    "## 用户的明确指令与约束\n无\n"
    "## 已完成的工作与关键结论\n若干轮查询已完成\n"
    "## 关键决策及理由\n无\n"
    "## 产物与引用\n见上下文占位符\n"
    "## 未完成的事项与下一步\n继续查询\n"
    "## 需要警惕的信息\n无\n</summary>"
)


def good_summarizer(n: int = 20) -> FakeLLMClient:
    return FakeLLMClient(script=[text_message(GOOD_SUMMARY) for _ in range(n)])


def bad_summarizer(n: int = 10) -> FakeLLMClient:
    """空回复 -> Compactor._summarize 抛"压缩器返回了空摘要" -> 熔断。"""
    return FakeLLMClient(script=[text_message("") for _ in range(n)])


class Findings:
    def __init__(self):
        self.violations: list[str] = []
        self.flags: list[str] = []

    def require(self, ok: bool, promise: str, detail: str = "") -> None:
        print(f"  {'✅' if ok else '❌'} [{promise}] {detail}")
        if not ok:
            self.violations.append(f"{promise}: {detail}")

    def flag(self, message: str) -> None:
        print(f"  ⚠️  FINDING: {message}")
        self.flags.append(message)

    def report(self) -> None:
        section("最终报告")
        if not self.violations:
            print("✅ 全部硬性承诺兑现。")
        else:
            print(f"❌ {len(self.violations)} 条承诺被打破:")
            for v in self.violations:
                print(f"   - {v}")
        if self.flags:
            print(f"\n⚠️  {len(self.flags)} 条待人工判断的现象:")
            for x in self.flags:
                print(f"   - {x}")
        else:
            print("⚠️  0 条待判断现象。")


def check_prefix_identity(original: list, current: list, n: int, f: Findings, tag: str) -> None:
    """is 恒等而非内容相等:压缩允许重建等值消息糊弄内容比较,
    对象恒等才证明 prefix 真的一个字节没被碰过。"""
    ok = len(current) >= n and all(current[i] is original[i] for i in range(n))
    f.require(ok, f"P1 前缀恒等 [{tag}]", f"prefix_len={n}")


def check_tool_pairing(messages: list, f: Findings, tag: str) -> None:
    pending, orphans = {}, []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "assistant")
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        if tcs and role != "tool":
            for tc in tcs:
                pending[tc.get("id") if isinstance(tc, dict) else tc.id] = True
        if isinstance(m, dict) and m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid in pending:
                del pending[cid]
            else:
                orphans.append(cid)
    f.require(not pending and not orphans, f"P2 工具配对完整 [{tag}]",
              f"未配对={list(pending)} 孤儿={orphans}")


def check_ledger_complete(cm: ContextManager, base_dir: Path, trace_id: str,
                          f: Findings, tag: str) -> None:
    """offload_records 修复的核心回归:磁盘上每一个文件,账本必须认领。
    all_refs() = offload_records(L1/L2 来源) ∪ compaction_history.middle_ref(L3 来源)。"""
    d = base_dir / trace_id
    disk = {str(p.relative_to(base_dir)) for p in d.glob("*.txt")} if d.is_dir() else set()
    ledger = cm.all_refs()
    missing = disk - ledger
    f.require(not missing, f"账本完整:磁盘文件全部被 all_refs 认领 [{tag}]",
              f"磁盘={len(disk)} 账本={len(ledger)} 漏记={sorted(missing)[:3]}")


def make_cm(tmp: Path, **budget_overrides) -> tuple[ContextManager, ContextBudget, OffloadStore]:
    defaults = dict(
        max_tokens=3000, output_reserve=200, compaction_reserve=300,
        compaction_trigger_margin=200, min_compact_tokens=400,
        default_tool_result_max_chars=2600, offload_preview_chars=200,
        offload_dir=tmp, tool_result_clear_after_rounds=2, keep_recent_rounds=2,
    )
    defaults.update(budget_overrides)
    budget = ContextBudget(**defaults)
    store = OffloadStore(tmp)
    return ContextManager(budget, store, Compactor()), budget, store


def seed_and_rounds(cm: ContextManager, n_rounds: int, chars_per_result: int,
                    args_pad: str = "") -> list:
    """种 seed + 模拟 n 轮工具调用。返回 seed 供 P1 恒等断言使用
    (v2 教训:不捕获 seed,前缀断言就只能自己比自己,恒真)。"""
    seed = [
        {"role": "system", "content": "系统提示:测试任务。"},
        {"role": "user", "content": "请多次调用工具完成任务。"},
    ]
    cm.init(seed, prefix_len=len(seed))
    for i in range(1, n_rounds + 1):
        marker = f"R{i}"
        cm.append(tool_call_message(
            [tool_call(f"call_{i}", "lookup", {"q": marker + args_pad})]))
        text = cm.offload_tool_result("t", f"call_{i}",
                                      make_content(marker, chars_per_result))
        cm.append({"role": "tool", "tool_call_id": f"call_{i}", "content": text})
    return seed


# ══ Part A｜L1 卸载 + 取回工具完整分页 ═══════════════════════════════════

async def part_a_l1(f: Findings) -> None:
    section("Part A｜L1 卸载 + 取回工具完整分页")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_a_"))
    cm, budget, store = make_cm(tmp, default_tool_result_max_chars=600)
    trace_id = "A"

    seed = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    cm.init(seed, prefix_len=2)

    big = make_content("BIG", 3000)
    cm.append(tool_call_message([tool_call("c0", "big_lookup", {"q": "x"})]))
    text = cm.offload_tool_result(trace_id, "c0", big)
    cm.append({"role": "tool", "tool_call_id": "c0", "content": text})

    f.require(text.startswith(OFFLOAD_MARKER), "L1 超阈值即卸载",
              f"{len(big)}字符 -> 预览{len(text)}字符")
    # v3 修正:断言引用 len(big) 而非字面量 3000(标记额外占 18 字符)
    f.require(len(cm.offload_records) == 1
              and cm.offload_records[0].original_chars == len(big),
              "L1 卸载进账本(original_chars 精确)",
              f"original_chars={cm.offload_records[0].original_chars} == len(big)={len(big)}")
    f.require("BIG>>" in text and "BIG_END" in text,
              "预览含头尾(报错在尾部原则)", "")

    # 小于阈值的结果原样通过,不产文件
    small = make_content("SMALL", 100)
    passed = cm.offload_tool_result(trace_id, "c1", small)
    f.require(passed == small and len(cm.offload_records) == 1,
              "L1 未超阈值原样通过、不落盘", "")

    # 取回:循环分页读到底(v2 教训:必须读到没有续读提示为止,不许只追固定页数)
    tool_def = build_retrieval_tool(store, max_return_chars=500)
    ref = cm.offload_records[0].ref
    assembled, offset, pages = "", 0, 0
    while True:
        chunk = await tool_def.func(ref=ref, offset=offset)
        pages += 1
        m = re.search(r"offset=(\d+)\]$", chunk)
        if m:
            assembled += chunk[:chunk.rfind("\n...[未完")]
            offset = int(m.group(1))
        else:
            assembled += chunk
            break
        if pages > 20:
            break
    f.require(assembled == big, "分页取回逐页拼接 == 原文(逐字节)",
              f"{pages} 页, 拼接{len(assembled)}字符 vs 原文{len(big)}字符")

    # 路径穿越防护(取回参数来自模型输出,不可信输入)
    try:
        store.load("../../etc/passwd")
        f.require(False, "路径穿越被拒绝", "竟然没抛异常")
    except (ValueError, FileNotFoundError) as e:
        f.require(isinstance(e, ValueError), "路径穿越被拒绝", type(e).__name__)

    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part B｜L2 沉底换页 + 账本回传 ════════════════════════════════════════

async def part_b_l2(f: Findings) -> None:
    section("Part B｜L2 沉底换页 + 账本回传(offload_records 修复的回归)")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_b_"))
    cm, budget, store = make_cm(tmp, tool_result_clear_after_rounds=2)

    seed = seed_and_rounds(cm, n_rounds=5, chars_per_result=400)
    ledger_before = len(cm.offload_records)

    cleared = cm._clear_stale("t")

    f.require(cleared == 3, "5 轮保 2 轮,清理恰好 3 条", f"cleared={cleared}")
    f.require(len(cm.offload_records) == ledger_before + cleared,
              "★修复回归:每次换页都进账本",
              f"账本 {ledger_before} -> {len(cm.offload_records)}")

    # 每条占位符里的 ref 都可还原,且内容含原始标记
    for m in cm.messages:
        if isinstance(m, dict) and m.get("role") == "tool" \
                and m["content"].startswith(CLEARED_MARKER):
            rm = re.search(r"引用[:：]\s*(\S+?)\(", m["content"])
            f.require(rm is not None, "占位符携带 ref", m["content"][:60])
            if rm:
                content = store.load(rm.group(1))
                mk = re.search(r"<<(R\d+)>>", content)
                f.require(mk is not None, "P3 换页内容可还原", f"ref={rm.group(1)}")

    # 幂等:重复清理不重复落盘
    cleared2 = cm._clear_stale("t")
    f.require(cleared2 == 0 and len(cm.offload_records) == ledger_before + cleared,
              "重复清理幂等(占位符不二次换页)", f"第二次 cleared={cleared2}")

    check_ledger_complete(cm, tmp, "t", f, "L2后")
    check_tool_pairing(cm.messages, f, "L2后")
    check_prefix_identity(seed, cm.messages, 2, f, "L2后")
    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part C｜L3 摘要压缩(manual 确定性驱动) ═══════════════════════════════

async def part_c_l3(f: Findings) -> None:
    section("Part C｜L3 摘要压缩(trigger=manual 确定性驱动)")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_c_"))
    # clear_after 调大隔离掉 L2,让本节只测 L3
    cm, budget, store = make_cm(tmp, tool_result_clear_after_rounds=99)
    seed = seed_and_rounds(cm, n_rounds=5, chars_per_result=400)

    outcome = await cm.maybe_compact(good_summarizer(), "t", trigger="manual")

    f.require(outcome is not None and not outcome.degraded, "manual 触发压缩成功", "")
    if outcome is None:
        shutil.rmtree(tmp, ignore_errors=True)
        return

    r = outcome.result
    # 5 轮保 2 轮 -> 中间段 = 前 3 轮 = 6 条消息
    f.require(r.dropped_message_count == 6, "中间段条数符合边界规则(5轮保2轮=丢6条)",
              f"dropped={r.dropped_message_count}")
    f.require(r.kept_tail_count == 4, "尾部保留条数正确(2轮=4条)",
              f"kept={r.kept_tail_count}")

    check_prefix_identity(seed, cm.messages, 2, f, "L3后")
    check_tool_pairing(cm.messages, f, "L3后")

    # 摘要消息:位置紧贴 prefix,剥离了 <analysis>,携带 middle_ref 取回指引
    smsg = cm.messages[2]
    is_summary = isinstance(smsg, dict) and smsg.get("role") == "user" \
                 and "已被压缩" in smsg.get("content", "")
    f.require(is_summary, "摘要消息在 prefix 之后第一位", "")
    if is_summary:
        f.require("<analysis>" not in smsg["content"] and "草稿区" not in smsg["content"],
                  "analysis 草稿被剥离", "")
        f.require("## 用户消息全录" in smsg["content"], "七章节结构保留", "")
        f.require(r.middle_ref is not None and r.middle_ref in smsg["content"]
                  and RETRIEVAL_TOOL_NAME in smsg["content"],
                  "摘要携带 middle_ref + 取回指引", f"ref={r.middle_ref}")

    # middle_ref 可还原,且含被丢弃轮次的原始标记
    if r.middle_ref:
        raw = store.load(r.middle_ref)
        f.require(all(f"<<R{i}>>" in raw for i in (1, 2, 3)),
                  "P3 middle_ref 含全部被压轮次原文", "")

    # 记账:压缩后 counter 与独立估算一致(reset 语义)
    diff = abs(cm.counter.current_tokens - estimate_messages_tokens(cm.messages))
    f.require(diff <= 1, "压缩后记账与独立估算一致(reset生效)",
              f"diff={diff} tokens {r.tokens_before}->{r.tokens_after}")
    f.require(r.tokens_after < r.tokens_before, "压缩确实减少了占用",
              f"{r.tokens_before} -> {r.tokens_after}")

    f.require(len(cm.compaction_history) == 1, "P4 审计记录恰好一条", "")
    check_ledger_complete(cm, tmp, "t", f, "L3后")

    print("  [说明] 摘要器是脚本化假模型,本节只验证结构性承诺;"
          "\"摘要内容召回率\"必须用真实模型测,已在 Phase 5 计划内,此处不假装。")
    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part D｜熔断降级:一档与二档 ═══════════════════════════════════════════

async def part_d_breaker(f: Findings) -> None:
    section("Part D1｜熔断一档(manual/threshold 路径:规则瘦身 + ref 线索)")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_d1_"))
    cm, budget, store = make_cm(tmp, tool_result_clear_after_rounds=99)
    seed = seed_and_rounds(cm, n_rounds=5, chars_per_result=400)   # v3:捕获 seed
    n_before = len(cm.messages)

    outcome = await cm.maybe_compact(bad_summarizer(), "t", trigger="manual")

    f.require(outcome is not None and outcome.degraded, "摘要连续失败 -> degraded=True", "")
    if outcome:
        f.require(outcome.user_notice is None, "一档不打扰用户(user_notice=None)", "")
        # v3:拍板落地后一档多出 1 条 ref 提示消息
        f.require(len(cm.messages) == n_before + 1,
                  "一档不丢消息,只清内容(+1 条 ref 提示)",
                  f"{n_before} -> {len(cm.messages)}")
        n_cleared = sum(1 for m in cm.messages if isinstance(m, dict)
                        and m.get("role") == "tool"
                        and m["content"].startswith(CLEARED_MARKER))
        f.require(n_cleared == 3, "中间段 3 条工具结果被清占位(尾部 2 轮不动)",
                  f"cleared={n_cleared}")
        check_tool_pairing(cm.messages, f, "一档后")
        check_prefix_identity(seed, cm.messages, 2, f, "一档后")   # v3:真断言

        # v3:flag 转硬断言——全局不变量"任何内容离开窗口,窗口内必留可见指针"
        ref = outcome.result.middle_ref
        hint_msg = next((m for m in cm.messages if isinstance(m, dict)
                         and ref and ref in (m.get("content") or "")), None)
        f.require(hint_msg is not None,
                  "★拍板落地:一档降级后上下文携带 middle_ref 线索", f"ref={ref}")
        if hint_msg:
            f.require(RETRIEVAL_TOOL_NAME in hint_msg["content"],
                      "线索含取回工具指引", "")
            raw = store.load(ref)
            f.require("<<R1>>" in raw,
                      "P3 一档线索的 ref 可还原被清内容", "")
        check_ledger_complete(cm, tmp, "t", f, "一档后")
    shutil.rmtree(tmp, ignore_errors=True)

    section("Part D2｜熔断二档(overflow 路径:一档不够 -> 换页丢弃 + 双向告知)")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_d2_"))
    # 强制二档的构造:尾部保留轮次本身巨大(一档不碰尾部),一档瘦身后
    # 必然仍 > fit_within(effective_window=2500),只能走二档
    cm, budget, store = make_cm(tmp, tool_result_clear_after_rounds=99,
                                default_tool_result_max_chars=99_999)
    seed = seed_and_rounds(cm, n_rounds=5, chars_per_result=3000)  # 尾2轮≈3000tok>2500
    n_before = len(cm.messages)

    outcome = await cm.maybe_compact(bad_summarizer(), "t", trigger="overflow")

    f.require(outcome is not None and outcome.degraded, "overflow+熔断进入降级", "")
    if outcome:
        f.require(outcome.user_notice is not None, "二档必须告知用户(user_notice非空)",
                  (outcome.user_notice or "")[:60])
        f.require(len(cm.messages) < n_before, "二档确实丢弃了中间段",
                  f"{n_before} -> {len(cm.messages)}")
        placeholder = next((m for m in cm.messages if isinstance(m, dict)
                            and "[系统提示]" in (m.get("content") or "")), None)
        f.require(placeholder is not None, "占位符告知模型(防幻觉连续)", "")
        if placeholder and outcome.result.middle_ref:
            f.require(outcome.result.middle_ref in placeholder["content"]
                      and RETRIEVAL_TOOL_NAME in placeholder["content"],
                      "二档占位符携带 ref_hint(模型可自主恢复)", "")
            raw = store.load(outcome.result.middle_ref)
            f.require("<<R1>>" in raw, "P3 被丢中间段可从 middle_ref 还原", "")
        check_tool_pairing(cm.messages, f, "二档后")
        check_prefix_identity(seed, cm.messages, 2, f, "二档后")
        check_ledger_complete(cm, tmp, "t", f, "二档后")
    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part E｜组合涌现:三级共存的长跑时间线 ═════════════════════════════════

async def part_e_combined(f: Findings) -> None:
    section("Part E｜组合涌现(只观察时间线,不预设哪级先开火)")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_e_"))
    cm, budget, store = make_cm(tmp)   # 2000<2600 不触发 L1(独立通道已在 A 测过)
    summarizer = good_summarizer(20)
    seed = [
        {"role": "system", "content": "系统提示。"},
        {"role": "user", "content": "请多次调用工具完成任务。"},
    ]
    cm.init(seed, prefix_len=2)

    timeline: list[str] = []
    l3_round, hist = None, 0
    for i in range(1, 41):
        cm.append(tool_call_message([tool_call(f"c{i}", "lookup", {"q": f"R{i}"})]))
        text = cm.offload_tool_result("t", f"c{i}", make_content(f"R{i}", 2000))
        cm.append({"role": "tool", "tool_call_id": f"c{i}", "content": text})

        before = cm.counter.current_tokens
        stale_before = sum(1 for r in cm.offload_records if "stale_" in r.ref)
        outcome = await cm.maybe_compact(summarizer, "t", trigger="threshold")
        stale_after = sum(1 for r in cm.offload_records if "stale_" in r.ref)
        after = cm.counter.current_tokens

        events = []
        if stale_after > stale_before:
            events.append(f"L2清{stale_after - stale_before}条")
        if outcome:
            events.append(f"L3压缩 dropped={outcome.result.dropped_message_count}"
                          f" degraded={outcome.degraded}")
            if l3_round is None:
                l3_round = i
        # P4 每轮验证:有事件恰好 +1 条审计,无事件恰好 +0
        new_hist = len(cm.compaction_history)
        if new_hist != hist + (1 if outcome else 0):
            f.require(False, "P4 事件/审计一一对应", f"round={i}")
        hist = new_hist

        line = f"round {i:>2}  tokens {before:>5} -> {after:>5}" + \
               ("  ← " + " + ".join(events) if events else "")
        timeline.append(line)
        print("  " + line)

        if l3_round and i >= l3_round + 3:
            break

    f.require(l3_round is not None,
              "组合场景下 L3 最终接手(L2 兜底被增长追上)",
              f"首次 L3 在 round {l3_round}")
    f.require(any("L2清" in ln for ln in timeline),
              "组合场景下 L2 确实先于/伴随 L3 工作", "")
    check_tool_pairing(cm.messages, f, "长跑后")
    check_prefix_identity(seed, cm.messages, 2, f, "长跑后")
    check_ledger_complete(cm, tmp, "t", f, "长跑后")
    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part F｜AgentLoop 溢出接线(真循环里的紧急压缩) ════════════════════════

async def part_f_loop_overflow(f: Findings) -> None:
    section("Part F｜AgentLoop 溢出接线(ContextOverflowError -> 紧急压缩 -> 重试)")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_f_"))

    async def lookup(q: str) -> str:
        return make_content(f"F:{q}", 300)

    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="lookup", description="查询",
        parameters={"q": {"type": "string"}}, required=["q"], func=lookup,
    ))

    # 脚本时序推演(主对话与压缩共用同一客户端,顺序必须精确):
    #  1-3  三轮工具调用(积累出可压的中间段:3轮保2轮 -> 中间=第1轮)
    #  4    Overflow -> 循环捕获 -> maybe_compact(trigger=overflow)
    #  5    被 Compactor._summarize 消费 -> 摘要
    #  6    压缩后重试本轮 -> 最终文本收口
    # "脚本恰好耗尽"断言守护这条推演:实际执行顺序与推演不符会立刻暴露。
    llm = FakeLLMClient(script=[
        tool_call_message([tool_call("f1", "lookup", {"q": "a"})]),
        tool_call_message([tool_call("f2", "lookup", {"q": "b"})]),
        tool_call_message([tool_call("f3", "lookup", {"q": "c"})]),
        Overflow(),
        text_message(GOOD_SUMMARY),
        text_message("已完成全部查询,任务结束。"),
    ])

    agent = Agent(
        llm_client=llm, tool_executor=executor,
        system_prompt="测试 agent。", termination=AnswerTermination(),
        context_config=ContextManagerConfig(
            budget=ContextBudget(offload_dir=tmp, keep_recent_rounds=2),
            offload_store=OffloadStore(tmp), compactor=Compactor(),
        ),
        name="smoke-f",
    )

    events = []
    outcome = None
    async for ev in agent.events("连续调用 lookup 三次然后总结。"):
        events.append(ev["type"])
        if ev["type"] == "context_compacted":
            f.require(ev["trigger"] == "overflow", "压缩事件标注 overflow 触发",
                      f"tokens {ev['tokens_before']}->{ev['tokens_after']}"
                      f" degraded={ev['degraded']}")
        if ev["type"] == "outcome":
            outcome = ev["outcome"]

    f.require("context_compacted" in events, "循环吐出了 context_compacted 事件",
              f"事件序列={events}")
    f.require(outcome is not None and outcome.status == "completed",
              "溢出恢复后正常收口(不是 overflow 状态)",
              f"status={outcome.status if outcome else None}")
    if outcome:
        has_summary = any(isinstance(m, dict) and "已被压缩" in (m.get("content") or "")
                          for m in outcome.messages)
        f.require(has_summary, "最终历史中存在摘要消息", "")
        f.require(not llm._script, "脚本恰好耗尽(时序推演与实际执行一致)",
                  f"剩余{len(llm._script)}项")
    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part G｜P5 记账口径修复的回归(大参数 tool_calls) ══════════════════════

async def part_g_drift(f: Findings) -> None:
    section("Part G｜P5 记账口径修复回归(故意喂大 tool_calls 参数)")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_g_"))
    cm, budget, store = make_cm(tmp, min_compact_tokens=999_999)  # 禁掉压缩,纯测记账
    # 参数塞 2000 字符(模拟"把整段代码作为参数传给子任务"的真实形态)
    seed_and_rounds(cm, n_rounds=5, chars_per_result=100, args_pad="x" * 2000)

    reported = cm.counter.current_tokens
    independent = estimate_messages_tokens(cm.messages)
    diff = independent - reported
    rel = abs(diff) / max(1, independent)
    print(f"  增量记账={reported}  独立全量估算={independent}  "
          f"差值={diff} token ({rel:.0%})")
    # v3:修复已落地,flag 转硬断言(修复前实测低估 94%)
    f.require(rel <= 0.15, "★修复回归:大参数下增量记账与全量估算一致",
              f"差值={diff} token ({rel:.0%})")
    shutil.rmtree(tmp, ignore_errors=True)


# ══ 入口 ═══════════════════════════════════════════════════════════════

async def main() -> None:
    f = Findings()
    parts = (part_a_l1, part_b_l2, part_c_l3, part_d_breaker,
             part_e_combined, part_f_loop_overflow, part_g_drift)
    for part in parts:
        try:
            await part(f)
        except Exception as e:
            f.require(False, f"{part.__name__} 无异常跑完", f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    f.report()


if __name__ == "__main__":
    asyncio.run(main())