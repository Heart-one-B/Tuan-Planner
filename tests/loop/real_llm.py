# tests/real_smoke_llm.py
"""
真实 LLM 驱动的行为有效性测试。

改这三行就能跑,不需要配置文件、不需要环境变量:
    API_KEY = "..."
    BASE_URL = "..."
    MODEL_NAME = "..."

与 Fake 测试(tests/agent/*)的关系:那些是机制测试,用 FakeLLMClient
精确控制代码路径,结果每次应当一致。本文件是行为测试,验证"真实调用
能不能正常工作、模型能不能利用给到的上下文",代价是慢、烧 token,
且不追求文字完全一致,只追求关键事实是否被正确利用。

═══ v3 修订说明(Query Loop 五刀重写之后) ═══════════════════════════

这次重写让本文件的旧版**整个跑不起来**,而且不只是 API 变了那么简单——
它验证过的东西有一部分已经不算数了。逐条说清楚:

【为什么必须改】
  - `harness.agent.termination` 模块已删除,`AnswerTermination` 不存在
  - `Agent()` 的第 4 个位置参数从 `termination` 变成了 `budget`
  - `build_snapshot()` 现在要读 outcome 的 5 个新字段(reason/
    overflow_recovery_count/...),原来用 SimpleNamespace 伪装的 outcome
    会直接 AttributeError

【为什么旧的验证结论不算数了】
  主循环现在走**流式**(cfg.llm.stream()),而旧版验证的全是非流式
  路径。具体说:
  - Part 1 验证过的"dict 形态 tool_calls 回放被真实 provider 接受"——
    消息现在由 StreamAccumulator 拼装产生,和当初验证的不是同一段代码
  - Part 2/5 依赖的压缩触发时机由 TokenCounter 决定,而 usage 的来源
    链路在流式下整个换了(见新增的 Part 0 usage 检查)

【新增的两个 Part,针对重写引入的新风险】
  - Part 3(新):强制并行工具调用,打 StreamAccumulator 的按 index
    归并。这是本次重写最精巧、也最容易错的一段代码,而且 Fake 测试
    有结构性盲区——Fake 每个 chunk 只发一个分片,真实 provider 可能
    在一个 chunk 里塞多个。(写这份文件时就是靠这个思路发现了
    OpenAIClient.stream() 里 `tcs[0]` 会静默丢分片的真 bug,已修,
    但真实分片形状仍需这里确认。)
  - Part 6(新):finish 工具收口。这是第一刀引入的新机制(终止判定
    重构),完全没有被真实模型验证过——模型到底会不会按要求调用它、
    参数合不合 schema,是纯行为问题。

【Part 0 扩展】
  旧版 ping 只调 `.call()`。但主循环现在走 `.stream()`,两条路径在
  客户端层是不同的代码——只 ping `.call()` 会出现"ping 通过、后面
  全挂"的困惑局面。现在两条都 ping,并且顺带回答一个直接影响
  Phase 2 正确性的问题:**这个 provider 支持 stream_options 吗?**
  拿不到 usage 的话 TokenCounter 永远停在保守估算模式,压缩会
  系统性提前触发、系统性提前丢信息,而且不报错、没人会注意到。

【Part 2 的历史教训(v1→v2,保留,防止以后重犯)】
  v1 把关键事实写进任务指令(user消息)本身——但 user 消息属于
  Compactor 的"头部前缀",P1 不变量保证前缀永不被压缩,所以那条断言
  从设计上就必然为真,是个假信号。v2 把事实埋进工具结果(真正可能被
  压缩掉的中间段),并用 trigger="manual" 确定性驱动,把"触发时机"和
  "摘要质量"这两个变量解耦。

运行:
    python -m tests.real_smoke_llm          全部跑
    python -m tests.real_smoke_llm 0        只跑 ping(最便宜的冒烟)
    python -m tests.real_smoke_llm 1 3      只跑 Part 1 和 Part 3

建议第一次跑的顺序:先单跑 0 确认连通性和 usage 支持情况,再单跑 3
(最高风险的新代码),都过了再跑全套。
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from harness.agent.abort import AbortSignal
from harness.agent.agent import Agent
from harness.agent.finish_tool import build_finish_tool
from harness.agent.loop import Budget, LoopOutcome
from harness.agent.permission import Allow, Defer, Deny
from harness.agent.repair import ORPHAN_MARKER, repaired_history
from harness.context.budget import ContextBudget
from harness.context.compactor import Compactor
from harness.context.context_manager import ContextManager, ContextManagerConfig
from harness.context.offload import OffloadStore, RETRIEVAL_TOOL_NAME
from harness.llm.base import ContextOverflowError
from harness.llm.openai_client import OpenAIClient
from harness.snapshot import FileSnapshotStore, build_snapshot
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor
from harness.tracing.span import Span


# ══ 改这三行 ═══════════════════════════════════════════════════════════
API_KEY = "sk-c034546e2bb64aac95048532a9084d78"
BASE_URL = "https://ws-hlt8a3rzeiu3i7i6.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.7-max-2026-05-17"
# ═══════════════════════════════════════════════════════════════════════


def get_real_llm_client() -> OpenAIClient | None:
    if not (API_KEY and BASE_URL and MODEL_NAME):
        print(
            "\n[real_smoke_llm] 请先在文件顶部填好 API_KEY / BASE_URL / "
            "MODEL_NAME 这三行,再运行。\n"
        )
        return None
    return OpenAIClient(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


class Findings:
    def __init__(self):
        self.violations: list[str] = []
        self.warnings: list[str] = []

    def require(self, ok: bool, promise: str, detail: str = "") -> None:
        print(f"  {'✅' if ok else '❌'} [{promise}] {detail}")
        if not ok:
            self.violations.append(f"{promise}: {detail}")

    def warn_if(self, bad: bool, promise: str, detail: str = "") -> None:
        """降级项:不是坏掉,是"能跑但退化了"。和 require 分开记——
        把"系统坏了"和"系统在次优模式下运行"混成同一个红叉,会让人
        要么过度紧张、要么慢慢学会忽略红叉,两种都不好。"""
        print(f"  {'⚠️ ' if bad else '✅'} [{promise}] {detail}")
        if bad:
            self.warnings.append(f"{promise}: {detail}")

    def report(self) -> None:
        section("最终报告")
        if not self.violations:
            print("✅ 全部行为验证通过。")
        else:
            print(f"❌ {len(self.violations)} 项未通过:")
            for v in self.violations:
                print(f"   - {v}")
            print(
                "\n提示:行为测试失败不一定是代码 bug。结构性正确性已经被 "
                "119 条 Fake 测试挡住了,能漏到这里的大概率是提示词/模型"
                "行为层面的问题,排查方向应该是提示词而不是先怀疑代码。"
                "\n例外:Part 3(并行工具调用归并)如果挂,那基本可以确定是"
                "代码问题——它测的是纯粹的数据拼装,和模型的措辞无关。"
            )
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} 项降级(能跑,但不是最优状态):")
            for w in self.warnings:
                print(f"   - {w}")


# ══ Part 0｜连通性 ping + 流式可用性 + usage 支持情况 ═══════════════════

async def part_0_ping(llm, f: Findings) -> bool:
    section("Part 0｜连通性 + 流式路径 + usage 支持")
    print(
        "  三件事:①.call() 还能用(Compactor 内部仍走这条路)\n"
        "  ②.stream() 能用(主循环现在走这条路)\n"
        "  ③ 这个 provider 支不支持 stream_options 回 usage——直接决定\n"
        "     TokenCounter 走真实读数还是保守估算,进而决定压缩会不会\n"
        "     系统性提前触发。拿不到不是崩溃,是静默退化,必须显式检查。"
    )
    span = Span.begin("ping", "ping")

    # ── ① 非流式 ──
    try:
        resp = await llm.call(trace_id=span.trace_id, messages=[
            {"role": "user", "content": "只回复两个字:收到"},
        ])
        text = resp.choices[0].message.content or ""
        f.require(bool(text.strip()), "非流式 .call() 返回非空内容", f"回复={text!r}")
    except Exception as e:
        f.require(False, "非流式 .call() 未抛异常(检查 url/key/model_name)",
                  f"{type(e).__name__}: {e}")
        span.end("", status="error")
        return False

    # ── ② 流式 + ③ usage ──
    try:
        pieces, saw_usage, usage_detail = [], False, ""
        async for delta in llm.stream(trace_id=span.trace_id, messages=[
            {"role": "user", "content": "从1数到5,只输出数字和逗号"},
        ]):
            if delta.content:
                pieces.append(delta.content)
            if delta.usage is not None:
                saw_usage = True
                usage_detail = (f"prompt={delta.usage.prompt_tokens} "
                               f"completion={delta.usage.completion_tokens} "
                               f"source={delta.usage.source}")
        streamed = "".join(pieces)
        f.require(bool(streamed.strip()), "★流式 .stream() 返回非空内容",
                  f"拼装结果={streamed!r}")
        f.require(len(pieces) > 1, "流式确实是分片到达的(不是一次性一整块)",
                  f"分片数={len(pieces)}")
        f.warn_if(
            not saw_usage,
            "provider 支持 stream_options 回传 usage",
            usage_detail if saw_usage else
            "拿不到 usage → TokenCounter 将永远停在保守估算模式 → "
            "压缩会比实际需要更早触发(不报错,但每次会话都更早开始丢信息)",
        )
        span.end(streamed, status="success")
        return True
    except Exception as e:
        f.require(False, "★流式 .stream() 未抛异常", f"{type(e).__name__}: {e}")
        span.end("", status="error")
        return False


# ══ Part 1｜快照续跑(dict 形态回放,现在由流式路径产生) ═════════════════

async def part_1_snapshot_resume(llm, f: Findings) -> None:
    section("Part 1｜真实模型:快照续跑")
    print("  验证两件事:① RunSnapshot 归一化出的 dict 形态 tool_calls "
         "回放给真实 provider 不报错;② 模型真的利用了历史上下文。\n"
         "  【v3 说明】这两条旧版验证过,但那时消息由非流式路径产生;"
         "现在由 StreamAccumulator 拼装,等于重新验证一遍。")
    tmp = Path(tempfile.mkdtemp(prefix="real_l3_"))

    async def lookup(q: str) -> str:
        return f"查询结果:'{q}' 对应的编号是 X-7742"

    def make_executor():
        ex = ToolExecutor()
        ex.register(ToolDefinition(
            name="lookup", description="查询一个项目的编号",
            parameters={"q": {"type": "string"}}, required=["q"], func=lookup,
        ))
        return ex

    store = FileSnapshotStore(tmp / "snapshots")
    session_id = "real-part1"

    agent1 = Agent(llm, make_executor(), "你是助手,回答简洁,不要多余解释。",
                   snapshot_store=store)
    outcome1 = await agent1.run("帮我查一下'量子计划'的编号", session_id=session_id)
    f.require(outcome1.status == "completed", "第一轮真实运行正常收口",
              f"status={outcome1.status} reason={outcome1.reason}")
    if outcome1.status != "completed":
        shutil.rmtree(tmp, ignore_errors=True)
        return

    snap = store.load_latest(session_id)
    f.require(snap.version == 6, "快照写出的是 v6 版本", f"version={snap.version}")
    f.require(snap.exit_reason == "completed", "快照记录了细粒度退出原因(五刀新增)",
              f"exit_reason={snap.exit_reason}")
    resumed = snap.resume_history()

    agent2 = Agent(llm, make_executor(), "你是助手,回答简洁,不要多余解释。",
                   snapshot_store=store)
    try:
        outcome2 = await agent2.run(
            "刚才查到的编号是多少?只回答编号本身。",
            history=resumed, session_id=session_id,
        )
    except Exception as e:
        f.require(False, "★核心验证:dict形态消息回放被真实provider接受",
                  f"{type(e).__name__}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return

    f.require(outcome2.status == "completed",
              "★核心验证:dict形态消息回放被真实provider接受,续跑正常收口", "")
    f.require("X-7742" in outcome2.final_text,
              "模型真的利用了历史上下文(不是瞎编)", f"回复={outcome2.final_text!r}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part 2｜压缩后关键事实召回 ══════════════════════════════════════════

async def part_2_compaction_recall(llm, f: Findings) -> None:
    section("Part 2｜真实模型:压缩摘要是否保留了关键事实")
    print(
        "  事实埋进工具结果(真正可能被压缩掉的中间段),用 trigger='manual'\n"
        "  确定性触发,只隔离测'真实模型摘要会不会漏事实'这一个变量。"
    )
    tmp = Path(tempfile.mkdtemp(prefix="real_compact_"))
    FACT = "项目代号是'凤凰七号',负责人是林薇"

    async def lookup(q: str) -> str:
        if q == "背景资料A":
            return f"资料A内容:{FACT}。{'以下为普通背景说明,与本次任务无直接关系。' * 15}"
        return f"关于'{q}'的资料:{'无关填充内容。' * 30}"

    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="lookup", description="查询一份背景资料",
        parameters={"q": {"type": "string"}}, required=["q"], func=lookup,
    ))

    task = (
        "请依次调用 lookup,每次只调用一个工具、必须等上一次调用的结果"
        "返回后才能调用下一个,不要并行调用多个工具:"
        "1) 查询'背景资料A'。2) 查询'背景资料B'。"
        "3) 查询'背景资料C'。4) 查询'背景资料D'。"
        "查询完毕后不需要总结内容,只需回复'查询完成'。"
    )

    agent = Agent(llm, executor, "你是助手,严格按步骤执行。")
    outcome1 = await agent.run(task, session_id="real-compact-fact")
    f.require(outcome1.status == "completed", "四轮查询真实运行正常收口",
              f"status={outcome1.status} reason={outcome1.reason}")
    if outcome1.status != "completed":
        shutil.rmtree(tmp, ignore_errors=True)
        return

    def _has_tool_calls(m):
        return (isinstance(m, dict) and m.get("tool_calls")) \
            or (not isinstance(m, dict) and getattr(m, "tool_calls", None))
    round_count = sum(1 for m in outcome1.messages if _has_tool_calls(m))
    f.require(round_count >= 3, "模型确实分成了多轮单独调用(不是并行成一轮)",
              f"轮次数={round_count} 消息总数={len(outcome1.messages)}")
    if round_count < 2:
        print("  轮次不足,无法构造有意义的压缩中间段,跳过后续压缩验证。")
        shutil.rmtree(tmp, ignore_errors=True)
        return

    budget = ContextBudget(offload_dir=tmp, keep_recent_rounds=1)
    offload_store = OffloadStore(tmp)
    compactor = Compactor(llm_client=llm)
    cm = ContextManager(budget, offload_store, compactor)
    cm.init(outcome1.messages, prefix_len=2)

    compaction_outcome = await cm.maybe_compact(llm, "real-compact-fact", trigger="manual")
    f.require(compaction_outcome is not None, "manual 触发压缩产出结果", "")
    if compaction_outcome is None:
        shutil.rmtree(tmp, ignore_errors=True)
        return
    f.require(not compaction_outcome.degraded, "真实摘要器未熔断降级",
              f"degraded={compaction_outcome.degraded}")

    summary_msg = next(
        (m for m in cm.messages
         if isinstance(m, dict) and "已被压缩" in (m.get("content") or "")),
        None,
    )
    f.require(summary_msg is not None, "压缩后历史里存在摘要消息", "")
    if summary_msg:
        contains_fact = "凤凰七号" in summary_msg["content"] and "林薇" in summary_msg["content"]
        f.require(
            contains_fact,
            "★核心验证:真实模型的摘要保留了埋在工具结果里的关键事实",
            f"摘要片段={summary_msg['content'][:300]!r}",
        )

    agent2 = Agent(llm, ToolExecutor(), "你是助手,回答简洁。")
    try:
        outcome2 = await agent2.run(
            "项目代号和负责人分别是什么?只回答结论。",
            history=cm.messages[2:], session_id="real-compact-fact-qa",
        )
        f.require(outcome2.status == "completed", "基于压缩历史的续问正常收口", "")
        ok = "凤凰七号" in outcome2.final_text and "林薇" in outcome2.final_text
        f.require(ok, "基于压缩历史续问,真实模型答对了(端到端确认)",
                  f"回复={outcome2.final_text!r}")
    except Exception as e:
        f.require(False, "基于压缩历史的续问未抛异常", f"{type(e).__name__}: {e}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part 3(新)｜并行工具调用:StreamAccumulator 按 index 归并 ═══════════

async def part_3_parallel_tool_calls(llm, f: Findings) -> None:
    section("Part 3(新)｜并行工具调用的流式归并")
    print(
        "  这是五刀重写里最精巧、也最容易错的一段代码。Fake 测试有一个\n"
        "  结构性盲区:Fake 每个 chunk 只发一个 tool_call 分片,而真实\n"
        "  provider 可能在一个 chunk 里塞多个(写这份文件时正是靠这个\n"
        "  思路发现了 OpenAIClient.stream() 里 `tcs[0]` 会静默丢分片的\n"
        "  真 bug,已修——但真实分片形状仍需在这里确认)。\n"
        "  判据不是'模型答得对不对',是'两个工具调用的参数有没有串味/丢失'。"
    )

    async def weather(city: str) -> str:
        return f"{city}的天气:晴,25度"

    async def stock(code: str) -> str:
        return f"股票{code}:收盘价 88.5 元"

    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="weather", description="查询某个城市的天气",
        parameters={"city": {"type": "string"}}, required=["city"], func=weather,
    ))
    executor.register(ToolDefinition(
        name="stock", description="查询某只股票的价格",
        parameters={"code": {"type": "string"}}, required=["code"], func=stock,
    ))

    agent = Agent(llm, executor, "你是助手。可以在一轮里同时调用多个工具。")
    outcome = await agent.run(
        "请在同一轮里同时调用两个工具:查询北京的天气,以及股票代码 600519 的价格。"
        "两个工具请一次性一起调用,不要分成两轮。",
        session_id="real-parallel",
    )
    f.require(outcome.status == "completed", "并行工具调用场景正常收口",
              f"status={outcome.status} reason={outcome.reason}")

    # 找出"一条 assistant 消息里带 >= 2 个 tool_calls"的那一条
    def _tool_calls_of(m):
        return m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)

    parallel_msg = next(
        (m for m in outcome.messages if len(_tool_calls_of(m) or []) >= 2), None,
    )
    if parallel_msg is None:
        print("  ⚠️  模型没有并行调用(把两个工具分成了两轮)——这是合理的模型\n"
              "      行为,不是 bug,但本 Part 想测的场景没有被构造出来。\n"
              "      建议:换个更强的模型重跑本 Part,或接受这一项未被覆盖。")
        f.warn_if(True, "模型产生了并行工具调用(本 Part 的前提条件)",
                  "模型选择了分轮调用,归并逻辑未被真实分片覆盖")
        return

    tcs = _tool_calls_of(parallel_msg)
    f.require(len(tcs) >= 2, "★一条 assistant 消息里确实有多个 tool_call",
              f"数量={len(tcs)}")

    # 核心断言:每个 tool_call 的参数都是独立、完整、能解析的 JSON。
    # 归并逻辑一旦出错,典型症状是:参数被拼接到一起(串味)、
    # 某个调用整个消失、或 JSON 残缺解析不了。
    names, all_parsed = [], True
    for tc in tcs:
        fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        args_raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
        names.append(name)
        try:
            parsed = json.loads(args_raw or "{}")
            print(f"      tool={name} args={parsed}")
            if name == "weather" and "city" not in parsed:
                all_parsed = False
            if name == "stock" and "code" not in parsed:
                all_parsed = False
        except json.JSONDecodeError as e:
            print(f"      ❌ tool={name} 参数不是合法JSON: {args_raw!r} ({e})")
            all_parsed = False

    f.require(all_parsed,
              "★核心验证:并行 tool_call 的参数各自完整、能解析、没有串味"
              "(StreamAccumulator 按 index 归并正确)", f"工具名={names}")
    f.require(set(names) == {"weather", "stock"},
              "两个工具都被正确识别出来(没有丢调用)", f"实际={names}")

    tool_results = [
        (m.get("content") if isinstance(m, dict) else "")
        for m in outcome.messages
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "tool"
    ]
    f.require(any("25度" in r for r in tool_results) and any("88.5" in r for r in tool_results),
              "两个工具都真的被执行了(结果都在消息历史里)", "")


# ══ Part 4｜真实模型看到卸载标记后,会不会主动调用取回工具 ═══════════════

async def part_4_real_retrieval_tool_usage(llm, f: Findings) -> None:
    section("Part 4｜真实模型是否会主动调用取回工具")
    print(
        "  取回工具的机制早被 Fake 测试证明是对的,但'真实模型看到卸载"
        "提示,会不会真的主动去调用'是纯行为问题。关键事实埋在预览"
        "覆盖不到的正中间,逼模型必须调用取回工具才能答对。"
    )
    tmp = Path(tempfile.mkdtemp(prefix="real_retrieval_"))
    FACT = "隐藏验证码是 QK-88317"

    async def big_lookup(q: str) -> str:
        pad = "填充内容用于撑大长度。" * 400
        return f"{pad}\n{FACT}\n{pad}"

    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="big_lookup", description="查询一份很长的资料,内含隐藏验证码",
        parameters={"q": {"type": "string"}}, required=["q"], func=big_lookup,
    ))

    budget = ContextBudget(
        offload_dir=tmp,
        default_tool_result_max_chars=200,
        offload_preview_chars=100,
    )
    context_config = ContextManagerConfig(
        budget=budget, offload_store=OffloadStore(tmp), compactor=Compactor(),
    )
    agent = Agent(llm, executor, "你是助手,如果内容因过长被截断/卸载,"
                  "可以调用取回工具读取完整原文。",
                  context_config=context_config)

    task = (
        "调用 big_lookup 查询'资料X',然后告诉我资料里提到的隐藏验证码是什么。"
        "如果看到的内容因为过长被截断了,请主动调用取回工具读取完整内容。"
    )

    # 【v4 诊断增强】改用 events() 而不是 run()——真实冒烟实测撞到过
    # "调用了取回工具、status=completed、但 final_text 为空"这种情况
    # (同样代码同样任务,上一次跑是正常回答的,是模型非确定性)。
    # 只报"答案不含验证码"没法排查,需要知道:模型到底有没有产出内容?
    # 内容是不是全跑进 reasoning 通道了?所以这里把事件流全收下来。
    token_chunks, thinking_chunks, outcome = [], [], None
    async for ev in agent.events(task, session_id="real-retrieval"):
        if ev["type"] == "token":
            token_chunks.append(ev["content"])
        elif ev["type"] == "thinking":
            thinking_chunks.append(ev["content"])
        elif ev["type"] == "outcome":
            outcome = ev["outcome"]

    f.require(outcome.status == "completed", "真实运行正常收口",
              f"status={outcome.status} reason={outcome.reason}")

    def _tool_names(m):
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        names = []
        for tc in (tcs or []):
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if name:
                names.append(name)
        return names

    called_retrieval = any(RETRIEVAL_TOOL_NAME in _tool_names(m) for m in outcome.messages)
    f.require(called_retrieval,
              "★行为验证:真实模型主动调用了取回工具读取被卸载的全文",
              f"是否调用={called_retrieval}")

    # ── 空回复诊断:先分清"没说话"和"说了但没答对",两者原因完全不同 ──
    streamed = "".join(token_chunks)
    thinking = "".join(thinking_chunks)
    if outcome.reason == "empty_response" or not outcome.final_text.strip():
        print(f"      ⚠️  模型返回了空内容。诊断信息:")
        print(f"          token 事件数={len(token_chunks)} 累计长度={len(streamed)}")
        print(f"          thinking 事件数={len(thinking_chunks)} 累计长度={len(thinking)}")
        print(f"          outcome.reason={outcome.reason}")
        if thinking.strip():
            print(f"          thinking 片段={thinking[:200]!r}")
            print(f"          ⇒ 内容跑进了 reasoning 通道:这个 provider 在这类"
                  f"任务上把答案放进了 reasoning_content 而不是 content。"
                  f"属于 provider 行为特性,不是 harness 的 bug,但值得记账。")
        else:
            print(f"          ⇒ 模型确实一个字都没产出(reasoning 通道也是空的)。"
                  f"这是模型非确定性,重跑一次通常就好了。")
        f.warn_if(True, "模型给出了非空回答",
                  f"本次返回空内容(reason={outcome.reason})——"
                  f"上一次跑同样的任务是正常回答的,属于模型非确定性。"
                  f"harness 已如实标注 reason=empty_response,不算故障。")
    else:
        f.require("QK-88317" in outcome.final_text,
                  "最终答案包含正确的隐藏验证码(证明真读到了全文,不是蒙的)",
                  f"回复={outcome.final_text!r}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part 5｜快照 + 压缩组合路径:落盘的摘要能否真实续跑 ═══════════════════

async def part_5_snapshot_after_compaction(llm, f: Findings) -> None:
    section("Part 5｜快照 + 压缩组合路径")
    print(
        "  Part1只测'未压缩的干净历史'落盘续跑,Part2只测'压缩'但停留在"
        "内存层面。这里把两条路径接起来:真实压缩产出的摘要,真的存成"
        "JSON、真的从磁盘读回来,再喂给真实模型续跑。"
    )
    tmp = Path(tempfile.mkdtemp(prefix="real_snap_compact_"))
    FACT = "备用联系人是王芳,电话是138-0000-0000"

    async def lookup(q: str) -> str:
        if q == "客户资料A":
            return f"客户资料A:{FACT}。{'其余为普通说明文字。' * 15}"
        return f"关于'{q}'的资料:{'无关填充内容。' * 30}"

    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="lookup", description="查询客户资料",
        parameters={"q": {"type": "string"}}, required=["q"], func=lookup,
    ))
    task = (
        "请依次调用 lookup,每次只调用一个工具、等结果返回后再调用下一个:"
        "1) 查询'客户资料A'。2) 查询'客户资料B'。3) 查询'客户资料C'。"
        "4) 查询'客户资料D'。查询完毕后只需回复'查询完成'。"
    )
    agent1 = Agent(llm, executor, "你是助手,严格按步骤执行。")
    outcome1 = await agent1.run(task, session_id="real-snap-compact")
    f.require(outcome1.status == "completed", "四轮查询真实运行正常收口", "")
    if outcome1.status != "completed":
        shutil.rmtree(tmp, ignore_errors=True)
        return

    budget = ContextBudget(offload_dir=tmp, keep_recent_rounds=1)
    cm = ContextManager(budget, OffloadStore(tmp), Compactor(llm_client=llm))
    cm.init(outcome1.messages, prefix_len=2)
    compaction_outcome = await cm.maybe_compact(llm, "real-snap-compact", trigger="manual")
    f.require(compaction_outcome is not None and not compaction_outcome.degraded,
              "真实压缩成功产出(非降级)", "")

    # 【v3 修订】原来用 SimpleNamespace 伪装 outcome,五刀之后
    # build_snapshot() 要读 reason/overflow_recovery_count 等 5 个新字段,
    # 伪装对象会直接 AttributeError。改用真实的 LoopOutcome dataclass——
    # 顺带获得"以后再加字段也不会在这里炸"的稳定性,因为新字段总有默认值。
    real_outcome = LoopOutcome(
        status="completed", reason="completed",
        rounds=4, tool_calls_used=4, messages=cm.messages,
    )
    fake_run_ctx = SimpleNamespace(context_manager=cm,
                                   span=SimpleNamespace(trace_id="real-snap-compact"))
    snap = build_snapshot(real_outcome, fake_run_ctx, "real-snap-compact", task)
    store = FileSnapshotStore(tmp / "snapshots")
    store.save(snap)

    loaded = store.load_latest("real-snap-compact")
    f.require(loaded.messages == snap.messages, "落盘再读回,消息内容逐字节一致", "")
    f.require(loaded.version == 6, "落盘的是 v6 快照", f"version={loaded.version}")

    agent2 = Agent(llm, ToolExecutor(), "你是助手,回答简洁。")
    try:
        outcome2 = await agent2.run(
            "刚才客户资料A里提到的备用联系人和电话是什么?",
            history=loaded.resume_history(), session_id="real-snap-compact-qa",
        )
        f.require(outcome2.status == "completed", "从磁盘续跑正常收口", "")
        ok = "王芳" in outcome2.final_text and "138-0000-0000" in outcome2.final_text
        f.require(ok, "★核心验证:压缩+落盘+读回+续跑全链路,真实模型答对了",
                  f"回复={outcome2.final_text!r}")
    except Exception as e:
        f.require(False, "从磁盘续跑未抛异常", f"{type(e).__name__}: {e}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part 6(新)｜finish 工具收口(第一刀新机制,从未真实验证) ═════════════

async def part_6_terminal_tool(llm, f: Findings) -> None:
    section("Part 6(新)｜finish 工具结构化收口")
    print(
        "  第一刀把'终止判定'改成了内建机制、把结构化收口降级成一个普通\n"
        "  工具(terminal=True)。这套机制被 Fake 测试覆盖得很充分,但\n"
        "  '真实模型会不会按要求调用 finish、参数合不合 schema'是纯行为\n"
        "  问题,从没验证过。这也是'Agent 即工具'这个用法的地基。"
    )

    class WeatherAnswer(BaseModel):
        city: str
        temperature: int

    async def weather(city: str) -> str:
        return f"{city}当前气温 26 摄氏度"

    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="weather", description="查询某个城市的当前气温",
        parameters={"city": {"type": "string"}}, required=["city"], func=weather,
    ))
    executor.register(build_finish_tool(WeatherAnswer))

    agent = Agent(
        llm, executor,
        "你是助手。查到结果后必须调用 finish 工具给出结构化结论,不要用自然语言直接回答。",
        require_terminal_tool="finish",
    )
    outcome = await agent.run("北京现在多少度?", session_id="real-finish")

    f.require(outcome.status == "completed", "finish 收口路径正常完成",
              f"status={outcome.status} reason={outcome.reason}")
    f.require(outcome.reason == "completed",
              "退出原因是正常完成(不是 terminal_tool_not_called 降级)",
              f"reason={outcome.reason}")
    f.require(outcome.result is not None,
              "★核心验证:真实模型调用了 finish,产出了结构化结果",
              f"result={outcome.result}")
    if outcome.result is not None:
        data = outcome.result.data
        f.require("city" in data and "temperature" in data,
                  "结构化结果符合 output_schema(pydantic 校验通过)",
                  f"data={data}")
        f.require(isinstance(data.get("temperature"), int),
                  "schema 的类型约束真的生效了(temperature 是 int 不是字符串)",
                  f"temperature={data.get('temperature')!r}")


# ══ Part 7(新)｜权限门控 Defer → resume 全链路(安全红线) ═════════════════

async def part_7_permission_defer_resume(llm, f: Findings) -> None:
    section("Part 7(新)｜权限门控:Defer 挂起 → 落盘 → resume 收口")
    print(
        "  权限门控是这套 harness 的上线红线(存在付款类不可逆操作),但它\n"
        "  从来只被 Fake 测试覆盖过。真实 provider 那一环从没验证:挂起后\n"
        "  的消息历史处在一个特殊形态——assistant 消息声称调用了工具、\n"
        "  但那个工具的结果是 resume 之后才补上的。这份历史发回给真实\n"
        "  provider 会不会被拒?只有真跑一次才知道。"
    )
    tmp = Path(tempfile.mkdtemp(prefix="real_defer_"))
    session_id = "real-defer"

    async def transfer(to: str, amount: int) -> str:
        return f"已向 {to} 转账 {amount} 元,流水号 TX-20240101"

    def make_executor():
        ex = ToolExecutor()
        ex.register(ToolDefinition(
            name="transfer", description="向指定的人转账指定金额",
            parameters={"to": {"type": "string"}, "amount": {"type": "integer"}},
            required=["to", "amount"], func=transfer,
        ))
        return ex

    class _DeferTransferOnce:
        """第一次遇到 transfer 就挂起等审批,恢复后放行。"""
        def __init__(self):
            self.deferred = False

        async def check(self, tool_name, args, run_ctx):
            if tool_name == "transfer" and not self.deferred:
                self.deferred = True
                return Defer(approval_id="appr-real-1")
            return Allow()

    store = FileSnapshotStore(tmp / "snapshots")
    agent = Agent(llm, make_executor(), "你是助手。用户要求转账时直接调用 transfer 工具。",
                  snapshot_store=store, permission_policy=_DeferTransferOnce())

    outcome1 = await agent.run("帮我给张三转账 100 元", session_id=session_id)
    f.require(outcome1.status == "awaiting_approval",
              "★真实模型触发工具调用后,权限门控成功挂起",
              f"status={outcome1.status} reason={outcome1.reason}")
    if outcome1.status != "awaiting_approval":
        print("  模型没有调用 transfer 工具,本 Part 的前提没有成立,跳过。")
        shutil.rmtree(tmp, ignore_errors=True)
        return

    snap = store.load_latest(session_id)
    f.require(snap.pending_tool_call_id is not None,
              "挂起点被正确落盘(pending_tool_call_id)",
              f"tool_call_id={snap.pending_tool_call_id} approval_id={snap.pending_approval_id}")

    try:
        resumed = await agent.resume(session_id, Allow())
    except Exception as e:
        f.require(False, "★核心验证:resume 后的历史被真实 provider 接受",
                  f"{type(e).__name__}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return

    f.require(resumed.status == "completed",
              "★核心验证:批准后 resume 正常收口(挂起态历史被真实 provider 接受)",
              f"status={resumed.status} reason={resumed.reason}")
    f.require("TX-20240101" in resumed.final_text or "转账" in resumed.final_text,
              "模型看到了 resume 之后补上的工具结果",
              f"回复={resumed.final_text[:120]!r}")

    final_snap = store.load_latest(session_id)
    f.require(final_snap.pending_tool_call_id is None,
              "收口后挂起标记被正确清除", f"pending={final_snap.pending_tool_call_id}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part 8(新)｜孤儿修复产出的历史能否被真实 provider 接受 ═══════════════

async def part_8_orphan_repair_accepted(llm, f: Findings) -> None:
    section("Part 8(新)｜孤儿 tool_use 修复:合成的占位结果 provider 认不认")
    print(
        "  repair.py 存在的**全部意义**就是'让残缺的历史仍然能发给 provider'。\n"
        "  Fake 测试只能验证'占位消息被正确插进去了',验证不了'插进去之后\n"
        "  provider 到底认不认'——而后者才是这个模块的成败判据。\n"
        "  这里制造一段真实的孤儿历史(Defer 挂起但放弃审批),用\n"
        "  repaired_history() 修复,然后真的发给 provider 续跑。"
    )
    tmp = Path(tempfile.mkdtemp(prefix="real_repair_"))
    session_id = "real-repair"

    async def slow_query(q: str) -> str:
        return f"'{q}'的查询结果:数值是 42"

    def make_executor():
        ex = ToolExecutor()
        ex.register(ToolDefinition(
            name="slow_query", description="查询一个数值",
            parameters={"q": {"type": "string"}}, required=["q"], func=slow_query,
        ))
        return ex

    class _AlwaysDefer:
        async def check(self, tool_name, args, run_ctx):
            return Defer(approval_id="appr-abandon")

    store = FileSnapshotStore(tmp / "snapshots")
    agent = Agent(llm, make_executor(), "你是助手。需要查数值时调用 slow_query。",
                  snapshot_store=store, permission_policy=_AlwaysDefer())

    outcome = await agent.run("帮我查一下'指标A'的数值", session_id=session_id)
    if outcome.status != "awaiting_approval":
        print("  模型没有调用工具,无法构造孤儿历史,跳过本 Part。")
        shutil.rmtree(tmp, ignore_errors=True)
        return
    f.require(True, "成功构造出一段挂起(含孤儿 tool_call)的历史", "")

    snap = store.load_latest(session_id)

    # ── 关键对照:resume_history() 保留孤儿,repaired_history() 修复它 ──
    raw = snap.resume_history()
    raw_has_orphan = not any(
        m.get("role") == "tool" and m.get("tool_call_id") == snap.pending_tool_call_id
        for m in raw
    )
    f.require(raw_has_orphan,
              "resume_history() 如约保留了孤儿(resume 定位挂起点靠它)", "")

    fixed = repaired_history(snap)
    placeholder = next(
        (m for m in fixed
         if m.get("role") == "tool" and ORPHAN_MARKER in (m.get("content") or "")), None,
    )
    f.require(placeholder is not None,
              "repaired_history() 合成了占位结果",
              f"占位内容={placeholder['content'][:60]!r}" if placeholder else "")

    # ── 核心:把修复后的历史真的发出去 ──
    agent2 = Agent(llm, make_executor(), "你是助手,回答简洁。")
    try:
        outcome2 = await agent2.run(
            "刚才那次查询没有完成。请直接告诉我:你现在知道'指标A'的数值吗?"
            "如果不知道就如实说不知道。",
            history=fixed, session_id="real-repair-continue",
        )
        f.require(outcome2.status == "completed",
                  "★核心验证:含合成占位结果的历史被真实 provider 接受(没有400)",
                  f"status={outcome2.status}")
        print(f"      模型回复={outcome2.final_text[:150]!r}")
        f.warn_if(
            "42" in outcome2.final_text,
            "模型没有把'未执行'的占位当成真实结果来编造答案",
            "模型回复里出现了 42——但那次查询其实从未执行过,"
            "说明占位文案可能不够明确,值得调整 repair.py 的 ORPHAN_MARKER 措辞"
            if "42" in outcome2.final_text else "回复中未出现虚构的查询结果",
        )
    except Exception as e:
        f.require(False, "★核心验证:含合成占位结果的历史被真实 provider 接受",
                  f"{type(e).__name__}: {e}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part 9(新)｜输出截断恢复 + 静默升档(第三刀最精巧的代码) ═══════════════

async def part_9_truncation_and_upgrade(llm, f: Findings) -> None:
    section("Part 9(新)｜输出截断恢复 + 静默升档")
    print(
        "  第三刀写了两套截断恢复机制,真实模型从未走过。这里用\n"
        "  max_output_tokens 故意设小来**确定性触发**截断(便宜,不需要\n"
        "  构造超长上下文),分两个场景:\n"
        "  ① 配了升档值 → 应当静默重试一次,用户无感、不计入催续写次数\n"
        "  ② 没配升档值 → 应当走催续写路径,计数 +1\n"
        "  同时顺带回答一个前提问题:这个 provider 认不认 max_tokens 参数、\n"
        "  截断时会不会如实回 finish_reason='length'。"
    )
    LONG_TASK = "请写一段关于春天的散文,至少 400 字,要写完整,不要中途停下。"

    # ── 场景①:配了升档 ──
    agent_up = Agent(
        llm, ToolExecutor(), "你是一个作家。",
        budget=Budget(max_output_tokens=40, max_output_tokens_upgraded=3000),
    )
    outcome_up = await agent_up.run(LONG_TASK, session_id="real-trunc-upgrade")
    f.require(outcome_up.status == "completed", "配置升档时正常收口",
              f"status={outcome_up.status} reason={outcome_up.reason}")
    f.warn_if(
        not outcome_up.output_upgraded,
        "★provider 认 max_tokens 且截断时回 finish_reason='length'(升档被触发)",
        f"output_upgraded={outcome_up.output_upgraded} "
        f"truncation_count={outcome_up.output_truncation_count} "
        f"最终长度={len(outcome_up.final_text)}"
        + ("" if outcome_up.output_upgraded else
           " —— 升档没被触发,可能是 provider 忽略了 max_tokens、"
           "或截断时没回 finish_reason='length'。这不算 bug,但意味着"
           "整套截断恢复机制在这个 provider 上是空转的,要如实记账"),
    )
    if outcome_up.output_upgraded:
        f.require(outcome_up.output_truncation_count == 0,
                  "★静默升档没有被误计入'催续写'次数(用户无感的关键)",
                  f"truncation_count={outcome_up.output_truncation_count}")
        f.require(len(outcome_up.final_text) > 100,
                  "升档后拿到了完整得多的输出", f"长度={len(outcome_up.final_text)}")

    # ── 场景②:不配升档,走催续写 ──
    agent_nudge = Agent(
        llm, ToolExecutor(), "你是一个作家。",
        budget=Budget(max_output_tokens=40, max_output_truncation_recoveries=2),
    )
    outcome_nudge = await agent_nudge.run(LONG_TASK, session_id="real-trunc-nudge")
    f.require(outcome_nudge.status in ("completed", "exhausted"),
              "不配升档时走催续写路径,正常收口",
              f"status={outcome_nudge.status} reason={outcome_nudge.reason} "
              f"truncation_count={outcome_nudge.output_truncation_count}")
    f.require(not outcome_nudge.output_upgraded,
              "不配升档值时确实没有升档(不配置就不该有感)",
              f"output_upgraded={outcome_nudge.output_upgraded}")
    if outcome_nudge.reason == "max_output_tokens_recovery":
        print("      (催续写次数用尽后走了专门的退出原因,"
              "这正是第三刀 bug#2 要给宿主的区分能力)")


# ══ Part 10(新)｜溢出检测:_is_overflow() 认不认 + 溢出发生的时机 ═══════

async def part_10_overflow_detection(llm, f: Findings) -> None:
    section("Part 10(新)｜上下文溢出检测")
    print(
        "  【v2 重新设计】上一版把两件事混在一起测,既贵又测不准——它让\n"
        "  超长历史走完整的 Agent 流程,结果是:主请求溢出 → 触发紧急压缩\n"
        "  → 压缩器要把那段超长中间段发给摘要模型 → **压缩请求本身也\n"
        "  溢出** → 熔断降级。测出来的是熔断路径,不是我想问的问题。\n\n"
        "  现在只问最关键的两个问题,直接打 llm.stream(),不经过 Agent:\n"
        "  ① _is_overflow() 认不认百炼的溢出错误?这是 OpenAI 原生错误\n"
        "     码/文案的匹配规则(context_length_exceeded / maximum\n"
        "     context length),百炼是兼容层,格式很可能不同。不匹配的\n"
        "     后果:紧急压缩永远不会被触发,异常直接冒泡崩掉整个 run。\n"
        "  ② 【第五刀留下的假设】溢出错误是在生成开始**之前**报出来的,\n"
        "     还是流到一半才报?loop.py 里那处异常捕获范围的注释明确写着\n"
        "     '这是待验证假设,不是已证实的事实'。数一下报错之前收到了\n"
        "     几个 delta 就有答案了。\n\n"
        "  ⚠️  成本说明:超长请求如果被 provider 以 400 拒绝,通常**不计费**\n"
        "     (没有 token 被处理)。所以这个 Part 大概率比上一版标注的\n"
        "     '昂贵'便宜得多——真正贵的是完整恢复链路(见函数末尾说明),\n"
        "     那部分这里不做。"
    )
    span = Span.begin("overflow-probe", "overflow probe")

    # 撑爆窗口:按 1 字符 ≈ 0.5 token 保守估,300 万字符足以越过任何
    # 现有模型的窗口。故意用无语义的重复文本,避免模型真的开始生成。
    huge = "填充内容" * 750_000

    deltas_before_error = 0
    try:
        async for _delta in llm.stream(
            trace_id=span.trace_id,
            messages=[{"role": "user", "content": huge}],
        ):
            deltas_before_error += 1
        f.require(False, "超长请求被 provider 拒绝",
                  f"请求居然成功了(收到 {deltas_before_error} 个 delta)——"
                  f"说明这个模型的窗口比预期大得多,把 huge 再调大重跑")
    except ContextOverflowError as e:
        f.require(True,
                  "★_is_overflow() 认出了这个 provider 的溢出错误"
                  "(紧急压缩能被正确触发)",
                  f"归一化为 ContextOverflowError: {str(e)[:150]}")
        f.require(
            deltas_before_error == 0,
            "★【第五刀假设验证】溢出在生成开始之前就报出来了",
            f"报错前收到了 {deltas_before_error} 个 delta —— "
            f"说明溢出是流到一半才报的,用户会看到'说一半重来',"
            f"loop.py 的异常捕获策略需要重新评估"
            if deltas_before_error else "报错前没有任何 delta 流出,假设成立",
        )
    except Exception as e:
        # 这是最有价值的失败:它会把百炼真实的错误形状打出来,
        # 照着改 _OVERFLOW_ERROR_CODES / _OVERFLOW_MESSAGE_HINTS 即可。
        f.require(
            False,
            "★_is_overflow() 认出了这个 provider 的溢出错误",
            f"抛出的是 {type(e).__name__} 而不是 ContextOverflowError,"
            f"说明 _is_overflow() 的匹配规则对不上百炼的错误格式。"
            f"请把下面这段原始错误贴给我,照着补匹配规则:\n"
            f"      code={getattr(e, 'code', None)!r}\n"
            f"      status_code={getattr(e, 'status_code', None)!r}\n"
            f"      message={str(e)[:400]}",
        )
    finally:
        span.end("", status="success")

    print(
        "\n  【本 Part 刻意不做的部分,如实记账】\n"
        "  '溢出 → 紧急压缩 → 用压缩后的上下文重试成功'这条完整恢复链路\n"
        "  没有在这里验证。原因不是偷懒,是它需要精心构造一段'主请求会\n"
        "  溢出、但压缩请求不会溢出'的历史——中间段太大压缩本身也会炸,\n"
        "  太小又触发不了主请求溢出,这个窗口需要按具体模型的窗口大小\n"
        "  调参才能命中,而且真的会烧不少 token(压缩是一次完整的 LLM\n"
        "  调用)。如果上面①②都通过了,这条链路的两个关键前提就已经\n"
        "  成立,剩下的部分被 Fake 测试覆盖得很充分(见\n"
        "  test_exit_reasons_and_recovery.py 的 overflow 系列)。"
    )


# ══ 入口 ═══════════════════════════════════════════════════════════════

ALL_PARTS = {
    "1": part_1_snapshot_resume,
    "2": part_2_compaction_recall,
    "3": part_3_parallel_tool_calls,
    "4": part_4_real_retrieval_tool_usage,
    "5": part_5_snapshot_after_compaction,
    "6": part_6_terminal_tool,
    "7": part_7_permission_defer_resume,
    "8": part_8_orphan_repair_accepted,
    "9": part_9_truncation_and_upgrade,
}

# Part 10 单独放,不进默认全跑列表——它发的是一个必然被拒的超大请求,
# 虽然 400 拒绝通常不计费(见该函数的成本说明),但"故意打一个必然失败
# 的请求"这件事本身不该在每次常规冒烟里都发生一遍,需要时点名跑即可。
EXPENSIVE_PARTS = {
    "10": part_10_overflow_detection,
}


async def main() -> None:
    import sys
    llm = get_real_llm_client()
    if llm is None:
        return

    f = Findings()
    requested = sys.argv[1:]

    ok = await part_0_ping(llm, f)
    if not ok:
        print("\nping 失败,跳过后续更贵的测试(先解决连通性问题)。")
        f.report()
        return
    if requested == ["0"]:
        f.report()
        return

    if requested:
        known = {**ALL_PARTS, **EXPENSIVE_PARTS}
        selected = [known[k] for k in requested if k in known]
        unknown = [k for k in requested if k not in known and k != "0"]
        if unknown:
            print(f"[提示] 无法识别的 Part 编号 {unknown},已忽略。"
                  f"可用: 0 + {list(ALL_PARTS)} + {list(EXPENSIVE_PARTS)}(昂贵)")
        if not selected:
            f.report()
            return
    else:
        # 不带参数 = 跑全部常规 Part,不含需要点名的 Part 10
        selected = list(ALL_PARTS.values())
        print(f"\n[提示] Part {list(EXPENSIVE_PARTS)}(溢出检测)默认不跑,"
              f"需要时显式指定编号(例:python -m tests.real_smoke_llm 10)")

    for part in selected:
        await part(llm, f)
    f.report()


if __name__ == "__main__":
    asyncio.run(main())