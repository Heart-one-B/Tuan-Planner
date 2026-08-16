



# tests/real_smoke_llm.py
"""
真实 LLM 驱动的行为有效性测试。

改这三行就能跑,不需要配置文件、不需要环境变量:
    API_KEY = "..."
    BASE_URL = "..."
    MODEL_NAME = "..."

与 smoke_context_full.py / smoke_snapshot.py 的关系:那两套是机制测试,
用 FakeLLMClient 精确控制代码路径,结果每次应当一致。本文件是行为测试,
验证"真实调用能不能正常工作、模型能不能利用给到的上下文",代价是慢、
烧 token,且不追求文字完全一致,只追求关键事实是否被正确利用。

Part 2 的教训(v1→v2 的修订记录,写在这里防止以后重犯):
  v1 把关键事实写进任务指令(user消息)本身——但 user 消息属于
  Compactor 的"头部前缀",P1 不变量保证前缀永不被压缩,所以这个事实
  不管压缩触没触发、压得好不好,都会一直留在上下文里,"最终答案正确"
  这条断言从设计上就必然为真,是个假信号,压根没测到东西。
  v2(当前版本)把事实埋进某一次工具调用的结果里(这是真正可能被
  压缩掉的"中间段"),并且不再靠预算参数赌真实对话的 token 数会不会
  越过阈值,改用 trigger="manual" 确定性驱动——触发时机是工程问题,
  已经被 smoke_context_full.py 用 FakeLLM 精确测过了,这里只想单独
  测"真实模型的摘要会不会漏掉这个事实"这一个变量,manual 触发把它和
  触发时机彻底解耦,不再是两个不确定因素纠缠在一起赌概率。

运行:python -m tests.real_smoke_llm
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from types import SimpleNamespace

from harness.agent.agent import Agent
from harness.agent.termination import AnswerTermination
from harness.context.budget import ContextBudget
from harness.context.compactor import Compactor
from harness.context.context_manager import ContextManager, ContextManagerConfig
from harness.context.offload import OffloadStore, RETRIEVAL_TOOL_NAME
from harness.llm.openai_client import OpenAIClient
from harness.snapshot import FileSnapshotStore, build_snapshot
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor
from harness.tracing.span import Span


# ══ 改这三行 ═══════════════════════════════════════════════════════════
API_KEY = "sk-c034546e2bb64aac95048532a9084d78"
BASE_URL = "https://ws-hlt8a3rzeiu3i7i6.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v4-flash"
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

    def require(self, ok: bool, promise: str, detail: str = "") -> None:
        print(f"  {'✅' if ok else '❌'} [{promise}] {detail}")
        if not ok:
            self.violations.append(f"{promise}: {detail}")

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
                "smoke_context_full.py / smoke_snapshot.py 挡住了,能漏到这里的"
                "大概率是提示词/模型行为层面的问题,排查方向应该是提示词而不是"
                "先怀疑代码。"
            )


# ══ Part 0｜连通性 ping ══════════════════════════════════════════════════

async def part_0_ping(llm, f: Findings) -> bool:
    section("Part 0｜连通性 ping")
    span = Span.begin("ping", "ping")
    text = ""
    try:
        resp = await llm.call(trace_id=span.trace_id, messages=[
            {"role": "user", "content": "只回复两个字:收到"},
        ])
        text = resp.choices[0].message.content or ""
        f.require(bool(text.strip()), "真实 API 调用返回了非空内容", f"回复={text!r}")
        span.end(text, status="success")
        return True
    except Exception as e:
        f.require(False, "真实 API 调用未抛出异常(检查 url/key/model_name)",
                  f"{type(e).__name__}: {e}")
        span.end("", status="error")
        return False


# ══ Part 1｜快照续跑(核心验证:dict 形态回放是否被真实 provider 接受) ═══

async def part_1_snapshot_resume(llm, f: Findings) -> None:
    section("Part 1｜真实模型:快照续跑")
    print("  验证两件事:① RunSnapshot 归一化出的 dict 形态 tool_calls "
         "回放给真实 provider 不报错(这是设计阶段标注的唯一未验证假设);"
         "② 模型真的利用了历史上下文,不是凭空回答。")
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
                    AnswerTermination(), snapshot_store=store)
    outcome1 = await agent1.run("帮我查一下'量子计划'的编号", session_id=session_id)
    f.require(outcome1.status == "completed", "第一轮真实运行正常收口",
              f"status={outcome1.status}")
    if outcome1.status != "completed":
        shutil.rmtree(tmp, ignore_errors=True)
        return

    snap = store.load_latest(session_id)
    resumed = snap.resume_history()

    agent2 = Agent(llm, make_executor(), "你是助手,回答简洁,不要多余解释。",
                    AnswerTermination(), snapshot_store=store)
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


# ══ Part 2｜压缩后关键事实召回(v2:事实埋进工具结果 + manual 确定性触发) ══

async def part_2_compaction_recall(llm, f: Findings) -> None:
    section("Part 2｜真实模型:压缩摘要是否保留了关键事实(v2)")
    print(
        "  v1 的教训:关键事实写在任务指令(user消息)里会落进 Compactor 的\n"
        "  '头部前缀',P1 不变量保证前缀永不压缩,测试因此必然通过、什么都\n"
        "  没测到。v2 改为:事实埋进工具结果(真正可能被压缩掉的中间段),\n"
        "  且用 trigger='manual' 确定性触发,不再赌真实对话的 token 数能否\n"
        "  越过预算阈值——触发时机已被 FakeLLM 测试覆盖过,这里只想单独\n"
        "  隔离测'真实模型摘要会不会漏事实'这一个变量。"
    )
    tmp = Path(tempfile.mkdtemp(prefix="real_compact_"))
    FACT = "项目代号是'凤凰七号',负责人是林薇"

    async def lookup(q: str) -> str:
        if q == "背景资料A":
            # 关键事实混在这条工具结果里,不在任务指令里——这是相对 v1
            # 的核心修正,见函数顶部说明。
            return f"资料A内容:{FACT}。{'以下为普通背景说明,与本次任务无直接关系。' * 15}"
        return f"关于'{q}'的资料:{'无关填充内容。' * 30}"

    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="lookup", description="查询一份背景资料",
        parameters={"q": {"type": "string"}}, required=["q"], func=lookup,
    ))

    # 强制顺序单次调用:不加这条约束时,真实模型可能把多次查询并行成
    # 一轮(parallel tool calling,合理的模型行为),那样整个对话只有
    # 1 轮"带tool_calls的消息",不存在"更早的轮次"可供压缩,不是预算
    # 问题,是任务设计假设了多轮而模型没有照做。
    task = (
        "请依次调用 lookup,每次只调用一个工具、必须等上一次调用的结果"
        "返回后才能调用下一个,不要并行调用多个工具:"
        "1) 查询'背景资料A'。2) 查询'背景资料B'。"
        "3) 查询'背景资料C'。4) 查询'背景资料D'。"
        "查询完毕后不需要总结内容,只需回复'查询完成'。"
    )

    agent = Agent(llm, executor, "你是助手,严格按步骤执行。", AnswerTermination())
    outcome1 = await agent.run(task, session_id="real-compact-fact")
    f.require(outcome1.status == "completed", "四轮查询真实运行正常收口",
              f"status={outcome1.status}")
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

    # ── 拿到真实对话历史后,manual 触发一次真实压缩:确定性,不猜token ──
    budget = ContextBudget(offload_dir=tmp, keep_recent_rounds=1)
    offload_store = OffloadStore(tmp)
    compactor = Compactor(llm_client=llm)   # 摘要器用真实模型
    cm = ContextManager(budget, offload_store, compactor)
    cm.init(outcome1.messages, prefix_len=2)   # system + user(task) 是前缀

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
            "★核心验证:真实模型的摘要保留了埋在工具结果里的关键事实"
            "(不再是v1那种必然为真的假信号)",
            f"摘要片段={summary_msg['content'][:300]!r}",
        )

    # ── 端到端收尾:用压缩后的历史真实续问,验证下游真能用上摘要 ──
    agent2 = Agent(llm, ToolExecutor(), "你是助手,回答简洁。", AnswerTermination())
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


# ══ Part 4｜真实模型看到卸载标记后,会不会主动调用取回工具 ═══════════════

async def part_4_real_retrieval_tool_usage(llm, f: Findings) -> None:
    section("Part 4｜真实模型是否会主动调用取回工具")
    print(
        "  取回工具本身的机制早被FakeLLM测试证明是对的,但'真实模型看到"
        "卸载提示,会不会真的主动去调用这个工具'是纯行为问题,从没测过。"
        "这里把关键事实埋在预览覆盖不到的正中间,逼模型必须调用取回工具"
        "才能答对。"
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
        default_tool_result_max_chars=200,   # 故意设小,确保触发卸载
        offload_preview_chars=100,           # 预览头尾各约50字符,FACT在正中间够不到
    )
    context_config = ContextManagerConfig(
        budget=budget, offload_store=OffloadStore(tmp), compactor=Compactor(),
    )
    agent = Agent(llm, executor, "你是助手,如果内容因过长被截断/卸载,"
                  "可以调用取回工具读取完整原文。",
                  AnswerTermination(), context_config=context_config)

    task = (
        "调用 big_lookup 查询'资料X',然后告诉我资料里提到的隐藏验证码是什么。"
        "如果看到的内容因为过长被截断了,请主动调用取回工具读取完整内容。"
    )
    outcome = await agent.run(task, session_id="real-retrieval")
    f.require(outcome.status == "completed", "真实运行正常收口", f"status={outcome.status}")

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
    f.require("QK-88317" in outcome.final_text,
              "最终答案包含正确的隐藏验证码(证明真读到了全文,不是蒙的)",
              f"回复={outcome.final_text!r}")

    shutil.rmtree(tmp, ignore_errors=True)


# ══ Part 5｜快照 + 压缩组合路径:落盘的摘要能否真实续跑 ═══════════════════

async def part_5_snapshot_after_compaction(llm, f: Findings) -> None:
    section("Part 5｜快照 + 压缩组合路径")
    print(
        "  Part1只测了'未压缩的干净历史'落盘续跑,Part2只测了'压缩'但停留"
        "在内存对象层面,从没落盘。这里把两条路径接起来:真实压缩产出的"
        "摘要消息,真的存成JSON文件、真的从磁盘读回来,再喂给真实模型"
        "续跑——这个组合此前完全没被测过。"
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
    agent1 = Agent(llm, executor, "你是助手,严格按步骤执行。", AnswerTermination())
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

    # 复用真实的 build_snapshot(不手工重复实现一遍序列化逻辑,
    # 用 SimpleNamespace 伪装 outcome/run_ctx 走生产代码本身的路径)
    fake_outcome = SimpleNamespace(status="completed", rounds=4, tool_calls_used=4,
                                   messages=cm.messages)
    fake_run_ctx = SimpleNamespace(context_manager=cm,
                                   span=SimpleNamespace(trace_id="real-snap-compact"))
    snap = build_snapshot(fake_outcome, fake_run_ctx, "real-snap-compact", task)
    store = FileSnapshotStore(tmp / "snapshots")
    store.save(snap)

    loaded = store.load_latest("real-snap-compact")
    f.require(loaded.messages == snap.messages, "落盘再读回,消息内容逐字节一致", "")

    agent2 = Agent(llm, ToolExecutor(), "你是助手,回答简洁。", AnswerTermination())
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


# ══ 入口 ═══════════════════════════════════════════════════════════════

ALL_PARTS = {
    "1": part_1_snapshot_resume,
    "2": part_2_compaction_recall,
    "4": part_4_real_retrieval_tool_usage,
    "5": part_5_snapshot_after_compaction,
}


async def main() -> None:
    import sys
    llm = get_real_llm_client()
    if llm is None:
        return

    f = Findings()
    ok = await part_0_ping(llm, f)
    if not ok:
        print("\nping 失败,跳过后续更贵的测试(先解决连通性问题)。")
        f.report()
        return

    # 支持单独跑某几个 Part,不用每次都全套跑一遍:
    #   python -m tests.real_smoke_llm 1       只跑 Part 1
    #   python -m tests.real_smoke_llm 1 4     只跑 Part 1 和 4
    #   python -m tests.real_smoke_llm         不带参数,全部跑
    # (真实溢出检测 Part 因为需要发送超大请求、免费额度下经济账不划算,
    # 已按需去掉,只保留正常量级的行为验证。)
    requested = sys.argv[1:]
    selected = [ALL_PARTS[k] for k in requested if k in ALL_PARTS] or list(ALL_PARTS.values())
    if requested and not all(k in ALL_PARTS for k in requested):
        print(f"[提示] 参数里有无法识别的 Part 编号,已忽略。可用编号: {list(ALL_PARTS)}")

    for part in selected:
        await part(llm, f)
    f.report()


if __name__ == "__main__":
    asyncio.run(main())