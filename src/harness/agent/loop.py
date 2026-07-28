# harness/agent/loop.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Literal, Protocol

from harness.agent.permission import Allow, Defer, Deny
from harness.agent.run_context import RunContext
from harness.agent.state import LoopConfig, LoopState
from harness.llm.base import ContextOverflowError, StreamDelta
from harness.llm.streaming import StreamAccumulator

logger = logging.getLogger(__name__)


# ── 消息存储接口(任务 2.9:接口一字不改,硬约束) ────────────────────────
class MessageStore(Protocol):
    @property
    def messages(self) -> list: ...
    def append(self, message) -> None: ...
    def note_api_usage(self, prompt_tokens: int) -> None: ...
    def offload_tool_result(self, trace_id: str, tool_call_id: str,
                            content: str, max_chars: int | None = None) -> str: ...
    async def maybe_compact(self, llm_client, trace_id: str,
                            trigger: str = "threshold", focus: str | None = None): ...


class _PlainMessageStore:
    def __init__(self, messages: list):
        self._messages = messages

    @property
    def messages(self) -> list:
        return self._messages

    def append(self, message) -> None:
        self._messages.append(message)

    def note_api_usage(self, prompt_tokens: int) -> None:
        pass

    def offload_tool_result(self, trace_id, tool_call_id, content, max_chars=None) -> str:
        return content

    async def maybe_compact(self, llm_client, trace_id, trigger="threshold", focus=None):
        return None


def wrap_store(messages: list, run_ctx: RunContext) -> MessageStore:
    if run_ctx.context_manager is not None:
        run_ctx.context_manager.init(messages, prefix_len=len(messages))
        return run_ctx.context_manager
    return _PlainMessageStore(messages)


def _tool_call_view(tc):
    if isinstance(tc, dict):
        fn = tc.get("function", {})
        return SimpleNamespace(
            id=tc.get("id"),
            function=SimpleNamespace(name=fn.get("name"), arguments=fn.get("arguments")),
        )
    return tc


def _find_pending_call_context(messages: list, pending_tool_call_id: str):
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        tcs = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        if role == "assistant" and tcs:
            views = [_tool_call_view(tc) for tc in tcs]
            if any(v.id == pending_tool_call_id for v in views):
                return msg, views
    return None, []


# ── 预算策略 ─────────────────────────────────────────────────────────────
@dataclass
class Budget:
    max_tool_calls: int = 10
    max_rounds: int = 24
    exhausted_action: Literal["stop", "force_answer", "force_finish"] = "force_answer"
    exhausted_prompt: str = (
        "你已用完工具调用次数,请根据已有信息尽力完成任务;"
        "信息不足请如实说明缺少什么,不要编造。"
    )

    # ── 第三刀新增 ──
    max_overflow_recoveries: int = 3
    # 整个 run 累计最多做几次紧急压缩(任务 3.5,修 bug#1)。数字照抄
    # CC 的量级拍的,未经真实负载校准,记 Phase 5 账。与
    # compacted_this_round(每轮最多一次,防同轮死循环)是两道独立防线,
    # 缺一都不完整:只有"每轮一次"防不住"每轮都撞墙、次次都压"的
    # 长期烧钱模式;只有"累计上限"防不住同一轮内的死循环重试。
    max_output_truncation_recoveries: int = 3
    # 输出截断的"催续写"最多允许几次(任务 3.6)。与上面同理,
    # 数字抄 CC,未校准,记 Phase 5 账。
    max_output_tokens: int | None = None
    # 常规请求的输出长度上限。None(默认)= 不传这个参数,完全交给
    # provider 默认行为——"不配置就不该有感"(任务 3.8)。
    max_output_tokens_upgraded: int | None = None
    # 第一次遇到输出截断时,静默切换到的更大上限(不打断用户、不
    # nudge、模型对此无感,见 loop.py 的 _call_model)。None(默认)=
    # 不做这次升档,直接进入"催续写"恢复路径——这是"不配置就不该
    # 有感"的另一半:两个字段任一为 None,整套升档机制都不会被触发。


# ── 循环产出 ─────────────────────────────────────────────────────────────
@dataclass
class LoopOutcome:
    """
    【第三刀改动】status/reason 两级设计(任务 3.1/3.2):
      status  宿主做决策用的粗信号,值域不变(现有代码按 status 分支
              的地方不用改),新增 "aborted"(第四刀才会真正产出,
              这里先把类型开出来,不然第四刀落地时又要改一次 Literal)。
      reason  诊断/上报用的细信号,必填(不给默认值,逼着每个构造点
              显式想清楚"这次退出到底为什么",不允许遗漏)。取值见
              下表(设计方案 §4.3):

        status              reason
        completed           completed / empty_response /
                            terminal_tool_not_called
        exhausted           max_rounds / max_tool_calls /
                            max_output_tokens_recovery
        overflow            context_overflow
        awaiting_approval   awaiting_approval
        aborted             aborted_streaming / aborted_tools(第四刀)

    empty_response 是真实冒烟暴露后补上的:模型返回 finish_reason=
    "stop" 但 content 为空。status 仍是 completed(它确实正常结束了),
    但宿主需要能区分"给了答案"和"什么都没说",否则拿到空字符串却
    被告知一切正常,没有任何依据决定要不要重试。

    overflow_recovery_count/output_truncation_count/output_upgraded/
    terminal_nudge_count 四个字段是"这次 run 结束时,LoopState 对应
    计数器的最终值"——build_snapshot() 靠它们把这些计数写进快照
    (任务 3.9/3.10),不需要 Agent 层单独去读 LoopState(LoopState
    本身不跨层传递,只有 outcome 会)。
    """
    status: Literal["completed", "exhausted", "overflow", "awaiting_approval", "aborted"]
    reason: str
    final_text: str = ""
    result: Any = None
    rounds: int = 0
    tool_calls_used: int = 0
    messages: list = field(default_factory=list)
    pending_approval_id: str | None = None
    pending_tool_call_id: str | None = None
    overflow_recovery_count: int = 0
    output_truncation_count: int = 0
    output_upgraded: bool = False
    terminal_nudge_count: int = 0


_TERMINAL_NUDGE_PROMPT_TEMPLATE = (
    "请调用 {tool} 工具给出结构化结论,不要用自然语言直接回答。"
)

_TRUNCATION_NUDGE_TEXT = (
    "你上一条回复因达到输出长度限制被截断。请直接从断点续写,"
    "不要道歉、不要回顾刚才在做什么,并把剩余工作拆成更小的块输出。"
)


def _truncation_dropped_call_notice(tool_name: str) -> str:
    return (
        f"你上一条回复因达到输出长度限制被截断,其中对 {tool_name} 的调用"
        f"参数不完整,已丢弃、不会执行。请直接继续(如果刚才还有别的意图"
        f"没表达完,请重新完整地表达一次)。"
    )


def _build_assistant_message_dict(msg, tool_call_views: list) -> dict:
    """把一条 assistant 消息重新构造成 dict 形态,tool_calls 只保留
    传入的 views——用于任务 3.7"丢弃最后一个不完整 tool_call"的场景:
    原始 msg 里那个被截断的 tool_call 绝不能进入消息历史(哪怕只是
    作为一个'看起来完整但参数是垃圾'的记录),否则等于制造一条新的
    孤儿/畸形 tool_use,这正是整个"认错哲学"(设计方案 §5.6)要
    根治的问题类型,不能在引入截断处理的同时又埋一个新的同类坑。
    """
    d: dict = {"role": "assistant", "content": getattr(msg, "content", None)
              if not isinstance(msg, dict) else msg.get("content")}
    if tool_call_views:
        d["tool_calls"] = [{
            "id": v.id, "type": "function",
            "function": {"name": v.function.name, "arguments": v.function.arguments},
        } for v in tool_call_views]
    return d


def _to_wire_message(msg) -> dict:
    """【真实冒烟 Part 1 实测发现并修复】_call_model 产出的 msg 是
    StreamAccumulator.build_message() 构造的 SimpleNamespace(为了让
    下游 msg.content/msg.tool_calls/tc.function.name 这类属性访问
    保持和第一到四刀完全一致的写法,不用大改 loop.py 其余部分)。

    但 SimpleNamespace 本身不是 JSON 可序列化对象——如果把它原样
    append 进 state.store.messages,下一轮把完整历史发给 provider 时,
    OpenAI SDK 的请求体序列化器不认识这个类型,直接抛
    `TypeError: Object of type SimpleNamespace is not JSON serializable`。
    第一到四刀从未暴露这个问题,是因为那时候 assistant 消息来自
    `resp.choices[0].message`,是 OpenAI SDK 自己的 pydantic 对象,
    SDK 认得怎么把自己的对象序列化回请求体;第五刀把它换成了
    harness 自己手搓的 SimpleNamespace,SDK 不认识,而 119 条 Fake
    测试从不会真的走到"把消息序列化成 JSON 发出去"这一步,测不出来。

    这里复用 _build_assistant_message_dict(它是给任务 3.7"丢弃末位
    不完整 tool_call"场景写的,恰好就是"msg + 一份 tool_call_views
    列表 → 干净的 wire dict"这个转换),只是这次不做过滤、把 msg 全部
    的 tool_calls 都转换进去——本质是同一个转换,只是调用方不再要求
    "丢弃某几个",而是"全部保留"。转换之后的 dict 天然是 JSON 安全的,
    也让它和第一到四刀留下的行为保持一致(那时 append 进去的历史,
    最终落盘时也是靠 normalize_message() 转成 dict——现在只是把这个
    转换提前到了"进 state.store"这一步,不再拖到落盘才做)。
    """
    if isinstance(msg, dict):
        return msg
    views = [_tool_call_view(tc) for tc in (getattr(msg, "tool_calls", None) or [])]
    return _build_assistant_message_dict(msg, views)


def _counters(state: LoopState) -> dict:
    """把 LoopState 的四个计数器/标志位打包成 kwargs,喂给每一处
    LoopOutcome 构造——避免每个构造点都手写四行、漏写一行不会报错
    只会在快照里悄悄留一个错误的 0。"""
    return {
        "overflow_recovery_count": state.overflow_recovery_count,
        "output_truncation_count": state.output_truncation_count,
        "output_upgraded": state.output_upgraded,
        "terminal_nudge_count": state.terminal_nudge_count,
    }


# ── 第四层:纯循环本体 ────────────────────────────────────────────────────
async def run_query_loop(
    cfg: LoopConfig, state: LoopState, run_ctx: RunContext,
) -> AsyncGenerator[dict, None]:
    """queryLoop 等价物。只负责"转圈",不做任何生命周期收尾——不捕获
    异常、不调用 span.end,这两件事收口到第三层(query.py)。
    """
    span = run_ctx.span
    tools = cfg.tools

    if state.resume_point is not None:
        got_outcome = False
        async for ev in _finish_interrupted_round(cfg, state, run_ctx):
            yield ev
            if ev["type"] == "outcome":
                got_outcome = True
        if got_outcome:
            return
        state.resume_point = None

    while True:
        state.rounds += 1
        state.compacted_this_round = False   # 每轮开头重置(任务 3.4)

        # ── 中断检查点①:每轮开头(任务 4.7)。还没调模型、还没往
        # 消息历史里追加任何东西——这是最干净的中断时机,不需要任何
        # 修复就能直接收口。reason=aborted_streaming:这个检查点在
        # 设计上对应"即将开始接收模型输出"这个阶段,第五刀接入真实
        # 流式后,这里和"流式消费循环内"那个检查点会共用同一个
        # reason,语义是一致的(都是"模型输出阶段被打断",不是
        # "工具执行阶段被打断")。
        if run_ctx.abort is not None and run_ctx.abort.is_set():
            yield _aborted_outcome(state, run_ctx, reason="aborted_streaming")
            return

        if state.rounds > cfg.budget.max_rounds:
            async for ev in _exhaust(cfg, state, run_ctx, reason="max_rounds"):
                yield ev
            return

        try:
            msg = None
            call_aborted = False
            async for ev in _call_model(cfg, state, run_ctx, tools):
                if ev["type"] == "_message":
                    msg = ev["message"]
                elif ev["type"] == "outcome":
                    # 检查点②(流式消费循环内)命中:_call_model 直接
                    # 产出了终态,原样转发并结束,不会再有 "_message" 事件
                    yield ev
                    call_aborted = True
                    break
                else:
                    yield ev
            if call_aborted:
                return
        except ContextOverflowError as e:
            # ── 压缩语义修正(任务 3.4,修 bug#1)──────────────────────
            # 两道独立防线:compacted_this_round 防同一轮内死循环重试
            # (CC 的 hasAttemptedReactiveCompact 语义,每轮重置);
            # overflow_recovery_count 防整个 run 长期烧压缩预算(累计,
            # 不清零,上限 Budget.max_overflow_recoveries,跨 resume
            # 边界存活——第二刀只是把字段搬进了 State,这里才是行为
            # 真正被修正的地方)。旧版(第一/二刀)只有后者、且上限
            # 硬编码为 1,第 3 轮压过一次,第 20 轮再溢出直接判死刑,
            # 即便压缩完全可能成功——那不是防死循环,是自己写死自己。
            #
            # 【第五刀改动】异常捕获范围从"单个 await"扩到"整个流式
            # 消费过程"——_call_model 现在是 async generator,溢出错误
            # 理论上应该总在生成开始前就报出来(不会在已经吐出一部分
            # token 之后才报),但这是一个待真实模型冒烟验证的假设,
            # 不是已证实的事实,如实记账。
            if state.compacted_this_round or state.overflow_recovery_count >= cfg.budget.max_overflow_recoveries:
                logger.error(f"[run_query_loop] 溢出恢复用尽(同轮已压={state.compacted_this_round},"
                             f"累计已压={state.overflow_recovery_count}/{cfg.budget.max_overflow_recoveries}): {e}")
                yield _overflow_outcome(state, run_ctx)
                return
            logger.warning(f"[run_query_loop] 上下文溢出,触发紧急压缩: {e}")
            outcome = await state.store.maybe_compact(cfg.llm, span.trace_id, trigger="overflow")
            state.compacted_this_round = True
            state.overflow_recovery_count += 1
            if outcome and outcome.user_notice:
                yield {"type": "notice", "content": outcome.user_notice}
            if outcome:
                yield {"type": "context_compacted", "trigger": "overflow",
                       "tokens_before": outcome.result.tokens_before,
                       "tokens_after": outcome.result.tokens_after,
                       "degraded": outcome.degraded}
            continue

        # 【第五刀改动】usage 记账、thinking 事件都已经在 _call_model
        # 内部处理完了(usage 在拿到完整 msg 后立刻记账;thinking 是
        # 随着 delta 边到达边吐出的,不再是这里读一次完整字段再吐一次)。
        # 这里不需要重复做这两件事。

        finish_reason = getattr(msg, "finish_reason", None)

        # ── 截断处理:提到 tool_calls 判断之前(任务 3.7)──────────────
        # 有 tool_calls 且被截断时,原来的代码会把最后一个(必定不完整
        # 的)tool_call 原样送进权限门控/执行器,JSON 解析大概率失败,
        # 模型平白花一轮换回一句报错。现在统一在这里拦截处理。
        if finish_reason == "length":
            if state.output_truncation_count >= cfg.budget.max_output_truncation_recoveries:
                logger.warning(
                    f"[run_query_loop] 输出截断恢复已用尽"
                    f"({state.output_truncation_count}/{cfg.budget.max_output_truncation_recoveries}),"
                    f"终止"
                )
                yield _plain_outcome(
                    "exhausted", msg.content or "", None, state, run_ctx,
                    reason="max_output_tokens_recovery", trace_status="timeout",
                )
                return
            state.output_truncation_count += 1

            if msg.tool_calls:
                all_views = [_tool_call_view(tc) for tc in msg.tool_calls]
                complete_views, dropped_view = all_views[:-1], all_views[-1]
                if complete_views:
                    state.store.append(_build_assistant_message_dict(msg, complete_views))
                    suspended = False
                    async for ev in _process_tool_calls(cfg, state, run_ctx, complete_views):
                        if ev["type"] == "outcome":
                            yield ev
                            suspended = True
                            break
                        yield ev
                    if suspended:
                        return
                    terminal_outcome = _maybe_terminal_completion(state, run_ctx)
                    if terminal_outcome is not None:
                        yield terminal_outcome
                        return
                state.store.append({
                    "role": "user",
                    "content": _truncation_dropped_call_notice(dropped_view.function.name),
                })
            else:
                logger.warning("[run_query_loop] 响应被截断(finish_reason=length),催促续写")
                state.store.append(_to_wire_message(msg))
                state.store.append({"role": "user", "content": _TRUNCATION_NUDGE_TEXT})
            continue

        # ── 决策点:CC 语义,只看有没有 tool_calls ──────────────────────
        if not msg.tool_calls:
            if (cfg.require_terminal_tool is not None
                    and state.terminal_nudge_count < cfg.max_terminal_nudges):
                state.terminal_nudge_count += 1
                state.store.append(_to_wire_message(msg))
                state.store.append({
                    "role": "user",
                    "content": _TERMINAL_NUDGE_PROMPT_TEMPLATE.format(
                        tool=cfg.require_terminal_tool),
                })
                continue

            state.store.append(_to_wire_message(msg))
            # 【第五刀改动】不再逐字符伪造 token 事件——msg.content 在
            # _call_model 消费流式 delta 期间已经实时吐给宿主了,这里
            # 再吐一遍会造成重复(旧版 _stream_text_as_tokens 存在的
            # 前提是"非流式路径下,内容是一次性拿到的",第五刀这个
            # 前提不再成立)。
            final_text = msg.content or ""
            # 【真实冒烟 Part 4 暴露的健壮性缺口】模型偶尔会返回
            # finish_reason="stop" 但 content 为空(实测在"从超长工具
            # 结果里找一个特定事实"这类任务上出现过,同样的代码同样的
            # 任务,上一次跑是正常回答的——是模型的非确定性,不是代码
            # 回归)。原来这种情况报的是 reason="completed",宿主拿到
            # 一个空答案却被告知"正常完成",没有任何信号可以据此判断
            # 要不要重试——而 reason 字段存在的全部意义就是让宿主能
            # 区分不同的收口情况。
            #
            # status 仍然是 completed(模型确实按 CC 语义结束了、没有
            # 报错、也没有耗尽预算,把它归成 exhausted/aborted 都是
            # 撒谎),只在 reason 上如实标注"这次收口时模型什么都没说",
            # 宿主可以据此选择重试、告警,或者干脆忽略。
            #
            # 刻意不做"content 为空就退回用 reasoning 当答案"这种兜底:
            # reasoning 是思考过程不是结论,拿它冒充答案是伪造确定性,
            # 比诚实地报告"空"更糟。
            if cfg.require_terminal_tool is not None:
                reason = "terminal_tool_not_called"
            elif not final_text.strip():
                reason = "empty_response"
            else:
                reason = "completed"
            yield _plain_outcome("completed", final_text, None, state, run_ctx, reason=reason)
            return

        # ── 工具批次:权限门控 + 执行 + 回填 ──────────────────────────────
        state.store.append(_to_wire_message(msg))
        suspended = False
        async for ev in _process_tool_calls(cfg, state, run_ctx, msg.tool_calls):
            if ev["type"] == "outcome":
                yield ev
                suspended = True
                break
            yield ev
        if suspended:
            return

        terminal_outcome = _maybe_terminal_completion(state, run_ctx)
        if terminal_outcome is not None:
            yield terminal_outcome
            return

        outcome_ev = None
        async for ev in _wrap_up_round(cfg, state, run_ctx):
            if ev["type"] == "outcome":
                outcome_ev = ev
                break
            yield ev
        if outcome_ev is not None:
            yield outcome_ev
            return


# ── 调模型(任务 3.8:输出截断静默升档) ───────────────────────────────────
def _delta_to_events(delta: StreamDelta):
    """把一个 StreamDelta 转成对外可见的事件(token/thinking)。
    tool_call_delta/finish_reason/usage 不产生独立事件——这些是
    内部记账信息,通过 StreamAccumulator 攒进最终的 msg,不是给宿主
    看的增量(现有事件词表只有 token/thinking 两种,tool_start/
    tool_end 要等完整 tool_calls 确定之后才在 _process_tool_calls
    里产生,这个分工第五刀不改变)。

    【第五刀的一处行为变更,如实记账】round1-4 用一个启发式区分
    "thinking" 和普通输出:`msg.reasoning or (msg.content if msg.
    tool_calls else None)`——也就是说,如果模型在调用工具的同时
    附带了一段 content(常见的"我先查一下"这类前缀文本),会被当成
    "thinking"处理,而不是当成"token"实时展示。这个启发式在流式场景
    下无法成立:content 增量到达的时刻,我们还不知道这次响应最终
    会不会带 tool_calls(tool_calls 本身也是增量到达、往往和 content
    交错),没有办法在增量层面做这个事后才能确定的判断。

    第五刀改用更直接、也更符合各字段本来含义的规则:reasoning 字段
    永远当 thinking,content 字段永远当 token,不再看这次响应最后
    有没有 tool_calls。content 本来就是模型的输出文本,不是内部
    推理,当年把它并入 thinking 是一个针对"部分 provider 没有独立
    reasoning 通道"的权宜启发式,现在借流式改造的机会一并理顺,不是
    顺带夹带的无关改动。影响面:只有"响应同时带 content 和
    tool_calls"这一种场景的展示方式变了(以前是一次性 thinking 事件,
    现在是实时 token 事件),不影响任何终止判定/预算/压缩逻辑。
    """
    if delta.reasoning:
        yield {"type": "thinking", "content": delta.reasoning}
    if delta.content:
        yield {"type": "token", "content": delta.content}


async def _call_model(
    cfg: LoopConfig, state: LoopState, run_ctx: RunContext, tools: list | None,
) -> AsyncGenerator[dict, None]:
    """第五刀:调用模型改走流式接口 cfg.llm.stream(...),边收边吐
    token/thinking 事件,最后 yield 一个 {"type":"_message",
    "message":msg} 信号交出归一化后的完整消息。调用方按事件类型
    区分:看到 "_message" 意味着这次调用结束,可以继续走终止判定;
    其他类型(token/thinking)照常转发给宿主。

    tools=None 时不把 "tools" 键放进发给 llm.stream() 的 kwargs
    (不是传 tools=None)——force_answer 场景要的是"完全不提供工具"
    的语义,省略这个键和显式传 None 在部分 provider SDK 下行为可能
    不同,省略是更保守、更明确的选择。

    静默升档(任务 3.8)与流式的取舍(见设计复盘"发现一"):如果配置
    了 max_output_tokens_upgraded 且这个 run 还没升档过,第一次调用
    的 delta **不会**实时转发——先在本地缓冲,等确认这次没有被截断
    才补发;如果被截断,直接丢弃缓冲(用户从未看到这次尝试),改用
    升档后的预算重试一次,这次的 delta 才是真正实时转发的。这是
    "用户对升档完全无感"这个承诺在流式场景下唯一自洽的实现方式——
    代价是配置了升档的用户,第一次调用会失去流式的即时性,这是需要
    显式接受的权衡,不是免费的。不配置升档(两个字段都是 None,
    默认)完全不受影响,从第一个 delta 开始就是真正实时转发,不缓冲
    任何东西。

    usage 记账(state.store.note_api_usage)挪到这里统一做(原来在
    run_query_loop 里),好处是 _exhaust 的 force_answer/force_finish
    复用这个函数时,也顺带获得了原来没有的 usage 记账——这是第五刀
    改走共享路径带来的连带修复,不是专门为它加的特殊逻辑。

    【中断检查点②,任务 4.7 遗留项在这里补上】流式消费期间发现
    run_ctx.abort 已被设置,直接产出一个 status="aborted" 的 outcome
    事件并结束——这个时机不需要孤儿 tool_use 修复:调用方
    (run_query_loop)要等看到这里 yield 出的 "_message" 事件之后
    才会把 assistant 消息追加进 state.store,中断发生在那之前,
    消息历史里还没有任何东西需要修复,和检查点①(每轮开头)是同一类
    "干净的中断时机"。产出的事件类型是完整的 {"type":"outcome",...}
    (不是内部的 "_message" 信号),调用方(run_query_loop/_exhaust)
    看到 "outcome" 类型要原样转发并立即返回,不能继续假设后面还有
    "_message" 事件会到来。
    """
    span = run_ctx.span
    may_need_upgrade = (not state.output_upgraded
                        and cfg.budget.max_output_tokens_upgraded is not None)
    max_tokens = (cfg.budget.max_output_tokens_upgraded if state.output_upgraded
                 else cfg.budget.max_output_tokens)

    def _kwargs(mt: int | None) -> dict:
        kw = {"trace_id": span.trace_id, "messages": state.store.messages}
        if tools:
            kw["tools"] = tools
        if mt is not None:
            kw["max_tokens"] = mt
        return kw

    def _aborted() -> bool:
        return run_ctx.abort is not None and run_ctx.abort.is_set()

    acc = StreamAccumulator()

    if may_need_upgrade:
        buffered: list[dict] = []
        async for delta in cfg.llm.stream(**_kwargs(max_tokens)):
            if _aborted():
                yield _aborted_outcome(state, run_ctx, reason="aborted_streaming")
                return
            acc.feed(delta)
            buffered.extend(_delta_to_events(delta))
        msg = acc.build_message()

        if getattr(msg, "finish_reason", None) == "length":
            logger.info("[run_query_loop] 输出被截断,静默升档重试一次(用户无感)")
            state.output_upgraded = True
            acc = StreamAccumulator()
            async for delta in cfg.llm.stream(**_kwargs(cfg.budget.max_output_tokens_upgraded)):
                if _aborted():
                    yield _aborted_outcome(state, run_ctx, reason="aborted_streaming")
                    return
                acc.feed(delta)
                for ev in _delta_to_events(delta):
                    yield ev
            msg = acc.build_message()
        else:
            for ev in buffered:
                yield ev
    else:
        async for delta in cfg.llm.stream(**_kwargs(max_tokens)):
            if _aborted():
                yield _aborted_outcome(state, run_ctx, reason="aborted_streaming")
                return
            acc.feed(delta)
            for ev in _delta_to_events(delta):
                yield ev
        msg = acc.build_message()

    if msg.usage is not None:
        state.store.note_api_usage(msg.usage.prompt_tokens)

    yield {"type": "_message", "message": msg}


# ── 恢复入口的"后半段"处理 ────────────────────────────────────────────────
async def _finish_interrupted_round(
    cfg: LoopConfig, state: LoopState, run_ctx: RunContext,
) -> AsyncGenerator[dict, None]:
    resume_point = state.resume_point
    messages = state.store.messages
    assistant_msg, all_views = _find_pending_call_context(messages, resume_point.tool_call_id)
    if assistant_msg is None:
        raise ValueError(
            f"resume 失败:消息列表里找不到 pending_tool_call_id="
            f"{resume_point.tool_call_id} 对应的 assistant 消息(快照数据"
            f"与传入的挂起标识不一致,拒绝在不确定的假设下继续)"
        )
    pending_idx = next(i for i, v in enumerate(all_views) if v.id == resume_point.tool_call_id)
    remaining = all_views[pending_idx:]

    async for ev in _process_tool_calls(
        cfg, state, run_ctx, remaining,
        prefetched=(resume_point.tool_call_id, resume_point.decision),
    ):
        yield ev
        if ev["type"] == "outcome":
            return

    terminal_outcome = _maybe_terminal_completion(state, run_ctx)
    if terminal_outcome is not None:
        yield terminal_outcome
        return

    async for ev in _wrap_up_round(cfg, state, run_ctx):
        yield ev
        if ev["type"] == "outcome":
            return


# ── 处理一批工具调用 ────────────────────────────────────────────────────
async def _process_tool_calls(
    cfg: LoopConfig, state: LoopState, run_ctx: RunContext,
    tool_calls: list, *, prefetched: tuple[str, Allow | Deny] | None = None,
) -> AsyncGenerator[dict, None]:
    span = run_ctx.span
    for tc in tool_calls:
        # ── 中断检查点②:每个工具调用之间(任务 4.7),包括处理第一个
        # 之前——调用方(run_query_loop)在调这个函数之前已经把整条
        # assistant 消息(含全部 N 个 tool_calls)追加进了 state.store,
        # 所以哪怕一个都还没处理,消息历史里也已经有 N 个孤儿
        # tool_call 了,修复必须在这里无条件跑一遍,不能只在"处理到
        # 一半"时才补(任务 4.2/4.3 的落点:中断路径退出前修复)。
        if run_ctx.abort is not None and run_ctx.abort.is_set():
            from harness.agent.repair import repair_orphan_tool_calls
            repair_orphan_tool_calls(state.store.messages, reason="用户中断")
            yield _aborted_outcome(state, run_ctx, reason="aborted_tools")
            return

        view = _tool_call_view(tc)

        if prefetched is not None and view.id == prefetched[0]:
            decision = prefetched[1]
        else:
            decision = await cfg.permission_policy.check(
                view.function.name, _parse_tool_args(view), run_ctx,
            )

        if isinstance(decision, Defer):
            yield {"type": "approval_pending", "tool": view.function.name,
                   "approval_id": decision.approval_id, "tool_call_id": view.id}
            yield _awaiting_approval_outcome(state, run_ctx, decision.approval_id, view.id)
            return

        if isinstance(decision, Deny):
            state.tool_calls_used += 1
            yield {"type": "tool_denied", "tool": view.function.name,
                   "reason": decision.reason}
            state.store.append({
                "role": "tool", "tool_call_id": view.id,
                "content": f"[权限拒绝] {decision.reason}",
            })
            continue

        state.tool_calls_used += 1
        yield {"type": "tool_start", "tool": view.function.name,
               "args": view.function.arguments}
        result = await cfg.tool_executor.execute(
            view, trace_id=span.trace_id, run_ctx=run_ctx,
        )
        yield {"type": "tool_end", "tool": view.function.name, "result": result}
        if cfg.tool_executor.is_exempt_from_offload(view.function.name):
            text = str(result)
        else:
            text = state.store.offload_tool_result(
                span.trace_id, view.id, str(result),
                max_chars=cfg.tool_executor.max_result_chars_for(view.function.name),
            )
        state.store.append({"role": "tool", "tool_call_id": view.id, "content": text})


def _maybe_terminal_completion(state: LoopState, run_ctx: RunContext) -> dict | None:
    result = run_ctx.state.get("result")
    if result is None:
        return None
    span = run_ctx.span
    span.end(result.summary, status="success")
    return {"type": "outcome", "outcome": LoopOutcome(
        status="completed", reason="completed",
        final_text=result.summary, result=result,
        rounds=state.rounds, tool_calls_used=state.tool_calls_used,
        messages=state.store.messages,
        **_counters(state),
    )}


async def _wrap_up_round(
    cfg: LoopConfig, state: LoopState, run_ctx: RunContext,
) -> AsyncGenerator[dict, None]:
    span = run_ctx.span

    if state.tool_calls_used >= cfg.budget.max_tool_calls:
        async for ev in _exhaust(cfg, state, run_ctx, reason="max_tool_calls"):
            yield ev
        return

    outcome = await state.store.maybe_compact(cfg.llm, span.trace_id, trigger="threshold")
    if outcome:
        if outcome.user_notice:
            yield {"type": "notice", "content": outcome.user_notice}
        yield {"type": "context_compacted", "trigger": "threshold",
               "tokens_before": outcome.result.tokens_before,
               "tokens_after": outcome.result.tokens_after,
               "degraded": outcome.degraded}


# ── 预算耗尽收口(任务 3.3:reason 由调用方指明是撞了哪个预算) ─────────────
async def _exhaust(
    cfg: LoopConfig, state: LoopState, run_ctx: RunContext, reason: str,
) -> AsyncGenerator[dict, None]:
    span = run_ctx.span
    action = cfg.budget.exhausted_action
    logger.warning(f"[run_query_loop] budget exhausted (action={action}, reason={reason}, "
                   f"rounds={state.rounds}, tool_calls={state.tool_calls_used})")

    if action == "stop":
        span.end("", status="timeout")
        yield {"type": "outcome", "outcome": LoopOutcome(
            status="exhausted", reason=reason,
            rounds=state.rounds, tool_calls_used=state.tool_calls_used,
            messages=state.store.messages, **_counters(state),
        )}
        return

    state.store.append({"role": "user", "content": cfg.budget.exhausted_prompt})

    if action == "force_answer":
        # 【第五刀改动】不再手写一份基于旧版 .call(stream=True) 裸
        # chunk 格式的消费逻辑——统一走 _call_model,和主循环共用
        # 同一套流式路径/归一化/静默升档逻辑。tools=None:force_answer
        # 的原意就是"不给工具、只要一段文本",这里不传 cfg.tools。
        # 这些中间事件(token/thinking)照常转发给宿主——force_answer
        # 本来就是要把这段"强制作答"的文本流式展示出来,这是它的
        # 原有行为,不是第五刀新引入的。
        msg = None
        async for ev in _call_model(cfg, state, run_ctx, tools=None):
            if ev["type"] == "_message":
                msg = ev["message"]
            elif ev["type"] == "outcome":
                yield ev   # 检查点②命中,直接透传中断结果
                return
            else:
                yield ev
        final_text = msg.content or ""
        if final_text:
            state.store.append({"role": "assistant", "content": final_text})
        span.end(final_text, status="timeout")
        yield {"type": "outcome", "outcome": LoopOutcome(
            status="exhausted", reason=reason, final_text=final_text,
            rounds=state.rounds, tool_calls_used=state.tool_calls_used,
            messages=state.store.messages, **_counters(state),
        )}
        return

    # action == "force_finish"(或任何其他值,统一走这条兜底)
    # 【第五刀改动】同样改走 _call_model,但这里的中间事件(token/
    # thinking)**不**转发给宿主——延续第一刀就定下的选择("exhaust
    # 场景下不吐给外面...保持事件面简洁"),force_finish 是收尾路径
    # 不是正常轮次,这条界限第五刀不改变,只是换了内部实现方式。
    # 注意:检查点②命中产出的是完整的 "outcome" 事件,不是内部的
    # "_message" 信号,不能被下面这段"忽略非 _message 事件"的逻辑
    # 误吞——那样会导致中断被吞掉、msg 保持 None、后面 msg.content
    # 直接 AttributeError。
    msg = None
    async for ev in _call_model(cfg, state, run_ctx, tools=cfg.tools):
        if ev["type"] == "_message":
            msg = ev["message"]
        elif ev["type"] == "outcome":
            yield ev   # 检查点②命中,直接透传中断结果
            return
        # 其余中间事件(token/thinking)有意吞掉,不转发,不是遗漏
    result = None
    final_text = msg.content or ""
    if msg.tool_calls:
        state.store.append(_to_wire_message(msg))
        async for ev in _process_tool_calls(cfg, state, run_ctx, msg.tool_calls):
            if ev["type"] == "outcome":
                yield ev
                return
        result = run_ctx.state.get("result")
        if result is not None:
            final_text = result.summary
    else:
        state.store.append(_to_wire_message(msg))

    span.end(final_text, status="timeout")
    yield {"type": "outcome", "outcome": LoopOutcome(
        status="exhausted", reason=reason, final_text=final_text, result=result,
        rounds=state.rounds, tool_calls_used=state.tool_calls_used,
        messages=state.store.messages, **_counters(state),
    )}


# ── 内部工具 ────────────────────────────────────────────────────────────
def _plain_outcome(status, final_text, result, state: LoopState, run_ctx: RunContext,
                   reason: str, trace_status: str = "success") -> dict:
    run_ctx.span.end(final_text, status=trace_status)
    return {"type": "outcome", "outcome": LoopOutcome(
        status=status, reason=reason, final_text=final_text, result=result,
        rounds=state.rounds, tool_calls_used=state.tool_calls_used,
        messages=state.store.messages, **_counters(state),
    )}


def _overflow_outcome(state: LoopState, run_ctx: RunContext) -> dict:
    run_ctx.span.end("", status="overflow")
    return {"type": "outcome", "outcome": LoopOutcome(
        status="overflow", reason="context_overflow",
        rounds=state.rounds, tool_calls_used=state.tool_calls_used,
        messages=state.store.messages, **_counters(state),
    )}


def _aborted_outcome(state: LoopState, run_ctx: RunContext, reason: str) -> dict:
    """任务 4.8。reason 由调用方指明是在哪个检查点被中断的
    (aborted_streaming / aborted_tools),status 统一是 "aborted"——
    这个两级设计和 exhausted/overflow 是同一套原则(第三刀 §4.3):
    status 给宿主做决策,reason 给诊断/上报用。"""
    run_ctx.span.end("", status="aborted")
    return {"type": "outcome", "outcome": LoopOutcome(
        status="aborted", reason=reason,
        rounds=state.rounds, tool_calls_used=state.tool_calls_used,
        messages=state.store.messages, **_counters(state),
    )}


def _awaiting_approval_outcome(
    state: LoopState, run_ctx: RunContext, approval_id: str, tool_call_id: str,
) -> dict:
    run_ctx.span.end("", status="awaiting_approval")
    return {"type": "outcome", "outcome": LoopOutcome(
        status="awaiting_approval", reason="awaiting_approval",
        rounds=state.rounds, tool_calls_used=state.tool_calls_used,
        messages=state.store.messages,
        pending_approval_id=approval_id, pending_tool_call_id=tool_call_id,
        **_counters(state),
    )}


def _parse_tool_args(view) -> dict:
    import json
    try:
        return json.loads(view.function.arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}