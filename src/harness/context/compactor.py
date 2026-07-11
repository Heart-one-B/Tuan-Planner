# harness/context/compactor.py
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from harness.context.compaction_prompt import (
    DEFAULT_COMPACTION_PROMPT,
    build_compaction_messages,
    extract_summary,
    render_messages_for_compaction,
)
from harness.context.models import CompactionResult
from harness.context.offload import CLEARED_MARKER, OFFLOAD_MARKER, OffloadStore, RETRIEVAL_TOOL_NAME
from harness.context.token_counter import estimate_messages_tokens
from harness.llm.base import LLMClientBase

logger = logging.getLogger(__name__)


@dataclass
class CompactionOutcome:
    """一次压缩尝试的完整产出:新的消息列表 + 审计记录。
    degraded=True 表示 LLM 压缩失败、走了规则降级。
    user_notice 非 None 表示发生了必须让用户知情的信息损失——
    AgentLoop 以 notice 事件吐给上层,由 UI 决定展示;harness 只负责
    如实产生这条通知。"""
    new_messages: list
    result: CompactionResult
    degraded: bool = False
    user_notice: str | None = None


class Compactor:
    """带熔断器的 LLM 全量压缩。

    压缩边界(三条硬规则):
      1. 头部前缀永不动(prefix_len 由调用方声明;启发式仅兜底)
      2. 尾部最近 keep_recent_rounds 个工具调用轮次原文保留
      3. 切口不拆散工具调用配对

    可恢复压缩(对照修正):被摘要替换/被降级丢弃的中间段,替换前先
    渲染落盘 OffloadStore,摘要/占位消息携带 ref——压缩不再是销毁,
    是换页,模型可按需取回原文("任何不可逆压缩都有风险",Manus 原则)。

    摘要输入预瘦身(对照修正):中间段送进摘要器之前,先对渲染副本做
    规则瘦身(清大工具结果)。否则历史越大,压缩请求本身越必然超模型
    窗口,LLM 压缩系统性失败、大会话只能走降级——这是个隐藏的结构性
    失败模式。原始消息不动(反正将被摘要替换),只瘦渲染副本。

    熔断器:max_attempts 次内不成功即停止重试,走分档降级:
      threshold 触发(还有余量)→ 一档:剥思考+清工具结果+截超长文本
      overflow 触发(已撞墙)→ 先跑一档,重新估量,放得下就用一档结果;
        仍不够才二档:中间段整体换页丢弃 + 双向告知(占位告知模型防
        幻觉连续,user_notice 告知用户建议另起会话)
    """

    def __init__(
        self,
        llm_client: LLMClientBase | None = None,
        compaction_prompt: str = DEFAULT_COMPACTION_PROMPT,
        max_attempts: int = 2,
    ):
        self._own_client = llm_client
        self.compaction_prompt = compaction_prompt
        self.max_attempts = max_attempts

    # ── 主入口 ──────────────────────────────────────────────────────────

    async def compact(
        self,
        messages: list,
        fallback_client: LLMClientBase,
        trace_id: str,
        trigger: str = "threshold",
        focus: str | None = None,
        keep_recent_rounds: int = 2,
        prefix_len: int | None = None,
        offload_store: OffloadStore | None = None,
        fit_within: int | None = None,
    ) -> CompactionOutcome:
        """offload_store:提供时启用可恢复压缩(换页而非销毁)。
        fit_within:overflow 降级时的目标预算(估算值),一档瘦身后若已
        放得下则不必走二档丢弃;None 表示无法判断,overflow 直接二档。"""
        client = self._own_client or fallback_client
        tokens_before = estimate_messages_tokens(messages)

        prefix, middle, tail = self._split(messages, keep_recent_rounds, prefix_len)
        if not middle:
            logger.info("[Compactor] 无可压缩的中间段,跳过")
            return CompactionOutcome(
                new_messages=messages,
                result=CompactionResult(
                    trigger=trigger, tokens_before=tokens_before,
                    tokens_after=tokens_before, summary="",
                    dropped_message_count=0, kept_tail_count=len(tail),
                    focus=focus,
                ),
            )

        # ── 可恢复压缩:中间段原文先换页落盘 ──
        middle_ref: str | None = None
        if offload_store is not None:
            full_text = render_messages_for_compaction(middle)
            record = offload_store.save(
                trace_id, f"compacted_middle_{uuid.uuid4().hex[:8]}", full_text,
            )
            middle_ref = record.ref

        # ── LLM 压缩,带熔断;输入是预瘦身后的渲染副本 ──
        render_text = render_messages_for_compaction(self._shrink_for_render(middle))
        summary: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                summary = await self._summarize(client, render_text, trace_id,
                                                focus, allow_questions=(trigger == "manual"))
                break
            except Exception as e:
                logger.error(f"[Compactor] 压缩第 {attempt}/{self.max_attempts} 次失败: {e}")

        user_notice: str | None = None
        if summary is not None:
            ref_line = (
                f"\n(被压缩部分的原文已换页保存,引用: {middle_ref},"
                f"如需查阅某个具体细节可调用 {RETRIEVAL_TOOL_NAME} 读取)"
                if middle_ref else ""
            )
            summary_msg = {
                "role": "user",
                "content": (
                    "[本对话较早的部分因上下文接近上限已被压缩为以下摘要。"
                    "你正在继续先前的任务,不是从头开始,"
                    "请基于摘要接着当前进度往下做。]"
                    f"{ref_line}\n\n{summary}"
                ),
            }
            new_messages = prefix + [summary_msg] + tail
            outcome_degraded = False
        else:
            # ── 熔断:分档降级 ──
            shrunk = self._rule_based_shrink(middle)
            if trigger != "overflow":
                # threshold:还有余量,一档到此为止,不够就如实不够
                logger.warning("[Compactor] 熔断:LLM 压缩失败,第一档降级(规则清理)")
                new_messages = prefix + shrunk + tail
            else:
                # overflow:先看一档瘦身后是否已放得下(便宜优先也适用于紧急路径)
                candidate = prefix + shrunk + tail
                if fit_within is not None and estimate_messages_tokens(candidate) <= fit_within:
                    logger.warning("[Compactor] 熔断+溢出:一档瘦身后已可容纳,免于丢弃")
                    new_messages = candidate
                else:
                    logger.warning(f"[Compactor] 熔断+溢出:第二档降级,换页丢弃中间段 {len(middle)} 条")
                    ref_hint = (
                        f"原文已换页保存(引用: {middle_ref}),如其中可能包含你仍"
                        f"需要的关键信息(用户明确的偏好、约束、未记录的决策),"
                        f"可调用 {RETRIEVAL_TOOL_NAME} 分页取回。"
                        if middle_ref else "且不可恢复。"
                    )
                    placeholder = {
                        "role": "user",
                        "content": (
                            f"[系统提示] 上下文已满且压缩失败,此处 {len(middle)} 条"
                            f"历史消息已移出上下文。{ref_hint}"
                            f"请基于剩余信息尽力继续,无法确定的内容如实说明,"
                            f"不要臆测被移除的部分。"
                        ),
                    }
                    new_messages = prefix + [placeholder] + tail
                    recover_note = "(原文已换页保存,模型可按需取回部分内容)" if middle_ref \
                                   else "(不可恢复)"
                    user_notice = (
                        f"上下文窗口已满且自动压缩失败,系统已将中间 {len(middle)} 条"
                        f"对话历史移出上下文{recover_note}以维持运行。"
                        f"当前回答可能因信息缺失而不完整,"
                        f"建议保存现有结果后另起新会话继续。"
                    )
            summary = ""
            outcome_degraded = True

        tokens_after = estimate_messages_tokens(new_messages)
        return CompactionOutcome(
            new_messages=new_messages,
            result=CompactionResult(
                trigger=trigger, tokens_before=tokens_before,
                tokens_after=tokens_after, summary=summary,
                dropped_message_count=len(middle),
                kept_tail_count=len(tail), focus=focus,
                middle_ref=middle_ref,
            ),
            degraded=outcome_degraded,
            user_notice=user_notice,
        )

    # ── 边界切分 ────────────────────────────────────────────────────────

    @staticmethod
    def _split(messages: list, keep_recent_rounds: int,
              prefix_len: int | None = None) -> tuple[list, list, list]:
        if prefix_len is not None:
            i = max(0, min(prefix_len, len(messages)))
        else:
            i = 0
            while i < len(messages) and isinstance(messages[i], dict) \
                    and messages[i].get("role") == "system":
                i += 1
            if i < len(messages) and isinstance(messages[i], dict) \
                    and messages[i].get("role") == "user":
                i += 1
        prefix = list(messages[:i])
        rest = list(messages[i:])

        round_starts = [
            j for j, m in enumerate(rest)
            if (getattr(m, "tool_calls", None)
                or (isinstance(m, dict) and m.get("tool_calls")))
        ]
        if len(round_starts) <= keep_recent_rounds:
            return prefix, [], rest
        cut = round_starts[-keep_recent_rounds]
        return prefix, rest[:cut], rest[cut:]

    # ── 摘要输入预瘦身 ──────────────────────────────────────────────────

    _RENDER_TOOL_RESULT_LIMIT = 2_000   # 渲染副本里单条工具结果的保留上限

    @classmethod
    def _shrink_for_render(cls, middle: list) -> list:
        """给摘要器看的渲染副本瘦身:大工具结果截头尾。只动副本,原始
        消息不动。用户消息和 assistant 结论性文本不截——契约要求
        用户消息全录,截了枚举就不完整了。"""
        shrunk = []
        for m in middle:
            if isinstance(m, dict) and m.get("role") == "tool":
                content = m.get("content", "")
                if isinstance(content, str) and len(content) > cls._RENDER_TOOL_RESULT_LIMIT:
                    half = cls._RENDER_TOOL_RESULT_LIMIT // 2
                    m = {**m, "content": (
                        f"{content[:half]}\n...[摘要输入预瘦身:原文 {len(content)} 字符]...\n"
                        f"{content[-half:]}"
                    )}
            shrunk.append(m)
        return shrunk

    # ── LLM 摘要 ────────────────────────────────────────────────────────

    async def _summarize(self, client: LLMClientBase, render_text: str,
                        trace_id: str, focus: str | None,
                        allow_questions: bool) -> str:
        msgs = build_compaction_messages(render_text, self.compaction_prompt,
                                         focus, allow_questions=allow_questions)
        resp = await client.call(trace_id=trace_id, messages=msgs)
        raw = resp.choices[0].message.content
        if not raw or not raw.strip():
            raise ValueError("压缩器返回了空摘要")
        summary = extract_summary(raw)
        if not summary:
            raise ValueError("压缩器输出无法提取出摘要正文")
        return summary

    # ── 降级路径(第一档) ────────────────────────────────────────────────

    _LONG_TEXT_TRUNCATE_CHARS = 500

    @classmethod
    def _rule_based_shrink(cls, middle: list) -> list:
        """熔断后的第一档降级,三件套,全部无语义判断、必然成功:
          ① tool 消息:清空原文换占位
          ② 剥思考:带 tool_calls 的 assistant 消息的 content(行动意图
             陈述,行动已由工具调用+结果承载)清空;reasoning 属性置空。
             注意区分:不带 tool_calls 的纯 assistant 文本是结论不是
             思考,不清,只截超长(③)——"思考是最可丢的内容类别"是
             各 reasoning 模型 provider 的共同做法(跨轮不携带思考),
             但结论不在此列。
          ③ 非 tool 的超长纯文本:头尾截断
        """
        shrunk = []
        for m in middle:
            if isinstance(m, dict) and m.get("role") == "tool":
                content = m.get("content", "")
                if not (content.startswith(CLEARED_MARKER)
                        or content.startswith(OFFLOAD_MARKER)):
                    m = {**m, "content": f"{CLEARED_MARKER}(压缩降级) "
                                         f"tool_call_id={m.get('tool_call_id')}"}
            elif not isinstance(m, dict) and getattr(m, "tool_calls", None):
                # 剥思考:SDK 消息对象,带工具调用
                if getattr(m, "reasoning", None):
                    try:
                        m.reasoning = None
                    except Exception:
                        pass
                if getattr(m, "content", None):
                    try:
                        m.content = None
                    except Exception:
                        pass
            elif isinstance(m, dict):
                content = m.get("content")
                if isinstance(content, str) and len(content) > cls._LONG_TEXT_TRUNCATE_CHARS:
                    half = cls._LONG_TEXT_TRUNCATE_CHARS // 2
                    m = {**m, "content": (
                        f"{content[:half]}\n...[压缩降级:原文 {len(content)} 字符被截断]...\n"
                        f"{content[-half:]}"
                    )}
            shrunk.append(m)
        return shrunk