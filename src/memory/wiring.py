# src/memory/wiring.py
"""应用侧的记忆接线：装配、召回、提取。

【为什么是模块级单例而不是放进 state】
MemoryConfig 持有 Path、LLM 客户端、回调函数，不可序列化——塞进
LangGraph 的 AgentState 在配了 checkpointer 时会在落盘那一步炸掉
（和 root_trace_id 那次相反的坑：那次是"没声明所以被丢弃"，这次是
"塞了不该塞的东西"）。模块级单例 + configure_memory() 与 tracing 的
configure_storage() 同构，是这个仓库已有的模式。

单例还顺带解决了对照组的开关问题：实验脚本调
configure_memory(None) 就得到一个完全没有记忆的运行，
其余代码一行不改——单变量对比要求的正是这个。

【三个默认值必须显式覆盖，否则实验会静默产出空结果】
  recall_top_k            默认 None = 召回完全关闭
  extraction_every_n_runs 默认 None = 提取完全关闭
  recall_min_memories     默认 8 —— 记忆总数不足 8 条时直接返回，
                          不做任何召回。实验里只会有 2-3 条记忆，
                          不改这个值，召回永远不触发，而表现是
                          "主组和对照组一样"，看起来像"记忆无效"。
这三个是本文件存在的最大理由：把"必须改"写在代码里，
而不是留在某份文档里等人记得。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from harness.memory import (
    ExtractionBookkeeping,
    FileMemoryStore,
    MemoryConfig,
    SurfacedMemory,
    maybe_extract,
    maybe_recall,
)
from harness.tracing.span import Span

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_DIR = Path("data/memory")

_config: MemoryConfig | None = None
_store: FileMemoryStore | None = None


# ── 装配 ──────────────────────────────────────────────────────────────

def build_memory_config(
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    recall_top_k: int = 3,
    recall_min_memories: int = 1,
    extraction_every_n_runs: int = 1,
    staleness_days: int = 7,
) -> MemoryConfig:
    """构造应用用的 MemoryConfig。

    recall_min_memories=1：harness 的默认值 8 是为"记忆库已经积累到
    一定规模"的稳态场景定的——那时常驻索引本身已经够长够醒目，
    为几条候选专门烧一次选择器调用不划算。实验/早期使用阶段记忆
    只有 2-3 条，8 这个门槛等于把召回整个关掉。

    staleness_days=7：默认 2 天在行程助手场景下太短——"不吃辣"
    这类 feedback 类记忆的老化速度以月计，两天就挂过期警告会让
    模型对完全有效的偏好产生不必要的怀疑。
    """
    return MemoryConfig(
        memory_dir=Path(memory_dir),
        recall_top_k=recall_top_k,
        recall_min_memories=recall_min_memories,
        extraction_every_n_runs=extraction_every_n_runs,
        extraction_mode="manual",   # 由调用方在会话结束时显式触发，
                                    # 不让 harness 替宿主决定并发模型
        staleness_days=staleness_days,
        on_memory_event=_log_event,
    )


def _log_event(event: dict) -> None:
    """记忆子系统的事件出口。当前只记日志。

    如实记账：这些事件不进可查询存储，所以"100 次 run 里召回命中率
    多少"这类问题现在只能翻日志文本——这正是 Phase 5 里 B 类参数
    （recall_min_memories / recall_top_k / extraction_every_n_runs）
    无法校准的直接原因。补这个仪表是零 token 的工作，但不在当前
    时间盒内。
    """
    logger.info(f"[memory] {event.get('type')} session={event.get('session_id')} "
                f"count={event.get('count')} error={event.get('error')}")


def configure_memory(config: MemoryConfig | None) -> None:
    """设置（或关闭）记忆。传 None = 对照组：完全没有记忆的运行。"""
    global _config, _store
    _config = config
    _store = FileMemoryStore(config.memory_dir) if config else None
    logger.info(f"[memory] configured: {'ON' if config else 'OFF'}")


def memory_enabled() -> bool:
    return _config is not None


def current_config() -> MemoryConfig | None:
    return _config


# ── 召回（每轮 intent 之前） ───────────────────────────────────────────

@dataclass
class RecallResult:
    """preference_context 为空字符串表示"这轮没有可注入的东西"——
    调用方不需要区分是关闭、门槛未达、熔断、还是选择器判定无相关项，
    统一按"没有"处理即可（原因已经通过 on_memory_event 上报）。"""
    preference_context: str = ""
    surfaced_names: list[str] = None

    def __post_init__(self):
        if self.surfaced_names is None:
            self.surfaced_names = []


async def recall_preference_context(
    session_id: str,
    task: str,
    trace_id: str,
    already_surfaced_names: list[str],
    llm_client,
) -> RecallResult:
    """本轮该注入哪些记忆。

    【already_surfaced 传 hid=None 是刻意的】
    harness 的 maybe_recall 用 hid 判断"承载这条记忆的注入消息是否
    还在上下文里"——hid 在 history 里找得到就继续排除，找不到
    （被压缩吃掉了）就重新变成候选。这套机制的前提是调用方维护着
    一份连续的消息历史。

    我们的 graph 不维护消息历史（每个节点各自组装 messages），
    没有 hid 可以查活。传 hid=None 会走 harness 的 legacy 分支：
    保守地视作"仍然在场、不允许重推"。对本场景这恰好是正确语义
    ——一次会话内同一条记忆注入一次就够。

    如实记账：这意味着"压缩吃掉召回消息后自动重新召回"这个能力在
    本应用里不生效。它不生效不是因为坏了，是因为本应用没有它所
    依赖的那种消息历史。
    """
    if _config is None or _store is None:
        return RecallResult()

    outcome = await maybe_recall(
        session_id=session_id,
        task=task,
        trace_id=trace_id,
        history=[],
        already_surfaced=[SurfacedMemory(name=n, hid=None)
                          for n in (already_surfaced_names or [])],
        memory_store=_store,
        memory_config=_config,
        fallback_llm_client=llm_client,
    )

    if outcome.injection_message is None:
        return RecallResult(surfaced_names=list(already_surfaced_names or []))

    content = outcome.injection_message.get("content") or ""
    new_names = [m.name for m in outcome.newly_surfaced]
    merged = list(dict.fromkeys(list(already_surfaced_names or []) + new_names))
    logger.info(f"[memory] 召回 {len(new_names)} 条: {new_names}")
    return RecallResult(preference_context=content, surfaced_names=merged)


# ── 提取（会话结束） ──────────────────────────────────────────────────

async def extract_session(
    session_id: str,
    dialogue: list[dict],
    llm_client,
    parent_span: Span | None = None,
) -> int:
    """会话结束时扫一遍对话，把值得跨会话记住的写进记忆库。

    dialogue 是 [{"role": "user"|"assistant", "content": str}, ...]，
    由调用方（拥有多轮循环的那一层）累积——graph 本身每轮独立
    invoke，节点看不到跨轮的完整对话。

    extraction_mode="manual" 时 harness 不会自动触发，必须显式调
    这个函数。是同步 await 还是 asyncio.create_task 包成后台任务，
    由调用方决定——harness 不替宿主选并发模型，这里同样不替。

    返回本次写入的记忆条数（供实验脚本断言"提取确实发生了"）。
    失败返回 0 而不是抛异常：提取是增益不是主流程，它坏了不该让
    一次本来成功的会话变成失败。
    """
    if _config is None or _store is None:
        return 0
    if not dialogue:
        logger.warning(f"[memory] session={session_id} 对话为空，跳过提取")
        return 0

    before = len(_store.list_all())
    try:
        await maybe_extract(
            session_id=session_id,
            all_messages=list(dialogue),
            bookkeeping=ExtractionBookkeeping(),
            memory_store=_store,
            memory_config=_config,
            fallback_llm_client=llm_client,
            parent_span=parent_span,
        )
    except Exception as e:
        logger.error(f"[memory] 提取异常 session={session_id}: {e}", exc_info=True)
        return 0

    after = len(_store.list_all())
    # 用"记忆库前后条数差"而不是数 memory_write 调用次数：同名写入
    # 是覆盖更新，数调用次数会把"更新一条已有记忆"也算成新增。
    # 这个口径偏保守（更新不计入），但不会虚报。
    written = max(0, after - before)
    logger.info(f"[memory] session={session_id} 提取完成，记忆库 {before} → {after}")
    return written


def list_memories() -> list:
    """当前记忆库全量（实验脚本用来检查 Session 1 到底记下了什么）。"""
    return _store.list_all() if _store else []