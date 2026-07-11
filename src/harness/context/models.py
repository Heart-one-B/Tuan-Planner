# harness/context/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class OffloadRecord:
    """一次工具结果卸载的档案:原始内容在磁盘上的引用。

    ref 是相对于 offload_dir 的相对路径(不含 base 目录),形如
    "{trace_id}/{tool_call_id}.txt"——用相对引用是刻意的:
    base 目录是用户配置,快照跨机器恢复时 base 可能变,相对引用不变。
    """
    ref: str
    original_chars: int
    trace_id: str
    tool_call_id: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CompactionResult:
    """一次压缩动作的审计记录:压了什么、留下什么、为什么。

    trigger:
      threshold  占用超过预算阈值,主动压缩
      overflow   API 已报 prompt 超长,被动抢救(账本#1 的吸收点)
      manual     用户/上层显式调用 compact()
    """
    trigger: Literal["threshold", "overflow", "manual"]
    tokens_before: int
    tokens_after: int
    summary: str
    dropped_message_count: int
    kept_tail_count: int
    focus: str | None = None          # 手动压缩时的聚焦指令
    middle_ref: str | None = None     # 被压缩中间段原文的卸载引用——
                                      # 可恢复压缩:摘要替换不等于原文销毁,
                                      # 原文换页到磁盘,模型可按 ref 取回
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContextSnapshot:
    """某个时间点的完整上下文状态,可序列化落盘。
    Phase 3 状态持久化的核心载荷之一(届时作为 RunContext 快照的一个字段)。
    注意:messages/摘要里的卸载引用(OffloadRecord.ref)指向的磁盘文件
    不随快照序列化——快照恢复的完整性依赖那些文件仍然存在,
    这是"卸载文件清理权归 Phase 3 会话生命周期管理"这个决策的原因。
    """
    messages: list[dict]
    compaction_history: list[CompactionResult] = field(default_factory=list)
    offload_records: list[OffloadRecord] = field(default_factory=list)
    total_tokens: int = 0
    token_source: str = "estimated"    # 与 tracing 的约定一致:api_usage | estimated
    last_updated: datetime = field(default_factory=datetime.now)