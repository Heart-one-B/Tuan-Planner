# harness/snapshot/models.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SNAPSHOT_VERSION = 6
# v5 -> v6 变更(第三刀,退出原因两级化 + 恢复计数器):新增
# exit_reason(细粒度退出原因,配合 status 的两级设计——status 是宿主
# 决策用的粗信号,exit_reason 是诊断/上报用的细信号,详见
# harness/agent/loop.py LoopOutcome 的字段注释)+ overflow_recovery_count/
# output_truncation_count/output_upgraded/terminal_nudge_count 四个
# LoopState 计数器/标志位。
#
# 这一步是 bug#1("overflow_recovered 跨 resume 边界重置")真正被修复
# 的地方:第二刀只是把这个字段从局部变量搬进 LoopState,行为没变
# (resume 时仍然从 0/False 重新开始);v6 把它纳入快照,
# Agent.resume_events() 从这里读回真实值重建 LoopState,跨 resume
# 边界的记忆才算真正接上,不是又一次"搬家但没修"。


def normalize_message(msg) -> dict:
    if isinstance(msg, dict):
        out = {k: v for k, v in msg.items() if k != "reasoning_content"}
        return json.loads(json.dumps(out, ensure_ascii=False, default=str))

    out: dict = {"role": getattr(msg, "role", "assistant"), "content": getattr(msg, "content", None)}
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = [{
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        } for tc in tool_calls]
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id is not None:
        out["tool_call_id"] = tool_call_id
    return out


def result_to_dict(c) -> dict:
    d = asdict(c)
    d["timestamp"] = c.timestamp.isoformat()
    return d


def record_to_dict(r) -> dict:
    d = asdict(r)
    d["created_at"] = r.created_at.isoformat()
    return d


@dataclass
class RunSnapshot:
    session_id: str
    task: str
    trace_id: str
    status: str
    rounds: int
    tool_calls_used: int
    messages: list = field(default_factory=list)
    compaction_history: list = field(default_factory=list)
    offload_records: list = field(default_factory=list)
    total_tokens: int = 0
    token_source: str = "estimated"
    created_at: str = ""
    version: int = SNAPSHOT_VERSION
    extraction_boundary_hid: str | None = None
    runs_since_extraction: int = 0
    surfaced_memories: list = field(default_factory=list)
    pending_approval_id: str | None = None
    pending_tool_call_id: str | None = None
    exit_reason: str | None = None
    overflow_recovery_count: int = 0
    output_truncation_count: int = 0
    output_upgraded: bool = False
    terminal_nudge_count: int = 0

    def resume_history(self) -> list[dict]:
        i = 0
        while i < len(self.messages) and self.messages[i].get("role") == "system":
            i += 1
        return self.messages[i:]

    def all_refs(self) -> set[str]:
        refs = {r["ref"] for r in self.offload_records}
        refs |= {c["middle_ref"] for c in self.compaction_history if c.get("middle_ref")}
        return refs

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "RunSnapshot":
        version = data.get("version")
        if version == 1:
            data = migrate_v1_to_v2(data)
            version = data["version"]
        if version == 2:
            data = migrate_v2_to_v3(data)
            version = data["version"]
        if version == 3:
            data = migrate_v3_to_v4(data)
            version = data["version"]
        if version == 4:
            data = migrate_v4_to_v5(data)
            version = data["version"]
        if version == 5:
            data = migrate_v5_to_v6(data)
            version = data["version"]
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"快照版本不兼容: 文件版本={version}, 当前代码版本={SNAPSHOT_VERSION}。"
                f"拒绝半解析——版本迁移需要显式的迁移函数,不能假装能读。"
            )
        return cls(**data)


def migrate_v1_to_v2(data: dict) -> dict:
    migrated = dict(data)
    migrated["version"] = 2
    migrated.setdefault("extraction_watermark", None)
    migrated.setdefault("runs_since_extraction", 0)
    return migrated


def migrate_v2_to_v3(data: dict) -> dict:
    migrated = dict(data)
    migrated["version"] = 3
    migrated.setdefault("surfaced_memories", [])
    return migrated


def migrate_v3_to_v4(data: dict) -> dict:
    migrated = dict(data)
    migrated["version"] = 4
    migrated.pop("extraction_watermark", None)
    migrated.setdefault("extraction_boundary_hid", None)
    old_surfaced = migrated.get("surfaced_memories") or []
    if old_surfaced and isinstance(old_surfaced[0], str):
        migrated["surfaced_memories"] = [{"name": n, "hid": None} for n in old_surfaced]
    return migrated


def migrate_v4_to_v5(data: dict) -> dict:
    migrated = dict(data)
    migrated["version"] = 5
    migrated.setdefault("pending_approval_id", None)
    migrated.setdefault("pending_tool_call_id", None)
    return migrated


def migrate_v5_to_v6(data: dict) -> dict:
    """v5 没有细粒度退出原因和恢复计数器的概念。exit_reason=None 表示
    "未知"(老快照的 status 字段仍然准确,只是没有细分);四个新字段
    归零/False,是"这些恢复在 v5 时代不被追踪"的直接翻译——不假装
    能从旧数据反推出真实发生过几次紧急压缩/截断恢复。"""
    migrated = dict(data)
    migrated["version"] = 6
    migrated.setdefault("exit_reason", None)
    migrated.setdefault("overflow_recovery_count", 0)
    migrated.setdefault("output_truncation_count", 0)
    migrated.setdefault("output_upgraded", False)
    migrated.setdefault("terminal_nudge_count", 0)
    return migrated