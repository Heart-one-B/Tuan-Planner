# harness/snapshot/models.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SNAPSHOT_VERSION = 1


def normalize_message(msg) -> dict:
    """任意形态的消息(dict / SDK 对象 / SimpleNamespace)归一为可 JSON
    往返的 dict。这是快照的"存活契约":只有这里显式提取的字段能活过
    序列化。

    刻意排除 reasoning:回放前本就要剥(见 OpenAIClient._strip_reasoning_
    for_replay),它的观测价值已经在 tracing 层保留,快照不重复承担。

    assistant 消息的 content 字段即使为 None 也显式写入(而不是省略
    该 key)——OpenAI 兼容 API 的标准形态是 content 键存在、值为 null,
    省略键在部分 provider/SDK 上可能触发校验错误,写 None 更保守。
    """
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
    """CompactionResult(harness.context.models)→dict。刻意用 duck typing
    (不 import 该类型)——本模块的零依赖纪律优先于类型标注的精确性,
    调用方(build.py)传什么形状进来,这里就按属性读什么。"""
    d = asdict(c)
    d["timestamp"] = c.timestamp.isoformat()
    return d


def record_to_dict(r) -> dict:
    """OffloadRecord(harness.context.models)→dict,同上不 import。"""
    d = asdict(r)
    d["created_at"] = r.created_at.isoformat()
    return d


@dataclass
class RunSnapshot:
    """一次运行结束时的完整工作现场,可 JSON 往返。

    本文件零依赖 harness.agent / harness.context —— 和 context/models.py
    "不 import agent 下任何东西"是同一条纪律的延伸:纯数据结构应该能被
    任何消费者(未来的 DagScheduler、memory 系统、离线分析脚本)直接
    引用,不该背上"必须先装配一整套 Agent 运行时"的依赖负担。真正需要
    从 LoopOutcome/RunContext 提取快照的转换逻辑在 build.py。

    显式排除项及理由(不是遗漏,是决策):
      - Span 对象      只存 trace_id 字符串。tracing 是观测,快照是状态,
                       观测数据在 trace.db 里,两边用 trace_id 关联。
      - RunContext.state dict   任意业务对象,无法保证可序列化。记入
                       缺口,Phase 6 的 ask_human 需要时回来解决。
      - 工具 schemas    工具是代码不是数据,恢复时由装配代码重新注册。
      - reasoning      见 normalize_message 的说明。

    version 字段是序列化系统的第一课:没有它,第一次改格式就会静默
    毁掉所有旧快照。加载时 version 不匹配必须显式报错,不允许半解析。
    """
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

    def resume_history(self) -> list[dict]:
        """恢复为新 run 的 history 参数:剥掉开头连续的 system 消息。

        指令是配置、对话是数据——恢复时应使用 Agent 当前的 system prompt
        (它可能已更新过),而不是快照里冻结的旧版,否则会拼出重复的
        system 消息。Claude Code 在 compact 后重新加载 CLAUDE.md 是
        同一条原则的体现。
        """
        i = 0
        while i < len(self.messages) and self.messages[i].get("role") == "system":
            i += 1
        return self.messages[i:]

    def all_refs(self) -> set[str]:
        """本快照引用的全部卸载文件——级联清理的存活判据之一
        (完整判据是"该 session 全部存活快照的 all_refs 并集")。"""
        refs = {r["ref"] for r in self.offload_records}
        refs |= {c["middle_ref"] for c in self.compaction_history if c.get("middle_ref")}
        return refs

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "RunSnapshot":
        version = data.get("version")
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"快照版本不兼容: 文件版本={version}, 当前代码版本={SNAPSHOT_VERSION}。"
                f"拒绝半解析——版本迁移需要显式的迁移函数,不能假装能读。"
            )
        return cls(**data)