# harness/context/offload.py
from __future__ import annotations

import logging
from pathlib import Path

from harness.context.models import OffloadRecord
from harness.tools.tool_definition import ToolDefinition

logger = logging.getLogger(__name__)

OFFLOAD_MARKER = "[结果过大已卸载]"
CLEARED_MARKER = "[工具结果已清理]"
RETRIEVAL_TOOL_NAME = "read_offloaded_result"


class OffloadStore:
    """卸载文件的读写。只写不删——清理权归 Phase 3 的会话生命周期管理,
    本层不掌握"这个文件是否还被某个快照引用"的知识,不做删除决策。

    目录组织:{base_dir}/{trace_id}/{tool_call_id}.txt
    对外引用一律用相对路径(ref),base_dir 是部署配置,不进引用。
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def save(self, trace_id: str, tool_call_id: str, content: str) -> OffloadRecord:
        safe_call_id = tool_call_id.replace("/", "_").replace("\\", "_")
        rel = Path(trace_id) / f"{safe_call_id}.txt"
        abs_path = self.base_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        logger.info(f"[Offload] saved {len(content)} chars -> {rel}")
        return OffloadRecord(
            ref=str(rel), original_chars=len(content),
            trace_id=trace_id, tool_call_id=tool_call_id,
        )

    def load(self, ref: str) -> str:
        """按相对引用读回全文。防路径穿越:解析后必须仍在 base_dir 内——
        取回工具的 ref 参数最终来自模型输出,属于不可信输入。"""
        abs_path = (self.base_dir / ref).resolve()
        base_resolved = self.base_dir.resolve()
        if not abs_path.is_relative_to(base_resolved):
            raise ValueError(f"非法引用(路径穿越): {ref}")
        if not abs_path.is_file():
            raise FileNotFoundError(f"卸载文件不存在: {ref}")
        return abs_path.read_text(encoding="utf-8")


def make_preview(content: str, record: OffloadRecord, preview_chars: int) -> str:
    """卸载后留在上下文里的替身:头尾预览 + 引用 + 取回指引。
    头尾各留而非只留头——日志的报错在尾部、列表的长尾在尾部,
    这是 Phase 1 设计截断时就确认过的原则,卸载预览沿用。"""
    half = max(1, preview_chars // 2)
    head, tail = content[:half], content[-half:]
    return (
        f"{OFFLOAD_MARKER} 原始长度 {record.original_chars} 字符,"
        f"完整内容引用: {record.ref}\n"
        f"(如需完整内容,调用 {RETRIEVAL_TOOL_NAME} 工具,参数 ref=\"{record.ref}\")\n"
        f"--- 开头预览 ---\n{head}\n--- 结尾预览 ---\n{tail}"
    )


def offload_if_oversized(
    content: str,
    store: OffloadStore,
    trace_id: str,
    tool_call_id: str,
    max_chars: int,
    preview_chars: int,
) -> tuple[str, OffloadRecord | None]:
    """管"太大":超阈值则落盘换预览,否则原样通过。
    返回 (进入上下文的文本, 卸载档案或None)。"""
    if len(content) <= max_chars:
        return content, None
    record = store.save(trace_id, tool_call_id, content)
    return make_preview(content, record, preview_chars), record


def clear_stale_tool_results(
    messages: list, keep_recent_rounds: int,
    store: "OffloadStore | None" = None, trace_id: str | None = None,
) -> int:
    """管"太旧":最近 N 个工具调用轮次之外的 tool 消息清出上下文。

    提供 store+trace_id 时是"换页"——原文先落盘,占位符携带 ref,
    模型可用取回工具读回。不提供时退回旧的"销毁"行为(仅测试/特殊场景)。

    换页而非销毁的原因(对照修正):无差别销毁会摧毁不可再生的结果——
    本 harness 里子 Agent 以工具形态被调用,其产出(可能花了几分钟和
    大量 token)就躺在 tool 消息里,销毁后无法靠"重跑工具"找回。
    Claude Code 的对应做法是只清可重新获取的工具结果、绝不裁剪子 agent
    输出;我们的通用化等价物是"清任何东西之前先换页",磁盘不要钱,
    换页对所有工具一视同仁,不需要业务标注哪些工具可再生。
    原地修改 messages,返回清理条数。"""
    round_starts = [
        i for i, m in enumerate(messages)
        if (getattr(m, "tool_calls", None) or (isinstance(m, dict) and m.get("tool_calls")))
    ]
    if len(round_starts) <= keep_recent_rounds:
        return 0
    cutoff = round_starts[-keep_recent_rounds]

    cleared = 0
    for i, m in enumerate(messages):
        if i >= cutoff:
            break
        if not (isinstance(m, dict) and m.get("role") == "tool"):
            continue
        content = m.get("content", "")
        if content.startswith(CLEARED_MARKER) or content.startswith(OFFLOAD_MARKER):
            continue   # 已是占位/预览,不重复清理(预览含取回引用,保留)
        call_id = m.get("tool_call_id")
        if store is not None and trace_id:
            record = store.save(trace_id, f"stale_{call_id}", content)
            m["content"] = (
                f"{CLEARED_MARKER} 原文已换页,引用: {record.ref}"
                f"(如需内容可调用 {RETRIEVAL_TOOL_NAME} 取回) tool_call_id={call_id}"
            )
        else:
            m["content"] = f"{CLEARED_MARKER} tool_call_id={call_id}"
        cleared += 1
    if cleared:
        logger.info(f"[Offload] 清理了 {cleared} 条沉底工具结果"
                    f"({'换页' if store else '销毁'})")
    return cleared


def build_retrieval_tool(store: OffloadStore, max_return_chars: int = 20_000) -> ToolDefinition:
    """取回工具——卸载设计的灵魂,没有它卸载在模型视角等于删除。
    注册进 ToolExecutor 后,模型看到预览里的 ref 就能自主取回全文。
    返回也设长度上限:取回一个 500KB 的文件原样塞回上下文,
    等于把卸载的努力原地清零,超长部分提示用 offset 分页读取。"""

    async def read_offloaded_result(ref: str, offset: int = 0) -> str:
        content = store.load(ref)
        chunk = content[offset:offset + max_return_chars]
        if offset + max_return_chars < len(content):
            chunk += (
                f"\n...[未完,总长 {len(content)} 字符,"
                f"继续读取请传 offset={offset + max_return_chars}]"
            )
        return chunk

    return ToolDefinition(
        name=RETRIEVAL_TOOL_NAME,
        description=(
            "读取先前因过大而被卸载到磁盘的工具结果全文。"
            "当你在对话历史中看到"
            f"'{OFFLOAD_MARKER}'标记及其 ref 引用,且预览信息不足以完成任务时调用。"
            "超长内容分页返回,用 offset 继续读取。"
        ),
        parameters={
            "ref": {"type": "string", "description": "卸载标记中给出的引用路径"},
            "offset": {"type": "integer", "description": "起始字符位置,默认0,分页时使用"},
        },
        required=["ref"],
        func=read_offloaded_result,
    )