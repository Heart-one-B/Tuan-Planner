# harness/memory/extraction.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness.llm.base import LLMClientBase
from harness.memory.inject import build_index_message
from harness.memory.instructions import MEMORY_DO_NOT_SAVE, MEMORY_RECALL_MARKER, MemoryConfig
from harness.memory.store import MemoryStore
from harness.memory.tools import MEMORY_WRITE_TOOL, build_memory_tools
from harness.message_id import find_index_by_hid
from harness.tools.tool_executor import ToolExecutor

if TYPE_CHECKING:
    from harness.agent.agent import Agent
    from harness.agent.loop import LoopOutcome

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
你是一个专职的记忆提取器,唯一职责是:扫一遍给你的对话片段,判断有没有
值得跨会话记住的内容,值得的话调用 memory_write 写下来。

你不参与、也看不到对话的后续走向——这段对话对用户而言可能已经翻篇了,
你的产出只对未来的会话有意义。所以:
- 只记跨会话仍然成立的东西,不要记"这次对话内部才有意义"的临时状态。
- 不确定是否值得记的,宁可不记——错误的记忆比没有记忆更糟。
- 你没有 memory_delete 权限,发现记忆冲突或过时不是你的职责范围。

四种记忆类型:
- user: 用户是谁——技能、背景、长期偏好。
- feedback: 怎么与用户协作——被纠正过的做法、明确的禁止事项。
- task: 当前任务的动态,且必须是跨会话仍有意义的进展/决策/截止日期。
- reference: 外部信息的位置——路径、链接、文档在哪。

写入纪律:
- feedback/task 类的 content 必须包含 Why(为什么)和 How to apply
  (什么情况下生效)两段。
- 一切日期写绝对日期,禁止"昨天""下周四"这类相对表述。
- description 要写得能唤起回忆,它是索引里唯一的一句话线索。
"""

EXTRACTION_TASK_TEMPLATE = """\
下面(作为 history)是自上次提取以来新增的对话片段,请扫一遍,判断有没有
值得跨会话记住的内容。

不要写入以下内容:
{do_not_save}

去重:
- 如果这段对话历史里已经出现过 memory_write 的调用,那部分已经记过了,
  不要重复写第二遍。
- 当前已有的记忆索引如下,索引里已经覆盖的内容不要重复写,除非这段
  新对话让某条记忆需要更新(此时用同名 memory_write 覆盖旧内容):

{current_index}

如果扫完之后确实没有新增的值得记内容,直接说"本轮无新增记忆"并结束,
不要为了写而写、不要凑数。
"""


def _build_extraction_agent(
    memory_store: MemoryStore,
    memory_config: MemoryConfig,
    llm_client: LLMClientBase,
) -> "Agent":
    """装配一个只有 memory_write/memory_read 权限的受限 Agent。

    【第一刀改动】不再传 termination=AnswerTermination()——AgentLoop
    不配置 require_terminal_tool 时的默认行为就是"无 tool_calls 即
    完成"(CC 语义),和原先 AnswerTermination 的行为完全等价,提取
    Agent 本来就是纯文本判定"本轮无新增记忆"就结束,不需要额外配置。
    """
    from harness.agent.agent import Agent  # 延迟导入,打破循环依赖
    from harness.agent.loop import Budget

    executor = ToolExecutor()
    for tool in build_memory_tools(memory_store, memory_config, include_delete=False):
        executor.register(tool)
    return Agent(
        llm_client=llm_client,
        tool_executor=executor,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        budget=Budget(max_tool_calls=8, max_rounds=6, exhausted_action="stop"),
        name="memory-extractor",
    )


def _strip_system_messages(messages: list) -> list:
    out = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "system":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(content, str) and content.startswith(MEMORY_RECALL_MARKER):
            continue
        out.append(m)
    return out


def _count_tool_calls(messages: list, tool_name: str) -> int:
    count = 0
    for m in messages:
        tool_calls = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        for tc in (tool_calls or []):
            if isinstance(tc, dict):
                name = (tc.get("function") or {}).get("name")
            else:
                name = getattr(getattr(tc, "function", None), "name", None)
            if name == tool_name:
                count += 1
    return count


_failure_streaks: dict[str, int] = {}


def _circuit_open(session_id: str, max_failures: int) -> bool:
    return _failure_streaks.get(session_id, 0) >= max_failures


def _note_failure(session_id: str) -> int:
    n = _failure_streaks.get(session_id, 0) + 1
    _failure_streaks[session_id] = n
    return n


def _note_success(session_id: str) -> None:
    _failure_streaks.pop(session_id, None)


def reset_circuit(session_id: str) -> None:
    _failure_streaks.pop(session_id, None)


def circuit_failure_count(session_id: str) -> int:
    return _failure_streaks.get(session_id, 0)


@dataclass
class ExtractionBookkeeping:
    boundary_hid: str | None = None
    runs_since: int = 0


def _emit(config: MemoryConfig, event_type: str, session_id: str,
          count: int = 0, error: str | None = None) -> None:
    logger.info(f"[MemoryExtraction] {event_type} session={session_id} "
               f"count={count} error={error}")
    if config.on_memory_event is None:
        return
    try:
        config.on_memory_event({
            "type": event_type, "session_id": session_id,
            "count": count, "error": error,
        })
    except Exception as e:
        logger.error(f"[MemoryExtraction] on_memory_event 回调自身异常: {e}")


async def maybe_extract(
    *,
    session_id: str,
    all_messages: list,
    bookkeeping: ExtractionBookkeeping,
    memory_store: MemoryStore,
    memory_config: MemoryConfig,
    fallback_llm_client: LLMClientBase,
    parent_span: "object | None" = None,
) -> ExtractionBookkeeping:
    if memory_config.extraction_every_n_runs is None:
        return bookkeeping

    new_runs_since = bookkeeping.runs_since + 1

    if new_runs_since < memory_config.extraction_every_n_runs:
        return ExtractionBookkeeping(boundary_hid=bookkeeping.boundary_hid,
                                     runs_since=new_runs_since)

    if _circuit_open(session_id, memory_config.extraction_max_failures):
        _emit(memory_config, "extraction_skipped_circuit_open", session_id)
        return ExtractionBookkeeping(boundary_hid=bookkeeping.boundary_hid,
                                     runs_since=new_runs_since)

    start = 0
    if bookkeeping.boundary_hid is not None:
        idx = find_index_by_hid(all_messages, bookkeeping.boundary_hid)
        if idx is not None:
            start = idx
        else:
            logger.warning(f"[MemoryExtraction] 提取锚点丢失(boundary_hid="
                           f"{bookkeeping.boundary_hid[:8]}...,大概率被压缩"
                           f"吞掉),退化为提取当前整个窗口")

    segment = _strip_system_messages(all_messages[start:])
    if not segment:
        return ExtractionBookkeeping(boundary_hid=None, runs_since=0)

    _emit(memory_config, "extraction_started", session_id)
    try:
        agent = _build_extraction_agent(
            memory_store, memory_config,
            memory_config.extraction_llm or fallback_llm_client,
        )
        index_msg = build_index_message(memory_store, memory_config)
        current_index = index_msg["content"] if index_msg else "(当前没有任何已存记忆)"
        task = EXTRACTION_TASK_TEMPLATE.format(
            do_not_save=MEMORY_DO_NOT_SAVE, current_index=current_index,
        )
        outcome: LoopOutcome = await agent.run(
            task=task, history=segment, session_id=f"{session_id}::extract",
            parent_span=parent_span,
        )
    except Exception as e:
        streak = _note_failure(session_id)
        _emit(memory_config, "extraction_failed", session_id, error=str(e))
        logger.error(f"[MemoryExtraction] 提取异常(连续失败 {streak} 次): {e}", exc_info=True)
        return ExtractionBookkeeping(boundary_hid=bookkeeping.boundary_hid,
                                     runs_since=new_runs_since)

    written = _count_tool_calls(outcome.messages, MEMORY_WRITE_TOOL)
    _note_success(session_id)
    _emit(memory_config, "extraction_done", session_id, count=written)
    return ExtractionBookkeeping(boundary_hid=None, runs_since=0)