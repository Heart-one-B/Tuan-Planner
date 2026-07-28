# harness/memory/recall.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from harness.llm.base import LLMClientBase
from harness.memory.instructions import MEMORY_RECALL_MARKER, MemoryConfig
from harness.memory.models import MemoryRecord
from harness.memory.store import MemoryStore
from harness.message_id import find_index_by_hid, get_hid, tag_message

logger = logging.getLogger(__name__)

# ── 选择器提示词 ─────────────────────────────────────────────────────
# 与 extraction.py 的 EXTRACTION_SYSTEM_PROMPT 是姊妹提示词,但形态不同:
# 提取器是一个会调工具、要跑好几轮的 Agent;选择器是一次性的分类判断,
# 不调任何工具,只产出一段严格 JSON。所以选择器不装配 Agent,直接一次
# LLMClientBase.call() 了事——见下方 _select() 的实现说明。
RECALL_SYSTEM_PROMPT = """\
你是一个记忆相关性选择器。给你一段任务描述和一份候选记忆清单(只有
name/type/最后更新日期/一句话描述,没有正文),判断清单里哪些记忆对
完成这个任务确实有帮助。

规则:
- 只选你确信有帮助的,不确定的宁可不选——选错的代价(把不相关的内容
  塞进上下文,可能误导后续判断、浪费上下文空间)比漏选的代价(这条
  记忆的索引依然常驻在对话里,后续模型仍可能自己想起来去读)更高。
- 最多选 {top_k} 条,按相关性从高到低排列。
- 只能选清单里出现过的 name,不要编造清单里没有的名字。
- 如果没有任何一条确信相关,选出空列表,不要为了给出结果而凑数。

严格只输出以下 JSON 格式,不要有任何其他文字、不要用 markdown 代码块
包裹:
{{"selected": ["name1", "name2"]}}
"""

RECALL_USER_TEMPLATE = """\
本轮任务:
{task}

候选记忆清单:
{candidates}
"""

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _format_candidates(records: list[MemoryRecord]) -> str:
    lines = []
    for r in records:
        date = r.updated_at.strftime("%Y-%m-%d")
        lines.append(f"- {r.name} [{r.type}] ({date}): {r.description or '(无描述)'}")
    return "\n".join(lines)


def _parse_selection(raw: str) -> list[str]:
    """解析选择器输出。模型输出是不可信文本,即便提示词要求"只输出
    JSON",实践中仍常见包一层 markdown 代码块——先尝试剥掉代码围栏
    再解析,不行就按失败处理,不做更激进的猜测式修复(宁可少召回一次,
    不要把畸形输出硬凑成看似合理的结果)。"""
    if not raw or not raw.strip():
        raise ValueError("选择器返回了空内容")
    cleaned = _JSON_FENCE_RE.sub("", raw.strip()).strip()
    data = json.loads(cleaned)  # 抛 JSONDecodeError 按失败处理,调用方捕获
    selected = data.get("selected")
    if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
        raise ValueError(f"选择器输出的 'selected' 字段不是字符串列表: {selected!r}")
    return selected


async def _select(
    llm_client: LLMClientBase,
    trace_id: str,
    task: str,
    candidates: list[MemoryRecord],
    top_k: int,
) -> list[str]:
    """一次性的分类调用,不装配 Agent、不走 AgentLoop。刻意复用调用方
    传入的 trace_id(而不是像提取那样开一个子 Span)——这个调用没有
    工具调用、没有独立的消息历史,概念上是"这次 run 的准备工作的一
    部分",不是一个独立身份的嵌套 Agent。这和 Compactor 对压缩摘要
    调用的处理方式是同一个道理:压缩、召回都是"服务于本次 run 的
    辅助计算",复用 trace_id 让它在 trace 树里正确地表现为主 run 的
    一部分,而不是无关的兄弟节点。"""
    system = RECALL_SYSTEM_PROMPT.format(top_k=top_k)
    user = RECALL_USER_TEMPLATE.format(task=task, candidates=_format_candidates(candidates))
    resp = await llm_client.call(
        trace_id=trace_id,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    msg = resp.choices[0].message
    if getattr(msg, "finish_reason", None) == "length":
        raise ValueError("选择器输出被截断(finish_reason=length),按失败处理")
    return _parse_selection(msg.content)


def _build_injection_message(records: list[MemoryRecord], config: MemoryConfig) -> dict:
    """把选中的记忆全文拼成一条注入消息。日期一律绝对日期——警告文本
    如果写"3 天前"这种相对表述,过几天再看同一条历史消息文案就变了,
    对已经留在 history 里、不会被剥离重建的推送消息来说,这会让同一条
    消息在不同时间点呈现不同内容,是自相矛盾的(消息一旦发生就该是
    定格的历史记录)。这个纪律和索引、压缩摘要里的"相对日期转绝对
    日期"是同一条,在这里是第三次应用,原因完全一样:防止内容随时间
    静默漂移。"""
    sections = []
    for r in records:
        age_days = (datetime.now() - r.updated_at).days
        warning = ""
        if age_days >= config.staleness_days:
            warning = (
                f"[过期警告] 该记忆最后更新于 {r.updated_at.strftime('%Y-%m-%d')}"
                f"(约 {age_days} 天前),其中的信息可能已经过时,"
                f"采纳前请先验证是否仍然准确。\n"
            )
        sections.append(
            f"--- 记忆 '{r.name}' (type={r.type}) ---\n{warning}{r.content}"
        )
    body = "\n\n".join(sections)
    content = (
        f"{MEMORY_RECALL_MARKER} 根据本轮任务,系统从记忆库中检索到以下"
        f"可能相关的记忆(记忆是历史快照,不代表当前一定仍然成立,"
        f"发现与现实冲突时以现实为准):\n\n{body}"
    )
    # 打稳定 ID——这条消息是否仍然留在上下文里(而不是被压缩吃掉),
    # 是判断"里面这些记忆算不算已经推送过"的唯一可靠依据。多条记忆
    # 共享同一个 hid 是刻意的、正确的语义:它们是同一条消息的一部分,
    # 要么一起在场、要么一起被压缩吞掉,不存在"半条消息还在"的中间态。
    return tag_message({"role": "user", "content": content})


# ── 进程级熔断状态(与提取的熔断分开计数) ────────────────────────────
# 独立于 extraction.py 的 _failure_streaks——提取和召回是两个独立的
# 记忆子系统能力,一个坏了不该连累另一个的熔断判定。同样的模块级字典
# 模式、同样的代价声明,见 extraction.py 对应位置的讨论,不重复贴。
_recall_failure_streaks: dict[str, int] = {}


def _recall_circuit_open(session_id: str, max_failures: int) -> bool:
    return _recall_failure_streaks.get(session_id, 0) >= max_failures


def _note_recall_failure(session_id: str) -> int:
    n = _recall_failure_streaks.get(session_id, 0) + 1
    _recall_failure_streaks[session_id] = n
    return n


def _note_recall_success(session_id: str) -> None:
    _recall_failure_streaks.pop(session_id, None)


def reset_recall_circuit(session_id: str) -> None:
    _recall_failure_streaks.pop(session_id, None)


def recall_circuit_failure_count(session_id: str) -> int:
    return _recall_failure_streaks.get(session_id, 0)


def _emit(config: MemoryConfig, event_type: str, session_id: str,
          count: int = 0, error: str | None = None) -> None:
    """与 extraction.py 的 _emit 共用同一套事件字段形状(type/session_id/
    count/error)——这是审查后主动统一的,理由见 extraction.py 对应
    位置的注释,不重复贴。"""
    logger.info(f"[MemoryRecall] {event_type} session={session_id} "
               f"count={count} error={error}")
    if config.on_memory_event is None:
        return
    try:
        config.on_memory_event({
            "type": event_type, "session_id": session_id,
            "count": count, "error": error,
        })
    except Exception as e:
        logger.error(f"[MemoryRecall] on_memory_event 回调自身异常: {e}")


@dataclass
class SurfacedMemory:
    """一条"曾经推送过"的记录。hid 是承载这条记忆的召回注入消息的
    稳定 ID(见 harness/message_id.py)——None 表示"不知道"(v3 迁移
    过来的旧数据,没有 hid 概念),这类条目保守处理为"仍视作已推送、
    不允许重推",宁可少推一次也不引入误判重推的风险。

    to_dict/from_dict 是它在 RunSnapshot(JSON 往返)里的序列化契约,
    与 CompactionResult/OffloadRecord 用 dataclasses.asdict 的方式不同
    (那两个没有需要挑选字段的必要),这里显式写是因为将来这个类型
    可能长出更多字段,显式的 to_dict/from_dict 让"哪些字段进快照"
    是一个可审查的显式决定,不是隐式地把 dataclass 全部字段导出。
    """
    name: str
    hid: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "hid": self.hid}

    @classmethod
    def from_dict(cls, d: dict) -> "SurfacedMemory":
        return cls(name=d["name"], hid=d.get("hid"))


@dataclass
class RecallOutcome:
    """一次召回判定的结果。injection_message 为 None 表示"这次不注入
    任何东西"——可能是因为关闭、门槛未达、熔断、候选为空、选择器
    判定无相关项、或调用失败,调用方(Agent)不需要区分这些子原因,
    统一按"没有可注入的东西"处理即可,原因已经通过 _emit 走事件通道
    上报过了。newly_surfaced 是这次新推送的记忆(带 hid),调用方要把
    它们并入持久化的 surfaced_memories 列表——同名覆盖(见 agent.py
    的合并逻辑),因为重新推送意味着旧 hid 已经失效、要用新的替换。"""
    injection_message: dict | None = None
    newly_surfaced: list = field(default_factory=list)  # list[SurfacedMemory]


async def maybe_recall(
    *,
    session_id: str,
    task: str,
    trace_id: str,
    history: list,
    already_surfaced: list,  # list[SurfacedMemory]
    memory_store: MemoryStore,
    memory_config: MemoryConfig,
    fallback_llm_client: LLMClientBase,
) -> RecallOutcome:
    """每次 run 开始时调用一次(在组装本次 messages 之前)。自己决定
    "这次该不该真的召回"——门槛、熔断、候选过滤、调用、校验、组装
    注入消息全部封装在这一个入口里,调用方拿到结果直接用就行。
    失败一律降级为"不注入"(fail open),不影响主 run 正常进行——
    拉模式(常驻索引 + memory_read)依然是兜底。

    history 是本次 run 组装 messages 之前、已经确定会带入这次上下文
    的历史消息(即调用方传给 Agent.run 的 history 参数)——用来判断
    already_surfaced 里每一条记录的 hid 是否还"活着":hid 能在 history
    里找到,说明对应的召回注入消息还在上下文里,继续排除;hid 找不到了
    (八成是被压缩摘要吞掉了),说明那条记忆事实上已经不在上下文里,
    重新变为可召回候选——这就是"压缩联动"的精确实现,不再需要像
    水位线/boundary_hid 那样在锚点失效时整体退化。
    """
    if memory_config.recall_top_k is None:
        return RecallOutcome()

    records = memory_store.list_all()
    if len(records) < memory_config.recall_min_memories:
        # 记忆总数太少,索引本身已经够短够醒目,拉模式完全覆盖,
        # 不值得为了这么点候选特地烧一次模型调用。
        return RecallOutcome()

    live_names: set[str] = set()
    for entry in already_surfaced:
        if entry.hid is None:
            live_names.add(entry.name)  # legacy 数据,保守排除
        elif find_index_by_hid(history, entry.hid) is not None:
            live_names.add(entry.name)  # 确认承载它的注入消息仍在场
        # else: hid 曾经存在但现在的 history 里找不到了——不排除,
        # 这条记忆重新变为候选。

    candidates = [r for r in records if r.name not in live_names]
    if not candidates:
        # 能选的都已经在本会话推送过、且仍然在场,没有新东西可召回。
        return RecallOutcome()

    if _recall_circuit_open(session_id, memory_config.recall_max_failures):
        _emit(memory_config, "recall_skipped_circuit_open", session_id)
        return RecallOutcome()

    try:
        selected_names = await _select(
            memory_config.recall_llm or fallback_llm_client,
            trace_id, task, candidates, memory_config.recall_top_k,
        )
    except Exception as e:
        streak = _note_recall_failure(session_id)
        _emit(memory_config, "recall_failed", session_id, error=str(e))
        logger.error(f"[MemoryRecall] 选择器调用异常(连续失败 {streak} 次): {e}",
                    exc_info=True)
        return RecallOutcome()

    # 校验:模型输出的名字是不可信输入,逐个核对是否真的在候选集里——
    # 幻觉出来的名字静默丢弃,不能让整轮注入因为一个编造的名字失败。
    candidate_names = {r.name for r in candidates}
    valid_names = [n for n in selected_names if n in candidate_names]
    dropped = len(selected_names) - len(valid_names)
    if dropped:
        logger.warning(f"[MemoryRecall] 选择器返回了 {dropped} 个不在候选集里的"
                       f"名字,已丢弃")
    valid_names = valid_names[:memory_config.recall_top_k]  # 防御性截断

    _note_recall_success(session_id)
    if not valid_names:
        _emit(memory_config, "recall_done", session_id, count=0)
        return RecallOutcome()

    by_name = {r.name: r for r in candidates}
    selected_records = [by_name[n] for n in valid_names]
    injection = _build_injection_message(selected_records, memory_config)
    injection_hid = get_hid(injection)
    newly_surfaced = [SurfacedMemory(name=n, hid=injection_hid) for n in valid_names]
    _emit(memory_config, "recall_done", session_id, count=len(valid_names))
    return RecallOutcome(injection_message=injection, newly_surfaced=newly_surfaced)