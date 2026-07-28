# harness/memory/instructions.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from harness.llm.base import LLMClientBase

logger = logging.getLogger(__name__)

MEMORY_RULES_TEMPLATE = """\

# 记忆系统使用规则

你拥有一个跨会话的记忆系统。对话开头会注入一份记忆索引(标记为
[记忆索引]),列出现有记忆的名称、类型和描述;需要某条的完整内容时,
调用 memory_read 读取。

## 四种记忆类型
- user: 用户是谁——技能、背景、长期偏好。老化极慢。
- feedback: 怎么与用户协作——被纠正过的做法、明确的禁止事项。老化慢。
- task: 当前任务的动态——已做的决策、截止日期、进行到哪。老化快。
- reference: 外部信息的位置——路径、链接、文档在哪。老化中等。

## 写入纪律(主对话专用:只写"显式信号")
你不是这个记忆系统唯一的写入者——后台还有一个专门的提取流程会定期
扫描整段对话、把该记的都记下来,不需要你顺手记。所以你只在以下两种
情况下调用 memory_write:
- 用户**明确要求**记住某件事("记住我...""以后都..."类似表达)
- 你**明确被用户纠正**,且这条纠正值得跨会话保留
除此之外一律不要主动写入——分心去判断"这次有没有什么该记的"会
拖慢你的主任务,这部分工作交给后台去做。

- 写入不需要征求用户同意(用户会通过事件流看到写入发生)。
- feedback 和 task 类必须包含 Why(为什么有这条规则/决策,通常是踩过
  的坑)和 How to apply(什么情况下生效)。只记规则不记原因,遇到边界
  情况就无法判断该不该破例。
- 一切日期写绝对日期(如 2026-07-14),禁止相对表述("周四之前"、
  "昨天说的")——相对日期几天后就失去意义。
- description 要写得能唤起回忆:未来的你只看得到索引里这一句话,
  靠它决定要不要读全文。

## 不要写入(负面清单)
- 能从代码/文件里直接读到的东西(架构、函数签名、文件路径下的内容)
- 版本控制系统里已有的信息(提交历史是权威,不要抄一份)
- 一次性的任务细节(本次对话结束就没有意义的东西)
- 系统提示/指令文件里已有的内容(不要重复)

## 使用纪律
- 记忆是线索,不是事实。"记忆说 X 存在"不等于"X 现在仍然存在"——
  尤其是带过期警告的记忆,采纳前先验证(去读一下文件、查一下现状)。
- 发现记忆与现实冲突时,以现实为准,并更新或删除那条记忆。
"""

# 负面清单的独立片段(第二期新增):提取提示词需要引用同一份负面清单,
# 从 MEMORY_RULES_TEMPLATE 里正则抠出来既脆弱又不透明,不如从一开始
# 就单独成一份常量、两处都引用同一个来源——"不抄两份"的具体做法。
MEMORY_DO_NOT_SAVE = """\
- 能从代码/文件里直接读到的东西(架构、函数签名、文件路径下的内容)
- 版本控制系统里已有的信息(提交历史是权威,不要抄一份)
- 一次性的任务细节(本次对话结束就没有意义的东西)
- 系统提示/指令文件里已有的内容(不要重复)\
"""

MEMORY_INDEX_MARKER = "[记忆索引]"

# 召回(第三期)注入消息的标记。与 MEMORY_INDEX_MARKER 刻意不同类:
# 索引是派生配置,每次 run 都会被 strip_old_index_messages 剥掉重注入
# (保护缓存前缀这件事不适用于它,它每次都不同);召回注入的消息是
# "针对那一轮任务给出的上下文",和那一轮对话是一体的,属于对话数据,
# 留在 history 里不剥离——剥掉 history 中段的消息会改变前缀,打穿
# 从那之后所有轮次的缓存。这个标记只用于人工排查日志/消息时识别,
# 不参与任何自动剥离逻辑,今后也不应该给它加一个 strip_old_recall_
# messages 之类的函数。
MEMORY_RECALL_MARKER = "[记忆召回]"


@dataclass
class MemoryConfig:
    """记忆系统装配配置。

    第二期新增字段(全部可选,默认值等价于"不启用后台提取",延续
    "新能力必须可选、不配置即无感"的老规矩):

    extraction_every_n_runs  每隔 N 次 run 结束触发一次后台提取;
                             None = 关闭提取(只有第一期的主对话
                             显式写入路径生效)。
    extraction_mode          "auto"(默认):Agent.run()/events() 自己
                             在收尾处同步 await 提取,行为最简单,适合
                             脚本形态或不在意阻塞的场景。
                             "manual":Agent 不自动触发,由宿主在拿到
                             run() 的结果后自行调用 agent.extract_now(...),
                             可以直接 await(同步)也可以用
                             asyncio.create_task(...) 包成后台任务——
                             并发模型的选择权完全交给宿主,harness 不替
                             它决定("库不该偷偷 spawn 任务")。
    extraction_llm           提取用的 LLM 客户端,建议用便宜模型;
                             None 时退回调用方传给提取入口的默认客户端
                             (通常是主 Agent 的 llm_client)。
    extraction_max_failures  连续失败达到此次数即熔断,本进程内停止
                             后续提取尝试(不落盘,进程重启即半开)。
    on_memory_event           宿主告警回调,记忆子系统(提取、第三期的
                             推模式召回)的关键事件都会调用它一次,
                             通过事件 dict 里的 "type" 字段区分来源
                             (如 "extraction_started"/"recall_failed");
                             None 时只记日志。一个回调口接全部记忆
                             子系统事件,比每个子系统各开一个回调字段
                             干净——宿主只需要接一次线。回调必须是
                             同步且不抛异常的轻量函数——它跑在记忆
                             流程内部,自己出错不该拖垮记忆流程本身
                             (extraction.py/recall.py 会包一层 try/except)。
    """
    memory_dir: Path
    managed_path: Path | None = None
    user_path: Path | None = None
    local_path: Path | None = None
    staleness_days: int = 2
    index_max_lines: int = 200
    index_max_bytes: int = 25_000

    extraction_every_n_runs: int | None = None
    extraction_mode: Literal["auto", "manual"] = "auto"
    extraction_llm: "LLMClientBase | None" = None
    extraction_max_failures: int = 3

    recall_top_k: int | None = None
    recall_min_memories: int = 8
    recall_llm: "LLMClientBase | None" = None
    recall_max_failures: int = 3

    on_memory_event: Callable[[dict], None] | None = None


_LAYER_HEADERS = (
    ("managed_path", "## 部署指令(最高优先级)"),
    ("user_path", "## 用户全局偏好"),
    ("local_path", "## 本地指令"),
)


def load_static_instructions(config: MemoryConfig) -> str:
    sections = []
    for attr, header in _LAYER_HEADERS:
        path: Path | None = getattr(config, attr)
        if path is None:
            continue
        path = Path(path)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"[Memory] 静态层文件读取失败,跳过 {path}: {e}")
            continue
        if text:
            sections.append(f"{header}\n{text}")
    return "\n\n".join(sections)