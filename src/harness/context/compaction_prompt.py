# harness/context/compaction_prompt.py
from __future__ import annotations

import re

# ── 压缩契约(默认版) ─────────────────────────────────────────────────
# 设计原则:
#   1. 领域无关,整体可替换(Compactor(compaction_prompt=...))。
#   2. 章节即优先级声明——压缩是有损信道,丢什么不交给模型临场发挥。
#   3. 验收标准是"重建级":新会话拿摘要能继续干活。
#   4. 卸载引用必须一字不差保留。
#   5. "用户消息全录"是枚举不是概括——用户中途改需求/加约束是任务方向
#      变化的关键信号,交给模型判断"哪些算指令"必有遗漏,全录不判断。
#   6. <analysis>草稿区:先想后写,草稿最终被剥离不进对话——零成本提高
#      摘要质量的技巧。

DEFAULT_COMPACTION_PROMPT = """\
你是一个上下文压缩器。下面是一段智能体执行任务的对话历史,它即将被你的摘要替换,\
之后的智能体只能看到你的摘要,再也看不到原文。你的摘要是它继续完成任务的唯一依据,\
所以标准是"重建级":宁可啰嗦,不可遗漏关键信息。

输出格式要求:先在 <analysis> 标签内做草稿分析(这段不会被保留,用来帮你想清楚\
哪些信息重要),然后在 <summary> 标签内输出正式摘要。

<summary> 内严格按以下章节输出(章节标题原样保留,某章节确无内容则写"无"):

## 任务目标
用户最初要求做什么,以及过程中对目标的任何修正。

## 用户消息全录
按时间顺序逐条枚举这段历史中所有 user 角色的消息(工具结果除外),每条一行,\
可精简措辞但不可省略任何一条——用户的每次发言都可能是任务方向变化的信号,\
这一项是枚举,不是概括。

## 用户的明确指令与约束
用户说过的所有必须遵守的要求、偏好、禁止事项。逐条列出,不要合并改写。

## 已完成的工作与关键结论
做了什么、得到了什么结果。事实性结论(数据、答案、发现)必须原样保留,不要模糊化。

## 关键决策及理由
过程中做过的重要选择和当时的依据(为什么选A不选B)。

## 产物与引用
所有产生的引用必须一字不差地保留,包括但不限于:
- 卸载引用(形如 "[结果过大已卸载]...引用: xxx" 或 "已换页,引用: xxx" 中的引用路径,\
及其内容的一句话描述)
- 链接、标识符、路径、编号
这些引用是后续取回完整数据的唯一线索,丢失即永久不可恢复。

## 未完成的事项与下一步
还有什么没做完、当前卡在哪里、接下来应该做什么。当前进度要写到最细颗粒度\
(不是"正在处理数据",而是"正在处理X数据的第Y步,刚发现Z问题,下一步准备W")。

## 需要警惕的信息
已知的错误、失败的尝试(避免重蹈覆辙)、待验证的假设。
"""

_NO_QUESTIONS_CLAUSE = (
    "\n\n重要:本次压缩发生在任务执行中途,摘要中不要提出任何需要用户回答的问题"
    "(如\"请问您希望...\"),否则会打断正在进行的任务流。"
)


def build_compaction_messages(
    conversation_text: str,
    base_prompt: str = DEFAULT_COMPACTION_PROMPT,
    focus: str | None = None,
    allow_questions: bool = False,
) -> list[dict]:
    """组装压缩调用的 messages。

    focus 拼接进契约而不是取代契约:固定章节依然全部要产出,
    focus 只提升特定内容在有损信道中的优先级。
    allow_questions:自动触发(threshold/overflow)时为 False——摘要里
    冒出"请问您想优先A还是B"会卡死任务流;手动触发是用户主动干预,
    模型确认性提问无妨,为 True。
    """
    system = base_prompt
    if focus:
        system += (
            f"\n\n本次压缩的额外聚焦要求(在满足上述所有章节的前提下,"
            f"对以下方面给予更完整的保留):{focus}"
        )
    if not allow_questions:
        system += _NO_QUESTIONS_CLAUSE
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"需要压缩的对话历史如下:\n\n{conversation_text}"},
    ]


def extract_summary(text: str) -> str:
    """从压缩器输出中提取 <summary> 正文,剥离 <analysis> 草稿。
    模型没按格式输出标签时兜底:剥掉 analysis 块后取剩余全文——
    格式偏差不该让一次本来可用的摘要作废。"""
    m = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL).strip()
    return cleaned


def render_messages_for_compaction(messages: list) -> str:
    """把待压缩的消息段渲染成给压缩器看的纯文本。
    覆盖 dict 消息和 SDK 消息对象两种形态;tool_calls 渲染名字和参数
    (它们承载了"做过什么动作"的信息,不能只渲染 content)。"""
    lines = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "?")
            content = m.get("content") or ""
            lines.append(f"[{role}] {content}")
        else:
            role = getattr(m, "role", "assistant")
            content = getattr(m, "content", None) or ""
            parts = [content] if content else []
            for tc in (getattr(m, "tool_calls", None) or []):
                fn = getattr(tc, "function", None)
                parts.append(
                    f"<调用工具 {getattr(fn, 'name', '?')} 参数={getattr(fn, 'arguments', '')}>"
                )
            lines.append(f"[{role}] " + " ".join(parts))
    return "\n".join(lines)