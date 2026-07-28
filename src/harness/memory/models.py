# harness/memory/models.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# 四类记忆(拍板:project 改名 task)。type 编码的是"老化速度与验证策略",
# 与 scope(存哪/谁可见,由嵌入方选目录决定)正交:
#   user       这个人是谁(技能/背景/长期偏好)          老化极慢
#   feedback   怎么和这个人协作(纠正过的做法/禁止事项)   老化慢
#   task       当前这摊事的动态(决策/截止日期/进度)      老化快,天级
#   reference  外部信息在哪(路径/链接/文档位置)          老化中
MemoryType = Literal["user", "feedback", "task", "reference"]
VALID_TYPES: tuple[str, ...] = ("user", "feedback", "task", "reference")


@dataclass
class MemoryRecord:
    """一条记忆 = 一个 md 文件。frontmatter 携带元数据,正文是内容本体。

    为什么记忆用 md 而快照用 json(同一条选型原则的两次应用):
    格式由"谁来读它"决定——记忆的读者是模型和人(按文本理解消费,
    人可以直接开编辑器改),md 合适;快照的读者是程序(逐字节精确重建),
    json 合适。
    """
    name: str                 # 文件名主体(不含 .md),也是工具读写的主键
    type: str                 # MemoryType 之一
    description: str          # 一句话描述,索引的原料——它的质量直接决定
                              # 模型"扫索引时能不能想起来要读这条"
    content: str              # 正文
    updated_at: datetime = field(default_factory=datetime.now)

    def render(self) -> str:
        """渲染成落盘的 md 全文(frontmatter + 正文)。
        updated_at 写绝对日期:相对时间('3天前')每天自己失效,
        绝对日期只在内容真变时才变——对 prompt cache 友好,
        也是压缩契约'相对日期转绝对日期'同一条纪律。"""
        return (
            f"---\n"
            f"name: {self.name}\n"
            f"type: {self.type}\n"
            f"description: {self.description}\n"
            f"updated_at: {self.updated_at.strftime('%Y-%m-%d')}\n"
            f"---\n\n"
            f"{self.content}\n"
        )


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_memory_file(text: str, fallback_name: str = "") -> MemoryRecord:
    """从 md 全文解析出 MemoryRecord。

    容错原则:frontmatter 缺失或字段残缺时不抛异常,尽力解析并给
    保守默认值(type 落回 reference——四类中语义最中性的一类)。
    理由:记忆文件允许人直接用编辑器创建/修改(这是选 md 的初衷之一),
    人写的文件不能要求机器级的格式严格性;一条格式不完美的记忆,
    有残缺元数据总好过整个读取失败。
    """
    m = _FRONTMATTER_RE.match(text)
    meta: dict[str, str] = {}
    body = text
    if m:
        body = text[m.end():]
        for line in m.group(1).splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()

    mem_type = meta.get("type", "")
    if mem_type not in VALID_TYPES:
        mem_type = "reference"

    updated_at = datetime.now()
    raw_date = meta.get("updated_at", "")
    if raw_date:
        try:
            updated_at = datetime.strptime(raw_date, "%Y-%m-%d")
        except ValueError:
            pass   # 日期格式坏了不致命,落回当前时间(宁可显得新也不丢记录)

    return MemoryRecord(
        name=meta.get("name") or fallback_name,
        type=mem_type,
        description=meta.get("description", ""),
        content=body.strip(),
        updated_at=updated_at,
    )