# harness/memory/store.py
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Protocol

from harness.memory.models import MemoryRecord, parse_memory_file

logger = logging.getLogger(__name__)

# name 清洗:与 FileSnapshotStore._sanitize_session_id 同构。
# memory_write 的 name 参数来自模型输出,是不可信输入,直接拼文件名
# 就是路径穿越攻击面——这个学费在 OffloadStore.load 和 SnapshotStore
# 已经付过两次,白名单清洗 + resolve 兜底双保险,不再犯第三次。
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_\-.\u4e00-\u9fff]")


def _sanitize_name(name: str) -> str:
    safe = _UNSAFE_CHARS.sub("_", name.strip())
    return safe or "_"


class MemoryStore(Protocol):
    """记忆存取协议。第一期只有 FileMemoryStore 一个实现,协议先立着——
    与 TraceStorageBase / SnapshotStore 同族:公共接口 + 可替换实现。"""

    def write(self, record: MemoryRecord) -> str: ...
    def read(self, name: str) -> MemoryRecord: ...
    def delete(self, name: str) -> bool: ...
    def list_all(self) -> list[MemoryRecord]: ...


class FileMemoryStore:
    """文件系统实现:{base_dir}/{name}.md,一条记忆一个文件。

    scope 的落法(之前拍板'scope 任意字符串不绑 git'):嵌入方选目录,
    FileMemoryStore(base / scope)。模型无权跨 scope——工具绑定在
    构造时的这一个目录上,跨 scope 是权限问题,归 Phase 6。

    只做平铺目录不做子目录:第一期记忆量级(百条内)不需要层级,
    索引双截断(200行)先到上限。层级化留给量级真的上来之后。
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def _path_of(self, name: str) -> Path:
        safe = _sanitize_name(name)
        p = (self.base_dir / f"{safe}.md").resolve()
        base_resolved = self.base_dir.resolve()
        if not p.is_relative_to(base_resolved):
            raise ValueError(f"非法记忆名(路径穿越): {name}")
        return p

    def write(self, record: MemoryRecord) -> str:
        """写入(覆盖即更新)。updated_at 强制刷为当天——记忆的新鲜度
        以'最后写入'为准,不信任调用方传入的旧时间戳。"""
        record.updated_at = datetime.now()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        p = self._path_of(record.name)
        p.write_text(record.render(), encoding="utf-8")
        logger.info(f"[Memory] write: {record.name} (type={record.type})")
        return str(p)

    def read(self, name: str) -> MemoryRecord:
        p = self._path_of(name)
        if not p.is_file():
            raise FileNotFoundError(f"记忆不存在: {name}")
        record = parse_memory_file(p.read_text(encoding="utf-8"), fallback_name=name)
        # 新鲜度以文件系统 mtime 为准,它是唯一不会被 frontmatter
        # 手写错误污染的时间来源
        record.updated_at = datetime.fromtimestamp(p.stat().st_mtime)
        return record

    def delete(self, name: str) -> bool:
        p = self._path_of(name)
        if not p.is_file():
            return False
        p.unlink()
        logger.info(f"[Memory] delete: {name}")
        return True

    def list_all(self) -> list[MemoryRecord]:
        """全量列出(索引生成的原料)。只读 frontmatter 需要的部分也得
        打开整个文件,但记忆文件都很小(索引双截断间接约束了总量),
        第一期不做惰性解析优化。按 updated_at 降序:最近更新的排前面,
        索引若被截断,牺牲的是最旧的条目——和沉底清理同一价值排序。"""
        if not self.base_dir.is_dir():
            return []
        records = []
        for p in sorted(self.base_dir.glob("*.md")):
            try:
                record = parse_memory_file(p.read_text(encoding="utf-8"),
                                           fallback_name=p.stem)
                record.updated_at = datetime.fromtimestamp(p.stat().st_mtime)
                records.append(record)
            except Exception as e:
                # 单个文件损坏不拖垮整个索引——跳过并记日志,
                # 这是 list 语义下唯一合理的失败处理
                logger.warning(f"[Memory] 跳过无法解析的记忆文件 {p.name}: {e}")
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records