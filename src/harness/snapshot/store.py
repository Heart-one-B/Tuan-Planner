# harness/snapshot/store.py
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Protocol

from harness.snapshot.models import RunSnapshot

logger = logging.getLogger(__name__)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_\-.]")


def _sanitize_session_id(session_id: str) -> str:
    safe = _UNSAFE_CHARS.sub("_", session_id)
    return safe or "_"


class SnapshotStore(Protocol):
    """注意:list_archived/purge 是级联清理(harness/session/cleanup.py)
    需要的能力,天然与"文件系统上有归档目录"这个具体实现细节绑定——
    一个假想的 PostgresSnapshotStore 大概率没有"归档文件"这个概念,
    会有自己完全不同的清理策略。这两个方法留在 Protocol 里是为了让
    "文件系统类实现都应该提供这个能力"显式可见,但 cleanup.py 的
    公开函数签名直接约束为 FileSnapshotStore(具体类)而不是这个
    Protocol,如实反映这个耦合,不假装它是后端无关的。"""
    def save(self, snapshot: RunSnapshot) -> str: ...
    def load_latest(self, session_id: str) -> RunSnapshot: ...
    def list_sessions(self) -> list[str]: ...
    def list_archived(self, session_id: str) -> list[RunSnapshot]: ...
    def purge(self, session_id: str) -> bool: ...


class FileSnapshotStore:
    def __init__(self, base_dir: Path, keep_archives: int = 3):
        self.base_dir = Path(base_dir)
        self.keep_archives = keep_archives

    def _session_dir(self, session_id: str) -> Path:
        safe = _sanitize_session_id(session_id)
        d = self.base_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, snapshot: RunSnapshot) -> str:
        session_dir = self._session_dir(snapshot.session_id)
        latest_path = session_dir / "latest.json"
        payload = snapshot.to_json()

        fd, tmp_path = tempfile.mkstemp(dir=session_dir, prefix=".latest_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, latest_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        if self.keep_archives > 0:
            self._write_archive(session_dir, payload)

        logger.info(f"[FileSnapshotStore] saved session={snapshot.session_id} "
                   f"rounds={snapshot.rounds} messages={len(snapshot.messages)}")
        return str(latest_path)

    def _write_archive(self, session_dir: Path, payload: str) -> None:
        archive_dir = session_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        (archive_dir / f"{stamp}.json").write_text(payload, encoding="utf-8")

        existing = sorted(archive_dir.glob("*.json"))
        excess = len(existing) - self.keep_archives
        for old in existing[:max(0, excess)]:
            old.unlink(missing_ok=True)

    def load_latest(self, session_id: str) -> RunSnapshot:
        safe = _sanitize_session_id(session_id)
        latest_path = (self.base_dir / safe / "latest.json").resolve()
        base_resolved = self.base_dir.resolve()
        if not latest_path.is_relative_to(base_resolved):
            raise ValueError(f"非法 session_id(路径穿越): {session_id}")
        if not latest_path.is_file():
            raise FileNotFoundError(f"未找到会话快照: {session_id}")

        data = json.loads(latest_path.read_text(encoding="utf-8"))
        return RunSnapshot.from_dict(data)

    def list_sessions(self) -> list[str]:
        if not self.base_dir.is_dir():
            return []
        return sorted(
            p.name for p in self.base_dir.iterdir()
            if p.is_dir() and (p / "latest.json").is_file()
        )

    def list_archived(self, session_id: str) -> list[RunSnapshot]:
        """列出这个 session 的全部归档快照(latest 之外的历史版本)。
        级联清理(harness/session/cleanup.py)靠它枚举"这个 session
        全部存活快照"——只看 latest 会漏掉 archive 里仍然引用着的
        卸载文件,把它们误判成孤儿删掉。单个归档文件损坏不该拖垮
        整个列表,跳过并记日志(和 FileMemoryStore.list_all 处理
        损坏文件的方式同构)。"""
        safe = _sanitize_session_id(session_id)
        archive_dir = (self.base_dir / safe / "archive").resolve()
        base_resolved = self.base_dir.resolve()
        if not archive_dir.is_relative_to(base_resolved):
            raise ValueError(f"非法 session_id(路径穿越): {session_id}")
        if not archive_dir.is_dir():
            return []
        snapshots = []
        for p in sorted(archive_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                snapshots.append(RunSnapshot.from_dict(data))
            except Exception as e:
                logger.warning(f"[FileSnapshotStore] 跳过无法解析的归档快照 {p}: {e}")
        return snapshots

    def purge(self, session_id: str) -> bool:
        """删除这个 session 的全部快照(latest.json + archive/ 整个
        目录)。是"删除用户全部数据"这类合规需求的原语——路径安全
        校验(resolve + is_relative_to)统一放在这里,而不是让调用方
        (级联清理)自己重新实现一遍对 session_id 的清洗逻辑,目录
        结构的知识只应该在这个类里存在一份。

        返回是否真的删除了什么;session 目录本来就不存在时返回
        False,不当作错误——"删除一个不存在的东西"本来就该是幂等
        的空操作。
        """
        safe = _sanitize_session_id(session_id)
        session_dir = (self.base_dir / safe).resolve()
        base_resolved = self.base_dir.resolve()
        if not session_dir.is_relative_to(base_resolved):
            raise ValueError(f"非法 session_id(路径穿越): {session_id}")
        if not session_dir.is_dir():
            return False
        shutil.rmtree(session_dir)
        logger.info(f"[FileSnapshotStore] purged session={session_id}")
        return True