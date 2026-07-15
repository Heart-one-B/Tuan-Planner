# harness/snapshot/store.py
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Protocol

from harness.snapshot.models import RunSnapshot

logger = logging.getLogger(__name__)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_\-.]")


def _sanitize_session_id(session_id: str) -> str:
    """session_id 可能来自外部输入,直接拼路径前必须清洗。策略与
    OffloadStore 的 tool_call_id 清洗同构:白名单外字符一律替换为
    '_'。这样任何路径分隔符('/','\\')都会被消灭,清洗结果必然是
    不含分隔符的单一路径段,不具备制造目录跳转的能力——即使结果里
    残留字面量 '..' 这样的字符序列,没有分隔符它就只是普通字符,
    没有穿越语义。load_latest 仍然做 resolve+is_relative_to 兜底,
    双保险不依赖单一防线。"""
    safe = _UNSAFE_CHARS.sub("_", session_id)
    return safe or "_"


class SnapshotStore(Protocol):
    def save(self, snapshot: RunSnapshot) -> str: ...
    def load_latest(self, session_id: str) -> RunSnapshot: ...
    def list_sessions(self) -> list[str]: ...


class FileSnapshotStore:
    """快照的文件系统实现:data/snapshots/{session_id}/latest.json
    (+ archive/{timestamp}.json)。

    为什么是 JSON 不是 md:格式由"谁来读它、读完要干什么"决定。快照
    的读者是程序,读完要做逐字节精确重建(消息 role/tool_call id 错一个,
    回放就会产出孤儿 tool 消息、被 API 拒收),这种消费方式要求无损
    往返,JSON 天然满足(dumps/loads 是恒等变换)。记忆文件走 md 是因为
    它的读者是人和模型、按文本理解消费——两类数据、两种格式,道理
    是同一条:读者决定格式。

    原子写是硬要求:快照系统存在的意义就是抵御进程死亡,如果写快照
    写到一半进程死了、还把上一份好快照带坏,它就失败于自身的存在
    理由。做法是写临时文件 + os.replace(同目录内的 rename 在几乎所有
    文件系统上是原子操作),绝不直接覆盖写 latest.json。

    archive 保留最近 keep_archives 份供回滚,keep_archives=0 即纯覆盖
    (不保留历史版本)。
    """

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