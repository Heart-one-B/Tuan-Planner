# harness/tracing/storage_sqlite.py
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from harness.tracing.models import Trace
from harness.tracing.storage_base import TraceStorageBase

_DEFAULT_DB_PATH = Path("data/trace.db")


@contextmanager
def _get_conn(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class SQLiteTraceStorage(TraceStorageBase):
    """SQLite-backed trace storage.

    Args:
        db_path: Path to the SQLite file.
                 Parent directories and the file are created if absent.

    注意:CREATE TABLE IF NOT EXISTS 不会给已存在的旧表补列。
    开发库直接删掉 db 文件重建即可;如需保留旧数据,手动执行:
        ALTER TABLE traces ADD COLUMN parent_trace_id TEXT;
        ALTER TABLE llm_calls ADD COLUMN reasoning TEXT;
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _get_conn(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id          TEXT PRIMARY KEY,
                    session_id        TEXT,
                    user_input        TEXT,
                    final_reply       TEXT,
                    total_duration_ms INTEGER,
                    tool_call_count   INTEGER,
                    llm_call_count    INTEGER,
                    status            TEXT,
                    parent_trace_id   TEXT,
                    created_at        TEXT
                );

                CREATE TABLE IF NOT EXISTS tool_events (
                    event_id    TEXT PRIMARY KEY,
                    trace_id    TEXT,
                    tool_name   TEXT,
                    args        TEXT,
                    result      TEXT,
                    status      TEXT,
                    duration_ms INTEGER,
                    timestamp   TEXT,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
                );

                CREATE TABLE IF NOT EXISTS llm_calls (
                    event_id          TEXT PRIMARY KEY,
                    trace_id          TEXT,
                    input_token_count INTEGER,
                    output            TEXT,
                    has_tool_calls    INTEGER,
                    duration_ms       INTEGER,
                    reasoning         TEXT,
                    timestamp         TEXT,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
                );
            """)

    # ── TraceStorageBase ──────────────────────────────────────────────────────

    def save_trace(self, trace: Trace) -> None:
        with _get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    trace.trace_id, trace.session_id, trace.user_input,
                    trace.final_reply, trace.total_duration_ms,
                    trace.tool_call_count, trace.llm_call_count,
                    trace.status, trace.parent_trace_id,
                    trace.created_at.isoformat(),
                ),
            )
            for e in trace.tool_events:
                conn.execute(
                    "INSERT OR REPLACE INTO tool_events VALUES (?,?,?,?,?,?,?,?)",
                    (
                        e.event_id, e.trace_id, e.tool_name, e.args,
                        e.result, e.status, e.duration_ms, e.timestamp.isoformat(),
                    ),
                )
            for c in trace.llm_calls:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_calls VALUES (?,?,?,?,?,?,?,?)",
                    (
                        c.event_id, c.trace_id, c.input_token_count,
                        c.output, int(c.has_tool_calls),
                        c.duration_ms, c.reasoning, c.timestamp.isoformat(),
                    ),
                )

    def get_traces_by_session(self, session_id: str) -> list[dict]:
        with _get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM traces WHERE session_id=? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]