# harness/tracing/storage_sqlite.py
"""
存了什么、哪些字段可能敏感:
  traces.user_input / final_reply   —— 完整的任务文本和最终答复
  tool_events.args / result         —— 完整的工具调用参数和返回值
                                        (付款场景下就是收款人/金额/账号)
  llm_calls.output / reasoning      —— 模型输出和推理过程(截断到 500 字符)
全部字段**明文存储**,没有加密。这和 snapshots/*.json、offload/*.txt
是同一个选择——都是明文,只加密其中一个没有意义。需要脱敏的部署方式
见 Tracer(redact=...) 参数。
"""
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from harness.tracing.models import LLMCall, ToolEvent, Trace, TraceNode
from harness.tracing.storage_base import TraceStorageBase

_DEFAULT_DB_PATH = Path("data/trace.db")

_SCHEMA = """
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
        prompt_tokens     INTEGER,
        completion_tokens INTEGER,
        token_source      TEXT,
        output            TEXT,
        has_tool_calls    INTEGER,
        duration_ms       INTEGER,
        reasoning         TEXT,
        timestamp         TEXT,
        FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
    );

    -- SQLite 不给 FOREIGN KEY 自动建索引(实测确认过:建表后
    -- sqlite_master 里只有主键的 sqlite_autoindex_*)。没有这些索引,
    -- 按 parent_trace_id 找子 trace、按 trace_id 找事件都是全表扫描——
    -- 代价随整库大小线性增长,trace 攒得越多、查审计越慢,恰好和
    -- "长期运行的生产库"这个真实场景相反。
    CREATE INDEX IF NOT EXISTS idx_traces_parent ON traces(parent_trace_id);
    CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
    CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
    CREATE INDEX IF NOT EXISTS idx_tool_events_trace ON tool_events(trace_id);
    CREATE INDEX IF NOT EXISTS idx_llm_calls_trace ON llm_calls(trace_id);
"""


@contextmanager
def _get_conn(db_path: Path):
    """短连接:每次调用新建、用完即关。get_trace/get_children 这类
    偶发的读查询用它就够,不需要为它们维护长连接的复杂度。

    注意:这个函数被 analytics.py 直接 import 使用。WAL 模式是数据库
    文件本身的持久属性(写在文件头里),不是连接级设置——一旦
    SQLiteTraceStorage.__init__ 通过长连接设置过一次 WAL,这里开的
    短连接会自动读到 WAL 模式生效的文件,不需要重复设置。
    """
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
        busy_timeout_ms: SQLite 写锁等待超时。SQLite 写入本质上是串行的
                        (同一时刻只有一个写者),并发 run 抢锁时,
                        默认(5000ms)会等而不是立刻报错;调大适应
                        高并发场景,调小让锁冲突更快暴露出来。

    【本轮:长连接 + WAL,实测数据支撑这个选择】
    20 次 run × 10 事件的逐事件写入基准测试:

        逐事件 + 短连接(每次新建)          37.9ms/run
        逐事件 + 短连接 + WAL              54.0ms/run  ← 单独加 WAL 反而更慢
        逐事件 + 长连接(不加 WAL)          34.8ms/run
        逐事件 + 长连接 + WAL              16.6ms/run  ← 本实现采用

    反直觉的一点:短连接 + WAL 比短连接 + 无 WAL **更慢**——WAL 模式的
    连接建立要额外读 `-wal` 文件、初始化共享内存,这个开销盖过了它
    带来的提交加速。**WAL 只有配长连接才有意义**,以后如果有人想
    "顺手优化性能"给某个短连接场景加 WAL,先看这组数字。

    长连接需要 threading.Lock 保护(SQLite 的 Connection 对象不是
    线程安全的),不用连接池——SQLite 写入本来就是串行化的,池只增加
    复杂度不增加吞吐。

    注意:CREATE TABLE IF NOT EXISTS 不会给已存在的旧表补列。
    开发库直接删掉 db 文件重建即可;如需保留旧数据,手动执行:
        ALTER TABLE llm_calls ADD COLUMN prompt_tokens INTEGER;
        ALTER TABLE llm_calls ADD COLUMN completion_tokens INTEGER;
        ALTER TABLE llm_calls ADD COLUMN token_source TEXT;
        (并把旧的 input_token_count 列数据按需迁移或直接丢弃)
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH, busy_timeout_ms: int = 5000):
        self.db_path = Path(db_path)
        self._busy_timeout_ms = busy_timeout_ms
        self._lock = threading.Lock()
        self._init_db()
        self._conn = self._open_persistent()

    # ── setup ─────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _get_conn(self.db_path) as conn:
            conn.executescript(_SCHEMA)
            # WAL 是文件级持久属性,在这里设一次,后续所有连接
            # (包括本类的长连接、_get_conn 开的短连接)都会读到它。
            conn.execute("PRAGMA journal_mode=WAL")

    def _open_persistent(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return conn

    def close(self) -> None:
        """显式关闭长连接。进程正常退出前调用是良好实践,但不调用
        不是致命问题——每次写操作都立即 commit,没有悬空的未提交事务,
        操作系统会在进程退出时回收文件句柄。"""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── TraceStorageBase:原有方法(短连接,不变) ─────────────────────────

    def save_trace(self, trace: Trace) -> None:
        with _get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?)",
                (trace.trace_id, trace.session_id, trace.user_input,
                 trace.final_reply, trace.total_duration_ms,
                 trace.tool_call_count, trace.llm_call_count,
                 trace.status, trace.parent_trace_id,
                 trace.created_at.isoformat()),
            )
            for e in trace.tool_events:
                conn.execute(
                    "INSERT OR REPLACE INTO tool_events VALUES (?,?,?,?,?,?,?,?)",
                    (e.event_id, e.trace_id, e.tool_name, e.args,
                     e.result, e.status, e.duration_ms, e.timestamp.isoformat()),
                )
            for c in trace.llm_calls:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (c.event_id, c.trace_id, c.prompt_tokens, c.completion_tokens,
                     c.token_source, c.output, int(c.has_tool_calls),
                     c.duration_ms, c.reasoning, c.timestamp.isoformat()),
                )

    def get_traces_by_session(self, session_id: str) -> list[dict]:
        with _get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM traces WHERE session_id=? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Phase 4:读取端(短连接,不变) ────────────────────────────────────

    @staticmethod
    def _row_to_trace(row) -> Trace:
        return Trace(
            trace_id=row["trace_id"], session_id=row["session_id"],
            user_input=row["user_input"], final_reply=row["final_reply"],
            total_duration_ms=row["total_duration_ms"],
            tool_call_count=row["tool_call_count"],
            llm_call_count=row["llm_call_count"], status=row["status"],
            parent_trace_id=row["parent_trace_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_tool_event(row) -> ToolEvent:
        return ToolEvent(
            event_id=row["event_id"], trace_id=row["trace_id"],
            tool_name=row["tool_name"], args=row["args"], result=row["result"],
            status=row["status"], duration_ms=row["duration_ms"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    @staticmethod
    def _row_to_llm_call(row) -> LLMCall:
        return LLMCall(
            event_id=row["event_id"], trace_id=row["trace_id"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            token_source=row["token_source"], output=row["output"],
            has_tool_calls=bool(row["has_tool_calls"]),
            duration_ms=row["duration_ms"], reasoning=row["reasoning"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def _attach_events(self, conn, traces: dict[str, Trace]) -> None:
        if not traces:
            return
        placeholders = ",".join("?" * len(traces))
        ids = list(traces)
        for row in conn.execute(
            f"SELECT * FROM tool_events WHERE trace_id IN ({placeholders}) "
            f"ORDER BY timestamp", ids,
        ):
            traces[row["trace_id"]].tool_events.append(self._row_to_tool_event(row))
        for row in conn.execute(
            f"SELECT * FROM llm_calls WHERE trace_id IN ({placeholders}) "
            f"ORDER BY timestamp", ids,
        ):
            traces[row["trace_id"]].llm_calls.append(self._row_to_llm_call(row))

    def get_trace(self, trace_id: str) -> Trace | None:
        with _get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM traces WHERE trace_id=?", (trace_id,)
            ).fetchone()
            if row is None:
                return None
            trace = self._row_to_trace(row)
            self._attach_events(conn, {trace_id: trace})
            return trace

    def get_children(self, parent_trace_id: str) -> list[Trace]:
        with _get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM traces WHERE parent_trace_id=? ORDER BY created_at",
                (parent_trace_id,),
            ).fetchall()
            traces = {r["trace_id"]: self._row_to_trace(r) for r in rows}
            self._attach_events(conn, traces)
            return [traces[r["trace_id"]] for r in rows]

    def get_trace_tree(self, root_trace_id: str) -> TraceNode | None:
        """覆盖基类的逐层展开,改用递归 CTE 一次查完整棵树(固定 3 次
        查询,与树的深度/节点数无关)。UNION(不是 UNION ALL)天然去重,
        即便磁盘数据因损坏形成环也不会无限展开。"""
        with _get_conn(self.db_path) as conn:
            rows = conn.execute("""
                WITH RECURSIVE subtree(trace_id) AS (
                    SELECT trace_id FROM traces WHERE trace_id = ?
                    UNION
                    SELECT t.trace_id FROM traces t
                    JOIN subtree s ON t.parent_trace_id = s.trace_id
                )
                SELECT * FROM traces
                WHERE trace_id IN (SELECT trace_id FROM subtree)
                ORDER BY created_at
            """, (root_trace_id,)).fetchall()

            if not rows:
                return None

            traces = {r["trace_id"]: self._row_to_trace(r) for r in rows}
            self._attach_events(conn, traces)

        nodes = {tid: TraceNode(trace=t, children=[]) for tid, t in traces.items()}
        root: TraceNode | None = None
        for tid, node in nodes.items():
            parent_id = node.trace.parent_trace_id
            if tid == root_trace_id:
                root = node
            elif parent_id is not None and parent_id in nodes:
                nodes[parent_id].children.append(node)
        return root

    # ── 本轮:持久化改造(长连接 + 锁) ────────────────────────────────────

    def begin_trace(self, trace: Trace) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?)",
                (trace.trace_id, trace.session_id, trace.user_input,
                 trace.final_reply, trace.total_duration_ms,
                 trace.tool_call_count, trace.llm_call_count,
                 trace.status, trace.parent_trace_id,
                 trace.created_at.isoformat()),
            )
            self._conn.commit()

    def record_event(self, trace_id: str, event) -> None:
        with self._lock:
            if isinstance(event, ToolEvent):
                self._conn.execute(
                    "INSERT OR REPLACE INTO tool_events VALUES (?,?,?,?,?,?,?,?)",
                    (event.event_id, event.trace_id, event.tool_name, event.args,
                     event.result, event.status, event.duration_ms,
                     event.timestamp.isoformat()),
                )
            elif isinstance(event, LLMCall):
                self._conn.execute(
                    "INSERT OR REPLACE INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (event.event_id, event.trace_id, event.prompt_tokens,
                     event.completion_tokens, event.token_source, event.output,
                     int(event.has_tool_calls), event.duration_ms,
                     event.reasoning, event.timestamp.isoformat()),
                )
            else:
                raise TypeError(f"record_event: 未知事件类型 {type(event)!r}")
            self._conn.commit()

    def finish_trace(self, trace: Trace) -> None:
        """用 UPDATE 不用 INSERT OR REPLACE——这条 trace 理应已经被
        begin_trace 写过一次了(durable 模式下),UPDATE 准确表达"这是
        对已有记录的收尾"。如果 begin_trace 因为某种原因没能成功执行,
        UPDATE 会是 0 行受影响的 no-op,不会静默造出一条缺失
        session_id/user_input/created_at 的残缺记录。"""
        with self._lock:
            self._conn.execute(
                "UPDATE traces SET final_reply=?, total_duration_ms=?, "
                "tool_call_count=?, llm_call_count=?, status=? WHERE trace_id=?",
                (trace.final_reply, trace.total_duration_ms,
                 trace.tool_call_count, trace.llm_call_count,
                 trace.status, trace.trace_id),
            )
            self._conn.commit()

    def delete_trace(self, trace_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM traces WHERE trace_id=?", (trace_id,))
            self._conn.execute("DELETE FROM tool_events WHERE trace_id=?", (trace_id,))
            self._conn.execute("DELETE FROM llm_calls WHERE trace_id=?", (trace_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def get_unfinished_traces(self, older_than: datetime | None = None) -> list[Trace]:
        with _get_conn(self.db_path) as conn:
            if older_than is not None:
                rows = conn.execute(
                    "SELECT * FROM traces WHERE status='running' AND created_at < ? "
                    "ORDER BY created_at",
                    (older_than.isoformat(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM traces WHERE status='running' ORDER BY created_at"
                ).fetchall()
            traces = {r["trace_id"]: self._row_to_trace(r) for r in rows}
            self._attach_events(conn, traces)
            return [traces[r["trace_id"]] for r in rows]