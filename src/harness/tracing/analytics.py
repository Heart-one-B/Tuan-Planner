from harness.tracing.storage_sqlite import SQLiteTraceStorage, _get_conn


def avg_tool_calls(storage: SQLiteTraceStorage) -> float:
    with _get_conn(storage.db_path) as conn:
        row = conn.execute(
            "SELECT AVG(tool_call_count) FROM traces WHERE status='success'"
        ).fetchone()
        return round(row[0] or 0, 2)


def tool_avg_duration_ms(storage: SQLiteTraceStorage) -> dict[str, int]:
    with _get_conn(storage.db_path) as conn:
        rows = conn.execute(
            "SELECT tool_name, AVG(duration_ms) FROM tool_events GROUP BY tool_name"
        ).fetchall()
        return {r[0]: int(r[1]) for r in rows}


def tool_error_rate(storage: SQLiteTraceStorage) -> dict[str, float]:
    with _get_conn(storage.db_path) as conn:
        rows = conn.execute("""
            SELECT tool_name,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS errors
            FROM tool_events
            GROUP BY tool_name
        """).fetchall()
        return {r[0]: round(r[2] / r[1], 2) for r in rows}


def slowest_traces(storage: SQLiteTraceStorage, n: int = 10) -> list[dict]:
    with _get_conn(storage.db_path) as conn:
        rows = conn.execute(
            "SELECT trace_id, user_input, total_duration_ms, tool_call_count "
            "FROM traces ORDER BY total_duration_ms DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]


def summary(storage: SQLiteTraceStorage) -> dict:
    return {
        "avg_tool_calls":      avg_tool_calls(storage),
        "tool_avg_duration_ms": tool_avg_duration_ms(storage),
        "tool_error_rate":     tool_error_rate(storage),
        "slowest_traces":      slowest_traces(storage, n=5),
    }