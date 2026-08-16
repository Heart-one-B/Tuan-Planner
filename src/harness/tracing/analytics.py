# harness/tracing/analytics.py
"""
【Phase 4 修订】Span 树读取端做出来之后,顺带照出了这里一个真实的
统计口径问题,一并修掉。

问题:提取子 Agent、召回选择器、压缩摘要器各自是独立的 trace(靠
parent_trace_id 挂在主 run 下面)。而原来的聚合是**全表扁平统计**,
把这些子 trace 当成了独立的用户请求。

实测的例子:1 次用户请求,顶层用了 6 个工具,下面挂 3 个各用 1 个
工具的子 Agent —— avg_tool_calls 报 2.25,因为它算的是
(6+1+1+1)/4。而"平均每次请求用几个工具"的正确答案不可能是 2.25。
slowest_traces 有同样的毛病:子 trace 和顶层 trace 混在同一张榜上。

修法:凡是"每次请求"口径的指标,只统计顶层 trace
(parent_trace_id IS NULL)。这**改变了现有数字**——如果你有依赖
这些数字的看板,换上来之后曲线会跳变。这是刻意的:让一个已知错误的
指标继续当默认值,比一次可见的跳变更糟。需要旧口径的话
传 top_level_only=False。

工具级指标(tool_avg_duration_ms / tool_error_rate)不受影响:
它们统计的是"某个工具的表现",子 Agent 调的工具也是真实调用,
本来就该算进去。
"""
from harness.tracing.models import TraceNode
from harness.tracing.storage_sqlite import SQLiteTraceStorage, _get_conn


def avg_tool_calls(storage: SQLiteTraceStorage, top_level_only: bool = True) -> float:
    """平均每次请求用了几个工具。

    top_level_only=True(默认):只统计顶层 trace,不把子 Agent 的 trace
    当成独立请求。见模块头说明。
    """
    where = "status='success'"
    if top_level_only:
        where += " AND parent_trace_id IS NULL"
    with _get_conn(storage.db_path) as conn:
        row = conn.execute(f"SELECT AVG(tool_call_count) FROM traces WHERE {where}").fetchone()
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


def slowest_traces(storage: SQLiteTraceStorage, n: int = 10,
                  top_level_only: bool = True) -> list[dict]:
    """最慢的 N 次请求。默认只看顶层——子 trace 混进榜单会让人误以为
    "某次提取"是一次独立的慢请求,而它其实只是某次慢请求的一部分。"""
    where = "WHERE parent_trace_id IS NULL" if top_level_only else ""
    with _get_conn(storage.db_path) as conn:
        rows = conn.execute(
            f"SELECT trace_id, user_input, total_duration_ms, tool_call_count "
            f"FROM traces {where} ORDER BY total_duration_ms DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Phase 4 新增:树感知的聚合(没有树查询之前根本做不到) ──────────────

def request_cost(storage: SQLiteTraceStorage, root_trace_id: str) -> dict:
    """一次用户请求的**真实**成本:整棵树的合计,含全部嵌套子 Agent。

    这是没有树查询之前根本回答不了的问题。只看顶层 trace 会系统性
    低估——提取子 Agent、召回选择器、压缩摘要器烧掉的 token 不出现在
    用户可见的对话里,但是真金白银。
    """
    tree: TraceNode | None = storage.get_trace_tree(root_trace_id)
    if tree is None:
        return {}
    prompt, completion = tree.total_tokens()
    return {
        "trace_id": root_trace_id,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "trace_count": len(tree.flatten()),      # 含自己 + 全部子 trace
        "tool_call_count": len(tree.all_tool_events()),
        "depth": tree.depth(),
        "duration_ms": tree.trace.total_duration_ms,
    }


def session_cost(storage: SQLiteTraceStorage, session_id: str) -> dict:
    """一个会话的累计成本。逐个顶层 trace 取整棵树再合计——
    不能直接 SUM(traces) 了事,那样会把同一次请求的子 trace 重复计入
    (虽然结果碰巧一样,但语义上是"把子 trace 当独立请求"的同一个错误,
    换个场景就会出问题,比如统计"请求次数")。
    """
    with _get_conn(storage.db_path) as conn:
        roots = [r[0] for r in conn.execute(
            "SELECT trace_id FROM traces "
            "WHERE session_id=? AND parent_trace_id IS NULL ORDER BY created_at",
            (session_id,),
        )]
    costs = [request_cost(storage, tid) for tid in roots]
    costs = [c for c in costs if c]
    return {
        "session_id": session_id,
        "request_count": len(costs),
        "total_tokens": sum(c["total_tokens"] for c in costs),
        "prompt_tokens": sum(c["prompt_tokens"] for c in costs),
        "completion_tokens": sum(c["completion_tokens"] for c in costs),
        "tool_call_count": sum(c["tool_call_count"] for c in costs),
        "requests": costs,
    }


def summary(storage: SQLiteTraceStorage) -> dict:
    return {
        "avg_tool_calls":      avg_tool_calls(storage),
        "tool_avg_duration_ms": tool_avg_duration_ms(storage),
        "tool_error_rate":     tool_error_rate(storage),
        "slowest_traces":      slowest_traces(storage, n=5),
    }