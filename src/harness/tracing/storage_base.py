# harness/tracing/storage_base.py
from abc import ABC, abstractmethod
from datetime import datetime

from harness.tracing.models import Trace, TraceNode


class TraceStorageBase(ABC):
    """Abstract persistence layer for traces.

    Implement this to swap backends (SQLite, Postgres, cloud, mock).
    Pass your implementation to configure_storage() at startup.
    The default provided implementation is SQLiteTraceStorage.

    ── Phase 4:Span 树读取端(get_trace/get_children/get_trace_tree) ──
    刻意不是 @abstractmethod,理由见下方对应方法的 docstring。

    ── 本轮:持久化改造(begin_trace/record_event/finish_trace/
    delete_trace/get_unfinished_traces)──

    起因:save_trace() 只在 run **结束时**调用一次。如果进程在结束前
    崩溃(付款类工具已经真实执行,但 end_trace 还没来得及跑),这条
    run 的全部审计记录——包括工具已经做了什么——一条都不会落盘。

    修法是把"记录"拆成三个时间点:run 开始时先写一行"正在跑"
    (begin_trace);每个工具/LLM 调用发生时立即记一笔
    (record_event);run 结束时补上最终状态(finish_trace)。这样
    即使中途崩溃,数据库里也留着"跑到哪一步、做过什么"的真实痕迹,
    而不是彻底的空白。

    同样刻意不是 @abstractmethod。默认实现:begin_trace/record_event
    是 no-op,finish_trace 退回调用 save_trace(trace)——这保证了
    没有实现新方法的后端,行为与改造前逐字节一致(改造前 end_trace
    做的就是调 save_trace,只在结束时发生一次)。
    """

    @abstractmethod
    def save_trace(self, trace: Trace) -> None:
        """Persist a completed trace and all its child events."""
        ...

    @abstractmethod
    def get_traces_by_session(self, session_id: str) -> list[dict]:
        """Return all traces for a session, newest first."""
        ...

    # ── 读取端(Phase 4,可选实现) ──────────────────────────────────────

    def get_trace(self, trace_id: str) -> Trace | None:
        """按 trace_id 取单条完整 trace(含 tool_events / llm_calls)。"""
        raise NotImplementedError(
            f"{type(self).__name__} 没有实现 get_trace(),无法使用 Span 树查询。"
            f"需要的话实现 get_trace / get_children 两个方法即可,"
            f"get_trace_tree 会自动可用。"
        )

    def get_children(self, parent_trace_id: str) -> list[Trace]:
        """取直接子 trace(只一层,不递归)。"""
        raise NotImplementedError(
            f"{type(self).__name__} 没有实现 get_children(),无法使用 Span 树查询。"
        )

    def get_trace_tree(self, root_trace_id: str) -> TraceNode | None:
        """取整棵 Span 树。基类通用实现:靠 get_trace + get_children
        逐层展开,任何实现了那两个原语的后端都能直接用。有环路保护
        (parent_trace_id 是从磁盘读回来的,不信任它天然无环)。"""
        root = self.get_trace(root_trace_id)
        if root is None:
            return None

        visited: set[str] = {root_trace_id}

        def build(trace: Trace) -> TraceNode:
            node = TraceNode(trace=trace, children=[])
            for child in self.get_children(trace.trace_id):
                if child.trace_id in visited:
                    continue
                visited.add(child.trace_id)
                node.children.append(build(child))
            return node

        return build(root)

    # ── 持久化改造(本轮,可选实现) ────────────────────────────────────

    def begin_trace(self, trace: Trace) -> None:
        """durable 模式下,run 开始时立即调用,把 status='running' 的行
        先写进去——即使进程在 finish_trace 之前崩溃,数据库里也留着
        "这次运行开始过、但没正常结束"的证据。

        默认 no-op:没实现的后端行为和改造前完全一致,持久化只发生在
        finish_trace(见下)。
        """
        pass

    def record_event(self, trace_id: str, event) -> None:
        """durable 模式下,每条 ToolEvent/LLMCall 产生时立即持久化,
        不等到整个 run 结束——这是审计数据不因崩溃而丢失的关键一环。

        event 是 harness.tracing.models.ToolEvent 或 LLMCall 的实例,
        用 isinstance 分派,不用额外的 event_type 参数——两种类型
        自身已经带着区分所需的全部信息,不需要第二份冗余标注。

        默认 no-op,行为与改造前一致。
        """
        pass

    def finish_trace(self, trace: Trace) -> None:
        """run 结束时调用,更新最终状态(final_reply/duration/
        tool_call_count/llm_call_count/status)。

        默认实现退回 save_trace(trace)——保证没有实现新方法的后端
        行为与改造前逐字节一致(改造前 end_trace 就是直接调
        save_trace,一次性写完整条记录)。SQLiteTraceStorage 会覆盖
        这个方法,改成对已有行做 UPDATE(因为 durable 模式下这条 trace
        理应已经被 begin_trace 写过一次了)。
        """
        self.save_trace(trace)

    def delete_trace(self, trace_id: str) -> bool:
        """删除一条被 discard 的 trace 的持久化痕迹(如果有)。

        discard 和 end 的语义区别很关键:discard 表示"这条不作数",
        而 durable 模式下 begin_trace 可能已经把它写进库里了——
        不删掉的话,discard 的语义就只在内存里成立,磁盘上还留着
        一条本不该存在的记录。

        默认 no-op、返回 False:非 durable 场景下 discard 本来就只是
        内存操作,没有东西可删。
        """
        return False

    def get_unfinished_traces(self, older_than: datetime | None = None) -> list[Trace]:
        """查询状态仍是 'running' 的孤儿 trace——通常意味着进程在
        run 正常结束前退出了(硬崩、被杀、断电)。

        只提供查询,不提供自动清扫:多进程部署下,A 进程主动清扫会把
        B 进程正在跑的 run 误标成"中断"。谁来处理这些孤儿、什么时候
        处理,是宿主的策略决定,不是 harness 该替它做的机制决定。

        older_than:只返回创建时间早于这个时间点的——避免把"刚
        begin_trace、还在正常执行中"的 run 误当成孤儿。

        默认返回空列表(没实现的后端等价于"看不到任何孤儿",
        而不是报错——这是一个只读的诊断能力,不该因为没实现就崩)。
        """
        return []