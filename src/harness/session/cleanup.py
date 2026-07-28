# harness/session/cleanup.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.snapshot.store import FileSnapshotStore

logger = logging.getLogger(__name__)

# 与 harness.snapshot.store._sanitize_session_id、
# harness.memory.store._sanitize_name 同构的白名单清洗——这个仓库对
# "从磁盘/快照读回的标识符,拼路径前必须清洗"这条纪律已经贯彻了三次
# (offload 的 tool_call_id、snapshot 的 session_id、memory 的 name),
# 这里是第四次,不是巧合,是同一条原则在第四个"用不可信字符串拼路径"
# 的场景下的应用。这次尤其要紧:trace_id 拼出来的路径要拿去做删除,
# 攻击面从"写到意外位置"升级为"删掉意外位置"。
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_\-.]")


def _sanitize_trace_id(trace_id: str) -> str:
    safe = _UNSAFE_CHARS.sub("_", trace_id)
    return safe or "_"


def _normalize_ref(ref: str) -> str:
    """把 ref 统一成正斜杠(POSIX)形式再参与比较。

    审查中发现的真实跨平台 bug:offload 的 ref 字段最初由
    OffloadStore.save() 用 `str(Path(trace_id) / f"{tool_call_id}.txt")`
    生成——在 Windows 上 `str(Path(...))` 产出反斜杠分隔的字符串
    (如 "trace-1\\keep.txt"),而这里从磁盘扫描目录、用
    `f.relative_to(base_resolved)` 反推 ref 时,同样受平台影响。
    两边分别在各自的时间点、各自的调用栈里生成字符串,如果不统一
    格式,Windows 上会出现"同一个文件、两种字符串表示"的情况——
    字符串比较 `ref in live_refs` 因此失配,把仍被引用的文件误判成
    孤儿删掉,是数据丢失级别的 bug,不是显示层面的小问题。

    刻意用纯字符串替换而不是 `Path(ref).as_posix()`:后者是否把
    反斜杠当分隔符解释,取决于**运行这段代码的机器当前是什么操作
    系统**(WindowsPath 认反斜杠为分隔符,PosixPath 不认,把它当成
    普通文件名字符)——这意味着同一个 ref 字符串,拿去 `Path()` 包一层
    再转换,在 Linux 上跑和在 Windows 上跑会得到不同结果,行为不确定。
    这里要做的是"不管 ref 字符串本身用的是哪种分隔符约定,统一转成
    正斜杠",这是纯字符串层面的语义,不该借助一个行为随平台变化的
    工具来做,直接替换字符更简单也更可预测。这个仓库自己生成的 ref
    不会含字面量反斜杠(trace_id 是 uuid4 hex、tool_call_id 是
    provider 调用 ID,都不含分隔符字符),盲替换是安全的。
    在比较发生的这一处(存活集合和磁盘扫描结果)双向归一化,不依赖
    调用方在写入时就用对格式,对已经用旧格式存在快照里的历史 ref
    同样兼容(存量数据不需要迁移)。"""
    return ref.replace("\\", "/")


@dataclass
class CleanupReport:
    """一次清理的完整审计:扫了什么、判定哪些还活着、删了什么、
    失败了什么。删除是不可逆动作,必须产生可核查的记录——这不是
    日志,是返回值,宿主拿它做告警、统计,或者 dry_run 模式下给人看
    "如果真删会删掉什么"再决定要不要继续。"""
    session_id: str
    dry_run: bool = False
    scanned_dirs: list = field(default_factory=list)     # 实际扫描过的 trace 目录(相对路径)
    live_refs: set = field(default_factory=set)           # 判定为存活、不会被删的引用
    deleted: list = field(default_factory=list)           # 真正删掉(或 dry_run 下"将会删掉")的引用
    failed: list = field(default_factory=list)             # [(ref_or_trace_id, 错误原因), ...]


def _collect_live_trace_ids(
    snapshot_store: FileSnapshotStore, session_id: str,
) -> tuple[list, set]:
    """收集该 session 全部存活快照(latest + archive/*)。返回
    (trace_id 列表, 全部存活快照的 all_refs 并集)。

    这一步是整个级联清理正确性的关键前提:offload 目录是多 session
    共享的(组织形式是 {base_dir}/{trace_id}/...,trace 归属于某个
    session,但目录层级上没有 session 这一段)。如果拿单个 session
    的引用集合去对整个 offload 目录做差集,会把其他 session 还活着
    的文件全部误删——这是数据销毁级的 bug,必须靠"只扫描这个 session
    自己名下的 trace 目录"在结构上杜绝,而不是指望调用方小心。

    latest 不存在(session 从未存过快照,或者只有归档、latest 已被
    别的机制清掉)时只看 archive;两者都没有则返回空,调用方据此得到
    "这个 session 没有任何东西可清"的正确结论,而不是报错。
    """
    trace_ids = []
    live_refs: set = set()
    try:
        latest = snapshot_store.load_latest(session_id)
        trace_ids.append(latest.trace_id)
        live_refs |= {_normalize_ref(r) for r in latest.all_refs()}
    except FileNotFoundError:
        pass
    for snap in snapshot_store.list_archived(session_id):
        trace_ids.append(snap.trace_id)
        live_refs |= {_normalize_ref(r) for r in snap.all_refs()}
    return trace_ids, live_refs


def _clean_trace_dirs(
    trace_ids: list, offload_base_dir: Path, live_refs: set, dry_run: bool,
) -> tuple[list, list, list]:
    """遍历 trace_ids 对应的目录,删除(或 dry_run 下预演删除)不在
    live_refs 里的文件。返回 (scanned_dirs, deleted, failed)——纯函数
    风格抽出来,cleanup_session 和 purge_session 共用这段核心循环,
    差别只在 live_refs 的构造方式(前者是真实并集,后者传空集合
    等价于"全部删光")。"""
    base_resolved = offload_base_dir.resolve()
    scanned_dirs: list = []
    deleted: list = []
    failed: list = []

    for trace_id in trace_ids:
        safe_trace_id = _sanitize_trace_id(trace_id)
        trace_dir = (offload_base_dir / safe_trace_id).resolve()
        if not trace_dir.is_relative_to(base_resolved):
            # trace_id 本身被污染(理论上不该发生,快照是磁盘上人手
            # 可编辑的 JSON,不能假设它一定干净)——拒绝处理这一个
            # 目录,记入失败但不中断其它目录的清理。
            failed.append((trace_id, "非法 trace_id(路径穿越),已跳过"))
            continue
        if not trace_dir.is_dir():
            continue
        scanned_dirs.append(trace_dir.relative_to(base_resolved).as_posix())

        for f in sorted(trace_dir.glob("*.txt")):
            ref = f.relative_to(base_resolved).as_posix()
            if ref in live_refs:
                continue
            if dry_run:
                deleted.append(ref)  # dry_run 下"deleted"语义是"将会被删"
                continue
            try:
                f.unlink()
                deleted.append(ref)
            except OSError as e:
                # 单个文件删不掉(权限、被占用)不中断整批——清理是
                # 尽力而为的维护操作,一个文件卡住不该让整批停摆。
                failed.append((ref, str(e)))

        if not dry_run:
            try:
                if trace_dir.is_dir() and not any(trace_dir.iterdir()):
                    trace_dir.rmdir()
            except OSError as e:
                logger.warning(f"[cleanup] 删空目录失败 {trace_dir}: {e}")

    return scanned_dirs, deleted, failed


def cleanup_session(
    session_id: str,
    snapshot_store: FileSnapshotStore,
    offload_base_dir: Path,
    dry_run: bool = False,
) -> CleanupReport:
    """清理一个 session 的孤儿卸载文件。

    存活判据 = 该 session 全部存活快照(latest + archive/*)的 all_refs
    并集。扫描范围 = 这些快照的 trace_id 对应的目录,绝不触碰其他
    session 的目录(见 _collect_live_trace_ids 的说明)。

    dry_run 不是可选的锦上添花——删除不可逆,宿主第一次在生产数据上
    跑,必须能先看"将要删什么"再决定,这是不可逆操作的标配,不是
    额外功能。

    不做定时器,只做显式调用的宿主 API——调度是宿主的事,和"库不该
    偷偷 spawn 任务"是同一条原则。

    已知边界(如实记账,不假装解决):
    ① 覆盖范围仅限"快照可见的 trace"。提取/召回子 Agent 走独立的
       Span.child(),有自己的 trace_id,但当前设计下它们不配置
       context_config、不产生卸载文件,所以不影响正确性——这里只是
       记一笔:如果将来子 Agent 也配了 context_config,它们的 trace
       目录不会被这里的扫描覆盖到,需要在那时候重新评估。
    ② 被 archive 轮换掉(超过 keep_archives)的旧快照,其独占引用的
       trace 目录会变成"无主目录",不属于任何还存活的 session 判据,
       这里不处理——这是预期行为(快照都没了,为它保文件没有意义),
       但清理这类目录需要"全局扫描 + 时间衰减"的另一套策略(Phase 7
       的 L3 时间衰减清理),第一期只做"按 session 显式触发"这一种,
       不做后台全局清扫,不能覆盖这类无主目录。

    孤儿文件在正常运行中就会自然产生,不是"以防万一"的理论风险:
    紧急压缩(overflow 触发)把中间段换页落盘后,如果这次 run 最终
    异常退出(loop 抛异常),_maybe_snapshot 根本不会执行,刚落盘的
    文件没有任何快照引用它,直接成为孤儿——这个功能因此是常规运行
    的必然结果需要的清扫,不是可选的维护脚本。
    """
    trace_ids, live_refs = _collect_live_trace_ids(snapshot_store, session_id)
    scanned_dirs, deleted, failed = _clean_trace_dirs(
        trace_ids, offload_base_dir, live_refs, dry_run,
    )
    report = CleanupReport(
        session_id=session_id, dry_run=dry_run,
        scanned_dirs=scanned_dirs, live_refs=live_refs,
        deleted=deleted, failed=failed,
    )
    logger.info(f"[cleanup_session] session={session_id} dry_run={dry_run} "
               f"扫描目录={len(scanned_dirs)} 删除={len(deleted)} 失败={len(failed)}")
    return report


def purge_session(
    session_id: str,
    snapshot_store: FileSnapshotStore,
    offload_base_dir: Path,
) -> CleanupReport:
    """整删:该 session 关联的全部卸载文件 + 全部快照(latest + archive)。
    是"删除用户全部数据"这类合规需求的 harness 侧原语,不支持
    dry_run——这个操作本身就是"确认要删"之后才会被调用的,调用方
    如果想预览,应该先调 cleanup_session(dry_run=True) 看孤儿文件,
    再决定是不是要整个 purge。

    顺序刻意:先收集 trace_id 清单(此时快照必须还在,它是找到关联
    文件的唯一线索)、删完 offload 文件和 trace 目录,最后才删快照
    本身——快照必须最后删,一旦先删快照,就再也无法知道这个 session
    名下还有哪些文件需要清理了。

    与 cleanup_session 的关键区别:live_refs 传空集合,等价于"这个
    session 名下扫到的每一个文件都删",因为 purge 的语义就是"这个
    session 的一切都不该再存在",不需要 all_refs() 判活。
    """
    trace_ids, _ = _collect_live_trace_ids(snapshot_store, session_id)
    scanned_dirs, deleted, failed = _clean_trace_dirs(
        trace_ids, offload_base_dir, live_refs=set(), dry_run=False,
    )
    snapshot_store.purge(session_id)
    report = CleanupReport(
        session_id=session_id, dry_run=False,
        scanned_dirs=scanned_dirs, live_refs=set(),
        deleted=deleted, failed=failed,
    )
    logger.info(f"[purge_session] session={session_id} "
               f"扫描目录={len(scanned_dirs)} 删除={len(deleted)} 失败={len(failed)}")
    return report