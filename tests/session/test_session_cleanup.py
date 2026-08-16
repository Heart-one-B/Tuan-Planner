# tests/test_session_cleanup.py
from pathlib import Path

import pytest

from harness.session.cleanup import cleanup_session, purge_session
from harness.snapshot.models import RunSnapshot
from harness.snapshot.store import FileSnapshotStore


def _snapshot(session_id: str, trace_id: str, refs: list[str] | None = None) -> RunSnapshot:
    """构造一份最小可用的快照,offload_records 里带上指定的引用
    (all_refs() 靠它算存活集合)。"""
    offload_records = [
        {"ref": r, "original_chars": 100, "trace_id": trace_id,
         "tool_call_id": "x", "created_at": "2026-07-01T00:00:00"}
        for r in (refs or [])
    ]
    return RunSnapshot(
        session_id=session_id, task="t", trace_id=trace_id, status="completed",
        rounds=1, tool_calls_used=0, offload_records=offload_records,
    )


def _write_offload_file(offload_dir: Path, trace_id: str, filename: str, content: str = "x") -> Path:
    d = offload_dir / trace_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / filename
    p.write_text(content, encoding="utf-8")
    return p


def test_orphan_file_is_deleted_live_ref_is_kept(tmp_path):
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots", keep_archives=0)
    offload_dir = tmp_path / "offload"

    snapshot_store.save(_snapshot("sess-a", "trace-1", refs=["trace-1/keep.txt"]))
    _write_offload_file(offload_dir, "trace-1", "keep.txt")
    _write_offload_file(offload_dir, "trace-1", "orphan.txt")

    report = cleanup_session("sess-a", snapshot_store, offload_dir)

    assert report.deleted == ["trace-1/orphan.txt"]
    assert (offload_dir / "trace-1" / "keep.txt").is_file()
    assert not (offload_dir / "trace-1" / "orphan.txt").exists()


def test_archived_snapshot_refs_are_also_kept_alive(tmp_path):
    """archive 里的旧快照(不是 latest)引用的文件也不该被删——
    级联清理的判据是"全部存活快照"的并集,不能只看 latest。"""
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots", keep_archives=3)
    offload_dir = tmp_path / "offload"

    # 第一次保存(会同时进 latest 和 archive),引用 old.txt
    snapshot_store.save(_snapshot("sess-a", "trace-1", refs=["trace-1/old.txt"]))
    _write_offload_file(offload_dir, "trace-1", "old.txt")
    # 第二次保存,latest 换成引用 new.txt——old.txt 只在 archive 里还被引用
    snapshot_store.save(_snapshot("sess-a", "trace-1", refs=["trace-1/new.txt"]))
    _write_offload_file(offload_dir, "trace-1", "new.txt")

    report = cleanup_session("sess-a", snapshot_store, offload_dir)

    assert "trace-1/old.txt" not in report.deleted  # archive 还引用着,不该删
    assert "trace-1/new.txt" not in report.deleted
    assert (offload_dir / "trace-1" / "old.txt").is_file()
    assert (offload_dir / "trace-1" / "new.txt").is_file()


def test_multi_session_isolation_never_touches_other_sessions_files(tmp_path):
    """最重要的一条:清理 session A 绝不能碰到 session B 的文件,
    即使它们碰巧共享同一个 offload 根目录(offload 目录是跨 session
    共享的,这是设计使然)。"""
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots", keep_archives=0)
    offload_dir = tmp_path / "offload"

    snapshot_store.save(_snapshot("sess-a", "trace-a", refs=[]))
    snapshot_store.save(_snapshot("sess-b", "trace-b", refs=["trace-b/keep.txt"]))

    _write_offload_file(offload_dir, "trace-a", "a_orphan.txt")
    _write_offload_file(offload_dir, "trace-b", "keep.txt")
    _write_offload_file(offload_dir, "trace-b", "b_orphan.txt")

    report = cleanup_session("sess-a", snapshot_store, offload_dir)

    # 只清了 sess-a 名下(trace-a)的文件
    assert report.scanned_dirs == ["trace-a"]
    assert report.deleted == ["trace-a/a_orphan.txt"]
    # sess-b 的文件完全没被碰过,不管它是不是孤儿
    assert (offload_dir / "trace-b" / "keep.txt").is_file()
    assert (offload_dir / "trace-b" / "b_orphan.txt").is_file()


def test_dry_run_reports_without_deleting(tmp_path):
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots", keep_archives=0)
    offload_dir = tmp_path / "offload"
    snapshot_store.save(_snapshot("sess-a", "trace-1", refs=[]))
    orphan = _write_offload_file(offload_dir, "trace-1", "orphan.txt")

    report = cleanup_session("sess-a", snapshot_store, offload_dir, dry_run=True)

    assert report.deleted == ["trace-1/orphan.txt"]  # 报告里说"会删"
    assert orphan.is_file()  # 但实际文件还在


def test_cleanup_session_with_no_snapshot_is_noop(tmp_path):
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots")
    offload_dir = tmp_path / "offload"
    report = cleanup_session("never-existed", snapshot_store, offload_dir)
    assert report.deleted == []
    assert report.failed == []
    assert report.scanned_dirs == []


def test_cleanup_session_failure_does_not_abort_batch(tmp_path, monkeypatch):
    """单个文件删除失败应该记入 failed,不能让整批清理中断。"""
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots", keep_archives=0)
    offload_dir = tmp_path / "offload"
    snapshot_store.save(_snapshot("sess-a", "trace-1", refs=[]))
    _write_offload_file(offload_dir, "trace-1", "orphan1.txt")
    _write_offload_file(offload_dir, "trace-1", "orphan2.txt")

    original_unlink = Path.unlink
    call_count = {"n": 0}

    def flaky_unlink(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("模拟权限错误")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    report = cleanup_session("sess-a", snapshot_store, offload_dir)

    assert len(report.failed) == 1
    assert len(report.deleted) == 1  # 第二个文件依然被正常删除了


def test_purge_session_deletes_everything_including_live_refs(tmp_path):
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots", keep_archives=0)
    offload_dir = tmp_path / "offload"
    snapshot_store.save(_snapshot("sess-a", "trace-1", refs=["trace-1/live.txt"]))
    _write_offload_file(offload_dir, "trace-1", "live.txt")
    _write_offload_file(offload_dir, "trace-1", "orphan.txt")

    report = purge_session("sess-a", snapshot_store, offload_dir)

    assert set(report.deleted) == {"trace-1/live.txt", "trace-1/orphan.txt"}
    assert not (offload_dir / "trace-1").exists()
    with pytest.raises(FileNotFoundError):
        snapshot_store.load_latest("sess-a")


def test_purge_session_does_not_touch_other_sessions(tmp_path):
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots", keep_archives=0)
    offload_dir = tmp_path / "offload"
    snapshot_store.save(_snapshot("sess-a", "trace-a", refs=[]))
    snapshot_store.save(_snapshot("sess-b", "trace-b", refs=["trace-b/keep.txt"]))
    _write_offload_file(offload_dir, "trace-b", "keep.txt")

    purge_session("sess-a", snapshot_store, offload_dir)

    assert (offload_dir / "trace-b" / "keep.txt").is_file()
    snap_b = snapshot_store.load_latest("sess-b")  # 不抛异常,sess-b 完好无损
    assert snap_b.session_id == "sess-b"


def test_malicious_trace_id_is_neutralized_by_sanitization(tmp_path):
    """快照是磁盘上人手可编辑的 JSON——手工构造一个带路径穿越企图的
    trace_id。第一层防御(_sanitize_trace_id 的白名单清洗)会把 '/'
    替换掉,'../outside' 变成一个不含分隔符的字面量目录名,根本不会
    真的指向 offload_dir 之外——诱饵文件应该完全没被碰到,而且因为
    清洗后的目录本来就不存在,cleanup 会静默跳过它(不产生 failed
    记录,这是符合预期的:清洗已经在更早的一层把恶意输入变成了
    无害字符串,不需要靠后面的 resolve 检查再报一次错)。"""
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots", keep_archives=0)
    offload_dir = tmp_path / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    bait = outside_dir / "bait.txt"
    bait.write_text("不该被碰到")

    malicious_snap = _snapshot("sess-evil", "../outside", refs=[])
    snapshot_store.save(malicious_snap)

    report = cleanup_session("sess-evil", snapshot_store, offload_dir)

    assert bait.is_file()
    assert bait.read_text() == "不该被碰到"
    assert report.deleted == []
    assert report.scanned_dirs == []  # 清洗后的目录不存在,连"扫描过"都算不上


def test_cross_platform_ref_normalization_prevents_false_orphan(tmp_path):
    """审查中发现的真实 bug 回归测试:offload 的 ref 字符串在 Windows
    上是反斜杠分隔的(str(Path(...)) 的行为),如果快照里存的 ref 和
    从磁盘扫描算出来的 ref 格式不一致,字符串比较会失配,把仍被引用
    的文件误判成孤儿删掉——这是数据丢失级别的 bug。这里人为构造一个
    反斜杠格式的 ref(模拟"快照是在 Windows 上生成的"这个场景),
    不管当前测试实际跑在哪个操作系统上,都要确认比较逻辑能正确
    识别它仍然存活。"""
    snapshot_store = FileSnapshotStore(tmp_path / "snapshots", keep_archives=0)
    offload_dir = tmp_path / "offload"

    # 故意用反斜杠格式的 ref,模拟旧数据/Windows 上产生的快照
    snapshot_store.save(_snapshot("sess-a", "trace-1", refs=["trace-1\\keep.txt"]))
    _write_offload_file(offload_dir, "trace-1", "keep.txt")

    report = cleanup_session("sess-a", snapshot_store, offload_dir)

    assert report.deleted == []  # 不该被误判成孤儿
    assert (offload_dir / "trace-1" / "keep.txt").is_file()


def test_normalize_ref_converts_backslash_to_forward_slash():
    from harness.session.cleanup import _normalize_ref

    assert _normalize_ref("trace-1\\keep.txt") == "trace-1/keep.txt"
    assert _normalize_ref("trace-1/keep.txt") == "trace-1/keep.txt"  # 已经是正斜杠,不变


@pytest.mark.parametrize("malicious", [
    "../outside",
    "../../etc/passwd",
    "/etc/passwd",
    "..\\..\\windows",
    "C:\\Windows\\System32",
    "trace/../../../outside",
])
def test_sanitize_trace_id_neutralizes_all_traversal_attempts(malicious):
    """直接对清洗函数做恶意输入穷举:所有路径分隔符(/ 和 \\)、盘符
    冒号都不在白名单里,会被替换成 '_',结果必然是一个不含任何分隔符
    的单一字面量字符串——不管输入里包含多少层 '..',拼接后都不可能
    真正跳出 base_dir。这比"构造一次攻击、断言诱饵文件没事"的集成
    测试更直接地证明了这道防线本身的健壮性,不依赖某一次具体的
    文件系统状态。"""
    from harness.session.cleanup import _sanitize_trace_id

    safe = _sanitize_trace_id(malicious)
    assert "/" not in safe
    assert "\\" not in safe
    assert ":" not in safe