# tests/test_snapshot_migration.py
import pytest

from harness.snapshot.models import (
    RunSnapshot,
    migrate_v1_to_v2,
    migrate_v2_to_v3,
    migrate_v3_to_v4,
)


def _v1_payload() -> dict:
    return {
        "session_id": "s1",
        "task": "do something",
        "trace_id": "t1",
        "status": "completed",
        "rounds": 2,
        "tool_calls_used": 1,
        "messages": [{"role": "user", "content": "hi"}],
        "compaction_history": [],
        "offload_records": [],
        "total_tokens": 100,
        "token_source": "estimated",
        "created_at": "2026-07-01T00:00:00",
        "version": 1,
    }


def test_migrate_v1_to_v2_sets_extraction_defaults():
    migrated = migrate_v1_to_v2(_v1_payload())
    assert migrated["version"] == 2
    assert migrated["extraction_watermark"] is None
    assert migrated["runs_since_extraction"] == 0


def test_migrate_v2_to_v3_sets_surfaced_memories_default():
    v2_payload = migrate_v1_to_v2(_v1_payload())
    migrated = migrate_v2_to_v3(v2_payload)
    assert migrated["version"] == 3
    assert migrated["surfaced_memories"] == []


def test_migrate_v3_to_v4_drops_watermark_and_upgrades_surfaced():
    v3_payload = migrate_v2_to_v3(migrate_v1_to_v2(_v1_payload()))
    v3_payload["extraction_watermark"] = 7
    v3_payload["surfaced_memories"] = ["mem_a", "mem_b"]

    migrated = migrate_v3_to_v4(v3_payload)
    assert migrated["version"] == 4
    assert "extraction_watermark" not in migrated  # 整数下标没有对应物,直接丢弃
    assert migrated["extraction_boundary_hid"] is None
    assert migrated["surfaced_memories"] == [
        {"name": "mem_a", "hid": None}, {"name": "mem_b", "hid": None},
    ]


def test_migrate_v3_to_v4_handles_empty_surfaced():
    v3_payload = migrate_v2_to_v3(migrate_v1_to_v2(_v1_payload()))
    migrated = migrate_v3_to_v4(v3_payload)
    assert migrated["surfaced_memories"] == []


def test_from_dict_chains_v1_all_the_way_to_v4():
    """v1 快照直接 from_dict 应该一路链式迁移到当前版本(v4),
    不需要调用方手动分步迁移。"""
    snap = RunSnapshot.from_dict(_v1_payload())
    assert snap.version == 4
    assert snap.extraction_boundary_hid is None
    assert snap.runs_since_extraction == 0
    assert snap.surfaced_memories == []
    assert snap.session_id == "s1"


def test_from_dict_migrates_v2_to_v4():
    payload = _v1_payload()
    payload["version"] = 2
    payload["extraction_watermark"] = 5
    payload["runs_since_extraction"] = 3
    snap = RunSnapshot.from_dict(payload)
    assert snap.version == 4
    assert snap.extraction_boundary_hid is None  # 旧下标无法换算,如实归零
    assert snap.runs_since_extraction == 3
    assert snap.surfaced_memories == []


def test_from_dict_migrates_v3_to_v4_preserves_names():
    payload = _v1_payload()
    payload["version"] = 3
    payload["extraction_watermark"] = 5
    payload["runs_since_extraction"] = 3
    payload["surfaced_memories"] = ["old_pref"]
    snap = RunSnapshot.from_dict(payload)
    assert snap.version == 4
    assert snap.surfaced_memories == [{"name": "old_pref", "hid": None}]


def test_from_dict_rejects_unknown_future_version():
    payload = _v1_payload()
    payload["version"] = 99
    with pytest.raises(ValueError, match="快照版本不兼容"):
        RunSnapshot.from_dict(payload)


def test_from_dict_accepts_current_v4_directly():
    payload = _v1_payload()
    payload["version"] = 4
    payload["extraction_boundary_hid"] = "abc123"
    payload["runs_since_extraction"] = 3
    payload["surfaced_memories"] = [{"name": "user_pref_a", "hid": "def456"}]
    snap = RunSnapshot.from_dict(payload)
    assert snap.extraction_boundary_hid == "abc123"
    assert snap.runs_since_extraction == 3
    assert snap.surfaced_memories == [{"name": "user_pref_a", "hid": "def456"}]


def test_round_trip_json():
    snap = RunSnapshot(
        session_id="s2", task="t", trace_id="tr", status="completed",
        rounds=1, tool_calls_used=0,
        extraction_boundary_hid="hid-xyz", runs_since_extraction=2,
        surfaced_memories=[{"name": "a", "hid": "h1"}, {"name": "b", "hid": None}],
    )
    import json
    payload = json.loads(snap.to_json())
    restored = RunSnapshot.from_dict(payload)
    assert restored.extraction_boundary_hid == "hid-xyz"
    assert restored.runs_since_extraction == 2
    assert restored.surfaced_memories == [
        {"name": "a", "hid": "h1"}, {"name": "b", "hid": None},
    ]
