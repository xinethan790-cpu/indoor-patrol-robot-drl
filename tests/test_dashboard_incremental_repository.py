from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from indoor_patrol_rl.application import (
    PatrolLiveOutcome,
    PatrolLiveReadings,
    PatrolLiveSnapshot,
    PatrolRuntimeIdentity,
    PatrolTaskState,
)
from indoor_patrol_rl.dashboard import DashboardEvidenceError, PatrolDashboardRepository


def _snapshot(sequence: int) -> PatrolLiveSnapshot:
    return PatrolLiveSnapshot(
        run_id="incremental-run",
        sequence=sequence,
        state=PatrolTaskState.NAVIGATING,
        outcome=PatrolLiveOutcome.RUNNING,
        seed=7000,
        control_step=sequence,
        simulation_time_s=sequence * 0.128,
        position=(float(sequence) / 10_000.0, 0.0),
        current_checkpoint_id="P1",
        current_checkpoint_ordinal=1,
        completed_checkpoint_count=0,
        checkpoint_count=4,
        progress_ratio=0.0,
        local_goal=(1.0, 0.0),
        remaining_distance_m=max(0.0, 1.0 - sequence / 10_000.0),
        latest_readings=PatrolLiveReadings(),
        active_alerts=(),
        runtime_identity=PatrolRuntimeIdentity(
            model_id="incremental-model",
            model_sha256="a" * 64,
            safety_shield_enabled=False,
        ),
    )


def _encoded(snapshot: PatrolLiveSnapshot) -> bytes:
    return json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_status(run_directory: Path, snapshot: PatrolLiveSnapshot) -> None:
    (run_directory / "status.json").write_bytes(_encoded(snapshot))


def _write_stream(run_directory: Path, snapshots: list[PatrolLiveSnapshot]) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "events.jsonl").write_bytes(
        b"".join(_encoded(snapshot) + b"\n" for snapshot in snapshots)
    )
    _write_status(run_directory, snapshots[-1])


def _append_event(path: Path, snapshot: PatrolLiveSnapshot) -> None:
    with path.open("ab") as stream:
        stream.write(_encoded(snapshot) + b"\n")


def test_incremental_append_is_equivalent_to_fresh_full_read(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    snapshots = [_snapshot(0)]
    _write_stream(run_directory, snapshots)
    repository = PatrolDashboardRepository(run_directory)
    repository.read()

    for sequence in range(1, 8):
        snapshot = _snapshot(sequence)
        snapshots.append(snapshot)
        _append_event(run_directory / "events.jsonl", snapshot)
        _write_status(run_directory, snapshot)
        incremental = repository.read()

    fresh = PatrolDashboardRepository(run_directory).read()
    assert incremental == fresh
    assert incremental.snapshots == tuple(snapshots)
    assert repository.event_stream_diagnostics.parse_attempts == len(snapshots)


def test_partial_record_is_staged_until_newline_and_status_stays_fail_closed(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    first = _snapshot(0)
    second = _snapshot(1)
    _write_stream(run_directory, [first])
    repository = PatrolDashboardRepository(run_directory)
    assert repository.read().snapshots == (first,)

    encoded = _encoded(second)
    split_at = len(encoded) // 2
    with (run_directory / "events.jsonl").open("ab") as stream:
        stream.write(encoded[:split_at])
    _write_status(run_directory, second)

    with pytest.raises(DashboardEvidenceError, match="超前"):
        repository.read()
    assert repository.event_stream_diagnostics.parse_attempts == 1

    with (run_directory / "events.jsonl").open("ab") as stream:
        stream.write(encoded[split_at:] + b"\n")
    evidence = repository.read()
    assert evidence.current == second
    assert evidence.snapshots == (first, second)
    assert repository.event_stream_diagnostics.parse_attempts == 2


def test_truncate_rebuilds_cache_from_current_file(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    original = [_snapshot(0), _snapshot(1), _snapshot(2)]
    _write_stream(run_directory, original)
    repository = PatrolDashboardRepository(run_directory)
    assert repository.read().snapshots == tuple(original)
    events_path = run_directory / "events.jsonl"
    inode = events_path.stat().st_ino

    replacement = _snapshot(10)
    events_path.write_bytes(_encoded(replacement) + b"\n")
    _write_status(run_directory, replacement)

    assert events_path.stat().st_ino == inode
    assert repository.read().snapshots == (replacement,)
    assert repository.event_stream_diagnostics.cache_rebuilds == 2


def test_inode_replace_rebuilds_cache_even_when_size_does_not_shrink(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    original = _snapshot(0)
    _write_stream(run_directory, [original])
    repository = PatrolDashboardRepository(run_directory)
    assert repository.read().snapshots == (original,)
    events_path = run_directory / "events.jsonl"
    old_identity = (events_path.stat().st_dev, events_path.stat().st_ino)

    replacement = _snapshot(100)
    temporary = run_directory / "replacement.jsonl"
    temporary.write_bytes(_encoded(replacement) + b"\n")
    os.replace(temporary, events_path)
    _write_status(run_directory, replacement)

    assert (events_path.stat().st_dev, events_path.stat().st_ino) != old_identity
    assert repository.read().snapshots == (replacement,)
    assert repository.event_stream_diagnostics.cache_rebuilds == 2


def test_invalid_complete_append_is_never_skipped(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    snapshot = _snapshot(0)
    _write_stream(run_directory, [snapshot])
    repository = PatrolDashboardRepository(run_directory)
    assert repository.read().snapshots == (snapshot,)
    events_path = run_directory / "events.jsonl"
    valid_size = events_path.stat().st_size

    with events_path.open("ab") as stream:
        stream.write(b"{not valid json}\n")
    with pytest.raises(DashboardEvidenceError, match="Invalid JSON"):
        repository.read()
    with pytest.raises(DashboardEvidenceError, match="Invalid JSON"):
        repository.read()

    with events_path.open("r+b") as stream:
        stream.truncate(valid_size)
    assert repository.read().snapshots == (snapshot,)


def test_3600_frame_stream_is_not_read_or_parsed_again_when_unchanged(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    snapshots = [_snapshot(sequence) for sequence in range(3600)]
    _write_stream(run_directory, snapshots)
    repository = PatrolDashboardRepository(run_directory)

    evidence = repository.read()
    assert len(evidence.snapshots) == 3600
    initial = repository.event_stream_diagnostics
    assert initial.data_read_operations == 1
    assert initial.parse_attempts == 3600

    for _ in range(5):
        assert repository.read().current.sequence == 3599
    assert repository.event_stream_diagnostics == initial
