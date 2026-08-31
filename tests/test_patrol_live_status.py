from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

import indoor_patrol_rl.application.live_status as live_status_module
from indoor_patrol_rl.application import (
    PatrolLiveAlert,
    PatrolLiveOutcome,
    PatrolLiveReadings,
    PatrolLiveSnapshot,
    PatrolLiveStatusWriter,
    PatrolRuntimeIdentity,
    PatrolTaskState,
    read_patrol_live_events,
    read_patrol_live_status,
)


def _runtime_identity() -> PatrolRuntimeIdentity:
    return PatrolRuntimeIdentity(
        model_id="r2b-targeted-step51200",
        model_sha256="a" * 64,
        safety_shield_enabled=True,
        safety_shield_id="front-brake-v1",
        safety_shield_sha256="b" * 64,
    )


def _snapshot(
    *,
    sequence: int = 0,
    control_step: int = 0,
    simulation_time_s: float = 0.0,
    run_id: str = "patrol-run-001",
) -> PatrolLiveSnapshot:
    return PatrolLiveSnapshot(
        run_id=run_id,
        sequence=sequence,
        state=PatrolTaskState.NAVIGATING,
        outcome=PatrolLiveOutcome.RUNNING,
        seed=11,
        control_step=control_step,
        simulation_time_s=simulation_time_s,
        position=(-2.0 + sequence * 0.1, 0.0),
        current_checkpoint_id="P1",
        current_checkpoint_ordinal=1,
        completed_checkpoint_count=0,
        checkpoint_count=4,
        progress_ratio=0.0,
        local_goal=(-1.0, 0.0),
        remaining_distance_m=max(0.0, 1.0 - sequence * 0.1),
        latest_readings=PatrolLiveReadings(),
        active_alerts=(),
        runtime_identity=_runtime_identity(),
    )


def test_writer_atomically_replaces_status_and_appends_one_json_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def record_replace(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        assert source_path.parent == tmp_path
        assert source_path.name.startswith(".status-")
        replace_calls.append((source_path, target_path))
        real_replace(source, target)

    monkeypatch.setattr(live_status_module.os, "replace", record_replace)
    writer = PatrolLiveStatusWriter(tmp_path, run_id="patrol-run-001")
    first = _snapshot()
    second = _snapshot(sequence=1, control_step=4, simulation_time_s=0.128)

    writer.publish(first)
    writer.publish(second)

    assert [target for _, target in replace_calls] == [
        tmp_path / "status.json",
        tmp_path / "status.json",
    ]
    assert read_patrol_live_status(tmp_path) == second
    assert not tuple(tmp_path.glob(".status-*.tmp"))

    event_lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 2
    assert all(isinstance(json.loads(line), dict) for line in event_lines)
    assert [item.sequence for item in read_patrol_live_events(tmp_path)] == [0, 1]


def test_writer_rejects_run_mismatch_and_non_monotonic_updates(tmp_path: Path) -> None:
    writer = PatrolLiveStatusWriter(tmp_path, run_id="patrol-run-001")
    writer.publish(_snapshot(sequence=3, control_step=30, simulation_time_s=1.0))

    with pytest.raises(ValueError, match="does not match"):
        writer.publish(
            _snapshot(
                sequence=4,
                control_step=31,
                simulation_time_s=1.1,
                run_id="another-run",
            )
        )
    with pytest.raises(ValueError, match="sequence must increase"):
        writer.publish(_snapshot(sequence=3, control_step=31, simulation_time_s=1.1))
    with pytest.raises(ValueError, match="control_step must be monotonic"):
        writer.publish(_snapshot(sequence=4, control_step=29, simulation_time_s=1.1))
    with pytest.raises(ValueError, match="simulation_time_s must be monotonic"):
        writer.publish(_snapshot(sequence=4, control_step=31, simulation_time_s=0.9))

    assert [item.sequence for item in read_patrol_live_events(tmp_path)] == [3]
    assert read_patrol_live_status(tmp_path).sequence == 3


def test_writer_rejects_runtime_identity_or_invalid_progress_changes(tmp_path: Path) -> None:
    first = _snapshot()

    identity_writer = PatrolLiveStatusWriter(tmp_path / "identity", run_id=first.run_id)
    identity_writer.publish(first)
    changed_identity = replace(
        _snapshot(sequence=1, control_step=1, simulation_time_s=0.1),
        runtime_identity=replace(_runtime_identity(), model_sha256="c" * 64),
    )
    with pytest.raises(ValueError, match="runtime_identity cannot change"):
        identity_writer.publish(changed_identity)

    progress_writer = PatrolLiveStatusWriter(tmp_path / "progress", run_id=first.run_id)
    progress_writer.publish(first)
    skipped_checkpoint = replace(
        _snapshot(sequence=1, control_step=1, simulation_time_s=0.1),
        current_checkpoint_id="P3",
        current_checkpoint_ordinal=3,
        completed_checkpoint_count=2,
        progress_ratio=0.5,
    )
    with pytest.raises(ValueError, match="advance by at most one"):
        progress_writer.publish(skipped_checkpoint)

    changed_checkpoint = replace(
        _snapshot(sequence=1, control_step=1, simulation_time_s=0.1),
        current_checkpoint_id="P2",
    )
    with pytest.raises(ValueError, match="cannot change before progress advances"):
        progress_writer.publish(changed_checkpoint)


def test_writer_rejects_updates_after_a_terminal_snapshot(tmp_path: Path) -> None:
    terminal = replace(
        _snapshot(),
        state=PatrolTaskState.FAILED,
        outcome=PatrolLiveOutcome.ERROR,
        local_goal=None,
        remaining_distance_m=None,
    )
    writer = PatrolLiveStatusWriter(tmp_path, run_id=terminal.run_id)
    writer.publish(terminal)

    with pytest.raises(ValueError, match="terminal live status"):
        writer.publish(_snapshot(sequence=1, control_step=1, simulation_time_s=0.1))


def test_safety_stop_is_a_replayable_failure_outcome() -> None:
    terminal = replace(
        _snapshot(),
        state=PatrolTaskState.FAILED,
        outcome=PatrolLiveOutcome.SAFETY_STOP,
        local_goal=None,
        remaining_distance_m=None,
    )

    assert PatrolLiveSnapshot.from_dict(terminal.to_dict()) == terminal


def test_writer_resumes_existing_stream_and_keeps_event_order(tmp_path: Path) -> None:
    first_writer = PatrolLiveStatusWriter(tmp_path, run_id="patrol-run-001")
    first_writer.publish(_snapshot(sequence=7, control_step=70, simulation_time_s=2.0))

    resumed_writer = PatrolLiveStatusWriter(tmp_path, run_id="patrol-run-001")
    assert resumed_writer.last_snapshot == _snapshot(
        sequence=7, control_step=70, simulation_time_s=2.0
    )
    resumed_writer.publish(_snapshot(sequence=9, control_step=72, simulation_time_s=2.1))

    events = read_patrol_live_events(tmp_path)
    assert [item.sequence for item in events] == [7, 9]
    assert [item.simulation_time_s for item in events] == [2.0, 2.1]


def test_writer_retries_a_transient_status_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace
    attempts = 0

    def fail_once(source: str | Path, target: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient replace failure")
        real_replace(source, target)

    monkeypatch.setattr(live_status_module.os, "replace", fail_once)
    snapshot = _snapshot(sequence=1, control_step=4, simulation_time_s=0.128)
    PatrolLiveStatusWriter(tmp_path, run_id=snapshot.run_id).publish(snapshot)

    assert attempts == 2
    assert read_patrol_live_status(tmp_path) == snapshot
    assert read_patrol_live_events(tmp_path) == (snapshot,)


def test_writer_repairs_status_from_a_committed_event_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = PatrolLiveStatusWriter(tmp_path, run_id="patrol-run-001")
    first = _snapshot(sequence=1, control_step=4, simulation_time_s=0.128)
    second = _snapshot(sequence=2, control_step=8, simulation_time_s=0.256)
    writer.publish(first)

    real_replace = os.replace

    def fail_replace(_source: str | Path, _target: str | Path) -> None:
        raise OSError("persistent replace failure")

    monkeypatch.setattr(live_status_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="persistent"):
        writer.publish(second)
    assert read_patrol_live_status(tmp_path) == first
    assert read_patrol_live_events(tmp_path) == (first, second)

    monkeypatch.setattr(live_status_module.os, "replace", real_replace)
    resumed = PatrolLiveStatusWriter(tmp_path, run_id="patrol-run-001")
    assert resumed.last_snapshot == second
    assert read_patrol_live_status(tmp_path) == second
    assert not tuple(tmp_path.glob(".status-*.tmp"))


def test_complete_readings_alert_and_runtime_identity_round_trip(tmp_path: Path) -> None:
    readings = PatrolLiveReadings(
        temperature_c=27.4,
        smoke_obscuration_pct_per_m=4.6,
        carbon_monoxide_ppm=42.0,
        sampled_at_s=4.0,
        checkpoint_id="P2",
    )
    alert = PatrolLiveAlert(
        rule_id="co-warning",
        sensor_id="carbon_monoxide",
        severity="warning",
        observed_value=42.0,
        threshold=35.0,
        unit="ppm",
        message="Carbon monoxide warning",
        raised_at_s=4.0,
    )
    snapshot = PatrolLiveSnapshot(
        run_id="patrol-run-001",
        sequence=8,
        state=PatrolTaskState.DWELLING,
        outcome=PatrolLiveOutcome.RUNNING,
        seed=11,
        control_step=125,
        simulation_time_s=4.0,
        position=(1.0, -1.0),
        current_checkpoint_id="P3",
        current_checkpoint_ordinal=3,
        completed_checkpoint_count=2,
        checkpoint_count=4,
        progress_ratio=0.5,
        local_goal=None,
        remaining_distance_m=None,
        latest_readings=readings,
        active_alerts=(alert,),
        runtime_identity=_runtime_identity(),
    )
    PatrolLiveStatusWriter(tmp_path, run_id=snapshot.run_id).publish(snapshot)

    restored = read_patrol_live_status(tmp_path / "status.json")
    assert restored == snapshot
    assert restored.latest_readings.carbon_monoxide_ppm == 42.0
    assert restored.active_alerts == (alert,)
    assert restored.runtime_identity.safety_shield_enabled


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("position", (math.nan, 0.0), "finite"),
        ("simulation_time_s", math.inf, "finite"),
        ("progress_ratio", 1.1, "within"),
        ("progress_ratio", 0.25, "must equal"),
        ("remaining_distance_m", -0.1, "non-negative"),
    ],
)
def test_snapshot_rejects_nan_and_invalid_progress_or_distance(
    field: str, value: object, message: str
) -> None:
    constructor_arguments = {
        "run_id": "patrol-run-001",
        "sequence": 0,
        "state": PatrolTaskState.NAVIGATING,
        "outcome": PatrolLiveOutcome.RUNNING,
        "seed": 11,
        "control_step": 0,
        "simulation_time_s": 0.0,
        "position": (-2.0, 0.0),
        "current_checkpoint_id": "P1",
        "current_checkpoint_ordinal": 1,
        "completed_checkpoint_count": 0,
        "checkpoint_count": 4,
        "progress_ratio": 0.0,
        "local_goal": (-1.0, 0.0),
        "remaining_distance_m": 1.0,
        "latest_readings": PatrolLiveReadings(),
        "active_alerts": (),
        "runtime_identity": _runtime_identity(),
    }
    constructor_arguments[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        PatrolLiveSnapshot(**constructor_arguments)


def test_readers_reject_unknown_fields_and_out_of_order_jsonl(tmp_path: Path) -> None:
    first = _snapshot(sequence=2, control_step=20, simulation_time_s=1.0)
    unknown = first.to_dict()
    unknown["unexpected"] = True
    (tmp_path / "status.json").write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        read_patrol_live_status(tmp_path)

    later = _snapshot(sequence=3, control_step=21, simulation_time_s=1.1)
    lines = [json.dumps(later.to_dict()), json.dumps(first.to_dict())]
    (tmp_path / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sequence must increase"):
        read_patrol_live_events(tmp_path)


def test_partial_or_non_finite_readings_and_invalid_identity_are_rejected() -> None:
    with pytest.raises(ValueError, match="set together"):
        PatrolLiveReadings(temperature_c=25.0)
    with pytest.raises(ValueError, match="finite"):
        PatrolLiveReadings(
            temperature_c=25.0,
            smoke_obscuration_pct_per_m=math.nan,
            carbon_monoxide_ppm=2.0,
            sampled_at_s=1.0,
            checkpoint_id="P1",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        PatrolRuntimeIdentity(
            model_id="model",
            model_sha256="not-a-digest",
            safety_shield_enabled=False,
        )
    with pytest.raises(ValueError, match="Disabled"):
        PatrolRuntimeIdentity(
            model_id="model",
            model_sha256="a" * 64,
            safety_shield_enabled=False,
            safety_shield_id="unexpected",
        )
