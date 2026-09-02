from __future__ import annotations

import json
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from indoor_patrol_rl.application import (
    RUNTIME_ABORT_EVIDENCE_SCOPE,
    RUNTIME_ABORT_FILENAME,
    PatrolLiveOutcome,
    PatrolLiveReadings,
    PatrolLiveSnapshot,
    PatrolLiveStatusWriter,
    PatrolRuntimeAbort,
    PatrolRuntimeIdentity,
    PatrolTaskState,
    write_patrol_runtime_abort,
)
from indoor_patrol_rl.dashboard import (
    DashboardEvidenceError,
    DashboardMap,
    PatrolDashboardRepository,
    project_dashboard,
)
from indoor_patrol_rl.relay import (
    PatrolRelayPublisher,
    RelayAcknowledgement,
    RelayRetryableError,
)

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
MAP_CONFIG = REPOSITORY_ROOT / "configs/maps/patrol_route.yaml"


def _running_snapshot(*, run_id: str = "runtime-abort-run") -> PatrolLiveSnapshot:
    return PatrolLiveSnapshot(
        run_id=run_id,
        sequence=0,
        state=PatrolTaskState.NAVIGATING,
        outcome=PatrolLiveOutcome.RUNNING,
        seed=7000,
        control_step=0,
        simulation_time_s=0.0,
        position=(-1.6, 0.0),
        current_checkpoint_id="P1",
        current_checkpoint_ordinal=1,
        completed_checkpoint_count=0,
        checkpoint_count=4,
        progress_ratio=0.0,
        local_goal=(-1.0, -0.6),
        remaining_distance_m=4.0,
        latest_readings=PatrolLiveReadings(),
        active_alerts=(),
        runtime_identity=PatrolRuntimeIdentity(
            model_id="models/demo.zip",
            model_sha256="a" * 64,
            safety_shield_enabled=True,
            safety_shield_id="shield-v1",
            safety_shield_sha256="b" * 64,
        ),
    )


def _failed_snapshot(running: PatrolLiveSnapshot) -> PatrolLiveSnapshot:
    return replace(
        running,
        sequence=1,
        state=PatrolTaskState.FAILED,
        outcome=PatrolLiveOutcome.ERROR,
        local_goal=None,
        remaining_distance_m=None,
    )


def _write_aborted_run(
    run_directory: Path,
    *,
    raw_error_message: str = "credential=/private/secret",
) -> PatrolRuntimeAbort:
    running = _running_snapshot()
    failed = _failed_snapshot(running)
    writer = PatrolLiveStatusWriter(run_directory, run_id=running.run_id)
    writer.publish(running)
    evidence = PatrolRuntimeAbort.from_exception(
        RuntimeError(raw_error_message),
        run_id=failed.run_id,
        sequence=failed.sequence,
        seed=failed.seed,
        control_step=failed.control_step,
        simulation_time_s=failed.simulation_time_s,
        checkpoints_visited=failed.completed_checkpoint_count,
        checkpoints_total=failed.checkpoint_count,
        collision_count=0,
        sample_count=0,
    )
    # This mirrors the controller ordering: the bounded marker is durable before
    # the matching failed status becomes visible.
    write_patrol_runtime_abort(run_directory, evidence)
    writer.publish(failed)
    return evidence


def test_runtime_abort_is_redacted_bounded_and_projects_as_incomplete(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    secret_message = "database-password=do-not-upload"
    abort = _write_aborted_run(run_directory, raw_error_message=secret_message)

    raw = (run_directory / RUNTIME_ABORT_FILENAME).read_text(encoding="utf-8")
    assert secret_message not in raw
    assert abort.failure_reason == "unhandled controller exception (RuntimeError)"
    assert stat.S_IMODE((run_directory / RUNTIME_ABORT_FILENAME).stat().st_mode) == 0o600

    evidence = PatrolDashboardRepository(run_directory).read()
    payload = project_dashboard(evidence, DashboardMap.from_yaml(MAP_CONFIG))

    assert evidence.runtime_abort == abort
    assert evidence.summary is None
    assert evidence.inspection_summary is None
    assert evidence.terminal_validation == RUNTIME_ABORT_EVIDENCE_SCOPE
    assert evidence.alarm_history_source == "runtime_abort_snapshot_diff"
    assert payload["source"]["mode"] == "aborted"
    assert payload["run"]["state"] == "failed"
    assert payload["run"]["outcome"] == "error"
    assert payload["run"]["terminal"] is True
    assert payload["inspection"] == {
        "available": False,
        "sample_count": None,
        "visit_count": 0,
        "visits": [],
    }
    assert payload["result"] == {
        "available": True,
        "validated": False,
        "validation": RUNTIME_ABORT_EVIDENCE_SCOPE,
        "incomplete": True,
        "outcome": "aborted",
        "success": False,
        "failure_reason": "unhandled controller exception (RuntimeError)",
        "failure_code": "unhandled_controller_exception",
        "failure_type": "RuntimeError",
        "checkpoints_visited": 0,
        "checkpoints_total": 4,
        "collision_count": 0,
        "sample_count": 0,
        "elapsed_simulation_time_s": 0.0,
    }


def test_runtime_abort_binding_is_strict_and_normal_failed_terminal_stays_strict(
    tmp_path: Path,
) -> None:
    missing_abort = tmp_path / "missing-abort"
    running = _running_snapshot(run_id="normal-failed-run")
    writer = PatrolLiveStatusWriter(missing_abort, run_id=running.run_id)
    writer.publish(running)
    writer.publish(_failed_snapshot(running))
    with pytest.raises(DashboardEvidenceError, match="summary.json"):
        PatrolDashboardRepository(missing_abort).read()

    mismatched = tmp_path / "mismatched-abort"
    _write_aborted_run(mismatched)
    path = mismatched / RUNTIME_ABORT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sequence"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DashboardEvidenceError, match="sequence"):
        PatrolDashboardRepository(mismatched).read()

    wrong_outcome = tmp_path / "wrong-outcome"
    running = _running_snapshot(run_id="collision-run")
    failed = replace(
        _failed_snapshot(running),
        outcome=PatrolLiveOutcome.COLLISION,
    )
    writer = PatrolLiveStatusWriter(wrong_outcome, run_id=running.run_id)
    writer.publish(running)
    write_patrol_runtime_abort(
        wrong_outcome,
        PatrolRuntimeAbort.from_exception(
            RuntimeError("must stay private"),
            run_id=failed.run_id,
            sequence=failed.sequence,
            seed=failed.seed,
            control_step=failed.control_step,
            simulation_time_s=failed.simulation_time_s,
            checkpoints_visited=0,
            checkpoints_total=4,
            collision_count=1,
            sample_count=0,
        ),
    )
    writer.publish(failed)
    with pytest.raises(DashboardEvidenceError, match="outcome=error"):
        PatrolDashboardRepository(wrong_outcome).read()


def test_abort_marker_race_stays_live_until_matching_terminal_and_errors_are_wrapped(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "race"
    running = _running_snapshot(run_id="race-run")
    failed = _failed_snapshot(running)
    writer = PatrolLiveStatusWriter(run_directory, run_id=running.run_id)
    writer.publish(running)
    abort = PatrolRuntimeAbort.from_exception(
        RuntimeError("private detail"),
        run_id=failed.run_id,
        sequence=failed.sequence,
        seed=failed.seed,
        control_step=failed.control_step,
        simulation_time_s=failed.simulation_time_s,
        checkpoints_visited=0,
        checkpoints_total=4,
        collision_count=0,
        sample_count=0,
    )
    write_patrol_runtime_abort(run_directory, abort)

    live = PatrolDashboardRepository(run_directory).read()
    live_payload = project_dashboard(live, DashboardMap.from_yaml(MAP_CONFIG))
    assert live.terminal is False
    assert live.runtime_abort is None
    assert live_payload["source"]["mode"] == "live"
    assert live_payload["result"]["available"] is False

    writer.publish(failed)
    (run_directory / RUNTIME_ABORT_FILENAME).write_text("{", encoding="utf-8")
    with pytest.raises(DashboardEvidenceError, match="runtime_abort.json"):
        PatrolDashboardRepository(run_directory).read()
    (run_directory / RUNTIME_ABORT_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(DashboardEvidenceError, match="巡检证据读取失败"):
        PatrolDashboardRepository(run_directory).read()


class _CommitThenLoseAck:
    def __init__(self) -> None:
        self.body: bytes | None = None

    def upload(self, snapshot: Any) -> RelayAcknowledgement:
        self.body = snapshot.body
        raise RelayRetryableError("connection closed after commit")


class _ExactRetryTransport:
    def __init__(self, expected_body: bytes) -> None:
        self.expected_body = expected_body
        self.uploads = 0

    def upload(self, snapshot: Any) -> RelayAcknowledgement:
        self.uploads += 1
        assert snapshot.body == self.expected_body
        return RelayAcknowledgement(
            device_id=snapshot.device_id,
            run_id=snapshot.run_id,
            sequence=snapshot.sequence,
            terminal=snapshot.terminal,
        )


def test_runtime_abort_relay_restarts_with_the_exact_terminal_body_after_lost_ack(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    _write_aborted_run(run_directory)
    repository = PatrolDashboardRepository(run_directory)
    dashboard_map = DashboardMap.from_yaml(MAP_CONFIG)
    first_transport = _CommitThenLoseAck()
    first = PatrolRelayPublisher(
        repository,
        dashboard_map,
        first_transport,
        device_id="macmini-m4",
    )

    with pytest.raises(RelayRetryableError, match="after commit"):
        first.publish_latest()
    assert first_transport.body is not None
    prepared = json.loads(first_transport.body)
    assert prepared["run"]["terminal"] is True
    assert prepared["result"]["incomplete"] is True
    assert prepared["result"]["validated"] is False

    retry_transport = _ExactRetryTransport(first_transport.body)
    resumed = PatrolRelayPublisher(
        PatrolDashboardRepository(run_directory),
        dashboard_map,
        retry_transport,
        device_id="macmini-m4",
        projector=lambda *_args: pytest.fail("persisted abort body must not be reprojected"),
    )

    acknowledgement = resumed.run_until_terminal_acknowledged()

    assert acknowledgement == RelayAcknowledgement(
        "macmini-m4",
        "runtime-abort-run",
        1,
        True,
    )
    assert retry_transport.uploads == 1
