from __future__ import annotations

import http.client
import json
import threading
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from indoor_patrol_rl.application import (
    PatrolLiveAlert,
    PatrolLiveAvoidance,
    PatrolLiveDynamicObstacle,
    PatrolLiveOutcome,
    PatrolLiveReadings,
    PatrolLiveSnapshot,
    PatrolRuntimeIdentity,
    PatrolTaskState,
)
from indoor_patrol_rl.dashboard import (
    DashboardEvidenceError,
    DashboardMap,
    PatrolDashboardRepository,
    PatrolDashboardServer,
    project_dashboard,
)
from indoor_patrol_rl.inspection import (
    EnvironmentalInspection,
    load_inspection_config,
    write_inspection_bundle,
)

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
MAP_CONFIG = REPOSITORY_ROOT / "configs/maps/patrol_route.yaml"
INSPECTION_CONFIG = REPOSITORY_ROOT / "configs/inspection/environmental_patrol_v1.yaml"


def _identity() -> PatrolRuntimeIdentity:
    return PatrolRuntimeIdentity(
        model_id="r2a-demo",
        model_sha256="a" * 64,
        safety_shield_enabled=True,
        safety_shield_id="front-brake-v1",
        safety_shield_sha256="b" * 64,
    )


def _alert() -> PatrolLiveAlert:
    return PatrolLiveAlert(
        rule_id="high_temperature",
        sensor_id="temperature",
        severity="warning",
        observed_value=39.5,
        threshold=38.0,
        unit="°C",
        message="温度告警",
        raised_at_s=0.1,
    )


def _snapshot(
    sequence: int,
    *,
    active_alerts: tuple[PatrolLiveAlert, ...] = (),
    position: tuple[float, float] | None = None,
) -> PatrolLiveSnapshot:
    return PatrolLiveSnapshot(
        run_id="dashboard-run",
        sequence=sequence,
        state=PatrolTaskState.NAVIGATING,
        outcome=PatrolLiveOutcome.RUNNING,
        seed=7000,
        control_step=sequence,
        simulation_time_s=sequence * 0.128,
        position=position or (-1.6 + sequence * 0.001, 0.0),
        current_checkpoint_id="P1",
        current_checkpoint_ordinal=1,
        completed_checkpoint_count=0,
        checkpoint_count=4,
        progress_ratio=0.0,
        local_goal=(-1.0, -0.6),
        remaining_distance_m=max(0.0, 4.0 - sequence * 0.001),
        latest_readings=PatrolLiveReadings(),
        active_alerts=active_alerts,
        runtime_identity=_identity(),
    )


def _write_live_stream(run_directory: Path, snapshots: list[PatrolLiveSnapshot]) -> None:
    run_directory.mkdir()
    (run_directory / "events.jsonl").write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in snapshots),
        encoding="utf-8",
    )
    (run_directory / "status.json").write_text(
        json.dumps(snapshots[-1].to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )


def _terminal_snapshot(*, active_alerts: tuple[PatrolLiveAlert, ...]) -> PatrolLiveSnapshot:
    return replace(
        _snapshot(10, active_alerts=active_alerts, position=(-1.6, -1.15)),
        state=PatrolTaskState.COMPLETED,
        outcome=PatrolLiveOutcome.SUCCESS,
        current_checkpoint_id=None,
        current_checkpoint_ordinal=None,
        completed_checkpoint_count=4,
        progress_ratio=1.0,
        local_goal=None,
        remaining_distance_m=None,
        latest_readings=PatrolLiveReadings(
            temperature_c=23.9,
            smoke_obscuration_pct_per_m=0.06,
            carbon_monoxide_ppm=4.0,
            sampled_at_s=1.1,
            checkpoint_id="P4",
        ),
    )


def _write_terminal_files(
    run_directory: Path,
    *,
    inspection_positions: list[tuple[float, float]] | None = None,
) -> None:
    summary = {
        "schema_version": "1.0",
        "run_id": "dashboard-run",
        "backend": "Webots R2025a",
        "planner": "astar_local_waypoint_ppo",
        "outcome": "completed",
        "failure_reason": None,
        "scenario_seed": 7000,
        "model_path": "r2a-demo",
        "model_sha256": "a" * 64,
        "model_training_domain": "webots",
        "model_selection_reason": "dashboard fixture",
        "safety_shield_enabled": True,
        "safety_shield_id": "front-brake-v1",
        "safety_shield_sha256": "b" * 64,
        "checkpoints_total": 4,
        "checkpoints_visited": 4,
        "completion_ratio": 1.0,
        "control_step_count": 10,
        "physics_step_count": 10,
        "collision_count": 0,
        "elapsed_simulation_time_s": 1.28,
        "actual_path_length_m": 10.8,
        "planned_attempted_path_length_m": 10.8,
        "nominal_route_length_m": 10.8,
        "completed_path_ratio": 1.0,
        "minimum_clearance_m": 0.22,
        "minimum_lidar_range_m": 0.32,
        "critical_proximity_count": 0,
        "safety_shield_intervention_steps": 7,
        "safety_shield_intervention_rate": 0.7,
        "ttc_escape_guard_intervention_steps": 2,
        "ttc_escape_guard_activation_count": 1,
        "ttc_escape_guard_recovery_count": 1,
        "ttc_escape_guard_unrecoverable_count": 0,
        "inference_latency_p50_ms": 0.31,
        "inference_latency_p95_ms": 0.52,
        "dynamic_obstacle_count": 2,
        "dynamic_obstacle_speed_max_mps": 0.18,
        "patrol_scenario_id": "dashboard-demo-v1",
        "patrol_scenario_mode": "operational",
        "patrol_scenario_config_sha256": "c" * 64,
        "ttc_escape_guard_enabled": True,
        "ttc_escape_guard_id": "dashboard-ttc-guard-v1",
        "ttc_escape_guard_evidence_scope": "dashboard_fixture",
        "trajectory_file": "trajectory.csv",
        "checkpoints_file": "checkpoints.csv",
        "collisions_file": "collisions.csv",
        "checkpoint_records": [
            {
                "checkpoint_id": checkpoint_id,
                "visit_order": ordinal,
                "target_x_m": position[0],
                "target_y_m": position[1],
                "reached": True,
            }
            for ordinal, (checkpoint_id, position) in enumerate(
                [
                    ("P1", (1.6, 0.0)),
                    ("P2", (1.6, 1.3)),
                    ("P3", (-1.6, 1.3)),
                    ("P4", (-1.6, -1.15)),
                ],
                1,
            )
        ],
    }
    summary_path = run_directory / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    inspection = EnvironmentalInspection(
        load_inspection_config(INSPECTION_CONFIG),
        run_id="dashboard-run",
        scenario_seed=7000,
    )
    positions = inspection_positions or [
        (1.6, 0.0),
        (1.6, 1.3),
        (-1.6, 1.3),
        (-1.6, -1.15),
    ]
    for ordinal, (checkpoint_id, position) in enumerate(
        zip(("P1", "P2", "P3", "P4"), positions, strict=True),
        1,
    ):
        inspection.sample_visit(
            checkpoint_id,
            ordinal,
            ordinal * 2,
            ordinal * 0.05,
            position,
        )
    write_inspection_bundle(
        run_directory,
        inspection,
        INSPECTION_CONFIG,
        expected_checkpoint_count=4,
        patrol_summary_path=summary_path,
    )


def test_map_loads_p1_to_p4_and_central_cabinet_with_demo_boundary() -> None:
    dashboard_map = DashboardMap.from_yaml(MAP_CONFIG)

    assert [item.checkpoint_id for item in dashboard_map.checkpoints] == [
        "P1",
        "P2",
        "P3",
        "P4",
    ]
    assert [item.obstacle_id for item in dashboard_map.obstacles] == ["central_cabinet"]
    assert dashboard_map.origin == (-2.25, -1.75)


def test_running_projection_uses_snapshot_diffs_and_labels_reading_provenance(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    second = replace(
        _snapshot(1, active_alerts=(_alert(),)),
        latest_readings=PatrolLiveReadings(
            temperature_c=39.5,
            smoke_obscuration_pct_per_m=0.1,
            carbon_monoxide_ppm=3.0,
            sampled_at_s=0.1,
            checkpoint_id="P1",
        ),
    )
    _write_live_stream(run_directory, [_snapshot(0), second])

    evidence = PatrolDashboardRepository(run_directory).read()
    payload = project_dashboard(
        evidence,
        DashboardMap.from_yaml(MAP_CONFIG),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert payload["schema_version"] == "patrol-dashboard/v1"
    assert payload["generated_at_utc"] == "2026-09-01T00:00:00Z"
    assert payload["source"]["alarm_history_source"] == "live_snapshot_diff"
    assert payload["source"]["map_qualification"] == "demonstration_only"
    assert "不代表真实场地" in payload["source"]["map_boundary_note"]
    assert payload["sensors"]["provenance_label"] == "上次巡检点采样"
    assert payload["sensors"]["checkpoint_id"] == "P1"
    assert payload["alarms"]["active"][0]["checkpoint_id"] == "P1"
    assert payload["alarms"]["history"][0]["event_type"] == "raised"
    assert payload["inspection"] == {
        "available": False,
        "sample_count": None,
        "visit_count": 0,
        "visits": [],
    }
    assert payload["result"]["available"] is False


def test_projection_exposes_current_dynamic_obstacles_and_avoidance_audit(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    snapshot = replace(
        _snapshot(5),
        dynamic_obstacles=(
            PatrolLiveDynamicObstacle(
                obstacle_id="DYNAMIC_OBSTACLE_0",
                position=(-0.92, -0.25),
                radius_m=0.18,
                velocity_mps=(0.0, 0.18),
            ),
        ),
        avoidance=PatrolLiveAvoidance(
            measured_control_step=5,
            measured_at_s=0.64,
            minimum_lidar_range_m=0.36,
            robust_ttc_s=1.2,
            policy_action=(0.5, -0.1),
            filtered_action=(0.5, -0.1),
            executed_action=(0.11, -0.2),
        ),
    )
    _write_live_stream(run_directory, [snapshot])

    payload = project_dashboard(
        PatrolDashboardRepository(run_directory).read(),
        DashboardMap.from_yaml(MAP_CONFIG),
    )

    assert payload["map"]["dynamic_obstacles"] == [
        {
            "id": "DYNAMIC_OBSTACLE_0",
            "position": {"x_m": -0.92, "y_m": -0.25},
            "radius_m": 0.18,
            "velocity": {"x_mps": 0.0, "y_mps": 0.18},
        }
    ]
    assert payload["avoidance"]["minimum_lidar_range_m"] == pytest.approx(0.36)
    assert payload["avoidance"]["actions"]["policy"]["linear_normalized"] == 0.5


def test_live_diff_records_alarm_clear_transition(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(
        run_directory,
        [_snapshot(0), _snapshot(1, active_alerts=(_alert(),)), _snapshot(2)],
    )

    evidence = PatrolDashboardRepository(run_directory).read()

    assert [event["event_type"] for event in evidence.alarm_history] == [
        "raised",
        "cleared",
    ]


def test_trajectory_is_uniformly_bounded_and_preserves_endpoints(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    snapshots = [_snapshot(index) for index in range(401)]
    _write_live_stream(run_directory, snapshots)

    payload = project_dashboard(
        PatrolDashboardRepository(run_directory).read(),
        DashboardMap.from_yaml(MAP_CONFIG),
    )
    trajectory = payload["map"]["trajectory"]

    assert len(trajectory) == 300
    assert trajectory[0]["sequence"] == 0
    assert trajectory[-1]["sequence"] == 400
    assert payload["map"]["trajectory_truncated"] is True
    assert payload["map"]["trajectory_source_point_count"] == 401


def test_terminal_projection_uses_verified_bundle_and_displays_complete_result(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    terminal = _terminal_snapshot(active_alerts=())
    _write_live_stream(run_directory, [terminal])
    _write_terminal_files(run_directory)

    evidence = PatrolDashboardRepository(run_directory).read()
    payload = project_dashboard(evidence, DashboardMap.from_yaml(MAP_CONFIG))

    assert evidence.alarm_history_source == "verified_inspection_bundle"
    assert evidence.terminal_validation == "inspection_bundle_and_runtime_consistency"
    assert payload["alarms"]["history"][0]["checkpoint_id"] == "P4"
    assert payload["alarms"]["history"][0]["sensor_id"] == "carbon_monoxide"
    assert "alarm cleared" in payload["alarms"]["history"][0]["message"]
    assert payload["result"]["validated"] is True
    assert payload["result"]["success"] is True
    assert payload["result"]["sample_count"] == 36
    assert payload["result"]["minimum_clearance_m"] == 0.22
    assert payload["result"]["minimum_lidar_range_m"] == 0.32
    assert payload["result"]["safety_shield_intervention_steps"] == 7
    assert payload["result"]["safety_shield_intervention_rate"] == 0.7
    assert payload["result"]["ttc_escape_guard_activation_count"] == 1
    assert payload["result"]["ttc_escape_guard_recovery_count"] == 1
    assert payload["result"]["inference_latency_p95_ms"] == 0.52
    assert payload["result"]["dynamic_obstacle_count"] == 2
    assert payload["result"]["failure_reason"] is None
    assert payload["result"]["alarms_cleared"] == 2
    assert payload["result"]["highest_event_severity"] == "critical"
    assert [item["status"] for item in payload["map"]["checkpoints"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]


def test_graduation_demo_inspection_projection_is_traceable_and_redacted(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_terminal_snapshot(active_alerts=())])
    _write_terminal_files(
        run_directory,
        inspection_positions=[
            (1.513194446150353, -0.013197516580409211),
            (1.5610229464933314, 1.2203411484343563),
            (-1.5269347965522584, 1.2430406767963162),
            (-1.6865219703809506, -1.123538169770933),
        ],
    )

    payload = project_dashboard(
        PatrolDashboardRepository(run_directory).read(),
        DashboardMap.from_yaml(MAP_CONFIG),
    )
    inspection = payload["inspection"]

    assert inspection["available"] is True
    assert inspection["sample_count"] == 36
    assert inspection["visit_count"] == 4
    visits = {visit["checkpoint_id"]: visit for visit in inspection["visits"]}
    assert visits["P2"]["readings"]["temperature"]["value"] == 39.4
    assert visits["P3"]["readings"]["smoke_obscuration"]["value"] == 6.06
    assert visits["P3"]["readings"]["carbon_monoxide"]["value"] == 78.0
    assert set(visits["P3"]["readings"]["carbon_monoxide"]) == {
        "display_name",
        "value",
        "unit",
        "quality",
        "sample_index",
    }
    serialized = json.dumps(inspection, ensure_ascii=False)
    assert "truth_value" not in serialized
    assert "noise_value" not in serialized


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("run_id", "run_id"),
        ("checkpoint", "checkpoint|\u4e0a\u4e0b\u6587"),
        ("sensor_set", "\u4f20\u611f\u5668"),
        ("non_finite", "\u6709\u9650\u6570\u503c"),
        ("summary_count", "sample_count"),
    ],
)
def test_inspection_projection_fails_closed_on_in_memory_evidence_mismatch(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_terminal_snapshot(active_alerts=())])
    _write_terminal_files(run_directory)
    evidence = PatrolDashboardRepository(run_directory).read()
    assert evidence.inspection_samples is not None
    assert evidence.inspection_summary is not None

    samples = [dict(sample) for sample in evidence.inspection_samples]
    inspection_summary = dict(evidence.inspection_summary)
    if mutation == "run_id":
        samples[0]["run_id"] = "another-run"
    elif mutation == "checkpoint":
        samples[0]["checkpoint_id"] = "P2"
    elif mutation == "sensor_set":
        samples[2].update(
            sensor_id="temperature",
            quantity="temperature",
            display_name="温度",
            unit="°C",
        )
    elif mutation == "non_finite":
        samples[0]["observed_value"] = "nan"
    else:
        inspection_summary["sample_count"] = len(samples) - 1
    inconsistent = replace(
        evidence,
        inspection_samples=tuple(samples),
        inspection_summary=inspection_summary,
    )

    with pytest.raises(DashboardEvidenceError, match=message):
        project_dashboard(inconsistent, DashboardMap.from_yaml(MAP_CONFIG))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("equal_control_step", "control_step.*严格递增"),
        ("equal_simulation_time", "simulation_time_s.*严格递增"),
        ("control_step_after_terminal", "control_step.*超过终态"),
        ("simulation_time_after_terminal", "simulation_time_s.*超过终态"),
    ],
)
def test_inspection_projection_rejects_non_monotonic_or_future_visit_clocks(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_terminal_snapshot(active_alerts=())])
    _write_terminal_files(run_directory)
    evidence = PatrolDashboardRepository(run_directory).read()
    assert evidence.inspection_summary is not None
    inspection_summary = deepcopy(evidence.inspection_summary)
    visits = inspection_summary["visit_records"]
    if mutation == "equal_control_step":
        visits[1]["control_step"] = visits[0]["control_step"]
    elif mutation == "equal_simulation_time":
        visits[1]["simulation_time_s"] = visits[0]["simulation_time_s"]
    elif mutation == "control_step_after_terminal":
        visits[-1]["control_step"] = evidence.current.control_step + 1
    else:
        visits[-1]["simulation_time_s"] = evidence.current.simulation_time_s + 0.1

    with pytest.raises(DashboardEvidenceError, match=message):
        project_dashboard(
            replace(evidence, inspection_summary=inspection_summary),
            DashboardMap.from_yaml(MAP_CONFIG),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"outcome": "timeout", "failure_reason": "timeout", "collision_count": 1},
            "Only a collision outcome",
        ),
        (
            {
                "outcome": "safety_stop",
                "failure_reason": "safety stop",
                "collision_count": 0,
                "critical_proximity_count": 0,
            },
            "safety_stop outcome",
        ),
        ({"collision_count": True}, "collision_count must be an integer"),
    ],
)
def test_terminal_result_replays_patrol_summary_domain_invariants(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_terminal_snapshot(active_alerts=())])
    _write_terminal_files(run_directory)
    evidence = PatrolDashboardRepository(run_directory).read()
    assert evidence.summary is not None
    inconsistent_summary = dict(evidence.summary)
    inconsistent_summary.update(overrides)

    with pytest.raises(DashboardEvidenceError, match=message):
        project_dashboard(
            replace(evidence, summary=inconsistent_summary),
            DashboardMap.from_yaml(MAP_CONFIG),
        )


@pytest.mark.parametrize(
    "missing",
    [
        "summary.json",
        "inspection_summary.json",
        "alarm_events.jsonl",
        "inspection_manifest.json",
        "inspection_manifest.sha256",
    ],
)
def test_terminal_repository_requires_complete_verified_bundle(
    tmp_path: Path,
    missing: str,
) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_terminal_snapshot(active_alerts=())])
    _write_terminal_files(run_directory)
    (run_directory / missing).unlink()

    with pytest.raises(DashboardEvidenceError, match="缺少|bundle|manifest"):
        PatrolDashboardRepository(run_directory).read()


def test_repository_rejects_run_id_mismatch_in_summary(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_snapshot(0)])
    (run_directory / "summary.json").write_text(
        json.dumps({"run_id": "another-run"}), encoding="utf-8"
    )

    with pytest.raises(DashboardEvidenceError, match="run_id"):
        PatrolDashboardRepository(run_directory).read()


def test_repository_rejects_run_id_mismatch_in_alarm_events(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_terminal_snapshot(active_alerts=())])
    _write_terminal_files(run_directory)
    events_path = run_directory / "alarm_events.jsonl"
    records = [json.loads(line) for line in events_path.read_text().splitlines()]
    records[0]["run_id"] = "another-run"
    events_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(DashboardEvidenceError, match="bundle|SHA-256"):
        PatrolDashboardRepository(run_directory).read()


@pytest.mark.parametrize(
    "mismatch",
    ["model", "shield", "control_step", "simulation_time"],
)
def test_terminal_repository_binds_runtime_identity_and_clock(
    tmp_path: Path,
    mismatch: str,
) -> None:
    run_directory = tmp_path / "run"
    terminal = _terminal_snapshot(active_alerts=())
    if mismatch == "model":
        terminal = replace(
            terminal,
            runtime_identity=replace(
                terminal.runtime_identity,
                model_id="another-model",
                model_sha256="c" * 64,
            ),
        )
    elif mismatch == "shield":
        terminal = replace(
            terminal,
            runtime_identity=replace(
                terminal.runtime_identity,
                safety_shield_id="another-shield",
                safety_shield_sha256="d" * 64,
            ),
        )
    elif mismatch == "control_step":
        terminal = replace(terminal, control_step=11)
    else:
        terminal = replace(terminal, simulation_time_s=1.29)
    _write_live_stream(run_directory, [terminal])
    _write_terminal_files(run_directory)

    with pytest.raises(DashboardEvidenceError, match="运行身份|control_step|elapsed"):
        PatrolDashboardRepository(run_directory).read()


def test_terminal_repository_reuses_bundle_hash_verification(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_terminal_snapshot(active_alerts=())])
    _write_terminal_files(run_directory)
    samples = run_directory / "inspection_samples.csv"
    samples.write_text(samples.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(DashboardEvidenceError, match="bundle|SHA-256"):
        PatrolDashboardRepository(run_directory).read()


def test_projection_rejects_live_checkpoint_id_ordinal_mismatch(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(
        run_directory,
        [replace(_snapshot(0), current_checkpoint_id="P2")],
    )

    with pytest.raises(DashboardEvidenceError, match="id/ordinal"):
        project_dashboard(
            PatrolDashboardRepository(run_directory).read(),
            DashboardMap.from_yaml(MAP_CONFIG),
        )


def test_projection_rejects_terminal_checkpoint_coordinate_mismatch(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_terminal_snapshot(active_alerts=())])
    _write_terminal_files(run_directory)
    dashboard_map = DashboardMap.from_yaml(MAP_CONFIG)
    wrong_first = replace(dashboard_map.checkpoints[0], position=(1.5, 0.0))
    dashboard_map = replace(
        dashboard_map,
        checkpoints=(wrong_first, *dashboard_map.checkpoints[1:]),
    )

    with pytest.raises(DashboardEvidenceError, match="目标坐标"):
        project_dashboard(PatrolDashboardRepository(run_directory).read(), dashboard_map)


def test_alarm_history_payload_is_limited_to_two_hundred_rows(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    snapshots = [
        _snapshot(index, active_alerts=(_alert(),) if index % 2 else ()) for index in range(202)
    ]
    _write_live_stream(run_directory, snapshots)

    payload = project_dashboard(
        PatrolDashboardRepository(run_directory).read(),
        DashboardMap.from_yaml(MAP_CONFIG),
    )

    assert payload["alarms"]["history_count"] == 201
    assert payload["alarms"]["history_returned_count"] == 200
    assert len(payload["alarms"]["history"]) == 200
    assert payload["alarms"]["history_truncated"] is True


def test_repository_rejects_symlinked_evidence_that_escapes_run_directory(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_snapshot(0).to_dict()), encoding="utf-8")
    (run_directory / "status.json").symlink_to(outside)
    (run_directory / "events.jsonl").write_text(
        json.dumps(_snapshot(0).to_dict()) + "\n", encoding="utf-8"
    )

    with pytest.raises(DashboardEvidenceError, match="符号链接"):
        PatrolDashboardRepository(run_directory).read()


def test_terminal_repository_rejects_symlinked_bundle_member(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_terminal_snapshot(active_alerts=())])
    _write_terminal_files(run_directory)
    samples = run_directory / "inspection_samples.csv"
    outside = tmp_path / "outside.csv"
    samples.replace(outside)
    samples.symlink_to(outside)

    with pytest.raises(DashboardEvidenceError, match="符号链接"):
        PatrolDashboardRepository(run_directory).read()


def test_repository_rejects_status_and_event_disagreement(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_snapshot(0)])
    conflicting = replace(_snapshot(0), position=(-1.0, 1.0))
    (run_directory / "status.json").write_text(json.dumps(conflicting.to_dict()), encoding="utf-8")

    with pytest.raises(DashboardEvidenceError, match="内容不一致"):
        PatrolDashboardRepository(run_directory).read()


def test_map_rejects_checkpoint_outside_demonstration_boundary(tmp_path: Path) -> None:
    config = tmp_path / "map.yaml"
    config.write_text(
        """
map:
  origin: [0, 0]
  size: [2, 2]
  resolution: 0.1
  static_rectangles: []
robot_start: [0.5, 0.5]
checkpoints:
  - id: P1
    position: [3, 1]
""",
        encoding="utf-8",
    )

    with pytest.raises(DashboardEvidenceError, match="边界之外"):
        DashboardMap.from_yaml(config)


def test_http_server_is_loopback_only_and_serves_dto_static_and_security_headers(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_snapshot(0)])
    server = PatrolDashboardServer(
        PatrolDashboardRepository(run_directory),
        DashboardMap.from_yaml(MAP_CONFIG),
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        connection.request("GET", "/api/dashboard")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
        assert "default-src 'self'" in response.getheader("Content-Security-Policy")
        assert payload["schema_version"] == "patrol-dashboard/v1"

        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
        assert "室内智能巡检看板" in response.read().decode("utf-8")

        connection.request("GET", "/static/dashboard.css")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_server_rejects_traversal_and_mutating_methods(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    _write_live_stream(run_directory, [_snapshot(0)])
    server = PatrolDashboardServer(
        PatrolDashboardRepository(run_directory),
        DashboardMap.from_yaml(MAP_CONFIG),
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        connection.request("GET", "/static/%2e%2e/repository.py")
        assert connection.getresponse().status == 404
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        connection.request("POST", "/api/dashboard", body=b"{}")
        response = connection.getresponse()
        assert response.status == 405
        assert response.getheader("Allow") == "GET, HEAD"
        response.read()
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        connection.request("GET", "/healthz", headers={"Host": "attacker.example"})
        response = connection.getresponse()
        assert response.status == 400
        assert json.loads(response.read())["error"] == "invalid_host"
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        connection.request(
            "GET",
            "/healthz",
            headers={"Host": "127.0.0.1:65535"},
        )
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_frontend_contract_serializes_polling_and_exposes_accessible_status() -> None:
    static = REPOSITORY_ROOT / "src/indoor_patrol_rl/dashboard/static"
    javascript = (static / "dashboard.js").read_text(encoding="utf-8")
    html = (static / "index.html").read_text(encoding="utf-8")
    css = (static / "dashboard.css").read_text(encoding="utf-8")

    assert "window.setInterval" not in javascript
    assert "window.setTimeout(refresh, pollIntervalMs)" in javascript
    assert "data.run.sequence < lastSequence" in javascript
    assert "if (terminal) pollingStopped = true" in javascript
    assert "maxGridLinesPerAxis = 32" in javascript
    assert "maxHistoryRows = 200" in javascript
    assert "alarm.message" in javascript
    assert "alarm.sensor_id" in javascript
    assert "renderAvoidance(data)" in javascript
    assert "map.dynamic_obstacles" in javascript
    assert "safety_shield" in javascript
    assert "escape_guard" in javascript
    assert "异常中止 · 证据不完整" in javascript
    assert 'data.source.mode === "aborted"' in javascript
    assert 'id="avoidance-state"' in html
    assert 'aria-label="强化学习动作经过安全过滤后执行"' in html
    assert 'aria-label="巡检点完成进度"' in html
    assert 'aria-live="polite"' in html
    assert 'role="alert"' in html
    assert "prefers-reduced-motion: reduce" in css
