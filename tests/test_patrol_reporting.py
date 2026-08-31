from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from indoor_patrol_rl.evaluation import (
    PatrolCheckpointRecord,
    PatrolCollisionRecord,
    PatrolCriticalProximityEdgeTracker,
    PatrolCriticalProximityRecord,
    PatrolRunSummary,
    PatrolTrajectorySample,
    PatrolTTCEscapeGuardEventRecord,
    portable_model_path,
    select_patrol_model,
    write_patrol_outputs,
)


def test_critical_proximity_edge_tracker_records_one_event_per_interval() -> None:
    tracker = PatrolCriticalProximityEdgeTracker()

    assert [
        tracker.update(critical)
        for critical in (False, True, True, True, False, False, True, True)
    ] == [False, True, False, False, False, False, True, False]


def test_critical_proximity_edge_tracker_rejects_non_boolean_state() -> None:
    with pytest.raises(ValueError, match="active must be a boolean"):
        PatrolCriticalProximityEdgeTracker(active=1)

    tracker = PatrolCriticalProximityEdgeTracker()
    with pytest.raises(ValueError, match="critical must be a boolean"):
        tracker.update(1)


def _touch(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _write_formal_selection(root: Path, model: Path) -> Path:
    manifest = root / "artifacts/results/webots_validation_w1_selection_seed11/selection.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_status": "formal_selected",
                "model_variant": None,
                "validation_contract": {
                    "split": "validation",
                    "episodes": 20,
                    "model_variant": None,
                },
                "gate": {
                    "min_success_rate": 0.75,
                    "max_collision_rate": 0.15,
                    "requires_model_file": True,
                },
                "formal_selected": {
                    "model_path": model.relative_to(root).as_posix(),
                    "model_sha256": digest,
                    "model_variant": None,
                    "episodes": 20,
                    "success_rate": 0.80,
                    "collision_rate": 0.10,
                    "gate_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_model_selection_prefers_formal_validated_checkpoint(tmp_path: Path) -> None:
    latest = _touch(tmp_path, "artifacts/models/webots/seed_11/ppo_webots_latest.zip")
    selected = _touch(
        tmp_path,
        "artifacts/models/webots/seed_11/checkpoints/ppo_webots_s11_40960_steps.zip",
    )
    selected.write_bytes(b"formal-selection")
    manifest = _write_formal_selection(tmp_path, selected)

    selection = select_patrol_model(tmp_path, webots_model=latest)

    assert selection.path == selected.resolve()
    assert selection.training_domain == "webots"
    assert selection.selection_manifest == manifest.resolve()
    assert selection.model_sha256 == hashlib.sha256(b"formal-selection").hexdigest()


def test_unvalidated_fallback_prefers_webots_checkpoint_when_allowed(
    tmp_path: Path,
) -> None:
    warm_start = _touch(tmp_path, "artifacts/models/ppo_curriculum_stage2.zip")
    webots = _touch(tmp_path, "artifacts/models/webots/seed_11/ppo_webots_latest.zip")

    selection = select_patrol_model(
        tmp_path,
        selection_manifest=None,
        allow_unvalidated_fallback=True,
        webots_model=webots,
        warm_start_model=warm_start,
    )

    assert selection.path == webots.resolve()
    assert selection.training_domain == "webots"
    assert "UNVALIDATED" in selection.selection_reason


def test_model_selection_labels_kinematic_fallback(tmp_path: Path) -> None:
    warm_start = _touch(tmp_path, "artifacts/models/ppo_curriculum_stage2.zip")

    selection = select_patrol_model(
        tmp_path,
        selection_manifest=None,
        allow_unvalidated_fallback=True,
        webots_model="artifacts/models/webots/missing.zip",
        warm_start_model=warm_start,
    )

    assert selection.path == warm_start.resolve()
    assert selection.training_domain == "kinematic"
    assert "warm start" in selection.selection_reason


def test_model_selection_refuses_mutable_latest_by_default(tmp_path: Path) -> None:
    _touch(tmp_path, "artifacts/models/webots/seed_11/ppo_webots_latest.zip")

    with pytest.raises(FileNotFoundError, match="formal patrol selection manifest"):
        select_patrol_model(tmp_path)


def test_formal_selection_rejects_model_hash_mismatch(tmp_path: Path) -> None:
    selected = _touch(tmp_path, "artifacts/models/webots/seed_11/selected.zip")
    selected.write_bytes(b"selected")
    _write_formal_selection(tmp_path, selected)
    selected.write_bytes(b"changed-after-selection")

    with pytest.raises(ValueError, match="SHA-256 verification"):
        select_patrol_model(tmp_path)


def test_explicit_missing_model_does_not_silently_fall_back(tmp_path: Path) -> None:
    _touch(tmp_path, "artifacts/models/ppo_curriculum_stage2.zip")

    with pytest.raises(FileNotFoundError, match="Explicit patrol model"):
        select_patrol_model(tmp_path, "artifacts/models/not-there.zip")


def test_explicit_model_is_recorded_as_unvalidated(tmp_path: Path) -> None:
    model = _touch(tmp_path, "artifacts/models/webots/seed_11/manual.zip")

    selection = select_patrol_model(
        tmp_path,
        explicit_model=model,
        allow_unvalidated_fallback=True,
    )

    assert selection.path == model.resolve()
    assert selection.training_domain == "webots"
    assert selection.selection_reason.startswith("UNVALIDATED")


def test_explicit_model_requires_unvalidated_opt_in(tmp_path: Path) -> None:
    model = _touch(tmp_path, "artifacts/models/webots/seed_11/manual.zip")

    with pytest.raises(PermissionError, match="ALLOW_UNVALIDATED"):
        select_patrol_model(tmp_path, explicit_model=model)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("validation_contract", "episodes"), 1, "at least 20"),
        (("gate", "min_success_rate"), 0.0, "min_success_rate"),
        (("gate", "max_collision_rate"), 1.0, "max_collision_rate"),
        (("formal_selected", "success_rate"), 0.50, "success_rate"),
        (("formal_selected", "collision_rate"), 0.50, "collision_rate"),
    ],
)
def test_formal_selection_rejects_relaxed_or_failed_gate(
    tmp_path: Path,
    field_path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    selected = _touch(tmp_path, "artifacts/models/webots/seed_11/selected.zip")
    manifest = _write_formal_selection(tmp_path, selected)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field_path[0]][field_path[1]] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        select_patrol_model(tmp_path)


def test_patrol_outputs_preserve_webots_route_evidence(tmp_path: Path) -> None:
    trajectory = [
        PatrolTrajectorySample(
            control_step=1,
            simulation_time_s=0.128,
            position_x_m=-1.59,
            position_y_m=0.0,
            yaw_rad=0.0,
            checkpoint_id="P1",
            waypoint_index=1,
            local_target_x_m=-0.8,
            local_target_y_m=-0.7,
            remaining_path_m=3.5,
            requested_linear_mps=0.20,
            requested_angular_radps=-0.1,
            executed_linear_mps=0.064,
            executed_angular_radps=-0.1,
            minimum_lidar_range_m=0.75,
            clearance_m=0.65,
            collision=False,
            inference_latency_ms=1.2,
            policy_action_linear=0.20,
            policy_action_angular=-0.10,
            shielded_action_linear=-0.05,
            shielded_action_angular=0.15,
            safety_shield_active=True,
            safety_shield_intervention=math.hypot(-0.25, 0.25),
            shield_front_clearance_m=0.23,
            shield_left_clearance_m=0.41,
            shield_right_clearance_m=0.19,
            minimum_lidar_beam_index=17,
            minimum_lidar_angle_rad=-0.42,
            dynamic_obstacle_1_x_m=-0.35,
            dynamic_obstacle_1_y_m=0.62,
            dynamic_obstacle_2_x_m=0.88,
            dynamic_obstacle_2_y_m=-0.41,
            critical_proximity=False,
            proximity_closing_rate_mps=0.24,
            estimated_ttc_s=0.75,
            ttc_escape_guard_active=True,
            ttc_escape_guard_mode="reverse",
            ttc_escape_guard_target_linear_mps=-0.10,
            ttc_escape_guard_target_angular_radps=0.0,
        )
    ]
    trajectory.append(
        replace(
            trajectory[0],
            control_step=2,
            simulation_time_s=0.256,
            position_x_m=-1.50,
            minimum_lidar_range_m=0.31,
            minimum_lidar_beam_index=5,
            minimum_lidar_angle_rad=0.18,
            dynamic_obstacle_1_x_m=-0.31,
            dynamic_obstacle_1_y_m=0.60,
            dynamic_obstacle_2_x_m=0.84,
            dynamic_obstacle_2_y_m=-0.38,
            shielded_action_linear=0.20,
            shielded_action_angular=-0.10,
            safety_shield_active=False,
            safety_shield_intervention=0.0,
        )
    )
    guard_events = [
        PatrolTTCEscapeGuardEventRecord(
            event_index=1,
            control_step=1,
            simulation_time_s=0.0,
            phase="navigation",
            active=True,
            mode="reverse",
            target_linear_mps=-0.10,
            target_angular_radps=0.0,
            counts_as_intervention_step=True,
        ),
        PatrolTTCEscapeGuardEventRecord(
            event_index=2,
            control_step=2,
            simulation_time_s=0.128,
            phase="navigation",
            active=True,
            mode="reverse",
            target_linear_mps=-0.10,
            target_angular_radps=0.0,
            counts_as_intervention_step=True,
        ),
        PatrolTTCEscapeGuardEventRecord(
            event_index=3,
            control_step=2,
            simulation_time_s=0.256,
            phase="dwell",
            active=False,
            mode="inactive",
            target_linear_mps=None,
            target_angular_radps=None,
            counts_as_intervention_step=False,
        ),
    ]
    checkpoints = [
        PatrolCheckpointRecord(
            checkpoint_id="P1",
            visit_order=1,
            target_x_m=1.6,
            target_y_m=0.0,
            reached=True,
            reached_control_step=170,
            arrival_simulation_time_s=21.76,
            segment_control_steps=170,
            actual_segment_path_m=3.8,
            planned_segment_path_m=3.6,
            segment_path_ratio=3.8 / 3.6,
            segment_minimum_clearance_m=0.12,
        )
    ]
    collisions = [
        PatrolCollisionRecord(
            collision_index=1,
            control_step=2,
            simulation_time_s=24.32,
            position_x_m=1.5,
            position_y_m=0.4,
            detection_source="contact",
            minimum_lidar_range_m=0.12,
            minimum_lidar_beam_index=3,
            minimum_lidar_angle_rad=0.07,
            dynamic_obstacle_1_x_m=0.21,
            dynamic_obstacle_1_y_m=0.36,
            dynamic_obstacle_2_x_m=0.95,
            dynamic_obstacle_2_y_m=-0.22,
        )
    ]
    critical_proximities = [
        PatrolCriticalProximityRecord(
            proximity_index=1,
            control_step=88,
            simulation_time_s=11.264,
            position_x_m=-0.22,
            position_y_m=0.51,
            detection_source="lidar_threshold",
            minimum_lidar_range_m=0.18,
            consecutive_frames=3,
            closing_rate_mps=0.24,
            estimated_ttc_s=0.0,
            minimum_lidar_beam_index=11,
            minimum_lidar_angle_rad=-0.16,
            dynamic_obstacle_1_x_m=-0.10,
            dynamic_obstacle_1_y_m=0.55,
            dynamic_obstacle_2_x_m=0.72,
            dynamic_obstacle_2_y_m=-0.28,
        )
    ]
    summary = PatrolRunSummary(
        schema_version="1.0",
        run_id="webots-patrol-seed7000",
        backend="Webots R2025a",
        planner="astar_local_waypoint_ppo",
        scenario_seed=7000,
        model_path="artifacts/models/webots/seed_11/ppo_webots_latest.zip",
        model_sha256=hashlib.sha256(b"model").hexdigest(),
        model_training_domain="webots",
        model_selection_reason="test",
        outcome="collision",
        failure_reason="collision detected by contact",
        checkpoints_total=4,
        checkpoints_visited=1,
        completion_ratio=0.25,
        collision_count=1,
        control_step_count=2,
        physics_step_count=760,
        elapsed_simulation_time_s=24.32,
        actual_path_length_m=4.2,
        planned_attempted_path_length_m=5.0,
        nominal_route_length_m=10.0,
        completed_path_ratio=None,
        minimum_lidar_range_m=0.12,
        minimum_clearance_m=0.0,
        inference_latency_p50_ms=1.1,
        inference_latency_p95_ms=1.8,
        trajectory_file="trajectory.csv",
        checkpoints_file="checkpoints.csv",
        collisions_file="collisions.csv",
        system_deployment_id="webots_r2a_front_brake_fixed_system_validation_v1",
        system_deployment_profile_sha256="a" * 64,
        patrol_scenario_qualified=False,
        deployment_evidence_scope="validated_component_outside_patrol_scenario",
        safety_shield_id="r2a-front-brake-v1-354ce9b831f0",
        safety_shield_sha256="b" * 64,
        safety_shield_enabled=True,
        safety_shield_intervention_steps=1,
        safety_shield_intervention_rate=0.5,
        patrol_scenario_id="dual_dynamic_obstacles_patrol_v1",
        patrol_scenario_mode="stress",
        patrol_scenario_config_sha256="c" * 64,
        dynamic_obstacle_count=2,
        dynamic_obstacle_speed_max_mps=0.31,
        ttc_escape_guard_enabled=True,
        ttc_escape_guard_id="patrol-ttc-escape-guard-v2",
        ttc_escape_guard_evidence_scope="unqualified_patrol_application_escape_guard",
        ttc_escape_guard_intervention_steps=2,
        ttc_escape_guard_activation_count=1,
        ttc_escape_guard_recovery_count=1,
        ttc_escape_guard_unrecoverable_count=0,
    )

    with pytest.raises(ValueError, match="action-vector difference"):
        replace(trajectory[0], safety_shield_intervention=0.1)
    with pytest.raises(ValueError, match="index and angle"):
        replace(trajectory[0], minimum_lidar_angle_rad=None)
    with pytest.raises(ValueError, match="at most"):
        replace(trajectory[0], minimum_lidar_angle_rad=math.pi + 0.01)
    with pytest.raises(ValueError, match="dynamic-obstacle x/y"):
        replace(trajectory[0], dynamic_obstacle_2_y_m=None)
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(trajectory[0], collision=True, critical_proximity=True)
    with pytest.raises(ValueError, match="proximity_closing_rate_mps"):
        replace(trajectory[0], proximity_closing_rate_mps=math.inf)
    with pytest.raises(ValueError, match="estimated_ttc_s"):
        replace(trajectory[0], estimated_ttc_s=-0.01)
    with pytest.raises(ValueError, match="non-inactive mode"):
        replace(trajectory[0], ttc_escape_guard_mode="inactive")
    with pytest.raises(ValueError, match="mode is not supported"):
        replace(trajectory[0], ttc_escape_guard_mode="sidestep")
    with pytest.raises(ValueError, match="requires target velocities"):
        replace(
            trajectory[0],
            ttc_escape_guard_target_linear_mps=None,
            ttc_escape_guard_target_angular_radps=None,
        )
    with pytest.raises(ValueError, match="recorded together"):
        replace(trajectory[0], ttc_escape_guard_target_angular_radps=None)
    with pytest.raises(ValueError, match="target_linear_mps"):
        replace(trajectory[0], ttc_escape_guard_target_linear_mps=math.nan)
    with pytest.raises(ValueError, match="inactive.*without target"):
        replace(
            trajectory[0],
            ttc_escape_guard_active=False,
            ttc_escape_guard_mode="reverse",
        )
    with pytest.raises(ValueError, match="index and angle"):
        replace(collisions[0], minimum_lidar_beam_index=None)
    with pytest.raises(ValueError, match="dynamic-obstacle x/y"):
        replace(collisions[0], dynamic_obstacle_1_x_m=None)
    with pytest.raises(ValueError, match="must be 'contact'"):
        replace(collisions[0], detection_source="lidar_threshold")
    with pytest.raises(ValueError, match="lidar_threshold.*ttc_escape_guard_timeout"):
        replace(critical_proximities[0], detection_source="contact")
    with pytest.raises(ValueError, match="positive for lidar_threshold"):
        replace(critical_proximities[0], consecutive_frames=0)
    guard_timeout = replace(
        critical_proximities[0],
        detection_source="ttc_escape_guard_timeout",
        consecutive_frames=0,
        minimum_lidar_range_m=0.47,
        closing_rate_mps=0.0,
        estimated_ttc_s=None,
    )
    assert guard_timeout.consecutive_frames == 0
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(guard_timeout, consecutive_frames=-1)
    with pytest.raises(ValueError, match="positive estimated_ttc_s"):
        replace(
            critical_proximities[0],
            closing_rate_mps=None,
            estimated_ttc_s=0.5,
        )
    first_frame_at_threshold = replace(
        critical_proximities[0],
        consecutive_frames=1,
        closing_rate_mps=None,
        estimated_ttc_s=0.0,
    )
    assert first_frame_at_threshold.estimated_ttc_s == 0.0
    receding_without_ttc = replace(
        critical_proximities[0],
        closing_rate_mps=-0.05,
        estimated_ttc_s=None,
    )
    assert receding_without_ttc.closing_rate_mps == pytest.approx(-0.05)
    with pytest.raises(ValueError, match="at least"):
        replace(critical_proximities[0], estimated_ttc_s=-0.01)
    with pytest.raises(ValueError, match="finite number"):
        replace(critical_proximities[0], closing_rate_mps=math.inf)
    with pytest.raises(ValueError, match="scenario identity fields"):
        replace(summary, patrol_scenario_mode=None)
    with pytest.raises(ValueError, match="operational.*stress"):
        replace(summary, patrol_scenario_mode="benchmark")
    with pytest.raises(ValueError, match="both be zero or positive"):
        replace(summary, dynamic_obstacle_count=0)
    with pytest.raises(ValueError, match="cannot carry identity"):
        replace(summary, ttc_escape_guard_enabled=False)
    with pytest.raises(ValueError, match="must be a boolean"):
        replace(summary, ttc_escape_guard_enabled=1)
    with pytest.raises(ValueError, match="ttc_escape_guard_id"):
        replace(summary, ttc_escape_guard_id=None)
    with pytest.raises(ValueError, match="must be an integer"):
        replace(summary, ttc_escape_guard_activation_count=True)
    dwell_heavy_summary = replace(summary, ttc_escape_guard_intervention_steps=191)
    assert dwell_heavy_summary.ttc_escape_guard_intervention_steps == 191
    with pytest.raises(ValueError, match="activation_count cannot exceed"):
        replace(summary, ttc_escape_guard_intervention_steps=0)
    with pytest.raises(ValueError, match="recoveries plus unrecoverable"):
        replace(
            summary,
            ttc_escape_guard_recovery_count=1,
            ttc_escape_guard_unrecoverable_count=1,
        )
    with pytest.raises(ValueError, match="cannot record intervention evidence"):
        replace(
            summary,
            ttc_escape_guard_enabled=False,
            ttc_escape_guard_id=None,
            ttc_escape_guard_evidence_scope=None,
        )
    with pytest.raises(ValueError, match="safety_stop outcome"):
        replace(summary, outcome="safety_stop", collision_count=0)
    for filename_field in (
        "trajectory_file",
        "checkpoints_file",
        "collisions_file",
        "proximity_events_file",
        "ttc_escape_guard_events_file",
    ):
        with pytest.raises(ValueError, match="portable CSV basename"):
            replace(summary, **{filename_field: "../escaped.csv"})
    with pytest.raises(ValueError, match="mutually distinct"):
        replace(summary, collisions_file="trajectory.csv")
    for evidence_scope in (
        "validated_component_outside_patrol_scenario",
        "experimental_patrol_scenario",
        "qualified_patrol_scenario",
        "component_only",
    ):
        with pytest.raises(ValueError, match="qualification issuer"):
            replace(
                summary,
                patrol_scenario_qualified=True,
                deployment_evidence_scope=evidence_scope,
            )

    legacy_trajectory_fields = asdict(trajectory[0])
    for field_name in (
        "policy_action_linear",
        "policy_action_angular",
        "shielded_action_linear",
        "shielded_action_angular",
        "safety_shield_active",
        "safety_shield_intervention",
        "shield_front_clearance_m",
        "shield_left_clearance_m",
        "shield_right_clearance_m",
        "minimum_lidar_beam_index",
        "minimum_lidar_angle_rad",
        "dynamic_obstacle_1_x_m",
        "dynamic_obstacle_1_y_m",
        "dynamic_obstacle_2_x_m",
        "dynamic_obstacle_2_y_m",
        "critical_proximity",
        "proximity_closing_rate_mps",
        "estimated_ttc_s",
        "ttc_escape_guard_active",
        "ttc_escape_guard_mode",
        "ttc_escape_guard_target_linear_mps",
        "ttc_escape_guard_target_angular_radps",
    ):
        legacy_trajectory_fields.pop(field_name)
    legacy_trajectory = PatrolTrajectorySample(**legacy_trajectory_fields)
    assert legacy_trajectory.policy_action_linear is None
    assert legacy_trajectory.safety_shield_active is False
    assert legacy_trajectory.minimum_lidar_beam_index is None
    assert legacy_trajectory.dynamic_obstacle_1_x_m is None
    assert legacy_trajectory.proximity_closing_rate_mps is None
    assert legacy_trajectory.estimated_ttc_s is None
    assert legacy_trajectory.ttc_escape_guard_active is False
    assert legacy_trajectory.ttc_escape_guard_mode == "inactive"
    assert legacy_trajectory.ttc_escape_guard_target_linear_mps is None

    legacy_summary_fields = asdict(summary)
    for field_name in (
        "system_deployment_id",
        "system_deployment_profile_sha256",
        "patrol_scenario_qualified",
        "deployment_evidence_scope",
        "safety_shield_id",
        "safety_shield_sha256",
        "safety_shield_enabled",
        "safety_shield_intervention_steps",
        "safety_shield_intervention_rate",
        "critical_proximity_count",
        "proximity_events_file",
        "patrol_scenario_id",
        "patrol_scenario_mode",
        "patrol_scenario_config_sha256",
        "dynamic_obstacle_count",
        "dynamic_obstacle_speed_max_mps",
        "ttc_escape_guard_enabled",
        "ttc_escape_guard_id",
        "ttc_escape_guard_evidence_scope",
        "ttc_escape_guard_intervention_steps",
        "ttc_escape_guard_activation_count",
        "ttc_escape_guard_recovery_count",
        "ttc_escape_guard_unrecoverable_count",
        "ttc_escape_guard_events_file",
    ):
        legacy_summary_fields.pop(field_name)
    legacy_summary = PatrolRunSummary(**legacy_summary_fields)
    assert legacy_summary.system_deployment_id is None
    assert legacy_summary.safety_shield_enabled is False
    assert legacy_summary.safety_shield_intervention_rate == 0.0
    assert legacy_summary.critical_proximity_count == 0
    assert legacy_summary.proximity_events_file == "proximity_events.csv"
    assert legacy_summary.patrol_scenario_id is None
    assert legacy_summary.ttc_escape_guard_enabled is False
    assert legacy_summary.ttc_escape_guard_id is None
    assert legacy_summary.ttc_escape_guard_intervention_steps == 0
    assert legacy_summary.ttc_escape_guard_activation_count == 0
    assert legacy_summary.ttc_escape_guard_events_file == "ttc_escape_guard_events.csv"

    with pytest.raises(ValueError, match="safety_shield_intervention_steps"):
        write_patrol_outputs(
            tmp_path / "mismatched-shield-count",
            trajectory=trajectory,
            checkpoints=checkpoints,
            collisions=collisions,
            summary=replace(summary, safety_shield_intervention_steps=2),
            ttc_escape_guard_events=guard_events,
        )
    with pytest.raises(ValueError, match="exactly one sample"):
        write_patrol_outputs(
            tmp_path / "incomplete-trajectory",
            trajectory=trajectory[:-1],
            checkpoints=checkpoints,
            collisions=collisions,
            summary=summary,
            ttc_escape_guard_events=guard_events,
        )
    with pytest.raises(ValueError, match="safety_shield_intervention_rate"):
        write_patrol_outputs(
            tmp_path / "mismatched-shield-rate",
            trajectory=trajectory,
            checkpoints=checkpoints,
            collisions=collisions,
            summary=replace(summary, safety_shield_intervention_rate=0.25),
            ttc_escape_guard_events=guard_events,
        )
    with pytest.raises(ValueError, match="collision_count"):
        write_patrol_outputs(
            tmp_path / "mismatched-collision-count",
            trajectory=trajectory,
            checkpoints=checkpoints,
            collisions=collisions,
            summary=replace(
                summary,
                outcome="timeout",
                failure_reason="fixture timeout",
                collision_count=0,
            ),
            ttc_escape_guard_events=guard_events,
        )
    with pytest.raises(ValueError, match="aggregate counts"):
        write_patrol_outputs(
            tmp_path / "mismatched-guard-count",
            trajectory=trajectory,
            checkpoints=checkpoints,
            collisions=collisions,
            summary=replace(summary, ttc_escape_guard_recovery_count=0),
            ttc_escape_guard_events=guard_events,
        )
    with pytest.raises(ValueError, match="complete guard event stream"):
        write_patrol_outputs(
            tmp_path / "missing-guard-events",
            trajectory=trajectory,
            checkpoints=checkpoints,
            collisions=collisions,
            summary=summary,
        )

    symlink_output = tmp_path / "symlink-output"
    symlink_output.mkdir()
    (symlink_output / summary.trajectory_file).symlink_to(tmp_path / "outside.csv")
    with pytest.raises(ValueError, match="stay inside"):
        write_patrol_outputs(
            symlink_output,
            trajectory=trajectory,
            checkpoints=checkpoints,
            collisions=collisions,
            summary=summary,
            ttc_escape_guard_events=guard_events,
        )

    paths = write_patrol_outputs(
        tmp_path / "result",
        trajectory=trajectory,
        checkpoints=checkpoints,
        collisions=collisions,
        summary=summary,
        ttc_escape_guard_events=guard_events,
    )
    original_summary_bytes = paths["summary"].read_bytes()
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        write_patrol_outputs(
            tmp_path / "result",
            trajectory=trajectory,
            checkpoints=checkpoints,
            collisions=collisions,
            summary=summary,
            ttc_escape_guard_events=guard_events,
        )
    assert paths["summary"].read_bytes() == original_summary_bytes

    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert payload["backend"] == "Webots R2025a"
    assert payload["model_training_domain"] == "webots"
    assert payload["model_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert payload["system_deployment_id"] == "webots_r2a_front_brake_fixed_system_validation_v1"
    assert payload["system_deployment_profile_sha256"] == "a" * 64
    assert payload["patrol_scenario_qualified"] is False
    assert payload["deployment_evidence_scope"] == "validated_component_outside_patrol_scenario"
    assert payload["safety_shield_id"] == "r2a-front-brake-v1-354ce9b831f0"
    assert payload["safety_shield_sha256"] == "b" * 64
    assert payload["safety_shield_enabled"] is True
    assert payload["safety_shield_intervention_steps"] == 1
    assert payload["safety_shield_intervention_rate"] == pytest.approx(0.5)
    assert payload["critical_proximity_count"] == 0
    assert payload["proximity_events_file"] == "proximity_events.csv"
    assert payload["critical_proximity_records"] == []
    assert payload["patrol_scenario_id"] == "dual_dynamic_obstacles_patrol_v1"
    assert payload["patrol_scenario_mode"] == "stress"
    assert payload["patrol_scenario_config_sha256"] == "c" * 64
    assert payload["dynamic_obstacle_count"] == 2
    assert payload["dynamic_obstacle_speed_max_mps"] == pytest.approx(0.31)
    assert payload["ttc_escape_guard_enabled"] is True
    assert payload["ttc_escape_guard_id"] == "patrol-ttc-escape-guard-v2"
    assert payload["ttc_escape_guard_evidence_scope"] == (
        "unqualified_patrol_application_escape_guard"
    )
    assert payload["ttc_escape_guard_intervention_steps"] == 2
    assert payload["ttc_escape_guard_activation_count"] == 1
    assert payload["ttc_escape_guard_recovery_count"] == 1
    assert payload["ttc_escape_guard_unrecoverable_count"] == 0
    assert payload["ttc_escape_guard_events_file"] == "ttc_escape_guard_events.csv"
    assert len(payload["ttc_escape_guard_event_records"]) == 3
    assert payload["minimum_lidar_diagnostic"] == {
        "control_step": 2,
        "simulation_time_s": 0.256,
        "minimum_lidar_range_m": 0.31,
        "minimum_lidar_beam_index": 5,
        "minimum_lidar_angle_rad": 0.18,
        "robot_position_x_m": -1.50,
        "robot_position_y_m": 0.0,
        "dynamic_obstacle_1_x_m": -0.31,
        "dynamic_obstacle_1_y_m": 0.60,
        "dynamic_obstacle_2_x_m": 0.84,
        "dynamic_obstacle_2_y_m": -0.38,
    }
    assert payload["checkpoint_records"][0]["checkpoint_id"] == "P1"
    assert payload["collision_records"][0]["detection_source"] == "contact"
    assert payload["collision_records"][0]["minimum_lidar_beam_index"] == 3
    assert payload["collision_records"][0]["minimum_lidar_angle_rad"] == pytest.approx(0.07)
    assert payload["collision_records"][0]["dynamic_obstacle_1_x_m"] == pytest.approx(0.21)
    assert payload["collision_records"][0]["dynamic_obstacle_2_y_m"] == pytest.approx(-0.22)
    with paths["trajectory"].open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["checkpoint_id"] == "P1"
    assert float(row["position_x_m"]) == pytest.approx(-1.59)
    assert float(row["policy_action_linear"]) == pytest.approx(0.20)
    assert float(row["policy_action_angular"]) == pytest.approx(-0.10)
    assert float(row["shielded_action_linear"]) == pytest.approx(-0.05)
    assert float(row["shielded_action_angular"]) == pytest.approx(0.15)
    assert row["safety_shield_active"] == "True"
    assert float(row["safety_shield_intervention"]) == pytest.approx(math.hypot(-0.25, 0.25))
    assert float(row["shield_front_clearance_m"]) == pytest.approx(0.23)
    assert float(row["shield_left_clearance_m"]) == pytest.approx(0.41)
    assert float(row["shield_right_clearance_m"]) == pytest.approx(0.19)
    assert int(row["minimum_lidar_beam_index"]) == 17
    assert float(row["minimum_lidar_angle_rad"]) == pytest.approx(-0.42)
    assert float(row["dynamic_obstacle_1_x_m"]) == pytest.approx(-0.35)
    assert float(row["dynamic_obstacle_1_y_m"]) == pytest.approx(0.62)
    assert float(row["dynamic_obstacle_2_x_m"]) == pytest.approx(0.88)
    assert float(row["dynamic_obstacle_2_y_m"]) == pytest.approx(-0.41)
    assert row["critical_proximity"] == "False"
    assert float(row["proximity_closing_rate_mps"]) == pytest.approx(0.24)
    assert float(row["estimated_ttc_s"]) == pytest.approx(0.75)
    assert row["ttc_escape_guard_active"] == "True"
    assert row["ttc_escape_guard_mode"] == "reverse"
    assert float(row["ttc_escape_guard_target_linear_mps"]) == pytest.approx(-0.10)
    assert float(row["ttc_escape_guard_target_angular_radps"]) == 0.0
    with paths["collisions"].open(newline="", encoding="utf-8") as stream:
        collision_row = next(csv.DictReader(stream))
    assert int(collision_row["minimum_lidar_beam_index"]) == 3
    assert float(collision_row["minimum_lidar_angle_rad"]) == pytest.approx(0.07)
    assert float(collision_row["dynamic_obstacle_1_x_m"]) == pytest.approx(0.21)
    assert float(collision_row["dynamic_obstacle_1_y_m"]) == pytest.approx(0.36)
    assert float(collision_row["dynamic_obstacle_2_x_m"]) == pytest.approx(0.95)
    assert float(collision_row["dynamic_obstacle_2_y_m"]) == pytest.approx(-0.22)
    with paths["proximity_events"].open(newline="", encoding="utf-8") as stream:
        proximity_rows = list(csv.DictReader(stream))
    assert proximity_rows == []
    with paths["ttc_escape_guard_events"].open(newline="", encoding="utf-8") as stream:
        guard_rows = list(csv.DictReader(stream))
    assert [row["phase"] for row in guard_rows] == ["navigation", "navigation", "dwell"]
    assert [row["counts_as_intervention_step"] for row in guard_rows] == [
        "True",
        "True",
        "False",
    ]

    safety_summary = replace(
        summary,
        run_id="webots-patrol-safety-stop-seed7000",
        outcome="safety_stop",
        failure_reason="critical LiDAR proximity triggered a safe stop",
        collision_count=0,
        critical_proximity_count=1,
    )
    critical_trajectory = [trajectory[0], replace(trajectory[1], critical_proximity=True)]
    safety_paths = write_patrol_outputs(
        tmp_path / "safety-stop",
        trajectory=critical_trajectory,
        checkpoints=checkpoints,
        collisions=[],
        summary=safety_summary,
        critical_proximities=critical_proximities,
        ttc_escape_guard_events=guard_events,
    )
    safety_payload = json.loads(safety_paths["summary"].read_text(encoding="utf-8"))
    assert safety_payload["outcome"] == "safety_stop"
    assert safety_payload["collision_count"] == 0
    assert safety_payload["collision_records"] == []
    assert safety_payload["critical_proximity_count"] == 1
    assert safety_payload["critical_proximity_records"][0]["detection_source"] == (
        "lidar_threshold"
    )
    assert safety_payload["critical_proximity_records"][0]["consecutive_frames"] == 3
    assert safety_payload["critical_proximity_records"][0]["closing_rate_mps"] == (
        pytest.approx(0.24)
    )
    assert safety_payload["critical_proximity_records"][0]["estimated_ttc_s"] == 0.0
    with safety_paths["proximity_events"].open(newline="", encoding="utf-8") as stream:
        proximity_row = next(csv.DictReader(stream))
    assert int(proximity_row["proximity_index"]) == 1
    assert proximity_row["detection_source"] == "lidar_threshold"
    assert int(proximity_row["consecutive_frames"]) == 3
    assert float(proximity_row["closing_rate_mps"]) == pytest.approx(0.24)
    assert float(proximity_row["estimated_ttc_s"]) == 0.0
    assert int(proximity_row["minimum_lidar_beam_index"]) == 11
    assert float(proximity_row["dynamic_obstacle_1_x_m"]) == pytest.approx(-0.10)

    recovered_summary = replace(
        safety_summary,
        run_id="webots-patrol-recovered-seed7000",
        outcome="completed",
        failure_reason=None,
    )
    recovered_paths = write_patrol_outputs(
        tmp_path / "recovered-after-critical-proximity",
        trajectory=critical_trajectory,
        checkpoints=checkpoints,
        collisions=[],
        summary=recovered_summary,
        critical_proximities=critical_proximities,
        ttc_escape_guard_events=guard_events,
    )
    recovered_payload = json.loads(
        recovered_paths["summary"].read_text(encoding="utf-8")
    )
    assert recovered_payload["outcome"] == "completed"
    assert recovered_payload["critical_proximity_count"] == 1

    guard_timeout_only = replace(
        critical_proximities[0],
        detection_source="ttc_escape_guard_timeout",
        consecutive_frames=0,
    )
    with pytest.raises(ValueError, match="lidar_threshold proximity event"):
        write_patrol_outputs(
            tmp_path / "missing-threshold-edge",
            trajectory=critical_trajectory,
            checkpoints=checkpoints,
            collisions=[],
            summary=safety_summary,
            critical_proximities=[guard_timeout_only],
            ttc_escape_guard_events=guard_events,
        )

    with pytest.raises(ValueError, match="critical_proximity_count"):
        write_patrol_outputs(
            tmp_path / "mismatched-proximity-count",
            trajectory=critical_trajectory,
            checkpoints=checkpoints,
            collisions=[],
            summary=safety_summary,
            critical_proximities=[],
            ttc_escape_guard_events=guard_events,
        )


def test_portable_model_path_uses_repository_relative_path(tmp_path: Path) -> None:
    model = _touch(tmp_path, "artifacts/models/webots/model.zip")

    assert portable_model_path(model, tmp_path) == "artifacts/models/webots/model.zip"
