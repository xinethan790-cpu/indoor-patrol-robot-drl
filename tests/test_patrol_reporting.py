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
    PatrolRunSummary,
    PatrolTrajectorySample,
    portable_model_path,
    select_patrol_model,
    write_patrol_outputs,
)


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
        )
    )
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
            control_step=190,
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
        control_step_count=190,
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
        safety_shield_intervention_rate=1 / 190,
    )

    with pytest.raises(ValueError, match="action-vector difference"):
        replace(trajectory[0], safety_shield_intervention=0.1)
    with pytest.raises(ValueError, match="index and angle"):
        replace(trajectory[0], minimum_lidar_angle_rad=None)
    with pytest.raises(ValueError, match="at most"):
        replace(trajectory[0], minimum_lidar_angle_rad=math.pi + 0.01)
    with pytest.raises(ValueError, match="dynamic-obstacle x/y"):
        replace(trajectory[0], dynamic_obstacle_2_y_m=None)
    with pytest.raises(ValueError, match="index and angle"):
        replace(collisions[0], minimum_lidar_beam_index=None)
    with pytest.raises(ValueError, match="dynamic-obstacle x/y"):
        replace(collisions[0], dynamic_obstacle_1_x_m=None)
    for out_of_scope_label in (
        "validated_component_outside_patrol_scenario",
        "experimental_patrol_scenario",
    ):
        with pytest.raises(ValueError, match="out-of-scope evidence"):
            replace(
                summary,
                patrol_scenario_qualified=True,
                deployment_evidence_scope=out_of_scope_label,
            )
    qualified = replace(
        summary,
        patrol_scenario_qualified=True,
        deployment_evidence_scope="qualified_patrol_scenario",
    )
    assert qualified.patrol_scenario_qualified is True

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
    ):
        legacy_trajectory_fields.pop(field_name)
    legacy_trajectory = PatrolTrajectorySample(**legacy_trajectory_fields)
    assert legacy_trajectory.policy_action_linear is None
    assert legacy_trajectory.safety_shield_active is False
    assert legacy_trajectory.minimum_lidar_beam_index is None
    assert legacy_trajectory.dynamic_obstacle_1_x_m is None

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
    ):
        legacy_summary_fields.pop(field_name)
    legacy_summary = PatrolRunSummary(**legacy_summary_fields)
    assert legacy_summary.system_deployment_id is None
    assert legacy_summary.safety_shield_enabled is False
    assert legacy_summary.safety_shield_intervention_rate == 0.0

    paths = write_patrol_outputs(
        tmp_path / "result",
        trajectory=trajectory,
        checkpoints=checkpoints,
        collisions=collisions,
        summary=summary,
    )

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
    assert payload["safety_shield_intervention_rate"] == pytest.approx(1 / 190)
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
    with paths["collisions"].open(newline="", encoding="utf-8") as stream:
        collision_row = next(csv.DictReader(stream))
    assert int(collision_row["minimum_lidar_beam_index"]) == 3
    assert float(collision_row["minimum_lidar_angle_rad"]) == pytest.approx(0.07)
    assert float(collision_row["dynamic_obstacle_1_x_m"]) == pytest.approx(0.21)
    assert float(collision_row["dynamic_obstacle_1_y_m"]) == pytest.approx(0.36)
    assert float(collision_row["dynamic_obstacle_2_x_m"]) == pytest.approx(0.95)
    assert float(collision_row["dynamic_obstacle_2_y_m"]) == pytest.approx(-0.22)


def test_portable_model_path_uses_repository_relative_path(tmp_path: Path) -> None:
    model = _touch(tmp_path, "artifacts/models/webots/model.zip")

    assert portable_model_path(model, tmp_path) == "artifacts/models/webots/model.zip"
