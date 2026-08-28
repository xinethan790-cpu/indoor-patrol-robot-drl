from __future__ import annotations

import csv
import json
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


def test_model_selection_prefers_webots_checkpoint(tmp_path: Path) -> None:
    warm_start = _touch(tmp_path, "artifacts/models/ppo_curriculum_stage2.zip")
    webots = _touch(tmp_path, "artifacts/models/webots/seed_11/ppo_webots_latest.zip")

    selection = select_patrol_model(
        tmp_path,
        webots_model=webots,
        warm_start_model=warm_start,
    )

    assert selection.path == webots.resolve()
    assert selection.training_domain == "webots"


def test_model_selection_labels_kinematic_fallback(tmp_path: Path) -> None:
    warm_start = _touch(tmp_path, "artifacts/models/ppo_curriculum_stage2.zip")

    selection = select_patrol_model(
        tmp_path,
        webots_model="artifacts/models/webots/missing.zip",
        warm_start_model=warm_start,
    )

    assert selection.path == warm_start.resolve()
    assert selection.training_domain == "kinematic"
    assert "warm start" in selection.selection_reason


def test_explicit_missing_model_does_not_silently_fall_back(tmp_path: Path) -> None:
    _touch(tmp_path, "artifacts/models/ppo_curriculum_stage2.zip")

    with pytest.raises(FileNotFoundError, match="Explicit patrol model"):
        select_patrol_model(tmp_path, "artifacts/models/not-there.zip")


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
        )
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
            control_step=190,
            simulation_time_s=24.32,
            position_x_m=1.5,
            position_y_m=0.4,
            detection_source="contact",
            minimum_lidar_range_m=0.12,
        )
    ]
    summary = PatrolRunSummary(
        schema_version="1.0",
        run_id="webots-patrol-seed7000",
        backend="Webots R2025a",
        planner="astar_local_waypoint_ppo",
        scenario_seed=7000,
        model_path="artifacts/models/webots/seed_11/ppo_webots_latest.zip",
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
    )

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
    assert payload["checkpoint_records"][0]["checkpoint_id"] == "P1"
    assert payload["collision_records"][0]["detection_source"] == "contact"
    with paths["trajectory"].open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["checkpoint_id"] == "P1"
    assert float(row["position_x_m"]) == pytest.approx(-1.59)


def test_portable_model_path_uses_repository_relative_path(tmp_path: Path) -> None:
    model = _touch(tmp_path, "artifacts/models/webots/model.zip")

    assert portable_model_path(model, tmp_path) == "artifacts/models/webots/model.zip"
