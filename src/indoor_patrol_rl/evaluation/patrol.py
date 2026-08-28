"""Portable result schema for the multi-checkpoint Webots patrol demo.

This module intentionally contains no Webots imports.  It keeps model selection
and result serialization testable in ordinary Python while the live controller
is executed by Webots.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, TypeVar

ModelTrainingDomain = Literal["webots", "kinematic", "unknown"]
PatrolOutcome = Literal["completed", "collision", "timeout", "error"]


@dataclass(frozen=True)
class PatrolModelSelection:
    """Resolved model plus the evidence domain in which it was trained."""

    path: Path
    training_domain: ModelTrainingDomain
    selection_reason: str


@dataclass(frozen=True)
class PatrolTrajectorySample:
    """One post-action sample from the real Webots controller loop."""

    control_step: int
    simulation_time_s: float
    position_x_m: float
    position_y_m: float
    yaw_rad: float
    checkpoint_id: str
    waypoint_index: int
    local_target_x_m: float
    local_target_y_m: float
    remaining_path_m: float
    requested_linear_mps: float
    requested_angular_radps: float
    executed_linear_mps: float
    executed_angular_radps: float
    minimum_lidar_range_m: float
    clearance_m: float
    collision: bool
    inference_latency_ms: float


@dataclass(frozen=True)
class PatrolCheckpointRecord:
    """Per-checkpoint progress and path-efficiency record."""

    checkpoint_id: str
    visit_order: int
    target_x_m: float
    target_y_m: float
    reached: bool
    reached_control_step: int | None
    arrival_simulation_time_s: float | None
    segment_control_steps: int
    actual_segment_path_m: float
    planned_segment_path_m: float | None
    segment_path_ratio: float | None
    segment_minimum_clearance_m: float | None


@dataclass(frozen=True)
class PatrolCollisionRecord:
    """A collision latched by contact points or the LiDAR fallback."""

    collision_index: int
    control_step: int
    simulation_time_s: float
    position_x_m: float
    position_y_m: float
    detection_source: str
    minimum_lidar_range_m: float


@dataclass(frozen=True)
class PatrolRunSummary:
    """Machine-readable aggregate for one Webots multi-point patrol run."""

    schema_version: str
    run_id: str
    backend: str
    planner: str
    scenario_seed: int
    model_path: str
    model_training_domain: ModelTrainingDomain
    model_selection_reason: str
    outcome: PatrolOutcome
    failure_reason: str | None
    checkpoints_total: int
    checkpoints_visited: int
    completion_ratio: float
    collision_count: int
    control_step_count: int
    physics_step_count: int
    elapsed_simulation_time_s: float
    actual_path_length_m: float
    planned_attempted_path_length_m: float
    nominal_route_length_m: float
    completed_path_ratio: float | None
    minimum_lidar_range_m: float
    minimum_clearance_m: float
    inference_latency_p50_ms: float
    inference_latency_p95_ms: float
    trajectory_file: str
    checkpoints_file: str
    collisions_file: str


def select_patrol_model(
    repository_root: Path,
    explicit_model: str | Path | None = None,
    *,
    webots_model: str | Path = "artifacts/models/webots/seed_11/ppo_webots_latest.zip",
    warm_start_model: str | Path = (
        "artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip"
    ),
) -> PatrolModelSelection:
    """Prefer a Webots-fine-tuned model and honestly label the fallback domain.

    Relative paths are resolved inside ``repository_root``.  An explicitly
    requested missing model is an error rather than a silent fallback.
    """

    root = repository_root.resolve()
    if explicit_model is not None:
        explicit_path = _inside_root(root, explicit_model)
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Explicit patrol model does not exist: {explicit_path}")
        return PatrolModelSelection(
            explicit_path,
            _infer_training_domain(explicit_path),
            "explicit WEBOTS_PATROL_MODEL override",
        )

    webots_path = _inside_root(root, webots_model)
    if webots_path.is_file():
        return PatrolModelSelection(
            webots_path,
            "webots",
            "preferred Webots fine-tuned checkpoint",
        )

    warm_start_path = _inside_root(root, warm_start_model)
    if warm_start_path.is_file():
        return PatrolModelSelection(
            warm_start_path,
            "kinematic",
            "Webots checkpoint unavailable; using kinematic stage-2 warm start",
        )
    raise FileNotFoundError(
        "No patrol model is available. Expected either "
        f"{webots_path} or {warm_start_path}."
    )


def write_patrol_outputs(
    output_directory: Path,
    *,
    trajectory: list[PatrolTrajectorySample],
    checkpoints: list[PatrolCheckpointRecord],
    collisions: list[PatrolCollisionRecord],
    summary: PatrolRunSummary,
) -> dict[str, Path]:
    """Write full-path CSV evidence and a compact JSON aggregate."""

    if not checkpoints:
        raise ValueError("A patrol run must contain at least one checkpoint record.")
    output_directory.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_directory / summary.trajectory_file
    checkpoints_path = output_directory / summary.checkpoints_file
    collisions_path = output_directory / summary.collisions_file
    summary_path = output_directory / "summary.json"
    _write_dataclass_csv(trajectory_path, trajectory, PatrolTrajectorySample)
    _write_dataclass_csv(checkpoints_path, checkpoints, PatrolCheckpointRecord)
    _write_dataclass_csv(collisions_path, collisions, PatrolCollisionRecord)
    payload = asdict(summary)
    payload["checkpoint_records"] = [asdict(record) for record in checkpoints]
    payload["collision_records"] = [asdict(record) for record in collisions]
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "trajectory": trajectory_path,
        "checkpoints": checkpoints_path,
        "collisions": collisions_path,
        "summary": summary_path,
    }


def portable_model_path(path: Path, repository_root: Path) -> str:
    """Return a repository-relative model path when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


RecordT = TypeVar("RecordT")


def _write_dataclass_csv(
    path: Path,
    records: list[RecordT],
    record_type: type[RecordT],
) -> None:
    field_names = [field.name for field in fields(record_type)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _inside_root(repository_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _infer_training_domain(path: Path) -> ModelTrainingDomain:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "webots" in parts or "ppo_webots" in name:
        return "webots"
    if "curriculum" in name or "kinematic" in name:
        return "kinematic"
    return "unknown"
