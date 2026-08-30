"""Portable result schema for the multi-checkpoint Webots patrol demo.

This module intentionally contains no Webots imports.  It keeps model selection
and result serialization testable in ordinary Python while the live controller
is executed by Webots.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, TypeVar

from indoor_patrol_rl.evaluation.checkpoint_selection import sha256_file

ModelTrainingDomain = Literal["webots", "kinematic", "unknown"]
PatrolOutcome = Literal["completed", "collision", "timeout", "error"]

_FORMAL_MIN_EPISODES = 20
_FORMAL_MIN_SUCCESS_RATE = 0.75
_FORMAL_MAX_COLLISION_RATE = 0.15


@dataclass(frozen=True)
class PatrolModelSelection:
    """Resolved model plus the evidence domain in which it was trained."""

    path: Path
    training_domain: ModelTrainingDomain
    selection_reason: str
    model_sha256: str
    selection_manifest: Path | None = None


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
    model_sha256: str
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
    selection_manifest: str | Path | None = (
        "artifacts/results/webots_validation_w1_selection_seed11/selection.json"
    ),
    allow_unvalidated_fallback: bool = False,
    webots_model: str | Path = "artifacts/models/webots/seed_11/ppo_webots_latest.zip",
    warm_start_model: str | Path = (
        "artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip"
    ),
) -> PatrolModelSelection:
    """Resolve an immutable, validation-selected model unless explicitly overridden.

    Relative paths are resolved inside ``repository_root``.  An explicitly
    requested missing model is an error rather than a silent fallback.  Mutable
    ``latest`` and kinematic warm-start files are available only when the caller
    explicitly enables an unvalidated exploratory fallback.
    """

    root = repository_root.resolve()
    if explicit_model is not None:
        explicit_path = _inside_root(root, explicit_model)
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Explicit patrol model does not exist: {explicit_path}")
        if not allow_unvalidated_fallback:
            raise PermissionError(
                "An explicit patrol model bypasses formal validation. Set "
                "WEBOTS_PATROL_ALLOW_UNVALIDATED_MODEL=1 for an exploratory run."
            )
        return PatrolModelSelection(
            explicit_path,
            _infer_training_domain(explicit_path),
            "UNVALIDATED explicit WEBOTS_PATROL_MODEL override",
            sha256_file(explicit_path),
        )

    if selection_manifest is not None:
        manifest_path = _inside_root(root, selection_manifest)
        if manifest_path.is_file():
            return _selection_from_manifest(root, manifest_path)
        if not allow_unvalidated_fallback:
            raise FileNotFoundError(
                "The formal patrol selection manifest does not exist: "
                f"{manifest_path}. Run checkpoint selection first, or explicitly allow "
                "an unvalidated exploratory model."
            )
    elif not allow_unvalidated_fallback:
        raise FileNotFoundError(
            "No formal patrol selection manifest was configured. Pass an explicit model "
            "for an exploratory run or explicitly allow an unvalidated fallback."
        )

    webots_path = _inside_root(root, webots_model)
    if webots_path.is_file():
        return PatrolModelSelection(
            webots_path,
            "webots",
            "UNVALIDATED fallback to mutable Webots latest checkpoint",
            sha256_file(webots_path),
        )

    warm_start_path = _inside_root(root, warm_start_model)
    if warm_start_path.is_file():
        return PatrolModelSelection(
            warm_start_path,
            "kinematic",
            "UNVALIDATED fallback to kinematic stage-2 warm start",
            sha256_file(warm_start_path),
        )
    raise FileNotFoundError(
        "No patrol model is available. Expected either "
        f"{webots_path} or {warm_start_path}."
    )


def _selection_from_manifest(
    repository_root: Path,
    manifest_path: Path,
) -> PatrolModelSelection:
    """Load a formal validation selection and verify the selected model digest."""

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read patrol selection manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise ValueError("Patrol selection manifest must contain a JSON object.")
    if raw.get("schema_version") != 1:
        raise ValueError("Patrol selection manifest schema_version must be 1.")
    contract = raw.get("validation_contract")
    if not isinstance(contract, Mapping) or contract.get("split") != "validation":
        raise ValueError("Patrol deployment accepts only a validation selection manifest.")
    episodes = _manifest_integer(contract.get("episodes"), "validation_contract.episodes")
    if episodes < _FORMAL_MIN_EPISODES:
        raise ValueError(
            f"Formal patrol selection requires at least {_FORMAL_MIN_EPISODES} episodes."
        )
    manifest_gate = raw.get("gate")
    if not isinstance(manifest_gate, Mapping):
        raise ValueError("Patrol selection manifest is missing its validation gate.")
    min_success_rate = _manifest_rate(
        manifest_gate.get("min_success_rate"), "gate.min_success_rate"
    )
    max_collision_rate = _manifest_rate(
        manifest_gate.get("max_collision_rate"), "gate.max_collision_rate"
    )
    if min_success_rate < _FORMAL_MIN_SUCCESS_RATE:
        raise ValueError(
            "Formal patrol selection requires min_success_rate >= "
            f"{_FORMAL_MIN_SUCCESS_RATE:.2f}."
        )
    if max_collision_rate > _FORMAL_MAX_COLLISION_RATE:
        raise ValueError(
            "Formal patrol selection requires max_collision_rate <= "
            f"{_FORMAL_MAX_COLLISION_RATE:.2f}."
        )
    if manifest_gate.get("requires_model_file") is not True:
        raise ValueError("Formal patrol selection must require the model file.")
    if raw.get("selection_status") != "formal_selected":
        raise ValueError("Patrol selection manifest has no gate-passing formal selection.")
    selected = raw.get("formal_selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Patrol selection manifest is missing formal_selected details.")
    if selected.get("gate_passed") is not True:
        raise ValueError("formal_selected.gate_passed must be true.")
    selected_episodes = _manifest_integer(
        selected.get("episodes"), "formal_selected.episodes"
    )
    if selected_episodes != episodes:
        raise ValueError("formal_selected.episodes must match validation_contract.episodes.")
    selected_success_rate = _manifest_rate(
        selected.get("success_rate"), "formal_selected.success_rate"
    )
    selected_collision_rate = _manifest_rate(
        selected.get("collision_rate"), "formal_selected.collision_rate"
    )
    if selected_success_rate < max(min_success_rate, _FORMAL_MIN_SUCCESS_RATE):
        raise ValueError("formal_selected.success_rate does not pass the formal gate.")
    if selected_collision_rate > min(max_collision_rate, _FORMAL_MAX_COLLISION_RATE):
        raise ValueError("formal_selected.collision_rate does not pass the formal gate.")
    contract_variant = contract.get("model_variant")
    if raw.get("model_variant") != contract_variant:
        raise ValueError("Manifest and validation contract model_variant values disagree.")
    if selected.get("model_variant") != contract_variant:
        raise ValueError("formal_selected.model_variant does not match the contract.")

    raw_model_path = selected.get("model_path")
    expected_sha256 = selected.get("model_sha256")
    if not isinstance(raw_model_path, str) or not raw_model_path:
        raise ValueError("formal_selected.model_path must be a non-empty string.")
    if not isinstance(expected_sha256, str) or re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ) is None:
        raise ValueError("formal_selected.model_sha256 must be a SHA-256 digest.")

    model_path = _inside_root(repository_root, raw_model_path)
    try:
        model_path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("A formal selected model must remain inside the repository.") from error
    try:
        model_path.relative_to((repository_root / "artifacts/models/webots").resolve())
    except ValueError as error:
        raise ValueError(
            "A formal selected model must be inside artifacts/models/webots."
        ) from error
    if not model_path.is_file():
        raise FileNotFoundError(f"Formal selected patrol model does not exist: {model_path}")
    actual_sha256 = sha256_file(model_path)
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise ValueError(
            "Formal selected patrol model failed SHA-256 verification: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    return PatrolModelSelection(
        path=model_path,
        training_domain="webots",
        selection_reason="gate-passing formal validation selection",
        model_sha256=actual_sha256,
        selection_manifest=manifest_path,
    )


def _manifest_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _manifest_rate(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite rate between zero and one.")
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{label} must be a finite rate between zero and one.")
    return rate


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
