#!/usr/bin/env python3
"""Analyze the locked recovery-v1 Webots risk collection without running Webots.

The collection report and its adjacent SHA-256 sidecar are the trust root for
the diagnostic ``episode_context.jsonl`` schema 1.1 and matching
``episodes.csv``.  This analyzer diagnoses one preregistered model; it never
selects a checkpoint or accepts validation/held-out test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from indoor_patrol_rl.evaluation.episode_context import (
    EPISODE_CONTEXT_SCHEMA_VERSION,
    RISK_TRACE_COORDINATE_SEMANTICS,
    RISK_TRACE_SAMPLE_SEMANTICS,
    validate_risk_trace,
)
from indoor_patrol_rl.evaluation.metrics import EpisodeRecord

PROTOCOL_NAME = "relative_motion_cpa_v1"
COLLECTION_EXPERIMENT_ID = "webots_recovery_v1_risk_trajectory_v1"
COLLECTION_STATUS = "collection_complete"
RECOVERY_MODEL_PATH = (
    "artifacts/models/webots/seed_11/w1_recovery_v1/"
    "ppo_webots_s11_step000040960.zip"
)
RECOVERY_MODEL_SHA256 = (
    "c5e7c3de42c7664477097b2734975660fec840ad5b7f45425098158c27cb7275"
)
RECOVERY_MODEL_VARIANT = "w1_recovery_v1"
RECOVERY_TRAINING_SEED = 11
RECOVERY_CONDITION_ID = "DR1"
RECOVERY_PLANNER = "ppo_recovery_v1_risk_trace"
DIAGNOSTIC_SEED_START = 20_494
DIAGNOSTIC_EPISODES = 120
DIAGNOSTIC_SEEDS = tuple(
    range(DIAGNOSTIC_SEED_START, DIAGNOSTIC_SEED_START + DIAGNOSTIC_EPISODES)
)
EXPECTED_AXIS_COUNTS = {"horizontal": 60, "vertical": 60}
RESERVED_VALIDATION_SEEDS = frozenset(range(5_000, 5_020))
RESERVED_TEST_SEEDS = frozenset(range(10_000, 10_020))
ROBOT_RADIUS_M = 0.122
OBSTACLE_RADIUS_M = 0.20
CPA_HORIZON_S = 1.0
PROXY_HORIZON_S = 0.5
GEOMETRY_TOLERANCE_M = 0.02
LIDAR_TRIGGER_MARGIN_M = 0.01
RISK_PERSISTENCE_TICKS = 2
RESPONSE_WINDOW_S = 0.512
RESPONSE_BASELINE_TICKS = 3
DECELERATION_FRACTION = 0.30
DOMINANCE_FRACTION = 2.0 / 3.0
MINIMUM_CLASSIFIABLE_FAILURES = 24
MAXIMUM_AMBIGUOUS_FRACTION = 0.25
_EPSILON = 1e-9
_TIME_TOLERANCE = 1e-6
_SHA256_HEX_LENGTH = 64
_EPISODE_FIELDS = tuple(EpisodeRecord.__dataclass_fields__)
_CONTEXT_FIELDS = {
    "schema_version",
    "identity",
    "route_axis",
    "reverse_route",
    "start_xy",
    "goal_xy",
    "start_yaw",
    "dynamic_obstacles",
    "termination",
    "risk_trace",
}
_IDENTITY_FIELDS = {
    "episode_id",
    "split",
    "condition_id",
    "scenario_seed",
    "planner",
}
_TERMINATION_FIELDS = {
    "contact_collision",
    "lidar_collision",
    "terminal_step",
    "terminal_xy",
    "terminal_yaw",
    "goal_distance",
    "policy_action",
    "executed_action",
}
_DYNAMIC_OBSTACLE_FIELDS = {
    "def_name",
    "motion_axis",
    "initial_xy",
    "speed",
    "initial_direction",
    "lower",
    "upper",
}


@dataclass(frozen=True)
class EpisodeRow:
    episode_id: str
    split: str
    condition_id: str
    scenario_seed: int
    planner: str
    outcome: str


@dataclass(frozen=True)
class VelocityState:
    robot: np.ndarray
    obstacles: tuple[np.ndarray, ...]
    source: str


@dataclass(frozen=True)
class LockedCollection:
    report: dict[str, Any]
    report_path: Path
    report_sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    episodes_path: Path
    episodes_sha256: str
    context_path: Path
    context_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_literal(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _portable_path(value: Any, *, repository_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX repository path.")
    candidate = Path(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{label} must be a normalized relative repository path.")
    root = repository_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository root.") from error
    if resolved.relative_to(root).as_posix() != value:
        raise ValueError(f"{label} must be a normalized POSIX repository path.")
    return resolved


def _file_binding(
    value: Any,
    *,
    repository_root: Path,
    label: str,
    verify_file: bool,
) -> tuple[Path, str]:
    binding = _mapping(value, label)
    _exact_keys(binding, {"path", "sha256"}, label)
    path = _portable_path(
        binding["path"], repository_root=repository_root, label=f"{label}.path"
    )
    expected_sha256 = _sha256_literal(binding["sha256"], f"{label}.sha256")
    if verify_file:
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, "
                f"actual {actual_sha256}."
            )
    return path, expected_sha256


def _verify_collection_sidecar(report_path: Path) -> tuple[Path, str, str]:
    if report_path.name != "collection_report.json":
        raise ValueError("The locked report filename must be collection_report.json.")
    sidecar_path = report_path.with_suffix(".sha256")
    yaml_path = report_path.with_suffix(".yaml")
    if not sidecar_path.is_file() or not yaml_path.is_file():
        raise FileNotFoundError(
            "The locked collection report requires adjacent YAML and SHA-256 sidecar files."
        )
    lines = sidecar_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2:
        raise ValueError("collection_report.sha256 must contain exactly two bindings.")
    bindings: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ")
        if len(parts) != 2 or not parts[1] or parts[1] in bindings:
            raise ValueError("collection_report.sha256 has an invalid binding line.")
        bindings[parts[1]] = _sha256_literal(parts[0], "collection report sidecar digest")
    expected_names = {report_path.name, yaml_path.name}
    if set(bindings) != expected_names:
        raise ValueError(
            "collection_report.sha256 must bind collection_report.json and "
            "collection_report.yaml exactly."
        )
    report_sha256 = _sha256_file(report_path)
    yaml_sha256 = _sha256_file(yaml_path)
    if bindings[report_path.name] != report_sha256:
        raise ValueError("collection_report.json disagrees with its SHA-256 sidecar.")
    if bindings[yaml_path.name] != yaml_sha256:
        raise ValueError("collection_report.yaml disagrees with its SHA-256 sidecar.")
    return sidecar_path, _sha256_file(sidecar_path), report_sha256


def _expected_axis(seed: int) -> str:
    return str(np.random.default_rng(seed).choice(["horizontal", "vertical"]))


def load_locked_collection(
    report_path: Path,
    *,
    repository_root: Path,
) -> LockedCollection:
    """Validate the immutable recovery-v1 collection manifest and its artifacts."""

    root = repository_root.resolve()
    report_path = report_path.resolve()
    try:
        report_relative = report_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("collection_report.json must remain inside the repository.") from error
    if not report_path.is_file():
        raise FileNotFoundError(f"Locked collection report does not exist: {report_path}")
    sidecar_path, sidecar_sha256, report_sha256 = _verify_collection_sidecar(
        report_path
    )
    report = _mapping(
        json.loads(report_path.read_text(encoding="utf-8")), "collection_report"
    )
    _exact_keys(
        report,
        {
            "schema_version",
            "experiment_id",
            "status",
            "generated_at_utc",
            "split",
            "validation_split_was_run",
            "test_split_was_run",
            "performance_gate_applied",
            "source_bindings",
            "scenario",
            "collection",
            "model",
            "execution",
            "descriptive_metrics_not_used_as_a_gate",
            "artifacts",
            "access",
            "written_reports",
        },
        "collection_report",
    )
    expected_scalars = {
        "schema_version": 1,
        "experiment_id": COLLECTION_EXPERIMENT_ID,
        "status": COLLECTION_STATUS,
        "split": "diagnostic",
        "validation_split_was_run": False,
        "test_split_was_run": False,
        "performance_gate_applied": False,
    }
    for name, expected in expected_scalars.items():
        if report[name] != expected:
            raise PermissionError(
                f"collection_report.{name} must equal the locked value {expected!r}."
            )
    if not isinstance(report["generated_at_utc"], str) or not report["generated_at_utc"]:
        raise ValueError("collection_report.generated_at_utc must be non-empty.")

    access = _mapping(report["access"], "collection_report.access")
    _exact_keys(access, {"validation_or_test"}, "collection_report.access")
    if access["validation_or_test"] != "forbidden":
        raise PermissionError("Validation and held-out test access must be forbidden.")

    model = _mapping(report["model"], "collection_report.model")
    _exact_keys(
        model,
        {"path", "sha256", "variant", "training_seed"},
        "collection_report.model",
    )
    expected_model = {
        "path": RECOVERY_MODEL_PATH,
        "sha256": RECOVERY_MODEL_SHA256,
        "variant": RECOVERY_MODEL_VARIANT,
        "training_seed": RECOVERY_TRAINING_SEED,
    }
    if model != expected_model:
        raise PermissionError("collection_report.model is not the locked recovery-v1 model.")

    source_bindings = _mapping(
        report["source_bindings"], "collection_report.source_bindings"
    )
    _exact_keys(
        source_bindings,
        {
            "git_revision",
            "config",
            "runner",
            "controller",
            "world",
            "model",
            "webots_executable",
        },
        "collection_report.source_bindings",
    )
    git_revision = source_bindings["git_revision"]
    if (
        not isinstance(git_revision, str)
        or len(git_revision) != 40
        or any(character not in "0123456789abcdef" for character in git_revision)
    ):
        raise ValueError("source_bindings.git_revision must be a lowercase Git SHA.")
    for label in ("config", "runner", "controller", "world"):
        _file_binding(
            source_bindings[label],
            repository_root=root,
            label=f"collection_report.source_bindings.{label}",
            verify_file=False,
        )
    source_model = _mapping(
        source_bindings["model"], "collection_report.source_bindings.model"
    )
    _exact_keys(
        source_model, {"path", "sha256"}, "collection_report.source_bindings.model"
    )
    if source_model != {
        "path": RECOVERY_MODEL_PATH,
        "sha256": RECOVERY_MODEL_SHA256,
    }:
        raise PermissionError("source_bindings.model is not the locked recovery-v1 model.")
    webots_binding = _mapping(
        source_bindings["webots_executable"],
        "collection_report.source_bindings.webots_executable",
    )
    _exact_keys(
        webots_binding,
        {"path", "sha256"},
        "collection_report.source_bindings.webots_executable",
    )
    if not isinstance(webots_binding["path"], str) or not webots_binding["path"]:
        raise ValueError("source_bindings.webots_executable.path must be non-empty.")
    _sha256_literal(
        webots_binding["sha256"], "source_bindings.webots_executable.sha256"
    )

    expected_seeds = list(DIAGNOSTIC_SEEDS)
    scenario = _mapping(report["scenario"], "collection_report.scenario")
    _exact_keys(
        scenario,
        {
            "condition_id",
            "scenario_seed_start",
            "episodes",
            "scenario_seeds",
            "axis_counts",
            "obstacle_count",
            "obstacle_speed_range",
            "horizontal_route_half_length",
            "vertical_route_half_length",
            "safety_shield",
            "deterministic_policy",
        },
        "collection_report.scenario",
    )
    scenario_expected = {
        "condition_id": RECOVERY_CONDITION_ID,
        "scenario_seed_start": DIAGNOSTIC_SEED_START,
        "episodes": DIAGNOSTIC_EPISODES,
        "scenario_seeds": expected_seeds,
        "axis_counts": EXPECTED_AXIS_COUNTS,
        "obstacle_count": 1,
        "obstacle_speed_range": [0.15, 0.30],
        "horizontal_route_half_length": 1.10,
        "vertical_route_half_length": 1.10,
        "safety_shield": False,
        "deterministic_policy": True,
    }
    if scenario != scenario_expected:
        raise PermissionError("collection_report.scenario is not the locked diagnostic plan.")
    if set(expected_seeds) & (RESERVED_VALIDATION_SEEDS | RESERVED_TEST_SEEDS):
        raise RuntimeError("Locked diagnostic seeds overlap a reserved evaluation split.")

    collection = _mapping(report["collection"], "collection_report.collection")
    _exact_keys(
        collection,
        {
            "expected_episode_count",
            "episodes_csv_count",
            "episode_context_count",
            "risk_trace_count",
            "episode_context_schema_version",
            "expected_route_axis_counts",
            "adaptive_extension",
            "complete",
        },
        "collection_report.collection",
    )
    expected_collection = {
        "expected_episode_count": DIAGNOSTIC_EPISODES,
        "episodes_csv_count": DIAGNOSTIC_EPISODES,
        "episode_context_count": DIAGNOSTIC_EPISODES,
        "risk_trace_count": DIAGNOSTIC_EPISODES,
        "episode_context_schema_version": EPISODE_CONTEXT_SCHEMA_VERSION,
        "expected_route_axis_counts": EXPECTED_AXIS_COUNTS,
        "adaptive_extension": "forbidden",
        "complete": True,
    }
    if collection != expected_collection:
        raise PermissionError("collection_report.collection is incomplete or out of scope.")

    execution = _mapping(report["execution"], "collection_report.execution")
    _exact_keys(
        execution,
        {"command", "environment", "port", "exit_code"},
        "collection_report.execution",
    )
    if execution["exit_code"] != 0:
        raise ValueError("collection_report.execution.exit_code must be zero.")
    environment = _mapping(
        execution["environment"], "collection_report.execution.environment"
    )
    expected_environment = {
        "WEBOTS_MODEL_SEED": str(RECOVERY_TRAINING_SEED),
        "WEBOTS_EVAL_MODEL": RECOVERY_MODEL_PATH,
        "WEBOTS_EVAL_MODEL_SHA256": RECOVERY_MODEL_SHA256,
        "WEBOTS_MODEL_VARIANT": RECOVERY_MODEL_VARIANT,
        "WEBOTS_EVAL_SPLIT": "diagnostic",
        "WEBOTS_EVAL_SEED_START": str(DIAGNOSTIC_SEED_START),
        "WEBOTS_EVAL_EPISODES": str(DIAGNOSTIC_EPISODES),
        "WEBOTS_EVAL_CONDITION": RECOVERY_CONDITION_ID,
        "WEBOTS_EVAL_PLANNER_LABEL": RECOVERY_PLANNER,
        "WEBOTS_SAFETY_SHIELD": "0",
        "WEBOTS_EVAL_DETERMINISTIC_POLICY": "1",
        "WEBOTS_EVAL_EPISODE_CONTEXT": "1",
        "WEBOTS_RISK_TRAJECTORY_ID": COLLECTION_EXPERIMENT_ID,
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise PermissionError(
                f"collection_report.execution.environment.{name} is out of scope."
            )

    artifacts = _mapping(report["artifacts"], "collection_report.artifacts")
    _exact_keys(
        artifacts,
        {"result_directory", "summary_json", "episodes_csv", "episode_context_jsonl"},
        "collection_report.artifacts",
    )
    result_directory = _portable_path(
        artifacts["result_directory"],
        repository_root=root,
        label="collection_report.artifacts.result_directory",
    )
    summary_path, _summary_sha256 = _file_binding(
        artifacts["summary_json"],
        repository_root=root,
        label="collection_report.artifacts.summary_json",
        verify_file=True,
    )
    episodes_path, episodes_sha256 = _file_binding(
        artifacts["episodes_csv"],
        repository_root=root,
        label="collection_report.artifacts.episodes_csv",
        verify_file=True,
    )
    context_path, context_sha256 = _file_binding(
        artifacts["episode_context_jsonl"],
        repository_root=root,
        label="collection_report.artifacts.episode_context_jsonl",
        verify_file=True,
    )
    expected_artifact_names = {
        summary_path: "summary.json",
        episodes_path: "episodes.csv",
        context_path: "episode_context.jsonl",
    }
    if any(path.name != name for path, name in expected_artifact_names.items()) or any(
        path.parent != result_directory for path in expected_artifact_names
    ):
        raise ValueError("Collection artifacts must use fixed names in result_directory.")

    written_reports = _mapping(
        report["written_reports"], "collection_report.written_reports"
    )
    _exact_keys(
        written_reports,
        {"json", "yaml", "sha256_sidecar"},
        "collection_report.written_reports",
    )
    expected_written_reports = {
        "json": report_relative,
        "yaml": report_path.with_suffix(".yaml").relative_to(root).as_posix(),
        "sha256_sidecar": sidecar_path.relative_to(root).as_posix(),
    }
    if written_reports != expected_written_reports:
        raise ValueError("collection_report.written_reports does not bind adjacent reports.")
    return LockedCollection(
        report=report,
        report_path=report_path,
        report_sha256=report_sha256,
        sidecar_path=sidecar_path,
        sidecar_sha256=sidecar_sha256,
        episodes_path=episodes_path,
        episodes_sha256=episodes_sha256,
        context_path=context_path,
        context_sha256=context_sha256,
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"{label} schema mismatch; missing={missing}, unknown={unknown}.")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _vector(value: Any, label: str, *, length: int = 2) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must be a {length}-value array.")
    result = np.asarray([_finite(item, f"{label}[{index}]") for index, item in enumerate(value)])
    if result.shape != (length,):
        raise ValueError(f"{label} must be a {length}-value array.")
    return result


def _parse_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{label} must be True or False.")


def read_episode_rows(path: Path) -> dict[str, EpisodeRow]:
    """Read exact EpisodeRecord v1.0 rows for one diagnostic run."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _EPISODE_FIELDS:
            raise ValueError("episodes.csv schema does not match EpisodeRecord v1.0.")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("episodes.csv contains no episodes.")
    rows: dict[str, EpisodeRow] = {}
    seen_seeds: set[int] = set()
    for index, raw in enumerate(raw_rows):
        if raw["schema_version"] != "1.0":
            raise ValueError(f"episodes.csv row {index} schema_version must be 1.0.")
        if raw["split"] != "diagnostic":
            raise PermissionError("Risk trajectory analysis only permits diagnostic rows.")
        try:
            seed = int(raw["scenario_seed"])
        except ValueError as error:
            raise ValueError(f"episodes.csv row {index} scenario_seed is invalid.") from error
        if seed < 0 or seed in seen_seeds:
            raise ValueError("episodes.csv scenario seeds must be unique and non-negative.")
        seen_seeds.add(seed)
        episode_id = raw["episode_id"]
        if not episode_id or episode_id in rows:
            raise ValueError("episodes.csv episode_id values must be non-empty and unique.")
        outcome = raw["outcome"]
        if outcome not in {"success", "collision", "timeout", "error"}:
            raise ValueError(f"episodes.csv row {index} outcome is invalid.")
        success = _parse_bool(raw["is_success"], f"episodes.csv row {index} is_success")
        collision = _parse_bool(raw["collision"], f"episodes.csv row {index} collision")
        timeout = _parse_bool(raw["timeout"], f"episodes.csv row {index} timeout")
        if (
            success != (outcome == "success")
            or collision != (outcome == "collision")
            or timeout != (outcome == "timeout")
        ):
            raise ValueError(f"episodes.csv row {index} outcome flags are inconsistent.")
        rows[episode_id] = EpisodeRow(
            episode_id=episode_id,
            split="diagnostic",
            condition_id=raw["condition_id"],
            scenario_seed=seed,
            planner=raw["planner"],
            outcome=outcome,
        )
    return rows


def _validate_window_timing(
    window: Sequence[Mapping[str, Any]],
    *,
    sample_period_s: float,
    label: str,
) -> None:
    for index, sample in enumerate(window):
        step = int(sample["step"])
        elapsed = float(sample["post_elapsed_s"])
        if index and step != int(window[index - 1]["step"]) + 1:
            raise ValueError(f"{label} must contain consecutive control steps.")
        expected = step * sample_period_s
        if bool(sample["post_contact_collision"]):
            lower = (step - 1) * sample_period_s
            if not lower < elapsed <= expected + _TIME_TOLERANCE:
                raise ValueError(f"{label}[{index}] contact-step timing is invalid.")
        elif not math.isclose(elapsed, expected, rel_tol=0.0, abs_tol=_TIME_TOLERANCE):
            raise ValueError(
                f"{label}[{index}] post_elapsed_s breaks interval action semantics."
            )


def _deduplicated_samples(
    trace: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, set[str]]]:
    sample_period = float(trace["sample_period_s"])
    by_step: dict[int, dict[str, Any]] = {}
    memberships: dict[int, set[str]] = defaultdict(set)
    for window_name in ("terminal_window", "minimum_clearance_window"):
        window = trace[window_name]
        _validate_window_timing(
            window,
            sample_period_s=sample_period,
            label=f"risk_trace.{window_name}",
        )
        for sample in window:
            step = int(sample["step"])
            existing = by_step.get(step)
            if existing is not None and existing != sample:
                raise ValueError(
                    f"Risk windows disagree for duplicated control step {step}."
                )
            by_step[step] = dict(sample)
            memberships[step].add(window_name)
    return [by_step[step] for step in sorted(by_step)], memberships


def _validate_context_record(
    raw: Any,
    *,
    index: int,
    episode: EpisodeRow,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, set[str]]]:
    record = _mapping(raw, f"episode_context row {index}")
    _exact_keys(record, _CONTEXT_FIELDS, f"episode_context row {index}")
    if record["schema_version"] != EPISODE_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            f"episode_context row {index} must use schema "
            f"{EPISODE_CONTEXT_SCHEMA_VERSION}."
        )
    identity = _mapping(record["identity"], f"episode_context row {index}.identity")
    _exact_keys(identity, _IDENTITY_FIELDS, f"episode_context row {index}.identity")
    expected_identity = {
        "episode_id": episode.episode_id,
        "split": "diagnostic",
        "condition_id": episode.condition_id,
        "scenario_seed": episode.scenario_seed,
        "planner": episode.planner,
    }
    if identity != expected_identity:
        raise ValueError(f"episode_context row {index} identity disagrees with episodes.csv.")
    if record["route_axis"] not in {"horizontal", "vertical"}:
        raise ValueError(f"episode_context row {index} route_axis is invalid.")
    if type(record["reverse_route"]) is not bool:
        raise ValueError(f"episode_context row {index} reverse_route must be boolean.")
    _vector(record["start_xy"], f"episode_context row {index}.start_xy")
    _vector(record["goal_xy"], f"episode_context row {index}.goal_xy")
    _finite(record["start_yaw"], f"episode_context row {index}.start_yaw")

    obstacles = record["dynamic_obstacles"]
    if not isinstance(obstacles, list) or not obstacles:
        raise ValueError("Risk protocol requires at least one dynamic obstacle.")
    obstacle_names: set[str] = set()
    for obstacle_index, obstacle_value in enumerate(obstacles):
        obstacle = _mapping(
            obstacle_value,
            f"episode_context row {index}.dynamic_obstacles[{obstacle_index}]",
        )
        _exact_keys(
            obstacle,
            _DYNAMIC_OBSTACLE_FIELDS,
            f"episode_context row {index}.dynamic_obstacles[{obstacle_index}]",
        )
        name = obstacle["def_name"]
        if not isinstance(name, str) or not name or name in obstacle_names:
            raise ValueError("Dynamic obstacle def_name values must be non-empty and unique.")
        obstacle_names.add(name)
        if obstacle["motion_axis"] not in {"x", "y"}:
            raise ValueError("Dynamic obstacle motion_axis must be x or y.")
        _vector(obstacle["initial_xy"], "dynamic_obstacle.initial_xy")
        if _finite(obstacle["speed"], "dynamic_obstacle.speed") < 0.0:
            raise ValueError("Dynamic obstacle speed cannot be negative.")
        if _finite(obstacle["initial_direction"], "dynamic_obstacle.initial_direction") not in {
            -1.0,
            1.0,
        }:
            raise ValueError("Dynamic obstacle initial_direction must be -1 or 1.")
        lower = _finite(obstacle["lower"], "dynamic_obstacle.lower")
        upper = _finite(obstacle["upper"], "dynamic_obstacle.upper")
        if lower >= upper:
            raise ValueError("Dynamic obstacle motion bounds must be ordered.")

    termination = _mapping(
        record["termination"], f"episode_context row {index}.termination"
    )
    _exact_keys(termination, _TERMINATION_FIELDS, f"episode_context row {index}.termination")
    for name in ("contact_collision", "lidar_collision"):
        if type(termination[name]) is not bool:
            raise ValueError(f"termination.{name} must be boolean.")
    if type(termination["terminal_step"]) is not int or termination["terminal_step"] < 1:
        raise ValueError("termination.terminal_step must be positive.")
    _vector(termination["terminal_xy"], "termination.terminal_xy")
    _finite(termination["terminal_yaw"], "termination.terminal_yaw")
    _finite(termination["goal_distance"], "termination.goal_distance")
    _vector(termination["policy_action"], "termination.policy_action")
    _vector(termination["executed_action"], "termination.executed_action")

    trace = _mapping(record["risk_trace"], f"episode_context row {index}.risk_trace")
    validate_risk_trace(trace, field=f"episode_context row {index}.risk_trace")
    if trace["sample_semantics"] != RISK_TRACE_SAMPLE_SEMANTICS:
        raise ValueError("Risk trace has unsupported pre-action/action/post-state semantics.")
    if trace["coordinate_semantics"] != RISK_TRACE_COORDINATE_SEMANTICS:
        raise ValueError("Risk trace has unsupported robot coordinate semantics.")
    samples, memberships = _deduplicated_samples(trace)
    lidar_collision_threshold = (
        float(trace["lidar_min_range_m"]) + LIDAR_TRIGGER_MARGIN_M
    )
    for sample_index, sample in enumerate(samples):
        if len(sample["post_obstacle_xy"]) != len(obstacles):
            raise ValueError(
                f"Risk sample {sample_index} obstacle count disagrees with reset context."
            )
        expected_lidar_collision = bool(
            float(sample["post_lidar_argmin"]["range_m"])
            <= lidar_collision_threshold + 1e-12
        )
        if bool(sample["post_lidar_collision"]) != expected_lidar_collision:
            raise ValueError(
                f"Risk sample {sample_index} LiDAR collision flag disagrees with "
                "risk_trace.lidar_min_range_m plus the protocol trigger margin."
            )
    terminal_sample = trace["terminal_window"][-1]
    if int(terminal_sample["step"]) != termination["terminal_step"]:
        raise ValueError("Terminal risk sample step disagrees with termination context.")
    terminal_checks = {
        "post_robot_gps_xy": termination["terminal_xy"],
        "post_robot_yaw": termination["terminal_yaw"],
        "post_goal_distance": termination["goal_distance"],
        "interval_policy_action": termination["policy_action"],
        "interval_executed_action": termination["executed_action"],
        "post_contact_collision": termination["contact_collision"],
        "post_lidar_collision": termination["lidar_collision"],
    }
    for field, expected in terminal_checks.items():
        actual = terminal_sample[field]
        if isinstance(expected, list):
            if not np.allclose(actual, expected, rtol=0.0, atol=1e-9):
                raise ValueError(f"Terminal risk sample {field} disagrees with termination.")
        elif isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"Terminal risk sample {field} disagrees with termination.")
        elif actual != expected:
            raise ValueError(f"Terminal risk sample {field} disagrees with termination.")
    if episode.outcome == "collision" and not (
        termination["contact_collision"] or termination["lidar_collision"]
    ):
        raise ValueError("Collision outcome lacks both contact and LiDAR collision evidence.")
    if episode.outcome != "collision" and (
        termination["contact_collision"] or termination["lidar_collision"]
    ):
        raise ValueError("Non-collision outcome carries collision termination evidence.")
    return record, samples, memberships


def read_context_records(
    path: Path,
    *,
    episodes: Mapping[str, EpisodeRow],
) -> list[tuple[dict[str, Any], EpisodeRow, list[dict[str, Any]], dict[int, set[str]]]]:
    values: list[Any] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"episode_context.jsonl line {line_number} is blank.")
            values.append(json.loads(line))
    if len(values) != len(episodes):
        raise ValueError("episode_context.jsonl count does not match episodes.csv.")
    result = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        candidate = _mapping(raw, f"episode_context row {index}")
        identity = _mapping(candidate.get("identity"), f"episode_context row {index}.identity")
        episode_id = identity.get("episode_id")
        if not isinstance(episode_id, str) or episode_id not in episodes or episode_id in seen:
            raise ValueError("episode_context episode_id is missing, unknown, or duplicated.")
        seen.add(episode_id)
        episode = episodes[episode_id]
        record, samples, memberships = _validate_context_record(
            candidate,
            index=index,
            episode=episode,
        )
        result.append((record, episode, samples, memberships))
    if seen != set(episodes):
        raise ValueError("episode_context.jsonl does not cover every episode exactly once.")
    return result


def _validate_locked_episode_scope(
    episodes: Mapping[str, EpisodeRow],
    contexts: Sequence[
        tuple[dict[str, Any], EpisodeRow, list[dict[str, Any]], dict[int, set[str]]]
    ],
) -> None:
    seeds = {episode.scenario_seed for episode in episodes.values()}
    reserved = seeds & (RESERVED_VALIDATION_SEEDS | RESERVED_TEST_SEEDS)
    if reserved:
        raise PermissionError(
            f"Risk analysis rejects reserved validation/test seeds: {sorted(reserved)}."
        )
    if seeds != set(DIAGNOSTIC_SEEDS) or len(episodes) != DIAGNOSTIC_EPISODES:
        raise PermissionError(
            "Formal mechanism inference requires exact diagnostic seeds "
            f"{DIAGNOSTIC_SEEDS[0]}..{DIAGNOSTIC_SEEDS[-1]} ({DIAGNOSTIC_EPISODES})."
        )
    for episode in episodes.values():
        expected_episode_id = (
            f"diagnostic-{RECOVERY_CONDITION_ID}-{episode.scenario_seed}-"
            f"{RECOVERY_PLANNER}"
        )
        if (
            episode.condition_id != RECOVERY_CONDITION_ID
            or episode.planner != RECOVERY_PLANNER
            or episode.episode_id != expected_episode_id
        ):
            raise PermissionError(
                f"Episode seed {episode.scenario_seed} is outside the locked recovery-v1 run."
            )
    axis_counts = Counter(record["route_axis"] for record, *_rest in contexts)
    if dict(axis_counts) != EXPECTED_AXIS_COUNTS:
        raise PermissionError(
            f"Formal mechanism inference requires H/V=60/60, got {dict(axis_counts)}."
        )
    for record, episode, *_rest in contexts:
        expected_axis = _expected_axis(episode.scenario_seed)
        if record["route_axis"] != expected_axis:
            raise PermissionError(
                f"Seed {episode.scenario_seed} route axis disagrees with the locked plan."
            )


def _velocity_state(
    *,
    record: Mapping[str, Any],
    sample: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> VelocityState | None:
    step = int(sample["step"])
    if previous is not None and int(previous["step"]) + 1 == step:
        dt = float(sample["post_elapsed_s"]) - float(previous["post_elapsed_s"])
        previous_robot = _vector(
            previous["post_robot_root_xy"], "previous.post_robot_root_xy"
        )
        previous_obstacles = [
            _vector(value, "previous.post_obstacle_xy")
            for value in previous["post_obstacle_xy"]
        ]
        source = "adjacent_post_truth"
    elif step == 1:
        dt = float(sample["post_elapsed_s"])
        previous_robot = _vector(record["start_xy"], "start_xy")
        previous_obstacles = [
            _vector(value["initial_xy"], "dynamic_obstacle.initial_xy")
            for value in record["dynamic_obstacles"]
        ]
        source = "reset_to_post_truth"
    else:
        return None
    if dt <= 0.0 or len(previous_obstacles) != len(sample["post_obstacle_xy"]):
        return None
    robot = (
        _vector(sample["post_robot_root_xy"], "sample.post_robot_root_xy")
        - previous_robot
    ) / dt
    obstacles = tuple(
        (_vector(current, "sample.post_obstacle_xy") - old) / dt
        for current, old in zip(sample["post_obstacle_xy"], previous_obstacles, strict=True)
    )
    if not np.isfinite(robot).all() or any(not np.isfinite(value).all() for value in obstacles):
        raise ValueError("Finite-difference velocity became non-finite.")
    return VelocityState(robot=robot, obstacles=obstacles, source=source)


def _relative_metrics(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    horizon_s: float,
) -> dict[str, Any]:
    distance = float(np.linalg.norm(relative_position))
    combined_radius = ROBOT_RADIUS_M + OBSTACLE_RADIUS_M
    gap = distance - combined_radius
    if distance <= _EPSILON:
        closing = 0.0
    else:
        closing = -float(np.dot(relative_position, relative_velocity)) / distance

    # Exact constant-velocity first contact between two 2-D circular footprints:
    # ||relative_position + relative_velocity * t||^2 = combined_radius^2.
    quadratic_a = float(np.dot(relative_velocity, relative_velocity))
    quadratic_b = 2.0 * float(np.dot(relative_position, relative_velocity))
    quadratic_c = float(np.dot(relative_position, relative_position)) - combined_radius**2
    if quadratic_c <= 0.0:
        ttc = 0.0
        ttc_defined = True
    elif quadratic_a == 0.0:
        ttc = None
        ttc_defined = False
    else:
        discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
        if discriminant < -_EPSILON:
            ttc = None
            ttc_defined = False
        else:
            square_root = math.sqrt(max(discriminant, 0.0))
            roots = (
                (-quadratic_b - square_root) / (2.0 * quadratic_a),
                (-quadratic_b + square_root) / (2.0 * quadratic_a),
            )
            future_roots = [max(root, 0.0) for root in roots if root >= -_EPSILON]
            if future_roots:
                ttc = min(future_roots)
                ttc_defined = True
            else:
                ttc = None
                ttc_defined = False

    if quadratic_a == 0.0:
        t_cpa = 0.0
    else:
        raw_t_cpa = -float(np.dot(relative_position, relative_velocity)) / quadratic_a
        t_cpa = min(max(raw_t_cpa, 0.0), horizon_s)
    cpa_gap = (
        float(np.linalg.norm(relative_position + relative_velocity * t_cpa))
        - ROBOT_RADIUS_M
        - OBSTACLE_RADIUS_M
    )
    return {
        "gap_m": gap,
        "closing_rate_mps": closing,
        "ttc_s": ttc,
        "ttc_defined": ttc_defined,
        "t_cpa_s": t_cpa,
        "cpa_clearance_m": cpa_gap,
    }


def _ray_circle_intersection_range(
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    centre: np.ndarray,
    radius: float,
) -> float | None:
    relative = centre - origin
    projection = float(np.dot(relative, direction))
    perpendicular_squared = float(np.dot(relative, relative)) - projection**2
    if projection <= 0.0 or perpendicular_squared > radius**2:
        return None
    offset = math.sqrt(max(radius**2 - perpendicular_squared, 0.0))
    near = projection - offset
    return near if near >= 0.0 else None


def _attribute_lidar_ray(
    sample: Mapping[str, Any],
    obstacle_names: Sequence[str],
) -> str | None:
    origin = _vector(sample["post_robot_root_xy"], "post_robot_root_xy")
    yaw = float(sample["post_robot_yaw"])
    lidar = sample["post_lidar_argmin"]
    bearing = float(lidar["bearing_rad"])
    measured_range = float(lidar["range_m"])
    direction = np.asarray([math.cos(yaw + bearing), math.sin(yaw + bearing)])
    matches: list[str] = []
    for name, centre_value in zip(
        obstacle_names, sample["post_obstacle_xy"], strict=True
    ):
        expected = _ray_circle_intersection_range(
            origin=origin,
            direction=direction,
            centre=_vector(centre_value, "post_obstacle_xy"),
            radius=OBSTACLE_RADIUS_M,
        )
        if expected is not None and abs(expected - measured_range) <= GEOMETRY_TOLERANCE_M:
            matches.append(name)
    return matches[0] if len(matches) == 1 else None


def _derive_samples(
    record: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    memberships: Mapping[int, set[str]],
) -> list[dict[str, Any]]:
    obstacle_names = [str(value["def_name"]) for value in record["dynamic_obstacles"]]
    lidar_minimum = float(record["risk_trace"]["lidar_min_range_m"])
    lidar_maximum = float(record["risk_trace"]["lidar_max_range_m"])
    lidar_collision_threshold = lidar_minimum + LIDAR_TRIGGER_MARGIN_M
    by_step = {int(sample["step"]): sample for sample in samples}
    derived: list[dict[str, Any]] = []
    for sample in samples:
        step = int(sample["step"])
        previous = by_step.get(step - 1)
        velocity = _velocity_state(record=record, sample=sample, previous=previous)
        obstacle_metrics: list[dict[str, Any]] = []
        robot_position = _vector(sample["post_robot_root_xy"], "post_robot_root_xy")
        for obstacle_index, (name, obstacle_position_value) in enumerate(
            zip(obstacle_names, sample["post_obstacle_xy"], strict=True)
        ):
            obstacle_position = _vector(obstacle_position_value, "post_obstacle_xy")
            if velocity is None:
                centre_distance = float(np.linalg.norm(obstacle_position - robot_position))
                gap = centre_distance - ROBOT_RADIUS_M - OBSTACLE_RADIUS_M
                metrics: dict[str, Any] = {
                    "gap_m": gap,
                    "closing_rate_mps": None,
                    "ttc_s": None,
                    "ttc_defined": False,
                    "t_cpa_s": None,
                    "cpa_clearance_m": None,
                    "proxy_t_cpa_s": None,
                    "proxy_cpa_clearance_m": None,
                }
            else:
                relative_position = obstacle_position - robot_position
                relative_velocity = velocity.obstacles[obstacle_index] - velocity.robot
                metrics = _relative_metrics(
                    relative_position,
                    relative_velocity,
                    horizon_s=CPA_HORIZON_S,
                )
                proxy = _relative_metrics(
                    relative_position,
                    relative_velocity,
                    horizon_s=PROXY_HORIZON_S,
                )
                metrics["proxy_t_cpa_s"] = proxy["t_cpa_s"]
                metrics["proxy_cpa_clearance_m"] = proxy["cpa_clearance_m"]
            obstacle_metrics.append({"def_name": name, **metrics})
        nearest = min(obstacle_metrics, key=lambda value: float(value["gap_m"]))
        defined_ttc = [
            value for value in obstacle_metrics if bool(value["ttc_defined"])
        ]
        minimum_ttc = (
            min(defined_ttc, key=lambda value: float(value["ttc_s"]))
            if defined_ttc
            else None
        )
        risk_values = [
            value
            for value in obstacle_metrics
            if value["cpa_clearance_m"] is not None
            and float(value["closing_rate_mps"]) > 0.0
            and float(value["t_cpa_s"]) <= CPA_HORIZON_S
            and float(value["cpa_clearance_m"]) <= GEOMETRY_TOLERANCE_M
        ]
        lidar_source = _attribute_lidar_ray(sample, obstacle_names)
        derived.append(
            {
                "step": step,
                "post_elapsed_s": float(sample["post_elapsed_s"]),
                "window_membership": sorted(memberships[step]),
                "interval_action_decision_state_step": step - 1,
                "interval_action_post_state_step": step,
                "interval_commanded_linear_mps": float(
                    sample["interval_commanded_linear"]
                ),
                "interval_commanded_angular_rps": float(
                    sample["interval_commanded_angular"]
                ),
                "post_global_clearance_m": float(sample["post_global_clearance"]),
                "post_lidar_range_m": float(sample["post_lidar_argmin"]["range_m"]),
                "lidar_min_range_m": lidar_minimum,
                "lidar_max_range_m": lidar_maximum,
                "lidar_collision_threshold_m": lidar_collision_threshold,
                "lidar_threshold_margin_m": (
                    float(sample["post_lidar_argmin"]["range_m"])
                    - lidar_collision_threshold
                ),
                "post_lidar_bearing_rad": float(
                    sample["post_lidar_argmin"]["bearing_rad"]
                ),
                "lidar_dynamic_source": lidar_source,
                "velocity_defined": velocity is not None,
                "velocity_source": velocity.source if velocity is not None else None,
                "nearest_dynamic_def": nearest["def_name"],
                "dynamic_gap_m": nearest["gap_m"],
                "dynamic_closing_rate_mps": nearest["closing_rate_mps"],
                "dynamic_ttc_s": (
                    minimum_ttc["ttc_s"] if minimum_ttc is not None else None
                ),
                "dynamic_ttc_defined": minimum_ttc is not None,
                "dynamic_ttc_obstacle_def": (
                    minimum_ttc["def_name"] if minimum_ttc is not None else None
                ),
                "dynamic_t_cpa_s": nearest["t_cpa_s"],
                "dynamic_cpa_clearance_m": nearest["cpa_clearance_m"],
                "proxy_t_cpa_s": nearest["proxy_t_cpa_s"],
                "proxy_cpa_clearance_m": nearest["proxy_cpa_clearance_m"],
                "ground_truth_dynamic_risk": bool(risk_values),
                "post_contact_collision": bool(sample["post_contact_collision"]),
                "post_lidar_collision": bool(sample["post_lidar_collision"]),
                "obstacles": obstacle_metrics,
            }
        )
    return derived


def _risk_onset(derived: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for start in range(0, len(derived) - RISK_PERSISTENCE_TICKS + 1):
        window = derived[start : start + RISK_PERSISTENCE_TICKS]
        first_step = int(window[0]["step"])
        if all(
            int(value["step"]) == first_step + offset
            and value["ground_truth_dynamic_risk"]
            for offset, value in enumerate(window)
        ):
            return window[0]
    return None


def _reaction_metrics(
    derived: Sequence[Mapping[str, Any]],
    onset: Mapping[str, Any] | None,
) -> dict[str, Any]:
    def indeterminate(reason: str) -> dict[str, Any]:
        return {
            "deceleration_reaction_evaluable": False,
            "deceleration_reaction_indeterminate_reason": reason,
            "deceleration_reaction_detected": False,
            "deceleration_reaction_delay_s": None,
            "timely_deceleration_response": False,
        }

    if onset is None:
        return indeterminate("risk_onset_not_detected")
    onset_step = int(onset["step"])
    before = [
        value
        for value in derived
        if onset_step - RESPONSE_BASELINE_TICKS + 1 <= int(value["step"]) <= onset_step
    ]
    expected_baseline_steps = list(
        range(onset_step - RESPONSE_BASELINE_TICKS + 1, onset_step + 1)
    )
    if [int(value["step"]) for value in before] != expected_baseline_steps:
        return indeterminate("missing_contiguous_pre_reaction_baseline")
    baseline = statistics.median(
        float(value["interval_commanded_linear_mps"]) for value in before
    )
    if baseline <= _EPSILON:
        return {
            "deceleration_reaction_evaluable": True,
            "deceleration_reaction_indeterminate_reason": None,
            "deceleration_reaction_detected": True,
            "deceleration_reaction_delay_s": 0.0,
            "timely_deceleration_response": True,
        }
    threshold = baseline * (1.0 - DECELERATION_FRACTION)
    onset_time = float(onset["post_elapsed_s"])
    previous_step = onset_step
    for value in derived:
        step = int(value["step"])
        if step <= onset_step:
            continue
        if step != previous_step + 1:
            return indeterminate("noncontiguous_post_onset_response_window")
        previous_step = step
        delay = float(value["post_elapsed_s"]) - onset_time
        if float(value["interval_commanded_linear_mps"]) <= threshold:
            return {
                "deceleration_reaction_evaluable": True,
                "deceleration_reaction_indeterminate_reason": None,
                "deceleration_reaction_detected": True,
                "deceleration_reaction_delay_s": delay,
                "timely_deceleration_response": delay <= RESPONSE_WINDOW_S,
            }
        if delay >= RESPONSE_WINDOW_S - _TIME_TOLERANCE:
            return {
                "deceleration_reaction_evaluable": True,
                "deceleration_reaction_indeterminate_reason": None,
                "deceleration_reaction_detected": False,
                "deceleration_reaction_delay_s": None,
                "timely_deceleration_response": False,
            }
    return indeterminate("insufficient_post_onset_response_window")


def _terminal_classification(
    record: Mapping[str, Any],
    derived: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None]:
    termination = record["termination"]
    if termination["contact_collision"]:
        return "contact_confirmed", None
    if not termination["lidar_collision"]:
        return "not_collision_terminated", None
    terminal_step = int(termination["terminal_step"])
    terminal = next((value for value in derived if int(value["step"]) == terminal_step), None)
    previous = next(
        (value for value in derived if int(value["step"]) == terminal_step - 1),
        None,
    )
    if terminal is None or previous is None or not terminal["velocity_defined"]:
        return "ambiguous", "insufficient_contiguous_terminal_window"
    source = terminal["lidar_dynamic_source"]
    if source is None:
        return "ambiguous", "lidar_ray_not_uniquely_attributed_to_dynamic_obstacle"
    obstacle = next(
        (value for value in terminal["obstacles"] if value["def_name"] == source),
        None,
    )
    if obstacle is None or obstacle["proxy_cpa_clearance_m"] is None:
        return "ambiguous", "attributed_obstacle_has_no_relative_motion_estimate"
    gap = float(obstacle["gap_m"])
    proxy_cpa = float(obstacle["proxy_cpa_clearance_m"])
    if gap <= GEOMETRY_TOLERANCE_M or proxy_cpa <= 0.0:
        return "physical_imminent_dynamic", None
    if gap > GEOMETRY_TOLERANCE_M and proxy_cpa > GEOMETRY_TOLERANCE_M:
        return "lidar_proxy_compatible", None
    return "ambiguous", "geometry_falls_inside_proxy_tolerance_band"


def analyze_episode(
    record: Mapping[str, Any],
    episode: EpisodeRow,
    samples: Sequence[Mapping[str, Any]],
    memberships: Mapping[int, set[str]],
) -> dict[str, Any]:
    derived = _derive_samples(record, samples, memberships)
    onset = _risk_onset(derived)
    reaction = _reaction_metrics(derived, onset)
    classification, ambiguity_reason = _terminal_classification(record, derived)
    defined_ttc = [
        float(value["dynamic_ttc_s"])
        for value in derived
        if value["dynamic_ttc_defined"]
    ]
    closing_rates = [
        float(value["dynamic_closing_rate_mps"])
        for value in derived
        if value["dynamic_closing_rate_mps"] is not None
    ]
    cpa_clearances = [
        float(value["dynamic_cpa_clearance_m"])
        for value in derived
        if value["dynamic_cpa_clearance_m"] is not None
    ]
    terminal_time = float(derived[-1]["post_elapsed_s"])
    onset_time = float(onset["post_elapsed_s"]) if onset is not None else None
    return {
        "episode_id": episode.episode_id,
        "scenario_seed": episode.scenario_seed,
        "condition_id": episode.condition_id,
        "planner": episode.planner,
        "route_axis": record["route_axis"],
        "outcome": episode.outcome,
        "terminal_collision_source": (
            "both"
            if record["termination"]["contact_collision"]
            and record["termination"]["lidar_collision"]
            else "contact_only"
            if record["termination"]["contact_collision"]
            else "lidar_only"
            if record["termination"]["lidar_collision"]
            else "none"
        ),
        "mechanism_classification": classification,
        "ambiguity_reason": ambiguity_reason,
        "trace_unique_steps": len(derived),
        "trace_first_step": int(derived[0]["step"]),
        "trace_last_step": int(derived[-1]["step"]),
        "minimum_dynamic_gap_m": min(float(value["dynamic_gap_m"]) for value in derived),
        "minimum_dynamic_ttc_s": min(defined_ttc) if defined_ttc else None,
        "minimum_dynamic_ttc_defined": bool(defined_ttc),
        "minimum_dynamic_cpa_clearance_m": (
            min(cpa_clearances) if cpa_clearances else None
        ),
        "minimum_lidar_threshold_margin_m": min(
            float(value["lidar_threshold_margin_m"]) for value in derived
        ),
        "maximum_dynamic_closing_rate_mps": (
            max(closing_rates) if closing_rates else None
        ),
        "risk_onset_detected": onset is not None,
        "risk_onset_step": int(onset["step"]) if onset is not None else None,
        "risk_onset_time_s": onset_time,
        "risk_onset_to_terminal_s": (
            terminal_time - onset_time if onset_time is not None else None
        ),
        **reaction,
        "dynamic_short_horizon_response_insufficient_signature": bool(
            classification in {"contact_confirmed", "physical_imminent_dynamic"}
            and onset is not None
            and reaction["deceleration_reaction_evaluable"]
            and terminal_time - float(onset["post_elapsed_s"]) >= RESPONSE_WINDOW_S
            and not reaction["timely_deceleration_response"]
        ),
        "risk_curve": derived,
    }


def wilson_interval(successes: int, total: int) -> dict[str, float | int | None]:
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total.")
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "proportion": None,
            "low": None,
            "high": None,
        }
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "proportion": proportion,
        "low": max(0.0, centre - radius),
        "high": min(1.0, centre + radius),
    }


def _dominant(interval: Mapping[str, Any]) -> bool:
    proportion = interval["proportion"]
    low = interval["low"]
    return bool(
        proportion is not None
        and low is not None
        and proportion >= DOMINANCE_FRACTION
        and low > 0.5
    )


def geometry_decision(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare physical and proxy-compatible geometry on one shared denominator."""

    lidar_only = [
        value
        for value in episodes
        if value["terminal_collision_source"] == "lidar_only"
    ]
    ambiguous = [
        value
        for value in lidar_only
        if value["mechanism_classification"] == "ambiguous"
    ]
    classifiable = [
        value
        for value in lidar_only
        if value["mechanism_classification"]
        in {"physical_imminent_dynamic", "lidar_proxy_compatible"}
    ]
    physical_count = sum(
        value["mechanism_classification"] == "physical_imminent_dynamic"
        for value in classifiable
    )
    proxy_count = sum(
        value["mechanism_classification"] == "lidar_proxy_compatible"
        for value in classifiable
    )
    physical_interval = wilson_interval(physical_count, len(classifiable))
    proxy_interval = wilson_interval(proxy_count, len(classifiable))
    ambiguous_fraction = len(ambiguous) / len(lidar_only) if lidar_only else 1.0
    insufficient = (
        len(classifiable) < MINIMUM_CLASSIFIABLE_FAILURES
        or ambiguous_fraction > MAXIMUM_AMBIGUOUS_FRACTION
    )
    if insufficient:
        decision = "indeterminate"
    elif _dominant(physical_interval):
        decision = "physical_dynamic_geometry_dominant"
    elif _dominant(proxy_interval):
        decision = "lidar_proxy_dominant"
    else:
        decision = "mixed"
    return {
        "decision": decision,
        "denominator_definition": (
            "classifiable_lidar_only_physical_or_proxy_compatible"
        ),
        "lidar_only_episodes": len(lidar_only),
        "classifiable_lidar_only_episodes": len(classifiable),
        "ambiguous_lidar_only_episodes": len(ambiguous),
        "ambiguous_fraction": ambiguous_fraction,
        "physical_imminent_dynamic": physical_interval,
        "lidar_proxy_compatible": proxy_interval,
        "thresholds": {
            "minimum_classifiable_failures": MINIMUM_CLASSIFIABLE_FAILURES,
            "maximum_ambiguous_fraction": MAXIMUM_AMBIGUOUS_FRACTION,
            "dominance_fraction": DOMINANCE_FRACTION,
            "wilson_confidence": 0.95,
        },
    }


def response_decision(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assess response only where physical geometry and reaction timing are evaluable."""

    physical = [
        value
        for value in episodes
        if value["mechanism_classification"]
        in {"contact_confirmed", "physical_imminent_dynamic"}
    ]
    evaluable = [
        value for value in physical if value["deceleration_reaction_evaluable"]
    ]
    indeterminate = [
        value for value in physical if not value["deceleration_reaction_evaluable"]
    ]
    insufficient_count = sum(
        not bool(value["timely_deceleration_response"]) for value in evaluable
    )
    timely_count = len(evaluable) - insufficient_count
    insufficient_interval = wilson_interval(insufficient_count, len(evaluable))
    timely_interval = wilson_interval(timely_count, len(evaluable))
    indeterminate_fraction = len(indeterminate) / len(physical) if physical else 1.0
    insufficient_power = (
        len(evaluable) < MINIMUM_CLASSIFIABLE_FAILURES
        or indeterminate_fraction > MAXIMUM_AMBIGUOUS_FRACTION
    )
    if insufficient_power:
        decision = "indeterminate"
    elif _dominant(insufficient_interval):
        decision = "short_horizon_response_insufficient"
    elif _dominant(timely_interval):
        decision = "timely_response_dominant"
    else:
        decision = "mixed"
    return {
        "decision": decision,
        "denominator_definition": "reaction_evaluable_physical_episodes_only",
        "physical_episodes": len(physical),
        "reaction_evaluable_physical_episodes": len(evaluable),
        "reaction_indeterminate_physical_episodes": len(indeterminate),
        "reaction_indeterminate_fraction": indeterminate_fraction,
        "short_horizon_response_insufficient": insufficient_interval,
        "timely_deceleration_response": timely_interval,
        "thresholds": {
            "minimum_reaction_evaluable_physical_failures": (
                MINIMUM_CLASSIFIABLE_FAILURES
            ),
            "maximum_reaction_indeterminate_fraction": MAXIMUM_AMBIGUOUS_FRACTION,
            "dominance_fraction": DOMINANCE_FRACTION,
            "wilson_confidence": 0.95,
        },
    }


def mechanism_decision(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    geometry = geometry_decision(episodes)
    response = response_decision(episodes)
    geometry_result = geometry["decision"]
    response_result = response["decision"]
    if geometry_result == "lidar_proxy_dominant":
        decision = "lidar_proxy_dominant"
    elif geometry_result == "indeterminate" or response_result == "indeterminate":
        decision = "indeterminate"
    elif (
        geometry_result == "physical_dynamic_geometry_dominant"
        and response_result == "short_horizon_response_insufficient"
    ):
        decision = "dynamic_short_horizon_response_insufficient"
    else:
        decision = "mixed"
    return {
        "decision": decision,
        "combination_rule": (
            "proxy_geometry_dominant=>lidar_proxy_dominant; "
            "physical_geometry_dominant+insufficient_response=>dynamic_short_horizon_"
            "response_insufficient; unresolved_layer=>indeterminate; otherwise=>mixed"
        ),
        "geometry": geometry,
        "response": response,
    }


def _optional_mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _aggregate_group(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(values),
        "mechanism_classification_counts": dict(
            sorted(Counter(str(value["mechanism_classification"]) for value in values).items())
        ),
        "risk_onset_count": sum(bool(value["risk_onset_detected"]) for value in values),
        "minimum_dynamic_gap_m_mean": _optional_mean(
            [value["minimum_dynamic_gap_m"] for value in values]
        ),
        "minimum_dynamic_ttc_s_mean": _optional_mean(
            [value["minimum_dynamic_ttc_s"] for value in values]
        ),
        "minimum_dynamic_cpa_clearance_m_mean": _optional_mean(
            [value["minimum_dynamic_cpa_clearance_m"] for value in values]
        ),
        "maximum_dynamic_closing_rate_mps_mean": _optional_mean(
            [value["maximum_dynamic_closing_rate_mps"] for value in values]
        ),
        "deceleration_reaction_delay_s_mean": _optional_mean(
            [value["deceleration_reaction_delay_s"] for value in values]
        ),
    }


def aggregate_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    axes = sorted({str(value["route_axis"]) for value in episodes})
    outcomes = sorted({str(value["outcome"]) for value in episodes})
    return {
        "overall": _aggregate_group(episodes),
        "by_route_axis": [
            {
                "route_axis": axis,
                **_aggregate_group([value for value in episodes if value["route_axis"] == axis]),
            }
            for axis in axes
        ],
        "by_outcome": [
            {
                "outcome": outcome,
                **_aggregate_group([value for value in episodes if value["outcome"] == outcome]),
            }
            for outcome in outcomes
        ],
        "by_route_axis_outcome": [
            {
                "route_axis": axis,
                "outcome": outcome,
                **_aggregate_group(
                    [
                        value
                        for value in episodes
                        if value["route_axis"] == axis and value["outcome"] == outcome
                    ]
                ),
            }
            for axis in axes
            for outcome in outcomes
            if any(
                value["route_axis"] == axis and value["outcome"] == outcome
                for value in episodes
            )
        ],
    }


def _csv_row(value: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"risk_curve"}
    return {
        name: item
        for name, item in value.items()
        if name not in excluded and not isinstance(item, (dict, list))
    }


def analyze_risk_trajectory(
    *,
    collection_report_path: Path,
    output_directory: Path,
    repository_root: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite risk analysis: {output_directory}")
    locked = load_locked_collection(
        collection_report_path,
        repository_root=repository_root,
    )
    episode_rows = read_episode_rows(locked.episodes_path)
    contexts = read_context_records(locked.context_path, episodes=episode_rows)
    _validate_locked_episode_scope(episode_rows, contexts)
    episode_metrics = [
        analyze_episode(record, episode, samples, memberships)
        for record, episode, samples, memberships in contexts
    ]
    episode_metrics.sort(key=lambda value: int(value["scenario_seed"]))
    sidecar_path, sidecar_sha256, report_sha256 = _verify_collection_sidecar(
        locked.report_path
    )
    if (
        report_sha256 != locked.report_sha256
        or sidecar_path != locked.sidecar_path
        or sidecar_sha256 != locked.sidecar_sha256
        or _sha256_file(locked.context_path) != locked.context_sha256
        or _sha256_file(locked.episodes_path) != locked.episodes_sha256
    ):
        raise RuntimeError("Locked collection inputs changed while they were being parsed.")
    output_directory.mkdir(parents=True, exist_ok=False)
    json_path = output_directory / "risk_episode_metrics.json"
    csv_path = output_directory / "risk_episode_metrics.csv"
    json_path.write_text(
        json.dumps(episode_metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_rows = [_csv_row(value) for value in episode_metrics]
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {
        "schema_version": 1,
        "analysis_protocol": {
            "name": PROTOCOL_NAME,
            "robot_radius_m": ROBOT_RADIUS_M,
            "obstacle_radius_m": OBSTACLE_RADIUS_M,
            "cpa_horizon_s": CPA_HORIZON_S,
            "proxy_horizon_s": PROXY_HORIZON_S,
            "geometry_tolerance_m": GEOMETRY_TOLERANCE_M,
            "lidar_trigger_margin_m": LIDAR_TRIGGER_MARGIN_M,
            "risk_persistence_ticks": RISK_PERSISTENCE_TICKS,
            "response_window_s": RESPONSE_WINDOW_S,
            "response_baseline_ticks": RESPONSE_BASELINE_TICKS,
            "deceleration_fraction": DECELERATION_FRACTION,
            "undefined_ttc_encoding": {"ttc_s": None, "ttc_defined": False},
            "ttc_definition": "exact_2d_circular_footprint_quadratic_first_contact",
            "velocity_preference": "adjacent_post_truth_then_reset_to_post_truth",
            "sample_semantics": RISK_TRACE_SAMPLE_SEMANTICS,
            "coordinate_semantics": RISK_TRACE_COORDINATE_SEMANTICS,
            "geometry_robot_position_field": "post_robot_root_xy",
            "velocity_robot_position_field": "post_robot_root_xy",
            "terminal_position_validation_field": "post_robot_gps_xy",
        },
        "inputs": {
            "collection_report": {
                "path": str(locked.report_path),
                "sha256": locked.report_sha256,
                "sidecar_path": str(locked.sidecar_path),
                "sidecar_sha256": locked.sidecar_sha256,
                "experiment_id": COLLECTION_EXPERIMENT_ID,
                "status": COLLECTION_STATUS,
                "model_sha256": RECOVERY_MODEL_SHA256,
            },
            "episode_context": {
                "path": str(locked.context_path),
                "schema_version": EPISODE_CONTEXT_SCHEMA_VERSION,
                "sha256": locked.context_sha256,
            },
            "episodes_csv": {
                "path": str(locked.episodes_path),
                "schema_version": "1.0",
                "sha256": locked.episodes_sha256,
            },
        },
        "episode_count": len(episode_metrics),
        "scenario_seeds": [value["scenario_seed"] for value in episode_metrics],
        "aggregates": aggregate_metrics(episode_metrics),
        "mechanism_decision": mechanism_decision(episode_metrics),
        "artifacts": {
            "risk_episode_metrics_json": {
                "path": str(json_path.resolve()),
                "sha256": _sha256_file(json_path),
            },
            "risk_episode_metrics_csv": {
                "path": str(csv_path.resolve()),
                "sha256": _sha256_file(csv_path),
            },
        },
        "scope": {
            "split": "diagnostic",
            "formal_mechanism_inference": True,
            "collection_experiment_id": COLLECTION_EXPERIMENT_ID,
            "model_sha256": RECOVERY_MODEL_SHA256,
            "exact_scenario_seed_start": DIAGNOSTIC_SEED_START,
            "exact_scenario_seed_count": DIAGNOSTIC_EPISODES,
            "exact_route_axis_counts": EXPECTED_AXIS_COUNTS,
            "checkpoint_selection": "forbidden",
            "validation_or_test_access": "forbidden",
        },
    }
    report_path = output_directory / "risk_analysis_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    repository_root = Path(__file__).absolute().parents[1]
    try:
        report = analyze_risk_trajectory(
            collection_report_path=args.collection_report,
            output_directory=args.output_dir,
            repository_root=repository_root,
        )
    except Exception as error:
        print(
            f"WEBOTS_RISK_ANALYSIS_FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    print(
        "WEBOTS_RISK_ANALYSIS_OK "
        f"episodes={report['episode_count']} "
        f"decision={report['mechanism_decision']['decision']} "
        f"report={args.output_dir / 'risk_analysis_report.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
