#!/usr/bin/env python3
"""Run the pre-registered paired front-brake diagnostic in Webots.

The protocol has exactly one reference and one candidate.  It binds source,
model, simulator, scenario seeds, and controller parameters by SHA-256; records
schema-validated episode contexts and risk traces; and never opens validation or
held-out-test splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from indoor_patrol_rl.core.safety_shield import safety_shield_config_from_env
from indoor_patrol_rl.evaluation.episode_context import (
    EPISODE_CONTEXT_SCHEMA_VERSION,
    validate_risk_trace,
)
from indoor_patrol_rl.evaluation.metrics import EpisodeRecord
from indoor_patrol_rl.evaluation.shield_evidence import (
    SHIELD_EVIDENCE_SCHEMA_VERSION,
    validate_shield_evidence_record,
)
from indoor_patrol_rl.training.webots_curriculum import (
    choose_available_port,
    resolve_webots_executable,
)

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
SourceVerifier = Callable[[Path], str]

EXIT_PREFLIGHT_OR_RUNTIME_FAILED = 2
EXIT_CANDIDATE_GATE_FAILED = 3
EXIT_INVALID_MANIPULATION = 4
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_VALIDATION_SEEDS = frozenset(range(5_000, 5_020))
_TEST_SEEDS = frozenset(range(10_000, 10_020))
_FIXED_EXPERIMENT_ID = "webots_front_brake_paired_diagnostic_v1"
_FIXED_CONDITION_ID = "DB1"
_FIXED_TRAINING_SEED = 11
_FIXED_SEED_START = 25_728
_FIXED_EPISODES = 120
_FIXED_MODEL = "artifacts/models/webots/seed_11/w1_recovery_v1/ppo_webots_s11_step000040960.zip"
_FIXED_MODEL_SHA256 = "c5e7c3de42c7664477097b2734975660fec840ad5b7f45425098158c27cb7275"
_FIXED_MODEL_VARIANT = "w1_recovery_v1"
_WORLD = "worlds/webots_ppo_evaluate.wbt"
_CONTROLLER = "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
_SHIELD_IMPLEMENTATION = "src/indoor_patrol_rl/core/safety_shield.py"
_CONTROL_IMPLEMENTATION = "src/indoor_patrol_rl/core/control.py"
_ENVIRONMENT_IMPLEMENTATION = "src/indoor_patrol_rl/envs/webots_patrol.py"
_EVALUATION_EXPORTS = "src/indoor_patrol_rl/evaluation/__init__.py"
_METRICS_IMPLEMENTATION = "src/indoor_patrol_rl/evaluation/metrics.py"
_FORMAL_CONFIG_IMPLEMENTATION = (
    "src/indoor_patrol_rl/evaluation/webots_formal_config.py"
)
_CONTEXT_IMPLEMENTATION = "src/indoor_patrol_rl/evaluation/episode_context.py"
_SHIELD_EVIDENCE_IMPLEMENTATION = "src/indoor_patrol_rl/evaluation/shield_evidence.py"
_CURRICULUM_IMPLEMENTATION = "src/indoor_patrol_rl/training/webots_curriculum.py"
_RUNNER = "scripts/run_webots_front_brake_diagnostic.py"
_REPORT_DIRECTORY = "artifacts/results/{experiment_id}"
_EPISODE_FIELDS = tuple(EpisodeRecord.__dataclass_fields__)
_LOCKED_SHIELD_PARAMETERS = {
    "WEBOTS_SAFETY_SHIELD_CLEARANCE_QUANTILE": "0.0",
    "WEBOTS_SAFETY_SHIELD_FRONT_STOP_DISTANCE": "0.24",
    "WEBOTS_SAFETY_SHIELD_FRONT_SLOWDOWN_DISTANCE": "0.75",
    "WEBOTS_SAFETY_SHIELD_FRONT_HALF_ANGLE": "1.0471975511965976",
    "WEBOTS_SAFETY_SHIELD_SIDE_INNER_ANGLE": "1.0471975511965976",
    "WEBOTS_SAFETY_SHIELD_SIDE_SPEED_FLOOR": "1.0",
    "WEBOTS_SAFETY_SHIELD_TURN_BLEND": "0.0",
}
_REFERENCE_ENV = {"WEBOTS_SAFETY_SHIELD": "0", **_LOCKED_SHIELD_PARAMETERS}
_CANDIDATE_ENV = {"WEBOTS_SAFETY_SHIELD": "1", **_LOCKED_SHIELD_PARAMETERS}
_FIXED_STRATA = {
    "horizontal": {"non_reverse": 30, "reverse": 30},
    "vertical": {"non_reverse": 30, "reverse": 30},
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key is forbidden: {key!r}.")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ModelBinding:
    portable_path: str
    path: Path
    sha256: str
    variant: str


@dataclass(frozen=True)
class Scenario:
    obstacle_count: int
    obstacle_speed_range: tuple[float, float]
    horizontal_route_half_length: float
    vertical_route_half_length: float
    deterministic_policy: bool


@dataclass(frozen=True)
class Arm:
    arm_id: str
    role: str
    run_name: str
    planner_label: str
    controller_env: Mapping[str, str]


@dataclass(frozen=True)
class CandidateGate:
    require_all: bool
    minimum_relative_collision_reduction: float
    exact_one_sided_mcnemar_significance_level: float
    maximum_new_timeout_count: int
    maximum_median_completion_time_ratio: float
    minimum_common_success_count: int
    maximum_error_count_per_arm: int


@dataclass(frozen=True)
class DiagnosticConfig:
    source_path: Path
    source_portable_path: str
    source_sha256: str
    experiment_id: str
    condition_id: str
    training_seed: int
    scenario_seed_start: int
    episodes: int
    model: ModelBinding
    scenario: Scenario
    arms: tuple[Arm, Arm]
    gate: CandidateGate
    decision: Mapping[str, str]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.scenario_seed_start, self.scenario_seed_start + self.episodes))

    def arm_for_role(self, role: str) -> Arm:
        return next(arm for arm in self.arms if arm.role == role)


@dataclass(frozen=True)
class DiagnosticRun:
    arm: Arm
    port: int
    command: tuple[str, ...]
    environment: Mapping[str, str]
    result_directory: Path


@dataclass(frozen=True)
class ParsedArmResult:
    arm: Arm
    rows_by_seed: Mapping[int, Mapping[str, Any]]
    contexts_by_seed: Mapping[int, Mapping[str, Any]]
    shield_evidence_by_seed: Mapping[int, Mapping[str, Any]]
    shield_evidence_metrics: Mapping[str, Any]
    metrics: Mapping[str, Any]
    artifacts: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_clean_source(repository_root: Path) -> str:
    """Return HEAD only for a clean tracked and non-ignored-untracked source tree."""

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"git {' '.join(arguments)} failed: {detail or completed.returncode}"
            )
        return completed.stdout.strip()

    revision = git("rev-parse", "--verify", "HEAD").casefold()
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("Git HEAD is not a full lowercase revision.")
    tracked = git("status", "--porcelain", "--untracked-files=no")
    untracked = git("ls-files", "--others", "--exclude-standard")
    if tracked or untracked:
        preview = " | ".join((tracked + "\n" + untracked).splitlines()[:8])
        raise RuntimeError("Front-brake audit requires a clean source tree: " + preview)
    return revision


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"{label} schema mismatch; missing={missing}, unknown={unknown}.")


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n"):
        raise ValueError(f"{label} must be a non-empty single-line string.")
    return value


def _safe_name(value: Any, label: str) -> str:
    result = _text(value, label)
    if _SAFE_NAME.fullmatch(result) is None:
        raise ValueError(f"{label} must be a path-safe name.")
    return result


def _inside_repository(repository_root: Path, value: Any, label: str) -> tuple[str, Path]:
    portable = _text(value, label)
    configured = Path(portable)
    if configured.is_absolute() or "\\" in portable or portable in {".", ".."}:
        raise ValueError(f"{label} must be a repository-relative POSIX path.")
    if ".." in configured.parts:
        raise ValueError(f"{label} escapes the repository.")
    resolved = (repository_root / configured).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository.") from error
    return configured.as_posix(), resolved


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, actual {actual}.")


def _scenario_covariates_for_seed(seed: int) -> tuple[str, bool, float]:
    """Reproduce pre-treatment route and obstacle-direction RNG draws."""

    random_generator = np.random.default_rng(seed)
    route_axis = str(random_generator.choice(["horizontal", "vertical"]))
    offset_limit = 0.55 if route_axis == "horizontal" else 0.70
    random_generator.uniform(-offset_limit, offset_limit)
    random_generator.uniform(-offset_limit, offset_limit)
    reverse_route = bool(random_generator.integers(0, 2))
    random_generator.uniform(-math.pi, math.pi)  # start yaw
    if route_axis == "horizontal":
        random_generator.uniform(-0.55, 0.55)
        random_generator.uniform(-1.05, 1.05)
    else:
        random_generator.uniform(-1.55, 1.55)
        random_generator.uniform(-0.45, 0.45)
    random_generator.uniform(0.15, 0.30)  # obstacle speed
    obstacle_direction = float(random_generator.choice([-1.0, 1.0]))
    return route_axis, reverse_route, obstacle_direction


def _route_cell_for_seed(seed: int) -> tuple[str, bool]:
    axis, reverse, _direction = _scenario_covariates_for_seed(seed)
    return axis, reverse


def _strata_counts(seeds: Sequence[int]) -> dict[str, dict[str, int]]:
    counts: Counter[tuple[str, bool]] = Counter(_route_cell_for_seed(seed) for seed in seeds)
    return {
        axis: {
            "non_reverse": counts[(axis, False)],
            "reverse": counts[(axis, True)],
        }
        for axis in ("horizontal", "vertical")
    }


def _obstacle_direction_counts(seeds: Sequence[int]) -> dict[str, int]:
    counts = Counter(_scenario_covariates_for_seed(seed)[2] for seed in seeds)
    return {"negative": counts[-1.0], "positive": counts[1.0]}


def _first_balanced_seed_window(search_start: int, episodes: int) -> int:
    """Find the first window using only deterministic pre-treatment covariates."""

    if search_start < 0 or episodes < 1:
        raise ValueError("Seed-window search bounds must be positive.")
    window = tuple(range(search_start, search_start + episodes))
    route_counts = Counter(_route_cell_for_seed(seed) for seed in window)
    direction_counts = Counter(_scenario_covariates_for_seed(seed)[2] for seed in window)

    def balanced() -> bool:
        return all(
            route_counts[(axis, reverse)] == 30
            for axis in ("horizontal", "vertical")
            for reverse in (False, True)
        ) and direction_counts == {-1.0: 60, 1.0: 60}

    start = search_start
    while start <= _FIXED_SEED_START:
        if balanced():
            return start
        leaving = _scenario_covariates_for_seed(start)
        entering = _scenario_covariates_for_seed(start + episodes)
        route_counts[(leaving[0], leaving[1])] -= 1
        direction_counts[leaving[2]] -= 1
        route_counts[(entering[0], entering[1])] += 1
        direction_counts[entering[2]] += 1
        start += 1
    raise ValueError("No balanced pre-treatment seed window was found by the locked bound.")


def _load_locked_strata(value: Any) -> None:
    raw = _mapping(value, "preregistration.expected_route_strata_counts")
    _exact_keys(raw, {"horizontal", "vertical"}, "expected_route_strata_counts")
    parsed: dict[str, dict[str, int]] = {}
    for axis in ("horizontal", "vertical"):
        axis_raw = _mapping(raw[axis], f"expected_route_strata_counts.{axis}")
        _exact_keys(axis_raw, {"non_reverse", "reverse"}, f"strata.{axis}")
        parsed[axis] = {
            direction: _integer(axis_raw[direction], f"strata.{axis}.{direction}")
            for direction in ("non_reverse", "reverse")
        }
    if parsed != _FIXED_STRATA:
        raise ValueError("Expected route strata must be exactly 30 in each axis/direction cell.")


def load_diagnostic_config(path: Path, *, repository_root: Path) -> DiagnosticConfig:
    """Load and verify the single supported front-brake protocol."""

    root = repository_root.resolve()
    source = (path if path.is_absolute() else root / path).resolve()
    try:
        source_portable = source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Diagnostic config must remain inside the repository.") from error
    if not source.is_file():
        raise FileNotFoundError(f"Diagnostic config does not exist: {source}")
    raw = _mapping(
        yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader),
        "config",
    )
    _exact_keys(
        raw,
        {
            "schema_version",
            "experiment_id",
            "split",
            "condition_id",
            "training_seed",
            "scenario_seed_start",
            "episodes",
            "model",
            "scenario",
            "arms",
            "preregistration",
            "candidate_gate",
            "decision",
        },
        "config",
    )
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must be exactly 1.")
    if raw["split"] != "diagnostic":
        raise PermissionError("Front-brake runner only permits split=diagnostic.")
    experiment_id = _safe_name(raw["experiment_id"], "experiment_id")
    condition_id = _safe_name(raw["condition_id"], "condition_id")
    training_seed = _integer(raw["training_seed"], "training_seed")
    seed_start = _integer(raw["scenario_seed_start"], "scenario_seed_start")
    episodes = _integer(raw["episodes"], "episodes", minimum=1)
    if (
        experiment_id,
        condition_id,
        training_seed,
        seed_start,
        episodes,
    ) != (
        _FIXED_EXPERIMENT_ID,
        _FIXED_CONDITION_ID,
        _FIXED_TRAINING_SEED,
        _FIXED_SEED_START,
        _FIXED_EPISODES,
    ):
        raise ValueError(
            "Protocol identity is fixed to DB1, training seed 11, and diagnostic seeds "
            "25728..25847 (120 episodes)."
        )
    seeds = tuple(range(seed_start, seed_start + episodes))
    validation_overlap = sorted(set(seeds) & _VALIDATION_SEEDS)
    test_overlap = sorted(set(seeds) & _TEST_SEEDS)
    if validation_overlap or test_overlap:
        raise PermissionError(
            "Diagnostic seeds overlap reserved splits: "
            f"validation={validation_overlap}, held_out_test={test_overlap}."
        )
    observed_strata = _strata_counts(seeds)
    if observed_strata != _FIXED_STRATA:
        raise ValueError(
            "Fixed seed window no longer reproduces four 30-episode route strata: "
            f"{observed_strata}."
        )
    if _obstacle_direction_counts(seeds) != {"negative": 60, "positive": 60}:
        raise ValueError("Fixed seed window no longer balances initial obstacle direction 60/60.")

    model_raw = _mapping(raw["model"], "model")
    _exact_keys(model_raw, {"path", "sha256", "variant"}, "model")
    model_portable, model_path = _inside_repository(root, model_raw["path"], "model.path")
    model_sha256 = _text(model_raw["sha256"], "model.sha256")
    if _SHA256.fullmatch(model_sha256) is None:
        raise ValueError("model.sha256 must be a lowercase SHA-256 digest.")
    if (
        model_portable,
        model_sha256,
        model_raw["variant"],
    ) != (_FIXED_MODEL, _FIXED_MODEL_SHA256, _FIXED_MODEL_VARIANT):
        raise ValueError("Both arms are locked to the recovery-v1 model and fixed SHA-256.")
    if "latest" in model_path.stem.casefold():
        raise ValueError("The diagnostic refuses mutable 'latest' model paths.")
    _verify_file(model_path, model_sha256, "Recovery-v1 model")
    model = ModelBinding(
        portable_path=model_portable,
        path=model_path,
        sha256=model_sha256,
        variant=_safe_name(model_raw["variant"], "model.variant"),
    )

    scenario_raw = _mapping(raw["scenario"], "scenario")
    _exact_keys(
        scenario_raw,
        {
            "obstacle_count",
            "obstacle_speed_range",
            "horizontal_route_half_length",
            "vertical_route_half_length",
            "deterministic_policy",
        },
        "scenario",
    )
    speeds = scenario_raw["obstacle_speed_range"]
    if not isinstance(speeds, list) or len(speeds) != 2:
        raise ValueError("scenario.obstacle_speed_range must be a two-value list.")
    speed_range = (
        _finite_number(speeds[0], "scenario.obstacle_speed_range[0]"),
        _finite_number(speeds[1], "scenario.obstacle_speed_range[1]"),
    )
    horizontal = _finite_number(
        scenario_raw["horizontal_route_half_length"],
        "scenario.horizontal_route_half_length",
    )
    vertical = _finite_number(
        scenario_raw["vertical_route_half_length"],
        "scenario.vertical_route_half_length",
    )
    obstacle_count = _integer(scenario_raw["obstacle_count"], "scenario.obstacle_count")
    if not (
        obstacle_count == 1
        and speed_range == (0.15, 0.30)
        and horizontal == 1.10
        and vertical == 1.10
        and scenario_raw["deterministic_policy"] is True
    ):
        raise ValueError(
            "Scenario is locked to DB1: one obstacle, speed [0.15,0.30], route halves "
            "1.10, and a deterministic policy."
        )
    scenario = Scenario(obstacle_count, speed_range, horizontal, vertical, True)

    arms_raw = raw["arms"]
    if not isinstance(arms_raw, list) or len(arms_raw) != 2:
        raise ValueError("arms must contain exactly one reference and one candidate.")
    expected_arms = {
        "reference": {
            "id": "recovery_v1_reference",
            "run_name": "webots_diagnostic_front_brake_v1_reference_seed11",
            "planner_label": "ppo_recovery_v1_front_brake_reference",
            "controller_env": _REFERENCE_ENV,
        },
        "candidate": {
            "id": "front_brake_candidate",
            "run_name": "webots_diagnostic_front_brake_v1_candidate_seed11",
            "planner_label": "ppo_recovery_v1_front_brake_candidate",
            "controller_env": _CANDIDATE_ENV,
        },
    }
    arms: list[Arm] = []
    for index, item in enumerate(arms_raw):
        arm_raw = _mapping(item, f"arms[{index}]")
        _exact_keys(
            arm_raw,
            {"id", "role", "run_name", "planner_label", "controller_env"},
            f"arms[{index}]",
        )
        role = _text(arm_raw["role"], f"arms[{index}].role")
        if role not in expected_arms:
            raise ValueError(f"arms[{index}].role must be reference or candidate.")
        controller_env = _mapping(arm_raw["controller_env"], f"arms[{index}].controller_env")
        if any(not isinstance(value, str) for value in controller_env.values()):
            raise ValueError(f"arms[{index}].controller_env values must be strings.")
        observed = {
            "id": arm_raw["id"],
            "run_name": arm_raw["run_name"],
            "planner_label": arm_raw["planner_label"],
            "controller_env": controller_env,
        }
        if observed != expected_arms[role]:
            raise ValueError(f"The {role} arm differs from its single pre-registered definition.")
        arms.append(
            Arm(
                arm_id=_safe_name(arm_raw["id"], f"arms[{index}].id"),
                role=role,
                run_name=_safe_name(arm_raw["run_name"], f"arms[{index}].run_name"),
                planner_label=_safe_name(arm_raw["planner_label"], f"arms[{index}].planner_label"),
                controller_env=dict(controller_env),
            )
        )
    if {arm.role for arm in arms} != {"reference", "candidate"}:
        raise ValueError("arms must contain exactly one reference and one candidate.")

    preregistration = _mapping(raw["preregistration"], "preregistration")
    _exact_keys(
        preregistration,
        {
            "paired_same_scenarios",
            "episode_context_schema_version",
            "risk_trace_required",
            "shield_evidence_schema_version",
            "shield_evidence_required",
            "expected_route_strata_counts",
            "expected_initial_obstacle_direction_counts",
            "seed_window_selection",
            "candidate_count",
            "parameter_scan",
            "adaptive_extension",
            "validation_or_test",
        },
        "preregistration",
    )
    _load_locked_strata(preregistration["expected_route_strata_counts"])
    direction_counts = _mapping(
        preregistration["expected_initial_obstacle_direction_counts"],
        "preregistration.expected_initial_obstacle_direction_counts",
    )
    _exact_keys(direction_counts, {"negative", "positive"}, "obstacle direction counts")
    if direction_counts != {"negative": 60, "positive": 60}:
        raise ValueError("Expected initial obstacle directions must be exactly 60/60.")
    selection = _mapping(
        preregistration["seed_window_selection"],
        "preregistration.seed_window_selection",
    )
    _exact_keys(
        selection,
        {"search_start", "rule", "covariates_only", "outcome_data_used"},
        "seed_window_selection",
    )
    if selection != {
        "search_start": 20_614,
        "rule": "first_contiguous_120_balancing_route_reverse_and_obstacle_direction",
        "covariates_only": [
            "route_axis",
            "reverse_route",
            "initial_obstacle_direction",
        ],
        "outcome_data_used": False,
    }:
        raise PermissionError("Seed-window selection must use only the locked covariate rule.")
    if _first_balanced_seed_window(20_614, 120) != _FIXED_SEED_START:
        raise RuntimeError("Pre-treatment seed-window search no longer selects seed 25728.")
    locked_preregistration = (
        preregistration["paired_same_scenarios"] is True
        and preregistration["episode_context_schema_version"] == "1.1"
        and preregistration["risk_trace_required"] is True
        and preregistration["shield_evidence_schema_version"] == "1.0"
        and preregistration["shield_evidence_required"] is True
        and preregistration["candidate_count"] == 1
        and preregistration["parameter_scan"] == "forbidden"
        and preregistration["adaptive_extension"] == "forbidden"
        and preregistration["validation_or_test"] == "forbidden"
    )
    if not locked_preregistration:
        raise PermissionError(
            "Preregistration must lock one paired candidate, schema-1.1 risk traces, "
            "schema-1.0 shield evidence, and forbid scanning, adaptive extension, "
            "validation, and test access."
        )
    if EPISODE_CONTEXT_SCHEMA_VERSION != "1.1":
        raise RuntimeError("Installed episode-context schema differs from locked schema 1.1.")
    if SHIELD_EVIDENCE_SCHEMA_VERSION != "1.0":
        raise RuntimeError("Installed shield-evidence schema differs from locked schema 1.0.")

    gate_raw = _mapping(raw["candidate_gate"], "candidate_gate")
    gate_fields = {
        "require_all",
        "minimum_relative_collision_reduction",
        "exact_one_sided_mcnemar_significance_level",
        "maximum_new_timeout_count",
        "maximum_median_completion_time_ratio",
        "minimum_common_success_count",
        "maximum_error_count_per_arm",
    }
    _exact_keys(gate_raw, gate_fields, "candidate_gate")
    gate = CandidateGate(
        require_all=gate_raw["require_all"] is True,
        minimum_relative_collision_reduction=_finite_number(
            gate_raw["minimum_relative_collision_reduction"],
            "candidate_gate.minimum_relative_collision_reduction",
        ),
        exact_one_sided_mcnemar_significance_level=_finite_number(
            gate_raw["exact_one_sided_mcnemar_significance_level"],
            "candidate_gate.exact_one_sided_mcnemar_significance_level",
        ),
        maximum_new_timeout_count=_integer(
            gate_raw["maximum_new_timeout_count"],
            "candidate_gate.maximum_new_timeout_count",
        ),
        maximum_median_completion_time_ratio=_finite_number(
            gate_raw["maximum_median_completion_time_ratio"],
            "candidate_gate.maximum_median_completion_time_ratio",
        ),
        minimum_common_success_count=_integer(
            gate_raw["minimum_common_success_count"],
            "candidate_gate.minimum_common_success_count",
        ),
        maximum_error_count_per_arm=_integer(
            gate_raw["maximum_error_count_per_arm"],
            "candidate_gate.maximum_error_count_per_arm",
        ),
    )
    if gate != CandidateGate(True, 0.25, 0.05, 2, 1.15, 24, 0):
        raise ValueError("candidate_gate differs from the fixed require-all thresholds.")

    decision_raw = _mapping(raw["decision"], "decision")
    _exact_keys(
        decision_raw,
        {"pass", "fail", "invalid_manipulation", "validation_or_test_access"},
        "decision",
    )
    decision = {name: _safe_name(value, f"decision.{name}") for name, value in decision_raw.items()}
    if decision != {
        "pass": "retain_front_brake_candidate_for_independent_confirmation",
        "fail": "reject_front_brake_candidate",
        "invalid_manipulation": "invalidate_front_brake_diagnostic",
        "validation_or_test_access": "forbidden",
    }:
        raise PermissionError("decision must preserve the diagnostic-only fixed outcomes.")

    return DiagnosticConfig(
        source_path=source,
        source_portable_path=source_portable,
        source_sha256=_sha256_file(source),
        experiment_id=experiment_id,
        condition_id=condition_id,
        training_seed=training_seed,
        scenario_seed_start=seed_start,
        episodes=episodes,
        model=model,
        scenario=scenario,
        arms=(arms[0], arms[1]),
        gate=gate,
        decision=decision,
    )


def _portable_binding(path: Path, repository_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(repository_root.resolve()).as_posix(),
        "sha256": _sha256_file(resolved),
    }


def _source_bindings(
    *,
    repository_root: Path,
    config: DiagnosticConfig,
    webots_executable: Path,
    git_revision: str,
) -> dict[str, Any]:
    root = repository_root.resolve()
    bindings: dict[str, Any] = {
        "git_revision": git_revision,
        "config": _portable_binding(config.source_path, root),
        "runner": _portable_binding(root / _RUNNER, root),
        "controller": _portable_binding(root / _CONTROLLER, root),
        "world": _portable_binding(root / _WORLD, root),
        "safety_shield": _portable_binding(root / _SHIELD_IMPLEMENTATION, root),
        "control": _portable_binding(root / _CONTROL_IMPLEMENTATION, root),
        "webots_patrol_environment": _portable_binding(root / _ENVIRONMENT_IMPLEMENTATION, root),
        "evaluation_exports": _portable_binding(root / _EVALUATION_EXPORTS, root),
        "evaluation_metrics": _portable_binding(root / _METRICS_IMPLEMENTATION, root),
        "webots_formal_config": _portable_binding(root / _FORMAL_CONFIG_IMPLEMENTATION, root),
        "episode_context": _portable_binding(root / _CONTEXT_IMPLEMENTATION, root),
        "shield_evidence": _portable_binding(root / _SHIELD_EVIDENCE_IMPLEMENTATION, root),
        "webots_curriculum": _portable_binding(root / _CURRICULUM_IMPLEMENTATION, root),
        "model": _portable_binding(config.model.path, root),
        "webots_executable": {
            "path": str(webots_executable.resolve()),
            "sha256": _sha256_file(webots_executable.resolve()),
        },
    }
    if bindings["config"]["sha256"] != config.source_sha256:
        raise RuntimeError("Config changed during preflight.")
    if bindings["model"]["sha256"] != config.model.sha256:
        raise RuntimeError("Model changed during preflight.")
    return bindings


def _verify_source_bindings(
    bindings: Mapping[str, Any], *, repository_root: Path, config: DiagnosticConfig
) -> None:
    root = repository_root.resolve()
    for label in (
        "config",
        "runner",
        "controller",
        "world",
        "safety_shield",
        "control",
        "webots_patrol_environment",
        "evaluation_exports",
        "evaluation_metrics",
        "webots_formal_config",
        "episode_context",
        "shield_evidence",
        "webots_curriculum",
        "model",
    ):
        binding = _mapping(bindings[label], f"source_bindings.{label}")
        _verify_file(root / str(binding["path"]), str(binding["sha256"]), label)
    executable = _mapping(bindings["webots_executable"], "webots_executable binding")
    _verify_file(Path(str(executable["path"])), str(executable["sha256"]), "Webots executable")
    _verify_file(config.model.path, config.model.sha256, "Recovery-v1 model")


def _clean_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("WEBOTS_") and name != "WEBOTS_HOME":
            environment.pop(name)
    environment.update(overrides)
    return environment


def build_run_plan(
    config: DiagnosticConfig,
    *,
    repository_root: Path,
    webots_executable: Path,
    port_base: int,
) -> tuple[DiagnosticRun, DiagnosticRun]:
    root = repository_root.resolve()
    world = (root / _WORLD).resolve()
    controller = (root / _CONTROLLER).resolve()
    for label, source in (("world", world), ("controller", controller)):
        if not source.is_file():
            raise FileNotFoundError(f"Webots evaluation {label} does not exist: {source}")
    if not 1_024 <= port_base <= 65_500:
        raise ValueError("port_base must be between 1024 and 65500.")
    report_directory = root / _REPORT_DIRECTORY.format(experiment_id=config.experiment_id)
    staging_directory = report_directory.with_name(report_directory.name + ".staging")
    destinations = [root / "artifacts/results" / arm.run_name for arm in config.arms]
    existing = [
        path for path in [*destinations, report_directory, staging_directory] if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Front-brake audit refuses to overwrite artifact directories: "
            + ", ".join(str(path) for path in existing)
        )
    reserved: set[int] = set()
    runs: list[DiagnosticRun] = []
    for arm in config.arms:
        port = choose_available_port(port_base, reserved)
        overrides = {
            "WEBOTS_PORT": str(port),
            "WEBOTS_MODEL_SEED": str(config.training_seed),
            "WEBOTS_EVAL_MODEL": config.model.portable_path,
            "WEBOTS_EVAL_MODEL_SHA256": config.model.sha256,
            "WEBOTS_MODEL_VARIANT": config.model.variant,
            "WEBOTS_EVAL_SPLIT": "diagnostic",
            "WEBOTS_EVAL_SEED_START": str(config.scenario_seed_start),
            "WEBOTS_EVAL_EPISODES": str(config.episodes),
            "WEBOTS_EVAL_CONDITION": config.condition_id,
            "WEBOTS_EVAL_PLANNER": "ppo",
            "WEBOTS_EVAL_PLANNER_LABEL": arm.planner_label,
            "WEBOTS_EVAL_RUN_NAME": arm.run_name,
            "WEBOTS_OBSTACLE_COUNT": str(config.scenario.obstacle_count),
            "WEBOTS_OBSTACLE_SPEED_MIN": str(config.scenario.obstacle_speed_range[0]),
            "WEBOTS_OBSTACLE_SPEED_MAX": str(config.scenario.obstacle_speed_range[1]),
            "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": str(
                config.scenario.horizontal_route_half_length
            ),
            "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": str(config.scenario.vertical_route_half_length),
            "WEBOTS_EVAL_DETERMINISTIC_POLICY": "1",
            "WEBOTS_EVAL_EPISODE_CONTEXT": "1",
            "WEBOTS_EVAL_SHIELD_EVIDENCE": "1",
            "WEBOTS_FRONT_BRAKE_DIAGNOSTIC_ID": config.experiment_id,
            "WEBOTS_FRONT_BRAKE_DIAGNOSTIC_CONFIG": config.source_portable_path,
            "WEBOTS_FRONT_BRAKE_DIAGNOSTIC_CONFIG_SHA256": config.source_sha256,
            **arm.controller_env,
        }
        command = (
            str(webots_executable.resolve()),
            f"--port={port}",
            "--batch",
            "--mode=fast",
            "--no-rendering",
            "--stdout",
            "--stderr",
            str(world),
        )
        runs.append(
            DiagnosticRun(
                arm=arm,
                port=port,
                command=command,
                environment=overrides,
                result_directory=root / "artifacts/results" / arm.run_name,
            )
        )
    return runs[0], runs[1]


def _csv_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{label} must be True or False in episodes.csv.")


def _read_episode_rows(
    path: Path, *, config: DiagnosticConfig, arm: Arm
) -> dict[int, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _EPISODE_FIELDS:
            raise ValueError("episodes.csv schema does not match EpisodeRecord v1.0.")
        raw_rows = list(reader)
    if len(raw_rows) != config.episodes:
        raise ValueError(f"episodes.csv must contain exactly {config.episodes} rows.")
    result: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        try:
            seed = int(raw["scenario_seed"])
        except ValueError as error:
            raise ValueError(f"episodes.csv row {index} has an invalid scenario_seed.") from error
        if seed in result:
            raise ValueError(f"episodes.csv contains duplicate scenario seed {seed}.")
        expected = {
            "schema_version": "1.0",
            "episode_id": f"diagnostic-{config.condition_id}-{seed}-{arm.planner_label}",
            "split": "diagnostic",
            "condition_id": config.condition_id,
            "planner": arm.planner_label,
        }
        for field, value in expected.items():
            if raw[field] != value:
                raise ValueError(f"episodes.csv row {index} {field} disagrees with the plan.")
        outcome = raw["outcome"]
        if outcome not in {"success", "collision", "timeout", "error"}:
            raise ValueError(f"episodes.csv row {index} has an invalid outcome.")
        success = _csv_bool(raw["is_success"], f"episodes.csv row {index} is_success")
        collision = _csv_bool(raw["collision"], f"episodes.csv row {index} collision")
        timeout = _csv_bool(raw["timeout"], f"episodes.csv row {index} timeout")
        if (
            (outcome == "success") != success
            or (outcome == "collision") != collision
            or (outcome == "timeout") != timeout
        ):
            raise ValueError(f"episodes.csv row {index} outcome flags are inconsistent.")
        try:
            step_count = int(raw["step_count"])
            elapsed_sim_s = float(raw["elapsed_sim_s"])
        except ValueError as error:
            raise ValueError(f"episodes.csv row {index} has invalid timing fields.") from error
        if step_count < 1 or not math.isfinite(elapsed_sim_s) or elapsed_sim_s <= 0.0:
            raise ValueError(f"episodes.csv row {index} has invalid timing fields.")
        completion_raw = raw["completion_time_s"].strip()
        completion_time: float | None = None
        if success:
            try:
                completion_time = float(completion_raw)
            except ValueError as error:
                raise ValueError(
                    f"episodes.csv row {index} success lacks completion_time_s."
                ) from error
            if not math.isfinite(completion_time) or completion_time <= 0.0:
                raise ValueError(f"episodes.csv row {index} completion_time_s is invalid.")
            if not math.isclose(completion_time, elapsed_sim_s, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"episodes.csv row {index} completion_time_s must equal elapsed_sim_s."
                )
        elif completion_raw:
            raise ValueError(f"episodes.csv row {index} non-success has completion_time_s.")
        result[seed] = {
            "episode_id": raw["episode_id"],
            "outcome": outcome,
            "is_success": success,
            "collision": collision,
            "timeout": timeout,
            "step_count": step_count,
            "elapsed_sim_s": elapsed_sim_s,
            "completion_time_s": completion_time,
        }
    if tuple(result) != config.seeds:
        raise ValueError("episodes.csv must preserve the exact pre-registered seed order.")
    return result


def _read_contexts(
    path: Path,
    *,
    config: DiagnosticConfig,
    arm: Arm,
    episode_rows: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    records: list[Any] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"episode_context.jsonl line {line_number} is blank.")
            records.append(json.loads(line))
    if len(records) != config.episodes:
        raise ValueError(f"episode_context.jsonl must contain exactly {config.episodes} records.")
    contexts: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(records):
        record = _mapping(value, f"episode_context row {index}")
        if record.get("schema_version") != "1.1":
            raise ValueError(f"episode_context row {index} must use schema 1.1.")
        identity = _mapping(record.get("identity"), f"episode_context row {index}.identity")
        _exact_keys(
            identity,
            {"episode_id", "split", "condition_id", "scenario_seed", "planner"},
            f"episode_context row {index}.identity",
        )
        seed = identity["scenario_seed"]
        if type(seed) is not int or seed not in episode_rows or seed in contexts:
            raise ValueError(f"episode_context row {index} has an invalid or duplicate seed.")
        expected_identity = {
            "episode_id": episode_rows[seed]["episode_id"],
            "split": "diagnostic",
            "condition_id": config.condition_id,
            "scenario_seed": seed,
            "planner": arm.planner_label,
        }
        if identity != expected_identity:
            raise ValueError(f"episode_context row {index} identity disagrees with episodes.csv.")
        expected_axis, expected_reverse, expected_obstacle_direction = (
            _scenario_covariates_for_seed(seed)
        )
        if record.get("route_axis") != expected_axis:
            raise ValueError(f"episode_context seed {seed} has an unexpected route_axis.")
        if record.get("reverse_route") is not expected_reverse:
            raise ValueError(f"episode_context seed {seed} has an unexpected reverse_route.")
        start_xy = record.get("start_xy")
        goal_xy = record.get("goal_xy")
        if not (
            isinstance(start_xy, list)
            and isinstance(goal_xy, list)
            and len(start_xy) == len(goal_xy) == 2
        ):
            raise ValueError(f"episode_context row {index} route coordinates are invalid.")
        route_index = 0 if expected_axis == "horizontal" else 1
        route_half = (
            config.scenario.horizontal_route_half_length
            if expected_axis == "horizontal"
            else config.scenario.vertical_route_half_length
        )
        expected_start = route_half if expected_reverse else -route_half
        expected_goal = -expected_start
        if not (
            math.isclose(float(start_xy[route_index]), expected_start, abs_tol=1e-12)
            and math.isclose(float(goal_xy[route_index]), expected_goal, abs_tol=1e-12)
        ):
            raise ValueError(f"episode_context seed {seed} route direction disagrees with RNG.")
        _finite_number(record.get("start_yaw"), f"episode_context row {index}.start_yaw")
        obstacles = record.get("dynamic_obstacles")
        if not isinstance(obstacles, list) or len(obstacles) != 1:
            raise ValueError(f"episode_context seed {seed} must contain one obstacle.")
        obstacle = _mapping(obstacles[0], f"episode_context seed {seed}.obstacle")
        speed = _finite_number(obstacle.get("speed"), "dynamic_obstacle.speed")
        if not 0.15 <= speed <= 0.30:
            raise ValueError(f"episode_context seed {seed} obstacle speed is out of range.")
        if obstacle.get("initial_direction") != expected_obstacle_direction:
            raise ValueError(
                f"episode_context seed {seed} initial obstacle direction disagrees with RNG."
            )
        termination = _mapping(
            record.get("termination"), f"episode_context row {index}.termination"
        )
        terminal_step = termination.get("terminal_step")
        if type(terminal_step) is not int or terminal_step < 1:
            raise ValueError(f"episode_context row {index} terminal_step is invalid.")
        validate_risk_trace(
            record.get("risk_trace"),
            field=f"episode_context row {index}.risk_trace",
        )
        trace = record["risk_trace"]
        if trace["terminal_window"][-1]["step"] != terminal_step:
            raise ValueError(f"episode_context row {index} risk trace misses termination.")
        for window_name in ("terminal_window", "minimum_clearance_window"):
            for sample in trace[window_name]:
                if len(sample["post_obstacle_xy"]) != 1:
                    raise ValueError(
                        f"episode_context row {index} risk trace obstacle count is invalid."
                    )
        final_sample = trace["terminal_window"][-1]
        for collision_source in ("contact", "lidar"):
            if (
                termination.get(f"{collision_source}_collision")
                != final_sample[f"post_{collision_source}_collision"]
            ):
                raise ValueError(
                    f"episode_context row {index} terminal collision disagrees with trace."
                )
        observed_collision = bool(
            termination.get("contact_collision") or termination.get("lidar_collision")
        )
        if observed_collision != episode_rows[seed]["collision"]:
            raise ValueError(f"episode_context row {index} collision disagrees with episodes.csv.")
        contexts[seed] = record
    if tuple(contexts) != config.seeds:
        raise ValueError("episode_context.jsonl must preserve the exact seed order.")
    if _strata_counts(tuple(contexts)) != _FIXED_STRATA:
        raise ValueError("Collected contexts do not contain four 30-episode route strata.")
    observed_directions = Counter(
        float(context["dynamic_obstacles"][0]["initial_direction"]) for context in contexts.values()
    )
    if observed_directions != {-1.0: 60, 1.0: 60}:
        raise ValueError("Collected contexts do not balance initial obstacle direction 60/60.")
    return contexts


def _read_shield_evidence(
    path: Path,
    *,
    config: DiagnosticConfig,
    arm: Arm,
    episode_rows: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    records: list[Any] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"shield_evidence.jsonl line {line_number} is blank.")
            records.append(json.loads(line))
    if len(records) != config.episodes:
        raise ValueError(f"shield_evidence.jsonl must contain exactly {config.episodes} records.")
    by_seed: dict[int, dict[str, Any]] = {}
    expected_enabled = arm.role == "candidate"
    for index, value in enumerate(records):
        record = _mapping(value, f"shield_evidence row {index}")
        validate_shield_evidence_record(record, field=f"shield_evidence row {index}")
        identity = _mapping(record["identity"], f"shield_evidence row {index}.identity")
        seed = identity["scenario_seed"]
        if type(seed) is not int or seed not in episode_rows or seed in by_seed:
            raise ValueError(f"shield_evidence row {index} has an invalid or duplicate seed.")
        expected_identity = {
            "episode_id": episode_rows[seed]["episode_id"],
            "split": "diagnostic",
            "condition_id": config.condition_id,
            "scenario_seed": seed,
            "planner": arm.planner_label,
        }
        if identity != expected_identity:
            raise ValueError(f"shield_evidence row {index} identity disagrees with episodes.csv.")
        summary = _mapping(record["summary"], f"shield_evidence row {index}.summary")
        if summary["shield_enabled"] is not expected_enabled:
            raise ValueError(f"shield_evidence row {index} enabled state disagrees with arm.")
        expected_epsilon = _expected_shield_config(arm)["intervention_epsilon"]
        if not math.isclose(
            float(summary["intervention_epsilon"]),
            expected_epsilon,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"shield_evidence row {index} intervention epsilon disagrees with profile."
            )
        if summary["step_count"] != episode_rows[seed]["step_count"]:
            raise ValueError(f"shield_evidence row {index} step_count disagrees with episodes.csv.")
        by_seed[seed] = record
    if tuple(by_seed) != config.seeds:
        raise ValueError("shield_evidence.jsonl must preserve the exact seed order.")
    return by_seed


def _shield_evidence_metrics(
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    steps = [step for record in records.values() for step in record["steps"]]
    summaries = [record["summary"] for record in records.values()]
    total_steps = len(steps)
    active_steps = sum(bool(step["shield_active"]) for step in steps)
    braking_steps = 0
    speed_cap_binding_steps = 0
    maximum_angular_delta = 0.0
    for step in steps:
        policy = step["policy_action"]
        shielded = step["shielded_action"]
        epsilon = 1e-6
        if float(shielded[0]) < float(policy[0]) - epsilon:
            braking_steps += 1
        speed_cap = float(step["shield_normalized_speed_limit"])
        if float(policy[0]) > speed_cap + epsilon and math.isclose(
            float(shielded[0]), speed_cap, rel_tol=0.0, abs_tol=1e-6
        ):
            speed_cap_binding_steps += 1
        maximum_angular_delta = max(
            maximum_angular_delta,
            abs(float(shielded[1]) - float(policy[1])),
        )
    return {
        "episode_count": len(records),
        "enabled_episode_count": sum(bool(summary["shield_enabled"]) for summary in summaries),
        "episode_with_active_step_count": sum(
            int(summary["active_step_count"] > 0) for summary in summaries
        ),
        "total_step_count": total_steps,
        "active_step_count": active_steps,
        "active_step_ratio": active_steps / total_steps,
        "intervention_sum": math.fsum(float(summary["intervention_sum"]) for summary in summaries),
        "braking_step_count": braking_steps,
        "speed_cap_binding_step_count": speed_cap_binding_steps,
        "maximum_absolute_angular_action_delta": maximum_angular_delta,
    }


def _outcome_counts(rows: Mapping[int, Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, int]:
    outcomes = Counter(str(rows[seed]["outcome"]) for seed in seeds)
    return {
        "episodes": len(seeds),
        "success_count": outcomes["success"],
        "collision_count": outcomes["collision"],
        "timeout_count": outcomes["timeout"],
        "error_count": outcomes["error"],
    }


def _count_metrics(
    rows: Mapping[int, Mapping[str, Any]],
    contexts: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    seeds = tuple(sorted(rows))

    def selected(
        *,
        axis: str | None = None,
        reverse: bool | None = None,
        obstacle_direction: float | None = None,
    ) -> list[int]:
        return [
            seed
            for seed in seeds
            if (axis is None or contexts[seed]["route_axis"] == axis)
            and (reverse is None or contexts[seed]["reverse_route"] is reverse)
            and (
                obstacle_direction is None
                or contexts[seed]["dynamic_obstacles"][0]["initial_direction"] == obstacle_direction
            )
        ]

    return {
        "overall": _outcome_counts(rows, seeds),
        "by_route_axis": {
            axis: _outcome_counts(rows, selected(axis=axis)) for axis in ("horizontal", "vertical")
        },
        "by_route_direction": {
            name: _outcome_counts(rows, selected(reverse=reverse))
            for name, reverse in (("non_reverse", False), ("reverse", True))
        },
        "by_initial_obstacle_direction": {
            name: _outcome_counts(rows, selected(obstacle_direction=direction))
            for name, direction in (("negative", -1.0), ("positive", 1.0))
        },
        "by_route_axis_and_direction": {
            axis: {
                name: _outcome_counts(rows, selected(axis=axis, reverse=reverse))
                for name, reverse in (("non_reverse", False), ("reverse", True))
            }
            for axis in ("horizontal", "vertical")
        },
    }


def _expected_shield_config(arm: Arm) -> dict[str, float]:
    environment = {
        name: value
        for name, value in arm.controller_env.items()
        if name.startswith("WEBOTS_SAFETY_SHIELD_")
    }
    return asdict(safety_shield_config_from_env(environment, lidar_max_range=3.5))


def _check_summary(
    summary_path: Path,
    *,
    repository_root: Path,
    config: DiagnosticConfig,
    arm: Arm,
    rows: Mapping[int, Mapping[str, Any]],
    contexts: Mapping[int, Mapping[str, Any]],
    episodes_sha256: str,
    context_path: Path,
    context_sha256: str,
    evidence_path: Path,
    evidence_sha256: str,
) -> None:
    summary = _mapping(json.loads(summary_path.read_text(encoding="utf-8")), "summary.json")
    _exact_keys(
        summary,
        {
            "schema_version",
            "split",
            "episode_count",
            "planners",
            "conditions",
            "by_planner_condition",
            "by_planner",
            "evaluation_provenance",
        },
        "summary.json",
    )
    for name, expected in {
        "schema_version": "1.0",
        "split": "diagnostic",
        "episode_count": config.episodes,
        "planners": [arm.planner_label],
        "conditions": [config.condition_id],
    }.items():
        if summary[name] != expected:
            raise ValueError(f"{arm.arm_id} summary.json {name} disagrees with the plan.")
    by_planner = summary["by_planner"]
    if not isinstance(by_planner, list) or len(by_planner) != 1:
        raise ValueError("summary.json must contain exactly one planner aggregate.")
    aggregate = _mapping(by_planner[0], "summary.by_planner[0]")
    expected_counts = _count_metrics(rows, contexts)["overall"]
    for name, expected in expected_counts.items():
        if aggregate.get(name) != expected:
            raise ValueError(f"summary aggregate {name} disagrees with episodes.csv.")

    provenance = _mapping(summary["evaluation_provenance"], "evaluation_provenance")
    expected_provenance_keys = {
        "schema_version",
        "model_path",
        "model_sha256",
        "model_variant",
        "training_seed",
        "split",
        "condition_id",
        "planner",
        "planner_label",
        "deterministic_policy",
        "scenario_seed_start",
        "scenario_seed_count",
        "scenario_seeds",
        "obstacle_count",
        "obstacle_cross_lanes",
        "obstacle_spawn_protocol",
        "obstacle_lane_seed_transform",
        "obstacle_speed_range",
        "horizontal_route_half_length",
        "vertical_route_half_length",
        "static_layout",
        "safety_shield_enabled",
        "safety_shield_config",
        "reference_path_definition",
        "robot_bounding_radius",
        "reset_health",
        "episode_context",
        "shield_evidence",
        "episodes_csv_sha256",
    }
    _exact_keys(provenance, expected_provenance_keys, "evaluation_provenance")
    expected_values = {
        "schema_version": 1,
        "model_path": config.model.portable_path,
        "model_sha256": config.model.sha256,
        "model_variant": config.model.variant,
        "training_seed": config.training_seed,
        "split": "diagnostic",
        "condition_id": config.condition_id,
        "planner": "ppo",
        "planner_label": arm.planner_label,
        "deterministic_policy": True,
        "scenario_seed_start": config.scenario_seed_start,
        "scenario_seed_count": config.episodes,
        "scenario_seeds": list(config.seeds),
        "obstacle_count": 1,
        "obstacle_cross_lanes": None,
        "obstacle_spawn_protocol": "training_random_cross_axis",
        "obstacle_lane_seed_transform": None,
        "obstacle_speed_range": [0.15, 0.30],
        "horizontal_route_half_length": 1.10,
        "vertical_route_half_length": 1.10,
        "static_layout": "train_a",
        "safety_shield_enabled": arm.role == "candidate",
        "safety_shield_config": _expected_shield_config(arm),
        "reference_path_definition": "euclidean_start_to_goal",
        "robot_bounding_radius": 0.122,
        "episodes_csv_sha256": episodes_sha256,
    }
    for name, expected in expected_values.items():
        if provenance[name] != expected:
            raise ValueError(f"evaluation_provenance.{name} disagrees with the plan.")
    expected_context = {
        "path": context_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "schema_version": "1.1",
        "count": config.episodes,
        "sha256": context_sha256,
    }
    if provenance["episode_context"] != expected_context:
        raise ValueError("summary episode_context provenance binding is invalid.")
    expected_evidence = {
        "path": evidence_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "schema_version": "1.0",
        "count": config.episodes,
        "sha256": evidence_sha256,
    }
    if provenance["shield_evidence"] != expected_evidence:
        raise ValueError("summary shield_evidence provenance binding is invalid.")
    reset_health = _mapping(provenance["reset_health"], "reset_health")
    _exact_keys(
        reset_health,
        {
            "attempts",
            "successful",
            "finite_settle_checks",
            "readback_checks",
            "maximum_translation_error",
            "maximum_start_xy_error",
            "maximum_gps_root_xy_distance",
            "maximum_yaw_error",
            "maximum_linear_speed",
            "maximum_angular_speed",
        },
        "reset_health",
    )
    for name, expected in {
        "attempts": config.episodes,
        "successful": config.episodes,
        "finite_settle_checks": 2 * config.episodes,
        "readback_checks": config.episodes,
    }.items():
        if reset_health[name] != expected:
            raise ValueError(f"reset_health.{name} disagrees with a complete run.")
    for name in (
        "maximum_translation_error",
        "maximum_start_xy_error",
        "maximum_gps_root_xy_distance",
        "maximum_yaw_error",
        "maximum_linear_speed",
        "maximum_angular_speed",
    ):
        value = _finite_number(reset_health[name], f"reset_health.{name}")
        if value < 0.0:
            raise ValueError(f"reset_health.{name} cannot be negative.")


def _parse_arm_result(
    run: DiagnosticRun,
    *,
    config: DiagnosticConfig,
    repository_root: Path,
) -> ParsedArmResult:
    expected_names = {
        "summary.json",
        "episodes.csv",
        "episode_context.jsonl",
        "shield_evidence.jsonl",
    }
    observed_names = {path.name for path in run.result_directory.iterdir()}
    if observed_names != expected_names:
        raise ValueError(
            f"{run.arm.arm_id} result files differ from the exact protocol: "
            f"{sorted(observed_names)}."
        )
    summary_path = run.result_directory / "summary.json"
    episodes_path = run.result_directory / "episodes.csv"
    context_path = run.result_directory / "episode_context.jsonl"
    evidence_path = run.result_directory / "shield_evidence.jsonl"
    episodes_sha256 = _sha256_file(episodes_path)
    context_sha256 = _sha256_file(context_path)
    evidence_sha256 = _sha256_file(evidence_path)
    rows = _read_episode_rows(episodes_path, config=config, arm=run.arm)
    contexts = _read_contexts(
        context_path,
        config=config,
        arm=run.arm,
        episode_rows=rows,
    )
    shield_evidence = _read_shield_evidence(
        evidence_path,
        config=config,
        arm=run.arm,
        episode_rows=rows,
    )
    _check_summary(
        summary_path,
        repository_root=repository_root,
        config=config,
        arm=run.arm,
        rows=rows,
        contexts=contexts,
        episodes_sha256=episodes_sha256,
        context_path=context_path,
        context_sha256=context_sha256,
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
    )
    root = repository_root.resolve()
    return ParsedArmResult(
        arm=run.arm,
        rows_by_seed=rows,
        contexts_by_seed=contexts,
        shield_evidence_by_seed=shield_evidence,
        shield_evidence_metrics=_shield_evidence_metrics(shield_evidence),
        metrics=_count_metrics(rows, contexts),
        artifacts={
            "result_directory": run.result_directory.resolve().relative_to(root).as_posix(),
            "summary_json": _portable_binding(summary_path, root),
            "episodes_csv": _portable_binding(episodes_path, root),
            "episode_context_jsonl": _portable_binding(context_path, root),
            "shield_evidence_jsonl": _portable_binding(evidence_path, root),
        },
    )


def _exact_one_sided_mcnemar_p(improvements: int, regressions: int) -> float:
    """Exact upper-tail binomial p for H1: candidate has higher success probability."""

    if min(improvements, regressions) < 0:
        raise ValueError("McNemar discordant counts cannot be negative.")
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    numerator = sum(math.comb(discordant, value) for value in range(improvements, discordant + 1))
    return numerator / (2**discordant)


def _paired_scope(
    reference: ParsedArmResult,
    candidate: ParsedArmResult,
    seeds: Sequence[int],
) -> dict[str, Any]:
    transitions = Counter(
        f"{reference.rows_by_seed[seed]['outcome']}_to_{candidate.rows_by_seed[seed]['outcome']}"
        for seed in seeds
    )
    reference_collision_count = sum(
        bool(reference.rows_by_seed[seed]["collision"]) for seed in seeds
    )
    candidate_collision_count = sum(
        bool(candidate.rows_by_seed[seed]["collision"]) for seed in seeds
    )
    success_rescues = sum(
        not bool(reference.rows_by_seed[seed]["is_success"])
        and bool(candidate.rows_by_seed[seed]["is_success"])
        for seed in seeds
    )
    success_regressions = sum(
        bool(reference.rows_by_seed[seed]["is_success"])
        and not bool(candidate.rows_by_seed[seed]["is_success"])
        for seed in seeds
    )
    relative_reduction = (
        (reference_collision_count - candidate_collision_count) / reference_collision_count
        if reference_collision_count
        else None
    )
    reference_timeout_count = sum(bool(reference.rows_by_seed[seed]["timeout"]) for seed in seeds)
    candidate_timeout_count = sum(bool(candidate.rows_by_seed[seed]["timeout"]) for seed in seeds)
    new_timeouts = sum(
        not bool(reference.rows_by_seed[seed]["timeout"])
        and bool(candidate.rows_by_seed[seed]["timeout"])
        for seed in seeds
    )
    resolved_timeouts = sum(
        bool(reference.rows_by_seed[seed]["timeout"])
        and not bool(candidate.rows_by_seed[seed]["timeout"])
        for seed in seeds
    )
    reference_success_seeds = [seed for seed in seeds if reference.rows_by_seed[seed]["is_success"]]
    paired_completion_seeds = [
        seed for seed in reference_success_seeds if candidate.rows_by_seed[seed]["is_success"]
    ]
    completion_time_ratios = [
        float(candidate.rows_by_seed[seed]["completion_time_s"])
        / float(reference.rows_by_seed[seed]["completion_time_s"])
        for seed in paired_completion_seeds
    ]
    ratio_quantiles = (
        np.quantile(np.asarray(completion_time_ratios), [0.25, 0.50, 0.75, 0.90])
        if completion_time_ratios
        else None
    )
    return {
        "episodes": len(seeds),
        "outcome_transitions": dict(sorted(transitions.items())),
        "collision": {
            "reference_count": reference_collision_count,
            "candidate_count": candidate_collision_count,
            "absolute_reduction_count": reference_collision_count - candidate_collision_count,
            "relative_reduction": relative_reduction,
        },
        "paired_success_endpoint": {
            "rescues_reference_non_success_to_candidate_success": success_rescues,
            "regressions_reference_success_to_candidate_non_success": success_regressions,
            "discordant_total": success_rescues + success_regressions,
            "exact_one_sided_mcnemar_p": _exact_one_sided_mcnemar_p(
                success_rescues, success_regressions
            ),
            "alternative": "candidate_success_probability_is_higher",
        },
        "timeouts": {
            "reference_count": reference_timeout_count,
            "candidate_count": candidate_timeout_count,
            "new_timeout_count": new_timeouts,
            "resolved_timeout_count": resolved_timeouts,
            "net_increase_count": candidate_timeout_count - reference_timeout_count,
        },
        "errors": {
            "reference_count": sum(
                reference.rows_by_seed[seed]["outcome"] == "error" for seed in seeds
            ),
            "candidate_count": sum(
                candidate.rows_by_seed[seed]["outcome"] == "error" for seed in seeds
            ),
        },
        "completion_delay_on_reference_success": {
            "reference_success_count": len(reference_success_seeds),
            "paired_success_count": len(paired_completion_seeds),
            "candidate_noncompletion_count": len(reference_success_seeds)
            - len(paired_completion_seeds),
            "eligible_transition": "success_to_success",
            "estimand": "median_of_per_scenario_candidate_to_reference_completion_time_ratio",
            "quantile_method": "numpy_linear",
            "median_ratio": (
                float(statistics.median(completion_time_ratios)) if completion_time_ratios else None
            ),
            "ratio_q25": float(ratio_quantiles[0]) if ratio_quantiles is not None else None,
            "ratio_q75": float(ratio_quantiles[2]) if ratio_quantiles is not None else None,
            "ratio_iqr": (
                float(ratio_quantiles[2] - ratio_quantiles[0])
                if ratio_quantiles is not None
                else None
            ),
            "ratio_p90": float(ratio_quantiles[3]) if ratio_quantiles is not None else None,
            "median_reference_completion_time_s": (
                float(
                    statistics.median(
                        float(reference.rows_by_seed[seed]["completion_time_s"])
                        for seed in paired_completion_seeds
                    )
                )
                if paired_completion_seeds
                else None
            ),
            "median_candidate_completion_time_s": (
                float(
                    statistics.median(
                        float(candidate.rows_by_seed[seed]["completion_time_s"])
                        for seed in paired_completion_seeds
                    )
                )
                if paired_completion_seeds
                else None
            ),
        },
    }


def _paired_analysis(
    reference: ParsedArmResult,
    candidate: ParsedArmResult,
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    by_seed: list[dict[str, Any]] = []
    for seed in seeds:
        for field in (
            "route_axis",
            "reverse_route",
            "start_xy",
            "goal_xy",
            "start_yaw",
            "dynamic_obstacles",
        ):
            if reference.contexts_by_seed[seed].get(field) != candidate.contexts_by_seed[seed].get(
                field
            ):
                raise ValueError(
                    f"Paired arms disagree on initial context {field} for seed {seed}."
                )
        by_seed.append(
            {
                "scenario_seed": seed,
                "route_axis": reference.contexts_by_seed[seed]["route_axis"],
                "reverse_route": reference.contexts_by_seed[seed]["reverse_route"],
                "reference_outcome": reference.rows_by_seed[seed]["outcome"],
                "candidate_outcome": candidate.rows_by_seed[seed]["outcome"],
                "transition": (
                    f"{reference.rows_by_seed[seed]['outcome']}_to_"
                    f"{candidate.rows_by_seed[seed]['outcome']}"
                ),
            }
        )

    def selected(
        *,
        axis: str | None = None,
        reverse: bool | None = None,
        obstacle_direction: float | None = None,
    ) -> list[int]:
        return [
            seed
            for seed in seeds
            if (axis is None or reference.contexts_by_seed[seed]["route_axis"] == axis)
            and (reverse is None or reference.contexts_by_seed[seed]["reverse_route"] is reverse)
            and (
                obstacle_direction is None
                or reference.contexts_by_seed[seed]["dynamic_obstacles"][0]["initial_direction"]
                == obstacle_direction
            )
        ]

    return {
        "by_seed": by_seed,
        "overall": _paired_scope(reference, candidate, seeds),
        "by_route_axis": {
            axis: _paired_scope(reference, candidate, selected(axis=axis))
            for axis in ("horizontal", "vertical")
        },
        "by_route_direction": {
            name: _paired_scope(reference, candidate, selected(reverse=reverse))
            for name, reverse in (("non_reverse", False), ("reverse", True))
        },
        "by_initial_obstacle_direction": {
            name: _paired_scope(
                reference,
                candidate,
                selected(obstacle_direction=direction),
            )
            for name, direction in (("negative", -1.0), ("positive", 1.0))
        },
        "by_route_axis_and_direction": {
            axis: {
                name: _paired_scope(
                    reference,
                    candidate,
                    selected(axis=axis, reverse=reverse),
                )
                for name, reverse in (("non_reverse", False), ("reverse", True))
            }
            for axis in ("horizontal", "vertical")
        },
    }


def _gate_report(gate: CandidateGate, *, paired: Mapping[str, Any]) -> dict[str, Any]:
    overall = _mapping(paired["overall"], "paired.overall")
    collision = _mapping(overall["collision"], "paired.overall.collision")
    success_endpoint = _mapping(
        overall["paired_success_endpoint"], "paired.overall.paired_success_endpoint"
    )
    timeouts = _mapping(overall["timeouts"], "paired.overall.timeouts")
    errors = _mapping(overall["errors"], "paired.overall.errors")
    completion = _mapping(
        overall["completion_delay_on_reference_success"],
        "paired.overall.completion_delay_on_reference_success",
    )
    checks = {
        "minimum_relative_collision_reduction": {
            "actual": collision["relative_reduction"],
            "operator": ">=",
            "threshold": gate.minimum_relative_collision_reduction,
        },
        "exact_one_sided_mcnemar_p": {
            "actual": success_endpoint["exact_one_sided_mcnemar_p"],
            "operator": "<",
            "threshold": gate.exact_one_sided_mcnemar_significance_level,
        },
        "maximum_new_timeout_count": {
            "actual": timeouts["new_timeout_count"],
            "operator": "<=",
            "threshold": gate.maximum_new_timeout_count,
        },
        "maximum_median_completion_time_ratio": {
            "actual": completion["median_ratio"],
            "operator": "<=",
            "threshold": gate.maximum_median_completion_time_ratio,
            "denominator": "success_to_success_scenarios",
        },
        "minimum_common_success_count": {
            "actual": completion["paired_success_count"],
            "operator": ">=",
            "threshold": gate.minimum_common_success_count,
        },
        "maximum_reference_error_count": {
            "actual": errors["reference_count"],
            "operator": "<=",
            "threshold": gate.maximum_error_count_per_arm,
        },
        "maximum_candidate_error_count": {
            "actual": errors["candidate_count"],
            "operator": "<=",
            "threshold": gate.maximum_error_count_per_arm,
        },
    }
    for check in checks.values():
        actual = check["actual"]
        threshold = check["threshold"]
        if actual is None:
            check["passed"] = False
        elif check["operator"] == ">=":
            check["passed"] = actual >= threshold
        elif check["operator"] == "<":
            check["passed"] = actual < threshold
        else:
            check["passed"] = actual <= threshold
    return {
        "require_all": gate.require_all,
        "checks": checks,
        "passed": all(bool(check["passed"]) for check in checks.values()),
    }


def _secondary_manipulation_check(
    reference: ParsedArmResult,
    candidate: ParsedArmResult,
    *,
    expected_episodes: int,
) -> dict[str, Any]:
    reference_metrics = reference.shield_evidence_metrics
    candidate_metrics = candidate.shield_evidence_metrics
    checks = {
        "reference_all_episodes_disabled": (reference_metrics["enabled_episode_count"] == 0),
        "reference_zero_active_steps": reference_metrics["active_step_count"] == 0,
        "reference_zero_intervention": math.isclose(
            float(reference_metrics["intervention_sum"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "candidate_all_episodes_enabled": (
            candidate_metrics["enabled_episode_count"] == expected_episodes
        ),
        "candidate_has_active_steps": candidate_metrics["active_step_count"] > 0,
        "candidate_has_braking_steps": candidate_metrics["braking_step_count"] > 0,
        "candidate_preserves_angular_action": (
            candidate_metrics["maximum_absolute_angular_action_delta"] <= 1e-6
        ),
        "candidate_active_steps_are_exact_speed_cap_braking": (
            candidate_metrics["active_step_count"]
            == candidate_metrics["braking_step_count"]
            == candidate_metrics["speed_cap_binding_step_count"]
        ),
    }
    return {
        "role": "required_experiment_validity_check_not_primary_efficacy_gate",
        "passed": all(checks.values()),
        "checks": checks,
        "reference_metrics": dict(reference_metrics),
        "candidate_metrics": dict(candidate_metrics),
    }


def _verify_parsed_artifacts(results: Sequence[ParsedArmResult], *, repository_root: Path) -> None:
    root = repository_root.resolve()
    for result in results:
        for artifact_name in (
            "summary_json",
            "episodes_csv",
            "episode_context_jsonl",
            "shield_evidence_jsonl",
        ):
            binding = _mapping(
                result.artifacts[artifact_name],
                f"{result.arm.arm_id}.artifacts.{artifact_name}",
            )
            _exact_keys(binding, {"path", "sha256"}, f"artifact {artifact_name}")
            portable = _text(binding["path"], f"artifact {artifact_name}.path")
            path = (root / portable).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"Artifact {artifact_name} escapes the repository.") from error
            _verify_file(path, _text(binding["sha256"], "artifact sha256"), artifact_name)


def _write_report(report: Mapping[str, Any], *, output_directory: Path) -> None:
    staging_directory = output_directory.with_name(output_directory.name + ".staging")
    if output_directory.exists() or staging_directory.exists():
        raise FileExistsError(
            "Report sealing refuses existing final or staging directories: "
            f"{output_directory}, {staging_directory}."
        )
    staging_directory.mkdir(parents=True, exist_ok=False)
    json_path = staging_directory / "paired_diagnostic_report.json"
    yaml_path = staging_directory / "paired_diagnostic_report.yaml"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    yaml_path.write_text(
        yaml.safe_dump(dict(report), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    sidecar_path = staging_directory / "paired_diagnostic_report.sha256"
    sidecar_path.write_text(
        f"{_sha256_file(json_path)}  {json_path.name}\n"
        f"{_sha256_file(yaml_path)}  {yaml_path.name}\n",
        encoding="utf-8",
    )
    expected_names = {
        "paired_diagnostic_report.json",
        "paired_diagnostic_report.yaml",
        "paired_diagnostic_report.sha256",
    }
    if {path.name for path in staging_directory.iterdir()} != expected_names:
        raise RuntimeError("Staged report does not contain exactly three protocol files.")
    if json.loads(json_path.read_text(encoding="utf-8")) != report:
        raise RuntimeError("Staged JSON report failed readback.")
    if yaml.safe_load(yaml_path.read_text(encoding="utf-8")) != report:
        raise RuntimeError("Staged YAML report failed readback.")
    expected_sidecar = (
        f"{_sha256_file(json_path)}  {json_path.name}\n"
        f"{_sha256_file(yaml_path)}  {yaml_path.name}\n"
    )
    if sidecar_path.read_text(encoding="utf-8") != expected_sidecar:
        raise RuntimeError("Staged SHA-256 sidecar failed readback.")
    staging_directory.replace(output_directory)


def execute_diagnostic(
    *,
    repository_root: Path,
    config_path: Path,
    webots_executable: Path,
    dry_run: bool,
    port_base: int = 13_680,
    timeout_seconds: float = 7_200.0,
    process_runner: ProcessRunner = subprocess.run,
    source_verifier: SourceVerifier = require_clean_source,
) -> tuple[int, Mapping[str, Any]]:
    """Preflight and optionally execute the single locked paired candidate."""

    root = repository_root.resolve()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive.")
    source_revision = source_verifier(root)
    if _GIT_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source_verifier returned an invalid Git revision.")
    config = load_diagnostic_config(config_path, repository_root=root)
    webots = resolve_webots_executable(webots_executable)
    runs = build_run_plan(
        config,
        repository_root=root,
        webots_executable=webots,
        port_base=port_base,
    )
    bindings = _source_bindings(
        repository_root=root,
        config=config,
        webots_executable=webots,
        git_revision=source_revision,
    )
    plan: dict[str, Any] = {
        "schema_version": 1,
        "dry_run": dry_run,
        "experiment_id": config.experiment_id,
        "split": "diagnostic",
        "source_bindings": bindings,
        "model": {
            "path": config.model.portable_path,
            "sha256": config.model.sha256,
            "variant": config.model.variant,
            "training_seed": config.training_seed,
        },
        "scenario_seeds": list(config.seeds),
        "route_strata_counts": _strata_counts(config.seeds),
        "initial_obstacle_direction_counts": _obstacle_direction_counts(config.seeds),
        "seed_window_selection": {
            "search_start": 20_614,
            "selected_start": config.scenario_seed_start,
            "rule": "first_contiguous_120_balancing_route_reverse_and_obstacle_direction",
            "uses_pre_treatment_covariates_only": True,
            "outcome_data_used": False,
        },
        "paired_same_scenarios": True,
        "shield_evidence_schema_version": "1.0",
        "shield_evidence_required": True,
        "candidate_count": 1,
        "parameter_scan": "forbidden",
        "adaptive_extension": "forbidden",
        "validation_or_test_access": "forbidden",
        "runs": [
            {
                "arm_id": run.arm.arm_id,
                "role": run.arm.role,
                "run_name": run.arm.run_name,
                "planner_label": run.arm.planner_label,
                "command": list(run.command),
                "environment": dict(run.environment),
            }
            for run in runs
        ],
    }
    if dry_run:
        return 0, plan

    parsed: list[ParsedArmResult] = []
    for run in runs:
        if source_verifier(root) != source_revision:
            raise RuntimeError("Git HEAD or source worktree changed before an audit arm.")
        _verify_source_bindings(bindings, repository_root=root, config=config)
        completed = process_runner(
            run.command,
            cwd=root,
            env=_clean_environment(run.environment),
            check=False,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Webots front-brake arm {run.arm.arm_id} failed with exit code "
                f"{completed.returncode}."
            )
        if source_verifier(root) != source_revision:
            raise RuntimeError("Git HEAD or source worktree changed during an audit arm.")
        _verify_source_bindings(bindings, repository_root=root, config=config)
        parsed.append(_parse_arm_result(run, config=config, repository_root=root))

    if source_verifier(root) != source_revision:
        raise RuntimeError("Git HEAD or source worktree changed during the paired diagnostic.")
    _verify_source_bindings(bindings, repository_root=root, config=config)
    reference = next(result for result in parsed if result.arm.role == "reference")
    candidate = next(result for result in parsed if result.arm.role == "candidate")
    paired = _paired_analysis(reference, candidate, seeds=config.seeds)
    gate = _gate_report(config.gate, paired=paired)
    manipulation = _secondary_manipulation_check(
        reference,
        candidate,
        expected_episodes=config.episodes,
    )
    primary_gate_passed = bool(gate["passed"])
    manipulation_valid = bool(manipulation["passed"])
    passed = primary_gate_passed and manipulation_valid
    if not manipulation_valid:
        status = "invalid_manipulation"
        selected_decision = config.decision["invalid_manipulation"]
        exit_code = EXIT_INVALID_MANIPULATION
    elif primary_gate_passed:
        status = "passed"
        selected_decision = config.decision["pass"]
        exit_code = 0
    else:
        status = "gate_failed"
        selected_decision = config.decision["fail"]
        exit_code = EXIT_CANDIDATE_GATE_FAILED
    report_directory = _REPORT_DIRECTORY.format(experiment_id=config.experiment_id)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": status,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "diagnostic",
        "validation_split_was_run": False,
        "test_split_was_run": False,
        "source_bindings": bindings,
        "model": {
            "path": config.model.portable_path,
            "sha256": config.model.sha256,
            "variant": config.model.variant,
            "training_seed": config.training_seed,
        },
        "scenario": {
            "condition_id": config.condition_id,
            "scenario_seed_start": config.scenario_seed_start,
            "episodes": config.episodes,
            "scenario_seeds": list(config.seeds),
            "route_strata_counts": _strata_counts(config.seeds),
            "initial_obstacle_direction_counts": _obstacle_direction_counts(config.seeds),
            "obstacle_count": 1,
            "obstacle_speed_range": [0.15, 0.30],
            "horizontal_route_half_length": 1.10,
            "vertical_route_half_length": 1.10,
            "deterministic_policy": True,
        },
        "preregistration": {
            "paired_same_scenarios": True,
            "episode_context_schema_version": "1.1",
            "risk_trace_required": True,
            "shield_evidence_schema_version": "1.0",
            "shield_evidence_required": True,
            "seed_window_selection": {
                "search_start": 20_614,
                "rule": ("first_contiguous_120_balancing_route_reverse_and_obstacle_direction"),
                "covariates_only": [
                    "route_axis",
                    "reverse_route",
                    "initial_obstacle_direction",
                ],
                "outcome_data_used": False,
            },
            "candidate_count": 1,
            "parameter_scan": "forbidden",
            "adaptive_extension": "forbidden",
            "validation_or_test": "forbidden",
        },
        "arms": [
            {
                "id": result.arm.arm_id,
                "role": result.arm.role,
                "run_name": result.arm.run_name,
                "planner_label": result.arm.planner_label,
                "model": {
                    "path": config.model.portable_path,
                    "sha256": config.model.sha256,
                    "variant": config.model.variant,
                    "training_seed": config.training_seed,
                },
                "controller_env": dict(result.arm.controller_env),
                "resolved_safety_shield_config": _expected_shield_config(result.arm),
                "shield_evidence_metrics": dict(result.shield_evidence_metrics),
                "execution": {
                    "command": list(run.command),
                    "environment": dict(run.environment),
                    "port": run.port,
                    "exit_code": 0,
                },
                "metrics": dict(result.metrics),
                "artifacts": dict(result.artifacts),
            }
            for result, run in zip(parsed, runs, strict=True)
        ],
        "paired_analysis": paired,
        "manipulation_integrity_check": manipulation,
        "experiment_validity": {
            "manipulation_integrity_required": True,
            "manipulation_integrity_passed": manipulation_valid,
            "primary_efficacy_gate_passed": primary_gate_passed,
            "overall_passed": passed,
        },
        "interpretation_scope": {
            "estimand": "paired_intent_to_treat_whole_closed_loop_profile_effect",
            "candidate_profile": (
                "front_brake_enabled_with_admitted_action_fed_back_into_next_observation"
            ),
            "only_treatment_parameter_difference": ("WEBOTS_SAFETY_SHIELD_enabled_0_vs_1"),
            "step_level_action_transformation_evidence_available": True,
            "outcome_effect_remains_intent_to_treat": True,
            "mechanism_evidence_scope": (
                "shield_activation_action_braking_speed_cap_and_angular_preservation"
            ),
        },
        "candidate_gate": gate,
        "decision": {
            "selected": selected_decision,
            "validation_or_test_access": "forbidden",
        },
        "written_reports": {
            "json": f"{report_directory}/paired_diagnostic_report.json",
            "yaml": f"{report_directory}/paired_diagnostic_report.yaml",
            "sha256_sidecar": f"{report_directory}/paired_diagnostic_report.sha256",
        },
    }
    _verify_parsed_artifacts(parsed, repository_root=root)
    if source_verifier(root) != source_revision:
        raise RuntimeError("Git HEAD or source worktree changed before report sealing.")
    _verify_source_bindings(bindings, repository_root=root, config=config)
    _write_report(
        report,
        output_directory=root / report_directory,
    )
    return exit_code, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/webots_front_brake_paired_diagnostic_v1.yaml"),
    )
    parser.add_argument("--webots", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=13_680)
    parser.add_argument("--timeout-seconds", type=float, default=7_200.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).absolute().parents[1]
    try:
        exit_code, result = execute_diagnostic(
            repository_root=root,
            config_path=args.config,
            webots_executable=args.webots,
            dry_run=args.dry_run,
            port_base=args.port_base,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as error:
        print(
            f"WEBOTS_FRONT_BRAKE_DIAGNOSTIC_FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_PREFLIGHT_OR_RUNTIME_FAILED
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("WEBOTS_FRONT_BRAKE_DIAGNOSTIC_DRY_RUN_OK", file=sys.stderr)
    else:
        labels = {
            0: "WEBOTS_FRONT_BRAKE_DIAGNOSTIC_OK",
            EXIT_CANDIDATE_GATE_FAILED: "WEBOTS_FRONT_BRAKE_DIAGNOSTIC_GATE_FAILED",
            EXIT_INVALID_MANIPULATION: (
                "WEBOTS_FRONT_BRAKE_DIAGNOSTIC_INVALID_MANIPULATION"
            ),
        }
        print(
            labels.get(exit_code, "WEBOTS_FRONT_BRAKE_DIAGNOSTIC_FAILED"),
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
