"""Fail-closed analysis for pre-registered paired Webots safety diagnostics.

The analyzer is deliberately offline.  It verifies the source and result hash
chain, recomputes outcome statistics from the two ``episodes.csv`` files, and
applies the gate declared by the input configuration.  It never selects or
copies a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

from indoor_patrol_rl.core import SafetyShieldConfig
from indoor_patrol_rl.evaluation.checkpoint_selection import sha256_file
from indoor_patrol_rl.evaluation.checkpoint_sweep import (
    WebotsSafetyShieldProfile,
    load_safety_shield_profile,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SAFE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
_FLOAT_REL_TOL = 1e-12
_FLOAT_ABS_TOL = 1e-12
_OUTCOMES = ("success", "collision", "timeout", "error")
_PROFILE_FIELDS = frozenset(SafetyShieldConfig.__dataclass_fields__) - {"lidar_max_range"}
_CSV_FIELDS = (
    "schema_version",
    "episode_id",
    "split",
    "condition_id",
    "scenario_seed",
    "planner",
    "outcome",
    "is_success",
    "collision",
    "timeout",
    "step_count",
    "elapsed_sim_s",
    "completion_time_s",
    "episode_return",
    "path_length_m",
    "reference_path_length_m",
    "path_length_ratio",
    "minimum_clearance_m",
    "near_collision_ratio",
    "action_delta_sq_mean",
    "inference_latency_p50_ms",
    "inference_latency_p95_ms",
    "error_type",
)

_ROOT_FIELDS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "split",
    "condition_id",
    "sweep_feature_revision",
    "training_seed",
    "scenario_seed_start",
    "episodes",
    "model",
    "scenario",
    "runtime_inputs",
    "arms",
    "preregistration",
    "candidate_gate",
    "decision",
}
_MODEL_FIELDS = {"path", "sha256", "variant"}
_SCENARIO_FIELDS = {
    "obstacle_count",
    "obstacle_speed_range",
    "horizontal_route_half_length",
    "vertical_route_half_length",
    "deterministic_policy",
    "static_layout",
    "action_repeat",
    "max_episode_steps",
}
_RUNTIME_FIELDS = {"world", "evaluator"}
_SOURCE_FIELDS = {"path", "sha256"}
_ARM_FIELDS = {
    "id",
    "role",
    "port_base",
    "shield_profile",
    "run_name",
    "sweep_directory",
}
_ARM_PROFILE_FIELDS = {
    "path",
    "source_sha256",
    "semantic_sha256",
    "result_token",
    "enabled",
}
_PREREGISTRATION_FIELDS = {
    "paired_same_scenarios",
    "same_model_and_controller",
    "only_operational_difference",
    "candidate_count",
    "parameter_scan",
    "adaptive_extension",
    "held_out_formal_test_access",
    "seed_window_selection",
}
_SEED_SELECTION_FIELDS = {
    "search_start",
    "selected_start",
    "rule",
    "covariates_only",
    "outcome_data_used",
    "expected_route_strata_counts",
    "expected_initial_obstacle_direction_counts",
}
_ROUTE_COUNT_FIELDS = {
    "horizontal_non_reverse",
    "horizontal_reverse",
    "vertical_non_reverse",
    "vertical_reverse",
}
_DIRECTION_COUNT_FIELDS = {"negative", "positive"}
_GATE_FIELDS = {
    "require_all",
    "minimum_success_rate",
    "maximum_collision_rate",
    "minimum_relative_collision_reduction",
    "exact_one_sided_mcnemar_significance_level",
    "maximum_new_timeout_count",
    "maximum_median_completion_time_ratio",
    "minimum_common_success_count",
    "maximum_error_count_per_arm",
}
_DECISION_FIELDS = {"pass", "fail", "formal_model_promotion", "output_directory"}
_MANIFEST_PROFILE_FIELDS = {
    "profile_id",
    "result_token",
    "semantic_sha256",
    "source_path",
    "source_sha256",
    "enabled",
    "config",
    "lidar_max_range_source",
}
_GIT_BINDING_FIELDS = {
    "schema_version",
    "kind",
    "config_path",
    "preregistration_revision",
    "config_blob_oid",
    "config_sha256",
}


@dataclass(frozen=True)
class PairedSafetyArm:
    """One source-bound arm declared by a paired diagnostic configuration."""

    arm_id: str
    role: str
    port_base: int
    profile: WebotsSafetyShieldProfile
    run_name: str
    sweep_directory: Path


@dataclass(frozen=True)
class PairedSafetyConfig:
    """Validated input contract for a paired diagnostic."""

    repository_root: Path
    source_path: Path
    source_sha256: str
    raw: Mapping[str, Any]
    model_path: Path
    world_path: Path
    evaluator_path: Path
    reference: PairedSafetyArm
    candidate: PairedSafetyArm
    git_revision: str
    git_binding_path: Path
    git_binding_sha256: str
    preregistration_revision: str
    preregistration_blob_oid: str
    preregistration_committed_at: datetime

    @property
    def output_directory(self) -> Path:
        return _repository_path(
            self.repository_root,
            self.raw["decision"]["output_directory"],
            "decision.output_directory",
            require_file=False,
        )

    @property
    def markdown_path(self) -> Path:
        experiment_id = str(self.raw["experiment_id"])
        stem = experiment_id.removesuffix("_paired_diagnostic_v1")
        return self.repository_root / "reports" / stem / "diagnostic_report.md"


@dataclass(frozen=True)
class _Episode:
    scenario_seed: int
    outcome: str
    completion_time_s: float | None
    reference_path_length_m: float


@dataclass(frozen=True)
class _LoadedArm:
    specification: PairedSafetyArm
    manifest_path: Path
    manifest_sha256: str
    summary_path: Path
    summary_sha256: str
    episodes_path: Path
    episodes_sha256: str
    summary: Mapping[str, Any]
    episodes: Mapping[int, _Episode]
    planner: str
    controller_planner: str
    shield_config: Mapping[str, float]
    counts: Mapping[str, int]
    started_at: datetime


@dataclass(frozen=True)
class _GitBlobBinding:
    binding_path: Path
    binding_sha256: str
    preregistration_revision: str
    config_blob_oid: str
    config_sha256: str
    committed_at: datetime


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _exact_fields(value: Any, expected: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    actual = set(mapping)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise ValueError(
            f"{label} fields do not match the schema; missing={missing}, unexpected={unexpected}."
        )
    return mapping


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    if strictly_positive and number <= 0.0:
        raise ValueError(f"{label} must be positive.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be at most {maximum}.")
    return number


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean.")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _same_number(left: Any, right: Any) -> bool:
    try:
        if isinstance(left, bool) or isinstance(right, bool):
            return False
        return math.isclose(
            float(left),
            float(right),
            rel_tol=_FLOAT_REL_TOL,
            abs_tol=_FLOAT_ABS_TOL,
        )
    except (TypeError, ValueError):
        return False


def _portable(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _repository_path(
    repository_root: Path,
    value: Any,
    label: str,
    *,
    require_file: bool,
) -> Path:
    raw = _string(value, label)
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or raw != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{label} must be a normalized repository-relative path.")
    resolved = (repository_root / pure).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository.") from error
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"{label} is not valid YAML.") from error
    return _mapping(raw, label)


def _reject_duplicate_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON.") from error
    return _mapping(raw, label)


def _verify_file_sha256(path: Path, expected: Any, label: str) -> str:
    digest = _sha256(expected, f"{label}.sha256")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 mismatch: expected {digest}, got {actual}.")
    return actual


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"Git verification failed: {detail}")
    return completed.stdout.strip()


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git verification failed: {detail}")
    return completed.stdout


def _parse_timestamp(value: Any, label: str) -> datetime:
    raw = _string(value, label)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset.")
    return parsed


def _git_blob_oid(repository_root: Path, revision: str, relative_path: str, label: str) -> str:
    output = _git_output(repository_root, "ls-tree", revision, "--", relative_path)
    lines = output.splitlines()
    if len(lines) != 1 or "\t" not in lines[0]:
        raise ValueError(f"{label} must resolve to exactly one committed Git blob.")
    metadata, actual_path = lines[0].split("\t", 1)
    parts = metadata.split()
    if (
        len(parts) != 3
        or parts[0] != "100644"
        or parts[1] != "blob"
        or _GIT_REVISION_PATTERN.fullmatch(parts[2]) is None
        or actual_path != relative_path
    ):
        raise ValueError(f"{label} must be a regular 100644 Git blob at the declared path.")
    return parts[2]


def _git_index_blob_oid(repository_root: Path, relative_path: str, label: str) -> str:
    output = _git_output(repository_root, "ls-files", "--stage", "--", relative_path)
    lines = output.splitlines()
    if len(lines) != 1 or "\t" not in lines[0]:
        raise ValueError(f"{label} must have exactly one tracked index entry.")
    metadata, actual_path = lines[0].split("\t", 1)
    parts = metadata.split()
    if (
        len(parts) != 3
        or parts[0] != "100644"
        or _GIT_REVISION_PATTERN.fullmatch(parts[1]) is None
        or parts[2] != "0"
        or actual_path != relative_path
    ):
        raise ValueError(f"{label} index entry must be one regular stage-zero blob.")
    return parts[1]


def _require_git_ancestor(
    repository_root: Path,
    ancestor: str,
    descendant: str,
    label: str,
) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"{label} is not an ancestor of the required revision.")


def _verify_clean_head_file(repository_root: Path, path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular repository file, not a symlink.")
    relative_path = _portable(path, repository_root)
    status = _git_output(
        repository_root,
        "status",
        "--porcelain=v2",
        "--untracked-files=all",
        "--",
        relative_path,
    )
    if status:
        raise ValueError(f"{label} must be tracked and clean in both the index and worktree.")
    head_oid = _git_blob_oid(repository_root, "HEAD", relative_path, label)
    if _git_index_blob_oid(repository_root, relative_path, label) != head_oid:
        raise ValueError(f"{label} index blob does not match HEAD.")
    if _git_bytes(repository_root, "cat-file", "blob", head_oid) != path.read_bytes():
        raise ValueError(f"{label} working bytes do not match the committed HEAD blob.")
    return head_oid


def _verify_git_revision(repository_root: Path, feature_revision: Any) -> str:
    feature = _string(feature_revision, "sweep_feature_revision")
    if _GIT_REVISION_PATTERN.fullmatch(feature) is None:
        raise ValueError("sweep_feature_revision must be a full lowercase Git revision.")
    current = _git_output(repository_root, "rev-parse", "HEAD")
    if _GIT_REVISION_PATTERN.fullmatch(current) is None:
        raise ValueError("git rev-parse HEAD did not return a full revision.")
    _git_output(repository_root, "cat-file", "-e", f"{feature}^{{commit}}")
    _require_git_ancestor(
        repository_root,
        feature,
        current,
        "sweep_feature_revision",
    )
    return current


def _verify_preregistration_git_binding(
    repository_root: Path,
    config_path: Path,
    *,
    feature_revision: str,
    current_revision: str,
) -> _GitBlobBinding:
    binding_path = config_path.with_suffix(".binding.yaml")
    _verify_clean_head_file(repository_root, binding_path, "preregistration Git binding")
    binding = _exact_fields(
        _load_yaml(binding_path, "preregistration Git binding"),
        _GIT_BINDING_FIELDS,
        "preregistration Git binding",
    )
    if _integer(binding["schema_version"], "Git binding schema_version") != 1:
        raise ValueError("Git binding schema_version must be 1.")
    if binding["kind"] != "paired_safety_preregistration_git_binding":
        raise ValueError("Git binding kind is not supported.")
    relative_config = _portable(config_path, repository_root)
    if binding["config_path"] != relative_config:
        raise ValueError("Git binding config_path does not match the analyzed configuration.")
    revision = _string(binding["preregistration_revision"], "preregistration_revision")
    blob_oid = _string(binding["config_blob_oid"], "config_blob_oid")
    config_sha256 = _sha256(binding["config_sha256"], "config_sha256")
    if _GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("preregistration_revision must be a full lowercase Git revision.")
    if _GIT_REVISION_PATTERN.fullmatch(blob_oid) is None:
        raise ValueError("config_blob_oid must be a full lowercase Git object ID.")
    _git_output(repository_root, "cat-file", "-e", f"{revision}^{{commit}}")
    _require_git_ancestor(
        repository_root,
        feature_revision,
        revision,
        "sweep_feature_revision",
    )
    _require_git_ancestor(
        repository_root,
        revision,
        current_revision,
        "preregistration_revision",
    )
    preregistered_oid = _git_blob_oid(
        repository_root,
        revision,
        relative_config,
        "pre-registered config",
    )
    if preregistered_oid != blob_oid:
        raise ValueError("Git binding config_blob_oid does not match the pre-registration commit.")
    if _git_blob_oid(repository_root, "HEAD", relative_config, "HEAD config") != blob_oid:
        raise ValueError("HEAD config blob does not match the pre-registered blob.")
    if _git_index_blob_oid(repository_root, relative_config, "config") != blob_oid:
        raise ValueError("Config index blob does not match the pre-registered blob.")
    status = _git_output(
        repository_root,
        "status",
        "--porcelain=v2",
        "--untracked-files=all",
        "--",
        relative_config,
    )
    if status:
        raise ValueError("Pre-registered config must be clean in both the index and worktree.")
    committed_bytes = _git_bytes(repository_root, "cat-file", "blob", blob_oid)
    if committed_bytes != config_path.read_bytes():
        raise ValueError("Config working bytes do not match the pre-registered Git blob.")
    actual_sha256 = hashlib.sha256(committed_bytes).hexdigest()
    if actual_sha256 != config_sha256:
        raise ValueError("Git binding config_sha256 does not match the pre-registered blob.")
    committed_at = _parse_timestamp(
        _git_output(repository_root, "show", "-s", "--format=%cI", revision),
        "pre-registration commit time",
    )
    return _GitBlobBinding(
        binding_path=binding_path,
        binding_sha256=sha256_file(binding_path),
        preregistration_revision=revision,
        config_blob_oid=blob_oid,
        config_sha256=config_sha256,
        committed_at=committed_at,
    )


@cache
def _seed_selection_covariates(
    seed: int,
    obstacle_speed_min: float,
    obstacle_speed_max: float,
) -> tuple[str, bool, str]:
    """Replay the outcome-independent random draws used by a one-obstacle reset."""

    random_generator = np.random.default_rng(seed)
    route_axis = str(random_generator.choice(["horizontal", "vertical"]))
    offset_limit = 0.55 if route_axis == "horizontal" else 0.70
    random_generator.uniform(-offset_limit, offset_limit)
    random_generator.uniform(-offset_limit, offset_limit)
    reverse_route = bool(random_generator.integers(0, 2))
    random_generator.uniform(-math.pi, math.pi)
    if route_axis == "horizontal":
        random_generator.uniform(-0.55, 0.55)
        random_generator.uniform(-1.05, 1.05)
    else:
        random_generator.uniform(-1.55, 1.55)
        random_generator.uniform(-0.45, 0.45)
    random_generator.uniform(obstacle_speed_min, obstacle_speed_max)
    direction = float(random_generator.choice([-1.0, 1.0]))
    direction_key = "negative" if direction < 0.0 else "positive"
    return route_axis, reverse_route, direction_key


def _first_balanced_seed_window(
    *,
    search_start: int,
    selected_start: int,
    episodes: int,
    obstacle_speed_min: float,
    obstacle_speed_max: float,
    route_target: Mapping[str, int],
    direction_target: Mapping[str, int],
) -> int | None:
    route_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    window: deque[tuple[str, str]] = deque()
    final_seed = selected_start + episodes - 1
    for seed in range(search_start, final_seed + 1):
        route_axis, reverse_route, direction = _seed_selection_covariates(
            seed,
            obstacle_speed_min,
            obstacle_speed_max,
        )
        route = f"{route_axis}_{'reverse' if reverse_route else 'non_reverse'}"
        window.append((route, direction))
        route_counts[route] += 1
        direction_counts[direction] += 1
        if len(window) > episodes:
            removed_route, removed_direction = window.popleft()
            route_counts[removed_route] -= 1
            direction_counts[removed_direction] -= 1
        if len(window) != episodes:
            continue
        window_start = seed - episodes + 1
        if all(route_counts[name] == count for name, count in route_target.items()) and all(
            direction_counts[name] == count for name, count in direction_target.items()
        ):
            return window_start
    return None


def _validate_preregistration(raw: Mapping[str, Any], episodes: int, seed_start: int) -> None:
    prereg = _exact_fields(raw["preregistration"], _PREREGISTRATION_FIELDS, "preregistration")
    required_literals = {
        "paired_same_scenarios": True,
        "same_model_and_controller": True,
        "only_operational_difference": "safety_shield_enabled",
        "candidate_count": 1,
        "parameter_scan": "forbidden",
        "adaptive_extension": "forbidden",
        "held_out_formal_test_access": "forbidden",
    }
    for name, expected in required_literals.items():
        if prereg[name] != expected or type(prereg[name]) is not type(expected):
            raise ValueError(f"preregistration.{name} must equal {expected!r}.")
    selection = _exact_fields(
        prereg["seed_window_selection"],
        _SEED_SELECTION_FIELDS,
        "preregistration.seed_window_selection",
    )
    search_start = _integer(
        selection["search_start"], "seed_window_selection.search_start", minimum=0
    )
    if (
        _integer(selection["selected_start"], "seed_window_selection.selected_start", minimum=0)
        != seed_start
    ):
        raise ValueError("seed_window_selection.selected_start must match scenario_seed_start.")
    if search_start > seed_start:
        raise ValueError("seed_window_selection.search_start cannot follow selected_start.")
    if selection["rule"] != "first_contiguous_120_balancing_route_reverse_and_obstacle_direction":
        raise ValueError("seed_window_selection.rule is not the locked selection rule.")
    if selection["covariates_only"] != [
        "route_axis",
        "reverse_route",
        "initial_obstacle_direction",
    ]:
        raise ValueError(
            "seed_window_selection.covariates_only does not match the locked covariates."
        )
    if _boolean(selection["outcome_data_used"], "seed_window_selection.outcome_data_used"):
        raise ValueError("seed_window_selection.outcome_data_used must be false.")
    if episodes != 120:
        raise ValueError("The locked first_contiguous_120 seed rule requires exactly 120 episodes.")
    if raw["scenario"]["obstacle_count"] != 1:
        raise ValueError("The locked seed-selection replay requires exactly one obstacle.")
    route_counts = _exact_fields(
        selection["expected_route_strata_counts"],
        _ROUTE_COUNT_FIELDS,
        "seed_window_selection.expected_route_strata_counts",
    )
    direction_counts = _exact_fields(
        selection["expected_initial_obstacle_direction_counts"],
        _DIRECTION_COUNT_FIELDS,
        "seed_window_selection.expected_initial_obstacle_direction_counts",
    )
    declared_route_counts = {
        name: _integer(value, f"route count {name}", minimum=0)
        for name, value in route_counts.items()
    }
    declared_direction_counts = {
        name: _integer(value, f"direction count {name}", minimum=0)
        for name, value in direction_counts.items()
    }
    expected_route_counts = {name: episodes // 4 for name in _ROUTE_COUNT_FIELDS}
    expected_direction_counts = {name: episodes // 2 for name in _DIRECTION_COUNT_FIELDS}
    if declared_route_counts != expected_route_counts:
        raise ValueError("Expected route-stratum counts must declare exact four-way balance.")
    if declared_direction_counts != expected_direction_counts:
        raise ValueError("Expected obstacle-direction counts must declare exact two-way balance.")
    speed_min, speed_max = raw["scenario"]["obstacle_speed_range"]
    first_start = _first_balanced_seed_window(
        search_start=search_start,
        selected_start=seed_start,
        episodes=episodes,
        obstacle_speed_min=float(speed_min),
        obstacle_speed_max=float(speed_max),
        route_target=expected_route_counts,
        direction_target=expected_direction_counts,
    )
    if first_start != seed_start:
        raise ValueError(
            "Selected seed window is not the first covariate-balanced contiguous 120 window "
            f"from search_start (computed {first_start!r}, declared {seed_start})."
        )


def _validate_gate(raw: Mapping[str, Any], episodes: int) -> None:
    gate = _exact_fields(raw["candidate_gate"], _GATE_FIELDS, "candidate_gate")
    if not _boolean(gate["require_all"], "candidate_gate.require_all"):
        raise ValueError("candidate_gate.require_all must be true.")
    for name in (
        "minimum_success_rate",
        "maximum_collision_rate",
        "minimum_relative_collision_reduction",
        "exact_one_sided_mcnemar_significance_level",
    ):
        _number(gate[name], f"candidate_gate.{name}", minimum=0.0, maximum=1.0)
    _integer(
        gate["maximum_new_timeout_count"], "candidate_gate.maximum_new_timeout_count", minimum=0
    )
    _number(
        gate["maximum_median_completion_time_ratio"],
        "candidate_gate.maximum_median_completion_time_ratio",
        strictly_positive=True,
    )
    common = _integer(
        gate["minimum_common_success_count"],
        "candidate_gate.minimum_common_success_count",
        minimum=1,
    )
    if common > episodes:
        raise ValueError("candidate_gate.minimum_common_success_count cannot exceed episodes.")
    _integer(
        gate["maximum_error_count_per_arm"], "candidate_gate.maximum_error_count_per_arm", minimum=0
    )


def _load_arm(
    value: Any,
    *,
    repository_root: Path,
    index: int,
) -> PairedSafetyArm:
    label = f"arms[{index}]"
    raw = _exact_fields(value, _ARM_FIELDS, label)
    arm_id = _string(raw["id"], f"{label}.id")
    if _SAFE_NAME_PATTERN.fullmatch(arm_id) is None:
        raise ValueError(f"{label}.id must be a lowercase path-safe name.")
    role = _string(raw["role"], f"{label}.role")
    if role not in {"reference", "candidate"}:
        raise ValueError(f"{label}.role must be reference or candidate.")
    port_base = _integer(raw["port_base"], f"{label}.port_base", minimum=1024)
    if port_base > 65535:
        raise ValueError(f"{label}.port_base must be at most 65535.")
    profile_spec = _exact_fields(
        raw["shield_profile"],
        _ARM_PROFILE_FIELDS,
        f"{label}.shield_profile",
    )
    profile_path = _repository_path(
        repository_root,
        profile_spec["path"],
        f"{label}.shield_profile.path",
        require_file=True,
    )
    profile_raw = _load_yaml(profile_path, f"{label} shield profile")
    _exact_fields(
        profile_raw,
        {"schema_version", "profile_id", "enabled", "config"},
        f"{label} shield profile",
    )
    _exact_fields(profile_raw["config"], _PROFILE_FIELDS, f"{label} shield profile.config")
    source_sha256 = _verify_file_sha256(
        profile_path,
        profile_spec["source_sha256"],
        f"{label}.shield_profile",
    )
    profile = load_safety_shield_profile(profile_path, repository_root=repository_root)
    if profile.source_sha256 != source_sha256:
        raise ValueError(f"{label} shield profile source binding is inconsistent.")
    if profile.semantic_sha256 != _sha256(
        profile_spec["semantic_sha256"], f"{label}.shield_profile.semantic_sha256"
    ):
        raise ValueError(f"{label} shield profile semantic SHA-256 mismatch.")
    if profile.result_token != profile_spec["result_token"]:
        raise ValueError(f"{label} shield profile result_token mismatch.")
    if profile.enabled is not _boolean(profile_spec["enabled"], f"{label}.shield_profile.enabled"):
        raise ValueError(f"{label} shield profile enabled value mismatch.")
    run_name = _string(raw["run_name"], f"{label}.run_name")
    if _SAFE_NAME_PATTERN.fullmatch(run_name) is None:
        raise ValueError(f"{label}.run_name must be a lowercase path-safe name.")
    if not run_name.endswith(f"_shield-{profile.result_token}"):
        raise ValueError(f"{label}.run_name is not bound to the shield result token.")
    sweep_directory = _repository_path(
        repository_root,
        raw["sweep_directory"],
        f"{label}.sweep_directory",
        require_file=False,
    )
    return PairedSafetyArm(
        arm_id=arm_id,
        role=role,
        port_base=port_base,
        profile=profile,
        run_name=run_name,
        sweep_directory=sweep_directory,
    )


def load_paired_safety_config(
    path: Path,
    *,
    repository_root: Path,
) -> PairedSafetyConfig:
    """Load and fully validate one paired diagnostic preregistration."""

    root = Path(repository_root).resolve()
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()
    try:
        source_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "Paired diagnostic configuration must stay inside the repository."
        ) from error
    if not source_path.is_file():
        raise FileNotFoundError(f"Paired diagnostic configuration does not exist: {source_path}")
    raw = _exact_fields(
        _load_yaml(source_path, "paired diagnostic configuration"),
        _ROOT_FIELDS,
        "paired diagnostic configuration",
    )
    if _integer(raw["schema_version"], "schema_version") != 1:
        raise ValueError("schema_version must be 1.")
    experiment_id = _string(raw["experiment_id"], "experiment_id")
    if _SAFE_NAME_PATTERN.fullmatch(experiment_id) is None:
        raise ValueError("experiment_id must be a lowercase path-safe name.")
    if raw["purpose"] != "exploratory_safety_profile_diagnostic":
        raise ValueError("purpose must be exploratory_safety_profile_diagnostic.")
    if raw["split"] != "validation":
        raise ValueError("Paired safety diagnostics may only use the validation split.")
    _string(raw["condition_id"], "condition_id")
    _integer(raw["training_seed"], "training_seed", minimum=0)
    seed_start = _integer(raw["scenario_seed_start"], "scenario_seed_start", minimum=0)
    episodes = _integer(raw["episodes"], "episodes", minimum=1)

    model = _exact_fields(raw["model"], _MODEL_FIELDS, "model")
    model_path = _repository_path(root, model["path"], "model.path", require_file=True)
    _verify_file_sha256(model_path, model["sha256"], "model")
    _string(model["variant"], "model.variant")

    scenario = _exact_fields(raw["scenario"], _SCENARIO_FIELDS, "scenario")
    _integer(scenario["obstacle_count"], "scenario.obstacle_count", minimum=0)
    speed_range = _sequence(scenario["obstacle_speed_range"], "scenario.obstacle_speed_range")
    if len(speed_range) != 2:
        raise ValueError("scenario.obstacle_speed_range must contain exactly two values.")
    speed_min = _number(speed_range[0], "scenario.obstacle_speed_range[0]", minimum=0.0)
    speed_max = _number(speed_range[1], "scenario.obstacle_speed_range[1]", minimum=0.0)
    if speed_min > speed_max:
        raise ValueError("scenario obstacle speed minimum cannot exceed its maximum.")
    _number(
        scenario["horizontal_route_half_length"],
        "scenario.horizontal_route_half_length",
        strictly_positive=True,
    )
    _number(
        scenario["vertical_route_half_length"],
        "scenario.vertical_route_half_length",
        strictly_positive=True,
    )
    if not _boolean(scenario["deterministic_policy"], "scenario.deterministic_policy"):
        raise ValueError("scenario.deterministic_policy must be true.")
    if scenario["static_layout"] != "train_a":
        raise ValueError("scenario.static_layout must be train_a for validation diagnostics.")
    _integer(scenario["action_repeat"], "scenario.action_repeat", minimum=1)
    _integer(scenario["max_episode_steps"], "scenario.max_episode_steps", minimum=1)

    runtime = _exact_fields(raw["runtime_inputs"], _RUNTIME_FIELDS, "runtime_inputs")
    world = _exact_fields(runtime["world"], _SOURCE_FIELDS, "runtime_inputs.world")
    evaluator = _exact_fields(runtime["evaluator"], _SOURCE_FIELDS, "runtime_inputs.evaluator")
    world_path = _repository_path(
        root, world["path"], "runtime_inputs.world.path", require_file=True
    )
    evaluator_path = _repository_path(
        root,
        evaluator["path"],
        "runtime_inputs.evaluator.path",
        require_file=True,
    )
    _verify_file_sha256(world_path, world["sha256"], "runtime_inputs.world")
    _verify_file_sha256(evaluator_path, evaluator["sha256"], "runtime_inputs.evaluator")

    arms_raw = _sequence(raw["arms"], "arms")
    if len(arms_raw) != 2:
        raise ValueError("arms must contain exactly one reference and one candidate.")
    arms = tuple(
        _load_arm(value, repository_root=root, index=index) for index, value in enumerate(arms_raw)
    )
    by_role = {arm.role: arm for arm in arms}
    if set(by_role) != {"reference", "candidate"} or len({arm.arm_id for arm in arms}) != 2:
        raise ValueError("arms must contain uniquely named reference and candidate entries.")
    reference = by_role["reference"]
    candidate = by_role["candidate"]
    if reference.port_base == candidate.port_base:
        raise ValueError("Reference and candidate port_base values must differ.")
    if reference.profile.enabled is not False or candidate.profile.enabled is not True:
        raise ValueError("Reference shield must be disabled and candidate shield must be enabled.")
    if reference.profile.profile_id != candidate.profile.profile_id:
        raise ValueError("Paired profiles must share one profile_id.")
    if reference.profile.config_values != candidate.profile.config_values:
        raise ValueError("Paired profile configurations must be identical apart from enabled.")

    _validate_preregistration(raw, episodes, seed_start)
    _validate_gate(raw, episodes)
    decision = _exact_fields(raw["decision"], _DECISION_FIELDS, "decision")
    if decision["formal_model_promotion"] != "forbidden":
        raise ValueError("decision.formal_model_promotion must be forbidden.")
    if _string(decision["pass"], "decision.pass") == _string(decision["fail"], "decision.fail"):
        raise ValueError("decision.pass and decision.fail must differ.")
    _repository_path(
        root, decision["output_directory"], "decision.output_directory", require_file=False
    )

    revision = _verify_git_revision(root, raw["sweep_feature_revision"])
    git_binding = _verify_preregistration_git_binding(
        root,
        source_path,
        feature_revision=raw["sweep_feature_revision"],
        current_revision=revision,
    )
    return PairedSafetyConfig(
        repository_root=root,
        source_path=source_path,
        source_sha256=sha256_file(source_path),
        raw=raw,
        model_path=model_path,
        world_path=world_path,
        evaluator_path=evaluator_path,
        reference=reference,
        candidate=candidate,
        git_revision=revision,
        git_binding_path=git_binding.binding_path,
        git_binding_sha256=git_binding.binding_sha256,
        preregistration_revision=git_binding.preregistration_revision,
        preregistration_blob_oid=git_binding.config_blob_oid,
        preregistration_committed_at=git_binding.committed_at,
    )


def _parse_bool(value: Any, label: str) -> bool:
    if not isinstance(value, str) or value.casefold() not in {"true", "false"}:
        raise ValueError(f"{label} must be True or False.")
    return value.casefold() == "true"


def _parse_csv_integer(value: Any, label: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer.") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(number)


def _parse_csv_number(value: Any, label: str, *, strictly_positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(number) or (strictly_positive and number <= 0.0):
        raise ValueError(f"{label} must be a finite positive number.")
    return number


def _load_episodes(
    path: Path,
    *,
    role: str,
    split: str,
    condition_id: str,
    expected_seeds: tuple[int, ...],
) -> tuple[dict[int, _Episode], str]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or tuple(reader.fieldnames) != _CSV_FIELDS:
            raise ValueError(f"{role} episodes.csv columns do not match the expected schema.")
        episodes: dict[int, _Episode] = {}
        planner: str | None = None
        for row_number, row in enumerate(reader, start=2):
            label = f"{role} episodes.csv row {row_number}"
            if _parse_csv_number(row["schema_version"], f"{label} schema_version") != 1.0:
                raise ValueError(f"{label} has unsupported schema_version.")
            _string(row["episode_id"], f"{label} episode_id")
            if row["split"] != split or row["condition_id"] != condition_id:
                raise ValueError(f"{label} split or condition_id does not match the configuration.")
            row_planner = _string(row["planner"], f"{label} planner")
            if planner is None:
                planner = row_planner
            elif row_planner != planner:
                raise ValueError(f"{role} episodes.csv must contain exactly one planner.")
            seed = _parse_csv_integer(row["scenario_seed"], f"{label} scenario_seed")
            if seed in episodes:
                raise ValueError(f"{role} episodes.csv repeats scenario_seed {seed}.")
            outcome = row["outcome"]
            if outcome not in _OUTCOMES:
                raise ValueError(f"{label} has invalid outcome {outcome!r}.")
            flags = {
                "success": _parse_bool(row["is_success"], f"{label} is_success"),
                "collision": _parse_bool(row["collision"], f"{label} collision"),
                "timeout": _parse_bool(row["timeout"], f"{label} timeout"),
            }
            expected_flags = {
                "success": outcome == "success",
                "collision": outcome == "collision",
                "timeout": outcome == "timeout",
            }
            if flags != expected_flags:
                raise ValueError(f"{label} outcome and boolean flags disagree.")
            error_type = row["error_type"]
            if (outcome == "error") != bool(error_type):
                raise ValueError(f"{label} error_type does not agree with outcome.")
            completion: float | None = None
            if outcome == "success":
                completion = _parse_csv_number(
                    row["completion_time_s"],
                    f"{label} completion_time_s",
                    strictly_positive=True,
                )
            elif row["completion_time_s"] != "":
                raise ValueError(f"{label} non-success completion_time_s must be empty.")
            reference_length = _parse_csv_number(
                row["reference_path_length_m"],
                f"{label} reference_path_length_m",
                strictly_positive=True,
            )
            episodes[seed] = _Episode(
                scenario_seed=seed,
                outcome=outcome,
                completion_time_s=completion,
                reference_path_length_m=reference_length,
            )
    if tuple(sorted(episodes)) != expected_seeds:
        raise ValueError(
            f"{role} episodes.csv does not contain the exact preregistered seed window."
        )
    if planner is None:
        raise ValueError(f"{role} episodes.csv is empty.")
    return episodes, planner


def _recomputed_counts(episodes: Mapping[int, _Episode]) -> dict[str, int]:
    counts = {outcome: 0 for outcome in _OUTCOMES}
    for episode in episodes.values():
        counts[episode.outcome] += 1
    return {"episodes": len(episodes), **{f"{name}_count": count for name, count in counts.items()}}


def _validate_aggregate(
    aggregate: Any,
    *,
    label: str,
    planner: str,
    condition_id: str | None,
    counts: Mapping[str, int],
    episodes: Mapping[int, _Episode],
) -> None:
    raw = _mapping(aggregate, label)
    if raw.get("planner") != planner:
        raise ValueError(f"{label}.planner does not match episodes.csv.")
    if condition_id is not None and raw.get("condition_id") != condition_id:
        raise ValueError(f"{label}.condition_id does not match episodes.csv.")
    for name in ("episodes", "success_count", "collision_count", "timeout_count", "error_count"):
        if raw.get(name) != counts[name] or type(raw.get(name)) is not int:
            raise ValueError(f"{label}.{name} does not match episodes.csv.")
    total = counts["episodes"]
    expected_rates = {
        "success_rate": counts["success_count"] / total,
        "collision_rate": counts["collision_count"] / total,
        "timeout_rate": counts["timeout_count"] / total,
    }
    for name, expected in expected_rates.items():
        if not _same_number(raw.get(name), expected):
            raise ValueError(f"{label}.{name} does not match episodes.csv.")
    completion_values = [
        episode.completion_time_s
        for episode in episodes.values()
        if episode.completion_time_s is not None
    ]
    if not completion_values:
        raise ValueError(f"{label} has no successful completion times.")
    if not _same_number(raw.get("completion_time_s_mean"), statistics.fmean(completion_values)):
        raise ValueError(f"{label}.completion_time_s_mean does not match episodes.csv.")


def _validate_manifest_profile(
    value: Any,
    *,
    arm: PairedSafetyArm,
    repository_root: Path,
) -> None:
    label = f"{arm.role} sweep request safety_shield_profile"
    raw = _exact_fields(value, _MANIFEST_PROFILE_FIELDS, label)
    expected = arm.profile.manifest_record(repository_root)
    if set(raw["config"]) != _PROFILE_FIELDS:
        raise ValueError(f"{label}.config fields do not match the profile.")
    for name in _PROFILE_FIELDS:
        if not _same_number(raw["config"].get(name), expected["config"][name]):
            raise ValueError(f"{label}.config.{name} does not match the profile.")
    for name in _MANIFEST_PROFILE_FIELDS - {"config"}:
        if raw[name] != expected[name] or type(raw[name]) is not type(expected[name]):
            raise ValueError(f"{label}.{name} does not match the profile.")


def _validate_sweep_request(
    request: Any,
    *,
    config: PairedSafetyConfig,
    arm: PairedSafetyArm,
) -> None:
    raw = _mapping(request, f"{arm.role} sweep request")
    expected_exact = {
        "condition": config.raw["condition_id"],
        "episodes": config.raw["episodes"],
        "seed": config.raw["training_seed"],
        "seed_start": config.raw["scenario_seed_start"],
        "scenario_seeds": list(
            range(
                config.raw["scenario_seed_start"],
                config.raw["scenario_seed_start"] + config.raw["episodes"],
            )
        ),
        "model_variant": config.raw["model"]["variant"],
        "obstacle_count": config.raw["scenario"]["obstacle_count"],
        "static_layout": config.raw["scenario"]["static_layout"],
        "action_repeat": config.raw["scenario"]["action_repeat"],
        "max_episode_steps": config.raw["scenario"]["max_episode_steps"],
        "safety_shield_enabled": arm.profile.enabled,
    }
    for name, expected in expected_exact.items():
        if raw.get(name) != expected or type(raw.get(name)) is not type(expected):
            raise ValueError(f"{arm.role} sweep request.{name} does not match the configuration.")
    expected_numbers = {
        "horizontal_route_half_length": config.raw["scenario"]["horizontal_route_half_length"],
        "vertical_route_half_length": config.raw["scenario"]["vertical_route_half_length"],
    }
    for name, expected in expected_numbers.items():
        if not _same_number(raw.get(name), expected):
            raise ValueError(f"{arm.role} sweep request.{name} does not match the configuration.")
    speed_range = raw.get("obstacle_speed_range")
    expected_speeds = config.raw["scenario"]["obstacle_speed_range"]
    if (
        not isinstance(speed_range, list)
        or len(speed_range) != 2
        or any(
            not _same_number(actual, expected)
            for actual, expected in zip(speed_range, expected_speeds, strict=True)
        )
    ):
        raise ValueError(f"{arm.role} sweep obstacle_speed_range does not match the configuration.")
    _validate_manifest_profile(
        raw.get("safety_shield_profile"), arm=arm, repository_root=config.repository_root
    )


def _load_result_arm(config: PairedSafetyConfig, arm: PairedSafetyArm) -> _LoadedArm:
    manifest_path = arm.sweep_directory / "checkpoint_sweep.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{arm.role} checkpoint_sweep.json does not exist: {manifest_path}")
    manifest = _load_json(manifest_path, f"{arm.role} sweep manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "webots_checkpoint_validation_sweep"
    ):
        raise ValueError(f"{arm.role} sweep manifest has unsupported identity.")
    if manifest.get("split") != "validation" or manifest.get("status") != "completed":
        raise ValueError(f"{arm.role} sweep must be a completed validation run.")
    _validate_sweep_request(manifest.get("request"), config=config, arm=arm)
    runtime = _mapping(manifest.get("runtime"), f"{arm.role} sweep runtime")
    expected_runtime = {
        "evaluation_world": config.raw["runtime_inputs"]["world"]["path"],
        "evaluation_world_sha256": config.raw["runtime_inputs"]["world"]["sha256"],
        "evaluator_controller": config.raw["runtime_inputs"]["evaluator"]["path"],
        "evaluator_controller_sha256": config.raw["runtime_inputs"]["evaluator"]["sha256"],
    }
    for name, expected in expected_runtime.items():
        if runtime.get(name) != expected:
            raise ValueError(f"{arm.role} sweep runtime.{name} does not match the configuration.")
    runs = _sequence(manifest.get("runs"), f"{arm.role} sweep runs")
    matches = [
        _mapping(run, f"{arm.role} sweep run")
        for run in runs
        if isinstance(run, dict) and run.get("run_name") == arm.run_name
    ]
    if len(matches) != 1:
        raise ValueError(f"{arm.role} sweep must contain exactly one configured run_name.")
    run = matches[0]
    if run.get("status") != "completed" or run.get("exit_code") != 0:
        raise ValueError(f"{arm.role} configured sweep run is not completed successfully.")
    started_at = _parse_timestamp(run.get("started_at"), f"{arm.role} sweep run started_at")
    finished_at = _parse_timestamp(run.get("finished_at"), f"{arm.role} sweep run finished_at")
    if started_at <= config.preregistration_committed_at:
        raise ValueError(
            f"{arm.role} sweep started before the bound pre-registration commit time."
        )
    if finished_at < started_at:
        raise ValueError(f"{arm.role} sweep finished_at precedes started_at.")
    model = config.raw["model"]
    if run.get("model_path") != model["path"] or run.get("model_sha256") != model["sha256"]:
        raise ValueError(f"{arm.role} sweep run model binding does not match the configuration.")
    summary_path = config.repository_root / "artifacts" / "results" / arm.run_name / "summary.json"
    if run.get("summary_path") != _portable(summary_path, config.repository_root):
        raise ValueError(f"{arm.role} sweep run summary_path does not match run_name.")
    if not summary_path.is_file():
        raise FileNotFoundError(f"{arm.role} summary.json does not exist: {summary_path}")
    summary_sha256 = _verify_file_sha256(
        summary_path,
        run.get("summary_sha256"),
        f"{arm.role} summary.json",
    )
    summary = _load_json(summary_path, f"{arm.role} summary.json")
    provenance = _mapping(summary.get("evaluation_provenance"), f"{arm.role} summary provenance")
    episodes_path = summary_path.with_name("episodes.csv")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"{arm.role} episodes.csv does not exist: {episodes_path}")
    episodes_sha256 = _verify_file_sha256(
        episodes_path,
        provenance.get("episodes_csv_sha256"),
        f"{arm.role} episodes.csv",
    )
    seed_start = config.raw["scenario_seed_start"]
    expected_seeds = tuple(range(seed_start, seed_start + config.raw["episodes"]))
    episodes, planner = _load_episodes(
        episodes_path,
        role=arm.role,
        split=config.raw["split"],
        condition_id=config.raw["condition_id"],
        expected_seeds=expected_seeds,
    )
    counts = _recomputed_counts(episodes)
    if (
        summary.get("split") != config.raw["split"]
        or summary.get("episode_count") != counts["episodes"]
    ):
        raise ValueError(f"{arm.role} summary split or episode_count does not match episodes.csv.")
    if summary.get("planners") != [planner] or summary.get("conditions") != [
        config.raw["condition_id"]
    ]:
        raise ValueError(
            f"{arm.role} summary planner or condition list does not match episodes.csv."
        )
    by_planner = _sequence(summary.get("by_planner"), f"{arm.role} summary.by_planner")
    by_condition = _sequence(
        summary.get("by_planner_condition"),
        f"{arm.role} summary.by_planner_condition",
    )
    if len(by_planner) != 1 or len(by_condition) != 1:
        raise ValueError(f"{arm.role} summary must contain one aggregate per scope.")
    _validate_aggregate(
        by_planner[0],
        label=f"{arm.role} summary.by_planner[0]",
        planner=planner,
        condition_id=None,
        counts=counts,
        episodes=episodes,
    )
    _validate_aggregate(
        by_condition[0],
        label=f"{arm.role} summary.by_planner_condition[0]",
        planner=planner,
        condition_id=config.raw["condition_id"],
        counts=counts,
        episodes=episodes,
    )
    expected_provenance = {
        "schema_version": 1,
        "model_path": model["path"],
        "model_sha256": model["sha256"],
        "model_variant": model["variant"],
        "training_seed": config.raw["training_seed"],
        "split": config.raw["split"],
        "condition_id": config.raw["condition_id"],
        "planner_label": planner,
        "scenario_seed_start": seed_start,
        "scenario_seed_count": config.raw["episodes"],
        "scenario_seeds": list(expected_seeds),
        "obstacle_count": config.raw["scenario"]["obstacle_count"],
        "static_layout": config.raw["scenario"]["static_layout"],
        "safety_shield_enabled": arm.profile.enabled,
    }
    for name, expected in expected_provenance.items():
        if provenance.get(name) != expected or type(provenance.get(name)) is not type(expected):
            raise ValueError(
                f"{arm.role} summary provenance.{name} does not match the configuration."
            )
    reset_health = _mapping(
        provenance.get("reset_health"),
        f"{arm.role} summary provenance.reset_health",
    )
    expected_reset_health = {
        "attempts": config.raw["episodes"],
        "successful": config.raw["episodes"],
        "finite_settle_checks": 2 * config.raw["episodes"],
        "readback_checks": config.raw["episodes"],
    }
    for name, expected in expected_reset_health.items():
        if reset_health.get(name) != expected or type(reset_health.get(name)) is not int:
            raise ValueError(
                f"{arm.role} summary reset_health.{name} does not match the "
                "completed seed window."
            )
    for name in ("horizontal_route_half_length", "vertical_route_half_length"):
        if not _same_number(provenance.get(name), config.raw["scenario"][name]):
            raise ValueError(f"{arm.role} summary provenance.{name} does not match the scenario.")
    speeds = provenance.get("obstacle_speed_range")
    if (
        not isinstance(speeds, list)
        or len(speeds) != 2
        or any(
            not _same_number(actual, expected)
            for actual, expected in zip(
                speeds,
                config.raw["scenario"]["obstacle_speed_range"],
                strict=True,
            )
        )
    ):
        raise ValueError(f"{arm.role} summary obstacle_speed_range does not match the scenario.")
    controller_planner = _string(provenance.get("planner"), f"{arm.role} provenance.planner")
    shield_raw = _exact_fields(
        provenance.get("safety_shield_config"),
        set(SafetyShieldConfig.__dataclass_fields__),
        f"{arm.role} provenance.safety_shield_config",
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in shield_raw.values()
    ):
        raise ValueError(f"{arm.role} provenance safety_shield_config must contain finite numbers.")
    try:
        shield_config = SafetyShieldConfig(**shield_raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{arm.role} provenance safety_shield_config is invalid.") from error
    normalized_shield = {
        name: float(getattr(shield_config, name))
        for name in SafetyShieldConfig.__dataclass_fields__
    }
    for name, expected in arm.profile.config_values.items():
        if not _same_number(normalized_shield[name], expected):
            raise ValueError(
                f"{arm.role} provenance shield field {name} does not match the profile."
            )
    return _LoadedArm(
        specification=arm,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        summary_path=summary_path,
        summary_sha256=summary_sha256,
        episodes_path=episodes_path,
        episodes_sha256=episodes_sha256,
        summary=summary,
        episodes=episodes,
        planner=planner,
        controller_planner=controller_planner,
        shield_config=normalized_shield,
        counts=counts,
        started_at=started_at,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("A percentile requires at least one value.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _exact_one_sided_mcnemar(reference_only: int, candidate_only: int) -> float:
    discordant = reference_only + candidate_only
    if discordant == 0:
        return 1.0
    numerator = sum(math.comb(discordant, value) for value in range(reference_only, discordant + 1))
    return numerator / (2**discordant)


def _gate_check(
    operator: str, observed: float | int | None, threshold: float | int
) -> dict[str, Any]:
    if observed is None:
        passed = False
    elif operator == ">=":
        passed = observed >= threshold
    elif operator == "<=":
        passed = observed <= threshold
    else:  # pragma: no cover - private call sites fix the operator
        raise ValueError(f"Unsupported gate operator: {operator}")
    return {
        "operator": operator,
        "observed": observed,
        "threshold": threshold,
        "passed": passed,
    }


def build_paired_safety_report(config: PairedSafetyConfig) -> dict[str, Any]:
    """Verify both result arms, recompute paired metrics, and apply the gate."""

    # Recheck mutable inputs immediately before consuming result artifacts.
    if sha256_file(config.source_path) != config.source_sha256:
        raise ValueError("Paired diagnostic configuration changed after it was loaded.")
    for path, expected, label in (
        (config.model_path, config.raw["model"]["sha256"], "model"),
        (config.world_path, config.raw["runtime_inputs"]["world"]["sha256"], "world"),
        (
            config.evaluator_path,
            config.raw["runtime_inputs"]["evaluator"]["sha256"],
            "evaluator controller",
        ),
    ):
        _verify_file_sha256(path, expected, label)
    config.reference.profile.verify_source(config.repository_root)
    config.candidate.profile.verify_source(config.repository_root)
    if _git_output(config.repository_root, "rev-parse", "HEAD") != config.git_revision:
        raise ValueError("Git revision changed after the paired configuration was loaded.")
    current_binding = _verify_preregistration_git_binding(
        config.repository_root,
        config.source_path,
        feature_revision=config.raw["sweep_feature_revision"],
        current_revision=config.git_revision,
    )
    if (
        current_binding.binding_sha256 != config.git_binding_sha256
        or current_binding.preregistration_revision != config.preregistration_revision
        or current_binding.config_blob_oid != config.preregistration_blob_oid
        or current_binding.committed_at != config.preregistration_committed_at
    ):
        raise ValueError("Pre-registration Git binding changed after it was loaded.")

    reference = _load_result_arm(config, config.reference)
    candidate = _load_result_arm(config, config.candidate)
    if (
        reference.planner != candidate.planner
        or reference.controller_planner != candidate.controller_planner
    ):
        raise ValueError("Reference and candidate results must use the same planner identity.")
    if reference.shield_config != candidate.shield_config:
        raise ValueError(
            "Reference and candidate runtime shield configs must differ only by enabled."
        )
    seeds = tuple(sorted(reference.episodes))
    if tuple(sorted(candidate.episodes)) != seeds:
        raise ValueError("Reference and candidate results do not share the same scenario seeds.")

    pair_counts = {f"{left}_{right}": 0 for left in _OUTCOMES for right in _OUTCOMES}
    rescue_seeds: list[int] = []
    regression_seeds: list[int] = []
    common_collision_seeds: list[int] = []
    common_success_rows: list[dict[str, float | int]] = []
    reference_only_collisions = 0
    candidate_only_collisions = 0
    new_timeouts = 0
    for seed in seeds:
        ref_episode = reference.episodes[seed]
        cand_episode = candidate.episodes[seed]
        if not _same_number(
            ref_episode.reference_path_length_m,
            cand_episode.reference_path_length_m,
        ):
            raise ValueError(
                f"Scenario seed {seed} has different reference path lengths across arms."
            )
        pair_counts[f"{ref_episode.outcome}_{cand_episode.outcome}"] += 1
        if ref_episode.outcome == "collision" and cand_episode.outcome != "collision":
            reference_only_collisions += 1
        if ref_episode.outcome != "collision" and cand_episode.outcome == "collision":
            candidate_only_collisions += 1
        if ref_episode.outcome != "timeout" and cand_episode.outcome == "timeout":
            new_timeouts += 1
        if ref_episode.outcome == "collision" and cand_episode.outcome == "success":
            rescue_seeds.append(seed)
        if ref_episode.outcome == "success" and cand_episode.outcome == "collision":
            regression_seeds.append(seed)
        if ref_episode.outcome == cand_episode.outcome == "collision":
            common_collision_seeds.append(seed)
        if ref_episode.outcome == cand_episode.outcome == "success":
            assert ref_episode.completion_time_s is not None
            assert cand_episode.completion_time_s is not None
            common_success_rows.append(
                {
                    "scenario_seed": seed,
                    "reference_completion_time_s": ref_episode.completion_time_s,
                    "candidate_completion_time_s": cand_episode.completion_time_s,
                    "candidate_to_reference_ratio": (
                        cand_episode.completion_time_s / ref_episode.completion_time_s
                    ),
                }
            )
    ratios = [float(row["candidate_to_reference_ratio"]) for row in common_success_rows]
    median_ratio = statistics.median(ratios) if ratios else None
    p90_ratio = _percentile(ratios, 0.9) if ratios else None
    reference_collisions = reference.counts["collision_count"]
    collision_reduction = reference_collisions - candidate.counts["collision_count"]
    relative_reduction = (
        collision_reduction / reference_collisions if reference_collisions else None
    )
    mcnemar_p = _exact_one_sided_mcnemar(
        reference_only_collisions,
        candidate_only_collisions,
    )
    gate = config.raw["candidate_gate"]
    candidate_total = candidate.counts["episodes"]
    success_rate = candidate.counts["success_count"] / candidate_total
    collision_rate = candidate.counts["collision_count"] / candidate_total
    checks = {
        "minimum_success_rate": _gate_check(">=", success_rate, gate["minimum_success_rate"]),
        "maximum_collision_rate": _gate_check("<=", collision_rate, gate["maximum_collision_rate"]),
        "minimum_relative_collision_reduction": _gate_check(
            ">=", relative_reduction, gate["minimum_relative_collision_reduction"]
        ),
        "exact_one_sided_mcnemar_significance_level": _gate_check(
            "<=", mcnemar_p, gate["exact_one_sided_mcnemar_significance_level"]
        ),
        "maximum_new_timeout_count": _gate_check(
            "<=", new_timeouts, gate["maximum_new_timeout_count"]
        ),
        "maximum_median_completion_time_ratio": _gate_check(
            "<=", median_ratio, gate["maximum_median_completion_time_ratio"]
        ),
        "minimum_common_success_count": _gate_check(
            ">=", len(common_success_rows), gate["minimum_common_success_count"]
        ),
        "maximum_error_count_per_arm": _gate_check(
            "<=",
            max(reference.counts["error_count"], candidate.counts["error_count"]),
            gate["maximum_error_count_per_arm"],
        ),
    }
    passed = all(check["passed"] for check in checks.values())
    decision = config.raw["decision"]

    def arm_binding(loaded: _LoadedArm) -> dict[str, Any]:
        profile = loaded.specification.profile
        return {
            "arm_id": loaded.specification.arm_id,
            "profile": {
                "path": _portable(profile.source_path, config.repository_root),
                "source_sha256": profile.source_sha256,
                "semantic_sha256": profile.semantic_sha256,
                "result_token": profile.result_token,
                "enabled": profile.enabled,
            },
            "sweep_manifest": {
                "path": _portable(loaded.manifest_path, config.repository_root),
                "sha256": loaded.manifest_sha256,
            },
            "summary": {
                "path": _portable(loaded.summary_path, config.repository_root),
                "sha256": loaded.summary_sha256,
            },
            "episodes_csv": {
                "path": _portable(loaded.episodes_path, config.repository_root),
                "sha256": loaded.episodes_sha256,
            },
            "started_at": loaded.started_at.isoformat(),
        }

    def arm_metrics(loaded: _LoadedArm) -> dict[str, Any]:
        total = loaded.counts["episodes"]
        return {
            **loaded.counts,
            "success_rate": loaded.counts["success_count"] / total,
            "collision_rate": loaded.counts["collision_count"] / total,
            "timeout_rate": loaded.counts["timeout_count"] / total,
            "error_rate": loaded.counts["error_count"] / total,
        }

    seed_payload = json.dumps(seeds, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "kind": "webots_paired_safety_diagnostic",
        "experiment_id": config.raw["experiment_id"],
        "purpose": config.raw["purpose"],
        "split": config.raw["split"],
        "condition_id": config.raw["condition_id"],
        "source_bindings": {
            "input_config": {
                "path": _portable(config.source_path, config.repository_root),
                "sha256": config.source_sha256,
                "git_binding": {
                    "path": _portable(config.git_binding_path, config.repository_root),
                    "sha256": config.git_binding_sha256,
                    "preregistration_revision": config.preregistration_revision,
                    "preregistration_committed_at": (
                        config.preregistration_committed_at.isoformat()
                    ),
                    "config_blob_oid": config.preregistration_blob_oid,
                    "config_clean": True,
                    "committed_before_both_runs": True,
                },
            },
            "git_revision": config.git_revision,
            "sweep_feature_revision": config.raw["sweep_feature_revision"],
            "model": {
                "path": _portable(config.model_path, config.repository_root),
                "sha256": config.raw["model"]["sha256"],
            },
            "world": {
                "path": _portable(config.world_path, config.repository_root),
                "sha256": config.raw["runtime_inputs"]["world"]["sha256"],
            },
            "controller": {
                "path": _portable(config.evaluator_path, config.repository_root),
                "sha256": config.raw["runtime_inputs"]["evaluator"]["sha256"],
            },
            "arms": {
                "reference": arm_binding(reference),
                "candidate": arm_binding(candidate),
            },
        },
        "provenance_limits": {
            "evaluation_runtime_worktree_state_cryptographically_bound": False,
            "webots_executable_binary_sha256_bound": False,
            "webots_environment_adapter_sha256_preregistered": False,
            "interpretation": (
                "The sweep binds the evaluated model, profile, world, controller, "
                "summaries, and episode CSV files. It did not record a cryptographic "
                "snapshot of the evaluation-time worktree, the imported Webots environment "
                "adapter, or the Webots executable."
            ),
        },
        "paired_validation": {
            "episode_count": len(seeds),
            "scenario_seed_start": seeds[0],
            "scenario_seed_end": seeds[-1],
            "scenario_seeds_sha256": hashlib.sha256(seed_payload).hexdigest(),
            "same_seed_window": True,
            "same_condition": True,
            "same_model": True,
            "same_scenario": True,
            "same_controller": True,
            "only_operational_difference": "safety_shield_enabled",
            "seed_window_rule_recomputed": True,
            "seed_window_search_start": config.raw["preregistration"][
                "seed_window_selection"
            ]["search_start"],
            "first_balanced_seed_window_start": seeds[0],
        },
        "arms": {
            "reference": arm_metrics(reference),
            "candidate": arm_metrics(candidate),
        },
        "paired_outcomes": {
            "rescue_count": len(rescue_seeds),
            "regression_count": len(regression_seeds),
            "common_success_count": len(common_success_rows),
            "common_collision_count": len(common_collision_seeds),
            "new_timeout_count": new_timeouts,
            "rescue_seeds": rescue_seeds,
            "regression_seeds": regression_seeds,
            "common_collision_seeds": common_collision_seeds,
            "full_outcome_table": pair_counts,
        },
        "collision_analysis": {
            "absolute_collision_reduction_count": collision_reduction,
            "relative_collision_reduction": relative_reduction,
            "reference_only_collision_count": reference_only_collisions,
            "candidate_only_collision_count": candidate_only_collisions,
            "exact_one_sided_mcnemar_p": mcnemar_p,
            "alternative": "candidate_collision_probability_is_lower",
        },
        "common_success_completion_time": {
            "count": len(common_success_rows),
            "median_candidate_to_reference_ratio": median_ratio,
            "p90_candidate_to_reference_ratio": p90_ratio,
            "ratios_by_seed": common_success_rows,
        },
        "gate": {
            "require_all": True,
            "checks": checks,
            "passed": passed,
        },
        "decision": {
            "passed": passed,
            "outcome": decision["pass"] if passed else decision["fail"],
            "formal_model_promotion": "forbidden",
            "model_alias_created": False,
        },
    }


def render_paired_safety_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable diagnostic report."""

    arms = report["arms"]
    paired = report["paired_outcomes"]
    collision = report["collision_analysis"]
    timing = report["common_success_completion_time"]
    decision = report["decision"]
    status = "PASS" if decision["passed"] else "FAIL"
    common_outcomes = f"{paired['common_success_count']} / {paired['common_collision_count']}"
    relative_reduction = _format_optional(collision["relative_collision_reduction"])
    median_ratio = _format_optional(timing["median_candidate_to_reference_ratio"])
    p90_ratio = _format_optional(timing["p90_candidate_to_reference_ratio"])
    lines = [
        f"# {report['experiment_id']} diagnostic report",
        "",
        f"Decision: **{status}** — `{decision['outcome']}`.",
        "",
        "> Validation-only exploratory safety diagnostic. Formal model promotion is "
        "forbidden; no model alias was created.",
        "",
        "## Arm outcomes",
        "",
        "| Arm | Success | Collision | Timeout | Error | Success rate | Collision rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role in ("reference", "candidate"):
        arm = arms[role]
        lines.append(
            f"| {role} | {arm['success_count']} | {arm['collision_count']} | "
            f"{arm['timeout_count']} | {arm['error_count']} | "
            f"{arm['success_rate']:.6f} | {arm['collision_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired analysis",
            "",
            f"- Rescue / regression: {paired['rescue_count']} / {paired['regression_count']}",
            f"- Common success / common collision: {common_outcomes}",
            f"- Relative collision reduction: {relative_reduction}",
            f"- Exact one-sided McNemar p: {collision['exact_one_sided_mcnemar_p']:.12g}",
            f"- Common-success completion-time ratio median / p90: {median_ratio} / {p90_ratio}",
            "",
            "## Preregistered gate",
            "",
            "| Check | Observed | Requirement | Result |",
            "| --- | ---: | ---: | :---: |",
        ]
    )
    for name, check in report["gate"]["checks"].items():
        lines.append(
            f"| `{name}` | {_format_optional(check['observed'])} | "
            f"{check['operator']} {_format_optional(check['threshold'])} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    bindings = report["source_bindings"]
    preregistration = bindings["input_config"]["git_binding"]
    lines.extend(
        [
            "",
            "## Audit bindings",
            "",
            f"- Input config SHA-256: `{bindings['input_config']['sha256']}`",
            f"- Pre-registration revision: `{preregistration['preregistration_revision']}`",
            f"- Pre-registration Git blob: `{preregistration['config_blob_oid']}`",
            f"- Pre-registration committed at: `{preregistration['preregistration_committed_at']}`",
            f"- Model SHA-256: `{bindings['model']['sha256']}`",
            f"- World SHA-256: `{bindings['world']['sha256']}`",
            f"- Controller SHA-256: `{bindings['controller']['sha256']}`",
            f"- Git revision: `{bindings['git_revision']}`",
            "",
            "## Provenance boundary",
            "",
            str(report["provenance_limits"]["interpretation"]),
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_paired_safety_reports(
    report: Mapping[str, Any],
    *,
    output_directory: Path,
    markdown_path: Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Write JSON, YAML, JSON digest, and Markdown without implicit overwrite."""

    output = Path(output_directory)
    markdown = Path(markdown_path)
    json_path = output / "paired_safety_report.json"
    yaml_path = output / "paired_safety_report.yaml"
    digest_path = output / "paired_safety_report.sha256"
    targets = (json_path, yaml_path, digest_path, markdown)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite paired safety report artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    json_text = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    yaml_text = yaml.safe_dump(
        json.loads(json_text),
        allow_unicode=True,
        sort_keys=False,
    )
    markdown_text = render_paired_safety_markdown(report)
    output.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    yaml_path.write_text(yaml_text, encoding="utf-8")
    digest = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    digest_path.write_text(f"{digest}  {json_path.name}\n", encoding="utf-8")
    markdown.write_text(markdown_text, encoding="utf-8")
    return json_path, yaml_path, digest_path, markdown


def analyze_paired_safety(
    config_path: Path,
    *,
    repository_root: Path,
    output_directory: Path | None = None,
    markdown_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Analyze one paired diagnostic and write all report artifacts."""

    config = load_paired_safety_config(config_path, repository_root=repository_root)
    report = build_paired_safety_report(config)
    write_paired_safety_reports(
        report,
        output_directory=output_directory or config.output_directory,
        markdown_path=markdown_path or config.markdown_path,
        overwrite=overwrite,
    )
    return report
