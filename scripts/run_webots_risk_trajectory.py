#!/usr/bin/env python3
"""Collect a SHA-bound, diagnostic-only risk trajectory corpus in Webots.

The runner deliberately has no performance gate.  A successful invocation only
means that all predeclared diagnostic episodes and their schema-validated risk
traces were collected and bound to immutable source and result artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from indoor_patrol_rl.evaluation.episode_context import (
    EPISODE_CONTEXT_SCHEMA_VERSION,
    validate_risk_trace,
)
from indoor_patrol_rl.evaluation.metrics import EpisodeRecord
from indoor_patrol_rl.training.webots_curriculum import (
    choose_available_port,
    resolve_webots_executable,
)

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
SourceVerifier = Callable[[Path], str]

EXIT_PREFLIGHT_OR_RUNTIME_FAILED = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SAFE_CONDITION = re.compile(r"[A-Za-z0-9]+")
_VALIDATION_SEEDS = frozenset(range(5_000, 5_020))
_TEST_SEEDS = frozenset(range(10_000, 10_020))
_FIXED_SEED_START = 20_494
_FIXED_EPISODES = 120
_FIXED_TRAINING_SEED = 11
_FIXED_MODEL = (
    "artifacts/models/webots/seed_11/w1_recovery_v1/"
    "ppo_webots_s11_step000040960.zip"
)
_FIXED_MODEL_SHA256 = (
    "c5e7c3de42c7664477097b2734975660fec840ad5b7f45425098158c27cb7275"
)
_FIXED_MODEL_VARIANT = "w1_recovery_v1"
_WORLD = "worlds/webots_ppo_evaluate.wbt"
_CONTROLLER = "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
_RUNNER = "scripts/run_webots_risk_trajectory.py"
_REPORT_DIRECTORY = "artifacts/results/{experiment_id}"
_EPISODE_FIELDS = tuple(EpisodeRecord.__dataclass_fields__)


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
class Scenario:
    obstacle_count: int
    obstacle_speed_range: tuple[float, float]
    horizontal_route_half_length: float
    vertical_route_half_length: float
    safety_shield: bool
    deterministic_policy: bool


@dataclass(frozen=True)
class ModelBinding:
    portable_path: str
    path: Path
    sha256: str
    variant: str
    run_name: str
    planner_label: str


@dataclass(frozen=True)
class RiskTrajectoryConfig:
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
    context_schema_version: str

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.scenario_seed_start, self.scenario_seed_start + self.episodes))


@dataclass(frozen=True)
class RiskTrajectoryRun:
    port: int
    command: tuple[str, ...]
    environment: Mapping[str, str]
    result_directory: Path
    report_directory: Path


@dataclass(frozen=True)
class ParsedResult:
    rows_by_seed: Mapping[int, Mapping[str, Any]]
    contexts_by_seed: Mapping[int, Mapping[str, Any]]
    metrics: Mapping[str, Any]
    artifacts: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_clean_tracked_source(repository_root: Path) -> str:
    """Return HEAD only when all tracked files match it exactly."""

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
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        preview = " | ".join(dirty.splitlines()[:5])
        raise RuntimeError(
            "Risk trajectory collection requires a clean tracked worktree: " + preview
        )
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
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, actual {actual}."
        )


def _axis_for_seed(seed: int) -> str:
    return str(np.random.default_rng(seed).choice(["horizontal", "vertical"]))


def load_risk_trajectory_config(
    path: Path, *, repository_root: Path
) -> RiskTrajectoryConfig:
    """Load the single supported diagnostic collection protocol."""

    root = repository_root.resolve()
    source = (path if path.is_absolute() else root / path).resolve()
    try:
        source_portable = source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Risk trajectory config must remain inside the repository.") from error
    if not source.is_file():
        raise FileNotFoundError(f"Risk trajectory config does not exist: {source}")
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
            "collection",
            "access",
        },
        "config",
    )
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must be exactly 1.")
    if raw["split"] != "diagnostic":
        raise PermissionError("Risk trajectory runner only permits split=diagnostic.")
    experiment_id = _safe_name(raw["experiment_id"], "experiment_id")
    condition_id = _text(raw["condition_id"], "condition_id")
    if _SAFE_CONDITION.fullmatch(condition_id) is None:
        raise ValueError("condition_id must contain only letters and digits.")

    training_seed = _integer(raw["training_seed"], "training_seed")
    seed_start = _integer(raw["scenario_seed_start"], "scenario_seed_start")
    episodes = _integer(raw["episodes"], "episodes", minimum=1)
    seeds = tuple(range(seed_start, seed_start + episodes))
    validation_overlap = sorted(set(seeds) & _VALIDATION_SEEDS)
    test_overlap = sorted(set(seeds) & _TEST_SEEDS)
    if validation_overlap or test_overlap:
        raise PermissionError(
            "Diagnostic seeds overlap reserved splits: "
            f"validation={validation_overlap}, held_out_test={test_overlap}."
        )
    if (training_seed, seed_start, episodes) != (
        _FIXED_TRAINING_SEED,
        _FIXED_SEED_START,
        _FIXED_EPISODES,
    ):
        raise ValueError(
            "This protocol fixes training_seed=11, scenario seeds 20494..20613, "
            "and episodes=120."
        )
    axis_counts = Counter(_axis_for_seed(seed) for seed in seeds)
    if axis_counts != {"horizontal": 60, "vertical": 60}:
        raise ValueError(
            "Fixed diagnostic seed window must map to H/V=60/60; observed "
            f"{dict(axis_counts)}."
        )

    model_raw = _mapping(raw["model"], "model")
    _exact_keys(
        model_raw,
        {"path", "sha256", "variant", "run_name", "planner_label"},
        "model",
    )
    model_portable, model_path = _inside_repository(root, model_raw["path"], "model.path")
    model_sha256 = _text(model_raw["sha256"], "model.sha256")
    if _SHA256.fullmatch(model_sha256) is None:
        raise ValueError("model.sha256 must be a lowercase SHA-256 digest.")
    if (
        model_portable != _FIXED_MODEL
        or model_sha256 != _FIXED_MODEL_SHA256
        or model_raw["variant"] != _FIXED_MODEL_VARIANT
    ):
        raise ValueError("Risk trajectory collection is locked to the recovery-v1 model and SHA.")
    if "latest" in model_path.stem.casefold():
        raise ValueError("Risk trajectory collection refuses mutable 'latest' models.")
    _verify_file(model_path, model_sha256, "Recovery-v1 model")
    model = ModelBinding(
        portable_path=model_portable,
        path=model_path,
        sha256=model_sha256,
        variant=_safe_name(model_raw["variant"], "model.variant"),
        run_name=_safe_name(model_raw["run_name"], "model.run_name"),
        planner_label=_safe_name(model_raw["planner_label"], "model.planner_label"),
    )
    if model.run_name == experiment_id:
        raise ValueError("model.run_name and experiment_id must use different directories.")

    scenario_raw = _mapping(raw["scenario"], "scenario")
    _exact_keys(
        scenario_raw,
        {
            "obstacle_count",
            "obstacle_speed_range",
            "horizontal_route_half_length",
            "vertical_route_half_length",
            "safety_shield",
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
    locked_scenario = (
        obstacle_count == 1
        and speed_range == (0.15, 0.30)
        and horizontal == 1.10
        and vertical == 1.10
        and scenario_raw["safety_shield"] is False
        and scenario_raw["deterministic_policy"] is True
    )
    if not locked_scenario:
        raise ValueError(
            "Scenario is locked to obstacle1, speed=[0.15,0.30], route halves=1.10, "
            "shield=false, deterministic=true."
        )
    scenario = Scenario(obstacle_count, speed_range, horizontal, vertical, False, True)

    collection = _mapping(raw["collection"], "collection")
    _exact_keys(
        collection,
        {
            "episode_context_schema_version",
            "risk_trace_required",
            "required_episodes",
            "expected_route_axis_counts",
            "completion_rule",
            "performance_gate",
            "adaptive_extension",
        },
        "collection",
    )
    expected_axes = _mapping(
        collection["expected_route_axis_counts"],
        "collection.expected_route_axis_counts",
    )
    _exact_keys(
        expected_axes,
        {"horizontal", "vertical"},
        "collection.expected_route_axis_counts",
    )
    locked_collection = (
        collection["episode_context_schema_version"] == "1.1"
        and collection["risk_trace_required"] is True
        and _integer(collection["required_episodes"], "collection.required_episodes")
        == episodes
        and _integer(
            expected_axes["horizontal"],
            "collection.expected_route_axis_counts.horizontal",
        )
        == 60
        and _integer(
            expected_axes["vertical"],
            "collection.expected_route_axis_counts.vertical",
        )
        == 60
        and collection["completion_rule"] == "complete_artifact_collection_only"
        and collection["performance_gate"] == "none"
        and collection["adaptive_extension"] == "forbidden"
    )
    if not locked_collection:
        raise ValueError("collection must require schema 1.1 risk traces with no performance gate.")
    if EPISODE_CONTEXT_SCHEMA_VERSION != collection["episode_context_schema_version"]:
        raise RuntimeError(
            "Installed episode-context schema does not match the locked collection schema."
        )
    access = _mapping(raw["access"], "access")
    _exact_keys(access, {"validation_or_test"}, "access")
    if access["validation_or_test"] != "forbidden":
        raise PermissionError("Validation and held-out test access must remain forbidden.")

    return RiskTrajectoryConfig(
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
        context_schema_version=collection["episode_context_schema_version"],
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
    config: RiskTrajectoryConfig,
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
    bindings: Mapping[str, Any], *, repository_root: Path, config: RiskTrajectoryConfig
) -> None:
    root = repository_root.resolve()
    for label in ("config", "runner", "controller", "world", "model"):
        binding = _mapping(bindings[label], f"source_bindings.{label}")
        _verify_file(root / str(binding["path"]), str(binding["sha256"]), label)
    executable = _mapping(bindings["webots_executable"], "source_bindings.webots_executable")
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
    config: RiskTrajectoryConfig,
    *,
    repository_root: Path,
    webots_executable: Path,
    port_base: int,
) -> RiskTrajectoryRun:
    root = repository_root.resolve()
    world = (root / _WORLD).resolve()
    controller = (root / _CONTROLLER).resolve()
    for label, path in (("world", world), ("controller", controller)):
        if not path.is_file():
            raise FileNotFoundError(f"Webots evaluation {label} does not exist: {path}")
    if not 1_024 <= port_base <= 65_535:
        raise ValueError("port_base must be between 1024 and 65535.")
    result_directory = root / "artifacts/results" / config.model.run_name
    report_directory = root / _REPORT_DIRECTORY.format(experiment_id=config.experiment_id)
    existing = [path for path in (result_directory, report_directory) if path.exists()]
    if existing:
        raise FileExistsError(
            "Risk trajectory runner refuses to overwrite artifact directories: "
            + ", ".join(str(path) for path in existing)
        )
    port = choose_available_port(port_base)
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
        "WEBOTS_EVAL_PLANNER_LABEL": config.model.planner_label,
        "WEBOTS_EVAL_RUN_NAME": config.model.run_name,
        "WEBOTS_OBSTACLE_COUNT": str(config.scenario.obstacle_count),
        "WEBOTS_OBSTACLE_SPEED_MIN": str(config.scenario.obstacle_speed_range[0]),
        "WEBOTS_OBSTACLE_SPEED_MAX": str(config.scenario.obstacle_speed_range[1]),
        "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": str(
            config.scenario.horizontal_route_half_length
        ),
        "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": str(
            config.scenario.vertical_route_half_length
        ),
        "WEBOTS_SAFETY_SHIELD": "0",
        "WEBOTS_EVAL_DETERMINISTIC_POLICY": "1",
        "WEBOTS_EVAL_EPISODE_CONTEXT": "1",
        "WEBOTS_RISK_TRAJECTORY_ID": config.experiment_id,
        "WEBOTS_RISK_TRAJECTORY_CONFIG": config.source_portable_path,
        "WEBOTS_RISK_TRAJECTORY_CONFIG_SHA256": config.source_sha256,
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
    return RiskTrajectoryRun(
        port=port,
        command=command,
        environment=overrides,
        result_directory=result_directory,
        report_directory=report_directory,
    )


def _csv_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{label} must be True or False in episodes.csv.")


def _read_episode_rows(
    path: Path, *, config: RiskTrajectoryConfig
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
            "episode_id": (
                f"diagnostic-{config.condition_id}-{seed}-{config.model.planner_label}"
            ),
            "split": "diagnostic",
            "condition_id": config.condition_id,
            "planner": config.model.planner_label,
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
        result[seed] = {
            "episode_id": raw["episode_id"],
            "outcome": outcome,
            "is_success": success,
            "collision": collision,
            "timeout": timeout,
        }
    if set(result) != set(config.seeds):
        raise ValueError("episodes.csv does not contain the exact diagnostic seed set.")
    return result


def _read_contexts(
    path: Path,
    *,
    config: RiskTrajectoryConfig,
    episode_rows: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    records: list[Any] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"episode_context.jsonl line {line_number} is blank.")
            records.append(json.loads(line))
    if len(records) != config.episodes:
        raise ValueError(
            f"episode_context.jsonl must contain exactly {config.episodes} rows."
        )
    contexts: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(records):
        record = _mapping(value, f"episode_context row {index}")
        if record.get("schema_version") != config.context_schema_version:
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
            "planner": config.model.planner_label,
        }
        if identity != expected_identity:
            raise ValueError(f"episode_context row {index} identity disagrees with episodes.csv.")
        axis = record.get("route_axis")
        if axis != _axis_for_seed(seed):
            raise ValueError(f"episode_context seed {seed} has an unexpected route_axis.")
        if type(record.get("reverse_route")) is not bool:
            raise ValueError(f"episode_context row {index} reverse_route must be boolean.")
        start_xy = record.get("start_xy")
        goal_xy = record.get("goal_xy")
        if not (
            isinstance(start_xy, list)
            and isinstance(goal_xy, list)
            and len(start_xy) == len(goal_xy) == 2
        ):
            raise ValueError(f"episode_context row {index} route coordinates are invalid.")
        route_index = 0 if axis == "horizontal" else 1
        route_half_length = (
            config.scenario.horizontal_route_half_length
            if axis == "horizontal"
            else config.scenario.vertical_route_half_length
        )
        if not (
            math.isclose(abs(float(start_xy[route_index])), route_half_length, abs_tol=1e-12)
            and math.isclose(abs(float(goal_xy[route_index])), route_half_length, abs_tol=1e-12)
        ):
            raise ValueError(f"episode_context seed {seed} route length disagrees with the plan.")
        obstacles = record.get("dynamic_obstacles")
        if not isinstance(obstacles, list) or len(obstacles) != config.scenario.obstacle_count:
            raise ValueError(f"episode_context seed {seed} has an invalid obstacle count.")
        for obstacle in obstacles:
            if not isinstance(obstacle, dict):
                raise ValueError(f"episode_context seed {seed} has an invalid obstacle.")
            speed = _finite_number(obstacle.get("speed"), "dynamic_obstacle.speed")
            if not config.scenario.obstacle_speed_range[0] <= speed <= (
                config.scenario.obstacle_speed_range[1]
            ):
                raise ValueError(f"episode_context seed {seed} obstacle speed is out of range.")
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
                if len(sample["post_obstacle_xy"]) != config.scenario.obstacle_count:
                    raise ValueError(
                        f"episode_context row {index} trace obstacle count is invalid."
                    )
        final_sample = trace["terminal_window"][-1]
        for source in ("contact", "lidar"):
            termination_key = f"{source}_collision"
            sample_key = f"post_{source}_collision"
            if termination.get(termination_key) != final_sample[sample_key]:
                raise ValueError(
                    f"episode_context row {index} terminal {source} collision disagrees "
                    "with the risk trace."
                )
        contexts[seed] = record
    if set(contexts) != set(config.seeds):
        raise ValueError("episode_context.jsonl does not contain the exact diagnostic seed set.")
    axis_counts = Counter(record["route_axis"] for record in contexts.values())
    if axis_counts != {"horizontal": 60, "vertical": 60}:
        raise ValueError(f"Collected contexts are not H/V=60/60: {dict(axis_counts)}.")
    return contexts


def _count_metrics(
    rows: Mapping[int, Mapping[str, Any]],
    contexts: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    def counts(seeds: Sequence[int]) -> dict[str, int]:
        outcomes = Counter(str(rows[seed]["outcome"]) for seed in seeds)
        return {
            "episodes": len(seeds),
            "success_count": outcomes["success"],
            "collision_count": outcomes["collision"],
            "timeout_count": outcomes["timeout"],
            "error_count": outcomes["error"],
        }

    seeds = sorted(rows)
    horizontal = [seed for seed in seeds if contexts[seed]["route_axis"] == "horizontal"]
    vertical = [seed for seed in seeds if contexts[seed]["route_axis"] == "vertical"]
    return {
        "overall": counts(seeds),
        "by_route_axis": {
            "horizontal": counts(horizontal),
            "vertical": counts(vertical),
        },
    }


def _check_summary(
    summary_path: Path,
    *,
    repository_root: Path,
    config: RiskTrajectoryConfig,
    episode_rows: Mapping[int, Mapping[str, Any]],
    contexts: Mapping[int, Mapping[str, Any]],
    episodes_sha256: str,
    context_path: Path,
    context_sha256: str,
) -> None:
    summary = _mapping(json.loads(summary_path.read_text(encoding="utf-8")), "summary.json")
    for name, expected in {
        "schema_version": "1.0",
        "split": "diagnostic",
        "episode_count": config.episodes,
        "planners": [config.model.planner_label],
        "conditions": [config.condition_id],
    }.items():
        if summary.get(name) != expected:
            raise ValueError(f"summary.json {name} disagrees with the collection plan.")
    by_planner = summary.get("by_planner")
    if not isinstance(by_planner, list) or len(by_planner) != 1:
        raise ValueError("summary.json must contain exactly one planner aggregate.")
    aggregate = _mapping(by_planner[0], "summary.by_planner[0]")
    expected_counts = _count_metrics(episode_rows, contexts)["overall"]
    for name, expected in expected_counts.items():
        if aggregate.get(name) != expected:
            raise ValueError(f"summary aggregate {name} disagrees with episodes.csv.")
    provenance = _mapping(summary.get("evaluation_provenance"), "evaluation_provenance")
    expected_provenance = {
        "schema_version": 1,
        "model_path": config.model.portable_path,
        "model_sha256": config.model.sha256,
        "model_variant": config.model.variant,
        "training_seed": config.training_seed,
        "split": "diagnostic",
        "condition_id": config.condition_id,
        "planner": "ppo",
        "planner_label": config.model.planner_label,
        "deterministic_policy": True,
        "scenario_seed_start": config.scenario_seed_start,
        "scenario_seed_count": config.episodes,
        "scenario_seeds": list(config.seeds),
        "obstacle_count": config.scenario.obstacle_count,
        "obstacle_speed_range": list(config.scenario.obstacle_speed_range),
        "horizontal_route_half_length": config.scenario.horizontal_route_half_length,
        "vertical_route_half_length": config.scenario.vertical_route_half_length,
        "safety_shield_enabled": False,
        "episodes_csv_sha256": episodes_sha256,
    }
    for name, expected in expected_provenance.items():
        if provenance.get(name) != expected:
            raise ValueError(f"evaluation_provenance.{name} disagrees with the plan.")
    expected_context = {
        "path": context_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "schema_version": config.context_schema_version,
        "count": config.episodes,
        "sha256": context_sha256,
    }
    if provenance.get("episode_context") != expected_context:
        raise ValueError("summary episode_context provenance binding is invalid.")
    reset_health = _mapping(provenance.get("reset_health"), "reset_health")
    for name, expected in {
        "attempts": config.episodes,
        "successful": config.episodes,
        "finite_settle_checks": 2 * config.episodes,
        "readback_checks": config.episodes,
    }.items():
        if reset_health.get(name) != expected:
            raise ValueError(f"reset_health.{name} disagrees with a complete collection.")


def _parse_result(
    run: RiskTrajectoryRun,
    *,
    config: RiskTrajectoryConfig,
    repository_root: Path,
) -> ParsedResult:
    summary_path = run.result_directory / "summary.json"
    episodes_path = run.result_directory / "episodes.csv"
    context_path = run.result_directory / "episode_context.jsonl"
    for path in (summary_path, episodes_path, context_path):
        if not path.is_file():
            raise FileNotFoundError(f"Collection omitted required artifact: {path}")
    episodes_sha256 = _sha256_file(episodes_path)
    context_sha256 = _sha256_file(context_path)
    rows = _read_episode_rows(episodes_path, config=config)
    contexts = _read_contexts(context_path, config=config, episode_rows=rows)
    _check_summary(
        summary_path,
        repository_root=repository_root,
        config=config,
        episode_rows=rows,
        contexts=contexts,
        episodes_sha256=episodes_sha256,
        context_path=context_path,
        context_sha256=context_sha256,
    )
    root = repository_root.resolve()
    artifacts = {
        "result_directory": run.result_directory.resolve().relative_to(root).as_posix(),
        "summary_json": _portable_binding(summary_path, root),
        "episodes_csv": _portable_binding(episodes_path, root),
        "episode_context_jsonl": _portable_binding(context_path, root),
    }
    return ParsedResult(
        rows_by_seed=rows,
        contexts_by_seed=contexts,
        metrics=_count_metrics(rows, contexts),
        artifacts=artifacts,
    )


def _write_report(report: Mapping[str, Any], *, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=False)
    json_path = output_directory / "collection_report.json"
    yaml_path = output_directory / "collection_report.yaml"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    yaml_path.write_text(
        yaml.safe_dump(dict(report), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_directory / "collection_report.sha256").write_text(
        f"{_sha256_file(json_path)}  {json_path.name}\n"
        f"{_sha256_file(yaml_path)}  {yaml_path.name}\n",
        encoding="utf-8",
    )


def execute_collection(
    *,
    repository_root: Path,
    config_path: Path,
    webots_executable: Path,
    dry_run: bool,
    port_base: int = 13_640,
    timeout_seconds: float = 7_200.0,
    process_runner: ProcessRunner = subprocess.run,
    source_verifier: SourceVerifier = require_clean_tracked_source,
) -> tuple[int, Mapping[str, Any]]:
    """Preflight and optionally execute the locked single-model collection."""

    root = repository_root.resolve()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive.")
    source_revision = source_verifier(root)
    if _GIT_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source_verifier returned an invalid Git revision.")
    config = load_risk_trajectory_config(config_path, repository_root=root)
    webots = resolve_webots_executable(webots_executable)
    run = build_run_plan(
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
    axis_counts = dict(Counter(_axis_for_seed(seed) for seed in config.seeds))
    plan: dict[str, Any] = {
        "schema_version": 1,
        "dry_run": dry_run,
        "experiment_id": config.experiment_id,
        "split": "diagnostic",
        "source_bindings": bindings,
        "scenario_seeds": list(config.seeds),
        "axis_counts": axis_counts,
        "risk_trace_schema_version": config.context_schema_version,
        "required_episodes": config.episodes,
        "expected_route_axis_counts": {"horizontal": 60, "vertical": 60},
        "adaptive_extension": "forbidden",
        "performance_gate_applied": False,
        "validation_or_test_access": "forbidden",
        "run": {
            "model": config.model.portable_path,
            "model_sha256": config.model.sha256,
            "run_name": config.model.run_name,
            "command": list(run.command),
            "environment": dict(run.environment),
        },
    }
    if dry_run:
        return 0, plan

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
            f"Webots risk trajectory collection failed with exit code {completed.returncode}."
        )
    _verify_source_bindings(bindings, repository_root=root, config=config)
    parsed = _parse_result(run, config=config, repository_root=root)
    if source_verifier(root) != source_revision:
        raise RuntimeError("Git HEAD or tracked worktree changed during collection.")
    _verify_source_bindings(bindings, repository_root=root, config=config)

    report_directory = _REPORT_DIRECTORY.format(experiment_id=config.experiment_id)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "collection_complete",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "diagnostic",
        "validation_split_was_run": False,
        "test_split_was_run": False,
        "performance_gate_applied": False,
        "source_bindings": bindings,
        "scenario": {
            "condition_id": config.condition_id,
            "scenario_seed_start": config.scenario_seed_start,
            "episodes": config.episodes,
            "scenario_seeds": list(config.seeds),
            "axis_counts": axis_counts,
            "obstacle_count": config.scenario.obstacle_count,
            "obstacle_speed_range": list(config.scenario.obstacle_speed_range),
            "horizontal_route_half_length": config.scenario.horizontal_route_half_length,
            "vertical_route_half_length": config.scenario.vertical_route_half_length,
            "safety_shield": False,
            "deterministic_policy": True,
        },
        "collection": {
            "expected_episode_count": config.episodes,
            "episodes_csv_count": len(parsed.rows_by_seed),
            "episode_context_count": len(parsed.contexts_by_seed),
            "risk_trace_count": len(parsed.contexts_by_seed),
            "episode_context_schema_version": config.context_schema_version,
            "expected_route_axis_counts": {"horizontal": 60, "vertical": 60},
            "adaptive_extension": "forbidden",
            "complete": True,
        },
        "model": {
            "path": config.model.portable_path,
            "sha256": config.model.sha256,
            "variant": config.model.variant,
            "training_seed": config.training_seed,
        },
        "execution": {
            "command": list(run.command),
            "environment": dict(run.environment),
            "port": run.port,
            "exit_code": completed.returncode,
        },
        "descriptive_metrics_not_used_as_a_gate": dict(parsed.metrics),
        "artifacts": dict(parsed.artifacts),
        "access": {"validation_or_test": "forbidden"},
        "written_reports": {
            "json": f"{report_directory}/collection_report.json",
            "yaml": f"{report_directory}/collection_report.yaml",
            "sha256_sidecar": f"{report_directory}/collection_report.sha256",
        },
    }
    _write_report(report, output_directory=run.report_directory)
    return 0, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/webots_recovery_v1_risk_trajectory_v1.yaml"),
    )
    parser.add_argument("--webots", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=13_640)
    parser.add_argument("--timeout-seconds", type=float, default=7_200.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).absolute().parents[1]
    try:
        exit_code, result = execute_collection(
            repository_root=root,
            config_path=args.config,
            webots_executable=args.webots,
            dry_run=args.dry_run,
            port_base=args.port_base,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as error:
        print(
            f"WEBOTS_RISK_TRAJECTORY_FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_PREFLIGHT_OR_RUNTIME_FAILED
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("WEBOTS_RISK_TRAJECTORY_DRY_RUN_OK", file=sys.stderr)
    else:
        print("WEBOTS_RISK_TRAJECTORY_COLLECTION_COMPLETE", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
