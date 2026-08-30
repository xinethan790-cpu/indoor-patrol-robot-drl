#!/usr/bin/env python3
"""Run the predeclared paired H1 diagnostic checkpoint audit.

This runner is deliberately diagnostic-only.  It binds both model artifacts by
SHA-256, runs the same scenarios for both arms, and refuses any reserved
validation or held-out test seed before Webots can be started.
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

from indoor_patrol_rl.evaluation.metrics import EpisodeRecord
from indoor_patrol_rl.training.webots_curriculum import (
    choose_available_port,
    resolve_webots_executable,
)

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
SourceVerifier = Callable[[Path], str]

EXIT_PREFLIGHT_OR_RUNTIME_FAILED = 2
EXIT_CANDIDATE_GATE_FAILED = 3
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SAFE_CONDITION = re.compile(r"[A-Za-z0-9]+")
_VALIDATION_SEEDS = frozenset(range(5_000, 5_020))
_TEST_SEEDS = frozenset(range(10_000, 10_020))
_WORLD = "worlds/webots_ppo_evaluate.wbt"
_AUDIT_DIRECTORY = "artifacts/results/{experiment_id}"
_EPISODE_FIELDS = tuple(EpisodeRecord.__dataclass_fields__)
_GIT_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


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
class Arm:
    arm_id: str
    role: str
    model: str
    model_path: Path
    model_sha256: str
    model_variant: str
    run_name: str
    planner_label: str


@dataclass(frozen=True)
class CandidateGate:
    require_all: bool
    minimum_success_count: int
    maximum_collision_count: int
    minimum_horizontal_success_count: int
    minimum_vertical_success_count: int
    minimum_horizontal_paired_net_rescues: int
    maximum_vertical_paired_net_regressions: int


@dataclass(frozen=True)
class AuditConfig:
    source_path: Path
    source_portable_path: str
    source_sha256: str
    experiment_id: str
    condition_id: str
    training_seed: int
    scenario_seed_start: int
    episodes: int
    scenario: Scenario
    selection_basis: Mapping[str, Any]
    source_monitor_path: Path
    arms: tuple[Arm, Arm]
    gate: CandidateGate
    decision: Mapping[str, str]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.scenario_seed_start, self.scenario_seed_start + self.episodes))

    def arm_for_role(self, role: str) -> Arm:
        return next(arm for arm in self.arms if arm.role == role)


@dataclass(frozen=True)
class AuditRun:
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
    metrics: Mapping[str, Any]
    artifact_bindings: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_clean_tracked_source(repository_root: Path) -> str:
    """Bind the audit to HEAD and reject modifications to tracked source files."""

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
            "Diagnostic audit requires a clean tracked worktree: " + preview
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


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _inside_repository(repository_root: Path, value: Any, label: str) -> tuple[str, Path]:
    portable = _text(value, label)
    configured = Path(portable)
    if configured.is_absolute() or "\\" in portable or portable in {".", ".."}:
        raise ValueError(f"{label} must be a repository-relative POSIX path.")
    resolved = (repository_root / configured).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository.") from error
    if ".." in configured.parts:
        raise ValueError(f"{label} escapes the repository.")
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


def load_audit_config(path: Path, *, repository_root: Path) -> AuditConfig:
    """Load the one supported paired audit schema and bind immutable inputs."""

    root = repository_root.resolve()
    source = path if path.is_absolute() else root / path
    source = source.resolve()
    try:
        source_portable = source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Audit config must remain inside the repository.") from error
    if not source.is_file():
        raise FileNotFoundError(f"Audit config does not exist: {source}")
    raw = _mapping(yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader), "config")
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
            "scenario",
            "selection_basis",
            "arms",
            "candidate_gate",
            "decision",
        },
        "config",
    )
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must be exactly 1.")
    if raw["split"] != "diagnostic":
        raise PermissionError("Paired audit runner only permits split=diagnostic.")
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
            "Diagnostic audit seeds overlap reserved splits: "
            f"validation={validation_overlap}, held_out_test={test_overlap}."
        )
    axes = [_axis_for_seed(seed) for seed in seeds]
    axis_counts = Counter(axes)
    if episodes % 2 or axis_counts != {"horizontal": episodes // 2, "vertical": episodes // 2}:
        raise ValueError(
            "Diagnostic seed window must be exactly axis-balanced under the evaluator's "
            f"default route mapping; observed {dict(axis_counts)}."
        )

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
    obstacle_count = _integer(scenario_raw["obstacle_count"], "scenario.obstacle_count")
    if obstacle_count > 3:
        raise ValueError("scenario.obstacle_count must be <= 3.")
    speed_values = scenario_raw["obstacle_speed_range"]
    if not isinstance(speed_values, list) or len(speed_values) != 2:
        raise ValueError("scenario.obstacle_speed_range must be a two-value list.")
    speed_range = (
        _finite_number(speed_values[0], "scenario.obstacle_speed_range[0]"),
        _finite_number(speed_values[1], "scenario.obstacle_speed_range[1]"),
    )
    if speed_range[0] < 0.0 or speed_range[0] > speed_range[1]:
        raise ValueError("scenario obstacle speeds must be non-negative and ordered.")
    horizontal_half_length = _finite_number(
        scenario_raw["horizontal_route_half_length"],
        "scenario.horizontal_route_half_length",
    )
    vertical_half_length = _finite_number(
        scenario_raw["vertical_route_half_length"],
        "scenario.vertical_route_half_length",
    )
    if horizontal_half_length <= 0.0 or vertical_half_length <= 0.0:
        raise ValueError("scenario route half lengths must be positive.")
    if scenario_raw["safety_shield"] is not False:
        raise ValueError("This checkpoint audit requires scenario.safety_shield=false.")
    if scenario_raw["deterministic_policy"] is not True:
        raise ValueError("This checkpoint audit requires deterministic_policy=true.")
    scenario = Scenario(
        obstacle_count=obstacle_count,
        obstacle_speed_range=speed_range,
        horizontal_route_half_length=horizontal_half_length,
        vertical_route_half_length=vertical_half_length,
        safety_shield=False,
        deterministic_policy=True,
    )

    selection = _mapping(raw["selection_basis"], "selection_basis")
    _exact_keys(
        selection,
        {
            "source_monitor",
            "source_monitor_sha256",
            "block_size_steps",
            "rule",
            "selected_checkpoint_step",
            "selected_block_success_rate",
            "audit_window_results_seen_when_selected",
        },
        "selection_basis",
    )
    monitor_portable, monitor_path = _inside_repository(
        root, selection["source_monitor"], "selection_basis.source_monitor"
    )
    monitor_sha256 = _digest(
        selection["source_monitor_sha256"], "selection_basis.source_monitor_sha256"
    )
    _verify_file(monitor_path, monitor_sha256, "Selection monitor")
    block_size = _integer(
        selection["block_size_steps"],
        "selection_basis.block_size_steps",
        minimum=1,
    )
    selected_step = _integer(
        selection["selected_checkpoint_step"],
        "selection_basis.selected_checkpoint_step",
        minimum=1,
    )
    if selected_step % block_size:
        raise ValueError("selected_checkpoint_step must end on a complete selection block.")
    selected_rate = _finite_number(
        selection["selected_block_success_rate"],
        "selection_basis.selected_block_success_rate",
    )
    if not 0.0 <= selected_rate <= 1.0:
        raise ValueError("selected_block_success_rate must be between zero and one.")
    if selection["rule"] != "highest_non_overlapping_training_block_success_rate":
        raise ValueError("selection_basis.rule is unsupported.")
    if selection["audit_window_results_seen_when_selected"] is not False:
        raise PermissionError(
            "Checkpoint selection must precede access to this audit window."
        )
    selection_copy = dict(selection)
    selection_copy["source_monitor"] = monitor_portable

    arms_raw = raw["arms"]
    if not isinstance(arms_raw, list) or len(arms_raw) != 2:
        raise ValueError("arms must contain exactly two mappings.")
    arms: list[Arm] = []
    for index, item in enumerate(arms_raw):
        arm_raw = _mapping(item, f"arms[{index}]")
        _exact_keys(
            arm_raw,
            {"id", "role", "model", "model_sha256", "model_variant", "run_name", "planner_label"},
            f"arms[{index}]",
        )
        role = _text(arm_raw["role"], f"arms[{index}].role")
        if role not in {"reference", "candidate"}:
            raise ValueError(f"arms[{index}].role must be reference or candidate.")
        model_portable, model_path = _inside_repository(
            root, arm_raw["model"], f"arms[{index}].model"
        )
        model_sha256 = _digest(arm_raw["model_sha256"], f"arms[{index}].model_sha256")
        if "latest" in model_path.stem.casefold():
            raise ValueError("Audit arms refuse mutable 'latest' model paths.")
        _verify_file(model_path, model_sha256, f"Audit model {role}")
        arm = Arm(
            arm_id=_safe_name(arm_raw["id"], f"arms[{index}].id"),
            role=role,
            model=model_portable,
            model_path=model_path,
            model_sha256=model_sha256,
            model_variant=_safe_name(arm_raw["model_variant"], f"arms[{index}].model_variant"),
            run_name=_safe_name(arm_raw["run_name"], f"arms[{index}].run_name"),
            planner_label=_safe_name(arm_raw["planner_label"], f"arms[{index}].planner_label"),
        )
        arms.append(arm)
    if {arm.role for arm in arms} != {"reference", "candidate"}:
        raise ValueError("arms must contain exactly one reference and one candidate.")
    for label, values in {
        "arm id": [arm.arm_id for arm in arms],
        "run_name": [arm.run_name for arm in arms],
        "planner_label": [arm.planner_label for arm in arms],
        "model": [arm.model for arm in arms],
    }.items():
        if len(set(values)) != 2:
            raise ValueError(f"Audit {label} values must be unique.")
    candidate = next(arm for arm in arms if arm.role == "candidate")
    match = re.search(r"_(\d+)_steps\.zip$", candidate.model_path.name)
    if match is None or int(match.group(1)) != selected_step:
        raise ValueError("Candidate model filename does not match selected_checkpoint_step.")

    gate_raw = _mapping(raw["candidate_gate"], "candidate_gate")
    gate_fields = {
        "require_all",
        "minimum_success_count",
        "maximum_collision_count",
        "minimum_horizontal_success_count",
        "minimum_vertical_success_count",
        "minimum_horizontal_paired_net_rescues",
        "maximum_vertical_paired_net_regressions",
    }
    _exact_keys(gate_raw, gate_fields, "candidate_gate")
    if gate_raw["require_all"] is not True:
        raise ValueError("candidate_gate.require_all must be true.")
    gate_values = {
        name: _integer(gate_raw[name], f"candidate_gate.{name}")
        for name in gate_fields - {"require_all"}
    }
    for name in (
        "minimum_success_count",
        "maximum_collision_count",
    ):
        if gate_values[name] > episodes:
            raise ValueError(f"candidate_gate.{name} cannot exceed episodes.")
    for name in (
        "minimum_horizontal_success_count",
        "minimum_vertical_success_count",
        "minimum_horizontal_paired_net_rescues",
        "maximum_vertical_paired_net_regressions",
    ):
        if gate_values[name] > episodes // 2:
            raise ValueError(
                f"candidate_gate.{name} cannot exceed the per-axis episode count."
            )
    gate = CandidateGate(require_all=True, **gate_values)

    decision = _mapping(raw["decision"], "decision")
    _exact_keys(decision, {"pass", "fail", "validation_or_test_access"}, "decision")
    decision_copy = {
        name: _safe_name(value, f"decision.{name}")
        for name, value in decision.items()
    }
    if decision_copy["validation_or_test_access"] != "forbidden":
        raise PermissionError("decision.validation_or_test_access must be forbidden.")

    return AuditConfig(
        source_path=source,
        source_portable_path=source_portable,
        source_sha256=_sha256_file(source),
        experiment_id=experiment_id,
        condition_id=condition_id,
        training_seed=training_seed,
        scenario_seed_start=seed_start,
        episodes=episodes,
        scenario=scenario,
        selection_basis=selection_copy,
        source_monitor_path=monitor_path,
        arms=(arms[0], arms[1]),
        gate=gate,
        decision=decision_copy,
    )


def _clean_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("WEBOTS_") and name != "WEBOTS_HOME":
            environment.pop(name)
    environment.update(overrides)
    return environment


def build_run_plan(
    config: AuditConfig,
    *,
    repository_root: Path,
    webots_executable: Path,
    port_base: int,
) -> tuple[AuditRun, AuditRun]:
    root = repository_root.resolve()
    world = (root / _WORLD).resolve()
    if not world.is_file():
        raise FileNotFoundError(f"Webots evaluation world does not exist: {world}")
    if not 1_024 <= port_base <= 65_500:
        raise ValueError("port_base must be between 1024 and 65500.")
    audit_directory = root / _AUDIT_DIRECTORY.format(experiment_id=config.experiment_id)
    destinations = [root / "artifacts/results" / arm.run_name for arm in config.arms]
    destinations.append(audit_directory)
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Audit refuses to overwrite existing artifact directories: "
            + ", ".join(str(path) for path in existing)
        )
    reserved: set[int] = set()
    runs: list[AuditRun] = []
    for arm in config.arms:
        port = choose_available_port(port_base, reserved)
        overrides = {
            "WEBOTS_PORT": str(port),
            "WEBOTS_MODEL_SEED": str(config.training_seed),
            "WEBOTS_EVAL_MODEL": arm.model,
            "WEBOTS_EVAL_MODEL_SHA256": arm.model_sha256,
            "WEBOTS_MODEL_VARIANT": arm.model_variant,
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
            "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": str(
                config.scenario.vertical_route_half_length
            ),
            "WEBOTS_SAFETY_SHIELD": "0",
            "WEBOTS_EVAL_DETERMINISTIC_POLICY": "1",
            "WEBOTS_EVAL_EPISODE_CONTEXT": "1",
            "WEBOTS_DIAGNOSTIC_AUDIT_ID": config.experiment_id,
            "WEBOTS_DIAGNOSTIC_AUDIT_CONFIG": config.source_portable_path,
            "WEBOTS_DIAGNOSTIC_AUDIT_CONFIG_SHA256": config.source_sha256,
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
            AuditRun(
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
    path: Path,
    *,
    config: AuditConfig,
    arm: Arm,
) -> dict[int, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _EPISODE_FIELDS:
            raise ValueError("episodes.csv schema does not match EpisodeRecord v1.0.")
        raw_rows = list(reader)
    if len(raw_rows) != config.episodes:
        raise ValueError(f"{arm.arm_id} episodes.csv must contain exactly {config.episodes} rows.")
    result: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        try:
            seed = int(raw["scenario_seed"])
        except ValueError as error:
            raise ValueError(f"episodes.csv row {index} has an invalid scenario_seed.") from error
        if seed in result:
            raise ValueError(f"episodes.csv contains duplicate scenario seed {seed}.")
        expected_id = f"diagnostic-{config.condition_id}-{seed}-{arm.planner_label}"
        expected = {
            "schema_version": "1.0",
            "episode_id": expected_id,
            "split": "diagnostic",
            "condition_id": config.condition_id,
            "planner": arm.planner_label,
        }
        for field, value in expected.items():
            if raw[field] != value:
                raise ValueError(f"episodes.csv row {index} {field} does not match the audit plan.")
        outcome = raw["outcome"]
        if outcome not in {"success", "collision", "timeout", "error"}:
            raise ValueError(f"episodes.csv row {index} has an invalid outcome.")
        success = _csv_bool(raw["is_success"], f"episodes.csv row {index} is_success")
        collision = _csv_bool(raw["collision"], f"episodes.csv row {index} collision")
        timeout = _csv_bool(raw["timeout"], f"episodes.csv row {index} timeout")
        if (outcome == "success") != success or (outcome == "collision") != collision or (
            outcome == "timeout"
        ) != timeout:
            raise ValueError(f"episodes.csv row {index} outcome flags are inconsistent.")
        result[seed] = {
            "episode_id": raw["episode_id"],
            "outcome": outcome,
            "is_success": success,
            "collision": collision,
            "timeout": timeout,
        }
    if set(result) != set(config.seeds):
        raise ValueError(f"{arm.arm_id} episodes.csv does not contain the exact audit seed set.")
    return result


def _read_contexts(
    path: Path,
    *,
    config: AuditConfig,
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
        raise ValueError(
            f"{arm.arm_id} episode_context.jsonl must contain exactly {config.episodes} rows."
        )
    contexts: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(records):
        record = _mapping(value, f"episode_context row {index}")
        if record.get("schema_version") != "1.0":
            raise ValueError(f"episode_context row {index} has an invalid schema_version.")
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
            raise ValueError(f"episode_context row {index} identity does not match episodes.csv.")
        axis = record.get("route_axis")
        expected_axis = _axis_for_seed(seed)
        if axis != expected_axis:
            raise ValueError(
                f"episode_context seed {seed} route_axis {axis!r} != expected {expected_axis!r}."
            )
        if not isinstance(record.get("termination"), dict):
            raise ValueError(f"episode_context row {index} termination must be an object.")
        contexts[seed] = record
    if set(contexts) != set(config.seeds):
        raise ValueError("episode_context.jsonl does not contain the exact audit seed set.")
    counts = Counter(record["route_axis"] for record in contexts.values())
    if counts != {"horizontal": config.episodes // 2, "vertical": config.episodes // 2}:
        raise ValueError(f"Actual diagnostic contexts are not axis-balanced: {dict(counts)}.")
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

    all_seeds = sorted(rows)
    horizontal = [seed for seed in all_seeds if contexts[seed]["route_axis"] == "horizontal"]
    vertical = [seed for seed in all_seeds if contexts[seed]["route_axis"] == "vertical"]
    return {
        "overall": counts(all_seeds),
        "by_route_axis": {
            "horizontal": counts(horizontal),
            "vertical": counts(vertical),
        },
    }


def _check_summary(
    summary_path: Path,
    *,
    repository_root: Path,
    config: AuditConfig,
    arm: Arm,
    episode_rows: Mapping[int, Mapping[str, Any]],
    episodes_sha256: str,
    context_path: Path,
    context_sha256: str,
) -> None:
    summary = _mapping(json.loads(summary_path.read_text(encoding="utf-8")), "summary.json")
    expected_top = {
        "split": "diagnostic",
        "episode_count": config.episodes,
        "planners": [arm.planner_label],
        "conditions": [config.condition_id],
    }
    for name, value in expected_top.items():
        if summary.get(name) != value:
            raise ValueError(f"{arm.arm_id} summary.json {name} does not match the audit plan.")
    by_planner = summary.get("by_planner")
    if not isinstance(by_planner, list) or len(by_planner) != 1:
        raise ValueError("summary.json must contain exactly one planner aggregate.")
    aggregate = _mapping(by_planner[0], "summary.by_planner[0]")
    expected_counts = _count_metrics(
        episode_rows,
        {seed: {"route_axis": _axis_for_seed(seed)} for seed in episode_rows},
    )["overall"]
    for name in ("episodes", "success_count", "collision_count", "timeout_count", "error_count"):
        if aggregate.get(name) != expected_counts[name]:
            raise ValueError(f"summary aggregate {name} disagrees with episodes.csv.")
    provenance = _mapping(summary.get("evaluation_provenance"), "evaluation_provenance")
    expected_provenance = {
        "schema_version": 1,
        "model_path": arm.model,
        "model_sha256": arm.model_sha256,
        "model_variant": arm.model_variant,
        "training_seed": config.training_seed,
        "split": "diagnostic",
        "condition_id": config.condition_id,
        "planner": "ppo",
        "planner_label": arm.planner_label,
        "scenario_seed_start": config.scenario_seed_start,
        "scenario_seed_count": config.episodes,
        "scenario_seeds": list(config.seeds),
        "obstacle_count": config.scenario.obstacle_count,
        "obstacle_speed_range": list(config.scenario.obstacle_speed_range),
        "horizontal_route_half_length": config.scenario.horizontal_route_half_length,
        "vertical_route_half_length": config.scenario.vertical_route_half_length,
        "safety_shield_enabled": False,
        "deterministic_policy": True,
        "episodes_csv_sha256": episodes_sha256,
    }
    for name, value in expected_provenance.items():
        if provenance.get(name) != value:
            raise ValueError(
                f"{arm.arm_id} evaluation_provenance.{name} does not match the audit plan."
            )
    context_binding = _mapping(provenance.get("episode_context"), "episode_context binding")
    expected_context_path = context_path.resolve().relative_to(
        repository_root.resolve()
    ).as_posix()
    expected_context = {
        "path": expected_context_path,
        "schema_version": "1.0",
        "count": config.episodes,
        "sha256": context_sha256,
    }
    if context_binding != expected_context:
        raise ValueError("summary episode_context binding does not match the context artifact.")
    reset_health = _mapping(provenance.get("reset_health"), "evaluation_provenance.reset_health")
    for name, value in {
        "attempts": config.episodes,
        "successful": config.episodes,
        "finite_settle_checks": 2 * config.episodes,
        "readback_checks": config.episodes,
    }.items():
        if reset_health.get(name) != value:
            raise ValueError(f"reset_health.{name} does not match the completed audit run.")


def _parse_arm_result(
    run: AuditRun,
    *,
    config: AuditConfig,
    repository_root: Path,
) -> ParsedArmResult:
    summary_path = run.result_directory / "summary.json"
    episodes_path = run.result_directory / "episodes.csv"
    context_path = run.result_directory / "episode_context.jsonl"
    for path in (summary_path, episodes_path, context_path):
        if not path.is_file():
            raise FileNotFoundError(f"Audit arm {run.arm.arm_id} omitted required artifact: {path}")
    episodes_sha256 = _sha256_file(episodes_path)
    context_sha256 = _sha256_file(context_path)
    rows = _read_episode_rows(episodes_path, config=config, arm=run.arm)
    contexts = _read_contexts(
        context_path,
        config=config,
        arm=run.arm,
        episode_rows=rows,
    )
    _check_summary(
        summary_path,
        repository_root=repository_root,
        config=config,
        arm=run.arm,
        episode_rows=rows,
        episodes_sha256=episodes_sha256,
        context_path=context_path,
        context_sha256=context_sha256,
    )
    binding = {
        "result_directory": run.result_directory.resolve().relative_to(
            repository_root.resolve()
        ).as_posix(),
        "summary_json": {
            "path": summary_path.resolve().relative_to(repository_root.resolve()).as_posix(),
            "sha256": _sha256_file(summary_path),
        },
        "episodes_csv": {
            "path": episodes_path.resolve().relative_to(repository_root.resolve()).as_posix(),
            "sha256": episodes_sha256,
        },
        "episode_context_jsonl": {
            "path": context_path.resolve().relative_to(repository_root.resolve()).as_posix(),
            "sha256": context_sha256,
        },
    }
    return ParsedArmResult(
        arm=run.arm,
        rows_by_seed=rows,
        contexts_by_seed=contexts,
        metrics=_count_metrics(rows, contexts),
        artifact_bindings=binding,
    )


def _paired_analysis(
    reference: ParsedArmResult,
    candidate: ParsedArmResult,
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    transition_counts: dict[str, Counter[str]] = {
        "overall": Counter(),
        "horizontal": Counter(),
        "vertical": Counter(),
    }
    paired_counts = {
        axis: {"rescues": 0, "regressions": 0}
        for axis in ("overall", "horizontal", "vertical")
    }
    for seed in seeds:
        axis = str(reference.contexts_by_seed[seed]["route_axis"])
        candidate_axis = candidate.contexts_by_seed[seed]["route_axis"]
        if candidate_axis != axis:
            raise ValueError(f"Paired arms disagree on route_axis for seed {seed}.")
        for field in (
            "route_axis",
            "reverse_route",
            "start_xy",
            "goal_xy",
            "start_yaw",
            "dynamic_obstacles",
        ):
            if reference.contexts_by_seed[seed].get(field) != candidate.contexts_by_seed[
                seed
            ].get(field):
                raise ValueError(
                    f"Paired arms disagree on initial context {field} for seed {seed}."
                )
        reference_outcome = str(reference.rows_by_seed[seed]["outcome"])
        candidate_outcome = str(candidate.rows_by_seed[seed]["outcome"])
        transition = f"{reference_outcome}_to_{candidate_outcome}"
        transition_counts["overall"][transition] += 1
        transition_counts[axis][transition] += 1
        rescue = reference_outcome != "success" and candidate_outcome == "success"
        regression = reference_outcome == "success" and candidate_outcome != "success"
        for key in ("overall", axis):
            paired_counts[key]["rescues"] += int(rescue)
            paired_counts[key]["regressions"] += int(regression)
        rows.append(
            {
                "scenario_seed": seed,
                "route_axis": axis,
                "reference_outcome": reference_outcome,
                "candidate_outcome": candidate_outcome,
                "transition": transition,
            }
        )
    for value in paired_counts.values():
        value["net_rescues"] = value["rescues"] - value["regressions"]
        value["net_regressions"] = value["regressions"] - value["rescues"]
    return {
        "by_seed": rows,
        "transition_counts": {
            key: dict(sorted(value.items())) for key, value in transition_counts.items()
        },
        "paired_success_changes": paired_counts,
    }


def _gate_report(
    gate: CandidateGate,
    *,
    candidate_metrics: Mapping[str, Any],
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    overall = candidate_metrics["overall"]
    by_axis = candidate_metrics["by_route_axis"]
    changes = paired["paired_success_changes"]
    checks = {
        "minimum_success_count": {
            "actual": overall["success_count"],
            "operator": ">=",
            "threshold": gate.minimum_success_count,
        },
        "maximum_collision_count": {
            "actual": overall["collision_count"],
            "operator": "<=",
            "threshold": gate.maximum_collision_count,
        },
        "minimum_horizontal_success_count": {
            "actual": by_axis["horizontal"]["success_count"],
            "operator": ">=",
            "threshold": gate.minimum_horizontal_success_count,
        },
        "minimum_vertical_success_count": {
            "actual": by_axis["vertical"]["success_count"],
            "operator": ">=",
            "threshold": gate.minimum_vertical_success_count,
        },
        "minimum_horizontal_paired_net_rescues": {
            "actual": changes["horizontal"]["net_rescues"],
            "operator": ">=",
            "threshold": gate.minimum_horizontal_paired_net_rescues,
        },
        "maximum_vertical_paired_net_regressions": {
            "actual": changes["vertical"]["net_regressions"],
            "operator": "<=",
            "threshold": gate.maximum_vertical_paired_net_regressions,
        },
    }
    for value in checks.values():
        value["passed"] = (
            value["actual"] >= value["threshold"]
            if value["operator"] == ">="
            else value["actual"] <= value["threshold"]
        )
    passed = all(value["passed"] for value in checks.values())
    return {"require_all": gate.require_all, "checks": checks, "passed": passed}


def _write_report(
    report: Mapping[str, Any],
    *,
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=False)
    json_path = output_directory / "audit_report.json"
    yaml_path = output_directory / "audit_report.yaml"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    yaml_path.write_text(
        yaml.safe_dump(dict(report), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    digest_path = output_directory / "audit_report.sha256"
    digest_path.write_text(
        f"{_sha256_file(json_path)}  {json_path.name}\n"
        f"{_sha256_file(yaml_path)}  {yaml_path.name}\n",
        encoding="utf-8",
    )
    return json_path, yaml_path, digest_path


def execute_audit(
    *,
    repository_root: Path,
    config_path: Path,
    webots_executable: Path,
    dry_run: bool,
    port_base: int = 13_620,
    timeout_seconds: float = 3_600.0,
    process_runner: ProcessRunner = subprocess.run,
    source_verifier: SourceVerifier = require_clean_tracked_source,
) -> tuple[int, Mapping[str, Any]]:
    """Preflight, optionally execute, and return an auditable decision mapping."""

    root = repository_root.resolve()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive.")
    source_revision = source_verifier(root)
    config = load_audit_config(config_path, repository_root=root)
    webots = resolve_webots_executable(webots_executable)
    runs = build_run_plan(
        config,
        repository_root=root,
        webots_executable=webots,
        port_base=port_base,
    )
    plan = {
        "schema_version": 1,
        "dry_run": dry_run,
        "experiment_id": config.experiment_id,
        "source_revision": source_revision,
        "split": "diagnostic",
        "config": {"path": config.source_portable_path, "sha256": config.source_sha256},
        "scenario_seeds": list(config.seeds),
        "axis_counts": dict(Counter(_axis_for_seed(seed) for seed in config.seeds)),
        "validation_or_test_access": "forbidden",
        "runs": [
            {
                "arm_id": run.arm.arm_id,
                "role": run.arm.role,
                "model": run.arm.model,
                "model_sha256": run.arm.model_sha256,
                "run_name": run.arm.run_name,
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
        if run.result_directory.exists():
            raise FileExistsError(f"Audit refuses to overwrite {run.result_directory}.")
        _verify_file(run.arm.model_path, run.arm.model_sha256, f"Audit model {run.arm.arm_id}")
        completed = process_runner(
            run.command,
            cwd=root,
            env=_clean_environment(run.environment),
            check=False,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Webots audit arm {run.arm.arm_id} failed with exit code "
                f"{completed.returncode}."
            )
        _verify_file(run.arm.model_path, run.arm.model_sha256, f"Audit model {run.arm.arm_id}")
        parsed.append(_parse_arm_result(run, config=config, repository_root=root))

    _verify_file(config.source_path, config.source_sha256, "Audit config")
    if source_verifier(root) != source_revision:
        raise RuntimeError("Git HEAD changed while the diagnostic audit was running.")
    _verify_file(
        config.source_monitor_path,
        str(config.selection_basis["source_monitor_sha256"]),
        "Selection monitor",
    )
    reference = next(result for result in parsed if result.arm.role == "reference")
    candidate = next(result for result in parsed if result.arm.role == "candidate")
    paired = _paired_analysis(reference, candidate, seeds=config.seeds)
    gate = _gate_report(config.gate, candidate_metrics=candidate.metrics, paired=paired)
    passed = bool(gate["passed"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "passed" if passed else "gate_failed",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_revision": source_revision,
        "split": "diagnostic",
        "validation_split_was_run": False,
        "test_split_was_run": False,
        "config": {"path": config.source_portable_path, "sha256": config.source_sha256},
        "selection_basis": dict(config.selection_basis),
        "scenario": {
            "condition_id": config.condition_id,
            "scenario_seed_start": config.scenario_seed_start,
            "episodes": config.episodes,
            "scenario_seeds": list(config.seeds),
            "axis_counts": dict(Counter(_axis_for_seed(seed) for seed in config.seeds)),
            "obstacle_count": config.scenario.obstacle_count,
            "obstacle_speed_range": list(config.scenario.obstacle_speed_range),
            "horizontal_route_half_length": config.scenario.horizontal_route_half_length,
            "vertical_route_half_length": config.scenario.vertical_route_half_length,
            "safety_shield": False,
            "deterministic_policy": True,
        },
        "arms": [
            {
                "id": result.arm.arm_id,
                "role": result.arm.role,
                "model": {
                    "path": result.arm.model,
                    "sha256": result.arm.model_sha256,
                    "variant": result.arm.model_variant,
                    "training_seed": config.training_seed,
                },
                "run_name": result.arm.run_name,
                "planner_label": result.arm.planner_label,
                "execution": {
                    "command": list(run.command),
                    "environment": dict(run.environment),
                    "port": run.port,
                    "exit_code": 0,
                },
                "metrics": dict(result.metrics),
                "artifacts": dict(result.artifact_bindings),
            }
            for result, run in zip(parsed, runs, strict=True)
        ],
        "paired_analysis": paired,
        "candidate_gate": gate,
        "decision": {
            "selected": config.decision["pass"] if passed else config.decision["fail"],
            "validation_or_test_access": "forbidden",
        },
        "written_reports": {
            "json": (
                _AUDIT_DIRECTORY.format(experiment_id=config.experiment_id)
                + "/audit_report.json"
            ),
            "yaml": (
                _AUDIT_DIRECTORY.format(experiment_id=config.experiment_id)
                + "/audit_report.yaml"
            ),
            "sha256_sidecar": (
                _AUDIT_DIRECTORY.format(experiment_id=config.experiment_id)
                + "/audit_report.sha256"
            ),
        },
    }
    output_directory = root / _AUDIT_DIRECTORY.format(experiment_id=config.experiment_id)
    _write_report(report, output_directory=output_directory)
    return (0 if passed else EXIT_CANDIDATE_GATE_FAILED), report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/webots_h1_checkpoint_audit_v1.yaml"),
    )
    parser.add_argument("--webots", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=13_620)
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).absolute().parents[1]
    try:
        exit_code, result = execute_audit(
            repository_root=root,
            config_path=args.config,
            webots_executable=args.webots,
            dry_run=args.dry_run,
            port_base=args.port_base,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as error:
        print(f"WEBOTS_DIAGNOSTIC_AUDIT_FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_PREFLIGHT_OR_RUNTIME_FAILED
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("WEBOTS_DIAGNOSTIC_AUDIT_DRY_RUN_OK", file=sys.stderr)
    else:
        print(
            "WEBOTS_DIAGNOSTIC_AUDIT_OK"
            if exit_code == 0
            else "WEBOTS_DIAGNOSTIC_AUDIT_GATE_FAILED",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
