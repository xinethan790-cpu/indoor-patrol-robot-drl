#!/usr/bin/env python3
"""Run a sequential, identity-bound four-point Webots patrol campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from indoor_patrol_rl.application import (
    DEFAULT_OPERATIONAL_PATROL_SCENARIO_PATH,
    DEFAULT_STRESS_PATROL_SCENARIO_PATH,
    PATROL_TTC_ESCAPE_GUARD_ID,
    UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE,
    PatrolScenarioConfig,
    load_patrol_scenario_config,
)

CAMPAIGN_SCHEMA_VERSION = "patrol-campaign/v1"
RUN_SCHEMA_VERSION = "1.0"
RUN_BACKEND = "Webots R2025a"
RUN_PLANNER = "astar_local_waypoint_ppo"
RUN_MODEL_TRAINING_DOMAIN = "webots"
MINIMUM_ACCEPTANCE_EPISODES = 20
MINIMUM_COMPLETED_RATE = 0.75
MAXIMUM_COLLISION_RATE = 0.15
REQUIRED_ERROR_RATE = 0.0
ACCEPTANCE_FAILURE_EXIT_CODE = 5
SCENARIO_CONFIGS = {
    "operational": DEFAULT_OPERATIONAL_PATROL_SCENARIO_PATH,
    "stress": DEFAULT_STRESS_PATROL_SCENARIO_PATH,
    "hotel": Path("configs/patrol/scenario_hotel_corridor_v1.yaml"),
}
CRITICAL_SOURCE_PATHS = {
    "single_run_launcher": "scripts/run_webots_patrol.py",
    "patrol_controller": "controllers/patrol_route_ppo/patrol_route_ppo.py",
    "patrol_world": "worlds/patrol_route_ppo.wbt",
    "patrol_map": "configs/maps/patrol_route.yaml",
    "proximity_monitor": "src/indoor_patrol_rl/application/proximity.py",
    "ttc_escape_guard": "src/indoor_patrol_rl/application/emergency_guard.py",
    "patrol_evaluation": "src/indoor_patrol_rl/evaluation/patrol.py",
    "inspection_config": "configs/inspection/environmental_patrol_v1.yaml",
}
ACCEPTED_EXIT_CODES = {0, 2, 3, 4}
EXIT_CODE_BY_OUTCOME = {
    "completed": 0,
    "collision": 2,
    "timeout": 3,
    "safety_stop": 4,
}
OUTCOMES = ("completed", "collision", "safety_stop", "timeout", "error")
EXECUTION_IDENTITY_FIELDS = (
    "schema_version",
    "backend",
    "planner",
    "model_path",
    "model_sha256",
    "model_training_domain",
    "system_deployment_id",
    "system_deployment_profile_sha256",
    "patrol_scenario_id",
    "patrol_scenario_mode",
    "patrol_scenario_config_sha256",
    "deployment_evidence_scope",
    "patrol_scenario_qualified",
    "safety_shield_enabled",
    "safety_shield_id",
    "safety_shield_sha256",
    "ttc_escape_guard_enabled",
    "ttc_escape_guard_id",
    "ttc_escape_guard_evidence_scope",
    "dynamic_obstacle_count",
    "dynamic_obstacle_speed_max_mps",
)
EPISODE_FIELDS = (
    "episode_index",
    "scenario_seed",
    "run_id",
    "outcome",
    "exit_code",
    "failure_reason",
    "checkpoints_visited",
    "checkpoints_total",
    "completion_ratio",
    "collision_count",
    "critical_proximity_count",
    "control_step_count",
    "elapsed_simulation_time_s",
    "actual_path_length_m",
    "minimum_clearance_m",
    *EXECUTION_IDENTITY_FIELDS,
    "trajectory_file",
    "trajectory_row_count",
    "trajectory_sha256",
    "checkpoints_file",
    "checkpoints_row_count",
    "checkpoints_sha256",
    "collisions_file",
    "collisions_row_count",
    "collisions_sha256",
    "proximity_events_file",
    "proximity_events_row_count",
    "proximity_events_sha256",
    "ttc_escape_guard_events_file",
    "ttc_escape_guard_events_row_count",
    "ttc_escape_guard_events_sha256",
    "run_summary_path",
    "run_summary_sha256",
)

_PORTABLE_CSV_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.csv")
_AUDIT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class PatrolCampaignError(RuntimeError):
    """One sub-run or its evidence invalidated the complete campaign."""


@dataclass(frozen=True, slots=True)
class PatrolCampaignRequest:
    """Validated inputs for one sequential patrol acceptance campaign."""

    repository_root: Path
    scenario: str
    seed_start: int
    episodes: int
    output_directory: Path
    port: int
    episode_wall_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        repository_root = Path(self.repository_root).expanduser().resolve()
        if not repository_root.is_dir():
            raise FileNotFoundError(f"Repository root does not exist: {repository_root}")
        if self.scenario not in SCENARIO_CONFIGS:
            raise ValueError(f"scenario must be one of {sorted(SCENARIO_CONFIGS)}.")
        seed_start = _integer(self.seed_start, "seed_start", minimum=0)
        episodes = _integer(self.episodes, "episodes", minimum=1)
        port = _integer(self.port, "port", minimum=1)
        if port > 65535:
            raise ValueError("port must be between 1 and 65535.")
        episode_wall_timeout_s = _finite_number(
            self.episode_wall_timeout_s,
            "episode_wall_timeout_s",
            minimum=0.0,
        )
        if episode_wall_timeout_s == 0.0:
            raise ValueError("episode_wall_timeout_s must be positive.")
        output_directory = Path(self.output_directory).expanduser()
        if not output_directory.is_absolute():
            output_directory = repository_root / output_directory
        output_directory = output_directory.resolve()
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "seed_start", seed_start)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "episode_wall_timeout_s", episode_wall_timeout_s)
        object.__setattr__(self, "output_directory", output_directory)


ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIO_CONFIGS),
        default="operational",
    )
    parser.add_argument("--seed-start", type=int, default=7000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New campaign directory; existing paths are never reused.",
    )
    parser.add_argument("--port", type=int, default=1262)
    parser.add_argument(
        "--episode-wall-timeout-s",
        "--wall-timeout-s",
        "--episode-timeout-s",
        type=float,
        default=600.0,
        help="Maximum wall-clock seconds allowed for each Webots sub-run.",
    )
    return parser


def run_patrol_campaign(
    request: PatrolCampaignRequest,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> dict[str, Any]:
    """Run every requested seed, then write canonical CSV and JSON aggregates."""

    if not isinstance(request, PatrolCampaignRequest):
        raise TypeError("request must be a PatrolCampaignRequest value.")
    output_directory = request.output_directory
    if output_directory.exists():
        raise FileExistsError(
            "Patrol campaign output directory already exists; choose a new directory: "
            f"{output_directory}"
        )

    scenario = load_patrol_scenario_config(
        SCENARIO_CONFIGS[request.scenario],
        repository_root=request.repository_root,
    )
    if scenario.mode.value != request.scenario:
        raise PatrolCampaignError(
            "Scenario selector and loaded scenario mode disagree: "
            f"{request.scenario!r} != {scenario.mode.value!r}."
        )
    critical_sources = _bind_critical_sources(request.repository_root, scenario)
    runner_path = request.repository_root / CRITICAL_SOURCE_PATHS["single_run_launcher"]

    output_directory.mkdir(parents=True, exist_ok=False)
    runs_directory = output_directory / "runs"
    runs_directory.mkdir()
    episodes: list[dict[str, Any]] = []
    expected_execution_identity: dict[str, Any] | None = None

    for episode_index in range(request.episodes):
        _verify_critical_sources(request.repository_root, critical_sources)
        scenario.verify_source_integrity()
        seed = request.seed_start + episode_index
        run_id = f"patrol-{request.scenario}-seed{seed}"
        run_directory = runs_directory / f"seed-{seed}"
        command = [
            sys.executable,
            str(runner_path),
            "--headless",
            "--scenario",
            request.scenario,
            "--seed",
            str(seed),
            "--run-id",
            run_id,
            "--output-dir",
            str(run_directory),
            "--port",
            str(request.port),
        ]
        try:
            completed = process_runner(
                command,
                cwd=request.repository_root,
                check=False,
                timeout=request.episode_wall_timeout_s,
            )
        except subprocess.TimeoutExpired as error:
            _verify_critical_sources(request.repository_root, critical_sources)
            raise PatrolCampaignError(
                f"Patrol seed {seed} exceeded its {request.episode_wall_timeout_s:g}s "
                "wall-clock timeout."
            ) from error
        except OSError as error:
            _verify_critical_sources(request.repository_root, critical_sources)
            raise PatrolCampaignError(
                f"Patrol seed {seed} could not be launched: {error}"
            ) from error
        _verify_critical_sources(request.repository_root, critical_sources)

        exit_code = _integer(completed.returncode, "sub-run return code")
        if exit_code not in ACCEPTED_EXIT_CODES:
            raise PatrolCampaignError(
                f"Patrol seed {seed} returned campaign-error exit code {exit_code}."
            )

        summary_path = run_directory / "summary.json"
        if not summary_path.is_file():
            raise PatrolCampaignError(
                f"Patrol seed {seed} returned {exit_code} without summary.json."
            )
        try:
            raw_summary = _load_json_object(summary_path)
            episode = _validated_episode(
                raw_summary,
                episode_index=episode_index,
                expected_seed=seed,
                expected_run_id=run_id,
                exit_code=exit_code,
                scenario=scenario,
                repository_root=request.repository_root,
                output_directory=output_directory,
                run_directory=run_directory,
                summary_path=summary_path,
            )
        except (OSError, TypeError, ValueError) as error:
            raise PatrolCampaignError(
                f"Patrol seed {seed} summary evidence is invalid: {error}"
            ) from error

        execution_identity = _execution_identity(episode)
        if expected_execution_identity is None:
            expected_execution_identity = execution_identity
        elif execution_identity != expected_execution_identity:
            changed = [
                field
                for field in EXECUTION_IDENTITY_FIELDS
                if execution_identity[field] != expected_execution_identity[field]
            ]
            details = ", ".join(
                f"{field}: {expected_execution_identity[field]!r} -> {execution_identity[field]!r}"
                for field in changed
            )
            raise PatrolCampaignError(
                f"Patrol execution identity changed at seed {seed}: {details}."
            )
        episodes.append(episode)

    _verify_critical_sources(request.repository_root, critical_sources)
    scenario.verify_source_integrity()
    if expected_execution_identity is None:
        raise PatrolCampaignError("Patrol campaign produced no execution identity.")
    model_path = request.repository_root / expected_execution_identity["model_path"]
    if _sha256_file(model_path) != expected_execution_identity["model_sha256"]:
        raise PatrolCampaignError("Patrol model source changed before campaign finalization.")

    episodes_path = output_directory / "episodes.csv"
    _write_episodes(episodes_path, episodes)
    payload = _campaign_summary(
        request,
        scenario=scenario,
        episodes=episodes,
        execution_identity=expected_execution_identity,
        critical_sources=critical_sources,
        episodes_sha256=_sha256_file(episodes_path),
    )
    summary_path = output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _validated_episode(
    summary: Mapping[str, Any],
    *,
    episode_index: int,
    expected_seed: int,
    expected_run_id: str,
    exit_code: int,
    scenario: PatrolScenarioConfig,
    repository_root: Path,
    output_directory: Path,
    run_directory: Path,
    summary_path: Path,
) -> dict[str, Any]:
    schema_version = _exact_text(
        _required(summary, "schema_version"),
        "schema_version",
        RUN_SCHEMA_VERSION,
    )
    backend = _exact_text(_required(summary, "backend"), "backend", RUN_BACKEND)
    planner = _exact_text(_required(summary, "planner"), "planner", RUN_PLANNER)
    seed = _integer(_required(summary, "scenario_seed"), "scenario_seed", minimum=0)
    if seed != expected_seed:
        raise ValueError(f"scenario_seed must be {expected_seed}, got {seed}.")
    run_id = _text(_required(summary, "run_id"), "run_id")
    if run_id != expected_run_id:
        raise ValueError(f"run_id must be {expected_run_id!r}, got {run_id!r}.")
    outcome = _text(_required(summary, "outcome"), "outcome")
    if outcome not in OUTCOMES:
        raise ValueError(f"Unsupported patrol outcome: {outcome!r}.")
    expected_exit_code = EXIT_CODE_BY_OUTCOME.get(outcome)
    if expected_exit_code != exit_code:
        raise ValueError(
            f"Outcome {outcome!r} requires exit code {expected_exit_code}, got {exit_code}."
        )

    model_path = _portable_repository_file(
        _required(summary, "model_path"),
        label="model_path",
        repository_root=repository_root,
    )
    model_sha256 = _sha256(_required(summary, "model_sha256"), "model_sha256")
    actual_model_sha256 = _sha256_file(repository_root / model_path)
    if model_sha256 != actual_model_sha256:
        raise ValueError(
            f"model_sha256 does not match {model_path}: {model_sha256} != {actual_model_sha256}."
        )
    model_training_domain = _exact_text(
        _required(summary, "model_training_domain"),
        "model_training_domain",
        RUN_MODEL_TRAINING_DOMAIN,
    )
    system_deployment_id = _audit_identity(
        _required(summary, "system_deployment_id"),
        "system_deployment_id",
    )
    system_deployment_profile_sha256 = _sha256(
        _required(summary, "system_deployment_profile_sha256"),
        "system_deployment_profile_sha256",
    )

    scenario_id = _text(_required(summary, "patrol_scenario_id"), "patrol_scenario_id")
    scenario_mode = _text(
        _required(summary, "patrol_scenario_mode"),
        "patrol_scenario_mode",
    )
    scenario_sha256 = _sha256(
        _required(summary, "patrol_scenario_config_sha256"),
        "patrol_scenario_config_sha256",
    )
    if scenario_id != scenario.scenario_id:
        raise ValueError(f"patrol_scenario_id differs from the selected scenario: {scenario_id!r}.")
    if scenario_mode != scenario.mode.value:
        raise ValueError(
            f"patrol_scenario_mode differs from the selected scenario: {scenario_mode!r}."
        )
    if scenario_sha256 != scenario.source_sha256:
        raise ValueError("patrol_scenario_config_sha256 differs from the selected scenario.")
    scenario_qualified = _exact_bool(
        _required(summary, "patrol_scenario_qualified"),
        "patrol_scenario_qualified",
        False,
    )
    deployment_evidence_scope = _audit_identity(
        _required(summary, "deployment_evidence_scope"),
        "deployment_evidence_scope",
    )
    if deployment_evidence_scope != scenario.deployment_evidence_scope:
        raise ValueError(
            "deployment_evidence_scope differs from the selected scenario's "
            "unqualified evidence scope."
        )

    safety_shield_enabled = _exact_bool(
        _required(summary, "safety_shield_enabled"),
        "safety_shield_enabled",
        True,
    )
    safety_shield_id = _audit_identity(
        _required(summary, "safety_shield_id"),
        "safety_shield_id",
    )
    safety_shield_sha256 = _sha256(
        _required(summary, "safety_shield_sha256"),
        "safety_shield_sha256",
    )
    guard_enabled = _exact_bool(
        _required(summary, "ttc_escape_guard_enabled"),
        "ttc_escape_guard_enabled",
        True,
    )
    guard_id = _exact_text(
        _required(summary, "ttc_escape_guard_id"),
        "ttc_escape_guard_id",
        PATROL_TTC_ESCAPE_GUARD_ID,
    )
    guard_scope = _exact_text(
        _required(summary, "ttc_escape_guard_evidence_scope"),
        "ttc_escape_guard_evidence_scope",
        UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE,
    )

    dynamic_obstacle_count = _integer(
        _required(summary, "dynamic_obstacle_count"),
        "dynamic_obstacle_count",
        minimum=0,
    )
    if dynamic_obstacle_count != len(scenario.dynamic_obstacles):
        raise ValueError(
            "dynamic_obstacle_count differs from the selected scenario: "
            f"{dynamic_obstacle_count} != {len(scenario.dynamic_obstacles)}."
        )
    dynamic_obstacle_speed_max_mps = _finite_number(
        _required(summary, "dynamic_obstacle_speed_max_mps"),
        "dynamic_obstacle_speed_max_mps",
        minimum=0.0,
    )
    expected_maximum_speed = max(obstacle.speed_mps for obstacle in scenario.dynamic_obstacles)
    if not math.isclose(
        dynamic_obstacle_speed_max_mps,
        expected_maximum_speed,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "dynamic_obstacle_speed_max_mps differs from the selected scenario: "
            f"{dynamic_obstacle_speed_max_mps} != {expected_maximum_speed}."
        )

    checkpoints_total = _integer(
        _required(summary, "checkpoints_total"),
        "checkpoints_total",
        minimum=1,
    )
    checkpoints_visited = _integer(
        _required(summary, "checkpoints_visited"),
        "checkpoints_visited",
        minimum=0,
    )
    if checkpoints_visited > checkpoints_total:
        raise ValueError("checkpoints_visited cannot exceed checkpoints_total.")
    completion_ratio = _finite_number(
        _required(summary, "completion_ratio"),
        "completion_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    expected_ratio = checkpoints_visited / checkpoints_total
    if not math.isclose(completion_ratio, expected_ratio, abs_tol=1e-9):
        raise ValueError("completion_ratio must equal checkpoints_visited / checkpoints_total.")
    if outcome == "completed" and completion_ratio != 1.0:
        raise ValueError("A completed patrol must visit every checkpoint.")

    collision_count = _integer(
        _required(summary, "collision_count"),
        "collision_count",
        minimum=0,
    )
    critical_proximity_count = _integer(
        _required(summary, "critical_proximity_count"),
        "critical_proximity_count",
        minimum=0,
    )
    if outcome == "collision":
        if collision_count < 1:
            raise ValueError("A collision run requires collision evidence.")
    elif outcome == "safety_stop":
        if collision_count != 0 or critical_proximity_count < 1:
            raise ValueError("A safety_stop run requires proximity evidence only.")
    elif collision_count != 0:
        raise ValueError("Only a collision outcome may carry collision evidence.")
    control_step_count = _integer(
        _required(summary, "control_step_count"),
        "control_step_count",
        minimum=0,
    )
    elapsed = _finite_number(
        _required(summary, "elapsed_simulation_time_s"),
        "elapsed_simulation_time_s",
        minimum=0.0,
    )
    path_length = _finite_number(
        _required(summary, "actual_path_length_m"),
        "actual_path_length_m",
        minimum=0.0,
    )
    minimum_clearance = _finite_number(
        _required(summary, "minimum_clearance_m"),
        "minimum_clearance_m",
        minimum=0.0,
    )
    failure_reason = summary.get("failure_reason")
    if failure_reason is not None:
        failure_reason = _text(failure_reason, "failure_reason")
    if outcome == "completed" and failure_reason is not None:
        raise ValueError("A completed patrol cannot carry failure_reason.")

    artifacts: dict[str, Any] = {}
    for prefix, summary_field, expected_rows in (
        ("trajectory", "trajectory_file", control_step_count),
        ("checkpoints", "checkpoints_file", checkpoints_total),
        ("collisions", "collisions_file", collision_count),
        (
            "proximity_events",
            "proximity_events_file",
            critical_proximity_count,
        ),
        ("ttc_escape_guard_events", "ttc_escape_guard_events_file", None),
    ):
        artifacts.update(
            _validated_csv_artifact(
                summary,
                summary_field=summary_field,
                prefix=prefix,
                run_directory=run_directory,
                expected_rows=expected_rows,
            )
        )
    artifact_basenames = [
        artifacts[f"{prefix}_file"]
        for prefix in (
            "trajectory",
            "checkpoints",
            "collisions",
            "proximity_events",
            "ttc_escape_guard_events",
        )
    ]
    if len(set(artifact_basenames)) != len(artifact_basenames):
        raise ValueError("Patrol CSV artifact basenames must be unique within a run.")

    summary_relative = summary_path.resolve().relative_to(output_directory).as_posix()
    return {
        "episode_index": episode_index,
        "scenario_seed": seed,
        "run_id": run_id,
        "outcome": outcome,
        "exit_code": exit_code,
        "failure_reason": "" if failure_reason is None else failure_reason,
        "checkpoints_visited": checkpoints_visited,
        "checkpoints_total": checkpoints_total,
        "completion_ratio": completion_ratio,
        "collision_count": collision_count,
        "critical_proximity_count": critical_proximity_count,
        "control_step_count": control_step_count,
        "elapsed_simulation_time_s": elapsed,
        "actual_path_length_m": path_length,
        "minimum_clearance_m": minimum_clearance,
        "schema_version": schema_version,
        "backend": backend,
        "planner": planner,
        "model_path": model_path,
        "model_sha256": model_sha256,
        "model_training_domain": model_training_domain,
        "system_deployment_id": system_deployment_id,
        "system_deployment_profile_sha256": system_deployment_profile_sha256,
        "patrol_scenario_id": scenario_id,
        "patrol_scenario_mode": scenario_mode,
        "patrol_scenario_config_sha256": scenario_sha256,
        "deployment_evidence_scope": deployment_evidence_scope,
        "patrol_scenario_qualified": scenario_qualified,
        "safety_shield_enabled": safety_shield_enabled,
        "safety_shield_id": safety_shield_id,
        "safety_shield_sha256": safety_shield_sha256,
        "ttc_escape_guard_enabled": guard_enabled,
        "ttc_escape_guard_id": guard_id,
        "ttc_escape_guard_evidence_scope": guard_scope,
        "dynamic_obstacle_count": dynamic_obstacle_count,
        "dynamic_obstacle_speed_max_mps": dynamic_obstacle_speed_max_mps,
        **artifacts,
        "run_summary_path": summary_relative,
        "run_summary_sha256": _sha256_file(summary_path),
    }


def _execution_identity(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {field: episode[field] for field in EXECUTION_IDENTITY_FIELDS}


def _campaign_summary(
    request: PatrolCampaignRequest,
    *,
    scenario: PatrolScenarioConfig,
    episodes: Sequence[Mapping[str, Any]],
    execution_identity: Mapping[str, Any],
    critical_sources: Mapping[str, Mapping[str, str]],
    episodes_sha256: str,
) -> dict[str, Any]:
    counts = {
        outcome: sum(episode["outcome"] == outcome for episode in episodes) for outcome in OUTCOMES
    }
    episode_count = len(episodes)
    rates = {outcome: counts[outcome] / episode_count for outcome in OUTCOMES}
    acceptance_status, acceptance_gate, acceptance_failures = _acceptance_result(
        episode_count=episode_count,
        completed_rate=rates["completed"],
        collision_rate=rates["collision"],
        error_rate=rates["error"],
    )
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "execution_status": "completed",
        "acceptance_status": acceptance_status,
        "acceptance_gate": acceptance_gate,
        "acceptance_failures": acceptance_failures,
        "scenario": request.scenario,
        "patrol_scenario_id": scenario.scenario_id,
        "patrol_scenario_mode": scenario.mode.value,
        "patrol_scenario_config_path": scenario.source_path.relative_to(
            request.repository_root
        ).as_posix(),
        "patrol_scenario_config_sha256": scenario.source_sha256,
        "deployment_evidence_scope": scenario.deployment_evidence_scope,
        "patrol_scenario_qualified": False,
        "run_schema_version": execution_identity["schema_version"],
        "backend": execution_identity["backend"],
        "planner": execution_identity["planner"],
        "model_path": execution_identity["model_path"],
        "model_sha256": execution_identity["model_sha256"],
        "model_training_domain": execution_identity["model_training_domain"],
        "system_deployment_id": execution_identity["system_deployment_id"],
        "system_deployment_profile_sha256": execution_identity["system_deployment_profile_sha256"],
        "safety_shield_enabled": execution_identity["safety_shield_enabled"],
        "safety_shield_id": execution_identity["safety_shield_id"],
        "safety_shield_sha256": execution_identity["safety_shield_sha256"],
        "ttc_escape_guard_enabled": execution_identity["ttc_escape_guard_enabled"],
        "ttc_escape_guard_id": execution_identity["ttc_escape_guard_id"],
        "ttc_escape_guard_evidence_scope": execution_identity["ttc_escape_guard_evidence_scope"],
        "dynamic_obstacle_count": execution_identity["dynamic_obstacle_count"],
        "dynamic_obstacle_speed_max_mps": execution_identity["dynamic_obstacle_speed_max_mps"],
        "execution_identity": dict(execution_identity),
        "critical_sources": {label: dict(record) for label, record in critical_sources.items()},
        "seed_start": request.seed_start,
        "episode_count": episode_count,
        "scenario_seeds": [episode["scenario_seed"] for episode in episodes],
        "port": request.port,
        "episode_wall_timeout_s": request.episode_wall_timeout_s,
        **{f"{outcome}_count": counts[outcome] for outcome in OUTCOMES},
        **{f"{outcome}_rate": rates[outcome] for outcome in OUTCOMES},
        "completion_ratio_mean": math.fsum(episode["completion_ratio"] for episode in episodes)
        / episode_count,
        "elapsed_simulation_time_s_mean": math.fsum(
            episode["elapsed_simulation_time_s"] for episode in episodes
        )
        / episode_count,
        "actual_path_length_m_mean": math.fsum(
            episode["actual_path_length_m"] for episode in episodes
        )
        / episode_count,
        "minimum_clearance_m_global": min(episode["minimum_clearance_m"] for episode in episodes),
        "episodes_file": "episodes.csv",
        "episodes_sha256": episodes_sha256,
        "runs_directory": "runs",
    }


def _acceptance_result(
    *,
    episode_count: int,
    completed_rate: float,
    collision_rate: float,
    error_rate: float,
) -> tuple[str, dict[str, Any], list[str]]:
    enough_evidence = episode_count >= MINIMUM_ACCEPTANCE_EPISODES
    checks: dict[str, bool | None] = {
        "episode_count": enough_evidence,
        "completed_rate": (completed_rate >= MINIMUM_COMPLETED_RATE if enough_evidence else None),
        "collision_rate": (collision_rate <= MAXIMUM_COLLISION_RATE if enough_evidence else None),
        "error_rate": error_rate == REQUIRED_ERROR_RATE if enough_evidence else None,
    }
    gate = {
        "minimum_episode_count": MINIMUM_ACCEPTANCE_EPISODES,
        "minimum_completed_rate": MINIMUM_COMPLETED_RATE,
        "maximum_collision_rate": MAXIMUM_COLLISION_RATE,
        "required_error_rate": REQUIRED_ERROR_RATE,
        "evaluated": enough_evidence,
        "observed": {
            "episode_count": episode_count,
            "completed_rate": completed_rate,
            "collision_rate": collision_rate,
            "error_rate": error_rate,
        },
        "checks": checks,
    }
    if not enough_evidence:
        return (
            "insufficient_evidence",
            gate,
            [f"episode_count={episode_count} is below the required {MINIMUM_ACCEPTANCE_EPISODES}."],
        )
    failures = []
    if not checks["completed_rate"]:
        failures.append(
            f"completed_rate={completed_rate:.6f} is below {MINIMUM_COMPLETED_RATE:.6f}."
        )
    if not checks["collision_rate"]:
        failures.append(
            f"collision_rate={collision_rate:.6f} exceeds {MAXIMUM_COLLISION_RATE:.6f}."
        )
    if not checks["error_rate"]:
        failures.append(f"error_rate={error_rate:.6f} must equal {REQUIRED_ERROR_RATE:.6f}.")
    return ("pass" if not failures else "fail", gate, failures)


def _bind_critical_sources(
    repository_root: Path,
    scenario: PatrolScenarioConfig,
) -> dict[str, dict[str, str]]:
    paths = {
        label: repository_root / relative_path
        for label, relative_path in CRITICAL_SOURCE_PATHS.items()
    }
    paths["scenario_config"] = scenario.source_path
    result: dict[str, dict[str, str]] = {}
    for label, path in paths.items():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repository_root).as_posix()
        except ValueError as error:
            raise PatrolCampaignError(
                f"Critical source {label!r} must stay inside the repository."
            ) from error
        if not resolved.is_file():
            raise FileNotFoundError(f"Critical source {label!r} is missing: {resolved}")
        result[label] = {
            "path": relative,
            "sha256": _sha256_file(resolved),
        }
    return result


def _verify_critical_sources(
    repository_root: Path,
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    for label, record in expected.items():
        path = repository_root / record["path"]
        if not path.is_file():
            raise PatrolCampaignError(
                f"Critical source {label!r} disappeared during the campaign: {path}."
            )
        actual = _sha256_file(path)
        if actual != record["sha256"]:
            raise PatrolCampaignError(
                f"Critical source {label!r} changed during the campaign: "
                f"{record['sha256']} != {actual}."
            )


def _validated_csv_artifact(
    summary: Mapping[str, Any],
    *,
    summary_field: str,
    prefix: str,
    run_directory: Path,
    expected_rows: int | None,
) -> dict[str, Any]:
    basename = _portable_csv_basename(_required(summary, summary_field), summary_field)
    resolved_run_directory = run_directory.resolve()
    path = (resolved_run_directory / basename).resolve()
    if path.parent != resolved_run_directory:
        raise ValueError(f"{summary_field} must stay directly inside the run directory.")
    if not path.is_file():
        raise FileNotFoundError(f"{summary_field} does not exist: {path}")
    row_count = _csv_data_row_count(path)
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(f"{summary_field} row count must be {expected_rows}, got {row_count}.")
    return {
        f"{prefix}_file": basename,
        f"{prefix}_row_count": row_count,
        f"{prefix}_sha256": _sha256_file(path),
    }


def _portable_csv_basename(value: Any, label: str) -> str:
    text = _text(value, label)
    portable = PurePosixPath(text)
    if (
        portable.is_absolute()
        or portable.name != text
        or "\\" in text
        or _PORTABLE_CSV_PATTERN.fullmatch(text) is None
    ):
        raise ValueError(f"{label} must be a portable CSV basename.")
    return text


def _csv_data_row_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError(f"CSV artifact has no header: {path}") from error
            if not header or any(not column.strip() for column in header):
                raise ValueError(f"CSV artifact has an invalid header: {path}")
            return sum(1 for _ in reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError(f"CSV artifact is invalid UTF-8 CSV: {path}") from error


def _write_episodes(path: Path, episodes: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(episodes)


def _load_json_object(path: Path) -> Mapping[str, Any]:
    def reject_duplicate_fields(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON field is forbidden: {key!r}.")
            value[key] = item
        return value

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"Non-finite JSON number is forbidden: {value}.")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"summary.json is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("summary.json must contain one JSON object.")
    return payload


def _required(value: Mapping[str, Any], field: str) -> Any:
    if field not in value:
        raise ValueError(f"summary.json is missing required field {field!r}.")
    return value[field]


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def _finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be at most {maximum}.")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be non-empty text.")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain leading or trailing whitespace.")
    return value


def _exact_text(value: Any, label: str, expected: str) -> str:
    text = _text(value, label)
    if text != expected:
        raise ValueError(f"{label} must be {expected!r}, got {text!r}.")
    return text


def _exact_bool(value: Any, label: str, expected: bool) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean.")
    if value is not expected:
        raise ValueError(f"{label} must be {expected}.")
    return value


def _audit_identity(value: Any, label: str) -> str:
    text = _text(value, label)
    if _AUDIT_ID_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must be a portable audit identifier.")
    return text


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def _portable_repository_file(
    value: Any,
    *,
    label: str,
    repository_root: Path,
) -> str:
    text = _text(value, label)
    portable = PurePosixPath(text)
    if portable.is_absolute() or ".." in portable.parts or "\\" in text:
        raise ValueError(f"{label} must be a repository-relative portable path.")
    resolved = (repository_root / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository.") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return portable.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).absolute().parents[1].resolve()
    try:
        request = PatrolCampaignRequest(
            repository_root=repository_root,
            scenario=args.scenario,
            seed_start=args.seed_start,
            episodes=args.episodes,
            output_directory=args.output_dir,
            port=args.port,
            episode_wall_timeout_s=args.episode_wall_timeout_s,
        )
        summary = run_patrol_campaign(request)
    except (OSError, TypeError, ValueError, PatrolCampaignError) as error:
        print(f"Patrol campaign error: {error}", file=sys.stderr)
        return 1
    print(
        "Patrol campaign completed: "
        f"episodes={summary['episode_count']} "
        f"completed={summary['completed_count']} "
        f"collision={summary['collision_count']} "
        f"safety_stop={summary['safety_stop_count']} "
        f"timeout={summary['timeout_count']} "
        f"acceptance={summary['acceptance_status']}"
    )
    print(f"Wrote: {request.output_directory / 'episodes.csv'}")
    print(f"Wrote: {request.output_directory / 'summary.json'}")
    if summary["acceptance_status"] == "fail":
        return ACCEPTANCE_FAILURE_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
