"""Fail-closed orchestration for the RB2B development stress matrix.

This module is deliberately separate from the one-shot formal-test protocol.  It
accepts one exact, pre-registered validation-only matrix, checks every mutable
input before creating output, and delegates the three Webots runs to the strict
checkpoint-sweep implementation in exact-model mode.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from indoor_patrol_rl.core import SafetyShieldConfig
from indoor_patrol_rl.evaluation.checkpoint_selection import sha256_file
from indoor_patrol_rl.evaluation.checkpoint_sweep import (
    WebotsCheckpointSweepRequest,
    WebotsSafetyShieldProfile,
    load_safety_shield_profile,
    run_checkpoint_sweep,
)
from indoor_patrol_rl.training.webots_curriculum import choose_available_port

DYNAMIC_STRESS_EXPERIMENT_ID = "webots_rb2b_posthoc_dynamic_stress_v1"
DYNAMIC_STRESS_PURPOSE = "post_training_pre_observation_dynamic_stress_characterization"
DYNAMIC_STRESS_EVIDENCE_SCOPE = "development_validation"
DYNAMIC_STRESS_SEEDS = tuple(range(8000, 8020))
PROTECTED_TEST_SEEDS = tuple(range(10000, 10020))
RB2B_ENDPOINT_PATH = Path(
    "artifacts/models/webots/seed_11/r2b_rebaseline_v1/"
    "ppo_webots_s11_step000081920.zip"
)
RB2B_ENDPOINT_SHA256 = (
    "498d970cfcf539d55943ace88dcbaad83445713ad852291098b391fd38a86cfa"
)
RB2B_VARIANT = "r2b_rebaseline_v1"
RB2B_ENDPOINT_STEP = 81920
RB2B_TRAINING_SEED = 11
FRONT_BRAKE_PROFILE_PATH = Path("configs/eval/webots_r2a_front_brake_v1.yaml")
FRONT_BRAKE_PROFILE_SHA256 = (
    "6acd0b2bcafa7ffe48ac800ab94cf13636bc4a7c2dff2fb4e50fd7381262acf3"
)
FRONT_BRAKE_SEMANTIC_SHA256 = (
    "354ce9b831f01fb9e9c72a9ad1650574f18c98b1b5fafea26671f81ee000ec08"
)
FRONT_BRAKE_RESULT_TOKEN = "r2a-front-brake-v1-354ce9b831f0"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "evidence_scope",
    "split",
    "test_split_was_run",
    "training_seed",
    "scenario_seed_start",
    "episodes",
    "model",
    "scenario",
    "runtime_inputs",
    "shield_profile",
    "conditions",
    "preregistration",
    "gate",
    "decision",
}
_MODEL_FIELDS = {"path", "sha256", "variant", "endpoint_step"}
_SCENARIO_FIELDS = {
    "static_layout",
    "horizontal_route_half_length",
    "vertical_route_half_length",
    "deterministic_policy",
    "action_repeat",
    "max_episode_steps",
}
_SOURCE_FIELDS = {"path", "sha256"}
_RUNTIME_FIELDS = {"world", "evaluator", "environment_adapter"}
_PROFILE_FIELDS = {
    "path",
    "source_sha256",
    "semantic_sha256",
    "result_token",
    "enabled",
    "expected_resolved_config",
}
_CONDITION_FIELDS = {
    "id",
    "obstacle_count",
    "obstacle_speed_range",
    "port_base",
    "run_name",
    "sweep_directory",
}
_PREREGISTRATION_FIELDS = {
    "fixed_model",
    "fixed_profile",
    "paired_same_scenario_seeds",
    "condition_order",
    "run_all_conditions_regardless_of_gate_outcomes",
    "parameter_scan",
    "checkpoint_selection",
    "adaptive_extension",
    "adaptive_retries",
    "outcome_dependent_reruns",
    "failed_episode_exclusion",
    "held_out_formal_test_access",
    "protected_test_seed_window",
}
_GATE_FIELDS = {
    "apply_to_each_condition",
    "require_all",
    "required_episode_count",
    "minimum_success_rate",
    "maximum_collision_rate",
    "maximum_error_count",
    "required_reset_attempts",
    "required_successful_resets",
}
_DECISION_FIELDS = {"pass", "fail", "formal_model_promotion", "output_directory"}
_CONDITION_SPECS = (
    ("BASE1", 1, (0.18, 0.25), 14320),
    ("DENS3", 3, (0.18, 0.25), 14330),
    ("FAST3", 3, (0.26, 0.42), 14340),
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONDITION_PATTERN = re.compile(r"[A-Z0-9]+")
_OUTCOMES = {"success", "collision", "timeout", "error"}
_FORBIDDEN_RUNTIME_ENVIRONMENT = {
    "WEBOTS_CONFIRM_TEST",
    "WEBOTS_FORMAL_CAMPAIGN_ID",
    "WEBOTS_FORMAL_CONFIG",
    "WEBOTS_FINAL_LOCK_SHA256",
}


@dataclass(frozen=True)
class DynamicStressCondition:
    """One immutable slot in the three-condition matrix."""

    condition_id: str
    obstacle_count: int
    obstacle_speed_min: float
    obstacle_speed_max: float
    port_base: int
    run_name: str
    sweep_directory: Path


@dataclass(frozen=True)
class DynamicStressGate:
    """Descriptive gate applied after every condition has been scheduled."""

    required_episode_count: int
    minimum_success_rate: float
    maximum_collision_rate: float
    maximum_error_count: int
    required_reset_attempts: int
    required_successful_resets: int


@dataclass(frozen=True)
class DynamicStressConfig:
    """Loaded, source-bound RB2B development stress protocol."""

    repository_root: Path
    source_path: Path
    source_sha256: str
    source_git_revision: str
    raw: Mapping[str, Any]
    model_path: Path
    runtime_paths: Mapping[str, Path]
    shield_profile: WebotsSafetyShieldProfile
    expected_shield_config: Mapping[str, float]
    conditions: tuple[DynamicStressCondition, ...]
    gate: DynamicStressGate
    output_directory: Path


ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]
PortAllocator = Callable[[int, set[int]], int]
SweepRunner = Callable[..., Mapping[str, Any]]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _exact_fields(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    raw = _mapping(value, label)
    actual = set(raw)
    if actual != fields:
        raise ValueError(
            f"{label} fields differ from the locked protocol: "
            f"missing={sorted(fields - actual)}, "
            f"extra={sorted(str(name) for name in actual - fields)}."
        )
    return raw


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return number


def _same_number(actual: Any, expected: float) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isfinite(float(actual))
        and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _resolve_inside_repository(
    repository_root: Path,
    value: Any,
    label: str,
    *,
    require_file: bool = False,
) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        raise ValueError(f"{label} must be path-like.") from error
    if not str(path):
        raise ValueError(f"{label} must be non-empty.")
    if path.is_absolute():
        raise ValueError(f"{label} must be repository-relative.")
    resolved = (repository_root / path).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository.") from error
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _portable(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Artifact must stay inside the repository: {path}") from error


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"Cannot inspect Git repository ({' '.join(arguments)}): "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _require_clean_git(repository_root: Path) -> str:
    if _git_output(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(
            "Dynamic stress validation requires a completely clean Git worktree "
            "before any output is created or Webots is launched."
        )
    revision = _git_output(repository_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Cannot resolve a full clean Git HEAD revision.")
    return revision


def _verify_file(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected_sha256}.")
    return actual


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read dynamic stress config {path}: {error}") from error
    return _mapping(value, "dynamic stress config")


def _validate_runtime_inputs(
    raw: Any,
    *,
    repository_root: Path,
) -> dict[str, Path]:
    inputs = _exact_fields(raw, _RUNTIME_FIELDS, "runtime_inputs")
    expected_paths = {
        "world": "worlds/webots_ppo_evaluate.wbt",
        "evaluator": "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py",
        "environment_adapter": "src/indoor_patrol_rl/envs/webots_patrol.py",
    }
    resolved: dict[str, Path] = {}
    for name in sorted(_RUNTIME_FIELDS):
        item = _exact_fields(inputs[name], _SOURCE_FIELDS, f"runtime_inputs.{name}")
        if item["path"] != expected_paths[name]:
            raise ValueError(
                f"runtime_inputs.{name}.path must equal {expected_paths[name]!r}."
            )
        digest = _sha256(item["sha256"], f"runtime_inputs.{name}.sha256")
        path = _resolve_inside_repository(
            repository_root,
            item["path"],
            f"runtime_inputs.{name}.path",
            require_file=True,
        )
        _verify_file(path, digest, f"runtime input {name}")
        resolved[name] = path
    return resolved


def _validate_shield_profile(
    raw: Any,
    *,
    repository_root: Path,
) -> tuple[WebotsSafetyShieldProfile, dict[str, float]]:
    item = _exact_fields(raw, _PROFILE_FIELDS, "shield_profile")
    expected_identity: Mapping[str, Any] = {
        "path": FRONT_BRAKE_PROFILE_PATH.as_posix(),
        "source_sha256": FRONT_BRAKE_PROFILE_SHA256,
        "semantic_sha256": FRONT_BRAKE_SEMANTIC_SHA256,
        "result_token": FRONT_BRAKE_RESULT_TOKEN,
        "enabled": True,
    }
    for name, expected in expected_identity.items():
        if item[name] != expected or type(item[name]) is not type(expected):
            raise ValueError(f"shield_profile.{name} must equal {expected!r}.")
    profile = load_safety_shield_profile(
        FRONT_BRAKE_PROFILE_PATH,
        repository_root=repository_root,
    )
    if (
        profile.source_sha256 != FRONT_BRAKE_PROFILE_SHA256
        or profile.semantic_sha256 != FRONT_BRAKE_SEMANTIC_SHA256
        or profile.result_token != FRONT_BRAKE_RESULT_TOKEN
        or profile.enabled is not True
    ):
        raise ValueError("Loaded front-brake profile does not match its locked identity.")

    expected = _exact_fields(
        item["expected_resolved_config"],
        set(SafetyShieldConfig.__dataclass_fields__),
        "shield_profile.expected_resolved_config",
    )
    resolved = asdict(profile.config)
    for name, actual in expected.items():
        value = _number(actual, f"shield_profile.expected_resolved_config.{name}")
        if not _same_number(value, resolved[name]):
            raise ValueError(
                "shield_profile.expected_resolved_config does not match the loaded "
                f"profile field {name}."
            )
    return profile, {name: float(expected[name]) for name in expected}


def _validate_conditions(
    raw: Any,
    *,
    repository_root: Path,
    profile: WebotsSafetyShieldProfile,
) -> tuple[DynamicStressCondition, ...]:
    values = _sequence(raw, "conditions")
    if len(values) != 3:
        raise ValueError("conditions must contain exactly BASE1, DENS3 and FAST3.")
    parsed: list[DynamicStressCondition] = []
    for index, (value, spec) in enumerate(zip(values, _CONDITION_SPECS, strict=True)):
        item = _exact_fields(value, _CONDITION_FIELDS, f"conditions[{index}]")
        condition_id, obstacle_count, speeds, port_base = spec
        if item["id"] != condition_id or _CONDITION_PATTERN.fullmatch(
            str(item["id"])
        ) is None:
            raise ValueError(f"conditions[{index}].id must equal {condition_id!r}.")
        if _integer(item["obstacle_count"], f"conditions[{index}].obstacle_count") != (
            obstacle_count
        ):
            raise ValueError(
                f"conditions[{index}].obstacle_count must equal {obstacle_count}."
            )
        raw_speeds = _sequence(
            item["obstacle_speed_range"],
            f"conditions[{index}].obstacle_speed_range",
        )
        if len(raw_speeds) != 2 or any(
            not _same_number(actual, expected)
            for actual, expected in zip(raw_speeds, speeds, strict=True)
        ):
            raise ValueError(
                f"conditions[{index}].obstacle_speed_range must equal {list(speeds)}."
            )
        if _integer(item["port_base"], f"conditions[{index}].port_base", minimum=1024) != (
            port_base
        ):
            raise ValueError(f"conditions[{index}].port_base must equal {port_base}.")
        expected_run_name = (
            f"webots_validation_{condition_id.casefold()}_final_step{RB2B_ENDPOINT_STEP}_"
            f"seed{RB2B_TRAINING_SEED}_shield-{profile.result_token}"
        )
        if item["run_name"] != expected_run_name:
            raise ValueError(
                f"conditions[{index}].run_name must identify the exact final endpoint."
            )
        expected_sweep = (
            "artifacts/results/"
            f"webots_validation_{condition_id.casefold()}_checkpoint_sweep_"
            f"seed{RB2B_TRAINING_SEED}_{RB2B_VARIANT}_shield-{profile.result_token}"
        )
        if item["sweep_directory"] != expected_sweep:
            raise ValueError(
                f"conditions[{index}].sweep_directory must equal {expected_sweep!r}."
            )
        parsed.append(
            DynamicStressCondition(
                condition_id=condition_id,
                obstacle_count=obstacle_count,
                obstacle_speed_min=speeds[0],
                obstacle_speed_max=speeds[1],
                port_base=port_base,
                run_name=expected_run_name,
                sweep_directory=_resolve_inside_repository(
                    repository_root,
                    item["sweep_directory"],
                    f"conditions[{index}].sweep_directory",
                ),
            )
        )

    # The sequence is intentionally progressive: density is the only BASE1 ->
    # DENS3 change, then speed range is the only DENS3 -> FAST3 change.
    assert parsed[0].obstacle_speed_min == parsed[1].obstacle_speed_min
    assert parsed[0].obstacle_speed_max == parsed[1].obstacle_speed_max
    assert parsed[0].obstacle_count != parsed[1].obstacle_count
    assert parsed[1].obstacle_count == parsed[2].obstacle_count
    assert (
        parsed[1].obstacle_speed_min,
        parsed[1].obstacle_speed_max,
    ) != (
        parsed[2].obstacle_speed_min,
        parsed[2].obstacle_speed_max,
    )
    return tuple(parsed)


def _validate_preregistration(raw: Any) -> None:
    item = _exact_fields(raw, _PREREGISTRATION_FIELDS, "preregistration")
    expected: Mapping[str, Any] = {
        "fixed_model": True,
        "fixed_profile": True,
        "paired_same_scenario_seeds": True,
        "condition_order": ["BASE1", "DENS3", "FAST3"],
        "run_all_conditions_regardless_of_gate_outcomes": True,
        "parameter_scan": "forbidden",
        "checkpoint_selection": "forbidden",
        "adaptive_extension": "forbidden",
        "adaptive_retries": "forbidden",
        "outcome_dependent_reruns": "forbidden",
        "failed_episode_exclusion": "forbidden",
        "held_out_formal_test_access": "forbidden",
        "protected_test_seed_window": [10000, 10019],
    }
    for name, expected_value in expected.items():
        if item[name] != expected_value or type(item[name]) is not type(expected_value):
            raise ValueError(f"preregistration.{name} must equal {expected_value!r}.")


def _validate_gate(raw: Any) -> DynamicStressGate:
    item = _exact_fields(raw, _GATE_FIELDS, "gate")
    fixed: Mapping[str, Any] = {
        "apply_to_each_condition": True,
        "require_all": True,
        "required_episode_count": 20,
        "minimum_success_rate": 0.75,
        "maximum_collision_rate": 0.25,
        "maximum_error_count": 0,
        "required_reset_attempts": 20,
        "required_successful_resets": 20,
    }
    for name, expected in fixed.items():
        actual = item[name]
        matches = (
            _same_number(actual, expected)
            if isinstance(expected, float)
            else actual == expected and type(actual) is type(expected)
        )
        if not matches:
            raise ValueError(f"gate.{name} must equal {expected!r}.")
    return DynamicStressGate(
        required_episode_count=20,
        minimum_success_rate=0.75,
        maximum_collision_rate=0.25,
        maximum_error_count=0,
        required_reset_attempts=20,
        required_successful_resets=20,
    )


def _validate_decision(raw: Any, *, repository_root: Path) -> Path:
    item = _exact_fields(raw, _DECISION_FIELDS, "decision")
    expected: Mapping[str, str] = {
        "pass": "record_all_condition_development_gates_passed",
        "fail": "record_observed_development_stress_boundary_without_tuning_or_retry",
        "formal_model_promotion": "forbidden",
        "output_directory": "artifacts/results/webots_rb2b_posthoc_dynamic_stress_v1",
    }
    for name, expected_value in expected.items():
        if item[name] != expected_value:
            raise ValueError(f"decision.{name} must equal {expected_value!r}.")
    return _resolve_inside_repository(
        repository_root,
        item["output_directory"],
        "decision.output_directory",
    )


def load_dynamic_stress_config(
    path: Path,
    *,
    repository_root: Path,
) -> DynamicStressConfig:
    """Load and fully bind the one supported RB2B stress protocol."""

    root = repository_root.resolve()
    source_path = _resolve_inside_repository(
        root,
        path,
        "dynamic stress config path",
        require_file=True,
    )
    git_revision = _require_clean_git(root)
    raw = _exact_fields(_load_yaml(source_path), _TOP_LEVEL_FIELDS, "dynamic stress config")
    locked_scalars: Mapping[str, Any] = {
        "schema_version": 1,
        "experiment_id": DYNAMIC_STRESS_EXPERIMENT_ID,
        "purpose": DYNAMIC_STRESS_PURPOSE,
        "evidence_scope": DYNAMIC_STRESS_EVIDENCE_SCOPE,
        "split": "validation",
        "test_split_was_run": False,
        "training_seed": RB2B_TRAINING_SEED,
        "scenario_seed_start": DYNAMIC_STRESS_SEEDS[0],
        "episodes": len(DYNAMIC_STRESS_SEEDS),
    }
    for name, expected in locked_scalars.items():
        if raw[name] != expected or type(raw[name]) is not type(expected):
            raise ValueError(f"{name} must equal the locked value {expected!r}.")
    if set(DYNAMIC_STRESS_SEEDS) & set(PROTECTED_TEST_SEEDS):
        raise ValueError("Development stress seeds overlap the protected formal test split.")

    model = _exact_fields(raw["model"], _MODEL_FIELDS, "model")
    model_expected: Mapping[str, Any] = {
        "path": RB2B_ENDPOINT_PATH.as_posix(),
        "sha256": RB2B_ENDPOINT_SHA256,
        "variant": RB2B_VARIANT,
        "endpoint_step": RB2B_ENDPOINT_STEP,
    }
    for name, expected in model_expected.items():
        if model[name] != expected or type(model[name]) is not type(expected):
            raise ValueError(f"model.{name} must equal {expected!r}.")
    model_path = _resolve_inside_repository(
        root,
        model["path"],
        "model.path",
        require_file=True,
    )
    _verify_file(model_path, RB2B_ENDPOINT_SHA256, "RB2B endpoint model")

    scenario = _exact_fields(raw["scenario"], _SCENARIO_FIELDS, "scenario")
    scenario_expected: Mapping[str, Any] = {
        "static_layout": "train_a",
        "horizontal_route_half_length": 1.10,
        "vertical_route_half_length": 1.10,
        "deterministic_policy": True,
        "action_repeat": 4,
        "max_episode_steps": 300,
    }
    for name, expected in scenario_expected.items():
        actual = scenario[name]
        matches = (
            _same_number(actual, expected)
            if isinstance(expected, float)
            else actual == expected and type(actual) is type(expected)
        )
        if not matches:
            raise ValueError(f"scenario.{name} must equal {expected!r}.")

    runtime_paths = _validate_runtime_inputs(raw["runtime_inputs"], repository_root=root)
    profile, expected_profile = _validate_shield_profile(
        raw["shield_profile"],
        repository_root=root,
    )
    conditions = _validate_conditions(
        raw["conditions"],
        repository_root=root,
        profile=profile,
    )
    _validate_preregistration(raw["preregistration"])
    gate = _validate_gate(raw["gate"])
    output_directory = _validate_decision(raw["decision"], repository_root=root)
    return DynamicStressConfig(
        repository_root=root,
        source_path=source_path,
        source_sha256=sha256_file(source_path),
        source_git_revision=git_revision,
        raw=raw,
        model_path=model_path,
        runtime_paths=runtime_paths,
        shield_profile=profile,
        expected_shield_config=expected_profile,
        conditions=conditions,
        gate=gate,
        output_directory=output_directory,
    )


def build_dynamic_stress_requests(
    config: DynamicStressConfig,
    *,
    webots_executable: Path,
    timeout_seconds: float = 3600.0,
) -> tuple[WebotsCheckpointSweepRequest, ...]:
    """Resolve the exact three validation-only sweep requests without side effects."""

    requests: list[WebotsCheckpointSweepRequest] = []
    scenario = config.raw["scenario"]
    for condition in config.conditions:
        request = WebotsCheckpointSweepRequest(
            repository_root=config.repository_root,
            webots_executable=webots_executable,
            seed=RB2B_TRAINING_SEED,
            condition=condition.condition_id,
            model_variant=RB2B_VARIANT,
            steps=(RB2B_ENDPOINT_STEP,),
            episodes=len(DYNAMIC_STRESS_SEEDS),
            seed_start=DYNAMIC_STRESS_SEEDS[0],
            obstacle_count=condition.obstacle_count,
            obstacle_speed_min=condition.obstacle_speed_min,
            obstacle_speed_max=condition.obstacle_speed_max,
            horizontal_route_half_length=float(
                scenario["horizontal_route_half_length"]
            ),
            vertical_route_half_length=float(scenario["vertical_route_half_length"]),
            port_base=condition.port_base,
            workers=1,
            timeout_seconds=timeout_seconds,
            shield_profile=config.shield_profile,
            exact_model_path=config.model_path,
            exact_model_sha256=RB2B_ENDPOINT_SHA256,
        )
        if request.run_name(RB2B_ENDPOINT_STEP) != condition.run_name:
            raise ValueError(
                f"Condition {condition.condition_id} run_name drifted from the exact-model "
                "checkpoint-sweep contract."
            )
        requests.append(request)
    return tuple(requests)


def _recheck_preflight(config: DynamicStressConfig) -> None:
    revision = _require_clean_git(config.repository_root)
    if revision != config.source_git_revision:
        raise ValueError("Git HEAD changed after the dynamic stress config was loaded.")
    _verify_file(config.source_path, config.source_sha256, "dynamic stress config")
    _verify_file(config.model_path, RB2B_ENDPOINT_SHA256, "RB2B endpoint model")
    for name, path in config.runtime_paths.items():
        expected = config.raw["runtime_inputs"][name]["sha256"]
        _verify_file(path, expected, f"runtime input {name}")
    config.shield_profile.verify_source(config.repository_root)
    if config.shield_profile.semantic_sha256 != FRONT_BRAKE_SEMANTIC_SHA256:
        raise ValueError("Front-brake shield semantics changed after config load.")


def _preflight_output_paths(
    config: DynamicStressConfig,
    requests: Sequence[WebotsCheckpointSweepRequest],
) -> None:
    result_directories = [
        config.repository_root / "artifacts/results" / request.run_name(RB2B_ENDPOINT_STEP)
        for request in requests
    ]
    targets = [
        config.output_directory,
        *(condition.sweep_directory for condition in config.conditions),
        *result_directories,
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite dynamic stress outputs: "
            + ", ".join(str(path) for path in existing)
        )


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} from {path}: {error}") from error
    return _mapping(value, label)


def _parse_episodes(
    path: Path,
    *,
    condition: DynamicStressCondition,
) -> tuple[dict[str, int], str]:
    required_fields = {
        "schema_version",
        "episode_id",
        "split",
        "scenario_seed",
        "planner",
        "condition_id",
        "outcome",
        "is_success",
        "collision",
        "timeout",
        "error_type",
    }
    seen: set[int] = set()
    planner: str | None = None
    counts = {name: 0 for name in _OUTCOMES}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError("Dynamic stress episodes.csv lacks required audit columns.")
        for row_number, row in enumerate(reader, start=2):
            label = f"{condition.condition_id} episodes.csv row {row_number}"
            try:
                schema_version = float(row["schema_version"])
            except ValueError as error:
                raise ValueError(f"{label} schema_version must be numeric.") from error
            if not math.isfinite(schema_version) or schema_version != 1.0:
                raise ValueError(f"{label} schema_version must equal 1.0.")
            _string(row["episode_id"], f"{label} episode_id")
            if row["split"] != "validation" or row["condition_id"] != condition.condition_id:
                raise ValueError(f"{label} split or condition does not match the protocol.")
            try:
                seed = int(row["scenario_seed"])
            except ValueError as error:
                raise ValueError(f"{label} scenario_seed must be an integer.") from error
            if seed in seen:
                raise ValueError(f"{label} repeats scenario seed {seed}.")
            seen.add(seed)
            row_planner = _string(row["planner"], f"{label} planner")
            planner = planner or row_planner
            if row_planner != planner:
                raise ValueError(f"{condition.condition_id} episodes use multiple planners.")
            outcome = row["outcome"]
            if outcome not in _OUTCOMES:
                raise ValueError(f"{label} has invalid outcome {outcome!r}.")
            expected_flags = {
                "is_success": "True" if outcome == "success" else "False",
                "collision": "True" if outcome == "collision" else "False",
                "timeout": "True" if outcome == "timeout" else "False",
            }
            if any(row[name] != expected for name, expected in expected_flags.items()):
                raise ValueError(f"{label} outcome and boolean flags disagree.")
            if bool(row["error_type"]) is not (outcome == "error"):
                raise ValueError(f"{label} error_type does not agree with outcome.")
            counts[outcome] += 1
    if tuple(sorted(seen)) != DYNAMIC_STRESS_SEEDS:
        raise ValueError(
            f"{condition.condition_id} episodes.csv must contain exactly seeds 8000..8019."
        )
    if planner is None:
        raise ValueError(f"{condition.condition_id} episodes.csv is empty.")
    return counts, planner


def _gate_record(
    config: DynamicStressConfig,
    *,
    counts: Mapping[str, int],
    reset_health: Mapping[str, Any],
) -> dict[str, Any]:
    episodes = sum(counts.values())
    success_rate = counts["success"] / episodes
    collision_rate = counts["collision"] / episodes
    checks = {
        "required_episode_count": episodes == config.gate.required_episode_count,
        "minimum_success_rate": success_rate >= config.gate.minimum_success_rate,
        "maximum_collision_rate": collision_rate <= config.gate.maximum_collision_rate,
        "maximum_error_count": counts["error"] <= config.gate.maximum_error_count,
        "required_reset_attempts": (
            reset_health.get("attempts") == config.gate.required_reset_attempts
        ),
        "required_successful_resets": (
            reset_health.get("successful") == config.gate.required_successful_resets
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "episodes": episodes,
        "success_count": counts["success"],
        "collision_count": counts["collision"],
        "timeout_count": counts["timeout"],
        "error_count": counts["error"],
        "success_rate": success_rate,
        "collision_rate": collision_rate,
    }


def _condition_evidence(
    config: DynamicStressConfig,
    condition: DynamicStressCondition,
    request: WebotsCheckpointSweepRequest,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("status") != "completed" or manifest.get("split") != "validation":
        raise RuntimeError(f"{condition.condition_id} sweep did not complete validation.")
    request_record = _mapping(manifest.get("request"), "sweep request")
    forbidden = _FORBIDDEN_RUNTIME_ENVIRONMENT & set(request_record)
    if forbidden:
        raise ValueError(f"Sweep request contains forbidden formal/test fields: {forbidden}.")
    expected_request: Mapping[str, Any] = {
        "condition": condition.condition_id,
        "seed": RB2B_TRAINING_SEED,
        "seed_start": DYNAMIC_STRESS_SEEDS[0],
        "episodes": len(DYNAMIC_STRESS_SEEDS),
        "scenario_seeds": list(DYNAMIC_STRESS_SEEDS),
        "model_variant": RB2B_VARIANT,
        "exact_model_path": RB2B_ENDPOINT_PATH.as_posix(),
        "exact_model_sha256": RB2B_ENDPOINT_SHA256,
        "obstacle_count": condition.obstacle_count,
        "static_layout": "train_a",
        "safety_shield_enabled": True,
    }
    for name, expected in expected_request.items():
        if request_record.get(name) != expected:
            raise ValueError(
                f"{condition.condition_id} sweep request.{name} does not match config."
            )

    runs = _sequence(manifest.get("runs"), "sweep runs")
    matching = [
        run
        for run in runs
        if isinstance(run, dict) and run.get("run_name") == condition.run_name
    ]
    if len(matching) != 1:
        raise ValueError(f"{condition.condition_id} sweep must contain exactly one run.")
    run = _mapping(matching[0], "sweep run")
    if run.get("status") != "completed" or run.get("exit_code") != 0:
        raise RuntimeError(f"{condition.condition_id} Webots run failed infrastructure checks.")
    if run.get("planner") != request.planner(RB2B_ENDPOINT_STEP):
        raise ValueError(f"{condition.condition_id} run planner identity drifted.")
    environment = _mapping(run.get("environment"), "sweep run environment")
    forbidden_environment = {
        name
        for name in environment
        if name == "WEBOTS_CONFIRM_TEST"
        or name == "WEBOTS_FINAL_LOCK_SHA256"
        or name.startswith("WEBOTS_FORMAL_")
    }
    if forbidden_environment:
        raise ValueError(
            f"{condition.condition_id} run environment contains forbidden formal/test "
            f"fields: {sorted(forbidden_environment)}."
        )
    expected_environment = {
        "WEBOTS_EVAL_SPLIT": "validation",
        "WEBOTS_EVAL_CONDITION": condition.condition_id,
        "WEBOTS_EVAL_SEED_START": "8000",
        "WEBOTS_EVAL_EPISODES": "20",
        "WEBOTS_EVAL_MODEL": RB2B_ENDPOINT_PATH.as_posix(),
        "WEBOTS_EVAL_MODEL_SHA256": RB2B_ENDPOINT_SHA256,
        "WEBOTS_EVAL_PLANNER_LABEL": request.planner(RB2B_ENDPOINT_STEP),
        "WEBOTS_SAFETY_SHIELD": "true",
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise ValueError(
                f"{condition.condition_id} run environment {name} does not match config."
            )
    if run.get("model_path") != RB2B_ENDPOINT_PATH.as_posix():
        raise ValueError(f"{condition.condition_id} run used a different model path.")
    if run.get("model_sha256") != RB2B_ENDPOINT_SHA256:
        raise ValueError(f"{condition.condition_id} run used a different model SHA-256.")

    summary_path = _resolve_inside_repository(
        config.repository_root,
        run.get("summary_path"),
        f"{condition.condition_id} summary_path",
        require_file=True,
    )
    summary_sha256 = _verify_file(
        summary_path,
        _sha256(run.get("summary_sha256"), "run.summary_sha256"),
        f"{condition.condition_id} summary.json",
    )
    summary = _load_json(summary_path, f"{condition.condition_id} summary")
    provenance = _mapping(summary.get("evaluation_provenance"), "evaluation_provenance")
    episodes_path = summary_path.with_name("episodes.csv")
    episodes_sha256 = _verify_file(
        episodes_path,
        _sha256(
            provenance.get("episodes_csv_sha256"),
            "evaluation_provenance.episodes_csv_sha256",
        ),
        f"{condition.condition_id} episodes.csv",
    )
    counts, planner = _parse_episodes(episodes_path, condition=condition)

    expected_provenance: Mapping[str, Any] = {
        "model_path": RB2B_ENDPOINT_PATH.as_posix(),
        "model_sha256": RB2B_ENDPOINT_SHA256,
        "model_variant": RB2B_VARIANT,
        "training_seed": RB2B_TRAINING_SEED,
        "split": "validation",
        "condition_id": condition.condition_id,
        "scenario_seed_start": DYNAMIC_STRESS_SEEDS[0],
        "scenario_seed_count": len(DYNAMIC_STRESS_SEEDS),
        "scenario_seeds": list(DYNAMIC_STRESS_SEEDS),
        "obstacle_count": condition.obstacle_count,
        "static_layout": "train_a",
        "safety_shield_enabled": True,
    }
    for name, expected in expected_provenance.items():
        if provenance.get(name) != expected:
            raise ValueError(
                f"{condition.condition_id} summary provenance.{name} does not match."
            )
    raw_speeds = provenance.get("obstacle_speed_range")
    if not isinstance(raw_speeds, list) or len(raw_speeds) != 2 or any(
        not _same_number(actual, expected)
        for actual, expected in zip(
            raw_speeds,
            (condition.obstacle_speed_min, condition.obstacle_speed_max),
            strict=True,
        )
    ):
        raise ValueError(f"{condition.condition_id} obstacle speed provenance drifted.")
    raw_shield = _mapping(provenance.get("safety_shield_config"), "safety_shield_config")
    if set(raw_shield) != set(config.expected_shield_config):
        raise ValueError("Summary safety_shield_config must contain all 15 fields exactly.")
    for name, expected in config.expected_shield_config.items():
        if not _same_number(raw_shield[name], expected):
            raise ValueError(
                f"{condition.condition_id} summary shield field {name} drifted."
            )
    reset_health = _mapping(provenance.get("reset_health"), "reset_health")
    expected_reset_health = {
        "attempts": 20,
        "successful": 20,
        "finite_settle_checks": 40,
        "readback_checks": 20,
    }
    for name, expected in expected_reset_health.items():
        if reset_health.get(name) != expected or type(reset_health.get(name)) is not int:
            raise ValueError(
                f"{condition.condition_id} reset_health.{name} must equal {expected}."
            )
    gate = _gate_record(config, counts=counts, reset_health=reset_health)

    if summary.get("split") != "validation" or summary.get("episode_count") != 20:
        raise ValueError(f"{condition.condition_id} summary identity is invalid.")
    if summary.get("planners") != [planner] or summary.get("conditions") != [
        condition.condition_id
    ]:
        raise ValueError(f"{condition.condition_id} summary planner/condition drifted.")
    def validate_aggregate(value: Any, *, label: str, include_condition: bool) -> None:
        aggregate_item = _mapping(value, label)
        aggregate_expected: dict[str, Any] = {
            "planner": planner,
            "episodes": 20,
            "success_count": counts["success"],
            "collision_count": counts["collision"],
            "timeout_count": counts["timeout"],
            "error_count": counts["error"],
        }
        if include_condition:
            aggregate_expected["condition_id"] = condition.condition_id
        for name, expected in aggregate_expected.items():
            if aggregate_item.get(name) != expected:
                raise ValueError(
                    f"{condition.condition_id} {label}.{name} disagrees with episodes.csv."
                )
        if not _same_number(aggregate_item.get("success_rate"), gate["success_rate"]):
            raise ValueError(f"{condition.condition_id} {label}.success_rate drifted.")
        if not _same_number(aggregate_item.get("collision_rate"), gate["collision_rate"]):
            raise ValueError(f"{condition.condition_id} {label}.collision_rate drifted.")

    by_planner = _sequence(summary.get("by_planner"), "by_planner")
    by_condition = _sequence(
        summary.get("by_planner_condition"),
        "by_planner_condition",
    )
    if len(by_planner) != 1 or len(by_condition) != 1:
        raise ValueError(
            f"{condition.condition_id} summary must contain one aggregate per scope."
        )
    validate_aggregate(by_planner[0], label="by_planner[0]", include_condition=False)
    validate_aggregate(
        by_condition[0],
        label="by_planner_condition[0]",
        include_condition=True,
    )

    return {
        "condition_id": condition.condition_id,
        "scenario": {
            "obstacle_count": condition.obstacle_count,
            "obstacle_speed_range": [
                condition.obstacle_speed_min,
                condition.obstacle_speed_max,
            ],
        },
        "run_name": condition.run_name,
        "sweep_manifest": {
            "path": _portable(
                condition.sweep_directory / "checkpoint_sweep.json",
                config.repository_root,
            ),
            "sha256": sha256_file(
                condition.sweep_directory / "checkpoint_sweep.json"
            ),
        },
        "summary": {
            "path": _portable(summary_path, config.repository_root),
            "sha256": summary_sha256,
        },
        "episodes_csv": {
            "path": _portable(episodes_path, config.repository_root),
            "sha256": episodes_sha256,
        },
        "model_sha256": RB2B_ENDPOINT_SHA256,
        "shield_profile": {
            "source_sha256": config.shield_profile.source_sha256,
            "semantic_sha256": config.shield_profile.semantic_sha256,
            "resolved_config": dict(config.expected_shield_config),
        },
        "gate": gate,
    }


def _write_outputs(config: DynamicStressConfig, report: Mapping[str, Any]) -> dict[str, Path]:
    output = config.output_directory
    json_path = output / "dynamic_stress_report.json"
    yaml_path = output / "dynamic_stress_report.yaml"
    digest_path = output / "dynamic_stress_report.sha256"
    if any(path.exists() for path in (json_path, yaml_path, digest_path)):
        raise FileExistsError("Refusing to overwrite dynamic stress aggregate report.")
    json_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    yaml_text = yaml.safe_dump(json.loads(json_text), allow_unicode=True, sort_keys=False)
    digest = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    json_path.write_text(json_text, encoding="utf-8")
    yaml_path.write_text(yaml_text, encoding="utf-8")
    digest_path.write_text(f"{digest}  {json_path.name}\n", encoding="utf-8")
    return {"json": json_path, "yaml": yaml_path, "sha256": digest_path}


def run_dynamic_stress(
    config: DynamicStressConfig,
    *,
    webots_executable: Path,
    process_runner: ProcessRunner = subprocess.run,
    port_allocator: PortAllocator = choose_available_port,
    sweep_runner: SweepRunner = run_checkpoint_sweep,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Execute all three pre-registered conditions and write one bound report.

    Gate failures are recorded only after a condition has completed and never stop
    the remaining conditions.  Missing/tampered inputs and infrastructure failures
    remain hard errors.
    """

    _recheck_preflight(config)
    requests = build_dynamic_stress_requests(
        config,
        webots_executable=webots_executable,
        timeout_seconds=timeout_seconds,
    )
    _preflight_output_paths(config, requests)

    evidence: list[dict[str, Any]] = []
    for condition, request in zip(config.conditions, requests, strict=True):
        manifest = sweep_runner(
            request,
            output_directory=condition.sweep_directory,
            process_runner=process_runner,
            port_allocator=port_allocator,
        )
        evidence.append(
            _condition_evidence(config, condition, request, manifest)
        )

    all_passed = all(bool(item["gate"]["passed"]) for item in evidence)
    decision = config.raw["decision"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "webots_rb2b_development_dynamic_stress",
        "status": "completed",
        "experiment_id": DYNAMIC_STRESS_EXPERIMENT_ID,
        "purpose": DYNAMIC_STRESS_PURPOSE,
        "evidence_scope": DYNAMIC_STRESS_EVIDENCE_SCOPE,
        "split": "validation",
        "test_split_was_run": False,
        "source_git_revision": config.source_git_revision,
        "source_bindings": {
            "config": {
                "path": _portable(config.source_path, config.repository_root),
                "sha256": config.source_sha256,
            },
            "model": {
                "path": RB2B_ENDPOINT_PATH.as_posix(),
                "sha256": RB2B_ENDPOINT_SHA256,
                "variant": RB2B_VARIANT,
                "endpoint_step": RB2B_ENDPOINT_STEP,
            },
            "shield_profile": config.shield_profile.manifest_record(
                config.repository_root
            ),
            "runtime_inputs": {
                name: {
                    "path": _portable(path, config.repository_root),
                    "sha256": config.raw["runtime_inputs"][name]["sha256"],
                }
                for name, path in sorted(config.runtime_paths.items())
            },
        },
        "scenario_seeds": list(DYNAMIC_STRESS_SEEDS),
        "conditions": evidence,
        "restrictions": {
            "formal_model_promotion": "forbidden",
            "held_out_formal_test_access": "forbidden",
            "adaptive_retries": "forbidden",
            "outcome_dependent_reruns": "forbidden",
        },
        "decision": {
            "passed": all_passed,
            "outcome": decision["pass"] if all_passed else decision["fail"],
            "formal_model_promotion": "forbidden",
        },
    }
    report["output_paths"] = {
        "json": _portable(
            config.output_directory / "dynamic_stress_report.json",
            config.repository_root,
        ),
        "yaml": _portable(
            config.output_directory / "dynamic_stress_report.yaml",
            config.repository_root,
        ),
        "sha256": _portable(
            config.output_directory / "dynamic_stress_report.sha256",
            config.repository_root,
        ),
    }
    _write_outputs(config, report)
    return report


def dynamic_stress_exit_code(report: Mapping[str, Any]) -> int:
    """Return zero only when execution completed and all descriptive gates passed."""

    decision = report.get("decision")
    return (
        0
        if report.get("status") == "completed"
        and isinstance(decision, Mapping)
        and decision.get("passed") is True
        else 1
    )


__all__ = [
    "DYNAMIC_STRESS_EXPERIMENT_ID",
    "DYNAMIC_STRESS_SEEDS",
    "DynamicStressCondition",
    "DynamicStressConfig",
    "DynamicStressGate",
    "build_dynamic_stress_requests",
    "dynamic_stress_exit_code",
    "load_dynamic_stress_config",
    "run_dynamic_stress",
]
