"""Restartable validation-only Webots checkpoint sweeps.

The sweep deliberately stops at producing auditable validation summaries.  Model
ranking and copying remain the responsibility of ``select_webots_checkpoint.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from indoor_patrol_rl.core import (
    SAFETY_SHIELD_CONFIG_ENV_FIELDS,
    SafetyShieldConfig,
    safety_shield_config_from_env,
)
from indoor_patrol_rl.evaluation.checkpoint_selection import (
    CheckpointSelectionGate,
    load_strict_candidate_summary,
    make_candidate_record,
    parse_result_identity,
    sha256_file,
)
from indoor_patrol_rl.training.webots_curriculum import (
    choose_available_port,
    load_validation_metrics,
)

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
PortAllocator = Callable[[int, set[int]], int]

_CONDITION_PATTERN = re.compile(r"[A-Za-z0-9]+")
_MODEL_VARIANT_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
_SHIELD_PROFILE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EVALUATION_WORLD = "worlds/webots_ppo_evaluate.wbt"
_EVALUATOR_CONTROLLER = (
    "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
)


def utc_now() -> str:
    """Return a compact, timezone-aware audit timestamp."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def _validate_model_variant(value: str | None) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or "/" in value
        or "\\" in value
        or _MODEL_VARIANT_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            "model_variant must be one path-safe name containing only letters, "
            "digits, '.', '_' and '-'."
        )


def _positive_finite(value: float, label: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    lower_bound_ok = number >= 0.0 if allow_zero else number > 0.0
    if not math.isfinite(number) or not lower_bound_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a finite {qualifier} number.")
    return number


@dataclass(frozen=True)
class WebotsSafetyShieldProfile:
    """One source-bound, complete non-hardware safety-shield configuration."""

    source_path: Path
    source_sha256: str
    profile_id: str
    enabled: bool
    config: SafetyShieldConfig

    def __post_init__(self) -> None:
        try:
            source_path = Path(self.source_path).resolve()
        except TypeError as error:
            raise ValueError("Shield profile source_path must be path-like.") from error
        object.__setattr__(self, "source_path", source_path)
        if (
            not isinstance(self.source_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.source_sha256) is None
        ):
            raise ValueError("Shield profile source_sha256 must be lowercase SHA-256.")
        if (
            not isinstance(self.profile_id, str)
            or _SHIELD_PROFILE_ID_PATTERN.fullmatch(self.profile_id) is None
        ):
            raise ValueError(
                "Shield profile_id must be a lowercase path-safe name of at most "
                "64 characters."
            )
        if type(self.enabled) is not bool:
            raise ValueError("Shield profile enabled must be a boolean.")
        if not isinstance(self.config, SafetyShieldConfig):
            raise ValueError("Shield profile config must be a SafetyShieldConfig.")

    @property
    def config_values(self) -> dict[str, float]:
        values = asdict(self.config)
        values.pop("lidar_max_range")
        return {name: float(value) for name, value in values.items()}

    @property
    def controller_environment(self) -> dict[str, str]:
        return {
            environment_name: repr(self.config_values[field_name])
            for environment_name, field_name in SAFETY_SHIELD_CONFIG_ENV_FIELDS.items()
        }

    @property
    def semantic_sha256(self) -> str:
        payload = {
            "enabled": self.enabled,
            "config": self.config_values,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def result_token(self) -> str:
        return f"{self.profile_id}-{self.semantic_sha256[:12]}"

    def verify_source(self, repository_root: Path) -> None:
        resolved = self.source_path.resolve()
        try:
            resolved.relative_to(repository_root.resolve())
        except ValueError as error:
            raise ValueError("Shield profile must stay inside the repository.") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"Shield profile does not exist: {resolved}")
        if sha256_file(resolved) != self.source_sha256:
            raise ValueError("Shield profile source changed after it was loaded.")

    def manifest_record(self, repository_root: Path) -> dict[str, Any]:
        self.verify_source(repository_root)
        return {
            "profile_id": self.profile_id,
            "result_token": self.result_token,
            "semantic_sha256": self.semantic_sha256,
            "source_path": _portable(self.source_path, repository_root),
            "source_sha256": self.source_sha256,
            "enabled": self.enabled,
            "config": self.config_values,
            "lidar_max_range_source": "LDS-01.getMaxRange()",
        }


@dataclass(frozen=True)
class WebotsCheckpointSweepRequest:
    """Immutable scenario and model identity for one validation sweep.

    The default mode resolves each requested step through the historical
    checkpoint filename convention. ``exact_model_path`` and
    ``exact_model_sha256`` opt into a single final-endpoint evaluation without
    copying that endpoint into the checkpoint directory.
    """

    repository_root: Path
    webots_executable: Path
    seed: int
    condition: str
    model_variant: str | None
    steps: tuple[int, ...]
    episodes: int = 100
    seed_start: int = 5000
    obstacle_count: int = 1
    obstacle_speed_min: float = 0.10
    obstacle_speed_max: float = 0.20
    horizontal_route_half_length: float = 1.10
    vertical_route_half_length: float = 1.10
    port_base: int = 13400
    workers: int = 1
    timeout_seconds: float = 3600.0
    shield_profile: WebotsSafetyShieldProfile | None = None
    exact_model_path: Path | None = None
    exact_model_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_root", self.repository_root.resolve())
        object.__setattr__(self, "webots_executable", self.webots_executable.resolve())
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        condition = str(self.condition).upper()
        if _CONDITION_PATTERN.fullmatch(condition) is None:
            raise ValueError("condition must contain only letters and digits.")
        object.__setattr__(self, "condition", condition)
        _validate_model_variant(self.model_variant)

        if not self.steps or any(isinstance(step, bool) or step <= 0 for step in self.steps):
            raise ValueError("steps must contain one or more positive integers.")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("steps must not contain duplicates.")
        object.__setattr__(self, "steps", tuple(sorted(self.steps)))

        exact_path_provided = self.exact_model_path is not None
        exact_sha_provided = self.exact_model_sha256 is not None
        if exact_path_provided != exact_sha_provided:
            raise ValueError(
                "exact_model_path and exact_model_sha256 must be provided together."
            )
        if exact_path_provided:
            if len(self.steps) != 1:
                raise ValueError("Exact model mode requires exactly one step.")
            if self.model_variant is None:
                raise ValueError("Exact model mode requires a non-empty model_variant.")
            assert self.exact_model_path is not None
            assert self.exact_model_sha256 is not None
            exact_model_path = _resolve_inside_repository(
                Path(self.exact_model_path),
                self.repository_root,
                "exact_model_path",
            )
            step = self.steps[0]
            canonical_model_path = (
                self.repository_root
                / "artifacts/models/webots"
                / f"seed_{self.seed}"
                / self.model_variant
                / f"ppo_webots_s{self.seed}_step{step:09d}.zip"
            ).resolve()
            if exact_model_path != canonical_model_path:
                raise ValueError(
                    "exact_model_path must equal the canonical final endpoint path "
                    f"{canonical_model_path}."
                )
            if not exact_model_path.is_file():
                raise FileNotFoundError(
                    f"Exact model does not exist: {exact_model_path}"
                )
            if (
                not isinstance(self.exact_model_sha256, str)
                or _SHA256_PATTERN.fullmatch(self.exact_model_sha256) is None
            ):
                raise ValueError(
                    "exact_model_sha256 must be a lowercase SHA-256 digest."
                )
            object.__setattr__(self, "exact_model_path", exact_model_path)
            self.verify_exact_model()
        if isinstance(self.episodes, bool) or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer.")
        if isinstance(self.seed_start, bool) or self.seed_start < 0:
            raise ValueError("seed_start must be a non-negative integer.")
        if isinstance(self.obstacle_count, bool) or not 0 <= self.obstacle_count <= 3:
            raise ValueError("obstacle_count must be between zero and three.")

        speed_min = _positive_finite(
            self.obstacle_speed_min,
            "obstacle_speed_min",
            allow_zero=True,
        )
        speed_max = _positive_finite(
            self.obstacle_speed_max,
            "obstacle_speed_max",
            allow_zero=True,
        )
        if speed_min > speed_max:
            raise ValueError("obstacle speeds must be ordered min <= max.")
        _positive_finite(
            self.horizontal_route_half_length,
            "horizontal_route_half_length",
        )
        _positive_finite(
            self.vertical_route_half_length,
            "vertical_route_half_length",
        )
        if isinstance(self.port_base, bool) or not 1024 <= self.port_base <= 65500:
            raise ValueError("port_base must be between 1024 and 65500.")
        if isinstance(self.workers, bool) or self.workers <= 0:
            raise ValueError("workers must be a positive integer.")
        _positive_finite(self.timeout_seconds, "timeout_seconds")
        if self.shield_profile is not None:
            if not isinstance(self.shield_profile, WebotsSafetyShieldProfile):
                raise ValueError("shield_profile must be a WebotsSafetyShieldProfile.")
            self.shield_profile.verify_source(self.repository_root)

    @property
    def model_directory(self) -> Path:
        directory = (
            self.repository_root
            / "artifacts/models/webots"
            / f"seed_{self.seed}"
        )
        return directory / self.model_variant if self.model_variant else directory

    def model_path(self, step: int) -> Path:
        if self.exact_model_path is not None:
            if step != self.steps[0]:
                raise ValueError("Exact model mode only supports its declared step.")
            return self.exact_model_path
        variant_token = f"_{self.model_variant}" if self.model_variant else ""
        return (
            self.model_directory
            / "checkpoints"
            / f"ppo_webots_s{self.seed}{variant_token}_{step}_steps.zip"
        )

    def run_name(self, step: int) -> str:
        shield_suffix = (
            f"_shield-{self.shield_profile.result_token}"
            if self.shield_profile is not None
            else ""
        )
        result_identity = (
            f"final_step{step}"
            if self.exact_model_path is not None
            else f"checkpoint{step}"
        )
        return (
            f"webots_validation_{self.condition.casefold()}_"
            f"{result_identity}_seed{self.seed}{shield_suffix}"
        )

    def planner(self, step: int) -> str:
        return f"ppo_{self.condition.casefold()}_step{step}"

    @property
    def result_kind(self) -> str:
        """Return the result-directory identity used by the strict parser."""

        return "final" if self.exact_model_path is not None else "checkpoint"

    def verify_exact_model(self) -> str | None:
        """Fail if an exact final endpoint is missing or has changed."""

        if self.exact_model_path is None:
            return None
        assert self.exact_model_sha256 is not None
        if not self.exact_model_path.is_file():
            raise FileNotFoundError(
                f"Exact model does not exist: {self.exact_model_path}"
            )
        actual_sha256 = sha256_file(self.exact_model_path)
        if actual_sha256 != self.exact_model_sha256:
            raise ValueError(
                "Exact model SHA-256 mismatch: "
                f"{actual_sha256} != {self.exact_model_sha256}."
            )
        return actual_sha256


def _portable(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Sweep artifact must stay inside the repository: {path}") from error


def _resolve_inside_repository(path: Path, repository_root: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else repository_root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository: {path}") from error
    return resolved


def load_safety_shield_profile(
    path: Path,
    *,
    repository_root: Path,
) -> WebotsSafetyShieldProfile:
    """Load a complete repository-local profile and bind its exact source."""

    profile_path = _resolve_inside_repository(path, repository_root, "shield_profile")
    if not profile_path.is_file():
        raise FileNotFoundError(f"Shield profile does not exist: {profile_path}")
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read shield profile {profile_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("Shield profile must be a YAML mapping.")
    expected_root_fields = {"schema_version", "profile_id", "enabled", "config"}
    actual_root_fields = set(raw)
    if actual_root_fields != expected_root_fields:
        raise ValueError(
            "Shield profile fields differ from the schema; "
            f"expected {sorted(expected_root_fields)}, "
            f"got {sorted(str(name) for name in actual_root_fields)}."
        )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Shield profile schema_version must be 1.")
    profile_id = raw["profile_id"]
    if (
        not isinstance(profile_id, str)
        or _SHIELD_PROFILE_ID_PATTERN.fullmatch(profile_id) is None
    ):
        raise ValueError(
            "Shield profile_id must be a lowercase path-safe name of at most "
            "64 characters."
        )
    enabled = raw["enabled"]
    if type(enabled) is not bool:
        raise ValueError("Shield profile enabled must be a boolean.")
    raw_config = raw["config"]
    if not isinstance(raw_config, dict):
        raise ValueError("Shield profile config must be a YAML mapping.")
    expected_config_fields = set(SAFETY_SHIELD_CONFIG_ENV_FIELDS.values())
    actual_config_fields = set(raw_config)
    if actual_config_fields != expected_config_fields:
        raise ValueError(
            "Shield profile config must list every non-hardware SafetyShieldConfig "
            "field exactly once; "
            f"missing={sorted(expected_config_fields - actual_config_fields)}, "
            "extra="
            f"{sorted(str(name) for name in actual_config_fields - expected_config_fields)}."
        )
    config_environment: dict[str, str] = {}
    for environment_name, field_name in SAFETY_SHIELD_CONFIG_ENV_FIELDS.items():
        raw_value = raw_config[field_name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"Shield profile config.{field_name} must be a finite number.")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Shield profile config.{field_name} must be a finite number.")
        config_environment[environment_name] = repr(value)
    config = safety_shield_config_from_env(config_environment)
    return WebotsSafetyShieldProfile(
        source_path=profile_path,
        source_sha256=sha256_file(profile_path),
        profile_id=profile_id,
        enabled=enabled,
        config=config,
    )


def _same_float(actual: object, expected: float) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isfinite(float(actual))
        and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    )


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def validate_existing_summary(
    summary_path: Path,
    *,
    request: WebotsCheckpointSweepRequest,
    step: int,
    model_path: Path,
    model_sha256: str,
) -> dict[str, Any]:
    """Enforce the selection contract plus every sweep scenario parameter."""

    gate = CheckpointSelectionGate(
        seed=request.seed,
        condition=request.condition,
        episodes=request.episodes,
        min_success_rate=0.0,
        max_collision_rate=1.0,
        model_variant=request.model_variant,
        shield_profile_token=(
            request.shield_profile.result_token
            if request.shield_profile is not None
            else None
        ),
    )
    identity = parse_result_identity(summary_path)
    if identity.kind != request.result_kind or identity.step != step:
        raise ValueError(
            "Validation summary does not identify the requested "
            f"{request.result_kind} model."
        )
    if request.exact_model_path is None:
        candidate = make_candidate_record(
            summary_path,
            repository_root=request.repository_root,
            gate=gate,
        )
        candidate_model_sha256 = candidate["model_sha256"]
    else:
        candidate = load_strict_candidate_summary(
            summary_path,
            identity,
            gate,
            allow_no_success_metrics=True,
        )
        candidate_model_sha256 = candidate["evaluation_model_sha256"]
    metrics = load_validation_metrics(
        summary_path,
        repository_root=request.repository_root,
        expected_model_path=model_path,
        expected_model_sha256=model_sha256,
        expected_training_seed=request.seed,
        expected_model_variant=request.model_variant,
        expected_condition=request.condition,
        expected_planner=request.planner(step),
        expected_seed_start=request.seed_start,
        expected_episodes=request.episodes,
    )
    raw = _load_mapping(summary_path, "summary")
    provenance = raw.get("evaluation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("summary.evaluation_provenance must be a JSON object.")
    if provenance.get("obstacle_count") != request.obstacle_count:
        raise ValueError("Validation summary obstacle_count does not match the sweep.")
    speed_range = provenance.get("obstacle_speed_range")
    if (
        not isinstance(speed_range, list)
        or len(speed_range) != 2
        or not _same_float(speed_range[0], request.obstacle_speed_min)
        or not _same_float(speed_range[1], request.obstacle_speed_max)
    ):
        raise ValueError("Validation summary obstacle speed range does not match the sweep.")
    if not _same_float(
        provenance.get("horizontal_route_half_length"),
        request.horizontal_route_half_length,
    ):
        raise ValueError("Validation summary horizontal route length does not match the sweep.")
    if not _same_float(
        provenance.get("vertical_route_half_length"),
        request.vertical_route_half_length,
    ):
        raise ValueError("Validation summary vertical route length does not match the sweep.")
    expected_shield_enabled = (
        request.shield_profile.enabled if request.shield_profile is not None else False
    )
    if provenance.get("safety_shield_enabled") is not expected_shield_enabled:
        raise ValueError(
            "Validation summary safety_shield_enabled does not match the sweep."
        )
    if request.shield_profile is not None:
        raw_shield_config = provenance.get("safety_shield_config")
        if not isinstance(raw_shield_config, dict):
            raise ValueError(
                "Explicit shield-profile summaries require safety_shield_config provenance."
            )
        expected_fields = set(SafetyShieldConfig.__dataclass_fields__)
        if set(raw_shield_config) != expected_fields:
            raise ValueError(
                "Validation summary safety_shield_config fields do not match "
                "SafetyShieldConfig."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_shield_config.values()
        ):
            raise ValueError(
                "Validation summary safety_shield_config values must be finite numbers."
            )
        try:
            SafetyShieldConfig(**raw_shield_config)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Validation summary safety_shield_config is invalid."
            ) from error
        for field_name, expected_value in request.shield_profile.config_values.items():
            if not _same_float(raw_shield_config[field_name], expected_value):
                raise ValueError(
                    "Validation summary safety_shield_config does not match "
                    f"profile field {field_name}."
                )
    if provenance.get("static_layout") != "train_a":
        raise ValueError("Validation summary static_layout must be train_a.")

    episodes_path = summary_path.parent / "episodes.csv"
    episodes_sha256 = provenance.get("episodes_csv_sha256")
    if (
        not isinstance(episodes_sha256, str)
        or _SHA256_PATTERN.fullmatch(episodes_sha256) is None
        or not episodes_path.is_file()
        or sha256_file(episodes_path) != episodes_sha256
    ):
        raise ValueError("Validation episodes.csv is missing or fails its SHA-256 binding.")
    if candidate_model_sha256 != model_sha256:
        raise ValueError("Validation candidate model SHA-256 does not match the sweep model.")
    return {
        "episodes": int(metrics["episodes"]),
        "success_rate": float(metrics["success_rate"]),
        "collision_rate": float(metrics["collision_rate"]),
        "success_count": int(candidate["success_count"]),
        "collision_count": int(candidate["collision_count"]),
        "timeout_count": int(candidate["timeout_count"]),
        "error_count": int(candidate["error_count"]),
        "path_length_ratio_mean": (
            None
            if candidate["path_length_ratio_mean"] is None
            else float(candidate["path_length_ratio_mean"])
        ),
        "completion_time_s_mean": (
            None
            if candidate["completion_time_s_mean"] is None
            else float(candidate["completion_time_s_mean"])
        ),
    }


def _clean_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("WEBOTS_")
    }
    environment.update(overrides)
    return environment


def _environment(
    request: WebotsCheckpointSweepRequest,
    *,
    step: int,
    model_path: Path,
    model_sha256: str,
) -> dict[str, str]:
    environment = {
        "WEBOTS_MODEL_SEED": str(request.seed),
        "WEBOTS_EVAL_MODEL": _portable(model_path, request.repository_root),
        "WEBOTS_EVAL_MODEL_SHA256": model_sha256,
        "WEBOTS_EVAL_EPISODES": str(request.episodes),
        "WEBOTS_EVAL_SEED_START": str(request.seed_start),
        "WEBOTS_EVAL_SPLIT": "validation",
        "WEBOTS_EVAL_CONDITION": request.condition,
        "WEBOTS_EVAL_PLANNER": "ppo",
        "WEBOTS_EVAL_PLANNER_LABEL": request.planner(step),
        "WEBOTS_EVAL_RUN_NAME": request.run_name(step),
        "WEBOTS_OBSTACLE_COUNT": str(request.obstacle_count),
        "WEBOTS_OBSTACLE_SPEED_MIN": str(request.obstacle_speed_min),
        "WEBOTS_OBSTACLE_SPEED_MAX": str(request.obstacle_speed_max),
        "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": str(
            request.horizontal_route_half_length
        ),
        "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": str(request.vertical_route_half_length),
        "WEBOTS_SAFETY_SHIELD": (
            "true"
            if request.shield_profile is not None and request.shield_profile.enabled
            else "false"
        ),
    }
    if request.model_variant is not None:
        environment["WEBOTS_MODEL_VARIANT"] = request.model_variant
    if request.shield_profile is not None:
        environment.update(request.shield_profile.controller_environment)
    return _clean_environment(environment)


def _command(request: WebotsCheckpointSweepRequest, port: int) -> list[str]:
    return [
        str(request.webots_executable),
        f"--port={port}",
        "--batch",
        "--mode=fast",
        "--no-rendering",
        "--stdout",
        "--stderr",
        str(request.repository_root / _EVALUATION_WORLD),
    ]


def _write_text(path: Path, value: str | bytes | None) -> None:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value or ""
    path.write_text(text, encoding="utf-8")


def _execute_candidate(
    base_record: Mapping[str, Any],
    *,
    request: WebotsCheckpointSweepRequest,
    output_directory: Path,
    process_runner: ProcessRunner,
) -> dict[str, Any]:
    record = dict(base_record)
    step = int(record["step"])
    port = int(record["port"])
    model_path = request.model_path(step).resolve()
    model_sha256 = str(record["model_sha256"])
    environment = _environment(
        request,
        step=step,
        model_path=model_path,
        model_sha256=model_sha256,
    )
    command = _command(request, port)
    log_identity = (
        f"final_step{step}"
        if request.exact_model_path is not None
        else f"checkpoint{step}"
    )
    stdout_path = output_directory / f"{log_identity}.stdout.log"
    stderr_path = output_directory / f"{log_identity}.stderr.log"
    record.update(
        {
            "status": "running",
            "started_at": utc_now(),
            "command": command,
            "environment": {
                name: environment[name]
                for name in sorted(environment)
                if name.startswith("WEBOTS_")
            },
            "stdout_log": _portable(stdout_path, request.repository_root),
            "stderr_log": _portable(stderr_path, request.repository_root),
        }
    )
    try:
        request.verify_exact_model()
        completed = process_runner(
            command,
            cwd=request.repository_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=request.timeout_seconds,
        )
        _write_text(stdout_path, completed.stdout)
        _write_text(stderr_path, completed.stderr)
        record["exit_code"] = int(completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError(f"Webots exited with code {completed.returncode}.")
        metrics = validate_existing_summary(
            request.repository_root
            / "artifacts/results"
            / request.run_name(step)
            / "summary.json",
            request=request,
            step=step,
            model_path=model_path,
            model_sha256=model_sha256,
        )
        request.verify_exact_model()
        summary_path = (
            request.repository_root
            / "artifacts/results"
            / request.run_name(step)
            / "summary.json"
        )
        record.update(
            {
                "status": "completed",
                "summary_sha256": sha256_file(summary_path),
                "metrics": metrics,
            }
        )
    except subprocess.TimeoutExpired as error:
        _write_text(stdout_path, error.stdout)
        _write_text(stderr_path, error.stderr)
        record.update(
            {
                "status": "failed",
                "exit_code": None,
                "error": f"Webots timed out after {request.timeout_seconds:g} seconds.",
            }
        )
    except (OSError, RuntimeError, ValueError) as error:
        if not stdout_path.exists():
            _write_text(stdout_path, "")
        if not stderr_path.exists():
            _write_text(stderr_path, "")
        record.update({"status": "failed", "error": str(error)})
    record["finished_at"] = utc_now()
    return record


def _base_record(
    request: WebotsCheckpointSweepRequest,
    *,
    step: int,
    model_path: Path,
    model_sha256: str | None,
) -> dict[str, Any]:
    summary_path = (
        request.repository_root
        / "artifacts/results"
        / request.run_name(step)
        / "summary.json"
    )
    return {
        "step": step,
        "run_name": request.run_name(step),
        "planner": request.planner(step),
        "model_path": _portable(model_path, request.repository_root),
        "model_sha256": model_sha256,
        "summary_path": _portable(summary_path, request.repository_root),
    }


def _write_manifest(manifest: Mapping[str, Any], output_directory: Path) -> tuple[Path, Path]:
    json_path = output_directory / "checkpoint_sweep.json"
    yaml_path = output_directory / "checkpoint_sweep.yaml"
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    yaml_path.write_text(
        yaml.safe_dump(dict(manifest), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return json_path, yaml_path


def run_checkpoint_sweep(
    request: WebotsCheckpointSweepRequest,
    *,
    output_directory: Path,
    process_runner: ProcessRunner = subprocess.run,
    port_allocator: PortAllocator = choose_available_port,
) -> dict[str, Any]:
    """Reuse strict summaries and execute all remaining checkpoints."""

    repository_root = request.repository_root.resolve()
    # Exact endpoint identity is fail-fast: a missing or drifted model must be
    # rejected before output setup, port allocation, or any Webots subprocess.
    request.verify_exact_model()
    executable = request.webots_executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Webots executable does not exist: {executable}")
    world_path = repository_root / _EVALUATION_WORLD
    controller_path = repository_root / _EVALUATOR_CONTROLLER
    if not world_path.is_file():
        raise FileNotFoundError(f"Webots evaluation world does not exist: {world_path}")
    if not controller_path.is_file():
        raise FileNotFoundError(f"Webots evaluator does not exist: {controller_path}")
    model_bindings: list[tuple[int, Path, str | None]] = []
    for step in request.steps:
        model_path = request.model_path(step).resolve()
        if not model_path.is_file():
            model_bindings.append((step, model_path, None))
            continue
        actual_model_sha256 = sha256_file(model_path)
        if request.exact_model_path is not None:
            assert request.exact_model_sha256 is not None
            if actual_model_sha256 != request.exact_model_sha256:
                raise ValueError(
                    "Exact model SHA-256 mismatch: "
                    f"{actual_model_sha256} != {request.exact_model_sha256}."
                )
            model_sha256 = request.exact_model_sha256
        else:
            model_sha256 = actual_model_sha256
        model_bindings.append((step, model_path, model_sha256))

    request.verify_exact_model()
    output_directory = _resolve_inside_repository(
        output_directory,
        repository_root,
        "output_directory",
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    reserved_ports: set[int] = set()
    records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for index, (step, model_path, model_sha256) in enumerate(model_bindings):
        record = _base_record(
            request,
            step=step,
            model_path=model_path,
            model_sha256=model_sha256,
        )
        if model_sha256 is None:
            record.update({"status": "failed", "error": "Checkpoint model is missing."})
            records.append(record)
            continue
        summary_path = repository_root / str(record["summary_path"])
        if summary_path.is_file():
            try:
                request.verify_exact_model()
                metrics = validate_existing_summary(
                    summary_path,
                    request=request,
                    step=step,
                    model_path=model_path,
                    model_sha256=model_sha256,
                )
                request.verify_exact_model()
            except (OSError, ValueError) as error:
                record["prior_summary_rejection"] = str(error)
            else:
                record.update(
                    {
                        "status": "reused",
                        "summary_sha256": sha256_file(summary_path),
                        "metrics": metrics,
                    }
                )
                records.append(record)
                continue
        request.verify_exact_model()
        record["port"] = port_allocator(request.port_base + index, reserved_ports)
        pending.append(record)

    if pending:
        with ThreadPoolExecutor(max_workers=min(request.workers, len(pending))) as executor:
            futures = {
                executor.submit(
                    _execute_candidate,
                    record,
                    request=request,
                    output_directory=output_directory,
                    process_runner=process_runner,
                ): int(record["step"])
                for record in pending
            }
            for future in as_completed(futures):
                step = futures[future]
                try:
                    records.append(future.result())
                except Exception as error:  # pragma: no cover - final worker guard
                    fallback = next(record for record in pending if record["step"] == step)
                    failed = dict(fallback)
                    failed.update({"status": "failed", "error": str(error)})
                    records.append(failed)

    records.sort(key=lambda record: int(record["step"]))
    status = (
        "completed"
        if all(record["status"] in {"completed", "reused"} for record in records)
        else "failed"
    )
    request_record: dict[str, Any] = {
        "seed": request.seed,
        "condition": request.condition,
        "model_variant": request.model_variant,
        "steps": list(request.steps),
        "episodes": request.episodes,
        "seed_start": request.seed_start,
        "scenario_seeds": list(
            range(request.seed_start, request.seed_start + request.episodes)
        ),
        "obstacle_count": request.obstacle_count,
        "obstacle_speed_range": [
            request.obstacle_speed_min,
            request.obstacle_speed_max,
        ],
        "horizontal_route_half_length": request.horizontal_route_half_length,
        "vertical_route_half_length": request.vertical_route_half_length,
        "safety_shield_enabled": (
            request.shield_profile.enabled
            if request.shield_profile is not None
            else False
        ),
        "static_layout": "train_a",
        "action_repeat": 4,
        "max_episode_steps": 300,
        "port_base": request.port_base,
        "workers": request.workers,
        "timeout_seconds": request.timeout_seconds,
    }
    if request.shield_profile is not None:
        request_record["safety_shield_profile"] = (
            request.shield_profile.manifest_record(repository_root)
        )
    if request.exact_model_path is not None:
        request_record["result_kind"] = "final"
        request_record["exact_model_path"] = _portable(
            request.exact_model_path,
            repository_root,
        )
        request_record["exact_model_sha256"] = request.exact_model_sha256

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "webots_checkpoint_validation_sweep",
        "status": status,
        "split": "validation",
        "started_at": started_at,
        "finished_at": utc_now(),
        "request": request_record,
        "runtime": {
            "webots_executable": str(executable),
            "evaluation_world": _portable(world_path, repository_root),
            "evaluation_world_sha256": sha256_file(world_path),
            "evaluator_controller": _portable(controller_path, repository_root),
            "evaluator_controller_sha256": sha256_file(controller_path),
        },
        "runs": records,
    }
    manifest["manifest_paths"] = {
        "json": _portable(output_directory / "checkpoint_sweep.json", repository_root),
        "yaml": _portable(output_directory / "checkpoint_sweep.yaml", repository_root),
    }
    _write_manifest(manifest, output_directory)
    return manifest


def sweep_exit_code(manifest: Mapping[str, Any]) -> int:
    """Return non-zero unless every requested checkpoint has a valid summary."""

    return 0 if manifest.get("status") == "completed" else 2
