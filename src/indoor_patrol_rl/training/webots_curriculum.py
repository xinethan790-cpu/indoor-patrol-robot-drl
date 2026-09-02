"""Restartable orchestration for staged Webots PPO training and validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from indoor_patrol_rl.core.safety_shield import (
    SafetyShieldConfig,
    safety_shield_config_from_env,
)

_MODEL_STEP_PATTERN = re.compile(r"step(?P<steps>\d+)\.zip$")
_CHECKPOINT_STEP_PATTERN = re.compile(r"_(?P<steps>\d+)_steps\.zip$")
_MODEL_VARIANT_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
_EVALUATION_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_CONTROLLER_ENV_PATTERN = re.compile(r"WEBOTS_[A-Z0-9_]+")
_RUN_MANIFEST_SCHEMA_VERSION = 2
_DEFAULT_TRAINING_WORLD = "webots_ppo_train.wbt"
_EVALUATION_WORLD = "webots_ppo_evaluate.wbt"
_EVALUATION_SHARED_ENV = frozenset(
    {
        "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH",
        "WEBOTS_OBSTACLE_SPEED_MAX",
        "WEBOTS_OBSTACLE_SPEED_MIN",
        "WEBOTS_SAFETY_SHIELD",
        "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH",
    }
)
_EXPECTED_NOT_SET = object()


@dataclass(frozen=True)
class ValidationGate:
    """Acceptance thresholds evaluated only on the validation split."""

    seed_start: int
    episodes: int
    min_success_rate: float
    max_collision_rate: float

    def __post_init__(self) -> None:
        if self.seed_start < 0 or self.episodes <= 0:
            raise ValueError("Validation seed_start must be non-negative and episodes positive.")
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between zero and one.")
        if not 0.0 <= self.max_collision_rate <= 1.0:
            raise ValueError("max_collision_rate must be between zero and one.")


@dataclass(frozen=True)
class WebotsStage:
    """One simulator training segment followed by a validation gate."""

    stage_id: str
    name: str
    obstacle_count: int
    training_steps: int
    gate: ValidationGate
    controller_env: Mapping[str, str] | None = None
    preferred_resume_model: Path | None = None
    preferred_resume_sha256: str | None = None
    world: str = _DEFAULT_TRAINING_WORLD

    def __post_init__(self) -> None:
        if not self.stage_id or not self.name:
            raise ValueError("A Webots curriculum stage must have an id and name.")
        if re.fullmatch(r"[A-Za-z0-9]+", self.stage_id) is None:
            raise ValueError(
                "A Webots curriculum stage id must contain only letters and digits so "
                "its validation result can be consumed by checkpoint selection."
            )
        if not 0 <= self.obstacle_count <= 3:
            raise ValueError("Webots obstacle_count must be between zero and three.")
        if self.training_steps <= 0:
            raise ValueError("Webots training_steps must be positive.")
        if self.controller_env is not None and not isinstance(self.controller_env, Mapping):
            raise ValueError("controller_env must be a mapping.")
        if self.preferred_resume_sha256 is not None:
            if self.preferred_resume_model is None:
                raise ValueError(
                    "preferred_resume_sha256 requires preferred_resume_model."
                )
            if _SHA256_PATTERN.fullmatch(self.preferred_resume_sha256) is None:
                raise ValueError("preferred_resume_sha256 must be a full SHA-256 digest.")
            object.__setattr__(
                self,
                "preferred_resume_sha256",
                self.preferred_resume_sha256.casefold(),
            )
        if not isinstance(self.world, str) or not self.world:
            raise ValueError("A Webots curriculum stage world must be a non-empty string.")
        world = self.world.replace("\\", "/")
        world_path = Path(world)
        if (
            world in {".", ".."}
            or world_path.is_absolute()
            or ".." in world_path.parts
            or re.match(r"^[A-Za-z]:/", world)
        ):
            raise ValueError("A Webots curriculum stage world must be relative to worlds/.")
        object.__setattr__(self, "world", world_path.as_posix())
        controller_env = normalize_controller_env(self.controller_env or {})
        object.__setattr__(self, "controller_env", controller_env)


@dataclass(frozen=True)
class WebotsCurriculumConfig:
    """Validated configuration for the external Webots process orchestrator."""

    transfer_model: Path
    port_base: int
    stages: tuple[WebotsStage, ...]
    model_variant: str | None = None
    policy_transfer_model: Path | None = None
    warm_start_requirement: Path | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not 1024 <= self.port_base <= 65500:
            raise ValueError("port_base must be between 1024 and 65500.")
        if not self.stages:
            raise ValueError("The Webots curriculum must contain at least one stage.")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("Webots curriculum stage ids must be unique.")
        _validate_model_variant(self.model_variant)
        for stage in self.stages:
            stage_variant = stage.controller_env.get("WEBOTS_MODEL_VARIANT")
            if stage_variant is not None and stage_variant != self.model_variant:
                raise ValueError(
                    f"Stage {stage.stage_id} WEBOTS_MODEL_VARIANT must match model_variant."
                )


@dataclass(frozen=True)
class ModelSnapshot:
    """A resumable model artifact and the cumulative Webots step it represents."""

    path: Path
    steps: int
    obstacle_count: int | None = None
    metadata_path: Path | None = None
    is_completed_run: bool = False
    model_variant: str | None = None
    starting_steps: int | None = None
    additional_steps_requested: int | None = None
    obstacle_speed_range: tuple[float, float] | None = None
    route_half_lengths: tuple[float, float] | None = None
    monitor_log: str | None = None
    modified_at_ns: int | None = None


def utc_now() -> str:
    """Return a stable UTC timestamp suitable for manifests."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _validate_model_variant(model_variant: str | None) -> None:
    if model_variant is None:
        return
    if not isinstance(model_variant, str):
        raise ValueError("model_variant must be a string when provided.")
    if (
        model_variant in {"", ".", ".."}
        or "/" in model_variant
        or "\\" in model_variant
        or _MODEL_VARIANT_PATTERN.fullmatch(model_variant) is None
    ):
        raise ValueError(
            "model_variant must be a non-empty path-safe name containing only "
            "letters, digits, '.', '_' and '-'."
        )


def validate_evaluation_run_name(value: str) -> str:
    """Return one path-safe result directory name, rejecting dot-directory escapes."""

    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or "/" in value
        or "\\" in value
        or _EVALUATION_RUN_NAME_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            "WEBOTS_EVAL_RUN_NAME must be one non-empty path-safe directory name."
        )
    return value


def _environment_value(value: Any, name: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)) and not isinstance(value, complex):
        return str(value)
    raise ValueError(f"controller_env value for {name} must be a string, number, or boolean.")


def _environment_flag_value(
    environ: Mapping[str, str],
    name: str,
    *,
    default: bool = False,
) -> bool:
    """Mirror the Webots controller's strict environment-boolean parser."""

    raw_value = environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag.")


def normalize_controller_env(raw: Mapping[str, Any]) -> dict[str, str]:
    """Validate controller overrides and make held-out test execution impossible."""

    normalized: dict[str, str] = {}
    for raw_name, raw_value in raw.items():
        name = str(raw_name)
        if _CONTROLLER_ENV_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"controller_env key {name!r} is invalid; only uppercase WEBOTS_* is allowed."
            )
        if name == "WEBOTS_CONFIRM_TEST":
            raise ValueError("controller_env must never set WEBOTS_CONFIRM_TEST.")
        value = _environment_value(raw_value, name)
        if name == "WEBOTS_EVAL_SPLIT" and value.casefold() != "validation":
            raise ValueError("controller_env WEBOTS_EVAL_SPLIT may only be validation.")
        normalized[name] = value
    return normalized


def evaluation_controller_env(controller_env: Mapping[str, str]) -> dict[str, str]:
    """Select scenario/evaluation values that the evaluator actually consumes."""

    return {
        name: value
        for name, value in controller_env.items()
        if name in _EVALUATION_SHARED_ENV
        or name.startswith("WEBOTS_EVAL_")
        or name.startswith("WEBOTS_SAFETY_SHIELD_")
    }


def load_webots_curriculum(path: Path) -> WebotsCurriculumConfig:
    """Load the simulator curriculum and reject any automatic test split."""

    with path.open(encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), "Webots curriculum")
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported Webots curriculum schema_version.")
    validation = _mapping(raw.get("validation"), "validation")
    split = str(validation.get("split", "validation"))
    if split != "validation":
        raise ValueError("The Webots curriculum may only run the validation split automatically.")
    default_gate = ValidationGate(
        seed_start=int(validation["seed_start"]),
        episodes=int(validation["episodes"]),
        min_success_rate=float(validation["min_success_rate"]),
        max_collision_rate=float(validation["max_collision_rate"]),
    )
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list):
        raise ValueError("Webots curriculum stages must be a list.")
    stages = []
    for raw_stage_value in raw_stages:
        raw_stage = _mapping(raw_stage_value, "stage")
        raw_gate_value = raw_stage.get("validation")
        if raw_gate_value is None:
            gate = default_gate
        else:
            raw_gate = _mapping(raw_gate_value, "stage.validation")
            gate = ValidationGate(
                seed_start=int(raw_gate.get("seed_start", default_gate.seed_start)),
                episodes=int(raw_gate.get("episodes", default_gate.episodes)),
                min_success_rate=float(
                    raw_gate.get("min_success_rate", default_gate.min_success_rate)
                ),
                max_collision_rate=float(
                    raw_gate.get("max_collision_rate", default_gate.max_collision_rate)
                ),
            )
        controller_env_value = raw_stage.get("controller_env", {})
        controller_env = normalize_controller_env(
            _mapping(controller_env_value, "stage.controller_env")
        )
        stages.append(
            WebotsStage(
                stage_id=str(raw_stage["id"]),
                name=str(raw_stage["name"]),
                obstacle_count=int(raw_stage["obstacle_count"]),
                training_steps=int(raw_stage["training_steps"]),
                gate=gate,
                controller_env=controller_env,
                preferred_resume_model=(
                    Path(str(raw_stage["preferred_resume_model"]))
                    if raw_stage.get("preferred_resume_model")
                    else None
                ),
                preferred_resume_sha256=(
                    str(raw_stage["preferred_resume_sha256"])
                    if raw_stage.get("preferred_resume_sha256")
                    else None
                ),
                world=raw_stage.get("world", _DEFAULT_TRAINING_WORLD),
            )
        )
    model_variant_value = raw.get("model_variant")
    if model_variant_value is not None and not isinstance(model_variant_value, str):
        raise ValueError("model_variant must be a string when provided.")
    model_variant = model_variant_value
    policy_transfer_value = raw.get("policy_transfer_model")
    warm_start_requirement_value = raw.get("warm_start_requirement")
    return WebotsCurriculumConfig(
        transfer_model=Path(str(raw["transfer_model"])),
        port_base=int(raw.get("port_base", 12490)),
        stages=tuple(stages),
        model_variant=model_variant,
        policy_transfer_model=(
            Path(str(policy_transfer_value)) if policy_transfer_value else None
        ),
        warm_start_requirement=(
            Path(str(warm_start_requirement_value))
            if warm_start_requirement_value
            else None
        ),
        source_path=path,
    )


def apply_preferred_resume_overrides(
    config: WebotsCurriculumConfig, entries: Sequence[str]
) -> WebotsCurriculumConfig:
    """Apply repeatable ``STAGE=MODEL`` CLI overrides without mutating the config."""

    overrides: dict[str, Path] = {}
    stage_ids = {stage.stage_id.casefold(): stage.stage_id for stage in config.stages}
    for entry in entries:
        raw_stage_id, separator, raw_path = entry.partition("=")
        stage_id = stage_ids.get(raw_stage_id.casefold())
        if not separator or stage_id is None or not raw_path:
            raise ValueError(
                f"Invalid preferred resume override {entry!r}; expected STAGE=MODEL."
            )
        if stage_id in overrides:
            raise ValueError(f"Duplicate preferred resume override for {stage_id}.")
        overrides[stage_id] = Path(raw_path)
    if not overrides:
        return config
    stages = tuple(
        replace(
            stage,
            preferred_resume_model=overrides.get(
                stage.stage_id, stage.preferred_resume_model
            ),
        )
        for stage in config.stages
    )
    return replace(config, stages=stages)


def resolve_stage_index(stages: Sequence[WebotsStage], value: str | None, default: int) -> int:
    """Resolve a stage selector by case-insensitive id or zero-based index."""

    if value is None:
        return default
    if value.isdigit():
        index = int(value)
        if 0 <= index < len(stages):
            return index
    for index, stage in enumerate(stages):
        if value.casefold() == stage.stage_id.casefold():
            return index
    raise ValueError(f"Unknown Webots curriculum stage: {value}")


def resolve_webots_executable(explicit: Path | None = None) -> Path:
    """Resolve Webots on macOS, Linux, or Windows, with an explicit override."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
        if explicit.suffix == ".app" or explicit.name.endswith(".app"):
            candidates.append(explicit / "Contents/MacOS/webots")
    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        home = Path(webots_home)
        candidates.extend(
            (
                home / "webots",
                home / "webots.exe",
                home / "Contents/MacOS/webots",
                home / "msys64/mingw64/bin/webots.exe",
            )
        )
    discovered = shutil.which("webots") or shutil.which("webots.exe")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(
        (
            Path("/Applications/Webots.app/Contents/MacOS/webots"),
            Path("/usr/local/webots/webots"),
            Path("/snap/bin/webots"),
            Path("C:/Program Files/Webots/msys64/mingw64/bin/webots.exe"),
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    if explicit is not None:
        raise FileNotFoundError(f"Webots executable not found at {explicit}")
    raise FileNotFoundError("Webots executable was not found; pass --webots explicitly.")


def choose_available_port(preferred: int, reserved: set[int] | None = None) -> int:
    """Choose an unused localhost TCP port without reusing one in this run."""

    reserved = reserved if reserved is not None else set()
    for port in range(preferred, min(preferred + 200, 65536)):
        if port in reserved:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        reserved.add(port)
        return port
    raise RuntimeError(f"No available Webots port near {preferred}.")


def _path_from_metadata(value: object, repository_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repository_root / path


def _model_directory(repository_root: Path, seed: int, model_variant: str | None) -> Path:
    directory = repository_root / "artifacts/models/webots" / f"seed_{seed}"
    return directory / model_variant if model_variant else directory


def _metadata_float_pair(
    raw: Mapping[str, Any], name: str, first: str | None = None, second: str | None = None
) -> tuple[float, float] | None:
    value = raw.get(name)
    if first is not None and second is not None and isinstance(value, dict):
        try:
            return float(value[first]), float(value[second])
        except (KeyError, TypeError, ValueError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def discover_model_snapshots(
    repository_root: Path,
    seed: int,
    model_variant: str | None = None,
) -> tuple[ModelSnapshot, ...]:
    """Discover models recursively inside exactly one variant, never across variants."""

    model_directory = _model_directory(repository_root, seed, model_variant)
    snapshots: dict[tuple[Path, int], ModelSnapshot] = {}
    if model_variant:
        metadata_paths = sorted(model_directory.rglob("run_metadata_step*.yaml"))
        latest_metadata_paths = sorted(model_directory.rglob("run_metadata.yaml"))
        final_model_paths = model_directory.rglob("ppo_webots_s*_step*.zip")
        checkpoint_paths = model_directory.rglob("checkpoints/*.zip")
    else:
        metadata_paths = sorted(model_directory.glob("run_metadata_step*.yaml"))
        latest_metadata_paths = [model_directory / "run_metadata.yaml"]
        final_model_paths = model_directory.glob("ppo_webots_s*_step*.zip")
        checkpoint_paths = (model_directory / "checkpoints").rglob("*.zip")
    metadata_paths.extend(path for path in latest_metadata_paths if path.is_file())
    for metadata_path in metadata_paths:
        try:
            raw = _mapping(yaml.safe_load(metadata_path.read_text(encoding="utf-8")), "metadata")
            final_model = _path_from_metadata(raw["final_model"], repository_root)
            steps = int(raw["final_model_steps"])
            obstacle_count = int(raw["obstacle_count"])
            metadata_variant_value = raw.get("model_variant")
            metadata_variant = (
                str(metadata_variant_value) if metadata_variant_value else None
            )
            if metadata_variant != model_variant:
                continue
            if model_variant:
                final_model.absolute().relative_to(model_directory.absolute())
            elif final_model.absolute().parent != model_directory.absolute():
                continue
        except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError):
            continue
        if final_model.is_file():
            snapshot = ModelSnapshot(
                path=final_model,
                steps=steps,
                obstacle_count=obstacle_count,
                metadata_path=metadata_path,
                is_completed_run=True,
                model_variant=metadata_variant,
                starting_steps=(
                    int(raw["starting_webots_steps"])
                    if raw.get("starting_webots_steps") is not None
                    else None
                ),
                additional_steps_requested=(
                    int(raw["additional_steps_requested"])
                    if raw.get("additional_steps_requested") is not None
                    else None
                ),
                obstacle_speed_range=_metadata_float_pair(raw, "obstacle_speed_range"),
                route_half_lengths=_metadata_float_pair(
                    raw, "route_half_lengths", "horizontal", "vertical"
                ),
                monitor_log=(str(raw["monitor_log"]) if raw.get("monitor_log") else None),
                modified_at_ns=final_model.stat().st_mtime_ns,
            )
            snapshots[(snapshot.path, snapshot.steps)] = snapshot
    for model_path in final_model_paths:
        match = _MODEL_STEP_PATTERN.search(model_path.name)
        if match:
            steps = int(match.group("steps"))
            snapshots.setdefault(
                (model_path, steps),
                ModelSnapshot(
                    model_path,
                    steps,
                    model_variant=model_variant,
                    modified_at_ns=model_path.stat().st_mtime_ns,
                ),
            )
    for model_path in checkpoint_paths:
        match = _CHECKPOINT_STEP_PATTERN.search(model_path.name)
        if match:
            steps = int(match.group("steps"))
            snapshots.setdefault(
                (model_path, steps),
                ModelSnapshot(
                    model_path,
                    steps,
                    model_variant=model_variant,
                    modified_at_ns=model_path.stat().st_mtime_ns,
                ),
            )
    return tuple(sorted(snapshots.values(), key=lambda item: (item.steps, item.is_completed_run)))


def _environment_float(stage: WebotsStage, name: str) -> float | None:
    value = stage.controller_env.get(name)
    return float(value) if value is not None else None


def snapshot_matches_stage(snapshot: ModelSnapshot, stage: WebotsStage) -> bool:
    """Match same-obstacle variant stages using their recorded scenario signature."""

    if not snapshot.is_completed_run or snapshot.obstacle_count != stage.obstacle_count:
        return False
    expected_speed = (
        _environment_float(stage, "WEBOTS_OBSTACLE_SPEED_MIN"),
        _environment_float(stage, "WEBOTS_OBSTACLE_SPEED_MAX"),
    )
    if any(value is not None for value in expected_speed):
        if None in expected_speed or snapshot.obstacle_speed_range is None:
            return False
        if not all(
            math.isclose(float(expected), actual, rel_tol=0.0, abs_tol=1e-9)
            for expected, actual in zip(expected_speed, snapshot.obstacle_speed_range, strict=True)
        ):
            return False
    expected_route = (
        _environment_float(stage, "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH"),
        _environment_float(stage, "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH"),
    )
    if any(value is not None for value in expected_route):
        if None in expected_route or snapshot.route_half_lengths is None:
            return False
        if not all(
            math.isclose(float(expected), actual, rel_tol=0.0, abs_tol=1e-9)
            for expected, actual in zip(expected_route, snapshot.route_half_lengths, strict=True)
        ):
            return False
    expected_run_id = stage.controller_env.get("WEBOTS_RUN_ID")
    if expected_run_id and (
        snapshot.monitor_log is None
        or f"training_{expected_run_id}.monitor.csv" not in snapshot.monitor_log
    ):
        return False
    return True


def select_completed_snapshot(
    snapshots: Sequence[ModelSnapshot],
    *,
    obstacle_count: int,
    target_steps: int,
) -> ModelSnapshot | None:
    """Return the earliest recorded completed run that reaches a stage target."""

    matches = [
        snapshot
        for snapshot in snapshots
        if snapshot.is_completed_run
        and snapshot.obstacle_count == obstacle_count
        and snapshot.steps >= target_steps
    ]
    return min(matches, key=lambda item: item.steps, default=None)


def select_stage_completed_snapshot(
    snapshots: Sequence[ModelSnapshot],
    *,
    stage: WebotsStage,
    target_steps: int,
) -> ModelSnapshot | None:
    """Select a completed stage by signature only after it reaches the target."""

    matches = [
        snapshot
        for snapshot in snapshots
        if snapshot_matches_stage(snapshot, stage)
        and snapshot.steps >= target_steps
    ]
    return min(matches, key=lambda item: item.steps, default=None)


def model_steps_from_path(path: Path) -> int | None:
    """Read the cumulative step encoded in an immutable model/checkpoint filename."""

    for pattern in (_MODEL_STEP_PATTERN, _CHECKPOINT_STEP_PATTERN):
        match = pattern.search(path.name)
        if match:
            return int(match.group("steps"))
    return None


def _timestamp_ns(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value).timestamp() * 1_000_000_000)
    except ValueError:
        return None


def relative_artifact_path(path: Path, repository_root: Path) -> str:
    """Use repository-relative paths in controller environment variables when possible."""

    try:
        return str(path.absolute().relative_to(repository_root.absolute()))
    except ValueError:
        return str(path.absolute())


def _semantic_relative_path(path: Path | None, repository_root: Path) -> str | None:
    """Normalize a configured artifact path relative to the repository for hashing."""

    if path is None:
        return None
    root = repository_root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = os.path.relpath(candidate.resolve(), root)
    except ValueError as error:
        raise ValueError(
            f"Configured path {path!s} cannot be made relative to repository_root."
        ) from error
    return relative.replace("\\", "/")


def _semantic_config_sha256(payload: Mapping[str, Any]) -> str:
    """Hash semantic configuration with a stable, whitespace-free JSON encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validation_metrics(
    summary_path: Path,
    *,
    repository_root: Path | None = None,
    expected_model_path: Path | None = None,
    expected_model_sha256: str | None = None,
    expected_training_seed: int | None = None,
    expected_model_variant: object = _EXPECTED_NOT_SET,
    expected_condition: str | None = None,
    expected_planner: str | None = None,
    expected_seed_start: int | None = None,
    expected_episodes: int | None = None,
    expected_safety_shield_enabled: object = _EXPECTED_NOT_SET,
    expected_safety_shield_config: object = _EXPECTED_NOT_SET,
) -> dict[str, float | int]:
    """Read validation metrics and, when requested, enforce full evaluation provenance.

    The optional strict contract keeps the standalone parser backward compatible while
    allowing the curriculum runner to reject a fresh-looking summary produced by the
    wrong model, seed, condition, variant, scenario seed range, or safety profile.
    """

    with summary_path.open(encoding="utf-8") as stream:
        raw = _mapping(json.load(stream), "Webots validation summary")
    if raw.get("split") != "validation":
        raise ValueError("Refusing to gate on a non-validation Webots result.")
    planners = raw.get("by_planner")
    if not isinstance(planners, list) or len(planners) != 1:
        raise ValueError("Webots validation summary must contain exactly one planner.")
    planner = _mapping(planners[0], "Webots planner summary")
    metrics: dict[str, float | int] = {
        "episodes": int(planner["episodes"]),
        "success_rate": float(planner["success_rate"]),
        "collision_rate": float(planner["collision_rate"]),
    }
    if metrics["episodes"] <= 0:
        raise ValueError("Webots validation summary contains no episodes.")

    strict_contract = any(
        value is not None
        for value in (
            repository_root,
            expected_model_path,
            expected_model_sha256,
            expected_training_seed,
            expected_condition,
            expected_planner,
            expected_seed_start,
            expected_episodes,
        )
    ) or any(
        value is not _EXPECTED_NOT_SET
        for value in (
            expected_model_variant,
            expected_safety_shield_enabled,
            expected_safety_shield_config,
        )
    )
    if not strict_contract:
        return metrics
    required = {
        "repository_root": repository_root,
        "expected_model_path": expected_model_path,
        "expected_training_seed": expected_training_seed,
        "expected_condition": expected_condition,
        "expected_planner": expected_planner,
        "expected_seed_start": expected_seed_start,
        "expected_episodes": expected_episodes,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing or expected_model_variant is _EXPECTED_NOT_SET:
        missing.extend(
            ["expected_model_variant"]
            if expected_model_variant is _EXPECTED_NOT_SET
            else []
        )
        raise ValueError(f"Strict Webots validation contract is incomplete: {missing}")

    assert repository_root is not None
    assert expected_model_path is not None
    assert expected_seed_start is not None
    assert expected_episodes is not None
    root = repository_root.resolve()
    model_path = expected_model_path.resolve()
    try:
        model_path.relative_to(root)
    except ValueError as error:
        raise ValueError("The expected validation model must be inside the repository.") from error
    if not model_path.is_file():
        raise FileNotFoundError(f"Expected validation model does not exist: {model_path}")
    actual_model_sha256 = _sha256_file(model_path)
    if expected_model_sha256 is not None and (
        _SHA256_PATTERN.fullmatch(expected_model_sha256) is None
        or actual_model_sha256 != expected_model_sha256.casefold()
    ):
        raise ValueError("The validation model changed after evaluation was launched.")

    provenance = _mapping(raw.get("evaluation_provenance"), "evaluation_provenance")
    if provenance.get("schema_version") != 1:
        raise ValueError("evaluation_provenance.schema_version must be 1.")
    raw_model_path = provenance.get("model_path")
    if not isinstance(raw_model_path, str) or not raw_model_path:
        raise ValueError("evaluation_provenance.model_path must be a non-empty string.")
    provenance_model_path = Path(raw_model_path)
    if provenance_model_path.is_absolute():
        raise ValueError("evaluation_provenance.model_path must be repository-relative.")
    resolved_provenance_model = (root / provenance_model_path).resolve()
    try:
        resolved_provenance_model.relative_to(root)
    except ValueError as error:
        raise ValueError("evaluation_provenance.model_path escapes the repository.") from error
    if resolved_provenance_model != model_path:
        raise ValueError("Validation summary was produced by a different model path.")
    provenance_sha256 = provenance.get("model_sha256")
    if not isinstance(provenance_sha256, str) or (
        _SHA256_PATTERN.fullmatch(provenance_sha256) is None
        or provenance_sha256.casefold() != actual_model_sha256
    ):
        raise ValueError("Validation summary model SHA-256 does not match the expected model.")
    training_seed = provenance.get("training_seed")
    if isinstance(training_seed, bool) or training_seed != expected_training_seed:
        raise ValueError("Validation summary training_seed does not match the curriculum seed.")
    if provenance.get("model_variant") != expected_model_variant:
        raise ValueError("Validation summary model_variant does not match the curriculum.")
    if provenance.get("split") != "validation":
        raise ValueError("Validation provenance split must be validation.")
    if provenance.get("condition_id") != expected_condition:
        raise ValueError("Validation summary condition does not match the curriculum stage.")

    if expected_safety_shield_enabled is not _EXPECTED_NOT_SET:
        if type(expected_safety_shield_enabled) is not bool:
            raise ValueError("Expected safety_shield_enabled must be a boolean.")
        if provenance.get("safety_shield_enabled") is not expected_safety_shield_enabled:
            raise ValueError(
                "Validation summary safety_shield_enabled does not match the curriculum."
            )
    if expected_safety_shield_config is not _EXPECTED_NOT_SET:
        if not isinstance(expected_safety_shield_config, Mapping):
            raise ValueError("Expected safety_shield_config must be a mapping.")
        raw_shield_config = provenance.get("safety_shield_config")
        if not isinstance(raw_shield_config, dict):
            raise ValueError(
                "Validation summary safety_shield_config must be a JSON object."
            )
        expected_fields = set(SafetyShieldConfig.__dataclass_fields__)
        if set(raw_shield_config) != expected_fields:
            raise ValueError(
                "Validation summary safety_shield_config fields do not match "
                "SafetyShieldConfig."
            )
        if set(expected_safety_shield_config) != expected_fields:
            raise ValueError(
                "Expected safety_shield_config fields do not match SafetyShieldConfig."
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
        for field_name, expected_value in expected_safety_shield_config.items():
            if (
                isinstance(expected_value, bool)
                or not isinstance(expected_value, (int, float))
                or not math.isfinite(float(expected_value))
                or not math.isclose(
                    float(raw_shield_config[field_name]),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Validation summary safety_shield_config does not match "
                    f"curriculum field {field_name}."
                )

    expected_seed_values = list(
        range(
            expected_seed_start,
            expected_seed_start + expected_episodes,
        )
    )
    if provenance.get("scenario_seed_start") != expected_seed_start:
        raise ValueError("Validation summary scenario_seed_start does not match the gate.")
    if provenance.get("scenario_seed_count") != expected_episodes:
        raise ValueError("Validation summary scenario_seed_count does not match the gate.")
    if provenance.get("scenario_seeds") != expected_seed_values:
        raise ValueError("Validation summary scenario_seeds do not match the fixed gate seeds.")

    if raw.get("episode_count") != expected_episodes or metrics["episodes"] != expected_episodes:
        raise ValueError("Validation summary episode count does not exactly match the gate.")
    if raw.get("planners") != [expected_planner]:
        raise ValueError("Validation summary planner does not match the checkpoint identity.")
    if raw.get("conditions") != [expected_condition]:
        raise ValueError("Validation summary condition list does not match the stage.")
    if planner.get("planner") != expected_planner:
        raise ValueError("Validation aggregate planner does not match the checkpoint identity.")
    by_condition = raw.get("by_planner_condition")
    if not isinstance(by_condition, list) or len(by_condition) != 1:
        raise ValueError("Validation summary must contain one planner-condition aggregate.")
    condition_group = _mapping(by_condition[0], "Webots planner-condition summary")
    if (
        condition_group.get("planner") != expected_planner
        or condition_group.get("condition_id") != expected_condition
        or condition_group.get("episodes") != expected_episodes
    ):
        raise ValueError("Validation planner-condition aggregate does not match the gate.")
    return metrics


def gate_passed(metrics: Mapping[str, float | int], gate: ValidationGate) -> bool:
    """Evaluate the success and collision thresholds."""

    return (
        int(metrics["episodes"]) >= gate.episodes
        and float(metrics["success_rate"]) >= gate.min_success_rate
        and float(metrics["collision_rate"]) <= gate.max_collision_rate
    )


ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]


class WebotsCurriculumRunner:
    """Run training/evaluation processes and persist restart state after each action."""

    def __init__(
        self,
        *,
        repository_root: Path,
        config: WebotsCurriculumConfig,
        webots_executable: Path,
        seed: int,
        manifest_path: Path,
        process_runner: ProcessRunner = subprocess.run,
        dry_run: bool = False,
        success_rate_override: float | None = None,
        collision_rate_override: float | None = None,
        episodes_override: int | None = None,
        seed_start_override: int | None = None,
    ) -> None:
        self.repository_root = repository_root.absolute()
        self.config = config
        from indoor_patrol_rl.training.warm_start_preflight import (
            verify_warm_start_requirement,
        )

        self.warm_start_verification = verify_warm_start_requirement(
            repository_root=self.repository_root,
            config=self.config,
        )
        self.webots_executable = webots_executable.absolute()
        self.seed = seed
        manifest_path = manifest_path.absolute()
        if manifest_path.suffix.casefold() not in {".yaml", ".yml", ".json"}:
            manifest_path = manifest_path.with_suffix(".yaml")
        self.manifest_path = manifest_path
        self.process_runner = process_runner
        self.dry_run = dry_run
        self.success_rate_override = success_rate_override
        self.collision_rate_override = collision_rate_override
        self.episodes_override = episodes_override
        self.seed_start_override = seed_start_override
        self.reserved_ports: set[int] = set()
        self.curriculum_config = self._semantic_config()
        self.curriculum_config_sha256 = _semantic_config_sha256(
            self.curriculum_config
        )
        self.manifest = self._load_or_create_manifest()

    def _semantic_config(self) -> dict[str, Any]:
        """Return every configured input that can change training, gates, or resume."""

        stages = []
        for stage in self.config.stages:
            stage_payload = {
                "id": stage.stage_id,
                "name": stage.name,
                "world": stage.world,
                "obstacle_count": stage.obstacle_count,
                "training_steps": stage.training_steps,
                "gate": {
                    "seed_start": stage.gate.seed_start,
                    "episodes": stage.gate.episodes,
                    "min_success_rate": stage.gate.min_success_rate,
                    "max_collision_rate": stage.gate.max_collision_rate,
                },
                "controller_env": dict(sorted(stage.controller_env.items())),
                "preferred_resume_model": _semantic_relative_path(
                    stage.preferred_resume_model, self.repository_root
                ),
            }
            # Preserve existing schema-v2 fingerprints for stages that predate
            # immutable resume-source binding.
            if stage.preferred_resume_sha256 is not None:
                stage_payload["preferred_resume_sha256"] = (
                    stage.preferred_resume_sha256
                )
            stages.append(stage_payload)
        return {
            "semantic_schema_version": 1,
            "seed": self.seed,
            "transfer_model": _semantic_relative_path(
                self.config.transfer_model, self.repository_root
            ),
            "policy_transfer_model": _semantic_relative_path(
                self.config.policy_transfer_model, self.repository_root
            ),
            "warm_start_requirement": _semantic_relative_path(
                self.config.warm_start_requirement, self.repository_root
            ),
            "warm_start_requirement_sha256": (
                self.warm_start_verification.requirement_sha256
                if self.warm_start_verification is not None
                else None
            ),
            "model_variant": self.config.model_variant,
            "port_base": self.config.port_base,
            "automatic_split": "validation",
            "evaluation_world": _EVALUATION_WORLD,
            "gate_overrides": {
                "seed_start": self.seed_start_override,
                "episodes": self.episodes_override,
                "min_success_rate": self.success_rate_override,
                "max_collision_rate": self.collision_rate_override,
            },
            "stages": stages,
        }

    def _load_or_create_manifest(self) -> dict[str, Any]:
        if self.manifest_path.is_file():
            raw = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
            manifest = dict(_mapping(raw, "run manifest"))
            schema_version = manifest.get("schema_version")
            if schema_version != _RUN_MANIFEST_SCHEMA_VERSION:
                raise ValueError(
                    "Existing run manifest is legacy or unsupported: schema_version "
                    f"{schema_version!r} lacks the required schema_version 2 semantic "
                    "configuration fingerprint; refusing unsafe resume."
                )
            stored_sha256 = manifest.get("curriculum_config_sha256")
            if (
                not isinstance(stored_sha256, str)
                or _SHA256_PATTERN.fullmatch(stored_sha256) is None
            ):
                raise ValueError(
                    "Existing schema_version 2 run manifest is missing a valid "
                    "curriculum_config_sha256; refusing unsafe resume."
                )
            stored_config = manifest.get("curriculum_config")
            if not isinstance(stored_config, dict):
                raise ValueError(
                    "Existing schema_version 2 run manifest is missing the semantic "
                    "curriculum_config snapshot; refusing unsafe resume."
                )
            if _semantic_config_sha256(stored_config) != stored_sha256.casefold():
                raise ValueError(
                    "Existing run manifest curriculum_config does not match its "
                    "curriculum_config_sha256; refusing unsafe resume."
                )
            if int(manifest.get("seed", -1)) != self.seed:
                raise ValueError("Existing manifest seed does not match --seed.")
            if stored_sha256.casefold() != self.curriculum_config_sha256:
                raise ValueError(
                    "Existing manifest curriculum configuration fingerprint does not "
                    "match the current configuration; refusing unsafe resume."
                )
            if manifest.get("model_variant") != self.config.model_variant:
                raise ValueError("Existing manifest model_variant does not match the config.")
            return manifest
        return {
            "schema_version": _RUN_MANIFEST_SCHEMA_VERSION,
            "seed": self.seed,
            "model_variant": self.config.model_variant,
            "curriculum_config_sha256": self.curriculum_config_sha256,
            "curriculum_config": self.curriculum_config,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "planned" if self.dry_run else "running",
            "webots_executable": str(self.webots_executable),
            "automatic_split": "validation",
            "test_split_was_run": False,
            "stages": [],
        }

    def _write_manifest(self) -> None:
        self.manifest["updated_at"] = utc_now()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path = self.manifest_path.with_suffix(".yaml")
        json_path = self.manifest_path.with_suffix(".json")
        yaml_path.write_text(
            yaml.safe_dump(self.manifest, sort_keys=False), encoding="utf-8"
        )
        json_path.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _record_formal_w2_artifact_provenance(self) -> None:
        """Create training-time sidecars for every immutable W2 model artifact."""

        if self.dry_run or [stage.stage_id for stage in self.config.stages] != [
            "W0",
            "W1",
            "W2",
        ]:
            return
        from indoor_patrol_rl.evaluation.formal_protocol import (
            file_sha256,
            write_or_validate_model_provenance,
        )

        matching = [
            record for record in self.manifest["stages"] if record.get("id") == "W2"
        ]
        if len(matching) != 1:
            raise ValueError("Completed formal curriculum must contain exactly one W2 record.")
        record = matching[0]
        training = _mapping(record.get("training"), "W2 training record")
        metadata_value = training.get("metadata")
        if not isinstance(metadata_value, str) or not metadata_value:
            raise ValueError("Completed W2 training must record its immutable metadata path.")
        metadata_path = (self.repository_root / metadata_value).resolve()
        stage_start = int(record["stage_start_steps"])
        final_steps = int(training["final_model_steps"])
        snapshots = discover_model_snapshots(
            self.repository_root, self.seed, self.config.model_variant
        )
        candidates = sorted(
            (
                snapshot
                for snapshot in snapshots
                if stage_start < snapshot.steps <= final_steps
                and (
                    _CHECKPOINT_STEP_PATTERN.search(snapshot.path.name) is not None
                    or _MODEL_STEP_PATTERN.search(snapshot.path.name) is not None
                )
            ),
            key=lambda item: (item.steps, item.path.as_posix()),
        )
        if not candidates:
            raise ValueError("Completed W2 training has no immutable model artifacts.")
        provenance_records: list[dict[str, str]] = []
        for snapshot in candidates:
            provenance_path, _payload = write_or_validate_model_provenance(
                repository_root=self.repository_root,
                training_seed=self.seed,
                model_path=snapshot.path,
                metadata_path=metadata_path,
                curriculum_manifest_path=self.manifest_path.with_suffix(".yaml"),
                allow_create=True,
            )
            provenance_records.append(
                {
                    "model_path": relative_artifact_path(
                        snapshot.path, self.repository_root
                    ).replace("\\", "/"),
                    "model_sha256": file_sha256(snapshot.path),
                    "provenance_path": relative_artifact_path(
                        provenance_path, self.repository_root
                    ).replace("\\", "/"),
                    "provenance_sha256": file_sha256(provenance_path),
                }
            )
        record["artifact_provenance"] = provenance_records
        self._write_manifest()

    def _stage_record(self, stage: WebotsStage) -> dict[str, Any]:
        records = self.manifest["stages"]
        for record in records:
            if record.get("id") == stage.stage_id:
                return record
        record = {
            "id": stage.stage_id,
            "name": stage.name,
            "world": stage.world,
            "obstacle_count": stage.obstacle_count,
            "training_steps": stage.training_steps,
            "controller_env": dict(stage.controller_env),
            "preferred_resume_model": (
                str(stage.preferred_resume_model)
                if stage.preferred_resume_model is not None
                else None
            ),
            "status": "pending",
            "training_attempts": [],
        }
        if stage.preferred_resume_sha256 is not None:
            record["preferred_resume_sha256"] = stage.preferred_resume_sha256
        records.append(record)
        return record

    def _gate_for(self, stage: WebotsStage) -> ValidationGate:
        return ValidationGate(
            seed_start=(
                self.seed_start_override
                if self.seed_start_override is not None
                else stage.gate.seed_start
            ),
            episodes=(
                self.episodes_override
                if self.episodes_override is not None
                else stage.gate.episodes
            ),
            min_success_rate=(
                self.success_rate_override
                if self.success_rate_override is not None
                else stage.gate.min_success_rate
            ),
            max_collision_rate=(
                self.collision_rate_override
                if self.collision_rate_override is not None
                else stage.gate.max_collision_rate
            ),
        )

    def _webots_command(self, port: int, world: str) -> list[str]:
        return [
            str(self.webots_executable),
            f"--port={port}",
            "--batch",
            "--mode=fast",
            "--no-rendering",
            "--stdout",
            "--stderr",
            str(self.repository_root / "worlds" / world),
        ]

    def _run_process(self, command: list[str], overrides: dict[str, str]) -> int:
        if self.dry_run:
            return 0
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith("WEBOTS_") and name != "WEBOTS_HOME":
                environment.pop(name)
        environment.update(overrides)
        completed = self.process_runner(
            command,
            cwd=self.repository_root,
            env=environment,
            check=False,
        )
        return int(completed.returncode)

    def _evaluation_run_name(self, stage: WebotsStage, model: ModelSnapshot) -> str:
        """Build the exact result identity consumed by checkpoint selection."""

        encoded_steps = model_steps_from_path(model.path)
        if encoded_steps is not None and encoded_steps != model.steps:
            raise ValueError(
                f"Validation model filename encodes {encoded_steps} steps, "
                f"but the snapshot records {model.steps}."
            )
        kind = (
            "checkpoint"
            if _CHECKPOINT_STEP_PATTERN.search(model.path.name) is not None
            else "final_step"
        )
        return validate_evaluation_run_name(
            f"webots_validation_{stage.stage_id.casefold()}_{kind}{model.steps}_"
            f"seed{self.seed}"
        )

    def _expected_result_directory(self, run_name: str) -> Path:
        return self.repository_root / "artifacts/results" / run_name

    def _preferred_resume_path(self, stage: WebotsStage) -> Path | None:
        if stage.preferred_resume_model is None:
            return None
        path = self.repository_root / stage.preferred_resume_model
        model_directory = _model_directory(
            self.repository_root, self.seed, self.config.model_variant
        ).absolute()
        try:
            relative = path.absolute().relative_to(model_directory)
        except ValueError as error:
            raise ValueError(
                f"Stage {stage.stage_id} preferred_resume_model must be inside "
                f"{model_directory}."
            ) from error
        if self.config.model_variant is None and len(relative.parts) > 1:
            if relative.parts[0] != "checkpoints":
                raise ValueError(
                    f"Stage {stage.stage_id} preferred_resume_model points into "
                    "a non-legacy model variant."
                )
        return path.absolute()

    def _verify_preferred_resume_source(
        self,
        stage: WebotsStage,
    ) -> tuple[Path | None, str | None]:
        """Verify one immutable stage root before any reuse or skip decision."""

        path = self._preferred_resume_path(stage)
        if path is None:
            return None, None
        if not path.is_file():
            raise FileNotFoundError(
                f"Stage {stage.stage_id} preferred resume model does not exist: {path}"
            )
        if model_steps_from_path(path) is None:
            raise ValueError(
                f"Stage {stage.stage_id} preferred resume model must encode its step "
                "in an immutable filename."
            )
        actual_sha256 = _sha256_file(path)
        if (
            stage.preferred_resume_sha256 is not None
            and actual_sha256 != stage.preferred_resume_sha256
        ):
            raise ValueError(
                f"Stage {stage.stage_id} preferred resume model SHA-256 does not "
                "match the pre-registered digest."
            )
        return path, actual_sha256

    def _preferred_resume_snapshot(
        self,
        stage: WebotsStage,
        snapshots: Sequence[ModelSnapshot],
    ) -> ModelSnapshot | None:
        path, _actual_sha256 = self._verify_preferred_resume_source(stage)
        if path is None:
            return None
        snapshot = next(
            (item for item in snapshots if item.path.absolute() == path), None
        )
        if snapshot is not None:
            return snapshot
        steps = model_steps_from_path(path)
        if steps is None:
            raise ValueError(
                f"Stage {stage.stage_id} preferred resume model must encode its step "
                "in an immutable filename."
            )
        return ModelSnapshot(
            path=path,
            steps=steps,
            model_variant=self.config.model_variant,
            modified_at_ns=path.stat().st_mtime_ns,
        )

    def _training_source_binding(
        self,
        snapshot: ModelSnapshot,
    ) -> tuple[Path, str]:
        """Read and independently verify a completed run's source-model binding."""

        if snapshot.metadata_path is None:
            raise ValueError("Completed training snapshot is missing run metadata.")
        try:
            raw = _mapping(
                yaml.safe_load(snapshot.metadata_path.read_text(encoding="utf-8")),
                "training metadata",
            )
        except (OSError, yaml.YAMLError) as error:
            raise ValueError("Completed training metadata cannot be read.") from error
        raw_source = raw.get("source_model")
        raw_sha256 = raw.get("source_model_sha256")
        if not isinstance(raw_source, str) or not raw_source:
            raise ValueError("Completed training metadata is missing source_model.")
        if not isinstance(raw_sha256, str) or _SHA256_PATTERN.fullmatch(raw_sha256) is None:
            raise ValueError(
                "Completed training metadata is missing a valid source_model_sha256."
            )
        source_path = _path_from_metadata(raw_source, self.repository_root).resolve()
        try:
            source_path.relative_to(self.repository_root.resolve())
        except ValueError as error:
            raise ValueError(
                "Completed training metadata source_model must be inside the repository."
            ) from error
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Completed training source model does not exist: {source_path}"
            )
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != raw_sha256.casefold():
            raise ValueError(
                "Completed training source_model_sha256 does not match its artifact."
            )
        return source_path, actual_sha256

    def _validate_attempt_completion(
        self,
        snapshot: ModelSnapshot,
        *,
        resume_path: Path,
        resume_sha256: str,
        additional_steps_requested: int,
    ) -> tuple[Path, str]:
        """Bind returned metadata to the exact model launched for this attempt."""

        if _sha256_file(resume_path) != resume_sha256:
            raise ValueError("The resume model changed while Webots training was running.")
        source_path, source_sha256 = self._training_source_binding(snapshot)
        if source_path != resume_path.resolve():
            raise ValueError(
                "Completed training metadata source_model does not match this attempt."
            )
        if source_sha256 != resume_sha256:
            raise ValueError(
                "Completed training metadata source_model_sha256 does not match this "
                "attempt."
            )
        starting_steps = model_steps_from_path(resume_path)
        if starting_steps is None or snapshot.starting_steps != starting_steps:
            raise ValueError(
                "Completed training metadata starting_webots_steps does not match "
                "the resume model."
            )
        if snapshot.additional_steps_requested != additional_steps_requested:
            raise ValueError(
                "Completed training metadata additional_steps_requested does not "
                "match this attempt."
            )
        encoded_final_steps = model_steps_from_path(snapshot.path)
        if encoded_final_steps != snapshot.steps:
            raise ValueError(
                "Completed training model filename does not match final_model_steps."
            )
        minimum_final_steps = starting_steps + additional_steps_requested
        if snapshot.steps < minimum_final_steps:
            raise ValueError(
                "Completed training final_model_steps does not reach the requested "
                "cumulative step target."
            )
        return source_path, source_sha256

    def _validate_reused_completion_source(
        self,
        stage: WebotsStage,
        record: Mapping[str, Any] | None,
        snapshot: ModelSnapshot,
    ) -> tuple[Path, str, dict[str, Any] | None]:
        """Recheck a SHA-bound completion before any restart path reuses it."""

        preferred_path, preferred_sha256 = self._verify_preferred_resume_source(stage)
        if preferred_path is None or preferred_sha256 is None:
            raise ValueError("A SHA-bound stage is missing its preferred resume source.")
        expected_resume_path = preferred_path
        expected_resume_sha256 = preferred_sha256
        expected_additional_steps = stage.training_steps
        selected_attempt: dict[str, Any] | None = None
        if record is not None:
            for previous_attempt in reversed(record.get("training_attempts", [])):
                raw_resume_model = previous_attempt.get("resume_model")
                raw_resume_sha256 = previous_attempt.get("resume_model_sha256")
                if (
                    previous_attempt.get("status") in {None, "training", "completed"}
                    and isinstance(raw_resume_model, str)
                    and isinstance(raw_resume_sha256, str)
                ):
                    expected_resume_path = (
                        self.repository_root / raw_resume_model
                    ).resolve()
                    expected_resume_sha256 = raw_resume_sha256
                    expected_additional_steps = int(
                        previous_attempt["additional_steps_requested"]
                    )
                    selected_attempt = previous_attempt
                    break
        source_path, source_sha256 = self._validate_attempt_completion(
            snapshot,
            resume_path=expected_resume_path,
            resume_sha256=expected_resume_sha256,
            additional_steps_requested=expected_additional_steps,
        )
        return source_path, source_sha256, selected_attempt

    def _configured_stage_start_steps(
        self,
        stage: WebotsStage,
        record: Mapping[str, Any] | None,
        fallback_steps: int,
    ) -> int:
        if record is not None and record.get("stage_start_steps") is not None:
            return int(record["stage_start_steps"])
        preferred_path = self._preferred_resume_path(stage)
        if preferred_path is not None:
            preferred_steps = model_steps_from_path(preferred_path)
            if preferred_steps is None:
                raise ValueError(
                    f"Stage {stage.stage_id} preferred resume model must encode its step."
                )
            return preferred_steps
        return fallback_steps

    def _stage_generated_snapshots(
        self,
        stage: WebotsStage,
        record: Mapping[str, Any],
        snapshots: Sequence[ModelSnapshot],
        stage_start_steps: int,
    ) -> tuple[ModelSnapshot, ...]:
        generated = {
            (snapshot.path, snapshot.steps): snapshot
            for snapshot in snapshots
            if snapshot.steps > stage_start_steps
            and snapshot_matches_stage(snapshot, stage)
        }
        attempt_starts = [
            timestamp
            for attempt in record.get("training_attempts", [])
            if attempt.get("status") != "planned"
            if (timestamp := _timestamp_ns(attempt.get("started_at"))) is not None
        ]
        if attempt_starts:
            first_attempt = min(attempt_starts)
            for snapshot in snapshots:
                if (
                    snapshot.steps > stage_start_steps
                    and snapshot.modified_at_ns is not None
                    and snapshot.modified_at_ns >= first_attempt
                ):
                    generated[(snapshot.path, snapshot.steps)] = snapshot
        return tuple(generated.values())

    def _run_training(
        self,
        stage: WebotsStage,
        record: dict[str, Any],
        *,
        baseline_steps: int,
        snapshots: Sequence[ModelSnapshot],
    ) -> ModelSnapshot | None:
        preferred_path = self._preferred_resume_path(stage)
        preferred_sha256 = None
        if stage.preferred_resume_sha256 is not None:
            preferred_path, preferred_sha256 = self._verify_preferred_resume_source(
                stage
            )
        stage_start_steps = self._configured_stage_start_steps(
            stage, record, baseline_steps
        )
        target_steps = stage_start_steps + stage.training_steps
        completed = select_stage_completed_snapshot(
            snapshots,
            stage=stage,
            target_steps=target_steps,
        )
        if completed is not None:
            source_binding: tuple[Path, str, dict[str, Any] | None] | None = None
            if stage.preferred_resume_sha256 is not None:
                source_binding = self._validate_reused_completion_source(
                    stage,
                    record,
                    completed,
                )
                selected_attempt = source_binding[2]
                if selected_attempt is not None and selected_attempt.get("status") in {
                    None,
                    "training",
                }:
                    selected_attempt["status"] = "completed"
                    selected_attempt["recovered_after_restart"] = True
                    selected_attempt.setdefault("finished_at", utc_now())
            record["training"] = {
                "status": "reused_completed_metadata",
                "baseline_steps": (
                    completed.starting_steps
                    if completed.starting_steps is not None
                    else stage_start_steps
                ),
                "target_model_steps": target_steps,
                "final_model_steps": completed.steps,
                "model": relative_artifact_path(completed.path, self.repository_root),
                "metadata": (
                    relative_artifact_path(completed.metadata_path, self.repository_root)
                    if completed.metadata_path
                    else None
                ),
            }
            if source_binding is not None:
                record["training"].update(
                    {
                        "source_model": relative_artifact_path(
                            source_binding[0], self.repository_root
                        ),
                        "source_model_sha256": source_binding[1],
                    }
                )
            record["stage_start_steps"] = record["training"]["baseline_steps"]
            return completed

        preferred = self._preferred_resume_snapshot(stage, snapshots)
        generated = self._stage_generated_snapshots(
            stage, record, snapshots, stage_start_steps
        )
        resumable = max(generated, key=lambda item: item.steps, default=None)
        if resumable is None:
            resumable = preferred
        if resumable is None and self.config.model_variant is None:
            resumable = max(
                (snapshot for snapshot in snapshots if snapshot.steps >= stage_start_steps),
                key=lambda item: item.steps,
                default=None,
            )
        if (
            resumable is None
            and self.config.model_variant is not None
            and stage_start_steps > 0
        ):
            raise ValueError(
                f"Stage {stage.stage_id} requires preferred_resume_model (or "
                f"--preferred-resume {stage.stage_id}=MODEL) selected by validation; "
                "the variant endpoint is never chosen automatically."
            )
        if self.dry_run and resumable is None and baseline_steps > 0:
            model_directory = _model_directory(
                self.repository_root, self.seed, self.config.model_variant
            )
            resumable = ModelSnapshot(
                model_directory / "ppo_webots_latest.zip",
                stage_start_steps,
                model_variant=self.config.model_variant,
            )
        if (
            resumable is not None
            and preferred is None
            and record.get("stage_start_steps") is None
            and resumable.starting_steps is not None
        ):
            stage_start_steps = resumable.starting_steps
            target_steps = stage_start_steps + stage.training_steps
        progressed_steps = (
            0 if resumable is None else max(0, resumable.steps - stage_start_steps)
        )
        remaining_steps = max(1, stage.training_steps - progressed_steps)
        port = choose_available_port(self.config.port_base, self.reserved_ports)
        command = self._webots_command(port, stage.world)
        overrides = dict(stage.controller_env)
        overrides.update(
            {
                "WEBOTS_TRAIN_SEED": str(self.seed),
                "WEBOTS_TRAIN_STEPS": str(remaining_steps),
                "WEBOTS_OBSTACLE_COUNT": str(stage.obstacle_count),
            }
        )
        if self.config.model_variant:
            overrides["WEBOTS_MODEL_VARIANT"] = self.config.model_variant
        if resumable is not None:
            overrides.pop("WEBOTS_POLICY_TRANSFER_MODEL", None)
            overrides.pop("WEBOTS_TRANSFER_MODEL", None)
            overrides["WEBOTS_RESUME_MODEL"] = relative_artifact_path(
                resumable.path, self.repository_root
            )
        else:
            overrides.pop("WEBOTS_RESUME_MODEL", None)
            if self.config.policy_transfer_model is not None:
                overrides["WEBOTS_POLICY_TRANSFER_MODEL"] = relative_artifact_path(
                    self.repository_root / self.config.policy_transfer_model,
                    self.repository_root,
                )
                overrides.pop("WEBOTS_TRANSFER_MODEL", None)
            elif "WEBOTS_POLICY_TRANSFER_MODEL" not in overrides:
                overrides["WEBOTS_TRANSFER_MODEL"] = relative_artifact_path(
                    self.repository_root / self.config.transfer_model,
                    self.repository_root,
                )
        resume_model_sha256 = (
            _sha256_file(resumable.path)
            if resumable is not None and resumable.path.is_file()
            else None
        )
        attempt = {
            "started_at": utc_now(),
            "command": command,
            "environment": overrides,
            "port": port,
            "baseline_steps": stage_start_steps,
            "progressed_steps_before_attempt": progressed_steps,
            "additional_steps_requested": remaining_steps,
            "preferred_resume_model": (
                relative_artifact_path(preferred_path, self.repository_root)
                if preferred_path is not None
                else None
            ),
            "preferred_resume_sha256": stage.preferred_resume_sha256,
            "preferred_resume_actual_sha256": preferred_sha256,
            "resume_model": (
                relative_artifact_path(resumable.path, self.repository_root)
                if resumable is not None
                else None
            ),
            "resume_model_sha256": resume_model_sha256,
        }
        record["stage_start_steps"] = stage_start_steps
        record["training_attempts"].append(attempt)
        record["status"] = "training"
        self._write_manifest()
        try:
            if stage.preferred_resume_sha256 is not None:
                self._verify_preferred_resume_source(stage)
            if resumable is not None and not self.dry_run:
                if (
                    resume_model_sha256 is None
                    or _sha256_file(resumable.path) != resume_model_sha256
                ):
                    raise ValueError(
                        "The resume model changed before Webots training launch."
                    )
        except (OSError, ValueError) as error:
            attempt["finished_at"] = utc_now()
            attempt["status"] = "resume_source_invalid"
            attempt["error"] = str(error)
            record["status"] = "training_failed"
            self.manifest["status"] = "failed"
            self._write_manifest()
            raise
        return_code = self._run_process(command, overrides)
        attempt["finished_at"] = utc_now()
        attempt["exit_code"] = return_code
        if self.dry_run:
            attempt["status"] = "planned"
            model_directory = _model_directory(
                self.repository_root, self.seed, self.config.model_variant
            )
            return ModelSnapshot(
                model_directory / "ppo_webots_latest.zip",
                target_steps,
                stage.obstacle_count,
                model_variant=self.config.model_variant,
            )
        if return_code != 0:
            attempt["status"] = "failed"
            record["status"] = "training_failed"
            self.manifest["status"] = "failed"
            self._write_manifest()
            return None
        refreshed = discover_model_snapshots(
            self.repository_root, self.seed, self.config.model_variant
        )
        completed = select_stage_completed_snapshot(
            refreshed,
            stage=stage,
            target_steps=target_steps,
        )
        if completed is None:
            attempt["status"] = "metadata_missing"
            record["status"] = "training_failed"
            self.manifest["status"] = "failed"
            self._write_manifest()
            return None
        source_binding: tuple[Path, str] | None = None
        if stage.preferred_resume_sha256 is not None:
            if resumable is None or resume_model_sha256 is None:
                raise ValueError("A SHA-bound stage must launch from a resume model.")
            try:
                source_binding = self._validate_attempt_completion(
                    completed,
                    resume_path=resumable.path,
                    resume_sha256=resume_model_sha256,
                    additional_steps_requested=remaining_steps,
                )
            except (OSError, ValueError) as error:
                attempt["status"] = "metadata_source_invalid"
                attempt["error"] = str(error)
                record["status"] = "training_failed"
                self.manifest["status"] = "failed"
                self._write_manifest()
                raise
        attempt["status"] = "completed"
        record["training"] = {
            "status": "completed",
            "baseline_steps": stage_start_steps,
            "target_model_steps": target_steps,
            "final_model_steps": completed.steps,
            "model": relative_artifact_path(completed.path, self.repository_root),
            "metadata": relative_artifact_path(completed.metadata_path, self.repository_root),
        }
        if source_binding is not None:
            record["training"].update(
                {
                    "source_model": relative_artifact_path(
                        source_binding[0], self.repository_root
                    ),
                    "source_model_sha256": source_binding[1],
                }
            )
        self._write_manifest()
        return completed

    def _run_validation(
        self,
        stage: WebotsStage,
        record: dict[str, Any],
        model: ModelSnapshot,
    ) -> bool:
        gate = self._gate_for(stage)
        port = choose_available_port(self.config.port_base, self.reserved_ports)
        command = self._webots_command(port, _EVALUATION_WORLD)
        run_name = self._evaluation_run_name(stage, model)
        planner_label = f"ppo_{stage.stage_id.casefold()}_step{model.steps}"
        model_sha256 = None if self.dry_run else _sha256_file(model.path)
        overrides = evaluation_controller_env(stage.controller_env)
        expected_safety_shield_enabled = _environment_flag_value(
            overrides, "WEBOTS_SAFETY_SHIELD"
        )
        expected_safety_shield_config = asdict(
            safety_shield_config_from_env(overrides)
        )
        overrides.update(
            {
                "WEBOTS_MODEL_SEED": str(self.seed),
                "WEBOTS_EVAL_MODEL": relative_artifact_path(
                    model.path, self.repository_root
                ),
                "WEBOTS_EVAL_EPISODES": str(gate.episodes),
                "WEBOTS_EVAL_SEED_START": str(gate.seed_start),
                "WEBOTS_EVAL_SPLIT": "validation",
                "WEBOTS_EVAL_CONDITION": stage.stage_id,
                "WEBOTS_EVAL_PLANNER_LABEL": planner_label,
                "WEBOTS_EVAL_RUN_NAME": run_name,
                "WEBOTS_OBSTACLE_COUNT": str(stage.obstacle_count),
            }
        )
        if self.config.model_variant:
            overrides["WEBOTS_MODEL_VARIANT"] = self.config.model_variant
        validation = {
            "started_at": utc_now(),
            "command": command,
            "environment": overrides,
            "port": port,
            "model": relative_artifact_path(model.path, self.repository_root),
            "model_sha256": model_sha256,
            "expected_safety_shield_enabled": expected_safety_shield_enabled,
            "expected_safety_shield_config": expected_safety_shield_config,
            "gate": {
                "episodes": gate.episodes,
                "min_success_rate": gate.min_success_rate,
                "max_collision_rate": gate.max_collision_rate,
            },
        }
        record["validation"] = validation
        record["status"] = "validating"
        self._write_manifest()
        summary_path = self._expected_result_directory(run_name) / "summary.json"
        previous_summary_mtime = (
            summary_path.stat().st_mtime_ns if summary_path.is_file() else None
        )
        return_code = self._run_process(command, overrides)
        validation["finished_at"] = utc_now()
        validation["exit_code"] = return_code
        if self.dry_run:
            validation["status"] = "planned"
            record["status"] = "planned"
            self._write_manifest()
            return True
        if return_code != 0:
            validation["status"] = "process_failed"
            record["status"] = "validation_failed"
            self.manifest["status"] = "failed"
            self._write_manifest()
            return False
        if (
            previous_summary_mtime is not None
            and summary_path.is_file()
            and summary_path.stat().st_mtime_ns == previous_summary_mtime
        ):
            validation["status"] = "summary_stale"
            validation["error"] = "Evaluation did not update the pre-existing summary.json."
            record["status"] = "validation_failed"
            self.manifest["status"] = "failed"
            self._write_manifest()
            return False
        try:
            metrics = load_validation_metrics(
                summary_path,
                repository_root=self.repository_root,
                expected_model_path=model.path,
                expected_model_sha256=model_sha256,
                expected_training_seed=self.seed,
                expected_model_variant=self.config.model_variant,
                expected_condition=stage.stage_id,
                expected_planner=planner_label,
                expected_seed_start=gate.seed_start,
                expected_episodes=gate.episodes,
                expected_safety_shield_enabled=expected_safety_shield_enabled,
                expected_safety_shield_config=expected_safety_shield_config,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            validation["status"] = "summary_invalid"
            validation["error"] = str(error)
            record["status"] = "validation_failed"
            self.manifest["status"] = "failed"
            self._write_manifest()
            return False
        passed = gate_passed(metrics, gate)
        validation.update(
            {
                "status": "passed" if passed else "gate_failed",
                "summary": relative_artifact_path(summary_path, self.repository_root),
                "metrics": metrics,
                "passed": passed,
            }
        )
        record["status"] = "completed" if passed else "gate_failed"
        if not passed:
            self.manifest["status"] = "gate_failed"
        self._write_manifest()
        return passed

    def _revalidate_completed_stage_evidence(
        self,
        stage: WebotsStage,
        record: Mapping[str, Any],
        model: ModelSnapshot,
    ) -> None:
        """Recheck saved validation provenance before trusting a completed skip."""

        validation = _mapping(record.get("validation"), "completed validation record")
        gate = self._gate_for(stage)
        run_name = self._evaluation_run_name(stage, model)
        summary_path = self._expected_result_directory(run_name) / "summary.json"
        model_sha256 = _sha256_file(model.path)
        expected_model = relative_artifact_path(model.path, self.repository_root)
        expected_summary = relative_artifact_path(summary_path, self.repository_root)
        expected_safety_shield_enabled = _environment_flag_value(
            stage.controller_env, "WEBOTS_SAFETY_SHIELD"
        )
        expected_safety_shield_config = asdict(
            safety_shield_config_from_env(stage.controller_env)
        )
        if validation.get("model") != expected_model:
            raise ValueError("Completed validation record model path does not match.")
        if validation.get("model_sha256") != model_sha256:
            raise ValueError("Completed validation record model SHA-256 does not match.")
        if validation.get("summary") != expected_summary:
            raise ValueError("Completed validation record summary path does not match.")
        if (
            validation.get("expected_safety_shield_enabled")
            is not expected_safety_shield_enabled
            or validation.get("expected_safety_shield_config")
            != expected_safety_shield_config
        ):
            raise ValueError("Completed validation record safety profile does not match.")
        planner_label = f"ppo_{stage.stage_id.casefold()}_step{model.steps}"
        metrics = load_validation_metrics(
            summary_path,
            repository_root=self.repository_root,
            expected_model_path=model.path,
            expected_model_sha256=model_sha256,
            expected_training_seed=self.seed,
            expected_model_variant=self.config.model_variant,
            expected_condition=stage.stage_id,
            expected_planner=planner_label,
            expected_seed_start=gate.seed_start,
            expected_episodes=gate.episodes,
            expected_safety_shield_enabled=expected_safety_shield_enabled,
            expected_safety_shield_config=expected_safety_shield_config,
        )
        if validation.get("metrics") != metrics or not gate_passed(metrics, gate):
            raise ValueError("Completed validation metrics no longer satisfy the gate.")

    def run(self, *, start_stage: int, stop_after_stage: int) -> int:
        """Execute the selected inclusive stage range, stopping on the first failure."""

        if not 0 <= start_stage <= stop_after_stage < len(self.config.stages):
            raise ValueError("Invalid Webots curriculum stage range.")
        # SHA-bound roots are prerequisites, even when their completed stages are
        # reused or skipped. Verify them before inspecting any reusable snapshot.
        for stage in self.config.stages[: stop_after_stage + 1]:
            if stage.preferred_resume_sha256 is not None:
                self._verify_preferred_resume_source(stage)
        snapshots = discover_model_snapshots(
            self.repository_root, self.seed, self.config.model_variant
        )
        baseline_steps = 0
        for index, stage in enumerate(self.config.stages):
            existing_record = next(
                (
                    record
                    for record in self.manifest["stages"]
                    if record.get("id") == stage.stage_id
                ),
                None,
            )
            stage_start_steps = self._configured_stage_start_steps(
                stage, existing_record, baseline_steps
            )
            target_steps = stage_start_steps + stage.training_steps
            completed = select_stage_completed_snapshot(
                snapshots,
                stage=stage,
                target_steps=target_steps,
            )
            if index < start_stage:
                if completed is None and not self.dry_run:
                    raise FileNotFoundError(
                        f"Cannot start at {self.config.stages[start_stage].stage_id}: "
                        f"prerequisite {stage.stage_id} has no completed metadata."
                    )
                if (
                    completed is not None
                    and stage.preferred_resume_sha256 is not None
                ):
                    self._validate_reused_completion_source(
                        stage,
                        existing_record,
                        completed,
                    )
                baseline_steps = target_steps if completed is None else completed.steps
                continue
            if index > stop_after_stage:
                break

            record = self._stage_record(stage)
            if record.get("status") == "completed" and record.get("validation", {}).get(
                "passed"
            ):
                if stage.preferred_resume_sha256 is not None:
                    if completed is None:
                        raise FileNotFoundError(
                            f"Completed stage {stage.stage_id} has no matching training "
                            "metadata."
                        )
                    self._validate_reused_completion_source(
                        stage,
                        record,
                        completed,
                    )
                    self._revalidate_completed_stage_evidence(
                        stage,
                        record,
                        completed,
                    )
                baseline_steps = int(record["training"]["final_model_steps"])
                continue
            model = self._run_training(
                stage,
                record,
                baseline_steps=baseline_steps,
                snapshots=snapshots,
            )
            if model is None:
                return 2
            if not self._run_validation(stage, record, model):
                return 3
            baseline_steps = model.steps
            snapshots = discover_model_snapshots(
                self.repository_root, self.seed, self.config.model_variant
            )

        self.manifest["status"] = "planned" if self.dry_run else "completed"
        self._write_manifest()
        self._record_formal_w2_artifact_provenance()
        return 0
