"""Strict aggregation for the held-out 400-episode Webots result matrix.

The formal matrix is deliberately fail-closed.  It accepts one result pair per
planner and condition, recomputes every aggregate from ``episodes.csv``, binds
PPO rows to the evaluated model digest, and only reports statistics after the
entire paired matrix has been proven complete.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from indoor_patrol_rl.evaluation.formal_protocol import FinalLock, load_final_lock
from indoor_patrol_rl.evaluation.metrics import EpisodeRecord
from indoor_patrol_rl.evaluation.webots_formal_config import (
    WebotsFormalConfig,
    load_webots_formal_config,
)

FORMAL_SPLIT = "test"
FORMAL_CONDITIONS = ("G0", "G1", "G2", "G3")
FORMAL_SCENARIO_SEEDS = tuple(range(10000, 10020))
FORMAL_PPO_TRAINING_SEEDS = (11, 22, 33)
FORMAL_SERIES = (
    "reactive",
    "dwa",
    "ppo_seed11",
    "ppo_seed22",
    "ppo_seed33",
)
FORMAL_EPISODE_COUNT = len(FORMAL_CONDITIONS) * len(FORMAL_SCENARIO_SEEDS) * len(FORMAL_SERIES)

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_PPO_SERIES_PATTERN = re.compile(r"ppo_seed(?P<seed>\d+)")
_SUMMARY_INTEGER_FIELDS = (
    "episodes",
    "success_count",
    "collision_count",
    "timeout_count",
    "error_count",
)
_SUMMARY_FLOAT_FIELDS = (
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "completion_time_s_mean",
    "path_length_ratio_mean",
    "minimum_clearance_m_mean",
    "near_collision_ratio_mean",
    "action_delta_sq_mean",
    "inference_latency_p95_ms",
)
_CROSS_SEED_METRICS = (
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "error_rate",
    "completion_time_s_mean",
    "path_length_ratio_mean",
    "minimum_clearance_m_mean",
    "near_collision_ratio_mean",
    "action_delta_sq_mean",
    "inference_latency_p95_ms",
)
_T_CRITICAL_975 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
}
_DERIVED_METRIC_REL_TOL = 1e-9
_DERIVED_METRIC_ABS_TOL = 1e-9


@dataclass(frozen=True)
class FormalResultInput:
    """One single-planner, single-condition formal result pair."""

    episodes_path: Path
    summary_path: Path


@dataclass(frozen=True)
class _LoadedEpisode:
    record: EpisodeRecord
    scenario_fingerprint: str | None
    source_path: Path


@dataclass(frozen=True)
class _LoadedInput:
    specification: FormalResultInput
    episodes: tuple[_LoadedEpisode, ...]
    summary: Mapping[str, Any]
    series: str
    condition_id: str
    episodes_sha256: str
    summary_sha256: str
    model_path: str | None
    model_sha256: str | None
    training_seed: int | None


@dataclass(frozen=True)
class _FormalContract:
    final_lock: FinalLock
    final_lock_path: Path
    config: WebotsFormalConfig
    condition_semantic_sha256: Mapping[str, str]


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _repository_file(repository_root: Path, value: str | Path, label: str) -> Path:
    root = repository_root.resolve()
    configured = Path(value)
    candidate = configured if configured.is_absolute() else root / configured
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository.") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _load_formal_contract(*, repository_root: Path, final_lock_path: Path) -> _FormalContract:
    root = repository_root.resolve()
    lock_path = _repository_file(root, final_lock_path, "final lock")
    final_lock = load_final_lock(
        lock_path,
        repository_root=root,
        verify_files=True,
    )
    config_path = _repository_file(root, final_lock.evaluation_config_path, "locked formal config")
    config = load_webots_formal_config(config_path)
    if config.source_sha256 != final_lock.evaluation_config_sha256:
        raise ValueError("Locked formal config digest differs from the final lock.")
    if tuple(config.scenario_seeds) != final_lock.split_spec.test_scenario_seeds:
        raise ValueError("Locked formal config test seeds differ from the final lock.")
    if tuple(config.ppo_training_seeds) != final_lock.split_spec.training_seeds:
        raise ValueError("Locked formal config training seeds differ from the final lock.")
    if config.world != final_lock.world_path:
        raise ValueError("Locked formal config world differs from the final lock.")
    world_path = _repository_file(root, final_lock.world_path, "locked formal world")
    world_sha256 = _sha256(world_path)
    if world_sha256 != final_lock.world_sha256:
        raise ValueError("Locked formal world digest differs from the final lock.")
    condition_hashes = {
        condition.condition_id: condition.semantic_sha256 for condition in config.conditions
    }
    return _FormalContract(
        final_lock=final_lock,
        final_lock_path=lock_path,
        config=config,
        condition_semantic_sha256=condition_hashes,
    )


def _strict_bool(value: str | None, field: str, row_number: int) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"episodes.csv row {row_number} field {field} must be True or False.")


def _strict_int(value: str | None, field: str, row_number: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError as error:
        raise ValueError(
            f"episodes.csv row {row_number} field {field} must be an integer."
        ) from error
    return parsed


def _strict_float(
    value: str | None,
    field: str,
    row_number: int,
    *,
    optional: bool = False,
) -> float | None:
    if optional and (value is None or value.strip() == ""):
        return None
    try:
        parsed = float(value or "")
    except ValueError as error:
        raise ValueError(f"episodes.csv row {row_number} field {field} must be numeric.") from error
    if not math.isfinite(parsed):
        raise ValueError(f"episodes.csv row {row_number} field {field} must be finite.")
    return parsed


def _scenario_fingerprint(row: Mapping[str, str], row_number: int) -> str | None:
    values = {
        str(row[name]).strip()
        for name in ("scenario_fingerprint", "scenario_sha256")
        if row.get(name) is not None and str(row[name]).strip()
    }
    if len(values) > 1:
        raise ValueError(
            f"episodes.csv row {row_number} has conflicting scenario fingerprint columns."
        )
    value = next(iter(values), None)
    if value is not None and "scenario_sha256" in row:
        sha_value = str(row.get("scenario_sha256") or "").strip()
        if sha_value and _SHA256_PATTERN.fullmatch(sha_value) is None:
            raise ValueError(
                f"episodes.csv row {row_number} scenario_sha256 is not a SHA-256 digest."
            )
    return value


def _parse_episode(row: Mapping[str, str], row_number: int, path: Path) -> _LoadedEpisode:
    required = {
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
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"episodes.csv is missing columns: {sorted(missing)}")

    if row["schema_version"] != "1.0":
        raise ValueError(f"episodes.csv row {row_number} has unsupported schema_version.")
    episode_id = str(row["episode_id"]).strip()
    if not episode_id:
        raise ValueError(f"episodes.csv row {row_number} has an empty episode_id.")
    split = str(row["split"])
    condition_id = str(row["condition_id"])
    planner = str(row["planner"])
    outcome = str(row["outcome"])
    if split != FORMAL_SPLIT:
        raise ValueError(f"Formal results require split={FORMAL_SPLIT!r}; found {split!r}.")
    if condition_id not in FORMAL_CONDITIONS:
        raise ValueError(f"Unexpected formal condition_id: {condition_id!r}.")
    if planner not in FORMAL_SERIES:
        raise ValueError(f"Unexpected formal planner label: {planner!r}.")
    if outcome not in {"success", "collision", "timeout", "error"}:
        raise ValueError(f"episodes.csv row {row_number} has invalid outcome {outcome!r}.")

    scenario_seed = _strict_int(row["scenario_seed"], "scenario_seed", row_number)
    if scenario_seed not in FORMAL_SCENARIO_SEEDS:
        raise ValueError(f"Unexpected formal scenario_seed: {scenario_seed}.")
    is_success = _strict_bool(row["is_success"], "is_success", row_number)
    collision = _strict_bool(row["collision"], "collision", row_number)
    timeout = _strict_bool(row["timeout"], "timeout", row_number)
    expected_flags = {
        "success": (True, False, False),
        "collision": (False, True, False),
        "timeout": (False, False, True),
        "error": (False, False, False),
    }[outcome]
    if (is_success, collision, timeout) != expected_flags:
        raise ValueError(f"episodes.csv row {row_number} outcome and boolean flags disagree.")

    step_count = _strict_int(row["step_count"], "step_count", row_number)
    elapsed_sim_s = _strict_float(row["elapsed_sim_s"], "elapsed_sim_s", row_number)
    completion_time_s = _strict_float(
        row["completion_time_s"], "completion_time_s", row_number, optional=True
    )
    episode_return = _strict_float(row["episode_return"], "episode_return", row_number)
    path_length_m = _strict_float(row["path_length_m"], "path_length_m", row_number)
    reference_path_length_m = _strict_float(
        row["reference_path_length_m"],
        "reference_path_length_m",
        row_number,
        optional=True,
    )
    path_length_ratio = _strict_float(
        row["path_length_ratio"], "path_length_ratio", row_number, optional=True
    )
    minimum_clearance_m = _strict_float(
        row["minimum_clearance_m"], "minimum_clearance_m", row_number
    )
    near_collision_ratio = _strict_float(
        row["near_collision_ratio"], "near_collision_ratio", row_number
    )
    action_delta_sq_mean = _strict_float(
        row["action_delta_sq_mean"], "action_delta_sq_mean", row_number
    )
    inference_latency_p50_ms = _strict_float(
        row["inference_latency_p50_ms"], "inference_latency_p50_ms", row_number
    )
    inference_latency_p95_ms = _strict_float(
        row["inference_latency_p95_ms"], "inference_latency_p95_ms", row_number
    )
    if step_count <= 0 or float(elapsed_sim_s) < 0.0 or float(path_length_m) < 0.0:
        raise ValueError(f"episodes.csv row {row_number} has invalid non-positive metrics.")
    if reference_path_length_m is None or reference_path_length_m <= 0.0:
        raise ValueError(
            f"episodes.csv row {row_number} must bind a positive reference_path_length_m."
        )
    if float(minimum_clearance_m) < 0.0:
        raise ValueError(f"episodes.csv row {row_number} has negative minimum_clearance_m.")
    if not 0.0 <= float(near_collision_ratio) <= 1.0:
        raise ValueError(f"episodes.csv row {row_number} has invalid near_collision_ratio.")
    if float(action_delta_sq_mean) < 0.0:
        raise ValueError(f"episodes.csv row {row_number} has negative action_delta_sq_mean.")
    if float(inference_latency_p50_ms) < 0.0 or float(inference_latency_p95_ms) < 0.0:
        raise ValueError(f"episodes.csv row {row_number} has negative inference latency.")
    if float(inference_latency_p95_ms) < float(inference_latency_p50_ms):
        raise ValueError(f"episodes.csv row {row_number} has P95 latency below P50.")
    if is_success:
        if completion_time_s is None or completion_time_s < 0.0:
            raise ValueError(f"episodes.csv row {row_number} success requires completion_time_s.")
        if path_length_ratio is None or path_length_ratio <= 0.0:
            raise ValueError(f"episodes.csv row {row_number} success requires path_length_ratio.")
        if not math.isclose(
            completion_time_s,
            float(elapsed_sim_s),
            rel_tol=_DERIVED_METRIC_REL_TOL,
            abs_tol=_DERIVED_METRIC_ABS_TOL,
        ):
            raise ValueError(
                f"episodes.csv row {row_number} completion_time_s does not equal elapsed_sim_s."
            )
        expected_path_length_ratio = float(path_length_m) / reference_path_length_m
        if not math.isclose(
            path_length_ratio,
            expected_path_length_ratio,
            rel_tol=_DERIVED_METRIC_REL_TOL,
            abs_tol=_DERIVED_METRIC_ABS_TOL,
        ):
            raise ValueError(
                f"episodes.csv row {row_number} path_length_ratio does not equal "
                "path_length_m/reference_path_length_m."
            )
    elif completion_time_s is not None or path_length_ratio is not None:
        raise ValueError(
            f"episodes.csv row {row_number} non-success metrics must leave "
            "success-only fields blank."
        )
    error_type = str(row.get("error_type") or "").strip() or None
    if outcome == "error" and error_type is None:
        raise ValueError(f"episodes.csv row {row_number} error requires error_type.")
    if outcome != "error" and error_type is not None:
        raise ValueError(f"episodes.csv row {row_number} unexpectedly sets error_type.")

    record = EpisodeRecord(
        schema_version="1.0",
        episode_id=episode_id,
        split=split,
        condition_id=condition_id,
        scenario_seed=scenario_seed,
        planner=planner,
        outcome=outcome,  # type: ignore[arg-type]
        is_success=is_success,
        collision=collision,
        timeout=timeout,
        step_count=step_count,
        elapsed_sim_s=float(elapsed_sim_s),
        completion_time_s=completion_time_s,
        episode_return=float(episode_return),
        path_length_m=float(path_length_m),
        reference_path_length_m=reference_path_length_m,
        path_length_ratio=path_length_ratio,
        minimum_clearance_m=float(minimum_clearance_m),
        near_collision_ratio=float(near_collision_ratio),
        action_delta_sq_mean=float(action_delta_sq_mean),
        inference_latency_p50_ms=float(inference_latency_p50_ms),
        inference_latency_p95_ms=float(inference_latency_p95_ms),
        error_type=error_type,
    )
    return _LoadedEpisode(
        record=record,
        scenario_fingerprint=_scenario_fingerprint(row, row_number),
        source_path=path,
    )


def _read_episodes(path: Path) -> tuple[_LoadedEpisode, ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        rows = [
            _parse_episode(row, row_number, path) for row_number, row in enumerate(reader, start=2)
        ]
    if not rows:
        raise ValueError(f"{path} contains no episode rows.")
    return tuple(rows)


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def _summarize(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty episode group.")
    successes = [record for record in records if record.is_success]
    count = len(records)
    return {
        "episodes": count,
        "success_count": len(successes),
        "collision_count": sum(record.collision for record in records),
        "timeout_count": sum(record.timeout for record in records),
        "error_count": sum(record.outcome == "error" for record in records),
        "success_rate": len(successes) / count,
        "collision_rate": sum(record.collision for record in records) / count,
        "timeout_rate": sum(record.timeout for record in records) / count,
        "error_rate": sum(record.outcome == "error" for record in records) / count,
        "completion_time_s_mean": _mean([record.completion_time_s for record in successes]),
        "path_length_ratio_mean": _mean([record.path_length_ratio for record in successes]),
        "minimum_clearance_m_mean": _mean([record.minimum_clearance_m for record in records]),
        "near_collision_ratio_mean": _mean([record.near_collision_ratio for record in records]),
        "action_delta_sq_mean": _mean([record.action_delta_sq_mean for record in records]),
        "inference_latency_p95_ms": float(
            np.percentile([record.inference_latency_p95_ms for record in records], 95)
        ),
    }


def _same_number(actual: Any, expected: float | None) -> bool:
    if expected is None:
        return actual is None
    if isinstance(actual, bool):
        return False
    try:
        number = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and math.isclose(number, expected, rel_tol=1e-9, abs_tol=1e-12)


def _validate_summary_group(
    raw: Any,
    expected: Mapping[str, Any],
    *,
    series: str,
    condition_id: str | None,
    label: str,
) -> None:
    group = _require_mapping(raw, label)
    if group.get("planner") != series:
        raise ValueError(f"{label}.planner does not match episodes.csv.")
    if condition_id is not None and group.get("condition_id") != condition_id:
        raise ValueError(f"{label}.condition_id does not match episodes.csv.")
    for field in _SUMMARY_INTEGER_FIELDS:
        if field not in group or isinstance(group[field], bool) or group[field] != expected[field]:
            raise ValueError(f"{label}.{field} does not match counts recomputed from CSV.")
    for field in _SUMMARY_FLOAT_FIELDS:
        if field not in group or not _same_number(group[field], expected[field]):
            raise ValueError(f"{label}.{field} does not match metrics recomputed from CSV.")


def _declared_episode_hashes(summary: Mapping[str, Any]) -> tuple[str, ...]:
    candidates: list[Any] = []
    for name in ("episodes_sha256", "episodes_csv_sha256"):
        if name in summary:
            candidates.append(summary[name])
    provenance = summary.get("evaluation_provenance")
    if isinstance(provenance, Mapping):
        for name in ("episodes_sha256", "episodes_csv_sha256"):
            if name in provenance:
                candidates.append(provenance[name])
    artifacts = summary.get("artifacts")
    if isinstance(artifacts, Mapping):
        for name in ("episodes_sha256", "episodes_csv_sha256"):
            if name in artifacts:
                candidates.append(artifacts[name])
        episodes = artifacts.get("episodes")
        if isinstance(episodes, Mapping) and "sha256" in episodes:
            candidates.append(episodes["sha256"])
    normalized: list[str] = []
    for value in candidates:
        digest = str(value).casefold()
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("Declared episodes.csv SHA-256 is malformed.")
        normalized.append(digest)
    return tuple(normalized)


def _validate_provenance_contract(
    provenance: Mapping[str, Any],
    *,
    series: str,
    condition_id: str,
    contract: _FormalContract,
) -> int | None:
    required = {
        "schema_version",
        "split",
        "condition_id",
        "scenario_seed_start",
        "scenario_seed_count",
        "scenario_seeds",
    }
    missing = required - provenance.keys()
    if missing:
        raise ValueError(f"evaluation_provenance is missing fields: {sorted(missing)}")
    if provenance["schema_version"] != 1:
        raise ValueError("evaluation_provenance.schema_version must be 1.")
    if provenance["split"] != FORMAL_SPLIT:
        raise ValueError("evaluation_provenance.split does not describe the test split.")
    if provenance["condition_id"] != condition_id:
        raise ValueError("evaluation_provenance.condition_id does not match episodes.csv.")
    if provenance["scenario_seed_start"] != FORMAL_SCENARIO_SEEDS[0]:
        raise ValueError("evaluation_provenance.scenario_seed_start is not formal.")
    if provenance["scenario_seed_count"] != len(FORMAL_SCENARIO_SEEDS):
        raise ValueError("evaluation_provenance.scenario_seed_count is not formal.")
    scenario_seeds = _require_sequence(
        provenance["scenario_seeds"], "evaluation_provenance.scenario_seeds"
    )
    if tuple(scenario_seeds) != FORMAL_SCENARIO_SEEDS:
        raise ValueError("evaluation_provenance.scenario_seeds are not the formal test seeds.")

    final_lock = contract.final_lock
    expected_values: dict[str, Any] = {
        "formal_campaign_id": final_lock.campaign_id,
        "final_lock_sha256": final_lock.lock_sha256,
        "test_consumption_key": final_lock.test_consumption_key,
        "source_revision": final_lock.source_revision,
        "source_bundle_sha256": final_lock.source_bundle_sha256,
        "experiment_id": contract.config.experiment_id,
        "webots_version": contract.config.webots_version,
        "formal_config_path": final_lock.evaluation_config_path,
        "formal_config_sha256": final_lock.evaluation_config_sha256,
        "evaluation_config_sha256": final_lock.evaluation_config_sha256,
        "world_path": final_lock.world_path,
        "world_sha256": final_lock.world_sha256,
        "evaluator_sha256": final_lock.evaluator_sha256,
        "controller_sha256": final_lock.controller_sha256,
        "condition_semantic_sha256": contract.condition_semantic_sha256[condition_id],
        "planner": _method(series),
        "planner_label": series,
    }
    for name, expected in expected_values.items():
        if name not in provenance:
            raise ValueError(f"evaluation_provenance is missing {name}.")
        if provenance[name] != expected:
            raise ValueError(f"evaluation_provenance.{name} differs from the verified final lock.")

    match = _PPO_SERIES_PATTERN.fullmatch(series)
    training_seed = int(match.group("seed")) if match is not None else None
    if training_seed is None:
        if provenance.get("training_seed") is not None:
            raise ValueError("Baseline provenance must not claim a PPO training_seed.")
        return None
    if provenance.get("training_seed") != training_seed:
        raise ValueError("PPO planner label and evaluation_provenance.training_seed disagree.")
    return training_seed


def _validated_model_binding(
    provenance: Mapping[str, Any],
    *,
    repository_root: Path,
    training_seed: int | None,
    final_lock: FinalLock,
) -> tuple[str | None, str | None]:
    if training_seed is None:
        if provenance.get("model_path") is not None or provenance.get("model_sha256") is not None:
            raise ValueError("Baseline provenance must not bind a PPO model.")
        return None, None
    raw_path = provenance.get("model_path")
    raw_digest = provenance.get("model_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("PPO evaluation_provenance.model_path must be a non-empty string.")
    if not isinstance(raw_digest, str) or _SHA256_PATTERN.fullmatch(raw_digest) is None:
        raise ValueError("PPO evaluation_provenance.model_sha256 must be a SHA-256 digest.")
    manifests = {manifest.training_seed: manifest for manifest in final_lock.model_manifests}
    expected = manifests.get(training_seed)
    if expected is None:
        raise ValueError(f"Final lock has no PPO model for training seed {training_seed}.")
    if raw_path != expected.model_path or raw_digest != expected.model_sha256:
        raise ValueError(
            f"PPO model binding for training seed {training_seed} differs from the final lock."
        )
    configured = Path(raw_path)
    if configured.is_absolute():
        raise ValueError("Formal PPO model_path must be repository-relative.")
    root = repository_root.resolve()
    model_path = (root / configured).resolve()
    try:
        model_path.relative_to((root / "artifacts/models/webots").resolve())
    except ValueError as error:
        raise ValueError(
            "Formal PPO model_path must remain inside artifacts/models/webots."
        ) from error
    if not model_path.is_file():
        raise FileNotFoundError(f"Formal PPO model does not exist: {model_path}")
    actual = _sha256(model_path)
    if actual != raw_digest.casefold():
        raise ValueError(
            f"PPO model SHA-256 mismatch for training seed {training_seed}: "
            f"{actual} != {raw_digest.casefold()}"
        )
    return configured.as_posix(), actual


def _load_input(
    specification: FormalResultInput,
    repository_root: Path,
    contract: _FormalContract,
) -> _LoadedInput:
    episodes_path = specification.episodes_path.resolve()
    summary_path = specification.summary_path.resolve()
    if not episodes_path.is_file():
        raise FileNotFoundError(f"episodes.csv does not exist: {episodes_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json does not exist: {summary_path}")
    episodes_sha256 = _sha256(episodes_path)
    summary_sha256 = _sha256(summary_path)
    episodes = _read_episodes(episodes_path)
    with summary_path.open(encoding="utf-8") as stream:
        summary = _require_mapping(json.load(stream), str(summary_path))

    series_values = {episode.record.planner for episode in episodes}
    condition_values = {episode.record.condition_id for episode in episodes}
    seed_values = {episode.record.scenario_seed for episode in episodes}
    if len(series_values) != 1 or len(condition_values) != 1:
        raise ValueError(
            f"Each formal input must contain one planner and one condition: {episodes_path}"
        )
    series = next(iter(series_values))
    condition_id = next(iter(condition_values))
    if seed_values != set(FORMAL_SCENARIO_SEEDS) or len(episodes) != len(FORMAL_SCENARIO_SEEDS):
        raise ValueError(
            f"{episodes_path} must contain exactly one row for each formal scenario seed."
        )

    if summary.get("schema_version") != "1.0":
        raise ValueError("summary.schema_version must be '1.0'.")
    if summary.get("split") != FORMAL_SPLIT:
        raise ValueError("summary.split must be 'test'.")
    if summary.get("episode_count") != len(episodes):
        raise ValueError("summary.episode_count does not match rows recomputed from CSV.")
    if summary.get("planners") != [series]:
        raise ValueError("summary.planners must contain exactly the CSV planner.")
    if summary.get("conditions") != [condition_id]:
        raise ValueError("summary.conditions must contain exactly the CSV condition.")
    expected = _summarize([episode.record for episode in episodes])
    by_planner = _require_sequence(summary.get("by_planner"), "summary.by_planner")
    by_condition = _require_sequence(
        summary.get("by_planner_condition"), "summary.by_planner_condition"
    )
    if len(by_planner) != 1 or len(by_condition) != 1:
        raise ValueError("Formal input summary must contain exactly one aggregate group.")
    _validate_summary_group(
        by_planner[0], expected, series=series, condition_id=None, label="summary.by_planner[0]"
    )
    _validate_summary_group(
        by_condition[0],
        expected,
        series=series,
        condition_id=condition_id,
        label="summary.by_planner_condition[0]",
    )
    declared_episode_hashes = _declared_episode_hashes(summary)
    if not declared_episode_hashes:
        raise ValueError("summary must declare the episodes.csv SHA-256 digest.")
    for declared in declared_episode_hashes:
        if declared != episodes_sha256:
            raise ValueError(
                f"episodes.csv SHA-256 mismatch for {episodes_path}: "
                f"{episodes_sha256} != {declared}"
            )

    provenance = _require_mapping(
        summary.get("evaluation_provenance"), "summary.evaluation_provenance"
    )
    training_seed = _validate_provenance_contract(
        provenance,
        series=series,
        condition_id=condition_id,
        contract=contract,
    )
    model_path, model_sha256 = _validated_model_binding(
        provenance,
        repository_root=repository_root,
        training_seed=training_seed,
        final_lock=contract.final_lock,
    )
    return _LoadedInput(
        specification=FormalResultInput(episodes_path, summary_path),
        episodes=episodes,
        summary=summary,
        series=series,
        condition_id=condition_id,
        episodes_sha256=episodes_sha256,
        summary_sha256=summary_sha256,
        model_path=model_path,
        model_sha256=model_sha256,
        training_seed=training_seed,
    )


def _method(series: str) -> str:
    return "ppo" if series.startswith("ppo_seed") else series


def _training_seed(series: str) -> int | None:
    match = _PPO_SERIES_PATTERN.fullmatch(series)
    return int(match.group("seed")) if match is not None else None


def _series_group(
    records: Sequence[EpisodeRecord], *, series: str, condition_id: str | None = None
) -> dict[str, Any]:
    return {
        "series": series,
        "method": _method(series),
        "training_seed": _training_seed(series),
        **({"condition_id": condition_id} if condition_id is not None else {}),
        **_summarize(records),
    }


def _method_group(
    records: Sequence[EpisodeRecord], *, method: str, condition_id: str | None = None
) -> dict[str, Any]:
    return {
        "method": method,
        **({"condition_id": condition_id} if condition_id is not None else {}),
        **_summarize(records),
    }


def _cross_seed_stat(values_by_seed: Mapping[int, float | None]) -> dict[str, Any]:
    finite = {
        seed: float(value)
        for seed, value in values_by_seed.items()
        if value is not None and math.isfinite(value)
    }
    values = list(finite.values())
    if not values:
        return {
            "n": 0,
            "values_by_training_seed": {},
            "mean": None,
            "sample_sd": None,
            "ci95": None,
        }
    mean = statistics.fmean(values)
    if len(values) < 2:
        sample_sd = None
        ci95 = None
    else:
        sample_sd = statistics.stdev(values)
        critical = _T_CRITICAL_975[len(values) - 1]
        half_width = critical * sample_sd / math.sqrt(len(values))
        ci95 = {
            "lower": mean - half_width,
            "upper": mean + half_width,
            "half_width": half_width,
            "method": f"two-sided Student t, df={len(values) - 1}",
        }
    return {
        "n": len(values),
        "values_by_training_seed": {str(seed): finite[seed] for seed in sorted(finite)},
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95": ci95,
    }


def _ppo_cross_seed_rows(
    by_series_condition: Sequence[Mapping[str, Any]],
    by_series: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, Sequence[Mapping[str, Any]]]] = [
        (
            condition,
            [
                group
                for group in by_series_condition
                if group["condition_id"] == condition and group["method"] == "ppo"
            ],
        )
        for condition in FORMAL_CONDITIONS
    ]
    scopes.append(("ALL", [group for group in by_series if group["method"] == "ppo"]))
    for condition_id, groups in scopes:
        if {group["training_seed"] for group in groups} != set(FORMAL_PPO_TRAINING_SEEDS):
            raise ValueError(f"PPO cross-seed group {condition_id} is incomplete.")
        groups_by_seed = {int(group["training_seed"]): group for group in groups}
        rows.append(
            {
                "condition_id": condition_id,
                "training_seeds": list(FORMAL_PPO_TRAINING_SEEDS),
                "statistics": {
                    metric: _cross_seed_stat(
                        {seed: groups_by_seed[seed][metric] for seed in FORMAL_PPO_TRAINING_SEEDS}
                    )
                    for metric in _CROSS_SEED_METRICS
                },
            }
        )
    return rows


def _validate_complete_matrix(loaded: Sequence[_LoadedInput]) -> list[_LoadedEpisode]:
    expected_inputs = {
        (series, condition_id) for condition_id in FORMAL_CONDITIONS for series in FORMAL_SERIES
    }
    actual_inputs = [(item.series, item.condition_id) for item in loaded]
    duplicate_inputs = sorted(key for key in set(actual_inputs) if actual_inputs.count(key) > 1)
    if duplicate_inputs:
        raise ValueError(f"Duplicate formal result inputs: {duplicate_inputs}")
    missing_inputs = sorted(expected_inputs - set(actual_inputs))
    extra_inputs = sorted(set(actual_inputs) - expected_inputs)
    if missing_inputs or extra_inputs:
        raise ValueError(
            f"Formal result input matrix mismatch; missing={missing_inputs}, extra={extra_inputs}."
        )

    episodes = [episode for item in loaded for episode in item.episodes]
    if len(episodes) != FORMAL_EPISODE_COUNT:
        raise ValueError(
            f"Formal matrix must contain {FORMAL_EPISODE_COUNT} rows; found {len(episodes)}."
        )
    episode_ids: set[str] = set()
    keys: dict[tuple[str, int, str], _LoadedEpisode] = {}
    for episode in episodes:
        record = episode.record
        if record.episode_id in episode_ids:
            raise ValueError(f"Duplicate formal episode_id: {record.episode_id}")
        episode_ids.add(record.episode_id)
        key = (record.condition_id, record.scenario_seed, record.planner)
        if key in keys:
            raise ValueError(f"Duplicate formal episode key: {key}")
        keys[key] = episode

    expected_keys = {
        (condition_id, scenario_seed, series)
        for condition_id in FORMAL_CONDITIONS
        for scenario_seed in FORMAL_SCENARIO_SEEDS
        for series in FORMAL_SERIES
    }
    missing_keys = sorted(expected_keys - keys.keys())
    extra_keys = sorted(keys.keys() - expected_keys)
    if missing_keys or extra_keys:
        raise ValueError(
            f"Formal episode matrix mismatch; missing={missing_keys[:5]}, extra={extra_keys[:5]}."
        )

    for condition_id in FORMAL_CONDITIONS:
        for scenario_seed in FORMAL_SCENARIO_SEEDS:
            paired = [keys[(condition_id, scenario_seed, series)] for series in FORMAL_SERIES]
            reference_lengths = [
                float(episode.record.reference_path_length_m) for episode in paired
            ]
            if not all(
                math.isclose(value, reference_lengths[0], rel_tol=1e-9, abs_tol=1e-9)
                for value in reference_lengths[1:]
            ):
                raise ValueError(
                    "Paired planners do not share the same reference path for "
                    f"{condition_id} seed {scenario_seed}."
                )
            fingerprints = {episode.scenario_fingerprint for episode in paired}
            if fingerprints != {None} and len(fingerprints) != 1:
                raise ValueError(
                    f"Paired planners have mismatched scenario fingerprints for "
                    f"{condition_id} seed {scenario_seed}."
                )
    return episodes


def _validate_ppo_model_bindings(loaded: Sequence[_LoadedInput]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for seed in FORMAL_PPO_TRAINING_SEEDS:
        matching = [item for item in loaded if item.training_seed == seed]
        paths = {item.model_path for item in matching}
        digests = {item.model_sha256 for item in matching}
        if len(matching) != len(FORMAL_CONDITIONS) or len(paths) != 1 or len(digests) != 1:
            raise ValueError(
                f"All PPO condition runs for training seed {seed} must bind one exact model."
            )
        models.append(
            {
                "training_seed": seed,
                "model_path": next(iter(paths)),
                "model_sha256": next(iter(digests)),
            }
        )
    paths = [model["model_path"] for model in models]
    digests = [model["model_sha256"] for model in models]
    if len(set(paths)) != len(paths) or len(set(digests)) != len(digests):
        raise ValueError("Independent PPO training seeds must bind distinct model artifacts.")
    return models


def build_formal_report(
    inputs: Sequence[FormalResultInput],
    *,
    repository_root: Path,
    final_lock_path: Path,
) -> dict[str, Any]:
    """Validate one locked formal result matrix and return its aggregate report."""

    if not inputs:
        raise ValueError("At least one formal result input is required.")
    root = repository_root.resolve()
    contract = _load_formal_contract(
        repository_root=root,
        final_lock_path=final_lock_path,
    )
    loaded = [_load_input(specification, root, contract) for specification in inputs]
    episodes = _validate_complete_matrix(loaded)
    ppo_models = _validate_ppo_model_bindings(loaded)
    records = [episode.record for episode in episodes]

    by_series_condition = [
        _series_group(
            [
                record
                for record in records
                if record.planner == series and record.condition_id == condition_id
            ],
            series=series,
            condition_id=condition_id,
        )
        for condition_id in FORMAL_CONDITIONS
        for series in FORMAL_SERIES
    ]
    by_series = [
        _series_group([record for record in records if record.planner == series], series=series)
        for series in FORMAL_SERIES
    ]
    methods = ("reactive", "dwa", "ppo")
    by_method_condition = [
        _method_group(
            [
                record
                for record in records
                if _method(record.planner) == method and record.condition_id == condition_id
            ],
            method=method,
            condition_id=condition_id,
        )
        for condition_id in FORMAL_CONDITIONS
        for method in methods
    ]
    by_method = [
        _method_group(
            [record for record in records if _method(record.planner) == method],
            method=method,
        )
        for method in methods
    ]
    source_artifacts = [
        {
            "series": item.series,
            "condition_id": item.condition_id,
            "episodes_path": _portable_path(item.specification.episodes_path, repository_root),
            "episodes_sha256": item.episodes_sha256,
            "summary_path": _portable_path(item.specification.summary_path, repository_root),
            "summary_sha256": item.summary_sha256,
            "episode_count": len(item.episodes),
        }
        for item in sorted(
            loaded,
            key=lambda item: (
                FORMAL_CONDITIONS.index(item.condition_id),
                FORMAL_SERIES.index(item.series),
            ),
        )
    ]
    return {
        "schema_version": "1.0",
        "report_type": "webots_formal_test_matrix",
        "formal_provenance": {
            "campaign_id": contract.final_lock.campaign_id,
            "final_lock_path": _portable_path(contract.final_lock_path, root),
            "final_lock_sha256": contract.final_lock.lock_sha256,
            "test_consumption_key": contract.final_lock.test_consumption_key,
            "source_revision": contract.final_lock.source_revision,
            "source_bundle_sha256": contract.final_lock.source_bundle_sha256,
            "evaluation_config_path": contract.final_lock.evaluation_config_path,
            "evaluation_config_sha256": contract.final_lock.evaluation_config_sha256,
            "world_path": contract.final_lock.world_path,
            "world_sha256": contract.final_lock.world_sha256,
            "evaluator_path": contract.final_lock.evaluator_path,
            "evaluator_sha256": contract.final_lock.evaluator_sha256,
            "controller_path": contract.final_lock.controller_path,
            "controller_sha256": contract.final_lock.controller_sha256,
            "condition_semantic_sha256": dict(contract.condition_semantic_sha256),
        },
        "contract": {
            "split": FORMAL_SPLIT,
            "conditions": list(FORMAL_CONDITIONS),
            "scenario_seeds": list(FORMAL_SCENARIO_SEEDS),
            "baseline_series": ["reactive", "dwa"],
            "ppo_training_seeds": list(FORMAL_PPO_TRAINING_SEEDS),
            "series": list(FORMAL_SERIES),
            "expected_episode_count": FORMAL_EPISODE_COUNT,
            "pairing_key": ["condition_id", "scenario_seed"],
        },
        "matrix_validation": {
            "status": "passed",
            "input_pair_count": len(loaded),
            "episode_count": len(records),
            "unique_episode_ids": len({record.episode_id for record in records}),
            "unique_episode_keys": len(
                {(record.condition_id, record.scenario_seed, record.planner) for record in records}
            ),
            "paired_scenario_count": len(FORMAL_CONDITIONS) * len(FORMAL_SCENARIO_SEEDS),
            "series_per_scenario": len(FORMAL_SERIES),
        },
        "input_artifacts": source_artifacts,
        "ppo_models": ppo_models,
        "by_series_condition": by_series_condition,
        "by_series": by_series,
        "by_method_condition": by_method_condition,
        "by_method": by_method,
        "ppo_cross_training_seed": _ppo_cross_seed_rows(by_series_condition, by_series),
        "statistical_notes": {
            "ppo_unit_of_replication": "independent training seed",
            "sample_sd_ddof": 1,
            "ci95_method": "two-sided Student t interval across training-seed aggregates",
            "success_only_metrics": [
                "completion_time_s_mean",
                "path_length_ratio_mean",
            ],
        },
    }


def write_formal_report(report: Mapping[str, Any], output_path: Path) -> Path:
    """Write a deterministic JSON report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path
