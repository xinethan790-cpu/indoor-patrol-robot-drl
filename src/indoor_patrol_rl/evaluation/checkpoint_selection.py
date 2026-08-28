"""Auditable validation-only selection of Webots PPO checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_RESULT_DIRECTORY_PATTERN = re.compile(
    r"^webots_validation_(?P<condition>[a-z0-9]+)_"
    r"(?:(?P<checkpoint>checkpoint)(?P<checkpoint_step>\d+)"
    r"|(?P<final>final)_step(?P<final_step>\d+))_seed(?P<seed>\d+)$",
    re.IGNORECASE,
)
_FINAL_MODEL_PATTERN = re.compile(r"_step0*(?P<steps>\d+)\.zip$")


@dataclass(frozen=True)
class CheckpointSelectionGate:
    """Validation contract and the minimum acceptable safety/performance gate."""

    seed: int
    condition: str
    episodes: int
    min_success_rate: float
    max_collision_rate: float
    split: str = "validation"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("Selection seed must be non-negative.")
        if not self.condition:
            raise ValueError("Selection condition must not be empty.")
        if self.episodes <= 0:
            raise ValueError("Expected episode count must be positive.")
        if self.split != "validation":
            raise ValueError("Checkpoint selection is restricted to the validation split.")
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between zero and one.")
        if not 0.0 <= self.max_collision_rate <= 1.0:
            raise ValueError("max_collision_rate must be between zero and one.")


@dataclass(frozen=True)
class ResultIdentity:
    """Identity encoded by one Webots validation result directory."""

    condition: str
    kind: str
    step: int
    seed: int

    @property
    def expected_planner(self) -> str:
        return f"ppo_{self.condition.casefold()}_step{self.step}"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return number


def _portable_path(path: Path, repository_root: Path) -> str:
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(repository_root.absolute()))
    except ValueError:
        return str(absolute)


def sha256_file(path: Path) -> str:
    """Hash an artifact without loading the entire model into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_result_identity(summary_path: Path) -> ResultIdentity:
    """Parse checkpoint/final identity from the summary's parent directory."""

    if summary_path.name != "summary.json":
        raise ValueError(f"Expected a summary.json path, got: {summary_path}")
    match = _RESULT_DIRECTORY_PATTERN.fullmatch(summary_path.parent.name)
    if match is None:
        raise ValueError(
            "Unsupported Webots validation result directory name: "
            f"{summary_path.parent.name}"
        )
    kind = "checkpoint" if match.group("checkpoint") else "final"
    step_value = match.group("checkpoint_step") or match.group("final_step")
    return ResultIdentity(
        condition=match.group("condition").upper(),
        kind=kind,
        step=int(step_value),
        seed=int(match.group("seed")),
    )


def _validate_rate(count: int, episodes: int, rate: float, label: str) -> None:
    expected = count / episodes
    if not math.isclose(rate, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} does not match its count: {rate} != {expected}")


def _extract_group_metrics(
    group_value: Any,
    *,
    label: str,
    identity: ResultIdentity,
    gate: CheckpointSelectionGate,
    require_condition: bool,
) -> dict[str, float | int | str]:
    group = _mapping(group_value, label)
    planner = str(group.get("planner", ""))
    if planner != identity.expected_planner:
        raise ValueError(
            f"{label}.planner must be {identity.expected_planner!r}, got {planner!r}."
        )
    if require_condition:
        condition = str(group.get("condition_id", ""))
        if condition.casefold() != gate.condition.casefold():
            raise ValueError(
                f"{label}.condition_id must be {gate.condition!r}, got {condition!r}."
            )
    episodes = _integer(group.get("episodes"), f"{label}.episodes")
    if episodes != gate.episodes:
        raise ValueError(
            f"{label}.episodes must equal {gate.episodes}, got {episodes}."
        )
    counts = {
        name: _integer(group.get(name), f"{label}.{name}")
        for name in ("success_count", "collision_count", "timeout_count", "error_count")
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError(f"{label} outcome counts must be non-negative.")
    if sum(counts.values()) != episodes:
        raise ValueError(f"{label} outcome counts do not sum to episodes.")
    success_rate = _finite_number(group.get("success_rate"), f"{label}.success_rate")
    collision_rate = _finite_number(group.get("collision_rate"), f"{label}.collision_rate")
    if not 0.0 <= success_rate <= 1.0 or not 0.0 <= collision_rate <= 1.0:
        raise ValueError(f"{label} success/collision rates must be between zero and one.")
    _validate_rate(counts["success_count"], episodes, success_rate, f"{label}.success_rate")
    _validate_rate(
        counts["collision_count"], episodes, collision_rate, f"{label}.collision_rate"
    )
    path_length_ratio = _finite_number(
        group.get("path_length_ratio_mean"),
        f"{label}.path_length_ratio_mean",
        minimum=0.0,
    )
    completion_time = _finite_number(
        group.get("completion_time_s_mean"),
        f"{label}.completion_time_s_mean",
        minimum=0.0,
    )
    return {
        "planner": planner,
        "episodes": episodes,
        **counts,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "path_length_ratio_mean": path_length_ratio,
        "completion_time_s_mean": completion_time,
    }


def load_strict_candidate_summary(
    summary_path: Path,
    identity: ResultIdentity,
    gate: CheckpointSelectionGate,
) -> dict[str, float | int | str]:
    """Load one summary and enforce the complete validation selection contract."""

    try:
        raw = _mapping(json.loads(summary_path.read_text(encoding="utf-8")), "summary")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON from {summary_path}: {error}") from error
    if identity.seed != gate.seed:
        raise ValueError(
            f"Result seed {identity.seed} does not match expected seed {gate.seed}: {summary_path}"
        )
    if identity.condition.casefold() != gate.condition.casefold():
        raise ValueError(
            f"Result condition {identity.condition!r} does not match "
            f"{gate.condition!r}: {summary_path}"
        )
    split = str(raw.get("split", ""))
    if split != gate.split:
        raise ValueError(
            f"Refusing {split or 'missing'} split; expected validation: {summary_path}"
        )
    episode_count = _integer(raw.get("episode_count"), "summary.episode_count")
    if episode_count != gate.episodes:
        raise ValueError(
            f"summary.episode_count must equal {gate.episodes}, got {episode_count}."
        )
    planners = _list(raw.get("planners"), "summary.planners")
    if planners != [identity.expected_planner]:
        raise ValueError(
            "summary.planners must contain only the expected planner "
            f"{identity.expected_planner!r}."
        )
    conditions = _list(raw.get("conditions"), "summary.conditions")
    if len(conditions) != 1 or str(conditions[0]).casefold() != gate.condition.casefold():
        raise ValueError(
            f"summary.conditions must contain only condition {gate.condition!r}."
        )
    by_planner = _list(raw.get("by_planner"), "summary.by_planner")
    by_condition = _list(
        raw.get("by_planner_condition"), "summary.by_planner_condition"
    )
    if len(by_planner) != 1 or len(by_condition) != 1:
        raise ValueError(
            "summary must contain exactly one by_planner and one by_planner_condition row."
        )
    metrics = _extract_group_metrics(
        by_planner[0],
        label="summary.by_planner[0]",
        identity=identity,
        gate=gate,
        require_condition=False,
    )
    condition_metrics = _extract_group_metrics(
        by_condition[0],
        label="summary.by_planner_condition[0]",
        identity=identity,
        gate=gate,
        require_condition=True,
    )
    for name in (
        "episodes",
        "success_count",
        "collision_count",
        "timeout_count",
        "error_count",
        "success_rate",
        "collision_rate",
        "path_length_ratio_mean",
        "completion_time_s_mean",
    ):
        if metrics[name] != condition_metrics[name]:
            raise ValueError(f"Aggregate rows disagree on {name} in {summary_path}.")
    return metrics


def _metadata_final_model(
    metadata_path: Path, repository_root: Path, expected_step: int
) -> Path | None:
    try:
        raw = _mapping(yaml.safe_load(metadata_path.read_text(encoding="utf-8")), "metadata")
        if int(raw["final_model_steps"]) != expected_step:
            return None
        path = Path(str(raw["final_model"]))
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
        return None
    return path if path.is_absolute() else repository_root / path


def resolve_candidate_model(
    repository_root: Path,
    identity: ResultIdentity,
) -> tuple[Path, str]:
    """Map a validation result to its immutable checkpoint or final model path."""

    model_directory = (
        repository_root / "artifacts/models/webots" / f"seed_{identity.seed}"
    )
    if identity.kind == "checkpoint":
        expected = (
            model_directory
            / "checkpoints"
            / f"ppo_webots_s{identity.seed}_{identity.step}_steps.zip"
        )
        return expected, "checkpoint_filename" if expected.is_file() else "missing_checkpoint"

    expected = model_directory / f"ppo_webots_s{identity.seed}_step{identity.step:09d}.zip"
    possibilities: list[tuple[Path, str]] = [(expected, "zero_padded_final_filename")]
    for metadata_path in (
        model_directory / f"run_metadata_step{identity.step:09d}.yaml",
        model_directory / "run_metadata.yaml",
    ):
        if metadata_path.is_file():
            mapped = _metadata_final_model(metadata_path, repository_root, identity.step)
            if mapped is not None:
                possibilities.append((mapped, f"metadata:{metadata_path.name}"))
    for model_path in sorted(model_directory.glob(f"ppo_webots_s{identity.seed}_step*.zip")):
        match = _FINAL_MODEL_PATTERN.search(model_path.name)
        if match is not None and int(match.group("steps")) == identity.step:
            possibilities.append((model_path, "matched_final_filename"))

    existing: dict[Path, str] = {}
    for path, source in possibilities:
        if path.is_file():
            existing.setdefault(path.resolve(), source)
    if len(existing) > 1:
        hashes = {sha256_file(path) for path in existing}
        if len(hashes) > 1:
            names = ", ".join(str(path) for path in existing)
            raise ValueError(
                f"Ambiguous final-model mapping for step {identity.step}: {names}"
            )
    if existing:
        path = next(iter(existing))
        return path, existing[path]
    return expected, "missing_final_model"


def _gate_failures(
    metrics: Mapping[str, float | int | str],
    *,
    gate: CheckpointSelectionGate,
    model_exists: bool,
) -> list[str]:
    failures: list[str] = []
    if float(metrics["success_rate"]) < gate.min_success_rate:
        failures.append("success_rate_below_minimum")
    if float(metrics["collision_rate"]) > gate.max_collision_rate:
        failures.append("collision_rate_above_maximum")
    if not model_exists:
        failures.append("model_file_missing")
    return failures


def make_candidate_record(
    summary_path: Path,
    *,
    repository_root: Path,
    gate: CheckpointSelectionGate,
) -> dict[str, Any]:
    """Validate and materialize one auditable candidate record."""

    identity = parse_result_identity(summary_path)
    metrics = load_strict_candidate_summary(summary_path, identity, gate)
    model_path, resolution = resolve_candidate_model(repository_root, identity)
    model_exists = model_path.is_file()
    failures = _gate_failures(metrics, gate=gate, model_exists=model_exists)
    metric_gate_passed = (
        float(metrics["success_rate"]) >= gate.min_success_rate
        and float(metrics["collision_rate"]) <= gate.max_collision_rate
    )
    return {
        "kind": identity.kind,
        "step": identity.step,
        "seed": identity.seed,
        "condition": gate.condition,
        "planner": metrics["planner"],
        "summary_path": _portable_path(summary_path, repository_root),
        "summary_sha256": sha256_file(summary_path),
        "model_path": _portable_path(model_path, repository_root),
        "model_resolution": resolution,
        "model_exists": model_exists,
        "model_sha256": sha256_file(model_path) if model_exists else None,
        "episodes": metrics["episodes"],
        "success_count": metrics["success_count"],
        "collision_count": metrics["collision_count"],
        "timeout_count": metrics["timeout_count"],
        "error_count": metrics["error_count"],
        "success_rate": metrics["success_rate"],
        "collision_rate": metrics["collision_rate"],
        "path_length_ratio_mean": metrics["path_length_ratio_mean"],
        "completion_time_s_mean": metrics["completion_time_s_mean"],
        "metric_gate_passed": metric_gate_passed,
        "gate_passed": metric_gate_passed and model_exists,
        "gate_failures": failures,
    }


def candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, int, str]:
    """Apply the declared metric ordering plus deterministic provenance tie breakers."""

    return (
        -float(candidate["success_rate"]),
        float(candidate["collision_rate"]),
        float(candidate["path_length_ratio_mean"]),
        float(candidate["completion_time_s_mean"]),
        int(candidate["step"]),
        str(candidate["kind"]),
    )


def build_checkpoint_selection(
    summary_paths: Sequence[Path],
    *,
    repository_root: Path,
    gate: CheckpointSelectionGate,
    source_globs: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate, rank, gate, and report every discovered checkpoint candidate."""

    unique_paths = sorted({path.absolute() for path in summary_paths})
    if not unique_paths:
        raise ValueError("No checkpoint validation summaries matched the requested glob(s).")
    candidates = [
        make_candidate_record(path, repository_root=repository_root, gate=gate)
        for path in unique_paths
    ]
    candidates.sort(key=candidate_sort_key)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    available = [candidate for candidate in candidates if candidate["model_exists"]]
    formal = [candidate for candidate in candidates if candidate["gate_passed"]]
    ranking_winner = available[0] if available else None
    formal_selected = formal[0] if formal else None
    if formal_selected is not None:
        status = "formal_selected"
    elif available:
        status = "no_candidate_passed_gate"
    else:
        status = "no_available_model"

    def reference(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if candidate is None:
            return None
        return {
            "rank": candidate["rank"],
            "kind": candidate["kind"],
            "step": candidate["step"],
            "planner": candidate["planner"],
            "model_path": candidate["model_path"],
            "model_sha256": candidate["model_sha256"],
            "gate_passed": candidate["gate_passed"],
        }

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "selection_status": status,
        "selection_rule": [
            "success_rate_desc",
            "collision_rate_asc",
            "path_length_ratio_mean_asc",
            "completion_time_s_mean_asc",
        ],
        "deterministic_tie_breakers": ["step_asc", "kind_asc"],
        "source_globs": list(source_globs),
        "validation_contract": {
            "split": gate.split,
            "seed": gate.seed,
            "condition": gate.condition,
            "episodes": gate.episodes,
            "planner_template": f"ppo_{gate.condition.casefold()}_step{{step}}",
        },
        "gate": {
            "min_success_rate": gate.min_success_rate,
            "max_collision_rate": gate.max_collision_rate,
            "requires_model_file": True,
        },
        "candidate_count": len(candidates),
        "available_model_count": len(available),
        "gate_passed_count": len(formal),
        "ranking_winner": reference(ranking_winner),
        "formal_selected": reference(formal_selected),
        "candidates": candidates,
        "copy_selected": {
            "requested": False,
            "performed": False,
            "alias_path": None,
            "alias_sha256": None,
            "reason": "not_requested",
        },
    }


def write_checkpoint_selection(
    selection: dict[str, Any],
    *,
    output_directory: Path,
    repository_root: Path,
    copy_selected: bool = False,
    alias_path: Path | None = None,
) -> tuple[Path, Path]:
    """Optionally copy a gate-passing model, then write JSON and YAML manifests."""

    copy_record = selection["copy_selected"]
    copy_record["requested"] = copy_selected
    if copy_selected:
        selected = selection.get("formal_selected")
        if selected is None:
            copy_record["reason"] = "no_gate_passing_formal_selection"
        else:
            source = Path(str(selected["model_path"]))
            if not source.is_absolute():
                source = repository_root / source
            if alias_path is None:
                condition = str(selection["validation_contract"]["condition"]).casefold()
                seed = int(selection["validation_contract"]["seed"])
                alias_path = (
                    repository_root
                    / "artifacts/models/webots"
                    / f"seed_{seed}"
                    / f"ppo_webots_{condition}_validation_selected.zip"
                )
            elif not alias_path.is_absolute():
                alias_path = repository_root / alias_path
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, alias_path)
            alias_hash = sha256_file(alias_path)
            if alias_hash != selected["model_sha256"]:
                raise OSError("Copied model alias failed SHA-256 verification.")
            copy_record.update(
                {
                    "performed": True,
                    "alias_path": _portable_path(alias_path, repository_root),
                    "alias_sha256": alias_hash,
                    "reason": "copied_formal_selection",
                }
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "selection.json"
    yaml_path = output_directory / "selection.yaml"
    json_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    yaml_path.write_text(
        yaml.safe_dump(selection, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return json_path, yaml_path
