"""Fail-closed provenance checks for the formal Webots warm-start artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from indoor_patrol_rl.training.webots_curriculum import WebotsCurriculumConfig

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIREMENT_SCHEMA_VERSION = 1


class WarmStartPreflightError(ValueError):
    """The formal warm-start evidence is incomplete or internally inconsistent."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        message = "Formal Webots warm-start preflight failed:\n" + "\n".join(
            f"- {issue}" for issue in self.issues
        )
        super().__init__(message)


@dataclass(frozen=True)
class WarmStartVerification:
    """Hashes and metrics established by a successful readiness check."""

    requirement_sha256: str
    model_sha256: str
    summary_sha256: str
    episodes_sha256: str
    episode_count: int
    success_rate: float
    collision_rate: float


def _mapping(value: Any, label: str, issues: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, dict):
        issues.append(f"{label} must be a mapping")
        return None
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
    issues: list[str],
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        issues.append(f"{label} is missing fields: {missing}")
    if extra:
        issues.append(f"{label} contains unsupported fields: {extra}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str, issues: list[str]) -> str | None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        issues.append(f"{label} must be one lowercase SHA-256 digest")
        return None
    return value


def _relative_path(value: Any, label: str, issues: list[str]) -> Path | None:
    if not isinstance(value, str) or not value:
        issues.append(f"{label} must be a non-empty repository-relative path")
        return None
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if (
        normalized in {".", ".."}
        or path.is_absolute()
        or ".." in path.parts
        or re.match(r"^[A-Za-z]:/", normalized)
    ):
        issues.append(f"{label} must stay inside the repository")
        return None
    return Path(path.as_posix())


def _safe_file(
    repository_root: Path,
    relative: Path | None,
    label: str,
    issues: list[str],
) -> Path | None:
    if relative is None:
        return None
    root = repository_root.resolve(strict=True)
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            issues.append(f"{label} must not use a symbolic link: {relative.as_posix()}")
            return None
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        issues.append(f"{label} is missing: {relative.as_posix()}")
        return None
    if not resolved.is_relative_to(root):
        issues.append(f"{label} resolves outside the repository: {relative.as_posix()}")
        return None
    if not resolved.is_file():
        issues.append(f"{label} is not a regular file: {relative.as_posix()}")
        return None
    return resolved


def _integer(value: Any, label: str, issues: list[str], *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        issues.append(f"{label} must be an integer >= {minimum}")
        return None
    return value


def _rate(value: Any, label: str, issues: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(f"{label} must be a finite rate between zero and one")
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        issues.append(f"{label} must be a finite rate between zero and one")
        return None
    return number


def _load_yaml(path: Path, label: str, issues: list[str]) -> Mapping[str, Any] | None:
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label, issues)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        issues.append(f"{label} cannot be parsed: {type(error).__name__}")
        return None


def _load_json(path: Path, label: str, issues: list[str]) -> Mapping[str, Any] | None:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label, issues)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        issues.append(f"{label} cannot be parsed: {type(error).__name__}")
        return None


def _parse_boolean(value: str, label: str, issues: list[str]) -> bool | None:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    issues.append(f"{label} must be True or False")
    return None


def _row_metrics(
    rows: Sequence[Mapping[str, str]], label: str, issues: list[str]
) -> dict[str, Any]:
    success_count = collision_count = timeout_count = error_count = 0
    for index, row in enumerate(rows):
        outcome = row.get("outcome", "")
        is_success = _parse_boolean(
            row.get("is_success", ""), f"{label}[{index}].is_success", issues
        )
        collision = _parse_boolean(row.get("collision", ""), f"{label}[{index}].collision", issues)
        timeout = _parse_boolean(row.get("timeout", ""), f"{label}[{index}].timeout", issues)
        if is_success is not None:
            success_count += int(is_success)
            if is_success != (outcome == "success"):
                issues.append(f"{label}[{index}] success flag disagrees with outcome")
        if collision is not None:
            collision_count += int(collision)
            if collision != (outcome == "collision"):
                issues.append(f"{label}[{index}] collision flag disagrees with outcome")
        if timeout is not None:
            timeout_count += int(timeout)
            if timeout != (outcome == "timeout"):
                issues.append(f"{label}[{index}] timeout flag disagrees with outcome")
        error_count += int(outcome == "error")
        if outcome not in {"success", "collision", "timeout", "error"}:
            issues.append(f"{label}[{index}] has unsupported outcome {outcome!r}")
    episodes = len(rows)
    denominator = episodes or 1
    return {
        "episodes": episodes,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "success_rate": success_count / denominator,
        "collision_rate": collision_count / denominator,
        "timeout_rate": timeout_count / denominator,
    }


def _compare_aggregate(
    aggregate: Any,
    recomputed: Mapping[str, Any],
    label: str,
    issues: list[str],
) -> None:
    row = _mapping(aggregate, label, issues)
    if row is None:
        return
    for field in ("episodes", "success_count", "collision_count", "timeout_count", "error_count"):
        if row.get(field) != recomputed[field]:
            issues.append(
                f"{label}.{field} disagrees with episodes.csv: "
                f"{row.get(field)!r} != {recomputed[field]!r}"
            )
    for field in ("success_rate", "collision_rate", "timeout_rate"):
        value = row.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(float(value), float(recomputed[field]), abs_tol=1e-12)
        ):
            issues.append(f"{label}.{field} disagrees with episodes.csv")


def _verify_evidence(
    *,
    summary: Mapping[str, Any],
    episodes_path: Path,
    validation: Mapping[str, Any],
    issues: list[str],
) -> dict[str, Any] | None:
    try:
        with episodes_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as error:
        issues.append(f"validation episodes cannot be parsed: {type(error).__name__}")
        return None
    required_columns = {
        "split",
        "condition_id",
        "scenario_seed",
        "planner",
        "outcome",
        "is_success",
        "collision",
        "timeout",
    }
    if not rows or not required_columns.issubset(rows[0]):
        issues.append("validation episodes.csv is empty or missing required columns")
        return None

    split = validation.get("split")
    planner = validation.get("planner")
    conditions = validation.get("conditions")
    seed_start = validation.get("seed_start")
    seed_count = validation.get("seed_count")
    total_episode_count = validation.get("total_episode_count")
    expected_episode_count = validation.get("planner_episode_count")
    if summary.get("split") != split or {row["split"] for row in rows} != {split}:
        issues.append("validation split differs between requirement, summary, or episodes.csv")
    if summary.get("episode_count") != total_episode_count or len(rows) != total_episode_count:
        issues.append("validation total episode count differs between requirement, summary, or CSV")
    if summary.get("conditions") != conditions:
        issues.append("validation summary conditions differ from the requirement")
    if not isinstance(summary.get("planners"), list) or planner not in summary.get("planners", []):
        issues.append("validation summary does not contain the required planner")

    grouped: dict[tuple[str, str | None], list[Mapping[str, str]]] = {}
    planners = sorted({row["planner"] for row in rows})
    observed_conditions = sorted({row["condition_id"] for row in rows})
    for planner_name in planners:
        grouped[(planner_name, None)] = [row for row in rows if row["planner"] == planner_name]
        for condition in observed_conditions:
            group = [
                row
                for row in rows
                if row["planner"] == planner_name and row["condition_id"] == condition
            ]
            if group:
                grouped[(planner_name, condition)] = group

    by_planner = summary.get("by_planner")
    by_condition = summary.get("by_planner_condition")
    if not isinstance(by_planner, list) or not isinstance(by_condition, list):
        issues.append("validation summary aggregate lists are missing")
        return None
    planner_aggregates = {row.get("planner"): row for row in by_planner if isinstance(row, dict)}
    condition_aggregates = {
        (row.get("planner"), row.get("condition_id")): row
        for row in by_condition
        if isinstance(row, dict)
    }
    if set(planner_aggregates) != set(planners):
        issues.append("validation by_planner keys disagree with episodes.csv")
    if set(condition_aggregates) != {key for key in grouped if key[1] is not None}:
        issues.append("validation by_planner_condition keys disagree with episodes.csv")
    for key, group in grouped.items():
        metrics = _row_metrics(group, f"episodes[{key[0]}/{key[1] or 'all'}]", issues)
        aggregate = (
            planner_aggregates.get(key[0]) if key[1] is None else condition_aggregates.get(key)
        )
        _compare_aggregate(aggregate, metrics, f"summary[{key[0]}/{key[1] or 'all'}]", issues)

    selected_rows = grouped.get((str(planner), None), [])
    selected_metrics = _row_metrics(selected_rows, f"episodes[{planner}/selected]", issues)
    if selected_metrics["episodes"] != expected_episode_count:
        issues.append("required planner episode count does not match episodes.csv")
    if (
        isinstance(conditions, list)
        and all(isinstance(condition, str) and condition for condition in conditions)
        and isinstance(seed_start, int)
        and isinstance(seed_count, int)
    ):
        expected_grid = Counter(
            (condition, seed)
            for condition in conditions
            for seed in range(seed_start, seed_start + seed_count)
        )
        observed_grid: Counter[tuple[str, int]] = Counter()
        for index, row in enumerate(selected_rows):
            try:
                seed = int(row["scenario_seed"])
            except ValueError:
                issues.append(f"selected planner row {index} has an invalid scenario_seed")
                continue
            observed_grid[(row["condition_id"], seed)] += 1
        if observed_grid != expected_grid:
            issues.append("required planner condition/seed grid differs from episodes.csv")
    return selected_metrics


def verify_warm_start_requirement(
    *,
    repository_root: Path,
    config: WebotsCurriculumConfig,
) -> WarmStartVerification | None:
    """Verify every declared warm-start input before a formal runner may start."""

    if config.warm_start_requirement is None:
        return None
    issues: list[str] = []
    requirement_relative = _relative_path(
        config.warm_start_requirement.as_posix(),
        "warm_start_requirement",
        issues,
    )
    requirement_path = _safe_file(
        repository_root,
        requirement_relative,
        "warm-start requirement manifest",
        issues,
    )
    if requirement_path is None:
        raise WarmStartPreflightError(issues)
    requirement_sha256 = _sha256(requirement_path)
    raw = _load_yaml(requirement_path, "warm-start requirement manifest", issues)
    if raw is None:
        raise WarmStartPreflightError(issues)
    _exact_keys(
        raw,
        {"schema_version", "requirement_id", "curriculum", "model", "registry", "validation"},
        "warm-start requirement manifest",
        issues,
    )
    if raw.get("schema_version") != _REQUIREMENT_SCHEMA_VERSION:
        issues.append("warm-start requirement schema_version must equal 1")

    curriculum = _mapping(raw.get("curriculum"), "requirement.curriculum", issues) or {}
    model = _mapping(raw.get("model"), "requirement.model", issues) or {}
    registry = _mapping(raw.get("registry"), "requirement.registry", issues) or {}
    validation = _mapping(raw.get("validation"), "requirement.validation", issues) or {}
    _exact_keys(
        curriculum, {"config_path", "transfer_model_path"}, "requirement.curriculum", issues
    )
    _exact_keys(model, {"sha256", "actual_steps"}, "requirement.model", issues)
    _exact_keys(registry, {"path", "section", "candidate_id"}, "requirement.registry", issues)
    _exact_keys(
        validation,
        {
            "split",
            "planner",
            "conditions",
            "seed_start",
            "seed_count",
            "total_episode_count",
            "planner_episode_count",
            "success_count",
            "collision_count",
            "timeout_count",
            "error_count",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "summary_path",
            "summary_sha256",
            "episodes_path",
            "episodes_sha256",
        },
        "requirement.validation",
        issues,
    )

    config_relative = _relative_path(
        curriculum.get("config_path"), "curriculum.config_path", issues
    )
    transfer_relative = _relative_path(
        curriculum.get("transfer_model_path"), "curriculum.transfer_model_path", issues
    )
    registry_relative = _relative_path(registry.get("path"), "registry.path", issues)
    summary_relative = _relative_path(
        validation.get("summary_path"), "validation.summary_path", issues
    )
    episodes_relative = _relative_path(
        validation.get("episodes_path"), "validation.episodes_path", issues
    )
    model_digest = _digest(model.get("sha256"), "model.sha256", issues)
    summary_digest = _digest(validation.get("summary_sha256"), "validation.summary_sha256", issues)
    episodes_digest = _digest(
        validation.get("episodes_sha256"), "validation.episodes_sha256", issues
    )
    actual_steps = _integer(model.get("actual_steps"), "model.actual_steps", issues, minimum=1)
    integer_values = {
        name: _integer(
            validation.get(name),
            f"validation.{name}",
            issues,
            minimum=(
                1 if name in {"seed_count", "total_episode_count", "planner_episode_count"} else 0
            ),
        )
        for name in (
            "seed_start",
            "seed_count",
            "total_episode_count",
            "planner_episode_count",
            "success_count",
            "collision_count",
            "timeout_count",
            "error_count",
        )
    }
    success_rate = _rate(validation.get("success_rate"), "validation.success_rate", issues)
    collision_rate = _rate(validation.get("collision_rate"), "validation.collision_rate", issues)
    timeout_rate = _rate(validation.get("timeout_rate"), "validation.timeout_rate", issues)
    if validation.get("split") != "validation":
        issues.append("validation.split must be validation; held-out test evidence is forbidden")
    if not isinstance(validation.get("planner"), str) or not validation.get("planner"):
        issues.append("validation.planner must be a non-empty string")
    conditions = validation.get("conditions")
    conditions_are_strings = isinstance(conditions, list) and all(
        isinstance(condition, str) and condition for condition in conditions
    )
    if not conditions_are_strings or not conditions or len(conditions) != len(set(conditions)):
        issues.append("validation.conditions must contain unique condition ids")
    if all(
        integer_values[name] is not None
        for name in (
            "planner_episode_count",
            "success_count",
            "collision_count",
            "timeout_count",
            "error_count",
        )
    ):
        outcome_count = sum(
            int(integer_values[name])
            for name in ("success_count", "collision_count", "timeout_count", "error_count")
        )
        if outcome_count != integer_values["planner_episode_count"]:
            issues.append("validation outcome counts must sum to planner_episode_count")
    if (
        isinstance(conditions, list)
        and conditions
        and integer_values["seed_count"] is not None
        and integer_values["planner_episode_count"] is not None
        and len(conditions) * integer_values["seed_count"]
        != integer_values["planner_episode_count"]
    ):
        issues.append("validation conditions × seed_count must equal planner_episode_count")
    rate_count_pairs = (
        (success_rate, integer_values["success_count"], "success_rate"),
        (collision_rate, integer_values["collision_count"], "collision_rate"),
        (timeout_rate, integer_values["timeout_count"], "timeout_rate"),
    )
    planner_episode_count = integer_values["planner_episode_count"]
    if planner_episode_count:
        for rate, count, label in rate_count_pairs:
            if (
                rate is not None
                and count is not None
                and not math.isclose(rate, count / planner_episode_count, abs_tol=1e-12)
            ):
                issues.append(f"validation.{label} must equal its count / planner_episode_count")

    config_path = _safe_file(repository_root, config_relative, "curriculum config", issues)
    if config.source_path is None:
        issues.append("loaded curriculum does not record its source path")
    elif config_path is not None:
        source_path = config.source_path
        if source_path.is_absolute():
            try:
                source_relative = source_path.absolute().relative_to(repository_root.absolute())
            except ValueError:
                issues.append("loaded curriculum source must stay inside the repository")
                source_relative = None
        else:
            source_relative = source_path
        loaded_source = _safe_file(
            repository_root,
            source_relative,
            "loaded curriculum source",
            issues,
        )
        if loaded_source is not None and loaded_source != config_path:
            issues.append(
                "loaded curriculum source differs from requirement.curriculum.config_path"
            )
    configured_transfer = Path(config.transfer_model.as_posix())
    if transfer_relative is not None and configured_transfer != transfer_relative:
        issues.append("curriculum transfer_model differs from the warm-start requirement")
    if config.policy_transfer_model is not None:
        configured_policy_transfer = Path(config.policy_transfer_model.as_posix())
        if transfer_relative is not None and configured_policy_transfer != transfer_relative:
            issues.append(
                "curriculum policy_transfer_model differs from the warm-start requirement"
            )

    model_path = _safe_file(repository_root, transfer_relative, "warm-start model", issues)
    actual_model_digest = _sha256(model_path) if model_path is not None else None
    if (
        model_digest is not None
        and actual_model_digest is not None
        and actual_model_digest != model_digest
    ):
        issues.append(
            "warm-start model SHA-256 mismatch: restore the declared retrained artifact "
            "without overwriting the current legacy file"
        )

    registry_path = _safe_file(repository_root, registry_relative, "model registry", issues)
    registry_raw = (
        _load_yaml(registry_path, "model registry", issues) if registry_path is not None else None
    )
    candidate: Mapping[str, Any] | None = None
    if registry_raw is not None:
        section_name = registry.get("section")
        section = registry_raw.get(section_name) if isinstance(section_name, str) else None
        section_mapping = _mapping(section, "model registry section", issues)
        candidates = section_mapping.get("candidates") if section_mapping is not None else None
        if not isinstance(candidates, list):
            issues.append("model registry section.candidates must be a list")
        else:
            matches = [
                item
                for item in candidates
                if isinstance(item, dict) and item.get("id") == registry.get("candidate_id")
            ]
            if len(matches) != 1:
                issues.append("model registry must contain exactly one required candidate")
            else:
                candidate = matches[0]
    if candidate is not None:
        expected_registry = {
            "path": transfer_relative.as_posix() if transfer_relative is not None else None,
            "sha256": model_digest,
            "actual_steps": actual_steps,
            "validation_episodes": validation.get("planner_episode_count"),
            "validation_success_rate": success_rate,
            "validation_collision_rate": collision_rate,
            "selected": True,
            "role": "webots_policy_warm_start",
        }
        for field, expected in expected_registry.items():
            if candidate.get(field) != expected:
                issues.append(
                    f"model registry candidate.{field} differs from the requirement: "
                    f"{candidate.get(field)!r} != {expected!r}"
                )

    summary_path = _safe_file(repository_root, summary_relative, "validation summary", issues)
    episodes_path = _safe_file(repository_root, episodes_relative, "validation episodes", issues)
    actual_summary_digest = _sha256(summary_path) if summary_path is not None else None
    actual_episodes_digest = _sha256(episodes_path) if episodes_path is not None else None
    if (
        summary_digest is not None
        and actual_summary_digest is not None
        and summary_digest != actual_summary_digest
    ):
        issues.append("validation summary SHA-256 differs from the requirement")
    if (
        episodes_digest is not None
        and actual_episodes_digest is not None
        and episodes_digest != actual_episodes_digest
    ):
        issues.append("validation episodes SHA-256 differs from the requirement")

    metrics: dict[str, Any] | None = None
    if summary_path is not None and episodes_path is not None:
        summary = _load_json(summary_path, "validation summary", issues)
        if summary is not None:
            metrics = _verify_evidence(
                summary=summary,
                episodes_path=episodes_path,
                validation=validation,
                issues=issues,
            )
    if metrics is not None:
        expected_metrics = {
            "episodes": validation.get("planner_episode_count"),
            "success_count": validation.get("success_count"),
            "collision_count": validation.get("collision_count"),
            "timeout_count": validation.get("timeout_count"),
            "error_count": validation.get("error_count"),
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
        }
        for field, expected in expected_metrics.items():
            actual = metrics[field]
            if isinstance(expected, float):
                matches = math.isclose(float(actual), expected, abs_tol=1e-12)
            else:
                matches = actual == expected
            if not matches:
                issues.append(
                    f"validation {field} recomputed from CSV differs from requirement: "
                    f"{actual!r} != {expected!r}"
                )

    if issues:
        raise WarmStartPreflightError(issues)
    assert model_digest is not None
    assert summary_digest is not None
    assert episodes_digest is not None
    assert success_rate is not None
    assert collision_rate is not None
    return WarmStartVerification(
        requirement_sha256=requirement_sha256,
        model_sha256=model_digest,
        summary_sha256=summary_digest,
        episodes_sha256=episodes_digest,
        episode_count=int(validation["planner_episode_count"]),
        success_rate=success_rate,
        collision_rate=collision_rate,
    )
