"""Diagnostic-only, full-episode safety-shield intervention evidence.

This sidecar deliberately does not extend ``episode_context.jsonl`` schema 1.1.
Keeping the artifact separate preserves the byte-level contract used by the
completed risk analysis while allowing a pre-registered diagnostic to prove
that its safety-shield treatment was operationally active.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SHIELD_EVIDENCE_FILENAME = "shield_evidence.jsonl"
SHIELD_EVIDENCE_SCHEMA_VERSION = "1.0"

_IDENTITY_FIELDS = frozenset({"episode_id", "split", "condition_id", "scenario_seed", "planner"})
_STEP_FIELDS = frozenset(
    {
        "step",
        "policy_action",
        "shielded_action",
        "executed_action",
        "shield_enabled",
        "shield_active",
        "shield_intervention",
        "shield_front_clearance",
        "shield_left_clearance",
        "shield_right_clearance",
        "shield_speed_fraction_limit",
        "shield_normalized_speed_limit",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "shield_enabled",
        "step_count",
        "intervention_epsilon",
        "active_step_count",
        "active_step_ratio",
        "intervention_sum",
        "intervention_mean",
    }
)
_RECORD_FIELDS = frozenset({"schema_version", "identity", "steps", "summary"})


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def _finite_vector(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-value array.")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a finite two-value array.")
    if np.any(result < -1.0) or np.any(result > 1.0):
        raise ValueError(f"{field} values must remain within [-1, 1].")
    return result


def _validate_identity(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _IDENTITY_FIELDS:
        raise ValueError(f"{field} must contain exactly the shield-evidence identity fields.")
    if value["split"] != "diagnostic":
        raise PermissionError(f"{field}.split must be diagnostic.")
    for name in ("episode_id", "condition_id", "planner"):
        if not isinstance(value[name], str) or not value[name] or "\n" in value[name]:
            raise ValueError(f"{field}.{name} must be a non-empty single-line string.")
    if type(value["scenario_seed"]) is not int or value["scenario_seed"] < 0:
        raise ValueError(f"{field}.scenario_seed must be a non-negative integer.")
    return value


def _validate_step(value: Any, field: str, *, enabled: bool, epsilon: float) -> None:
    if not isinstance(value, dict) or set(value) != _STEP_FIELDS:
        raise ValueError(f"{field} must contain exactly the shield-evidence step fields.")
    if type(value["step"]) is not int or value["step"] < 1:
        raise ValueError(f"{field}.step must be a positive integer.")
    for name in ("shield_enabled", "shield_active"):
        if type(value[name]) is not bool:
            raise ValueError(f"{field}.{name} must be a boolean.")
    if value["shield_enabled"] is not enabled:
        raise ValueError(f"{field}.shield_enabled disagrees with the episode summary.")

    policy = _finite_vector(value["policy_action"], f"{field}.policy_action")
    shielded = _finite_vector(value["shielded_action"], f"{field}.shielded_action")
    _finite_vector(value["executed_action"], f"{field}.executed_action")
    intervention = _finite_number(value["shield_intervention"], f"{field}.shield_intervention")
    if intervention < 0.0:
        raise ValueError(f"{field}.shield_intervention cannot be negative.")
    expected_intervention = float(np.linalg.norm(shielded - policy))
    if not math.isclose(intervention, expected_intervention, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{field}.shield_intervention disagrees with the action delta.")
    expected_active = bool(enabled and intervention > epsilon)
    if value["shield_active"] is not expected_active:
        raise ValueError(f"{field}.shield_active disagrees with intervention_epsilon.")
    if not enabled and not np.array_equal(policy, shielded):
        raise ValueError(f"{field} disabled shield must preserve the policy action.")

    for name in (
        "shield_front_clearance",
        "shield_left_clearance",
        "shield_right_clearance",
    ):
        if _finite_number(value[name], f"{field}.{name}") < 0.0:
            raise ValueError(f"{field}.{name} cannot be negative.")
    fraction = _finite_number(
        value["shield_speed_fraction_limit"],
        f"{field}.shield_speed_fraction_limit",
    )
    normalized = _finite_number(
        value["shield_normalized_speed_limit"],
        f"{field}.shield_normalized_speed_limit",
    )
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{field}.shield_speed_fraction_limit must be within [0, 1].")
    if not -1.0 <= normalized <= 1.0:
        raise ValueError(f"{field}.shield_normalized_speed_limit must be within [-1, 1].")
    if not math.isclose(normalized, 2.0 * fraction - 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{field} speed-limit representations disagree.")


def validate_shield_evidence_record(value: Any, *, field: str = "shield_evidence") -> None:
    """Strictly validate one complete diagnostic episode of shield evidence."""

    if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
        raise ValueError(f"{field} must contain exactly the shield-evidence record fields.")
    if value["schema_version"] != SHIELD_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"{field}.schema_version must be {SHIELD_EVIDENCE_SCHEMA_VERSION}.")
    _validate_identity(value["identity"], f"{field}.identity")
    summary = value["summary"]
    if not isinstance(summary, dict) or set(summary) != _SUMMARY_FIELDS:
        raise ValueError(f"{field}.summary must contain exactly the summary fields.")
    if type(summary["shield_enabled"]) is not bool:
        raise ValueError(f"{field}.summary.shield_enabled must be a boolean.")
    if type(summary["step_count"]) is not int or summary["step_count"] < 1:
        raise ValueError(f"{field}.summary.step_count must be a positive integer.")
    epsilon = _finite_number(
        summary["intervention_epsilon"], f"{field}.summary.intervention_epsilon"
    )
    if epsilon <= 0.0:
        raise ValueError(f"{field}.summary.intervention_epsilon must be positive.")
    if type(summary["active_step_count"]) is not int or not (
        0 <= summary["active_step_count"] <= summary["step_count"]
    ):
        raise ValueError(f"{field}.summary.active_step_count is invalid.")

    steps = value["steps"]
    if not isinstance(steps, list) or len(steps) != summary["step_count"]:
        raise ValueError(f"{field}.steps must contain exactly step_count entries.")
    for index, step in enumerate(steps, start=1):
        _validate_step(
            step,
            f"{field}.steps[{index - 1}]",
            enabled=summary["shield_enabled"],
            epsilon=epsilon,
        )
        if step["step"] != index:
            raise ValueError(f"{field}.steps must be contiguous and start at one.")

    active_count = sum(bool(step["shield_active"]) for step in steps)
    intervention_sum = math.fsum(float(step["shield_intervention"]) for step in steps)
    expected_values = {
        "active_step_count": active_count,
        "active_step_ratio": active_count / len(steps),
        "intervention_sum": intervention_sum,
        "intervention_mean": intervention_sum / len(steps),
    }
    for name, expected in expected_values.items():
        actual = summary[name]
        if name == "active_step_count":
            if actual != expected:
                raise ValueError(f"{field}.summary.{name} disagrees with steps.")
            continue
        number = _finite_number(actual, f"{field}.summary.{name}")
        if number < 0.0 or not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{field}.summary.{name} disagrees with steps.")


def shield_evidence_step(info: Mapping[str, Any]) -> dict[str, Any]:
    """Project one Webots step-info mapping into the stable evidence schema."""

    required = _STEP_FIELDS - {"step"}
    missing = required - info.keys()
    if missing:
        raise ValueError(f"Shield evidence step is missing fields: {sorted(missing)}")
    if "step_count" not in info:
        raise ValueError("Shield evidence step is missing step_count.")
    result = {
        "step": info["step_count"],
        "policy_action": np.asarray(info["policy_action"], dtype=np.float64).tolist(),
        "shielded_action": np.asarray(info["shielded_action"], dtype=np.float64).tolist(),
        "executed_action": np.asarray(info["executed_action"], dtype=np.float64).tolist(),
        **{
            name: info[name]
            for name in required - {"policy_action", "shielded_action", "executed_action"}
        },
    }
    return result


def build_shield_evidence_record(
    *,
    identity: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    intervention_epsilon: float,
    terminal_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and cross-check one record against the environment's episode counters."""

    projected = [shield_evidence_step(step) for step in steps]
    if not projected:
        raise ValueError("Shield evidence requires at least one step.")
    enabled_values = {step["shield_enabled"] for step in projected}
    if len(enabled_values) != 1:
        raise ValueError("Shield evidence shield_enabled changed within an episode.")
    enabled = enabled_values.pop()
    active_count = sum(bool(step["shield_active"]) for step in projected)
    intervention_sum = math.fsum(float(step["shield_intervention"]) for step in projected)
    record = {
        "schema_version": SHIELD_EVIDENCE_SCHEMA_VERSION,
        "identity": dict(identity),
        "steps": projected,
        "summary": {
            "shield_enabled": enabled,
            "step_count": len(projected),
            "intervention_epsilon": float(intervention_epsilon),
            "active_step_count": active_count,
            "active_step_ratio": active_count / len(projected),
            "intervention_sum": intervention_sum,
            "intervention_mean": intervention_sum / len(projected),
        },
    }
    validate_shield_evidence_record(record)
    expected_terminal = {
        "step_count": len(projected),
        "shield_intervention_epsilon": float(intervention_epsilon),
        "episode_shield_intervention_ratio": record["summary"]["active_step_ratio"],
        "episode_shield_intervention_mean": record["summary"]["intervention_mean"],
    }
    for name, expected in expected_terminal.items():
        if name not in terminal_info:
            raise ValueError(f"Terminal info is missing {name}.")
        actual = terminal_info[name]
        if name == "step_count":
            if actual != expected:
                raise ValueError("Terminal step_count disagrees with shield evidence.")
        elif not math.isclose(
            _finite_number(actual, f"terminal_info.{name}"),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Terminal {name} disagrees with shield evidence.")
    return record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_shield_evidence_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Atomically create a bound sidecar without overwriting existing evidence."""

    if not records:
        raise ValueError("At least one shield-evidence record is required.")
    validated = [dict(record) for record in records]
    for index, record in enumerate(validated):
        validate_shield_evidence_record(record, field=f"shield_evidence row {index}")
    episode_ids = [record["identity"]["episode_id"] for record in validated]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Shield-evidence episode_id values must be unique.")

    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        portable = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Shield-evidence artifact must remain inside the repository.") from error
    if resolved.exists():
        raise FileExistsError(f"Shield-evidence artifact already exists: {resolved}")

    lines = [
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for record in validated
    ]
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Shield-evidence temporary artifact already exists: {temporary}")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.link(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": portable,
        "schema_version": SHIELD_EVIDENCE_SCHEMA_VERSION,
        "count": len(validated),
        "sha256": _sha256_file(resolved),
    }


__all__ = [
    "SHIELD_EVIDENCE_FILENAME",
    "SHIELD_EVIDENCE_SCHEMA_VERSION",
    "build_shield_evidence_record",
    "shield_evidence_step",
    "validate_shield_evidence_record",
    "write_shield_evidence_jsonl",
]
