#!/usr/bin/env python3
"""Generate the public v1-audit-to-v2 Webots front-brake report.

This generator is deliberately offline and dependency-free.  It verifies the
sealed v1 report, the post-run manipulation-boundary audit, the independent v2
confirmation report, and every v2 raw artifact before deriving the published
numbers and SVGs.  It never launches Webots and never rewrites experiment data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import struct
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

LOCKED_V1_FILES = {
    "paired_diagnostic_report.json": (
        "2779e8dcb6f8ccf2216464e196dc5c5f3a5ba1c8092eb0d75cb14fb62e2cc787"
    ),
    "paired_diagnostic_report.yaml": (
        "db161fa425e25255381b60d99aece212b072fdbfa87061532e4f2a3f9289ff75"
    ),
    "paired_diagnostic_report.sha256": (
        "b255ab30a0bb0214dbaa4dd7a1b52eab02dcac441683eb6d4f3ef071a17b1427"
    ),
}
LOCKED_AUDIT_FILES = {
    "manipulation_boundary_reaudit.json": (
        "aac1bf62e755a1868837c294adc4e2d13cf9949a4dcba484d0c0b18c4ffd16ed"
    ),
    "manipulation_boundary_reaudit.yaml": (
        "deb53cd3cd16c8a89cba5d26cf627d79fd4b04d7f9ad71a8d05e067f19d391e2"
    ),
    "manipulation_boundary_reaudit.sha256": (
        "75f36c1c73b5686b8319b4d1134a622ebee67d59532f428f628bfe911e8d6379"
    ),
}
LOCKED_V2_FILES = {
    "paired_diagnostic_report.json": (
        "62397f5ffa892183dcca24438ea8c5de98c32adb7dcacad13fde47291cbe23d0"
    ),
    "paired_diagnostic_report.yaml": (
        "cb51dc6d0d2ddb43c1d129476596988dbc37d695fd613ebe2ba599360dbe3309"
    ),
    "paired_diagnostic_report.sha256": (
        "fdff954b34eb2f3b0bedc01cf2ac652ce74967dca84e13c593372d70e0923890"
    ),
}
LOCKED_V2_RAW_FILES = {
    "reference": {
        "summary_json": (
            "artifacts/results/webots_diagnostic_front_brake_v2_reference_seed11/summary.json",
            "5153fb51ec749f205ed6e21615e68f3674f9946b93c65b7fda68305652d5e210",
        ),
        "episodes_csv": (
            "artifacts/results/webots_diagnostic_front_brake_v2_reference_seed11/episodes.csv",
            "e9c6f0d92d561f0d374baf446106c394b577cc165c58bc61c43077be01b3dad3",
        ),
        "episode_context_jsonl": (
            "artifacts/results/webots_diagnostic_front_brake_v2_reference_seed11/episode_context.jsonl",
            "fa0ee4cb70e5377ba138ea26595fcec783c1a317aeaa79b318e21c8c23f5e157",
        ),
        "shield_evidence_jsonl": (
            "artifacts/results/webots_diagnostic_front_brake_v2_reference_seed11/shield_evidence.jsonl",
            "247540a73ced04b1a0c4c5075c3ff0a85a9bd11c43fb25334e0c61cd89b1879b",
        ),
    },
    "candidate": {
        "summary_json": (
            "artifacts/results/webots_diagnostic_front_brake_v2_candidate_seed11/summary.json",
            "38867d29a074a15ef6e0fa10fd028a58cf7ebf2b7f4da4983b6035145970ec91",
        ),
        "episodes_csv": (
            "artifacts/results/webots_diagnostic_front_brake_v2_candidate_seed11/episodes.csv",
            "b35daccaddd6fdbb693d090f1b8d0de17a6e2bc3318272a689586732a0502643",
        ),
        "episode_context_jsonl": (
            "artifacts/results/webots_diagnostic_front_brake_v2_candidate_seed11/episode_context.jsonl",
            "c42cbb2b5f3383ff56d76fee80adeb0e28f2d285f93fd179451b100d48dc41c4",
        ),
        "shield_evidence_jsonl": (
            "artifacts/results/webots_diagnostic_front_brake_v2_candidate_seed11/shield_evidence.jsonl",
            "f3e2d822c339ae3123f750db5253f30e21df511a00221944d583d5a2aadf3f7a",
        ),
    },
}

EXPECTED_V1_SEEDS = tuple(range(25728, 25848))
EXPECTED_V2_SEEDS = tuple(range(54775, 54895))
OUTPUT_FILENAMES = (
    "confirmation_report.md",
    "outcome_rates.svg",
    "paired_transitions.svg",
    "safety_tradeoff.svg",
    "confirmation_report.sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object.")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array.")
    return value


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse {label}: {error}") from error


def _require_file_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise PermissionError(
            f"{label} failed its locked SHA-256: expected {expected}, actual {actual}."
        )
    return actual


def _verify_trilogy(
    json_path: Path,
    *,
    locked_files: Mapping[str, str],
    label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    directory = json_path.parent
    paths = {name: directory / name for name in locked_files}
    hashes = {
        name: _require_file_sha(path, locked_files[name], f"{label} {name}")
        for name, path in paths.items()
    }
    sidecar_name = next(name for name in locked_files if name.endswith(".sha256"))
    expected_lines = [
        f"{locked_files[name]}  {name}" for name in locked_files if not name.endswith(".sha256")
    ]
    sidecar_lines = paths[sidecar_name].read_text(encoding="utf-8").splitlines()
    if sidecar_lines != expected_lines:
        raise PermissionError(f"{label} SHA-256 sidecar does not exactly bind JSON and YAML.")
    return _mapping(_load_json(json_path, f"{label} JSON"), f"{label} JSON"), hashes


def _relative_artifact_path(repository_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} path must be a string.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError(f"{label} path must be repository-relative without '..'.")
    root = repository_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"{label} escapes the repository root.") from error
    return resolved


def _validate_v1(report: Mapping[str, Any]) -> None:
    scenario = _mapping(report.get("scenario"), "v1.scenario")
    validity = _mapping(report.get("experiment_validity"), "v1.experiment_validity")
    manipulation = _mapping(
        report.get("manipulation_integrity_check"), "v1.manipulation_integrity_check"
    )
    checks = _mapping(manipulation.get("checks"), "v1.manipulation checks")
    candidate_metrics = _mapping(manipulation.get("candidate_metrics"), "v1.candidate metrics")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != "webots_front_brake_paired_diagnostic_v1"
        or report.get("status") != "invalid_manipulation"
        or report.get("split") != "diagnostic"
        or report.get("validation_split_was_run") is not False
        or report.get("test_split_was_run") is not False
    ):
        raise PermissionError("The sealed v1 report must remain invalid_manipulation/diagnostic.")
    if (
        scenario.get("scenario_seed_start") != EXPECTED_V1_SEEDS[0]
        or scenario.get("episodes") != len(EXPECTED_V1_SEEDS)
        or scenario.get("scenario_seeds") != list(EXPECTED_V1_SEEDS)
    ):
        raise PermissionError("Unexpected sealed v1 seed window.")
    failed_checks = {name for name, passed in checks.items() if passed is not True}
    if failed_checks != {"candidate_active_steps_are_exact_speed_cap_braking"}:
        raise PermissionError("v1 must retain exactly its original manipulation-check failure.")
    if (
        manipulation.get("passed") is not False
        or validity
        != {
            "manipulation_integrity_required": True,
            "manipulation_integrity_passed": False,
            "primary_efficacy_gate_passed": True,
            "overall_passed": False,
        }
        or report.get("candidate_gate", {}).get("passed") is not True
        or candidate_metrics.get("active_step_count") != 3197
        or candidate_metrics.get("braking_step_count") != 3197
        or candidate_metrics.get("speed_cap_binding_step_count") != 3196
    ):
        raise PermissionError("Unexpected sealed v1 validity facts.")


def _validate_audit(audit: Mapping[str, Any]) -> None:
    facts = _mapping(audit.get("original_report_facts"), "audit.original_report_facts")
    counts = _mapping(facts.get("reported_counts"), "audit.reported_counts")
    boundary = _mapping(audit.get("numerical_boundary"), "audit.numerical_boundary")
    conclusion = _mapping(audit.get("checker_conclusion"), "audit.checker_conclusion")
    scope = _mapping(audit.get("scope"), "audit.scope")
    inputs = _mapping(audit.get("inputs"), "audit.inputs")
    report_artifacts = _mapping(
        inputs.get("paired_diagnostic_report_artifacts"), "audit.paired report artifacts"
    )
    if (
        audit.get("schema_version") != 1
        or audit.get("audit_id") != "webots_front_brake_v1_manipulation_boundary_reaudit_v1"
        or audit.get("status") != "passed"
        or audit.get("audit_type") != "offline_post_run_checker_audit"
    ):
        raise PermissionError("Unexpected manipulation-boundary audit identity or status.")
    for key, name in (
        ("json", "paired_diagnostic_report.json"),
        ("yaml", "paired_diagnostic_report.yaml"),
        ("sha256", "paired_diagnostic_report.sha256"),
    ):
        binding = _mapping(report_artifacts.get(key), f"audit paired report {key}")
        expected_path = f"artifacts/results/webots_front_brake_paired_diagnostic_v1/{name}"
        if binding != {"path": expected_path, "sha256": LOCKED_V1_FILES[name]}:
            raise PermissionError("Audit does not bind the exact sealed v1 report trilogy.")
    if (
        facts.get("status") != "invalid_manipulation"
        or facts.get("primary_efficacy_gate_passed") is not True
        or facts.get("only_failed_manipulation_check")
        != "candidate_active_steps_are_exact_speed_cap_braking"
        or counts
        != {
            "active_step_count": 3197,
            "braking_step_count": 3197,
            "speed_cap_binding_step_count": 3196,
        }
        or facts.get("original_report_was_modified") is not False
        or facts.get("original_status_is_not_rewritten") is not True
    ):
        raise PermissionError("Audit no longer preserves the original v1 result.")
    if (
        boundary.get("scenario_seed") != 25737
        or boundary.get("step") != 21
        or boundary.get("boundary_condition_verified") is not True
        or boundary.get("legacy_double_cap_test_passed") is not False
        or boundary.get("operational_float32_action_test_passed") is not True
        or boundary.get("classification")
        != "double_cap_vs_float32_admitted_action_threshold_boundary"
    ):
        raise PermissionError("Unexpected audited float32 numerical boundary.")
    if (
        conclusion.get("corrected_manipulation_integrity_passed") is not True
        or conclusion.get("original_paired_report_remains_invalid_manipulation") is not True
        or conclusion.get("primary_efficacy_result_is_not_promoted_by_this_post_hoc_audit")
        is not True
        or conclusion.get("next_allowed_stage") != "independent_diagnostic_confirmation_only"
        or conclusion.get("validation_or_test_confirmation_allowed") is not False
        or scope.get("webots_was_run") is not False
        or scope.get("validation_split_was_accessed") is not False
        or scope.get("test_split_was_accessed") is not False
        or scope.get("input_artifacts_were_modified") is not False
    ):
        raise PermissionError("Audit scope does not preserve the diagnostic-only boundary.")


def _validate_v2(report: Mapping[str, Any]) -> None:
    scenario = _mapping(report.get("scenario"), "v2.scenario")
    prior = _mapping(report.get("prior_evidence"), "v2.prior_evidence")
    restrictions = _mapping(prior.get("restrictions"), "v2.prior restrictions")
    source_bindings = _mapping(report.get("source_bindings"), "v2.source_bindings")
    preregistration = _mapping(report.get("preregistration"), "v2.preregistration")
    validity = _mapping(report.get("experiment_validity"), "v2.experiment_validity")
    interpretation = _mapping(report.get("interpretation_scope"), "v2.interpretation_scope")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != "webots_front_brake_paired_confirmation_v2"
        or report.get("status") != "passed"
        or report.get("split") != "diagnostic"
        or report.get("validation_split_was_run") is not False
        or report.get("test_split_was_run") is not False
    ):
        raise PermissionError("v2 must be the passed, diagnostic-only independent confirmation.")
    if (
        scenario.get("scenario_seed_start") != EXPECTED_V2_SEEDS[0]
        or scenario.get("episodes") != len(EXPECTED_V2_SEEDS)
        or scenario.get("scenario_seeds") != list(EXPECTED_V2_SEEDS)
        or set(EXPECTED_V1_SEEDS) & set(EXPECTED_V2_SEEDS)
    ):
        raise PermissionError("v2 does not use the locked independent seed window 54775..54894.")
    expected_restrictions = {
        "v1_rerun": "forbidden",
        "v1_reinterpretation": "forbidden",
        "v1_report_rewrite": "forbidden",
        "v1_v2_pooled_fallback": "forbidden",
    }
    if (
        restrictions != expected_restrictions
        or prior.get("v1_status_remains") != "invalid_manipulation"
        or prior.get("confirmation_uses_v2_overall_only") is not True
        or preregistration.get("primary_inference") != "v2_overall_only"
        or preregistration.get("parameter_scan") != "forbidden"
        or preregistration.get("adaptive_extension") != "forbidden"
        or preregistration.get("validation_or_test") != "forbidden"
    ):
        raise PermissionError(
            "v2 must forbid reinterpretation, pooling, and validation/test access."
        )
    for key, expected_path, expected_hash in (
        (
            "v1_paired_report",
            "artifacts/results/webots_front_brake_paired_diagnostic_v1/paired_diagnostic_report.json",
            LOCKED_V1_FILES["paired_diagnostic_report.json"],
        ),
        (
            "manipulation_boundary_reaudit",
            "artifacts/results/webots_front_brake_v1_manipulation_reaudit_v1/manipulation_boundary_reaudit.json",
            LOCKED_AUDIT_FILES["manipulation_boundary_reaudit.json"],
        ),
    ):
        binding = _mapping(source_bindings.get(key), f"v2.source_bindings.{key}")
        if binding != {"path": expected_path, "sha256": expected_hash}:
            raise PermissionError("v2 does not bind the exact sealed v1/audit JSON inputs.")
    if (
        validity
        != {
            "manipulation_integrity_required": True,
            "manipulation_integrity_passed": True,
            "primary_efficacy_gate_passed": True,
            "overall_passed": True,
        }
        or report.get("candidate_gate", {}).get("passed") is not True
        or any(
            value.get("passed") is not True
            for value in report.get("candidate_gate", {}).get("checks", {}).values()
        )
        or report.get("manipulation_integrity_check", {}).get("passed") is not True
        or report.get("decision", {}).get("selected")
        != "front_brake_independently_confirmed_in_diagnostic_v2"
        or report.get("decision", {}).get("validation_or_test_access") != "forbidden"
        or interpretation.get("estimand")
        != "paired_intent_to_treat_whole_closed_loop_profile_effect"
    ):
        raise PermissionError("Unexpected v2 gate, validity, or interpretation result.")
    arms = [_mapping(value, "v2.arms[]") for value in _list(report.get("arms"), "v2.arms")]
    by_role = {value.get("role"): value for value in arms}
    if set(by_role) != {"reference", "candidate"}:
        raise PermissionError("v2 must contain exactly one reference and one candidate arm.")
    reference_env = dict(_mapping(by_role["reference"].get("controller_env"), "reference env"))
    candidate_env = dict(_mapping(by_role["candidate"].get("controller_env"), "candidate env"))
    if reference_env.pop("WEBOTS_SAFETY_SHIELD", None) != "0":
        raise PermissionError("v2 reference shield must be disabled.")
    if candidate_env.pop("WEBOTS_SAFETY_SHIELD", None) != "1" or candidate_env != reference_env:
        raise PermissionError("v2 arm environment must differ only by shield enablement.")
    if by_role["reference"].get("model") != by_role["candidate"].get("model") or by_role[
        "reference"
    ].get("resolved_safety_shield_config") != by_role["candidate"].get(
        "resolved_safety_shield_config"
    ):
        raise PermissionError("v2 arms must bind the same model and shield parameters.")


def _load_episode_csv(path: Path, role: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(EXPECTED_V2_SEEDS):
        raise PermissionError(f"v2 {role} episodes.csv must have 120 rows.")
    parsed: list[dict[str, Any]] = []
    expected_planner = f"ppo_recovery_v1_front_brake_v2_{role}"
    for index, row in enumerate(rows):
        seed = int(row["scenario_seed"])
        outcome = row["outcome"]
        if (
            seed != EXPECTED_V2_SEEDS[index]
            or row["split"] != "diagnostic"
            or row["condition_id"] != "DB1"
            or row["planner"] != expected_planner
            or outcome not in {"success", "collision"}
            or row["error_type"]
        ):
            raise PermissionError(f"Unexpected v2 {role} episode row {index}.")
        parsed.append(
            {
                "seed": seed,
                "outcome": outcome,
                "step_count": int(row["step_count"]),
                "completion_time_s": (
                    float(row["completion_time_s"]) if row["completion_time_s"] else None
                ),
            }
        )
    return parsed


def _load_identity_jsonl(path: Path, role: str, label: str) -> list[int]:
    seeds: list[int] = []
    expected_planner = f"ppo_recovery_v1_front_brake_v2_{role}"
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            value = _mapping(json.loads(line), f"{label}[{index}]")
            identity = _mapping(value.get("identity"), f"{label}[{index}].identity")
            seed = identity.get("scenario_seed")
            if (
                seed != EXPECTED_V2_SEEDS[index]
                or identity.get("split") != "diagnostic"
                or identity.get("condition_id") != "DB1"
                or identity.get("planner") != expected_planner
            ):
                raise PermissionError(f"Unexpected {label} identity at row {index}.")
            seeds.append(int(seed))
    if seeds != list(EXPECTED_V2_SEEDS):
        raise PermissionError(f"{label} must contain exactly the ordered v2 seeds.")
    return seeds


def _load_shield_evidence(
    path: Path,
    role: str,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    expected_enabled = role == "candidate"
    total_steps = 0
    active_steps = 0
    active_episodes = 0
    braking_steps = 0
    cap_binding_steps = 0
    angular_unchanged_steps = 0
    with path.open(encoding="utf-8") as stream:
        lines = list(stream)
    if len(lines) != len(EXPECTED_V2_SEEDS):
        raise PermissionError(f"v2 {role} shield evidence must have 120 records.")
    for index, line in enumerate(lines):
        value = _mapping(json.loads(line), f"{role}.shield[{index}]")
        identity = _mapping(value.get("identity"), f"{role}.shield[{index}].identity")
        summary = _mapping(value.get("summary"), f"{role}.shield[{index}].summary")
        epsilon = float(summary.get("intervention_epsilon"))
        if not math.isclose(epsilon, 1e-6, rel_tol=0.0, abs_tol=1e-12):
            raise PermissionError(f"Unexpected v2 {role} shield epsilon at row {index}.")
        steps = [_mapping(step, "shield step") for step in _list(value.get("steps"), "steps")]
        if (
            identity.get("scenario_seed") != EXPECTED_V2_SEEDS[index]
            or identity.get("split") != "diagnostic"
            or identity.get("condition_id") != "DB1"
            or identity.get("planner") != f"ppo_recovery_v1_front_brake_v2_{role}"
        ):
            raise PermissionError(f"Unexpected v2 {role} shield seed at row {index}.")
        if len(steps) != episodes[index]["step_count"]:
            raise PermissionError(f"v2 {role} shield/episodes step count mismatch at row {index}.")
        flags = [step.get("shield_active") for step in steps]
        if any(not isinstance(flag, bool) for flag in flags):
            raise ValueError("shield_active must be boolean on every raw step.")
        if any(step.get("shield_enabled") is not expected_enabled for step in steps):
            raise PermissionError(f"Unexpected v2 {role} shield-enabled state.")
        episode_active = sum(flags)
        for step, is_active in zip(steps, flags, strict=True):
            if not is_active:
                continue
            policy = _list(step.get("policy_action"), "active policy_action")
            shielded = _list(step.get("shielded_action"), "active shielded_action")
            if len(policy) != 2 or len(shielded) != 2:
                raise ValueError("Active actions must each contain linear and angular values.")
            linear_policy = float(policy[0])
            linear_shielded = float(shielded[0])
            normalized_cap = float(step["shield_normalized_speed_limit"])
            braking_steps += _float32(linear_policy - linear_shielded) > epsilon
            cap_binding_steps += struct.pack("<f", linear_shielded) == struct.pack(
                "<f", normalized_cap
            )
            angular_unchanged_steps += struct.pack("<f", float(policy[1])) == struct.pack(
                "<f", float(shielded[1])
            )
        total_steps += len(steps)
        active_steps += episode_active
        active_episodes += episode_active > 0
    return {
        "total_steps": total_steps,
        "active_steps": active_steps,
        "active_episodes": active_episodes,
        "braking_steps": braking_steps,
        "cap_binding_steps": cap_binding_steps,
        "angular_unchanged_steps": angular_unchanged_steps,
    }


def _exact_one_sided_mcnemar_p(rescues: int, regressions: int) -> float:
    discordant = rescues + regressions
    return sum(math.comb(discordant, value) for value in range(rescues, discordant + 1)) / (
        2**discordant
    )


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    """Match NumPy's default linear quantile without adding a runtime dependency."""

    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("A non-empty sequence and probability in [0, 1] are required.")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _derive_raw_metrics(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    candidate_shield: Mapping[str, int],
) -> dict[str, Any]:
    if [value["seed"] for value in reference] != [value["seed"] for value in candidate]:
        raise PermissionError("v2 episode arms are not seed-paired.")
    outcomes = {
        "reference": Counter(value["outcome"] for value in reference),
        "candidate": Counter(value["outcome"] for value in candidate),
    }
    transitions = Counter(
        f"{reference_value['outcome']}_to_{candidate_value['outcome']}"
        for reference_value, candidate_value in zip(reference, candidate, strict=True)
    )
    ratios = [
        float(candidate_value["completion_time_s"]) / float(reference_value["completion_time_s"])
        for reference_value, candidate_value in zip(reference, candidate, strict=True)
        if reference_value["outcome"] == candidate_value["outcome"] == "success"
    ]
    rescues = transitions["collision_to_success"]
    regressions = transitions["success_to_collision"]
    reference_collisions = outcomes["reference"]["collision"]
    candidate_collisions = outcomes["candidate"]["collision"]
    return {
        "outcomes": outcomes,
        "transitions": transitions,
        "rescues": rescues,
        "regressions": regressions,
        "collision_relative_reduction": (reference_collisions - candidate_collisions)
        / reference_collisions,
        "mcnemar_p": _exact_one_sided_mcnemar_p(rescues, regressions),
        "common_success": len(ratios),
        "median_completion_ratio": statistics.median(ratios),
        "p90_completion_ratio": _linear_quantile(ratios, 0.90),
        "candidate_shield": dict(candidate_shield),
    }


def _crosscheck_report_metrics(report: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    arms = {value["role"]: value for value in report["arms"]}
    for role in ("reference", "candidate"):
        reported = arms[role]["metrics"]["overall"]
        counts = metrics["outcomes"][role]
        if reported != {
            "episodes": 120,
            "success_count": counts["success"],
            "collision_count": counts["collision"],
            "timeout_count": 0,
            "error_count": 0,
        }:
            raise PermissionError(f"v2 {role} report/raw outcome mismatch.")
    overall = report["paired_analysis"]["overall"]
    expected_transitions = {
        "collision_to_collision": metrics["transitions"]["collision_to_collision"],
        "collision_to_success": metrics["transitions"]["collision_to_success"],
        "success_to_collision": metrics["transitions"]["success_to_collision"],
        "success_to_success": metrics["transitions"]["success_to_success"],
    }
    if overall["outcome_transitions"] != expected_transitions:
        raise PermissionError("v2 paired transition report/raw mismatch.")
    if (
        not math.isclose(
            overall["collision"]["relative_reduction"],
            metrics["collision_relative_reduction"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or overall["paired_success_endpoint"]["rescues_reference_non_success_to_candidate_success"]
        != metrics["rescues"]
        or overall["paired_success_endpoint"][
            "regressions_reference_success_to_candidate_non_success"
        ]
        != metrics["regressions"]
        or not math.isclose(
            overall["paired_success_endpoint"]["exact_one_sided_mcnemar_p"],
            metrics["mcnemar_p"],
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        or overall["completion_delay_on_reference_success"]["paired_success_count"]
        != metrics["common_success"]
        or not math.isclose(
            overall["completion_delay_on_reference_success"]["median_ratio"],
            metrics["median_completion_ratio"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            overall["completion_delay_on_reference_success"]["ratio_p90"],
            metrics["p90_completion_ratio"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise PermissionError("v2 paired report/raw metric mismatch.")
    shield = arms["candidate"]["shield_evidence_metrics"]
    raw_shield = metrics["candidate_shield"]
    if (
        shield["active_step_count"] != raw_shield["active_steps"]
        or shield["total_step_count"] != raw_shield["total_steps"]
        or shield["episode_with_active_step_count"] != raw_shield["active_episodes"]
        or shield["braking_step_count"] != raw_shield["braking_steps"]
        or shield["speed_cap_binding_step_count"] != raw_shield["cap_binding_steps"]
        or shield["active_operational_semantics_step_count"] != raw_shield["active_steps"]
        or shield["active_angular_action_float32_bits_unchanged_step_count"]
        != raw_shield["angular_unchanged_steps"]
    ):
        raise PermissionError("v2 shield report/raw mismatch.")


def load_locked_inputs(
    *,
    repository_root: Path,
    v1_report_path: Path,
    audit_report_path: Path,
    v2_report_path: Path,
    generator_path: Path,
) -> dict[str, Any]:
    """Verify all locked reports/raw files and return raw-derived metrics."""

    repository_root = repository_root.resolve()
    v1, v1_hashes = _verify_trilogy(
        v1_report_path,
        locked_files=LOCKED_V1_FILES,
        label="sealed v1 report",
    )
    audit, audit_hashes = _verify_trilogy(
        audit_report_path,
        locked_files=LOCKED_AUDIT_FILES,
        label="manipulation-boundary audit",
    )
    v2, v2_hashes = _verify_trilogy(
        v2_report_path,
        locked_files=LOCKED_V2_FILES,
        label="sealed v2 report",
    )
    _validate_v1(v1)
    _validate_audit(audit)
    _validate_v2(v2)
    arms = {value["role"]: value for value in v2["arms"]}
    raw_paths: dict[str, dict[str, Path]] = {"reference": {}, "candidate": {}}
    raw_hashes: dict[str, str] = {}
    for role, locked in LOCKED_V2_RAW_FILES.items():
        artifacts = _mapping(arms[role].get("artifacts"), f"v2 {role} artifacts")
        for artifact_key, (expected_path, expected_hash) in locked.items():
            binding = _mapping(artifacts.get(artifact_key), f"v2 {role}.{artifact_key}")
            if binding != {"path": expected_path, "sha256": expected_hash}:
                raise PermissionError(f"v2 {role} {artifact_key} binding drifted.")
            path = _relative_artifact_path(repository_root, expected_path, artifact_key)
            _require_file_sha(path, expected_hash, f"v2 {role} {artifact_key}")
            raw_paths[role][artifact_key] = path
            raw_hashes[expected_path] = expected_hash

    before_paths = [
        *(v1_report_path.parent / name for name in LOCKED_V1_FILES),
        *(audit_report_path.parent / name for name in LOCKED_AUDIT_FILES),
        *(v2_report_path.parent / name for name in LOCKED_V2_FILES),
        *(path for values in raw_paths.values() for path in values.values()),
        generator_path,
    ]
    before = {
        _relative_or_absolute(repository_root, path): _sha256_file(path) for path in before_paths
    }
    generator_hash = before[_relative_or_absolute(repository_root, generator_path)]

    episode_rows = {
        role: _load_episode_csv(raw_paths[role]["episodes_csv"], role)
        for role in ("reference", "candidate")
    }
    for role in ("reference", "candidate"):
        summary = _mapping(
            _load_json(raw_paths[role]["summary_json"], f"v2 {role} summary"),
            f"v2 {role} summary",
        )
        counts = Counter(value["outcome"] for value in episode_rows[role])
        by_planner = _list(summary.get("by_planner"), f"v2 {role} summary.by_planner")
        if (
            summary.get("split") != "diagnostic"
            or summary.get("episode_count") != 120
            or len(by_planner) != 1
            or by_planner[0].get("success_count") != counts["success"]
            or by_planner[0].get("collision_count") != counts["collision"]
            or by_planner[0].get("timeout_count") != 0
            or by_planner[0].get("error_count") != 0
        ):
            raise PermissionError(f"v2 {role} summary/episodes mismatch.")
        _load_identity_jsonl(
            raw_paths[role]["episode_context_jsonl"], role, f"v2 {role} episode context"
        )
    shield_metrics = {
        role: _load_shield_evidence(
            raw_paths[role]["shield_evidence_jsonl"], role, episode_rows[role]
        )
        for role in ("reference", "candidate")
    }
    if shield_metrics["reference"] != {
        "total_steps": 9523,
        "active_steps": 0,
        "active_episodes": 0,
        "braking_steps": 0,
        "cap_binding_steps": 0,
        "angular_unchanged_steps": 0,
    }:
        raise PermissionError("Unexpected v2 reference shield evidence.")
    metrics = _derive_raw_metrics(
        episode_rows["reference"], episode_rows["candidate"], shield_metrics["candidate"]
    )
    _crosscheck_report_metrics(v2, metrics)
    expected_metrics = {
        "reference_success": 66,
        "reference_collision": 54,
        "candidate_success": 95,
        "candidate_collision": 25,
        "rescues": 35,
        "regressions": 6,
        "common_success": 60,
        "active_steps": 3740,
        "total_steps": 12741,
        "braking_steps": 3740,
        "cap_binding_steps": 3740,
        "angular_unchanged_steps": 3740,
    }
    actual_metrics = {
        "reference_success": metrics["outcomes"]["reference"]["success"],
        "reference_collision": metrics["outcomes"]["reference"]["collision"],
        "candidate_success": metrics["outcomes"]["candidate"]["success"],
        "candidate_collision": metrics["outcomes"]["candidate"]["collision"],
        "rescues": metrics["rescues"],
        "regressions": metrics["regressions"],
        "common_success": metrics["common_success"],
        "active_steps": metrics["candidate_shield"]["active_steps"],
        "total_steps": metrics["candidate_shield"]["total_steps"],
        "braking_steps": metrics["candidate_shield"]["braking_steps"],
        "cap_binding_steps": metrics["candidate_shield"]["cap_binding_steps"],
        "angular_unchanged_steps": metrics["candidate_shield"]["angular_unchanged_steps"],
    }
    if actual_metrics != expected_metrics:
        raise PermissionError(f"Unexpected locked v2 raw metrics: {actual_metrics}")
    if (
        not math.isclose(metrics["collision_relative_reduction"], 29 / 54, abs_tol=1e-15)
        or not math.isclose(metrics["mcnemar_p"], 2.436799150018487e-06, abs_tol=1e-18)
        or not math.isclose(metrics["median_completion_ratio"], 1.072172619047619, abs_tol=1e-15)
        or not math.isclose(metrics["p90_completion_ratio"], 1.1896752519596865, abs_tol=1e-15)
    ):
        raise PermissionError("Unexpected locked v2 raw-derived statistical values.")
    after = {
        _relative_or_absolute(repository_root, path): _sha256_file(path) for path in before_paths
    }
    if after != before:
        raise RuntimeError("A locked input changed while it was being verified.")
    all_hashes = {
        **{
            str((v1_report_path.parent / name).relative_to(repository_root)).replace(
                "\\", "/"
            ): digest
            for name, digest in v1_hashes.items()
        },
        **{
            str((audit_report_path.parent / name).relative_to(repository_root)).replace(
                "\\", "/"
            ): digest
            for name, digest in audit_hashes.items()
        },
        **{
            str((v2_report_path.parent / name).relative_to(repository_root)).replace(
                "\\", "/"
            ): digest
            for name, digest in v2_hashes.items()
        },
        **raw_hashes,
    }
    return {
        "v1": v1,
        "audit": audit,
        "v2": v2,
        "metrics": metrics,
        "hashes": all_hashes,
        "input_snapshot": before,
        "input_paths": before_paths,
        "generator_path": generator_path,
        "generator_sha256": generator_hash,
    }


def _relative_or_absolute(repository_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _svg_document(*, width: int, height: int, title: str, description: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"  <title>{escape(title)}</title>\n"
        f"  <desc>{escape(description)}</desc>\n"
        f'  <rect width="{width}" height="{height}" fill="#f8fafc"/>\n'
        f"{body}\n</svg>\n"
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#172033",
) -> str:
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{escape(value)}</text>'
    )


def render_outcome_rates(metrics: Mapping[str, Any]) -> str:
    width, height = 960, 560
    left, top, plot_height = 110.0, 125.0, 310.0
    colors = {"success": "#167d64", "collision": "#c94b55"}
    lines = [
        _text(48, 48, "Independent diagnostic confirmation: outcomes", size=27, weight=700),
        _text(48, 79, "v2 seeds 54775-54894 · 120 paired episodes", fill="#526078"),
    ]
    for tick in range(0, 121, 20):
        y = top + plot_height - tick / 120 * plot_height
        lines.append(
            f'  <line x1="{left:.1f}" y1="{y:.1f}" x2="900.0" y2="{y:.1f}" '
            'stroke="#d9e0ea" stroke-width="1"/>'
        )
        lines.append(_text(left - 14, y + 5, str(tick), size=13, anchor="end", fill="#526078"))
    centers = {"reference": 330.0, "candidate": 700.0}
    for role in ("reference", "candidate"):
        for index, outcome in enumerate(("success", "collision")):
            value = metrics["outcomes"][role][outcome]
            x = centers[role] + (-75 if index == 0 else 35)
            bar_height = value / 120 * plot_height
            y = top + plot_height - bar_height
            lines.append(
                f'  <rect x="{x:.1f}" y="{y:.1f}" width="80" height="{bar_height:.1f}" '
                f'rx="6" fill="{colors[outcome]}"/>'
            )
            lines.append(
                _text(
                    x + 40,
                    y - 10,
                    f"{value}/120 ({value / 120:.2%})",
                    size=14,
                    weight=700,
                    anchor="middle",
                )
            )
            lines.append(
                _text(x + 40, 462, outcome.title(), size=13, anchor="middle", fill="#526078")
            )
        lines.append(_text(centers[role], 510, role.title(), size=19, weight=700, anchor="middle"))
    lines.append(
        _text(
            480,
            548,
            "Collision reduction: 54 -> 25 (-53.70%)",
            size=16,
            weight=700,
            anchor="middle",
            fill="#9b3340",
        )
    )
    return _svg_document(
        width=width,
        height=height,
        title="v2 diagnostic outcome rates",
        description=(
            "Reference 66 successes and 54 collisions; candidate 95 successes and 25 collisions."
        ),
        body="\n".join(lines),
    )


def render_paired_transitions(metrics: Mapping[str, Any]) -> str:
    width, height = 980, 550
    values = [
        ("Collision -> collision", metrics["transitions"]["collision_to_collision"], "#8c98aa"),
        ("Rescue: collision -> success", metrics["rescues"], "#167d64"),
        ("Regression: success -> collision", metrics["regressions"], "#c94b55"),
        ("Success -> success", metrics["transitions"]["success_to_success"], "#2878b5"),
    ]
    lines = [
        _text(48, 48, "Paired outcome transitions", size=28, weight=700),
        _text(48, 79, "Every v2 scenario seed is compared arm-to-arm", fill="#526078"),
    ]
    max_value = 60
    for index, (label, value, color) in enumerate(values):
        y = 125 + index * 82
        width_value = value / max_value * 610
        lines.append(_text(48, y + 25, label, size=15, weight=600))
        lines.append(f'  <rect x="320" y="{y}" width="610" height="34" rx="6" fill="#e5eaf1"/>')
        lines.append(
            f'  <rect x="320" y="{y}" width="{width_value:.1f}" height="34" rx="6" fill="{color}"/>'
        )
        lines.append(
            _text(
                320 + width_value - 10 if value > 10 else 320 + width_value + 10,
                y + 24,
                str(value),
                size=16,
                weight=700,
                anchor="end" if value > 10 else "start",
                fill="#ffffff" if value > 10 else "#172033",
            )
        )
    lines.extend(
        [
            _text(48, 482, "Rescues 35 · regressions 6 · discordant pairs 41", size=17, weight=700),
            _text(
                48,
                515,
                "Exact one-sided McNemar p = 2.4368e-6",
                size=17,
                weight=700,
                fill="#167d64",
            ),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title="v2 paired outcome transitions",
        description=(
            "19 collision-to-collision, 35 rescues, 6 regressions, and 60 success-to-success pairs."
        ),
        body="\n".join(lines),
    )


def render_safety_tradeoff(metrics: Mapping[str, Any]) -> str:
    width, height = 1000, 500
    active = metrics["candidate_shield"]["active_steps"]
    total = metrics["candidate_shield"]["total_steps"]
    active_ratio = active / total
    cards = [
        (
            "Collision reduction",
            f"{metrics['collision_relative_reduction']:.2%}",
            "54 -> 25",
            "#167d64",
        ),
        ("Common successes", str(metrics["common_success"]), "paired completion set", "#2878b5"),
        (
            "Median time ratio",
            f"{metrics['median_completion_ratio']:.4f}",
            f"p90 {metrics['p90_completion_ratio']:.4f} (+18.97%)",
            "#b46b18",
        ),
        (
            "Shield-active steps",
            f"{active}/{total}",
            f"{active_ratio:.2%} of candidate steps",
            "#7847a8",
        ),
    ]
    lines = [
        _text(48, 48, "Safety effect and completion trade-off", size=28, weight=700),
        _text(
            48, 80, "Raw-derived v2 diagnostic metrics · no validation/test claim", fill="#526078"
        ),
    ]
    for index, (title, value, subtitle, color) in enumerate(cards):
        x = 48 + index * 238
        lines.append(
            f'  <rect x="{x}" y="125" width="214" height="250" rx="14" '
            'fill="#ffffff" stroke="#d9e0ea"/>'
        )
        lines.append(f'  <rect x="{x}" y="125" width="214" height="10" rx="5" fill="{color}"/>')
        lines.append(_text(x + 18, 177, title, size=15, weight=700, fill="#526078"))
        lines.append(_text(x + 107, 254, value, size=28, weight=700, anchor="middle", fill=color))
        lines.append(_text(x + 107, 310, subtitle, size=13, anchor="middle", fill="#526078"))
    lines.extend(
        [
            _text(
                500,
                430,
                "Manipulation integrity passed in v2: 3740 active = braking = cap-binding steps",
                size=16,
                weight=700,
                anchor="middle",
            ),
            _text(
                500,
                465,
                "Diagnostic confirmation only; not validation, held-out test, "
                "or generalization evidence",
                size=15,
                anchor="middle",
                fill="#9b3340",
            ),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title="v2 safety effect and trade-off",
        description=(
            "53.70 percent collision reduction, 60 common successes, median "
            "completion ratio 1.0722, p90 ratio 1.1897, and 3740 active shield "
            "steps out of 12741."
        ),
        body="\n".join(lines),
    )


def _relative_link(output_directory: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), output_directory.resolve())).as_posix()


def render_markdown(
    *,
    loaded: Mapping[str, Any],
    repository_root: Path,
    output_directory: Path,
    v1_report_path: Path,
    audit_report_path: Path,
    v2_report_path: Path,
) -> str:
    metrics = loaded["metrics"]
    hashes = loaded["hashes"]
    active = metrics["candidate_shield"]["active_steps"]
    total = metrics["candidate_shield"]["total_steps"]
    links = {
        "v1": _relative_link(output_directory, v1_report_path),
        "audit": _relative_link(output_directory, audit_report_path),
        "v2": _relative_link(output_directory, v2_report_path),
    }
    hash_rows = "\n".join(f"| `{path}` | `{digest}` |" for path, digest in sorted(hashes.items()))
    generator_relative = _relative_or_absolute(repository_root, loaded["generator_path"])
    return f"""# Webots 前向制动：v1 审计到 v2 独立确认（公开版）

> **结论边界：**密封 v1 结果仍是 **`invalid_manipulation`**，离线审计只定位并修正了 checker
> 对一个 float32 边界步的判定方式，**没有改写 v1、也没有把 v1 的效果结果升级为有效结论**。
> 随后在全新且不重叠的 `54775～54894` 种子上完成 v2；v2 的预注册门槛全部通过，因此得到
> **独立 diagnostic 确认**。这不是 validation、held-out test 或跨场景泛化结论，也没有合并 v1/v2。

## 1. 为什么需要 v2

v1 使用 `25728～25847` 的 120 个配对回合，主效果门槛本身通过，但完整报告因 manipulation
完整性检查失败而封存为 `invalid_manipulation`。后验审计发现唯一差异来自 seed `25737`、step `21`：
实际送入控制链的 float32 动作变化为 `1.013278961e-6`，超过 `1e-6` 介入阈值并与速度上限的
float32 表示一致；旧 checker 却再次用 double 精度上限计算出 `9.985105547e-7`，漏计了这一个
cap-binding 步。因此审计把 v1 的 `3197/3197/3196` 解释为 checker 数值边界，而不是执行动作偏离。

审计的权限仅到“允许开展一次新的独立 diagnostic 确认”。v1 报告状态保持不变；v2 明确禁止
重跑 v1、重写或重新解释 v1、参数扫描、自适应扩样，以及 v1/v2 pooling fallback。

证据入口：[`v1 paired report`]({links["v1"]}) →
[`offline boundary audit`]({links["audit"]}) →
[`independent v2 report`]({links["v2"]})。

## 2. v2 结果：独立 diagnostic 确认通过

图表和下列数字均由 v2 两臂的原始 `episodes.csv` 重新计算，并与密封 JSON 报告逐项核对：

![v2 两臂结果](./outcome_rates.svg)

- reference：成功 **66/120（55.00%）**，碰撞 **54/120（45.00%）**；
- front-brake candidate：成功 **95/120（79.17%）**，碰撞 **25/120（20.83%）**；
- 碰撞数相对下降 **{metrics["collision_relative_reduction"]:.2%}**（54 → 25）；
  两臂均无 timeout 或 error。

## 3. 配对证据

![v2 配对转移](./paired_transitions.svg)

120 个相同 scenario seed 的结果转移为：19 个 collision→collision、**35 个 rescue**、
**6 个 regression**、60 个 success→success。成功端点的精确单侧 McNemar
`p = {metrics["mcnemar_p"]:.10g}`（约 `2.4368e-6`），通过预注册的 `< 0.05` 门槛。
该检验只回答这组新 diagnostic 场景中候选闭环 profile 是否优于 reference，不外推到未运行分布。

## 4. 安全收益与完成时间代价

![安全收益与完成时间代价](./safety_tradeoff.svg)

- 共同成功样本为 **{metrics["common_success"]}** 个；逐 seed 候选/reference 完成时间比的中位数为
  **{metrics["median_completion_ratio"]:.4f}**，即共同成功回合的典型比例约增加 7.22%，仍低于
  预注册上限 1.15；第 90 百分位为 **{metrics["p90_completion_ratio"]:.4f}**（约增加 18.97%）。
  p90 只作尾部代价的描述性披露，不属于、也不改变预注册的 median 门槛；
- candidate 的原始 shield sidecar 共记录 **{active}/{total}** 个 active steps
  （{active / total:.2%}）；v2 中 active、braking、speed-cap-binding 均为 {active}，
  角速度动作保持不变；
- v2 manipulation integrity、主效果 gate 和 overall gate 均为 `passed`。这里报告的是
  `front-brake enabled` 整个闭环 profile 的 intent-to-treat 效果，不把相关性表述为单步因果机制。

## 5. 严格边界

- **v1 仍无效：**离线审计不能追认 v1 主效果，也没有修改其 JSON/YAML/sidecar；
- **v2 独立而不合并：**v2 种子与 v1 不重叠，唯一主推断来自 v2 overall，未做 pooled fallback；
- **只属 diagnostic：**validation 与 held-out test 均未访问；这不能证明跨地图、跨障碍数量、
  真实机器人
  或其他模型上的泛化；
- **不用于 checkpoint 选择：**本报告没有训练、筛选或替换模型，也没有运行 Webots。

## 6. 可审计来源

生成器首先验证三组密封 JSON/YAML/`.sha256`，再验证 v2 两臂各四个 raw artifacts 的路径和
SHA-256；随后从 raw CSV/JSONL 重算结果、配对转移、完成时间比例与 shield-active 计数。任一输入
字节、路径、范围或诊断边界变化都会拒绝生成。

| 输入 | SHA-256 |
|---|---|
{hash_rows}

生成脚本：`{generator_relative}`
脚本 SHA-256：`{loaded["generator_sha256"]}`

## 7. 离线复现

以下命令只刷新这个已知的派生展示目录；它不会运行 Webots，也不会修改上述输入：

```bash
python scripts/generate_webots_front_brake_confirmation_report.py --force
```

默认不允许覆盖已有目录；`--force` 也只接受恰好包含本报告五个已知派生文件的目录。输出
`confirmation_report.sha256` 同时绑定 Markdown、3 张 SVG、生成脚本和全部直接输入。
"""


def _build_sidecar(
    *,
    primary_outputs: Mapping[str, str],
    loaded: Mapping[str, Any],
    repository_root: Path,
) -> str:
    lines = [f"{_sha256_text(content)}  {name}" for name, content in primary_outputs.items()]
    generator_relative = _relative_or_absolute(repository_root, loaded["generator_path"])
    lines.append(f"{loaded['generator_sha256']}  {generator_relative}")
    lines.extend(f"{digest}  {path}" for path, digest in sorted(loaded["hashes"].items()))
    return "\n".join(lines) + "\n"


def _publish_outputs(
    *,
    output_directory: Path,
    outputs: Mapping[str, str],
    force: bool,
    post_publish_check: Callable[[], None],
) -> None:
    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.parent / f".{output_directory.name}.staging"
    backup = output_directory.parent / f".{output_directory.name}.backup"
    if staging.exists() or backup.exists():
        raise FileExistsError("A report staging/backup directory already exists; inspect it first.")
    if output_directory.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_directory}")
        if not output_directory.is_dir() or output_directory.is_symlink():
            raise PermissionError("--force only accepts a regular derived report directory.")
        children = list(output_directory.iterdir())
        if {child.name for child in children} != set(OUTPUT_FILENAMES) or any(
            not child.is_file() or child.is_symlink() for child in children
        ):
            raise PermissionError(
                "--force refused: output directory is not the exact known manifest."
            )
    try:
        staging.mkdir()
        for name, content in outputs.items():
            (staging / name).write_text(content, encoding="utf-8", newline="\n")
        if {path.name for path in staging.iterdir()} != set(OUTPUT_FILENAMES):
            raise RuntimeError("Staging manifest mismatch.")
        if output_directory.exists():
            output_directory.replace(backup)
            try:
                staging.replace(output_directory)
                post_publish_check()
            except Exception:
                if output_directory.exists():
                    shutil.rmtree(output_directory)
                backup.replace(output_directory)
                raise
            shutil.rmtree(backup)
        else:
            staging.replace(output_directory)
            try:
                post_publish_check()
            except Exception:
                shutil.rmtree(output_directory)
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def generate_report(
    *,
    repository_root: Path,
    v1_report_path: Path,
    audit_report_path: Path,
    v2_report_path: Path,
    output_directory: Path,
    generator_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Verify immutable evidence and atomically publish the derived report."""

    generator_path = (
        Path(__file__).resolve() if generator_path is None else generator_path.resolve()
    )
    loaded = load_locked_inputs(
        repository_root=repository_root,
        v1_report_path=v1_report_path,
        audit_report_path=audit_report_path,
        v2_report_path=v2_report_path,
        generator_path=generator_path,
    )
    primary_outputs = {
        "confirmation_report.md": render_markdown(
            loaded=loaded,
            repository_root=repository_root,
            output_directory=output_directory,
            v1_report_path=v1_report_path,
            audit_report_path=audit_report_path,
            v2_report_path=v2_report_path,
        ),
        "outcome_rates.svg": render_outcome_rates(loaded["metrics"]),
        "paired_transitions.svg": render_paired_transitions(loaded["metrics"]),
        "safety_tradeoff.svg": render_safety_tradeoff(loaded["metrics"]),
    }
    outputs = {
        **primary_outputs,
        "confirmation_report.sha256": _build_sidecar(
            primary_outputs=primary_outputs,
            loaded=loaded,
            repository_root=repository_root,
        ),
    }
    if tuple(outputs) != OUTPUT_FILENAMES:
        raise RuntimeError("Internal output manifest mismatch.")
    current = {
        _relative_or_absolute(repository_root, path): _sha256_file(path)
        for path in loaded["input_paths"]
    }
    if current != loaded["input_snapshot"]:
        raise RuntimeError("A locked input changed before report publication.")

    def require_inputs_unchanged() -> None:
        if {
            _relative_or_absolute(repository_root, path): _sha256_file(path)
            for path in loaded["input_paths"]
        } != loaded["input_snapshot"]:
            raise RuntimeError("A locked input changed during report publication.")

    _publish_outputs(
        output_directory=output_directory,
        outputs=outputs,
        force=force,
        post_publish_check=require_inputs_unchanged,
    )
    return {
        "output_directory": output_directory,
        "files": {name: _sha256_file(output_directory / name) for name in outputs},
        "v1_status": "invalid_manipulation",
        "audit_promotes_v1": False,
        "v2_status": "passed",
        "split": "diagnostic",
        "validation_or_test_access": "forbidden",
        "pooling": "forbidden",
        "generator_sha256": loaded["generator_sha256"],
        "input_snapshot": loaded["input_snapshot"],
        "metrics": loaded["metrics"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-report",
        type=Path,
        default=(
            repository_root
            / "artifacts/results/webots_front_brake_paired_diagnostic_v1"
            / "paired_diagnostic_report.json"
        ),
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=(
            repository_root
            / "artifacts/results/webots_front_brake_v1_manipulation_reaudit_v1"
            / "manipulation_boundary_reaudit.json"
        ),
    )
    parser.add_argument(
        "--v2-report",
        type=Path,
        default=(
            repository_root
            / "artifacts/results/webots_front_brake_paired_confirmation_v2"
            / "paired_diagnostic_report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "reports/webots_front_brake_confirmation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh only an existing directory with the exact known derived-file manifest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    try:
        summary = generate_report(
            repository_root=repository_root,
            v1_report_path=args.v1_report,
            audit_report_path=args.audit_report,
            v2_report_path=args.v2_report,
            output_directory=args.output_dir,
            force=args.force,
        )
    except Exception as error:
        print(f"WEBOTS_FRONT_BRAKE_REPORT_FAILED: {type(error).__name__}: {error}")
        return 2
    print(
        "WEBOTS_FRONT_BRAKE_REPORT_OK "
        f"v1={summary['v1_status']} v2={summary['v2_status']} "
        f"split={summary['split']} output={summary['output_directory']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
