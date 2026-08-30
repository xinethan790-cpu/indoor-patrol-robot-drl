#!/usr/bin/env python3
"""Offline numerical-boundary re-audit of the sealed front-brake diagnostic.

The original paired report and both arm directories are immutable inputs.  This
script does not launch Webots and can only create a new, previously absent audit
directory.  Its conclusion is deliberately limited to eligibility for a fresh
independent *diagnostic* confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

AUDIT_ID = "webots_front_brake_v1_manipulation_boundary_reaudit_v1"
ORIGINAL_REPORT_SHA256 = "2779e8dcb6f8ccf2216464e196dc5c5f3a5ba1c8092eb0d75cb14fb62e2cc787"
ORIGINAL_GIT_REVISION = "f957181777709cd8e064f213df1a718eb515ce84"
MODEL_SHA256 = "c5e7c3de42c7664477097b2734975660fec840ad5b7f45425098158c27cb7275"
MODEL_PATH = "artifacts/models/webots/seed_11/w1_recovery_v1/ppo_webots_s11_step000040960.zip"
EXPECTED_SEEDS = tuple(range(25728, 25848))
BOUNDARY_SEED = 25737
BOUNDARY_STEP = 21
ABSOLUTE_TOLERANCE = 1e-6

ORIGINAL_REPORT_ARTIFACTS = {
    "json": (
        "paired_diagnostic_report.json",
        ORIGINAL_REPORT_SHA256,
    ),
    "yaml": (
        "paired_diagnostic_report.yaml",
        "db161fa425e25255381b60d99aece212b072fdbfa87061532e4f2a3f9289ff75",
    ),
    "sha256": (
        "paired_diagnostic_report.sha256",
        "b255ab30a0bb0214dbaa4dd7a1b52eab02dcac441683eb6d4f3ef071a17b1427",
    ),
}

SOURCE_BINDINGS = {
    "config": (
        "configs/eval/webots_front_brake_paired_diagnostic_v1.yaml",
        "cf3f0b27877db0fe107f5dfe18ba3127c9efb90ebd1909f8735d7db952aa66e3",
    ),
    "runner": (
        "scripts/run_webots_front_brake_diagnostic.py",
        "7c886fef3c021c2afe890e7891cc6ee2e26be7600fd6dddf8e9078af796c87fd",
    ),
    "controller": (
        "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py",
        "92d6fc76e7d143e0d5ef1a4b5798274656d991fe154052c05c05ded64e60dbea",
    ),
    "world": (
        "worlds/webots_ppo_evaluate.wbt",
        "e0ae7ac8aa2a7981f94673d656d76940166cb12ceab8dd2b17a17329985d5425",
    ),
    "safety_shield": (
        "src/indoor_patrol_rl/core/safety_shield.py",
        "0affe799301c11a43b5d67a9d5eb37efe89ec54406c5fbd08be80f4853b3b50b",
    ),
    "control": (
        "src/indoor_patrol_rl/core/control.py",
        "5e95838266364c33894992f425f9e653a5ebe370c19bc73ac4cdd6852cf8e04a",
    ),
    "webots_patrol_environment": (
        "src/indoor_patrol_rl/envs/webots_patrol.py",
        "9c1a17eb436c3a3591a795b0dde843a4c6b8f2f5a4a44b2b76cfd2ed5c9c413a",
    ),
    "evaluation_exports": (
        "src/indoor_patrol_rl/evaluation/__init__.py",
        "181825c68b116edf4e525635073de332deb93bbcb761c4c64e29f3af014897f9",
    ),
    "evaluation_metrics": (
        "src/indoor_patrol_rl/evaluation/metrics.py",
        "355691402123ff9a5dbd0731c0e1b41aa3db81c5ac4d00c8ab9d2a0a676306e2",
    ),
    "webots_formal_config": (
        "src/indoor_patrol_rl/evaluation/webots_formal_config.py",
        "dea197c3a33ad7debb1b20a2a02ef8c57cd962faf83ce9cb090e47b688485af0",
    ),
    "episode_context": (
        "src/indoor_patrol_rl/evaluation/episode_context.py",
        "26562c242869aa3cf39f9438d8c170411d5d40605c3813fa702dfc4b682537c0",
    ),
    "shield_evidence": (
        "src/indoor_patrol_rl/evaluation/shield_evidence.py",
        "09f6400c653915a5966ec81fc3225e6c94762af0fb3cea202e0e62134d518e1b",
    ),
    "webots_curriculum": (
        "src/indoor_patrol_rl/training/webots_curriculum.py",
        "657b6f3843d4905bfce83fa640927f368c508702ef9e96bc487be0afa58d9f35",
    ),
}

ARM_ARTIFACTS = {
    "reference": {
        "summary_json": (
            "artifacts/results/webots_diagnostic_front_brake_v1_reference_seed11/summary.json",
            "2e6a078ae52ceb4da1424949266da6b10311a6a6eee5ad22eaecfd6499b69320",
        ),
        "episodes_csv": (
            "artifacts/results/webots_diagnostic_front_brake_v1_reference_seed11/episodes.csv",
            "4b8556ddc63717f7a10f133ca7f37a5f1c81d763cb157bc89411dd7a982ed2ec",
        ),
        "episode_context_jsonl": (
            "artifacts/results/webots_diagnostic_front_brake_v1_reference_seed11/"
            "episode_context.jsonl",
            "80ea387481260271bb0c5e437f4846485d4936650e030626a7ee628d0581c976",
        ),
        "shield_evidence_jsonl": (
            "artifacts/results/webots_diagnostic_front_brake_v1_reference_seed11/"
            "shield_evidence.jsonl",
            "bb714f144545eaf32d0d58e3e55ca17ed8a5a9bf5c3a31f69b6f3b8af8bd0a57",
        ),
    },
    "candidate": {
        "summary_json": (
            "artifacts/results/webots_diagnostic_front_brake_v1_candidate_seed11/summary.json",
            "90189b83a2aec57e4f97f0cf578eb58ab614618741fb07e3fc48bf1863a2842c",
        ),
        "episodes_csv": (
            "artifacts/results/webots_diagnostic_front_brake_v1_candidate_seed11/episodes.csv",
            "a11af45af4e7f108f97344a582d2185a4e0e59c52a5a084c018520aa44b6e95d",
        ),
        "episode_context_jsonl": (
            "artifacts/results/webots_diagnostic_front_brake_v1_candidate_seed11/"
            "episode_context.jsonl",
            "cd64d754fbdccd6419dc6a13d30242238a7fdcba64404c6b5e92754e65f64c3b",
        ),
        "shield_evidence_jsonl": (
            "artifacts/results/webots_diagnostic_front_brake_v1_candidate_seed11/"
            "shield_evidence.jsonl",
            "9ab05fb389221ad0b8a41a00e667169ddc7d2b8902665ed4609c86e8d45d955d",
        ),
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object.")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Locked {label} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise PermissionError(
            f"Locked {label} SHA-256 mismatch: expected {expected_sha256}, actual {actual}."
        )


def _verify_source_blobs(repository_root: Path) -> dict[str, Any]:
    """Verify that every locked source hash is present in the original commit."""

    commit_type = subprocess.run(
        ["git", "-C", str(repository_root), "cat-file", "-t", ORIGINAL_GIT_REVISION],
        check=False,
        capture_output=True,
    )
    if commit_type.returncode != 0 or commit_type.stdout.strip() != b"commit":
        detail = commit_type.stderr.decode("utf-8", errors="replace").strip()
        raise PermissionError(
            f"Locked original Git revision is unavailable or is not a commit: {detail}"
        )

    verified: dict[str, Any] = {}
    for name, (portable_path, expected_sha256) in SOURCE_BINDINGS.items():
        blob = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "cat-file",
                "blob",
                f"{ORIGINAL_GIT_REVISION}:{portable_path}",
            ],
            check=False,
            capture_output=True,
        )
        if blob.returncode != 0:
            detail = blob.stderr.decode("utf-8", errors="replace").strip()
            raise PermissionError(
                f"Locked source blob {name!r} is unavailable at the original revision: {detail}"
            )
        actual_sha256 = hashlib.sha256(blob.stdout).hexdigest()
        if actual_sha256 != expected_sha256:
            raise PermissionError(
                f"Locked source blob {name!r} SHA-256 mismatch: "
                f"expected {expected_sha256}, actual {actual_sha256}."
            )
        verified[name] = {
            "path": portable_path,
            "sha256": expected_sha256,
            "verified_from_git_blob": True,
        }
    return verified


def _binding(value: Any, expected: tuple[str, str], label: str) -> None:
    item = _mapping(value, label)
    if set(item) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256.")
    if item != {"path": expected[0], "sha256": expected[1]}:
        raise PermissionError(f"{label} is not the locked original binding.")


def _load_original_report(path: Path, repository_root: Path) -> dict[str, Any]:
    for name, (filename, expected_sha256) in ORIGINAL_REPORT_ARTIFACTS.items():
        _verify_file(path.parent / filename, expected_sha256, f"paired diagnostic report {name}")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "paired_report")
    report_yaml = yaml.safe_load(
        (path.parent / ORIGINAL_REPORT_ARTIFACTS["yaml"][0]).read_text(encoding="utf-8")
    )
    if report_yaml != report:
        raise PermissionError("The locked paired-report JSON and YAML do not agree.")
    expected_sidecar = (
        f"{ORIGINAL_REPORT_ARTIFACTS['json'][1]}  {ORIGINAL_REPORT_ARTIFACTS['json'][0]}\n"
        f"{ORIGINAL_REPORT_ARTIFACTS['yaml'][1]}  {ORIGINAL_REPORT_ARTIFACTS['yaml'][0]}\n"
    )
    sidecar_path = path.parent / ORIGINAL_REPORT_ARTIFACTS["sha256"][0]
    if sidecar_path.read_text(encoding="utf-8") != expected_sidecar:
        raise PermissionError("The locked paired-report SHA-256 sidecar is inconsistent.")
    if (
        report.get("experiment_id") != "webots_front_brake_paired_diagnostic_v1"
        or report.get("status") != "invalid_manipulation"
        or report.get("split") != "diagnostic"
        or report.get("validation_split_was_run") is not False
        or report.get("test_split_was_run") is not False
    ):
        raise PermissionError("The paired report is not the locked diagnostic-only invalid run.")

    sources = _mapping(report.get("source_bindings"), "source_bindings")
    expected_source_keys = {"git_revision", "model", "webots_executable", *SOURCE_BINDINGS}
    if set(sources) != expected_source_keys or sources.get("git_revision") != ORIGINAL_GIT_REVISION:
        raise PermissionError("The paired report source snapshot is not the locked original.")
    for name, expected in SOURCE_BINDINGS.items():
        _binding(sources[name], expected, f"source_bindings.{name}")
    _binding(sources["model"], (MODEL_PATH, MODEL_SHA256), "source_bindings.model")
    webots = _mapping(sources["webots_executable"], "source_bindings.webots_executable")
    if webots.get("sha256") != "34e3824b3bb1d64cea8ed58ec4089b607fb83c906d36290df8e2d4a8ee2227c0":
        raise PermissionError("The Webots executable binding is not the locked original.")
    _verify_file(repository_root / MODEL_PATH, MODEL_SHA256, "model checkpoint")

    model = _mapping(report.get("model"), "model")
    if model.get("path") != MODEL_PATH or model.get("sha256") != MODEL_SHA256:
        raise PermissionError("The paired report model is not the locked checkpoint.")

    gate = _mapping(report.get("candidate_gate"), "candidate_gate")
    checks = _mapping(gate.get("checks"), "candidate_gate.checks")
    if gate.get("passed") is not True or any(
        _mapping(check, f"candidate_gate.checks.{name}").get("passed") is not True
        for name, check in checks.items()
    ):
        raise PermissionError("The locked primary efficacy gate was not wholly true.")

    manipulation = _mapping(
        report.get("manipulation_integrity_check"), "manipulation_integrity_check"
    )
    manipulation_checks = _mapping(manipulation.get("checks"), "manipulation checks")
    failed = [name for name, passed in manipulation_checks.items() if passed is not True]
    expected_failed = ["candidate_active_steps_are_exact_speed_cap_braking"]
    if manipulation.get("passed") is not False or failed != expected_failed:
        raise PermissionError("The original report does not have the single locked failed check.")
    metrics = _mapping(manipulation.get("candidate_metrics"), "candidate metrics")
    expected_counts = {
        "active_step_count": 3197,
        "braking_step_count": 3197,
        "speed_cap_binding_step_count": 3196,
    }
    if {name: metrics.get(name) for name in expected_counts} != expected_counts:
        raise PermissionError("The original manipulation counts are not 3197/3197/3196.")

    arms = report.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError("The paired report must contain exactly two arms.")
    by_role = {_mapping(arm, "arm").get("role"): arm for arm in arms}
    if set(by_role) != {"reference", "candidate"}:
        raise ValueError("The paired report arm roles are invalid.")
    for role, expected_artifacts in ARM_ARTIFACTS.items():
        arm = _mapping(by_role[role], f"arms.{role}")
        arm_model = _mapping(arm.get("model"), f"arms.{role}.model")
        if arm_model.get("path") != MODEL_PATH or arm_model.get("sha256") != MODEL_SHA256:
            raise PermissionError(f"The {role} arm model binding is invalid.")
        artifacts = _mapping(arm.get("artifacts"), f"arms.{role}.artifacts")
        for name, expected in expected_artifacts.items():
            _binding(artifacts.get(name), expected, f"arms.{role}.artifacts.{name}")
            _verify_file(repository_root / expected[0], expected[1], f"{role} {name}")
    return report


def _float32_vector(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-value array.")
    source = np.asarray([_finite(item, label) for item in value], dtype=np.float64)
    result = source.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must remain finite after float32 reconstruction.")
    if not np.array_equal(result.astype(np.float64), source):
        raise ValueError(f"{label} is not an exact serialization of a float32 action.")
    return result


def _same_float32_bits(left: np.float32, right: np.float32) -> bool:
    return bool(left.tobytes() == right.tobytes())


def _active_step_semantics(
    step: dict[str, Any], epsilon: float, *, enabled: bool
) -> dict[str, Any]:
    """Reconstruct one shield decision using the original float32 action semantics."""

    policy = step.get("policy_action")
    shielded = step.get("shielded_action")
    policy_f32 = _float32_vector(policy, "policy action")
    shielded_f32 = _float32_vector(shielded, "shielded action")
    policy_linear = policy_f32[0]
    policy_angular = policy_f32[1]
    shielded_linear = shielded_f32[0]
    shielded_angular = shielded_f32[1]
    cap = _finite(step.get("shield_normalized_speed_limit"), "normalized speed cap")
    recorded_intervention = _finite(step.get("shield_intervention"), "shield intervention")
    recorded_active = step.get("shield_active")
    if type(recorded_active) is not bool:
        raise ValueError("shield_active must be a boolean.")

    delta_f32 = np.subtract(shielded_f32, policy_f32, dtype=np.float32)
    recomputed_intervention = float(np.linalg.norm(delta_f32))
    recomputed_active = bool(enabled and recomputed_intervention > epsilon)
    action_braking_delta_f32 = np.subtract(policy_linear, shielded_linear, dtype=np.float32)
    cap_f32 = np.float32(cap)
    admitted_linear_f32 = np.float32(min(float(policy_linear), cap))
    angular_bits_equal = _same_float32_bits(shielded_angular, policy_angular)
    cap_bits_equal = _same_float32_bits(shielded_linear, cap_f32)
    admitted_linear_bits_equal = _same_float32_bits(shielded_linear, admitted_linear_f32)
    recorded_intervention_matches = recorded_intervention == recomputed_intervention
    recorded_active_matches = recorded_active is recomputed_active
    linear_delta_exceeds_epsilon = bool(float(action_braking_delta_f32) > epsilon)
    legacy_double_cap_test_passed = bool(
        float(policy_linear) > cap + epsilon
        and math.isclose(float(shielded_linear), cap, rel_tol=0.0, abs_tol=1e-6)
    )
    operational_test_passed = bool(
        recomputed_active
        and recorded_active_matches
        and recorded_intervention_matches
        and angular_bits_equal
        and cap_bits_equal
        and admitted_linear_bits_equal
        and linear_delta_exceeds_epsilon
    )
    return {
        "recomputed_active_from_float32_action": recomputed_active,
        "recorded_active_matches_recomputed": recorded_active_matches,
        "recorded_intervention_matches_float32_norm": recorded_intervention_matches,
        "angular_action_float32_bits_unchanged": angular_bits_equal,
        "shielded_linear_matches_cap_float32_bits": cap_bits_equal,
        "shielded_linear_matches_admitted_float32_bits": admitted_linear_bits_equal,
        "linear_float32_delta_exceeds_epsilon": linear_delta_exceeds_epsilon,
        "legacy_double_cap_test_passed": legacy_double_cap_test_passed,
        "operational_float32_action_test_passed": operational_test_passed,
        "all_action_float32_bits_unchanged": policy_f32.tobytes() == shielded_f32.tobytes(),
        "angular_delta": abs(float(shielded_angular) - float(policy_angular)),
        "action_braking_delta": float(action_braking_delta_f32),
        "policy_to_double_cap_delta": float(policy_linear) - cap,
        "action_to_double_cap_error": abs(float(shielded_linear) - cap),
        "recorded_intervention": recorded_intervention,
        "recomputed_float32_intervention": recomputed_intervention,
        "epsilon": epsilon,
    }


def _audit_evidence(path: Path, *, expected_enabled: bool) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != len(EXPECTED_SEEDS):
        raise ValueError("Shield evidence must contain exactly 120 non-blank records.")
    seeds: list[int] = []
    active_count = 0
    semantic_count = 0
    legacy_double_cap_count = 0
    intervention_match_count = 0
    boundary: dict[str, Any] | None = None
    total_steps = 0
    for record_index, value in enumerate(records):
        record = _mapping(value, f"evidence row {record_index}")
        identity = _mapping(record.get("identity"), f"evidence row {record_index}.identity")
        seed = identity.get("scenario_seed")
        seeds.append(seed)
        summary = _mapping(record.get("summary"), f"evidence row {record_index}.summary")
        if summary.get("shield_enabled") is not expected_enabled:
            raise ValueError("Shield evidence enabled state disagrees with the arm.")
        epsilon = _finite(summary.get("intervention_epsilon"), "intervention epsilon")
        if not math.isclose(epsilon, 1e-6, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Shield evidence intervention epsilon is not locked to 1e-6.")
        steps = record.get("steps")
        if not isinstance(steps, list) or len(steps) != summary.get("step_count"):
            raise ValueError("Shield evidence steps disagree with the episode summary.")
        row_active = 0
        for expected_step, step_value in enumerate(steps, start=1):
            step = _mapping(step_value, "shield evidence step")
            if step.get("step") != expected_step:
                raise ValueError("Shield evidence steps must be contiguous and one-based.")
            if step.get("shield_enabled") is not expected_enabled:
                raise ValueError("Step shield_enabled disagrees with the arm.")
            semantics = _active_step_semantics(step, epsilon, enabled=expected_enabled)
            if not semantics["recorded_intervention_matches_float32_norm"]:
                raise ValueError(
                    "Recorded shield intervention disagrees with the reconstructed "
                    "float32 action norm."
                )
            intervention_match_count += 1
            if not semantics["recorded_active_matches_recomputed"]:
                raise ValueError(
                    "Step shield_active disagrees with reconstructed SafetyShield semantics."
                )
            if not expected_enabled and not semantics["all_action_float32_bits_unchanged"]:
                raise ValueError("Disabled shield changed the float32 policy action.")
            if not semantics["recomputed_active_from_float32_action"]:
                continue
            row_active += 1
            active_count += 1
            if semantics["operational_float32_action_test_passed"]:
                semantic_count += 1
            if semantics["legacy_double_cap_test_passed"]:
                legacy_double_cap_count += 1
            if seed == BOUNDARY_SEED and expected_step == BOUNDARY_STEP:
                boundary = {"scenario_seed": seed, "step": expected_step, **semantics}
        if row_active != summary.get("active_step_count"):
            raise ValueError("Evidence active steps disagree with the episode summary.")
        total_steps += len(steps)
    if tuple(seeds) != EXPECTED_SEEDS:
        raise PermissionError("Shield evidence does not preserve the locked seed window.")
    if expected_enabled and boundary is None:
        raise ValueError("The locked numerical-boundary step is missing.")
    return {
        "episode_count": len(records),
        "total_step_count": total_steps,
        "active_step_count": active_count,
        "active_steps_satisfying_operational_semantics": semantic_count,
        "legacy_double_cap_delta_count": legacy_double_cap_count,
        "recorded_intervention_matches_float32_norm_step_count": intervention_match_count,
        "boundary": boundary,
    }


def _portable(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _locked_input_bindings(
    *, paired_report_path: Path, repository_root: Path
) -> dict[str, tuple[Path, str]]:
    bindings: dict[str, tuple[Path, str]] = {
        f"paired_report_{name}": (paired_report_path.parent / filename, sha256)
        for name, (filename, sha256) in ORIGINAL_REPORT_ARTIFACTS.items()
    }
    bindings["model_checkpoint"] = (repository_root / MODEL_PATH, MODEL_SHA256)
    for role, artifacts in ARM_ARTIFACTS.items():
        for name, (portable_path, sha256) in artifacts.items():
            bindings[f"{role}_{name}"] = (repository_root / portable_path, sha256)
    return bindings


def _hash_snapshot(
    bindings: dict[str, tuple[Path, str]], *, repository_root: Path, stage: str
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha256) in bindings.items():
        _verify_file(path, expected_sha256, f"{stage} input {name}")
        snapshot[name] = {
            "path": _portable(path, repository_root),
            "sha256": _sha256_file(path),
            "matches_locked_sha256": True,
        }
    return snapshot


def _seal_report(report: dict[str, Any], output_directory: Path) -> None:
    staging = output_directory.with_name(output_directory.name + ".staging")
    if output_directory.exists() or staging.exists():
        raise FileExistsError(
            "Refusing to overwrite final or staging audit directory: "
            f"{output_directory}, {staging}."
        )
    staging.mkdir(parents=True, exist_ok=False)
    json_path = staging / "manipulation_boundary_reaudit.json"
    yaml_path = staging / "manipulation_boundary_reaudit.yaml"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    yaml_path.write_text(
        yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    sidecar = staging / "manipulation_boundary_reaudit.sha256"
    sidecar.write_text(
        f"{_sha256_file(json_path)}  {json_path.name}\n"
        f"{_sha256_file(yaml_path)}  {yaml_path.name}\n",
        encoding="utf-8",
    )
    staging.rename(output_directory)


def audit_front_brake_manipulation(
    *, paired_report_path: Path, output_directory: Path, repository_root: Path
) -> dict[str, Any]:
    """Re-audit the one-step boundary without mutating or re-running the experiment."""

    root = repository_root.resolve()
    report_path = paired_report_path.resolve()
    output = output_directory.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite final or staging audit directory: {output}, {staging}."
        )
    expected_report_name = ORIGINAL_REPORT_ARTIFACTS["json"][0]
    if report_path.name != expected_report_name:
        raise ValueError(f"Paired report filename must remain {expected_report_name!r}.")

    input_bindings = _locked_input_bindings(
        paired_report_path=report_path,
        repository_root=root,
    )
    entry_hashes = _hash_snapshot(input_bindings, repository_root=root, stage="entry")
    original = _load_original_report(report_path, root)
    source_blob_snapshot = _verify_source_blobs(root)
    reference_path = root / ARM_ARTIFACTS["reference"]["shield_evidence_jsonl"][0]
    candidate_path = root / ARM_ARTIFACTS["candidate"]["shield_evidence_jsonl"][0]
    reference = _audit_evidence(reference_path, expected_enabled=False)
    candidate = _audit_evidence(candidate_path, expected_enabled=True)
    boundary = _mapping(candidate["boundary"], "boundary")
    boundary_legacy_passed = bool(boundary["legacy_double_cap_test_passed"])
    boundary_operational_passed = bool(boundary["operational_float32_action_test_passed"])
    boundary_passed = bool(
        boundary_operational_passed
        and not boundary_legacy_passed
        and boundary["policy_to_double_cap_delta"] <= boundary["epsilon"]
    )
    corrected_passed = bool(
        reference["active_step_count"] == 0
        and reference["recorded_intervention_matches_float32_norm_step_count"]
        == reference["total_step_count"]
        and candidate["active_step_count"] == 3197
        and candidate["active_steps_satisfying_operational_semantics"] == 3197
        and candidate["legacy_double_cap_delta_count"] == 3196
        and candidate["recorded_intervention_matches_float32_norm_step_count"]
        == candidate["total_step_count"]
        and boundary_passed
    )
    exit_hashes = _hash_snapshot(input_bindings, repository_root=root, stage="exit")
    input_hashes_unchanged = entry_hashes == exit_hashes
    if not input_hashes_unchanged:
        raise PermissionError("Locked inputs changed during the offline audit.")
    audit_implementation_path = Path(__file__).resolve()
    audit_implementation = {
        "path": _portable(audit_implementation_path, root),
        "sha256": _sha256_file(audit_implementation_path),
    }
    original_gate = _mapping(original["candidate_gate"], "candidate_gate")
    original_manipulation = _mapping(
        original["manipulation_integrity_check"], "manipulation_integrity_check"
    )
    original_manipulation_checks = _mapping(
        original_manipulation["checks"], "manipulation_integrity_check.checks"
    )
    original_failed_checks = [
        name for name, passed in original_manipulation_checks.items() if passed is not True
    ]
    original_remains_invalid = original["status"] == "invalid_manipulation"
    next_allowed_stage = (
        "independent_diagnostic_confirmation_only"
        if corrected_passed and original_remains_invalid
        else "none_audit_failed"
    )
    validation_or_test_allowed = next_allowed_stage in {"validation", "test"}
    primary_result_not_promoted = bool(
        original_remains_invalid
        and original["split"] == "diagnostic"
        and not validation_or_test_allowed
    )
    result = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "passed" if corrected_passed else "failed",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "audit_type": "offline_post_run_checker_audit",
        "inputs": {
            "paired_diagnostic_report_artifacts": {
                name: {
                    "path": _portable(report_path.parent / filename, root),
                    "sha256": sha256,
                }
                for name, (filename, sha256) in ORIGINAL_REPORT_ARTIFACTS.items()
            },
            "source_snapshot": {
                "git_revision": ORIGINAL_GIT_REVISION,
                "bindings": source_blob_snapshot,
            },
            "audit_implementation": audit_implementation,
            "model": {"path": MODEL_PATH, "sha256": MODEL_SHA256},
            "arm_artifacts": {
                role: {
                    name: {"path": value[0], "sha256": value[1]}
                    for name, value in artifacts.items()
                }
                for role, artifacts in ARM_ARTIFACTS.items()
            },
            "hash_snapshots": {
                "entry": entry_hashes,
                "exit": exit_hashes,
                "unchanged": input_hashes_unchanged,
            },
        },
        "original_report_facts": {
            "status": original["status"],
            "primary_efficacy_gate_passed": original_gate["passed"],
            "only_failed_manipulation_check": original_failed_checks[0],
            "reported_counts": {
                "active_step_count": 3197,
                "braking_step_count": 3197,
                "speed_cap_binding_step_count": 3196,
            },
            "original_report_was_modified": (
                entry_hashes["paired_report_json"]["sha256"] != ORIGINAL_REPORT_SHA256
            ),
            "original_status_is_not_rewritten": (original_remains_invalid),
        },
        "operational_semantics": {
            "source": (
                "LidarSafetyShield.active is reconstructed as enabled and the norm "
                "of (shielded_action - policy_action) in float32 > epsilon"
            ),
            "active_step_requirements": {
                "angular_action": "float32_bits_unchanged",
                "shielded_linear": "float32_bits_equal_to_normalized_cap",
                "linear_action_delta": "float32_delta_strictly_greater_than_epsilon",
                "recorded_intervention": "exactly_equals_recomputed_float32_action_norm",
                "recorded_active": "equals_recomputed_active",
            },
        },
        "recomputed_counts": {
            "reference": reference,
            "candidate": {key: value for key, value in candidate.items() if key != "boundary"},
        },
        "numerical_boundary": {
            **boundary,
            "classification": (
                "double_cap_vs_float32_admitted_action_threshold_boundary"
                if boundary_passed
                else "boundary_not_verified"
            ),
            "legacy_double_cap_test_passed": boundary_legacy_passed,
            "operational_float32_action_test_passed": boundary_operational_passed,
            "boundary_condition_verified": boundary_passed,
        },
        "checker_conclusion": {
            "corrected_manipulation_integrity_passed": corrected_passed,
            "original_paired_report_remains_invalid_manipulation": original_remains_invalid,
            "primary_efficacy_result_is_not_promoted_by_this_post_hoc_audit": (
                primary_result_not_promoted
            ),
            "next_allowed_stage": next_allowed_stage,
            "validation_or_test_confirmation_allowed": validation_or_test_allowed,
        },
        "scope": {
            "webots_was_run": False,
            "checkpoint_selection_was_run": False,
            "validation_split_was_accessed": False,
            "test_split_was_accessed": False,
            "input_artifacts_were_modified": not input_hashes_unchanged,
            "output_is_a_new_non_overwriting_directory": (
                not output.exists() and not staging.exists()
            ),
        },
        "written_artifacts": {
            "json": _portable(output / "manipulation_boundary_reaudit.json", root),
            "yaml": _portable(output / "manipulation_boundary_reaudit.yaml", root),
            "sha256": _portable(output / "manipulation_boundary_reaudit.sha256", root),
        },
    }
    _seal_report(result, output)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-report",
        type=Path,
        default=Path(
            "artifacts/results/webots_front_brake_paired_diagnostic_v1/"
            "paired_diagnostic_report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/webots_front_brake_v1_manipulation_reaudit_v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    report = audit_front_brake_manipulation(
        paired_report_path=args.paired_report,
        output_directory=args.output_dir,
        repository_root=root,
    )
    counts = report["recomputed_counts"]["candidate"]
    print(
        f"audit={report['status']} active={counts['active_step_count']} "
        f"operational={counts['active_steps_satisfying_operational_semantics']} "
        "webots_run=false"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
