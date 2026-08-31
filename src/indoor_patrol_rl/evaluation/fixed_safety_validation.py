"""Fail-closed acceptance analysis for one fixed Webots safety system.

The fixed-system validator consumes exactly one pre-registered model/profile
combination and one completed validation run.  It reuses the paired diagnostic
artifact verifier, qualifies the profile through a bound paired report, and
only emits a deployment profile when every locked acceptance gate passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from indoor_patrol_rl.evaluation.checkpoint_selection import sha256_file
from indoor_patrol_rl.evaluation.paired_safety import (
    PairedSafetyArm,
    _boolean,
    _exact_fields,
    _gate_check,
    _git_blob_oid,
    _git_output,
    _integer,
    _load_arm,
    _load_json,
    _load_result_arm,
    _load_yaml,
    _mapping,
    _number,
    _portable,
    _repository_path,
    _sequence,
    _string,
    _verify_clean_head_file,
    _verify_file_sha256,
    _verify_git_revision,
    _verify_preregistration_git_binding,
    build_paired_safety_report,
    load_paired_safety_config,
)

_SAFE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
_CONDITION_PATTERN = re.compile(r"[A-Za-z0-9]+")
_FORMAL_SCENARIO_SEEDS = frozenset(range(10_000, 10_020))
_FIXED_SEED_START = 5_000
_FIXED_EPISODES = 100
_SELECTION_RULE = "post_selection_fixed_system_acceptance_5000_5099"

_ROOT_FIELDS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "split",
    "condition_id",
    "sweep_feature_revision",
    "training_seed",
    "scenario_seed_start",
    "episodes",
    "model",
    "scenario",
    "runtime_inputs",
    "shield_profile",
    "evaluation_arm",
    "diagnostic_qualification",
    "preregistration",
    "gate",
    "decision",
}
_MODEL_FIELDS = {"path", "sha256", "variant", "checkpoint_step"}
_SCENARIO_FIELDS = {
    "obstacle_count",
    "obstacle_speed_range",
    "horizontal_route_half_length",
    "vertical_route_half_length",
    "deterministic_policy",
    "static_layout",
    "action_repeat",
    "max_episode_steps",
}
_RUNTIME_FIELDS = {
    "world",
    "evaluator",
    "environment_adapter",
    "checkpoint_sweep_runner",
    "fixed_validator",
    "fixed_cli",
    "paired_validator",
}
_SOURCE_FIELDS = {"path", "sha256"}
_PROFILE_SPEC_FIELDS = {
    "path",
    "source_sha256",
    "semantic_sha256",
    "result_token",
    "enabled",
}
_ARM_FIELDS = {"id", "role", "port_base", "run_name", "sweep_directory"}
_DIAGNOSTIC_FIELDS = {
    "report_path",
    "report_sha256",
    "paired_config_path",
    "paired_config_sha256",
    "required_outcome",
    "require_passed",
}
_PREREGISTRATION_FIELDS = {
    "fixed_model",
    "fixed_profile",
    "candidate_count",
    "parameter_scan",
    "adaptive_extension",
    "held_out_formal_test_access",
    "diagnostic_seed_overlap",
    "selection_rule",
}
_GATE_FIELDS = {
    "minimum_success_rate",
    "maximum_collision_rate",
    "maximum_timeout_count",
    "maximum_error_count",
}
_DECISION_FIELDS = {
    "pass",
    "fail",
    "formal_model_promotion",
    "output_directory",
    "deployment_directory",
}
_PAIRED_DECISION_FIELDS = {
    "passed",
    "outcome",
    "formal_model_promotion",
    "model_alias_created",
}
_PAIRED_PROFILE_FIELDS = {
    "path",
    "source_sha256",
    "semantic_sha256",
    "result_token",
    "enabled",
}


@dataclass(frozen=True)
class FixedSafetyValidationConfig:
    """Fully validated, source-bound fixed-system acceptance contract."""

    repository_root: Path
    source_path: Path
    source_sha256: str
    raw: Mapping[str, Any]
    model_path: Path
    runtime_paths: Mapping[str, Path]
    arm: PairedSafetyArm
    git_revision: str
    git_binding_path: Path
    git_binding_sha256: str
    preregistration_revision: str
    preregistration_blob_oid: str
    preregistration_committed_at: datetime
    diagnostic_report_path: Path
    diagnostic_report_sha256: str
    diagnostic_report: Mapping[str, Any]
    paired_config_path: Path
    paired_config_sha256: str

    @property
    def output_directory(self) -> Path:
        return _repository_path(
            self.repository_root,
            self.raw["decision"]["output_directory"],
            "decision.output_directory",
            require_file=False,
        )

    @property
    def deployment_directory(self) -> Path:
        return _repository_path(
            self.repository_root,
            self.raw["decision"]["deployment_directory"],
            "decision.deployment_directory",
            require_file=False,
        )

    @property
    def markdown_path(self) -> Path:
        stem = str(self.raw["experiment_id"]).removesuffix("_v1")
        return self.repository_root / "reports" / stem / "fixed_safety_report.md"


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _verify_repository_clean(repository_root: Path) -> None:
    status = _git_output(
        repository_root,
        "status",
        "--porcelain=v2",
        "--untracked-files=all",
    )
    if status:
        raise ValueError(
            "Fixed safety analysis requires a completely clean repository; "
            "tracked, staged, or untracked changes are forbidden."
        )


def _normalized_paired_report(report: Mapping[str, Any]) -> str:
    try:
        normalized = json.loads(json.dumps(report, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Paired diagnostic report is not canonical JSON data.") from error
    bindings = _mapping(normalized.get("source_bindings"), "paired source_bindings")
    if "git_revision" not in bindings:
        raise ValueError("Paired diagnostic report is missing analysis-time git_revision.")
    del bindings["git_revision"]
    return json.dumps(
        normalized,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_paired_gate_checks(report: Mapping[str, Any], label: str) -> None:
    gate = _exact_fields(
        report.get("gate"),
        {"require_all", "checks", "passed"},
        f"{label} gate",
    )
    if gate["require_all"] is not True or gate["passed"] is not True:
        raise ValueError(f"{label} did not pass its require-all gate.")
    checks = _mapping(gate["checks"], f"{label} gate checks")
    if not checks:
        raise ValueError(f"{label} gate checks must not be empty.")
    for name, value in checks.items():
        check = _exact_fields(
            value,
            {"operator", "observed", "threshold", "passed"},
            f"{label} gate check {name}",
        )
        if check["passed"] is not True:
            raise ValueError(f"{label} gate check {name} did not pass.")
def _normalized_source_path(
    repository_root: Path,
    value: Any,
    label: str,
) -> Path:
    return _repository_path(repository_root, value, label, require_file=True)


def _validate_model(raw: Mapping[str, Any], repository_root: Path) -> Path:
    model = _exact_fields(raw["model"], _MODEL_FIELDS, "model")
    model_path = _normalized_source_path(repository_root, model["path"], "model.path")
    _verify_file_sha256(model_path, model["sha256"], "model")
    variant = _string(model["variant"], "model.variant")
    if _SAFE_NAME_PATTERN.fullmatch(variant) is None:
        raise ValueError("model.variant must be a lowercase path-safe name.")
    _integer(model["checkpoint_step"], "model.checkpoint_step", minimum=1)
    return model_path


def _validate_scenario(raw: Mapping[str, Any]) -> None:
    scenario = _exact_fields(raw["scenario"], _SCENARIO_FIELDS, "scenario")
    obstacle_count = _integer(scenario["obstacle_count"], "scenario.obstacle_count", minimum=0)
    if obstacle_count != 1:
        raise ValueError("Fixed R2a system validation requires exactly one obstacle.")
    speed_range = _sequence(scenario["obstacle_speed_range"], "scenario.obstacle_speed_range")
    if len(speed_range) != 2:
        raise ValueError("scenario.obstacle_speed_range must contain exactly two values.")
    speed_min = _number(speed_range[0], "scenario.obstacle_speed_range[0]", minimum=0.0)
    speed_max = _number(speed_range[1], "scenario.obstacle_speed_range[1]", minimum=0.0)
    if speed_min > speed_max:
        raise ValueError("scenario obstacle speed minimum cannot exceed its maximum.")
    _number(
        scenario["horizontal_route_half_length"],
        "scenario.horizontal_route_half_length",
        strictly_positive=True,
    )
    _number(
        scenario["vertical_route_half_length"],
        "scenario.vertical_route_half_length",
        strictly_positive=True,
    )
    if not _boolean(scenario["deterministic_policy"], "scenario.deterministic_policy"):
        raise ValueError("scenario.deterministic_policy must be true.")
    if scenario["static_layout"] != "train_a":
        raise ValueError("Fixed system validation requires the train_a static layout.")
    _integer(scenario["action_repeat"], "scenario.action_repeat", minimum=1)
    _integer(scenario["max_episode_steps"], "scenario.max_episode_steps", minimum=1)


def _validate_runtime_sources(
    raw: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Path]:
    runtime = _exact_fields(raw["runtime_inputs"], _RUNTIME_FIELDS, "runtime_inputs")
    paths: dict[str, Path] = {}
    for name in sorted(_RUNTIME_FIELDS):
        source = _exact_fields(runtime[name], _SOURCE_FIELDS, f"runtime_inputs.{name}")
        path = _normalized_source_path(
            repository_root,
            source["path"],
            f"runtime_inputs.{name}.path",
        )
        _verify_file_sha256(path, source["sha256"], f"runtime_inputs.{name}")
        paths[name] = path
    return paths


def _load_system_arm(
    raw: Mapping[str, Any],
    repository_root: Path,
) -> PairedSafetyArm:
    profile = _exact_fields(raw["shield_profile"], _PROFILE_SPEC_FIELDS, "shield_profile")
    arm = _exact_fields(raw["evaluation_arm"], _ARM_FIELDS, "evaluation_arm")
    if arm["role"] != "system":
        raise ValueError("evaluation_arm.role must be system.")
    internal = {
        "id": arm["id"],
        "role": "candidate",
        "port_base": arm["port_base"],
        "shield_profile": dict(profile),
        "run_name": arm["run_name"],
        "sweep_directory": arm["sweep_directory"],
    }
    result = _load_arm(internal, repository_root=repository_root, index=0)
    if result.profile.enabled is not True:
        raise ValueError("Fixed system validation requires an enabled shield profile.")
    return result


def _validate_preregistration(raw: Mapping[str, Any]) -> None:
    prereg = _exact_fields(
        raw["preregistration"],
        _PREREGISTRATION_FIELDS,
        "preregistration",
    )
    expected = {
        "fixed_model": True,
        "fixed_profile": True,
        "candidate_count": 1,
        "parameter_scan": "forbidden",
        "adaptive_extension": "forbidden",
        "held_out_formal_test_access": "forbidden",
        "diagnostic_seed_overlap": "forbidden",
        "selection_rule": _SELECTION_RULE,
    }
    for name, value in expected.items():
        if prereg[name] != value or type(prereg[name]) is not type(value):
            raise ValueError(f"preregistration.{name} must equal {value!r}.")
    if raw["scenario_seed_start"] != _FIXED_SEED_START or raw["episodes"] != _FIXED_EPISODES:
        raise ValueError("Fixed system acceptance must use scenario seeds 5000 through 5099.")
    fixed_seeds = set(range(_FIXED_SEED_START, _FIXED_SEED_START + _FIXED_EPISODES))
    if fixed_seeds & _FORMAL_SCENARIO_SEEDS:
        raise ValueError("Fixed system acceptance seeds overlap the locked formal seed window.")


def _validate_gate(raw: Mapping[str, Any]) -> None:
    gate = _exact_fields(raw["gate"], _GATE_FIELDS, "gate")
    locked_numbers = {
        "minimum_success_rate": 0.75,
        "maximum_collision_rate": 0.25,
    }
    for name, expected in locked_numbers.items():
        observed = _number(gate[name], f"gate.{name}", minimum=0.0, maximum=1.0)
        if not _same_number(observed, expected):
            raise ValueError(f"gate.{name} must equal the locked threshold {expected}.")
    if _integer(gate["maximum_timeout_count"], "gate.maximum_timeout_count", minimum=0) != 2:
        raise ValueError("gate.maximum_timeout_count must equal 2.")
    if _integer(gate["maximum_error_count"], "gate.maximum_error_count", minimum=0) != 0:
        raise ValueError("gate.maximum_error_count must equal 0.")


def _validate_paired_diagnostic(
    raw: Mapping[str, Any],
    *,
    repository_root: Path,
    model: Mapping[str, Any],
    arm: PairedSafetyArm,
) -> tuple[Path, str, Mapping[str, Any], Path, str]:
    qualification = _exact_fields(
        raw["diagnostic_qualification"],
        _DIAGNOSTIC_FIELDS,
        "diagnostic_qualification",
    )
    if not _boolean(
        qualification["require_passed"],
        "diagnostic_qualification.require_passed",
    ):
        raise ValueError("diagnostic_qualification.require_passed must be true.")
    required_outcome = _string(
        qualification["required_outcome"],
        "diagnostic_qualification.required_outcome",
    )
    report_path = _normalized_source_path(
        repository_root,
        qualification["report_path"],
        "diagnostic_qualification.report_path",
    )
    report_sha256 = _verify_file_sha256(
        report_path,
        qualification["report_sha256"],
        "diagnostic qualification report",
    )
    report = _load_json(report_path, "paired diagnostic report")
    paired_config_path = _normalized_source_path(
        repository_root,
        qualification["paired_config_path"],
        "diagnostic_qualification.paired_config_path",
    )
    paired_config_sha256 = _verify_file_sha256(
        paired_config_path,
        qualification["paired_config_sha256"],
        "paired diagnostic configuration",
    )
    paired_config = load_paired_safety_config(
        paired_config_path,
        repository_root=repository_root,
    )
    recomputed_report = build_paired_safety_report(paired_config)
    _validate_paired_gate_checks(recomputed_report, "recomputed paired diagnostic")
    if _normalized_paired_report(report) != _normalized_paired_report(recomputed_report):
        raise ValueError(
            "Saved paired diagnostic report does not match the recomputed evidence report."
        )
    if report.get("schema_version") != 1 or report.get("kind") != "webots_paired_safety_diagnostic":
        raise ValueError("Diagnostic qualification report has an unsupported identity.")
    if report.get("split") != "validation":
        raise ValueError("Diagnostic qualification report must use the validation split.")
    decision = _exact_fields(
        report.get("decision"),
        _PAIRED_DECISION_FIELDS,
        "paired diagnostic decision",
    )
    if (
        decision["passed"] is not True
        or decision["outcome"] != required_outcome
        or decision["formal_model_promotion"] != "forbidden"
        or decision["model_alias_created"] is not False
    ):
        raise ValueError("Paired diagnostic report does not carry the required passed decision.")
    _validate_paired_gate_checks(report, "saved paired diagnostic")
    bindings = _mapping(report.get("source_bindings"), "paired diagnostic source_bindings")
    paired_model = _exact_fields(bindings.get("model"), {"path", "sha256"}, "paired model")
    if paired_model["path"] != model["path"] or paired_model["sha256"] != model["sha256"]:
        raise ValueError("Fixed validation model does not match the paired diagnostic model.")
    paired_arms = _mapping(bindings.get("arms"), "paired diagnostic arms")
    candidate = _mapping(paired_arms.get("candidate"), "paired diagnostic candidate")
    paired_profile = _exact_fields(
        candidate.get("profile"),
        _PAIRED_PROFILE_FIELDS,
        "paired diagnostic candidate profile",
    )
    expected_profile = {
        "path": _portable(arm.profile.source_path, repository_root),
        "source_sha256": arm.profile.source_sha256,
        "semantic_sha256": arm.profile.semantic_sha256,
        "result_token": arm.profile.result_token,
        "enabled": True,
    }
    if paired_profile != expected_profile:
        raise ValueError("Fixed validation profile does not match the qualified paired candidate.")
    paired_window = _mapping(
        report.get("paired_validation"),
        "paired diagnostic seed window",
    )
    diagnostic_count = _integer(
        paired_window.get("episode_count"),
        "paired diagnostic episode_count",
        minimum=1,
    )
    diagnostic_start = _integer(
        paired_window.get("scenario_seed_start"),
        "paired diagnostic scenario_seed_start",
        minimum=0,
    )
    diagnostic_end = _integer(
        paired_window.get("scenario_seed_end"),
        "paired diagnostic scenario_seed_end",
        minimum=0,
    )
    if diagnostic_end != diagnostic_start + diagnostic_count - 1:
        raise ValueError("Paired diagnostic report does not declare one contiguous seed window.")
    fixed_seeds = set(range(_FIXED_SEED_START, _FIXED_SEED_START + _FIXED_EPISODES))
    diagnostic_seeds = set(range(diagnostic_start, diagnostic_end + 1))
    if fixed_seeds & diagnostic_seeds:
        raise ValueError("Fixed acceptance seeds overlap the paired diagnostic seed window.")
    return (
        report_path,
        report_sha256,
        report,
        paired_config_path,
        paired_config_sha256,
    )


def load_fixed_safety_validation_config(
    path: Path,
    *,
    repository_root: Path,
) -> FixedSafetyValidationConfig:
    """Load and fail-closed validate one fixed-system preregistration."""

    root = Path(repository_root).resolve()
    _verify_repository_clean(root)
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()
    try:
        source_path.relative_to(root)
    except ValueError as error:
        raise ValueError("Fixed safety configuration must stay inside the repository.") from error
    if not source_path.is_file():
        raise FileNotFoundError(f"Fixed safety configuration does not exist: {source_path}")
    raw = _exact_fields(
        _load_yaml(source_path, "fixed safety configuration"),
        _ROOT_FIELDS,
        "fixed safety configuration",
    )
    if _integer(raw["schema_version"], "schema_version") != 1:
        raise ValueError("schema_version must be 1.")
    experiment_id = _string(raw["experiment_id"], "experiment_id")
    if _SAFE_NAME_PATTERN.fullmatch(experiment_id) is None:
        raise ValueError("experiment_id must be a lowercase path-safe name.")
    if raw["purpose"] != "fixed_system_safety_acceptance":
        raise ValueError("purpose must be fixed_system_safety_acceptance.")
    if raw["split"] != "validation":
        raise ValueError("Fixed system acceptance may only use the validation split.")
    condition_id = _string(raw["condition_id"], "condition_id")
    if _CONDITION_PATTERN.fullmatch(condition_id) is None:
        raise ValueError("condition_id must contain only letters and digits.")
    _integer(raw["training_seed"], "training_seed", minimum=0)
    _integer(raw["scenario_seed_start"], "scenario_seed_start", minimum=0)
    _integer(raw["episodes"], "episodes", minimum=1)

    model_path = _validate_model(raw, root)
    _validate_scenario(raw)
    runtime_paths = _validate_runtime_sources(raw, root)
    arm = _load_system_arm(raw, root)
    _validate_preregistration(raw)
    _validate_gate(raw)
    decision = _exact_fields(raw["decision"], _DECISION_FIELDS, "decision")
    if decision["formal_model_promotion"] != "forbidden":
        raise ValueError("decision.formal_model_promotion must be forbidden.")
    if _string(decision["pass"], "decision.pass") == _string(
        decision["fail"],
        "decision.fail",
    ):
        raise ValueError("decision.pass and decision.fail must differ.")
    output = _repository_path(
        root,
        decision["output_directory"],
        "decision.output_directory",
        require_file=False,
    )
    deployment = _repository_path(
        root,
        decision["deployment_directory"],
        "decision.deployment_directory",
        require_file=False,
    )
    if output == deployment:
        raise ValueError("Report and deployment directories must differ.")
    try:
        output.relative_to(root / "artifacts" / "results")
    except ValueError as error:
        raise ValueError("Fixed safety reports must stay under artifacts/results.") from error
    try:
        deployment.relative_to(root / "artifacts" / "deployments")
    except ValueError as error:
        raise ValueError(
            "System deployment profiles must stay under artifacts/deployments."
        ) from error

    (
        diagnostic_path,
        diagnostic_sha256,
        diagnostic_report,
        paired_config_path,
        paired_config_sha256,
    ) = _validate_paired_diagnostic(
        raw,
        repository_root=root,
        model=raw["model"],
        arm=arm,
    )
    revision = _verify_git_revision(root, raw["sweep_feature_revision"])
    git_binding = _verify_preregistration_git_binding(
        root,
        source_path,
        feature_revision=raw["sweep_feature_revision"],
        current_revision=revision,
    )
    _verify_clean_head_file(root, diagnostic_path, "paired diagnostic report")
    diagnostic_relative = _portable(diagnostic_path, root)
    preregistered_diagnostic_oid = _git_blob_oid(
        root,
        git_binding.preregistration_revision,
        diagnostic_relative,
        "pre-registered paired diagnostic report",
    )
    if _git_blob_oid(root, "HEAD", diagnostic_relative, "HEAD paired diagnostic report") != (
        preregistered_diagnostic_oid
    ):
        raise ValueError("Paired diagnostic report changed after fixed-system preregistration.")
    _verify_clean_head_file(root, paired_config_path, "paired diagnostic configuration")
    paired_config_relative = _portable(paired_config_path, root)
    preregistered_paired_config_oid = _git_blob_oid(
        root,
        git_binding.preregistration_revision,
        paired_config_relative,
        "pre-registered paired diagnostic configuration",
    )
    if _git_blob_oid(
        root,
        "HEAD",
        paired_config_relative,
        "HEAD paired diagnostic configuration",
    ) != preregistered_paired_config_oid:
        raise ValueError("Paired diagnostic config changed after fixed-system preregistration.")

    return FixedSafetyValidationConfig(
        repository_root=root,
        source_path=source_path,
        source_sha256=sha256_file(source_path),
        raw=raw,
        model_path=model_path,
        runtime_paths=runtime_paths,
        arm=arm,
        git_revision=revision,
        git_binding_path=git_binding.binding_path,
        git_binding_sha256=git_binding.binding_sha256,
        preregistration_revision=git_binding.preregistration_revision,
        preregistration_blob_oid=git_binding.config_blob_oid,
        preregistration_committed_at=git_binding.committed_at,
        diagnostic_report_path=diagnostic_path,
        diagnostic_report_sha256=diagnostic_sha256,
        diagnostic_report=diagnostic_report,
        paired_config_path=paired_config_path,
        paired_config_sha256=paired_config_sha256,
    )


def _validate_single_sweep_run(config: FixedSafetyValidationConfig) -> None:
    manifest_path = config.arm.sweep_directory / "checkpoint_sweep.json"
    manifest = _load_json(manifest_path, "fixed system sweep manifest")
    request = _mapping(manifest.get("request"), "fixed system sweep request")
    expected_step = config.raw["model"]["checkpoint_step"]
    if request.get("steps") != [expected_step]:
        raise ValueError("Fixed system sweep must request exactly the pre-registered checkpoint.")
    if request.get("port_base") != config.raw["evaluation_arm"]["port_base"]:
        raise ValueError("Fixed system sweep port_base does not match the pre-registration.")
    if request.get("workers") != 1:
        raise ValueError("Fixed system sweep must use exactly one worker.")
    runs = _sequence(manifest.get("runs"), "fixed system sweep runs")
    if len(runs) != 1:
        raise ValueError("Fixed system sweep must contain exactly one run.")
    run = _mapping(runs[0], "fixed system sweep run")
    expected_planner = f"ppo_{config.raw['condition_id'].casefold()}_step{expected_step}"
    expected = {
        "run_name": config.raw["evaluation_arm"]["run_name"],
        "step": expected_step,
        "port": config.raw["evaluation_arm"]["port_base"],
        "planner": expected_planner,
    }
    for name, value in expected.items():
        if run.get(name) != value or type(run.get(name)) is not type(value):
            raise ValueError(f"Fixed system sweep run.{name} does not match pre-registration.")
    environment = _mapping(run.get("environment"), "fixed system sweep environment")
    expected_environment = {
        "WEBOTS_EVAL_MODEL": config.raw["model"]["path"],
        "WEBOTS_EVAL_MODEL_SHA256": config.raw["model"]["sha256"],
        "WEBOTS_EVAL_RUN_NAME": config.raw["evaluation_arm"]["run_name"],
        "WEBOTS_EVAL_CONDITION": config.raw["condition_id"],
        "WEBOTS_EVAL_SEED_START": str(config.raw["scenario_seed_start"]),
        "WEBOTS_EVAL_EPISODES": str(config.raw["episodes"]),
        "WEBOTS_SAFETY_SHIELD": "true",
    }
    for name, value in expected_environment.items():
        if environment.get(name) != value:
            raise ValueError(f"Fixed system sweep environment {name} does not match config.")
    profile_environment = config.arm.profile.controller_environment
    for name, value in profile_environment.items():
        if environment.get(name) != value:
            raise ValueError(f"Fixed system sweep environment {name} does not match profile.")


def _recheck_mutable_inputs(config: FixedSafetyValidationConfig) -> None:
    _verify_repository_clean(config.repository_root)
    if sha256_file(config.source_path) != config.source_sha256:
        raise ValueError("Fixed safety configuration changed after it was loaded.")
    _verify_file_sha256(config.model_path, config.raw["model"]["sha256"], "model")
    for name, path in config.runtime_paths.items():
        _verify_file_sha256(path, config.raw["runtime_inputs"][name]["sha256"], name)
    config.arm.profile.verify_source(config.repository_root)
    _verify_file_sha256(
        config.diagnostic_report_path,
        config.diagnostic_report_sha256,
        "paired diagnostic report",
    )
    _verify_file_sha256(
        config.paired_config_path,
        config.paired_config_sha256,
        "paired diagnostic configuration",
    )
    if _git_output(config.repository_root, "rev-parse", "HEAD") != config.git_revision:
        raise ValueError("Git revision changed after the fixed safety config was loaded.")
    binding = _verify_preregistration_git_binding(
        config.repository_root,
        config.source_path,
        feature_revision=config.raw["sweep_feature_revision"],
        current_revision=config.git_revision,
    )
    if (
        binding.binding_sha256 != config.git_binding_sha256
        or binding.preregistration_revision != config.preregistration_revision
        or binding.config_blob_oid != config.preregistration_blob_oid
        or binding.committed_at != config.preregistration_committed_at
    ):
        raise ValueError("Fixed safety pre-registration binding changed after loading.")
    paired_config = load_paired_safety_config(
        config.paired_config_path,
        repository_root=config.repository_root,
    )
    recomputed_report = build_paired_safety_report(paired_config)
    _validate_paired_gate_checks(recomputed_report, "recomputed paired diagnostic")
    if _normalized_paired_report(config.diagnostic_report) != _normalized_paired_report(
        recomputed_report
    ):
        raise ValueError(
            "Saved paired diagnostic report changed relative to recomputed evidence."
        )


def _source_binding(config: FixedSafetyValidationConfig, name: str) -> dict[str, str]:
    source = config.raw["runtime_inputs"][name]
    return {
        "path": _portable(config.runtime_paths[name], config.repository_root),
        "sha256": source["sha256"],
    }


def build_fixed_safety_report(config: FixedSafetyValidationConfig) -> dict[str, Any]:
    """Verify the fixed run, recompute metrics, and apply locked acceptance gates."""

    _recheck_mutable_inputs(config)
    _validate_single_sweep_run(config)
    loaded = _load_result_arm(config, config.arm)
    counts = loaded.counts
    total = counts["episodes"]
    success_rate = counts["success_count"] / total
    collision_rate = counts["collision_count"] / total
    timeout_rate = counts["timeout_count"] / total
    gate = config.raw["gate"]
    checks = {
        "minimum_success_rate": _gate_check(">=", success_rate, gate["minimum_success_rate"]),
        "maximum_collision_rate": _gate_check(
            "<=",
            collision_rate,
            gate["maximum_collision_rate"],
        ),
        "maximum_timeout_count": _gate_check(
            "<=",
            counts["timeout_count"],
            gate["maximum_timeout_count"],
        ),
        "maximum_error_count": _gate_check(
            "<=",
            counts["error_count"],
            gate["maximum_error_count"],
        ),
    }
    passed = all(check["passed"] for check in checks.values())
    decision = config.raw["decision"]
    profile = config.arm.profile
    diagnostic_decision = config.diagnostic_report["decision"]
    deployment_json = config.deployment_directory / "system_deployment_profile.json"
    return {
        "schema_version": 1,
        "kind": "webots_fixed_system_safety_validation",
        "experiment_id": config.raw["experiment_id"],
        "purpose": config.raw["purpose"],
        "split": config.raw["split"],
        "condition_id": config.raw["condition_id"],
        "source_bindings": {
            "input_config": {
                "path": _portable(config.source_path, config.repository_root),
                "sha256": config.source_sha256,
                "git_binding": {
                    "path": _portable(config.git_binding_path, config.repository_root),
                    "sha256": config.git_binding_sha256,
                    "preregistration_revision": config.preregistration_revision,
                    "preregistration_committed_at": (
                        config.preregistration_committed_at.isoformat()
                    ),
                    "config_blob_oid": config.preregistration_blob_oid,
                    "config_clean": True,
                    "committed_before_run": True,
                },
            },
            "git_revision": config.git_revision,
            "sweep_feature_revision": config.raw["sweep_feature_revision"],
            "model": {
                "path": _portable(config.model_path, config.repository_root),
                "sha256": config.raw["model"]["sha256"],
                "variant": config.raw["model"]["variant"],
                "checkpoint_step": config.raw["model"]["checkpoint_step"],
            },
            "shield_profile": {
                "path": _portable(profile.source_path, config.repository_root),
                "source_sha256": profile.source_sha256,
                "semantic_sha256": profile.semantic_sha256,
                "result_token": profile.result_token,
                "enabled": profile.enabled,
                "resolved_full_config": dict(loaded.shield_config),
            },
            "runtime_inputs": {
                name: _source_binding(config, name) for name in sorted(_RUNTIME_FIELDS)
            },
            "paired_diagnostic_report": {
                "path": _portable(config.diagnostic_report_path, config.repository_root),
                "sha256": config.diagnostic_report_sha256,
                "passed": diagnostic_decision["passed"],
                "outcome": diagnostic_decision["outcome"],
            },
            "paired_diagnostic_config": {
                "path": _portable(config.paired_config_path, config.repository_root),
                "sha256": config.paired_config_sha256,
            },
            "sweep_manifest": {
                "path": _portable(loaded.manifest_path, config.repository_root),
                "sha256": loaded.manifest_sha256,
            },
            "summary": {
                "path": _portable(loaded.summary_path, config.repository_root),
                "sha256": loaded.summary_sha256,
            },
            "episodes_csv": {
                "path": _portable(loaded.episodes_path, config.repository_root),
                "sha256": loaded.episodes_sha256,
            },
        },
        "provenance_limits": {
            "evaluation_runtime_worktree_state_cryptographically_bound": False,
            "webots_executable_binary_sha256_bound": False,
            "interpretation": (
                "The fixed acceptance binds its pre-registered source files and result "
                "artifacts, but the evaluation-time worktree and Webots executable were "
                "not captured as cryptographic runtime snapshots."
            ),
        },
        "fixed_validation": {
            "single_model": True,
            "single_profile": True,
            "single_seed_window": True,
            "single_run": True,
            "scenario_seed_start": config.raw["scenario_seed_start"],
            "scenario_seed_end": (
                config.raw["scenario_seed_start"] + config.raw["episodes"] - 1
            ),
            "episode_count": config.raw["episodes"],
            "diagnostic_seed_overlap": False,
            "formal_seed_overlap": False,
            "selection_rule": _SELECTION_RULE,
            "run_started_at": loaded.started_at.isoformat(),
        },
        "metrics": {
            **counts,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
            "error_rate": counts["error_count"] / total,
        },
        "gate": {
            "require_all": True,
            "checks": checks,
            "passed": passed,
        },
        "decision": {
            "passed": passed,
            "outcome": decision["pass"] if passed else decision["fail"],
            "formal_model_promotion": "forbidden",
            "policy_only_alias_created": False,
            "system_deployment_profile_eligible": passed,
            "expected_system_deployment_profile_path": (
                _portable(deployment_json, config.repository_root) if passed else None
            ),
        },
    }


def build_system_deployment_profile(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    report_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    """Build the system-level deployment identity from a passed report."""

    if report["decision"]["passed"] is not True:
        raise ValueError("A deployment profile requires a passed fixed safety report.")
    bindings = report["source_bindings"]
    fixed = report["fixed_validation"]
    return {
        "schema_version": 1,
        "kind": "webots_system_deployment_profile",
        "system_id": report["experiment_id"],
        "qualification": {
            "fixed_safety_report": {
                "path": _portable(report_path, repository_root),
                "sha256": report_sha256,
                "decision": report["decision"]["outcome"],
            },
            "paired_diagnostic_report": dict(bindings["paired_diagnostic_report"]),
            "paired_diagnostic_config": dict(bindings["paired_diagnostic_config"]),
            "git_revision": bindings["git_revision"],
            "preregistration_revision": bindings["input_config"]["git_binding"][
                "preregistration_revision"
            ],
        },
        "system": {
            "model": dict(bindings["model"]),
            "shield_profile": dict(bindings["shield_profile"]),
        },
        "validation_evidence": {
            "condition_id": report["condition_id"],
            "scenario_seed_start": fixed["scenario_seed_start"],
            "scenario_seed_end": fixed["scenario_seed_end"],
            "episode_count": fixed["episode_count"],
            "sweep_manifest": dict(bindings["sweep_manifest"]),
            "summary": dict(bindings["summary"]),
            "episodes_csv": dict(bindings["episodes_csv"]),
            "metrics": dict(report["metrics"]),
            "gate": dict(report["gate"]),
        },
        "runtime_inputs": dict(bindings["runtime_inputs"]),
        "restrictions": {
            "formal_model_promotion": "forbidden",
            "policy_only_alias_created": False,
            "model_and_shield_must_be_deployed_together": True,
        },
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def render_fixed_safety_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise, auditable fixed-system acceptance report."""

    decision = report["decision"]
    metrics = report["metrics"]
    status = "PASS" if decision["passed"] else "FAIL"
    lines = [
        f"# {report['experiment_id']} fixed-system safety report",
        "",
        f"Decision: **{status}** — `{decision['outcome']}`.",
        "",
        "> This validates one model-plus-shield system. It never creates a policy-only alias ",
        "> and does not promote a formal-test model.",
        "",
        "## Recomputed outcomes",
        "",
        "| Episodes | Success | Collision | Timeout | Error | Success rate | Collision rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {metrics['episodes']} | {metrics['success_count']} | "
            f"{metrics['collision_count']} | {metrics['timeout_count']} | "
            f"{metrics['error_count']} | {metrics['success_rate']:.6f} | "
            f"{metrics['collision_rate']:.6f} |"
        ),
        "",
        "## Locked gate",
        "",
        "| Check | Observed | Requirement | Result |",
        "| --- | ---: | ---: | :---: |",
    ]
    for name, check in report["gate"]["checks"].items():
        lines.append(
            f"| `{name}` | {_format_value(check['observed'])} | "
            f"{check['operator']} {_format_value(check['threshold'])} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    bindings = report["source_bindings"]
    lines.extend(
        [
            "",
            "## Audit bindings",
            "",
            f"- Model SHA-256: `{bindings['model']['sha256']}`",
            f"- Profile source SHA-256: `{bindings['shield_profile']['source_sha256']}`",
            f"- Profile semantic SHA-256: `{bindings['shield_profile']['semantic_sha256']}`",
            (
                "- Paired diagnostic report SHA-256: "
                f"`{bindings['paired_diagnostic_report']['sha256']}`"
            ),
            f"- Sweep manifest SHA-256: `{bindings['sweep_manifest']['sha256']}`",
            f"- Summary SHA-256: `{bindings['summary']['sha256']}`",
            f"- Episodes CSV SHA-256: `{bindings['episodes_csv']['sha256']}`",
            f"- Git revision: `{bindings['git_revision']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _json_yaml_digest_payload(value: Mapping[str, Any]) -> tuple[str, str, str]:
    json_text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    yaml_text = yaml.safe_dump(json.loads(json_text), allow_unicode=True, sort_keys=False)
    digest = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    return json_text, yaml_text, digest


def write_fixed_safety_outputs(
    report: Mapping[str, Any],
    *,
    repository_root: Path,
    output_directory: Path,
    deployment_directory: Path,
    markdown_path: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write reports and, only on pass, the system deployment profile."""

    output = Path(output_directory)
    deployment = Path(deployment_directory)
    markdown = Path(markdown_path)
    report_json = output / "fixed_safety_report.json"
    report_yaml = output / "fixed_safety_report.yaml"
    report_digest = output / "fixed_safety_report.sha256"
    targets = [report_json, report_yaml, report_digest, markdown]
    deployment_json = deployment / "system_deployment_profile.json"
    deployment_yaml = deployment / "system_deployment_profile.yaml"
    deployment_digest = deployment / "system_deployment_profile.sha256"
    deployment_targets = [deployment_json, deployment_yaml, deployment_digest]
    if report["decision"]["passed"]:
        targets.extend(deployment_targets)
    elif any(path.exists() for path in deployment_targets):
        raise FileExistsError(
            "Refusing to write a failed validation while stale deployment artifacts exist."
        )
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite fixed safety artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    report_json_text, report_yaml_text, report_sha256 = _json_yaml_digest_payload(report)
    deployment_profile: Mapping[str, Any] | None = None
    deployment_json_text = ""
    deployment_yaml_text = ""
    deployment_sha256 = ""
    if report["decision"]["passed"]:
        deployment_profile = build_system_deployment_profile(
            report,
            report_path=report_json,
            report_sha256=report_sha256,
            repository_root=repository_root,
        )
        deployment_json_text, deployment_yaml_text, deployment_sha256 = (
            _json_yaml_digest_payload(deployment_profile)
        )

    output.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(report_json_text, encoding="utf-8")
    report_yaml.write_text(report_yaml_text, encoding="utf-8")
    report_digest.write_text(f"{report_sha256}  {report_json.name}\n", encoding="utf-8")
    markdown.write_text(render_fixed_safety_markdown(report), encoding="utf-8")
    written = {
        "report_json": report_json,
        "report_yaml": report_yaml,
        "report_sha256": report_digest,
        "markdown": markdown,
    }
    if deployment_profile is not None:
        deployment.mkdir(parents=True, exist_ok=True)
        deployment_json.write_text(deployment_json_text, encoding="utf-8")
        deployment_yaml.write_text(deployment_yaml_text, encoding="utf-8")
        deployment_digest.write_text(
            f"{deployment_sha256}  {deployment_json.name}\n",
            encoding="utf-8",
        )
        written.update(
            {
                "deployment_json": deployment_json,
                "deployment_yaml": deployment_yaml,
                "deployment_sha256": deployment_digest,
            }
        )
    return written


def analyze_fixed_safety_validation(
    config_path: Path,
    *,
    repository_root: Path,
    output_directory: Path | None = None,
    deployment_directory: Path | None = None,
    markdown_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Analyze one fixed-system run and write its auditable artifacts."""

    config = load_fixed_safety_validation_config(
        config_path,
        repository_root=repository_root,
    )
    report = build_fixed_safety_report(config)
    resolved_deployment_directory = deployment_directory or config.deployment_directory
    if report["decision"]["passed"]:
        deployment_json = Path(resolved_deployment_directory) / "system_deployment_profile.json"
        report["decision"]["expected_system_deployment_profile_path"] = _portable(
            deployment_json,
            config.repository_root,
        )
    write_fixed_safety_outputs(
        report,
        repository_root=config.repository_root,
        output_directory=output_directory or config.output_directory,
        deployment_directory=resolved_deployment_directory,
        markdown_path=markdown_path or config.markdown_path,
        overwrite=overwrite,
    )
    return report
