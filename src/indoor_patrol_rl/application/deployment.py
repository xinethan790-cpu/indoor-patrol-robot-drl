"""Fail-closed loading of an evidence-bound Webots model-plus-shield profile.

The fixed-safety campaign evaluates a *system*, not a policy checkpoint by
itself.  This module is the application-layer boundary that turns its emitted
``system_deployment_profile.json`` into a small typed runtime object.  Loading
fails if the profile, model, shield source, qualification evidence, validation
evidence, or bound runtime inputs no longer match their recorded SHA-256.

The R2a evidence covers its recorded evaluation world and is not, by itself,
qualification of the separate four-point patrol application scenario.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from indoor_patrol_rl.core import SafetyShieldConfig
from indoor_patrol_rl.evaluation.checkpoint_sweep import (
    WebotsSafetyShieldProfile,
    load_safety_shield_profile,
)

SYSTEM_DEPLOYMENT_PROFILE_SCHEMA_VERSION = 1

DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH = Path(
    "artifacts/deployments/webots_r2a_front_brake_v1/"
    "model-051d1e2f07bb_profile-354ce9b831f0/system_deployment_profile.json"
)
AUDITED_R2A_SYSTEM_ID = "webots_r2a_front_brake_fixed_system_validation_v1"
AUDITED_R2A_CONDITION_ID = "R2ASV"
AUDITED_R2A_MODEL_VARIANT = "r2a_targeted_v2"
AUDITED_R2A_CHECKPOINT_STEP = 10_240
AUDITED_R2A_MODEL_SHA256 = "051d1e2f07bb147d247c91ded821e58af5b87987bedc2cee4e287fdbdfcdcbe0"
AUDITED_R2A_PROFILE_SHA256 = "f0d5bf258057589ac5f2da9fcd7e211c4b7a5639d147b61588bd6572ee579383"
AUDITED_R2A_SHIELD_RESULT_TOKEN = "r2a-front-brake-v1-354ce9b831f0"
AUDITED_R2A_SHIELD_SEMANTIC_SHA256 = (
    "354ce9b831f01fb9e9c72a9ad1650574f18c98b1b5fafea26671f81ee000ec08"
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SAFE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CONDITION_PATTERN = re.compile(r"[A-Za-z0-9]+")
_ROOT_FIELDS = {
    "schema_version",
    "kind",
    "system_id",
    "qualification",
    "system",
    "validation_evidence",
    "runtime_inputs",
    "restrictions",
}
_RUNTIME_INPUT_FIELDS = {
    "world",
    "evaluator",
    "environment_adapter",
    "checkpoint_sweep_runner",
    "fixed_validator",
    "fixed_cli",
    "paired_validator",
}
_METRIC_FIELDS = {
    "episodes",
    "success_count",
    "collision_count",
    "timeout_count",
    "error_count",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "error_rate",
}
_GATE_CHECKS = {
    "minimum_success_rate": ("success_rate", ">="),
    "maximum_collision_rate": ("collision_rate", "<="),
    "maximum_timeout_count": ("timeout_count", "<="),
    "maximum_error_count": ("error_count", "<="),
}


class DeploymentProfileError(ValueError):
    """A deployment profile is malformed or has inconsistent identities."""


class DeploymentIntegrityError(DeploymentProfileError):
    """A profile-bound file no longer matches its recorded digest."""


@dataclass(frozen=True, slots=True)
class DeploymentArtifactBinding:
    """One verified repository-local file named by a deployment profile."""

    name: str
    repository_path: str
    path: Path
    sha256: str

    def verify_integrity(self) -> None:
        """Recheck the artifact after loading and fail if it changed."""

        if not self.path.is_file():
            raise FileNotFoundError(f"Deployment artifact is missing: {self.path}")
        actual = _file_sha256(self.path)
        if actual != self.sha256:
            raise DeploymentIntegrityError(
                f"Deployment artifact {self.name} SHA-256 mismatch: "
                f"expected {self.sha256}, got {actual}."
            )


@dataclass(frozen=True, slots=True)
class VerifiedSystemDeploymentProfile:
    """A verified model-plus-shield profile and its recorded evidence identity."""

    repository_root: Path
    profile_path: Path
    profile_sha256: str
    profile_digest_path: Path
    system_id: str
    model: DeploymentArtifactBinding
    model_variant: str
    checkpoint_step: int
    shield_profile: WebotsSafetyShieldProfile
    shield_result_token: str
    shield_semantic_sha256: str
    validation_condition_id: str
    validation_episode_count: int
    validation_success_rate: float
    validation_collision_rate: float
    qualification_artifacts: tuple[DeploymentArtifactBinding, ...]
    validation_artifacts: tuple[DeploymentArtifactBinding, ...]
    runtime_artifacts: tuple[DeploymentArtifactBinding, ...]

    @property
    def model_path(self) -> Path:
        """Return the already verified PPO checkpoint path."""

        return self.model.path

    @property
    def model_sha256(self) -> str:
        """Return the immutable checkpoint identity used by live reporting."""

        return self.model.sha256

    @property
    def safety_shield_config(self) -> SafetyShieldConfig:
        """Return the exact shield configuration qualified with the model."""

        return self.shield_profile.config

    @property
    def safety_shield_id(self) -> str:
        """Return the result token that identifies profile semantics."""

        return self.shield_result_token

    @property
    def safety_shield_sha256(self) -> str:
        """Return the shield semantic SHA-256 used by runtime identity records."""

        return self.shield_semantic_sha256

    @property
    def all_artifacts(self) -> tuple[DeploymentArtifactBinding, ...]:
        """Return every file whose digest is bound by the deployment profile."""

        shield = DeploymentArtifactBinding(
            name="system.shield_profile",
            repository_path=_portable_existing_path(
                self.repository_root,
                self.shield_profile.source_path,
                "shield profile source",
            ),
            path=self.shield_profile.source_path,
            sha256=self.shield_profile.source_sha256,
        )
        return (
            self.model,
            shield,
            *self.qualification_artifacts,
            *self.validation_artifacts,
            *self.runtime_artifacts,
        )

    def verify_integrity(self) -> None:
        """Recheck the complete deployment bundle immediately before use."""

        if not self.profile_path.is_file():
            raise FileNotFoundError(f"Deployment profile is missing: {self.profile_path}")
        actual_profile_sha256 = _file_sha256(self.profile_path)
        if actual_profile_sha256 != self.profile_sha256:
            raise DeploymentIntegrityError("System deployment profile changed after it was loaded.")
        _verify_digest_sidecar(
            self.profile_digest_path,
            profile_path=self.profile_path,
            actual_sha256=actual_profile_sha256,
        )
        for artifact in self.all_artifacts:
            artifact.verify_integrity()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_rejecting_mapping(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DeploymentProfileError(f"Duplicate JSON field is forbidden: {key}.")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise DeploymentProfileError(f"Non-finite JSON number is forbidden: {value}.")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DeploymentProfileError(f"Cannot read {label} {path}: {error}") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_mapping,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DeploymentProfileError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise DeploymentProfileError(f"{label} must be a JSON object.")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DeploymentProfileError(f"{label} must be an object.")
    return value


def _exact_mapping(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    actual = set(mapping)
    if actual != expected:
        raise DeploymentProfileError(
            f"{label} fields differ from the deployment schema; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(str(name) for name in actual - expected)}."
        )
    return mapping


def _string(value: Any, label: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise DeploymentProfileError(f"{label} must be a non-empty string.")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise DeploymentProfileError(f"{label} has an invalid identity: {value!r}.")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DeploymentProfileError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DeploymentProfileError(f"{label} must be an integer >= {minimum}.")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeploymentProfileError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise DeploymentProfileError(f"{label} must be a finite number.")
    return result


def _same_number(left: Any, right: Any) -> bool:
    try:
        left_value = _number(left, "numeric identity")
        right_value = _number(right, "numeric identity")
    except DeploymentProfileError:
        return False
    return math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12)


def _portable_path(value: Any, label: str) -> str:
    text = _string(value, label)
    if "\\" in text:
        raise DeploymentProfileError(f"{label} must use forward slashes.")
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or text != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DeploymentProfileError(f"{label} must be a normalized repository-relative path.")
    return text


def _resolve_repository_path(repository_root: Path, value: Any, label: str) -> tuple[str, Path]:
    portable = _portable_path(value, label)
    path = (repository_root / Path(*PurePosixPath(portable).parts)).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as error:
        raise DeploymentProfileError(f"{label} escapes the repository.") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return portable, path


def _resolve_profile_path(repository_root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else repository_root / value
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise DeploymentProfileError(
            "System deployment profile must stay inside the repository."
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(f"System deployment profile does not exist: {resolved}")
    return resolved


def _portable_existing_path(repository_root: Path, path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise DeploymentProfileError(f"{label} escapes the repository.") from error


def _verify_digest_sidecar(
    digest_path: Path,
    *,
    profile_path: Path,
    actual_sha256: str,
) -> None:
    if not digest_path.is_file():
        raise FileNotFoundError(f"Deployment profile digest is missing: {digest_path}")
    try:
        lines = digest_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DeploymentProfileError(
            f"Cannot read deployment profile digest {digest_path}: {error}"
        ) from error
    expected_line = f"{actual_sha256}  {profile_path.name}"
    if lines != [expected_line]:
        raise DeploymentIntegrityError(
            "Deployment profile digest sidecar does not match the JSON artifact."
        )


def _artifact_binding(
    raw: Mapping[str, Any],
    *,
    name: str,
    repository_root: Path,
) -> DeploymentArtifactBinding:
    portable, path = _resolve_repository_path(
        repository_root,
        raw["path"],
        f"{name}.path",
    )
    expected = _sha256(raw["sha256"], f"{name}.sha256")
    binding = DeploymentArtifactBinding(
        name=name,
        repository_path=portable,
        path=path,
        sha256=expected,
    )
    binding.verify_integrity()
    return binding


def _validate_metrics(raw: Any, episode_count: int) -> tuple[float, float]:
    metrics = _exact_mapping(raw, _METRIC_FIELDS, "validation_evidence.metrics")
    counts = {
        name: _integer(metrics[name], f"validation_evidence.metrics.{name}")
        for name in (
            "episodes",
            "success_count",
            "collision_count",
            "timeout_count",
            "error_count",
        )
    }
    if counts["episodes"] != episode_count:
        raise DeploymentProfileError("Validation metrics episode count is inconsistent.")
    outcome_total = sum(
        counts[name]
        for name in ("success_count", "collision_count", "timeout_count", "error_count")
    )
    if outcome_total != episode_count:
        raise DeploymentProfileError("Validation outcome counts do not sum to episode_count.")
    rates: dict[str, float] = {}
    for outcome in ("success", "collision", "timeout", "error"):
        rate_name = f"{outcome}_rate"
        rate = _number(metrics[rate_name], f"validation_evidence.metrics.{rate_name}")
        expected = counts[f"{outcome}_count"] / episode_count
        if not math.isclose(rate, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise DeploymentProfileError(f"Validation {rate_name} is inconsistent with counts.")
        rates[outcome] = rate
    return rates["success"], rates["collision"]


def _validate_gate(raw: Any, metrics: Mapping[str, Any]) -> None:
    gate = _exact_mapping(raw, {"require_all", "checks", "passed"}, "validation_evidence.gate")
    if gate["require_all"] is not True or gate["passed"] is not True:
        raise DeploymentProfileError("Deployment requires a passed require-all validation gate.")
    checks = _exact_mapping(gate["checks"], set(_GATE_CHECKS), "validation_evidence.gate.checks")
    for name, (metric_name, operator) in _GATE_CHECKS.items():
        check = _exact_mapping(
            checks[name],
            {"observed", "operator", "passed", "threshold"},
            f"validation_evidence.gate.checks.{name}",
        )
        if check["operator"] != operator or check["passed"] is not True:
            raise DeploymentProfileError(
                f"Validation gate check {name} is not a passed {operator} check."
            )
        observed = _number(check["observed"], f"validation gate {name}.observed")
        threshold = _number(check["threshold"], f"validation gate {name}.threshold")
        if not _same_number(observed, metrics[metric_name]):
            raise DeploymentProfileError(f"Validation gate check {name} observed value is stale.")
        passed = observed >= threshold if operator == ">=" else observed <= threshold
        if not passed:
            raise DeploymentProfileError(f"Validation gate check {name} contradicts its threshold.")


def _validate_shield_config(
    raw: Any,
    shield_profile: WebotsSafetyShieldProfile,
) -> None:
    expected = asdict(shield_profile.config)
    actual = _exact_mapping(raw, set(expected), "system.shield_profile.resolved_full_config")
    for name, expected_value in expected.items():
        if not _same_number(actual[name], expected_value):
            raise DeploymentProfileError(
                f"Resolved safety-shield config field {name} differs from its source profile."
            )


def _validate_fixed_report_identity(
    report: Mapping[str, Any],
    *,
    profile_raw: Mapping[str, Any],
    profile_repository_path: str,
) -> None:
    if report.get("schema_version") != 1 or report.get("kind") != (
        "webots_fixed_system_safety_validation"
    ):
        raise DeploymentProfileError("Fixed safety report has the wrong schema identity.")
    if report.get("experiment_id") != profile_raw["system_id"]:
        raise DeploymentProfileError("Fixed safety report system identity does not match.")
    evidence = _mapping(profile_raw["validation_evidence"], "validation_evidence")
    if report.get("condition_id") != evidence["condition_id"]:
        raise DeploymentProfileError("Fixed safety report condition identity does not match.")
    decision = _mapping(report.get("decision"), "fixed safety report decision")
    qualification = _mapping(profile_raw["qualification"], "qualification")
    fixed_binding = _mapping(
        qualification["fixed_safety_report"],
        "qualification.fixed_safety_report",
    )
    expected_decision = {
        "passed": True,
        "system_deployment_profile_eligible": True,
        "formal_model_promotion": "forbidden",
        "policy_only_alias_created": False,
        "outcome": fixed_binding["decision"],
        "expected_system_deployment_profile_path": profile_repository_path,
    }
    for name, expected in expected_decision.items():
        if decision.get(name) != expected:
            raise DeploymentProfileError(
                f"Fixed safety report decision field {name} does not match deployment."
            )
    bindings = _mapping(report.get("source_bindings"), "fixed report source_bindings")
    system = _mapping(profile_raw["system"], "system")
    comparisons = {
        "model": system["model"],
        "shield_profile": system["shield_profile"],
        "runtime_inputs": profile_raw["runtime_inputs"],
        "paired_diagnostic_report": qualification["paired_diagnostic_report"],
        "paired_diagnostic_config": qualification["paired_diagnostic_config"],
        "sweep_manifest": evidence["sweep_manifest"],
        "summary": evidence["summary"],
        "episodes_csv": evidence["episodes_csv"],
    }
    for name, expected in comparisons.items():
        if bindings.get(name) != expected:
            raise DeploymentProfileError(
                f"Fixed safety report binding {name} differs from deployment profile."
            )
    if bindings.get("git_revision") != qualification["git_revision"]:
        raise DeploymentProfileError("Fixed safety report git identity does not match.")
    input_config = _mapping(bindings.get("input_config"), "fixed report input_config")
    git_binding = _mapping(
        input_config.get("git_binding"),
        "fixed report input_config.git_binding",
    )
    if (
        git_binding.get("preregistration_revision")
        != qualification["preregistration_revision"]
    ):
        raise DeploymentProfileError(
            "Fixed safety report preregistration identity does not match."
        )
    if report.get("metrics") != evidence["metrics"] or report.get("gate") != evidence["gate"]:
        raise DeploymentProfileError("Fixed safety report results differ from deployment evidence.")
    fixed = _mapping(report.get("fixed_validation"), "fixed report fixed_validation")
    for name in ("scenario_seed_start", "scenario_seed_end", "episode_count"):
        if fixed.get(name) != evidence[name]:
            raise DeploymentProfileError(
                f"Fixed safety report {name} differs from deployment evidence."
            )


def _validate_paired_report_identity(
    report: Mapping[str, Any],
    *,
    profile_raw: Mapping[str, Any],
) -> None:
    if report.get("schema_version") != 1 or report.get("kind") != (
        "webots_paired_safety_diagnostic"
    ):
        raise DeploymentProfileError("Paired diagnostic report has the wrong schema identity.")
    qualification = _mapping(profile_raw["qualification"], "qualification")
    paired = _mapping(qualification["paired_diagnostic_report"], "paired report binding")
    decision = _mapping(report.get("decision"), "paired diagnostic decision")
    expected_decision = {
        "passed": True,
        "outcome": paired["outcome"],
        "formal_model_promotion": "forbidden",
        "model_alias_created": False,
    }
    for name, expected in expected_decision.items():
        if decision.get(name) != expected:
            raise DeploymentProfileError(
                f"Paired diagnostic decision field {name} does not match deployment."
            )
    bindings = _mapping(report.get("source_bindings"), "paired source_bindings")
    model = _mapping(_mapping(profile_raw["system"], "system")["model"], "system.model")
    expected_model = {"path": model["path"], "sha256": model["sha256"]}
    if bindings.get("model") != expected_model:
        raise DeploymentProfileError("Paired diagnostic model identity does not match.")
    paired_config = _mapping(qualification["paired_diagnostic_config"], "paired config")
    input_config = _mapping(bindings.get("input_config"), "paired input_config")
    if (
        input_config.get("path") != paired_config["path"]
        or input_config.get("sha256") != (paired_config["sha256"])
    ):
        raise DeploymentProfileError("Paired diagnostic config identity does not match.")
    arms = _mapping(bindings.get("arms"), "paired source_bindings.arms")
    candidate = _mapping(arms.get("candidate"), "paired candidate arm")
    candidate_profile = _mapping(candidate.get("profile"), "paired candidate profile")
    shield = _mapping(_mapping(profile_raw["system"], "system")["shield_profile"], "shield")
    for name in ("path", "source_sha256", "semantic_sha256", "result_token", "enabled"):
        if candidate_profile.get(name) != shield[name]:
            raise DeploymentProfileError(
                f"Paired diagnostic candidate shield field {name} does not match."
            )


def _validate_sweep_identity(
    manifest: Mapping[str, Any],
    *,
    profile_raw: Mapping[str, Any],
) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("kind") != (
        "webots_checkpoint_validation_sweep"
    ):
        raise DeploymentProfileError("Validation sweep has the wrong schema identity.")
    if manifest.get("status") != "completed" or manifest.get("split") != "validation":
        raise DeploymentProfileError("Validation sweep is not a completed validation run.")
    request = _mapping(manifest.get("request"), "validation sweep request")
    evidence = _mapping(profile_raw["validation_evidence"], "validation_evidence")
    system = _mapping(profile_raw["system"], "system")
    model = _mapping(system["model"], "system.model")
    shield = _mapping(system["shield_profile"], "system.shield_profile")
    expected_request = {
        "condition": evidence["condition_id"],
        "episodes": evidence["episode_count"],
        "seed_start": evidence["scenario_seed_start"],
        "model_variant": model["variant"],
        "safety_shield_enabled": True,
    }
    for name, expected in expected_request.items():
        if request.get(name) != expected:
            raise DeploymentProfileError(f"Validation sweep request field {name} does not match.")
    if request.get("steps") != [model["checkpoint_step"]]:
        raise DeploymentProfileError("Validation sweep checkpoint identity does not match.")
    request_profile = _mapping(request.get("safety_shield_profile"), "sweep shield profile")
    shield_comparisons = {
        "source_path": shield["path"],
        "source_sha256": shield["source_sha256"],
        "semantic_sha256": shield["semantic_sha256"],
        "result_token": shield["result_token"],
        "enabled": True,
    }
    for name, expected in shield_comparisons.items():
        if request_profile.get(name) != expected:
            raise DeploymentProfileError(f"Validation sweep shield field {name} does not match.")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise DeploymentProfileError("Fixed deployment sweep must contain exactly one run.")
    run = _mapping(runs[0], "validation sweep run")
    expected_run = {
        "status": "completed",
        "step": model["checkpoint_step"],
        "model_path": model["path"],
        "model_sha256": model["sha256"],
        "summary_path": evidence["summary"]["path"],
        "summary_sha256": evidence["summary"]["sha256"],
    }
    for name, expected in expected_run.items():
        if run.get(name) != expected:
            raise DeploymentProfileError(f"Validation sweep run field {name} does not match.")


def _validate_summary_identity(
    summary: Mapping[str, Any],
    *,
    profile_raw: Mapping[str, Any],
) -> None:
    evidence = _mapping(profile_raw["validation_evidence"], "validation_evidence")
    system = _mapping(profile_raw["system"], "system")
    model = _mapping(system["model"], "system.model")
    shield = _mapping(system["shield_profile"], "system.shield_profile")
    if (
        summary.get("split") != "validation"
        or summary.get("episode_count") != (evidence["episode_count"])
    ):
        raise DeploymentProfileError("Validation summary split or episode identity does not match.")
    provenance = _mapping(summary.get("evaluation_provenance"), "summary provenance")
    expected = {
        "model_path": model["path"],
        "model_sha256": model["sha256"],
        "model_variant": model["variant"],
        "condition_id": evidence["condition_id"],
        "scenario_seed_start": evidence["scenario_seed_start"],
        "scenario_seed_count": evidence["episode_count"],
        "safety_shield_enabled": True,
        "episodes_csv_sha256": evidence["episodes_csv"]["sha256"],
    }
    for name, expected_value in expected.items():
        if provenance.get(name) != expected_value:
            raise DeploymentProfileError(
                f"Validation summary provenance field {name} does not match."
            )
    if provenance.get("safety_shield_config") != shield["resolved_full_config"]:
        raise DeploymentProfileError("Validation summary safety-shield config does not match.")


def _validate_episode_evidence(
    episodes_path: Path,
    *,
    condition_id: str,
    seed_start: int,
    episode_count: int,
    metrics: Mapping[str, Any],
) -> None:
    try:
        with episodes_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise DeploymentProfileError(
            f"Cannot parse validation episodes: {episodes_path}"
        ) from error
    if len(rows) != episode_count:
        raise DeploymentProfileError("Validation episodes row count does not match deployment.")
    expected_seeds = set(range(seed_start, seed_start + episode_count))
    try:
        actual_seeds = {int(row["scenario_seed"]) for row in rows}
        conditions = {row["condition_id"] for row in rows}
        outcomes = [row["outcome"] for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise DeploymentProfileError("Validation episodes are missing identity columns.") from error
    if actual_seeds != expected_seeds or conditions != {condition_id}:
        raise DeploymentProfileError("Validation episode seeds or condition identity do not match.")
    counts = {
        "success_count": outcomes.count("success"),
        "collision_count": outcomes.count("collision"),
        "timeout_count": outcomes.count("timeout"),
        "error_count": outcomes.count("error"),
    }
    for name, count in counts.items():
        if metrics[name] != count:
            raise DeploymentProfileError(
                f"Validation episode {name} differs from deployment metrics."
            )


def load_system_deployment_profile(
    path: Path,
    *,
    repository_root: Path,
) -> VerifiedSystemDeploymentProfile:
    """Load and verify one repository-local system deployment profile.

    This generic loader accepts any profile conforming to the version-1 fixed
    safety schema.  Use :func:`load_r2a_system_deployment_profile` for the
    project's audited default, which additionally pins all critical R2a IDs.
    """

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository root does not exist: {root}")
    profile_path = _resolve_profile_path(root, Path(path))
    profile_repository_path = _portable_existing_path(root, profile_path, "deployment profile")
    profile_sha256 = _file_sha256(profile_path)
    digest_path = profile_path.with_suffix(".sha256")
    _verify_digest_sidecar(
        digest_path,
        profile_path=profile_path,
        actual_sha256=profile_sha256,
    )
    raw = _exact_mapping(
        _load_json(profile_path, "system deployment profile"),
        _ROOT_FIELDS,
        "profile",
    )
    if raw["schema_version"] != SYSTEM_DEPLOYMENT_PROFILE_SCHEMA_VERSION:
        raise DeploymentProfileError("System deployment profile schema_version must be 1.")
    if raw["kind"] != "webots_system_deployment_profile":
        raise DeploymentProfileError("System deployment profile kind is invalid.")
    system_id = _string(raw["system_id"], "system_id", pattern=_SAFE_NAME_PATTERN)

    restrictions = _exact_mapping(
        raw["restrictions"],
        {
            "formal_model_promotion",
            "policy_only_alias_created",
            "model_and_shield_must_be_deployed_together",
        },
        "restrictions",
    )
    if restrictions != {
        "formal_model_promotion": "forbidden",
        "policy_only_alias_created": False,
        "model_and_shield_must_be_deployed_together": True,
    }:
        raise DeploymentProfileError(
            "Deployment restrictions do not enforce model-plus-shield use."
        )

    qualification = _exact_mapping(
        raw["qualification"],
        {
            "fixed_safety_report",
            "paired_diagnostic_report",
            "paired_diagnostic_config",
            "git_revision",
            "preregistration_revision",
        },
        "qualification",
    )
    _string(
        qualification["git_revision"],
        "qualification.git_revision",
        pattern=_GIT_REVISION_PATTERN,
    )
    _string(
        qualification["preregistration_revision"],
        "qualification.preregistration_revision",
        pattern=_GIT_REVISION_PATTERN,
    )
    fixed_raw = _exact_mapping(
        qualification["fixed_safety_report"],
        {"path", "sha256", "decision"},
        "qualification.fixed_safety_report",
    )
    _string(fixed_raw["decision"], "qualification.fixed_safety_report.decision")
    paired_report_raw = _exact_mapping(
        qualification["paired_diagnostic_report"],
        {"path", "sha256", "passed", "outcome"},
        "qualification.paired_diagnostic_report",
    )
    if paired_report_raw["passed"] is not True:
        raise DeploymentProfileError("Paired diagnostic qualification must be passed.")
    _string(paired_report_raw["outcome"], "qualification.paired_diagnostic_report.outcome")
    paired_config_raw = _exact_mapping(
        qualification["paired_diagnostic_config"],
        {"path", "sha256"},
        "qualification.paired_diagnostic_config",
    )
    qualification_artifacts = (
        _artifact_binding(
            fixed_raw,
            name="qualification.fixed_safety_report",
            repository_root=root,
        ),
        _artifact_binding(
            paired_report_raw,
            name="qualification.paired_diagnostic_report",
            repository_root=root,
        ),
        _artifact_binding(
            paired_config_raw,
            name="qualification.paired_diagnostic_config",
            repository_root=root,
        ),
    )

    system = _exact_mapping(raw["system"], {"model", "shield_profile"}, "system")
    model_raw = _exact_mapping(
        system["model"],
        {"path", "sha256", "variant", "checkpoint_step"},
        "system.model",
    )
    model = _artifact_binding(model_raw, name="system.model", repository_root=root)
    model_variant = _string(
        model_raw["variant"],
        "system.model.variant",
        pattern=_SAFE_NAME_PATTERN,
    )
    checkpoint_step = _integer(
        model_raw["checkpoint_step"],
        "system.model.checkpoint_step",
        minimum=1,
    )
    if model.path.suffix != ".zip" or not model.path.name.endswith(f"_{checkpoint_step}_steps.zip"):
        raise DeploymentProfileError("Model filename does not match checkpoint_step.")
    if model_variant not in PurePosixPath(model.repository_path).parts:
        raise DeploymentProfileError("Model path does not contain its declared variant.")

    shield_raw = _exact_mapping(
        system["shield_profile"],
        {
            "path",
            "source_sha256",
            "semantic_sha256",
            "result_token",
            "enabled",
            "resolved_full_config",
        },
        "system.shield_profile",
    )
    if shield_raw["enabled"] is not True:
        raise DeploymentProfileError("A deployable system safety shield must be enabled.")
    shield_source_sha256 = _sha256(
        shield_raw["source_sha256"],
        "system.shield_profile.source_sha256",
    )
    shield_semantic_sha256 = _sha256(
        shield_raw["semantic_sha256"],
        "system.shield_profile.semantic_sha256",
    )
    shield_result_token = _string(
        shield_raw["result_token"],
        "system.shield_profile.result_token",
        pattern=_SAFE_NAME_PATTERN,
    )
    shield_portable, shield_path = _resolve_repository_path(
        root,
        shield_raw["path"],
        "system.shield_profile.path",
    )
    shield_profile = load_safety_shield_profile(shield_path, repository_root=root)
    if shield_profile.source_sha256 != shield_source_sha256:
        raise DeploymentIntegrityError("Safety-shield source SHA-256 does not match deployment.")
    if shield_profile.semantic_sha256 != shield_semantic_sha256:
        raise DeploymentProfileError("Safety-shield semantic SHA-256 does not match deployment.")
    if shield_profile.result_token != shield_result_token:
        raise DeploymentProfileError("Safety-shield result token does not match deployment.")
    if (
        shield_profile.enabled is not True
        or _portable_existing_path(
            root,
            shield_profile.source_path,
            "shield profile source",
        )
        != shield_portable
    ):
        raise DeploymentProfileError("Safety-shield source identity does not match deployment.")
    _validate_shield_config(shield_raw["resolved_full_config"], shield_profile)

    validation = _exact_mapping(
        raw["validation_evidence"],
        {
            "condition_id",
            "scenario_seed_start",
            "scenario_seed_end",
            "episode_count",
            "sweep_manifest",
            "summary",
            "episodes_csv",
            "metrics",
            "gate",
        },
        "validation_evidence",
    )
    condition_id = _string(
        validation["condition_id"],
        "validation_evidence.condition_id",
        pattern=_CONDITION_PATTERN,
    )
    seed_start = _integer(validation["scenario_seed_start"], "scenario_seed_start")
    seed_end = _integer(validation["scenario_seed_end"], "scenario_seed_end")
    episode_count = _integer(validation["episode_count"], "episode_count", minimum=1)
    if seed_end != seed_start + episode_count - 1:
        raise DeploymentProfileError("Validation seed window is inconsistent with episode_count.")
    success_rate, collision_rate = _validate_metrics(validation["metrics"], episode_count)
    metrics = _mapping(validation["metrics"], "validation_evidence.metrics")
    _validate_gate(validation["gate"], metrics)
    validation_artifacts = tuple(
        _artifact_binding(
            _exact_mapping(validation[name], {"path", "sha256"}, f"validation_evidence.{name}"),
            name=f"validation_evidence.{name}",
            repository_root=root,
        )
        for name in ("sweep_manifest", "summary", "episodes_csv")
    )

    runtime = _exact_mapping(raw["runtime_inputs"], _RUNTIME_INPUT_FIELDS, "runtime_inputs")
    runtime_artifacts = tuple(
        _artifact_binding(
            _exact_mapping(runtime[name], {"path", "sha256"}, f"runtime_inputs.{name}"),
            name=f"runtime_inputs.{name}",
            repository_root=root,
        )
        for name in sorted(_RUNTIME_INPUT_FIELDS)
    )

    fixed_report = _load_json(qualification_artifacts[0].path, "fixed safety report")
    paired_report = _load_json(qualification_artifacts[1].path, "paired diagnostic report")
    sweep_manifest = _load_json(validation_artifacts[0].path, "validation sweep manifest")
    summary = _load_json(validation_artifacts[1].path, "validation summary")
    _validate_fixed_report_identity(
        fixed_report,
        profile_raw=raw,
        profile_repository_path=profile_repository_path,
    )
    _validate_paired_report_identity(paired_report, profile_raw=raw)
    _validate_sweep_identity(sweep_manifest, profile_raw=raw)
    _validate_summary_identity(summary, profile_raw=raw)
    _validate_episode_evidence(
        validation_artifacts[2].path,
        condition_id=condition_id,
        seed_start=seed_start,
        episode_count=episode_count,
        metrics=metrics,
    )

    return VerifiedSystemDeploymentProfile(
        repository_root=root,
        profile_path=profile_path,
        profile_sha256=profile_sha256,
        profile_digest_path=digest_path,
        system_id=system_id,
        model=model,
        model_variant=model_variant,
        checkpoint_step=checkpoint_step,
        shield_profile=shield_profile,
        shield_result_token=shield_result_token,
        shield_semantic_sha256=shield_semantic_sha256,
        validation_condition_id=condition_id,
        validation_episode_count=episode_count,
        validation_success_rate=success_rate,
        validation_collision_rate=collision_rate,
        qualification_artifacts=qualification_artifacts,
        validation_artifacts=validation_artifacts,
        runtime_artifacts=runtime_artifacts,
    )


def load_r2a_system_deployment_profile(
    *,
    repository_root: Path,
    path: Path = DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH,
) -> VerifiedSystemDeploymentProfile:
    """Load the pinned R2a model-plus-front-brake evaluation component profile."""

    root = Path(repository_root).resolve()
    profile_path = _resolve_profile_path(root, Path(path))
    profile_sha256 = _file_sha256(profile_path)
    if profile_sha256 != AUDITED_R2A_PROFILE_SHA256:
        raise DeploymentIntegrityError(
            "R2a deployment profile SHA-256 does not match the audited artifact: "
            f"expected {AUDITED_R2A_PROFILE_SHA256}, got {profile_sha256}."
        )
    profile = load_system_deployment_profile(path, repository_root=repository_root)
    expected = {
        "system_id": (profile.system_id, AUDITED_R2A_SYSTEM_ID),
        "condition_id": (profile.validation_condition_id, AUDITED_R2A_CONDITION_ID),
        "model_variant": (profile.model_variant, AUDITED_R2A_MODEL_VARIANT),
        "checkpoint_step": (profile.checkpoint_step, AUDITED_R2A_CHECKPOINT_STEP),
        "model_sha256": (profile.model_sha256, AUDITED_R2A_MODEL_SHA256),
        "shield_result_token": (
            profile.shield_result_token,
            AUDITED_R2A_SHIELD_RESULT_TOKEN,
        ),
        "shield_semantic_sha256": (
            profile.shield_semantic_sha256,
            AUDITED_R2A_SHIELD_SEMANTIC_SHA256,
        ),
    }
    mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    if mismatches:
        raise DeploymentProfileError(
            "Deployment profile is valid but is not the audited R2a system; "
            f"mismatched identities={mismatches}."
        )
    return profile
