"""Fail-closed contracts for a one-shot formal evaluation campaign.

The ordinary benchmark helpers intentionally remain convenient for development.
This module is the stricter boundary used immediately before consuming the held-
out test scenarios.  It fixes the experiment split, binds all three independently
trained models to validation evidence, hashes an immutable final lock, and records
the single allowed test attempt in an atomic ledger.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

FORMAL_TRAINING_SEEDS = (11, 22, 33)
FORMAL_VALIDATION_SEEDS = tuple(range(5_000, 5_020))
FORMAL_TEST_SEEDS = tuple(range(10_000, 10_020))

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_CAMPAIGN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ATTEMPT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_FORMAL_MIN_SUCCESS_RATE = 0.75
_FORMAL_MAX_COLLISION_RATE = 0.15
_LOCK_SCHEMA_VERSION = 2
_MODEL_MANIFEST_SCHEMA_VERSION = 2
_MODEL_PROVENANCE_SCHEMA_VERSION = 1
_LEDGER_SCHEMA_VERSION = 2
_CURRICULUM_MANIFEST_SCHEMA_VERSION = 2
_PROTECTED_TEST_PROTOCOL_ID = "webots-g0-g3-20x20-v1"
_FORMAL_EVALUATOR_PATH = "scripts/run_webots_formal_test.py"
_FORMAL_CONTROLLER_PATH = (
    "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
)
_FORMAL_WORLD_PATH = "worlds/webots_ppo_evaluate.wbt"
_FORMAL_SOURCE_ENTRY_PATHS = (
    _FORMAL_EVALUATOR_PATH,
    _FORMAL_CONTROLLER_PATH,
    "requirements-formal-lock.txt",
)
_MODEL_VARIANT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CHECKPOINT_NAME_PATTERN = re.compile(r"_(?P<step>\d+)_steps\.zip$")
_FINAL_MODEL_NAME_PATTERN = re.compile(r"_step(?P<step>\d+)\.zip$")
_RUNTIME_EVIDENCE_FIELDS = {
    "schema_version",
    "executable_path",
    "executable_sha256",
    "expected_version",
    "detected_version",
    "version_probe_stdout",
    "version_probe_stderr",
    "windows_file_version",
    "windows_product_version",
    "signature_status",
    "signature_status_message",
    "signer_subject",
    "signer_thumbprint",
    "probed_at",
}

EvaluationSplit = Literal["validation", "test"]
LedgerStatus = Literal["attempt", "success", "invalidated"]


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON object using a stable UTF-8 canonical representation."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{label} fields differ from the formal schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer.")
    return value


def _finite_rate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between zero and one.")
    return result


def _tuple_of_ints(value: Sequence[int], label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an integer sequence.")
    values = tuple(value)
    if any(type(item) is not int for item in values):
        raise ValueError(f"{label} must contain integers only.")
    return values


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _git_revision(value: Any, label: str = "source_revision") -> str:
    if not isinstance(value, str) or _GIT_REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40- or 64-character Git revision.")
    return value


def _portable_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path.")
    if "\\" in value or ":" in value:
        raise ValueError(f"{label} must use a portable repository-relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."} or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the repository.")
    if path.as_posix() != value:
        raise ValueError(f"{label} must be normalized POSIX syntax.")
    return value


def _resolve_repository_path(repository_root: Path, value: str, label: str) -> Path:
    root = repository_root.resolve()
    portable = _portable_path(value, label)
    resolved = root.joinpath(*PurePosixPath(portable).parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository after resolution.") from error
    return resolved


def _require_prefix(value: str, prefix: tuple[str, ...], label: str) -> None:
    parts = PurePosixPath(value).parts
    if parts[: len(prefix)] != prefix:
        raise ValueError(f"{label} must be below {'/'.join(prefix)}/.")


def _timestamp(now: datetime | None = None) -> str:
    instant = datetime.now(UTC) if now is None else now
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("Ledger timestamps must be timezone-aware.")
    return instant.astimezone(UTC).isoformat(timespec="seconds")


def _model_variant(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _MODEL_VARIANT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be null or one path-safe variant name.")
    return value


def _json_compatible_mapping(value: Any, label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    try:
        json.dumps(mapping, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON-compatible values.") from error
    return mapping


def _runtime_evidence(value: Any, label: str = "runtime_evidence") -> dict[str, Any]:
    evidence = dict(_json_compatible_mapping(value, label))
    _exact_keys(evidence, _RUNTIME_EVIDENCE_FIELDS, label)
    if evidence["schema_version"] != 1:
        raise ValueError(f"{label}.schema_version must be 1.")
    for name in (
        "executable_path",
        "expected_version",
        "detected_version",
        "signature_status",
        "probed_at",
    ):
        if not isinstance(evidence[name], str) or not evidence[name]:
            raise ValueError(f"{label}.{name} must be a non-empty string.")
    _sha256(evidence["executable_sha256"], f"{label}.executable_sha256")
    if evidence["expected_version"] != "R2025a" or evidence["detected_version"] != "R2025a":
        raise ValueError(f"{label} must identify the locked Webots R2025a runtime.")
    for name in (
        "version_probe_stdout",
        "version_probe_stderr",
        "windows_file_version",
        "windows_product_version",
        "signature_status_message",
        "signer_subject",
        "signer_thumbprint",
    ):
        if evidence[name] is not None and not isinstance(evidence[name], str):
            raise ValueError(f"{label}.{name} must be a string or null.")
    return evidence


def protected_test_protocol_payload(
    split_spec: SplitSpec | None = None,
) -> dict[str, Any]:
    """Return the campaign-independent identity of the protected test protocol.

    The consumption identity deliberately excludes campaign names, model choices,
    output paths, and timestamps.  A new campaign that addresses the same held-out
    dataset therefore resolves to the same global ledger.
    """

    spec = split_spec or SplitSpec()
    return {
        "schema_version": 1,
        "protocol_id": _PROTECTED_TEST_PROTOCOL_ID,
        "split": "test",
        "scenario_seeds": list(spec.test_scenario_seeds),
        "conditions": ["G0", "G1", "G2", "G3"],
        "planners": ["reactive", "dwa", "ppo_seed11", "ppo_seed22", "ppo_seed33"],
        "runs": 20,
        "episodes_per_run": 20,
        "episode_count": 400,
    }


def protected_test_consumption_key(split_spec: SplitSpec | None = None) -> str:
    """Return the one-shot key shared by every campaign using the test protocol."""

    return canonical_sha256(protected_test_protocol_payload(split_spec))


def protected_test_ledger_path(split_spec: SplitSpec | None = None) -> str:
    """Return the sole repository-global ledger path for the held-out protocol."""

    key = protected_test_consumption_key(split_spec)
    return f"artifacts/results/formal_ledgers/protected_test_{key}.json"


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """The only split definition accepted by the formal experiment API."""

    training_seeds: tuple[int, ...] = FORMAL_TRAINING_SEEDS
    validation_scenario_seeds: tuple[int, ...] = FORMAL_VALIDATION_SEEDS
    test_scenario_seeds: tuple[int, ...] = FORMAL_TEST_SEEDS
    schema_version: int = 1

    def __post_init__(self) -> None:
        training = _tuple_of_ints(self.training_seeds, "training_seeds")
        validation = _tuple_of_ints(
            self.validation_scenario_seeds, "validation_scenario_seeds"
        )
        test = _tuple_of_ints(self.test_scenario_seeds, "test_scenario_seeds")
        object.__setattr__(self, "training_seeds", training)
        object.__setattr__(self, "validation_scenario_seeds", validation)
        object.__setattr__(self, "test_scenario_seeds", test)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("SplitSpec schema_version must be 1.")
        if training != FORMAL_TRAINING_SEEDS:
            raise ValueError(f"Formal training seeds must be exactly {FORMAL_TRAINING_SEEDS}.")
        if validation != FORMAL_VALIDATION_SEEDS:
            raise ValueError(
                "Formal validation scenario seeds must be exactly 5000 through 5019."
            )
        if test != FORMAL_TEST_SEEDS:
            raise ValueError("Formal test scenario seeds must be exactly 10000 through 10019.")
        if set(training) & set(validation) or set(training) & set(test):
            raise ValueError("Training seeds must be disjoint from evaluation scenario seeds.")
        if set(validation) & set(test):
            raise ValueError("Validation and test scenario seeds must be disjoint.")

    def seeds_for(self, split: EvaluationSplit) -> tuple[int, ...]:
        """Return the immutable seed sequence for one evaluation split."""

        if split == "validation":
            return self.validation_scenario_seeds
        if split == "test":
            return self.test_scenario_seeds
        raise ValueError("Formal evaluation split must be exactly 'validation' or 'test'.")

    def require_training_seeds(self, seeds: Sequence[int]) -> tuple[int, ...]:
        """Reject missing, extra, duplicate, or reordered formal training seeds."""

        actual = _tuple_of_ints(seeds, "training seeds")
        if actual != self.training_seeds:
            raise ValueError(f"Formal training seeds must be exactly {self.training_seeds}.")
        return actual

    def require_evaluation_seeds(
        self,
        split: EvaluationSplit,
        seeds: Sequence[int],
        *,
        confirm_test: bool = False,
    ) -> tuple[int, ...]:
        """Validate an exact formal seed sequence and protect the held-out test split."""

        expected = self.seeds_for(split)
        actual = _tuple_of_ints(seeds, f"{split} scenario seeds")
        if actual != expected:
            raise ValueError(f"Formal {split} scenario seeds must be exactly {expected}.")
        if split == "test" and confirm_test is not True:
            raise PermissionError("Formal test seeds require explicit confirm_test=True.")
        return actual

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "training_seeds": list(self.training_seeds),
            "validation_scenario_seeds": list(self.validation_scenario_seeds),
            "test_scenario_seeds": list(self.test_scenario_seeds),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SplitSpec:
        raw = _mapping(value, "split_spec")
        _exact_keys(
            raw,
            {
                "schema_version",
                "training_seeds",
                "validation_scenario_seeds",
                "test_scenario_seeds",
            },
            "split_spec",
        )
        return cls(
            schema_version=_integer(raw["schema_version"], "split_spec.schema_version"),
            training_seeds=_tuple_of_ints(raw["training_seeds"], "training_seeds"),
            validation_scenario_seeds=_tuple_of_ints(
                raw["validation_scenario_seeds"], "validation_scenario_seeds"
            ),
            test_scenario_seeds=_tuple_of_ints(
                raw["test_scenario_seeds"], "test_scenario_seeds"
            ),
        )


@dataclass(frozen=True, slots=True)
class FormalSourceArtifact:
    """One repository source file bound into the formal execution bundle."""

    path: str
    sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("FormalSourceArtifact schema_version must be 1.")
        object.__setattr__(self, "path", _portable_path(self.path, "source artifact path"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "source artifact sha256"))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FormalSourceArtifact:
        raw = _mapping(value, "formal source artifact")
        _exact_keys(raw, {"schema_version", "path", "sha256"}, "formal source artifact")
        return cls(
            schema_version=_integer(raw["schema_version"], "source.schema_version"),
            path=raw["path"],
            sha256=raw["sha256"],
        )


def discover_formal_source_paths(repository_root: Path) -> tuple[str, ...]:
    """Discover the complete source bundle that can affect a formal execution."""

    root = repository_root.resolve()
    paths = set(_FORMAL_SOURCE_ENTRY_PATHS)
    source_root = root / "src" / "indoor_patrol_rl"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Formal source package does not exist: {source_root}")
    paths.update(path.relative_to(root).as_posix() for path in source_root.rglob("*.py"))
    missing = [value for value in sorted(paths) if not (root / value).is_file()]
    if missing:
        raise FileNotFoundError(f"Formal source bundle is missing files: {missing}.")
    return tuple(sorted(paths))


def build_formal_source_bundle(repository_root: Path) -> tuple[FormalSourceArtifact, ...]:
    """Hash the complete, deterministic source set used by the formal runner."""

    root = repository_root.resolve()
    return tuple(
        FormalSourceArtifact(path=value, sha256=file_sha256(root / value))
        for value in discover_formal_source_paths(root)
    )


def source_bundle_sha256(artifacts: Sequence[FormalSourceArtifact]) -> str:
    """Hash an ordered source bundle without depending on a campaign identifier."""

    return canonical_sha256(
        {"schema_version": 1, "files": [item.to_mapping() for item in artifacts]}
    )


@dataclass(frozen=True, slots=True)
class FormalModelManifest:
    """Immutable references needed to admit one PPO training seed to final test."""

    training_seed: int
    model_path: str
    model_sha256: str
    training_metadata_path: str
    training_metadata_sha256: str
    model_provenance_path: str
    model_provenance_sha256: str
    curriculum_manifest_path: str
    curriculum_manifest_sha256: str
    selection_manifest_path: str
    selection_manifest_sha256: str
    validation_episodes_path: str
    validation_episodes_sha256: str
    training_stage: str
    model_variant: str | None
    training_protocol_sha256: str
    algorithm_config_sha256: str
    schema_version: int = _MODEL_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _MODEL_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                "FormalModelManifest schema_version must be "
                f"{_MODEL_MANIFEST_SCHEMA_VERSION}."
            )
        if type(self.training_seed) is not int or self.training_seed not in FORMAL_TRAINING_SEEDS:
            raise ValueError(f"training_seed must be one of {FORMAL_TRAINING_SEEDS}.")
        for name in (
            "model_path",
            "training_metadata_path",
            "model_provenance_path",
            "curriculum_manifest_path",
            "selection_manifest_path",
            "validation_episodes_path",
        ):
            object.__setattr__(self, name, _portable_path(getattr(self, name), name))
        for name in (
            "model_sha256",
            "training_metadata_sha256",
            "model_provenance_sha256",
            "curriculum_manifest_sha256",
            "selection_manifest_sha256",
            "validation_episodes_sha256",
            "training_protocol_sha256",
            "algorithm_config_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.training_stage != "W2":
            raise ValueError("Formal model training_stage must be exactly W2.")
        object.__setattr__(
            self,
            "model_variant",
            _model_variant(self.model_variant, "model_variant"),
        )
        model_prefix = ("artifacts", "models", "webots", f"seed_{self.training_seed}")
        _require_prefix(self.model_path, model_prefix, "model_path")
        _require_prefix(self.training_metadata_path, model_prefix, "training_metadata_path")
        _require_prefix(self.model_provenance_path, model_prefix, "model_provenance_path")
        _require_prefix(
            self.curriculum_manifest_path,
            ("artifacts", "results"),
            "curriculum_manifest_path",
        )
        _require_prefix(
            self.selection_manifest_path,
            ("artifacts", "results"),
            "selection_manifest_path",
        )
        _require_prefix(
            self.validation_episodes_path,
            ("artifacts", "results"),
            "validation_episodes_path",
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "training_seed": self.training_seed,
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "training_metadata_path": self.training_metadata_path,
            "training_metadata_sha256": self.training_metadata_sha256,
            "model_provenance_path": self.model_provenance_path,
            "model_provenance_sha256": self.model_provenance_sha256,
            "curriculum_manifest_path": self.curriculum_manifest_path,
            "curriculum_manifest_sha256": self.curriculum_manifest_sha256,
            "selection_manifest_path": self.selection_manifest_path,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "validation_episodes_path": self.validation_episodes_path,
            "validation_episodes_sha256": self.validation_episodes_sha256,
            "training_stage": self.training_stage,
            "model_variant": self.model_variant,
            "training_protocol_sha256": self.training_protocol_sha256,
            "algorithm_config_sha256": self.algorithm_config_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FormalModelManifest:
        raw = _mapping(value, "formal model manifest")
        expected = {
            "schema_version",
            "training_seed",
            "model_path",
            "model_sha256",
            "training_metadata_path",
            "training_metadata_sha256",
            "model_provenance_path",
            "model_provenance_sha256",
            "curriculum_manifest_path",
            "curriculum_manifest_sha256",
            "selection_manifest_path",
            "selection_manifest_sha256",
            "validation_episodes_path",
            "validation_episodes_sha256",
            "training_stage",
            "model_variant",
            "training_protocol_sha256",
            "algorithm_config_sha256",
        }
        _exact_keys(raw, expected, "formal model manifest")
        return cls(
            schema_version=_integer(raw["schema_version"], "model.schema_version"),
            training_seed=_integer(raw["training_seed"], "model.training_seed"),
            model_path=raw["model_path"],
            model_sha256=raw["model_sha256"],
            training_metadata_path=raw["training_metadata_path"],
            training_metadata_sha256=raw["training_metadata_sha256"],
            model_provenance_path=raw["model_provenance_path"],
            model_provenance_sha256=raw["model_provenance_sha256"],
            curriculum_manifest_path=raw["curriculum_manifest_path"],
            curriculum_manifest_sha256=raw["curriculum_manifest_sha256"],
            selection_manifest_path=raw["selection_manifest_path"],
            selection_manifest_sha256=raw["selection_manifest_sha256"],
            validation_episodes_path=raw["validation_episodes_path"],
            validation_episodes_sha256=raw["validation_episodes_sha256"],
            training_stage=raw["training_stage"],
            model_variant=raw["model_variant"],
            training_protocol_sha256=raw["training_protocol_sha256"],
            algorithm_config_sha256=raw["algorithm_config_sha256"],
        )


def _verified_file(
    repository_root: Path,
    portable_path: str,
    expected_sha256: str,
    label: str,
) -> Path:
    path = _resolve_repository_path(repository_root, portable_path, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {actual_sha256} != {expected_sha256}.")
    return path


def _selected_candidate(
    selection: Mapping[str, Any], formal: Mapping[str, Any]
) -> Mapping[str, Any]:
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Selection manifest candidates must be a list.")
    matches = [
        _mapping(candidate, "selection candidate")
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("model_path") == formal.get("model_path")
        and candidate.get("model_sha256") == formal.get("model_sha256")
        and candidate.get("rank") == formal.get("rank")
    ]
    if len(matches) != 1:
        raise ValueError("Formal selection must resolve to exactly one candidate record.")
    return matches[0]


_NORMALIZED_PATH_KEYS = {
    "transfer_model",
    "policy_transfer_model",
    "preferred_resume_model",
    "WEBOTS_TRANSFER_MODEL",
    "WEBOTS_POLICY_TRANSFER_MODEL",
    "WEBOTS_RESUME_MODEL",
}
_NORMALIZED_SEED_KEYS = {"seed", "WEBOTS_TRAIN_SEED", "WEBOTS_MODEL_SEED"}


def _normalize_training_protocol_value(value: Any, *, key: str | None = None) -> Any:
    if key in _NORMALIZED_PATH_KEYS:
        return None if value is None else "<PATH>"
    if key in _NORMALIZED_SEED_KEYS:
        return "<SEED>"
    if key is not None and (key.endswith("_at") or key in {"created_at", "updated_at"}):
        return "<TIME>"
    if isinstance(value, dict):
        return {
            str(name): _normalize_training_protocol_value(item, key=str(name))
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_normalize_training_protocol_value(item) for item in value]
    return value


def _curriculum_training_identity(
    raw: Mapping[str, Any], *, training_seed: int
) -> tuple[str | None, str]:
    if raw.get("schema_version") != _CURRICULUM_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Curriculum manifest must use schema_version 2.")
    if _integer(raw.get("seed"), "curriculum.seed") != training_seed:
        raise ValueError("Curriculum manifest seed differs from the model manifest.")
    if raw.get("status") != "completed":
        raise ValueError("Formal curriculum manifest must have status=completed.")
    if raw.get("automatic_split") != "validation" or raw.get("test_split_was_run") is not False:
        raise ValueError("Curriculum evidence must be validation-only and never run test.")
    variant = _model_variant(raw.get("model_variant"), "curriculum.model_variant")
    config = _json_compatible_mapping(raw.get("curriculum_config"), "curriculum_config")
    declared = _sha256(
        raw.get("curriculum_config_sha256"), "curriculum.curriculum_config_sha256"
    )
    if canonical_sha256(config) != declared:
        raise ValueError("Curriculum semantic config does not match its declared SHA-256.")
    if config.get("semantic_schema_version") != 1:
        raise ValueError("Curriculum semantic config schema_version must be 1.")
    if config.get("automatic_split") != "validation":
        raise ValueError("Curriculum semantic config must be validation-only.")
    if _integer(config.get("seed"), "curriculum_config.seed") != training_seed:
        raise ValueError("Curriculum semantic config seed differs from its manifest.")
    if _model_variant(config.get("model_variant"), "curriculum_config.model_variant") != variant:
        raise ValueError("Curriculum model_variant differs from its semantic config.")
    config_stages = config.get("stages")
    if not isinstance(config_stages, list):
        raise ValueError("Curriculum semantic config stages must be a list.")
    if [item.get("id") for item in config_stages if isinstance(item, dict)] != [
        "W0",
        "W1",
        "W2",
    ]:
        raise ValueError("Formal curriculum semantic config must be exactly W0/W1/W2.")
    stages = raw.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Curriculum manifest stages must be a list.")
    stage_by_id = {
        item.get("id"): item for item in stages if isinstance(item, dict)
    }
    if tuple(stage_by_id) != ("W0", "W1", "W2"):
        raise ValueError("Formal curriculum evidence must contain W0/W1/W2 in order.")
    for stage_id in ("W0", "W1", "W2"):
        stage = _mapping(stage_by_id[stage_id], f"curriculum stage {stage_id}")
        validation = _mapping(stage.get("validation"), f"curriculum {stage_id} validation")
        if stage.get("status") != "completed" or validation.get("passed") is not True:
            raise ValueError(f"Curriculum stage {stage_id} must complete its validation gate.")
    normalized = _normalize_training_protocol_value(config)
    return variant, canonical_sha256(_mapping(normalized, "normalized curriculum config"))


_ALGORITHM_CONFIG_KEYS = (
    "mode",
    "model_variant",
    "obstacle_count",
    "obstacle_speed_range",
    "reward_config",
    "reward_safety_distance",
    "proximity_speed_distance",
    "proximity_speed_weight",
    "route_half_lengths",
    "ppo",
    "safety_shield_enabled",
    "safety_shield_intervention_penalty",
    "safety_shield_config",
    "starting_webots_steps",
    "additional_steps_requested",
    "final_model_steps",
)


def _algorithm_config_identity(metadata: Mapping[str, Any]) -> str:
    missing = [name for name in _ALGORITHM_CONFIG_KEYS if name not in metadata]
    if missing:
        raise ValueError(f"Training metadata lacks algorithm configuration: {missing}.")
    config = {name: metadata[name] for name in _ALGORITHM_CONFIG_KEYS}
    _json_compatible_mapping(config, "training algorithm config")
    return canonical_sha256(config)


def _metadata_path(
    value: Any, *, repository_root: Path, label: str
) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path.")
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository_root / candidate).resolve()
    )
    portable = _portable_existing_path(repository_root, resolved, label)
    return resolved, portable


def _selected_artifact_identity(
    selection: Mapping[str, Any], *, training_seed: int
) -> tuple[Mapping[str, Any], Mapping[str, Any], str, int, str | None]:
    if selection.get("schema_version") != 1:
        raise ValueError("Selection manifest schema_version must be 1.")
    if selection.get("selection_status") != "formal_selected":
        raise ValueError("Model manifest requires a gate-passing formal selection.")
    contract = _mapping(selection.get("validation_contract"), "validation contract")
    if contract.get("condition") != "W2":
        raise ValueError("Formal model selection condition must be exactly W2.")
    if _integer(contract.get("seed"), "validation_contract.seed") != training_seed:
        raise ValueError("Selection training seed differs from the model manifest.")
    variant = _model_variant(contract.get("model_variant"), "validation_contract.model_variant")
    if _model_variant(selection.get("model_variant"), "selection.model_variant") != variant:
        raise ValueError("Selection model_variant differs from its validation contract.")
    formal = _mapping(selection.get("formal_selected"), "formal_selected")
    kind = formal.get("kind")
    if kind not in {"checkpoint", "final"}:
        raise ValueError("Formal selection kind must be checkpoint or final.")
    step = _integer(formal.get("step"), "formal_selected.step")
    if step <= 0:
        raise ValueError("formal_selected.step must be positive.")
    if _model_variant(formal.get("model_variant"), "formal_selected.model_variant") != variant:
        raise ValueError("Formal selected model_variant differs from its contract.")
    candidate = _selected_candidate(selection, formal)
    if _model_variant(candidate.get("model_variant"), "candidate.model_variant") != variant:
        raise ValueError("Selected candidate model_variant differs from its contract.")
    return formal, candidate, kind, step, variant


def _build_model_provenance_payload(
    *,
    repository_root: Path,
    training_seed: int,
    model_path: Path,
    metadata_path: Path,
    curriculum_manifest_path: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    model_portable = _portable_existing_path(root, model_path, "model_path")
    metadata_portable = _portable_existing_path(root, metadata_path, "training_metadata_path")
    curriculum_portable = _portable_existing_path(
        root, curriculum_manifest_path, "curriculum_manifest_path"
    )
    metadata = _mapping(yaml.safe_load(metadata_path.read_text(encoding="utf-8")), "metadata")
    if _integer(metadata.get("seed"), "metadata.seed") != training_seed:
        raise ValueError("Training metadata seed differs from the requested model seed.")
    metadata_variant = _model_variant(metadata.get("model_variant"), "metadata.model_variant")
    git = _mapping(metadata.get("git"), "metadata.git")
    revision = _git_revision(git.get("revision"), "metadata.git.revision")
    if git.get("dirty") is not False:
        raise ValueError("Formal model training metadata must record a clean Git worktree.")
    algorithm_sha256 = _algorithm_config_identity(metadata)

    curriculum = _mapping(
        yaml.safe_load(curriculum_manifest_path.read_text(encoding="utf-8")),
        "curriculum manifest",
    )
    curriculum_variant, protocol_sha256 = _curriculum_training_identity(
        curriculum, training_seed=training_seed
    )
    if curriculum_variant != metadata_variant:
        raise ValueError("Training metadata and curriculum model_variant differ.")

    digest = file_sha256(model_path)

    final_model, final_portable = _metadata_path(
        metadata.get("final_model"), repository_root=root, label="metadata.final_model"
    )
    final_sha256 = _sha256(metadata.get("final_model_sha256"), "metadata.final_model_sha256")
    if not final_model.is_file() or file_sha256(final_model) != final_sha256:
        raise ValueError("Training metadata final-model artifact or digest is stale.")
    final_steps = _integer(metadata.get("final_model_steps"), "metadata.final_model_steps")
    checkpoint_match = _CHECKPOINT_NAME_PATTERN.search(model_path.name)
    final_match = _FINAL_MODEL_NAME_PATTERN.search(model_path.name)
    if checkpoint_match is not None:
        kind = "checkpoint"
        step = int(checkpoint_match.group("step"))
    elif final_match is not None:
        kind = "final"
        step = int(final_match.group("step"))
    else:
        raise ValueError("Model artifact filename does not encode a checkpoint/final step.")
    if kind == "final":
        if model_portable != final_portable or digest != final_sha256 or step != final_steps:
            raise ValueError("Selected final model differs from training metadata provenance.")
        expected = _FINAL_MODEL_NAME_PATTERN.search(model_path.name)
    else:
        expected = _CHECKPOINT_NAME_PATTERN.search(model_path.name)
        if model_path.parent.name != "checkpoints" or step > final_steps:
            raise ValueError("Selected checkpoint is outside its training checkpoint lineage.")
    if expected is None or int(expected.group("step")) != step:
        raise ValueError("Selected model filename does not encode its selected training step.")
    seed_root = root / "artifacts" / "models" / "webots" / f"seed_{training_seed}"
    variant_root = seed_root / metadata_variant if metadata_variant is not None else seed_root
    try:
        model_path.resolve().relative_to(variant_root.resolve())
        final_model.resolve().relative_to(variant_root.resolve())
    except ValueError as error:
        raise ValueError("Model provenance escapes its seed/variant artifact directory.") from error

    return {
        "schema_version": _MODEL_PROVENANCE_SCHEMA_VERSION,
        "artifact_kind": kind,
        "training_seed": training_seed,
        "training_stage": "W2",
        "model_variant": metadata_variant,
        "model_step": step,
        "model_path": model_portable,
        "model_sha256": digest,
        "training_metadata_path": metadata_portable,
        "training_metadata_sha256": file_sha256(metadata_path),
        "curriculum_manifest_path": curriculum_portable,
        "training_protocol_sha256": protocol_sha256,
        "algorithm_config_sha256": algorithm_sha256,
        "source_revision": revision,
        "source_dirty": False,
    }


def write_or_validate_model_provenance(
    *,
    repository_root: Path,
    training_seed: int,
    model_path: Path,
    metadata_path: Path,
    curriculum_manifest_path: Path,
    provenance_path: Path | None = None,
    allow_create: bool = False,
) -> tuple[Path, Mapping[str, Any]]:
    """Validate a training-time sidecar, optionally creating it for the trainer."""

    payload = _build_model_provenance_payload(
        repository_root=repository_root,
        training_seed=training_seed,
        model_path=model_path,
        metadata_path=metadata_path,
        curriculum_manifest_path=curriculum_manifest_path,
    )
    destination = provenance_path or model_path.with_suffix(
        model_path.suffix + ".formal-provenance.json"
    )
    _portable_existing_path(repository_root, destination, "model_provenance_path")
    if destination.exists():
        existing = _mapping(
            json.loads(destination.read_text(encoding="utf-8")), "model provenance"
        )
        if existing != payload:
            raise ValueError("Existing model provenance sidecar conflicts with its artifacts.")
    elif allow_create:
        _exclusive_write(destination, payload)
    else:
        raise FileNotFoundError(
            "Training-time model provenance sidecar is required before final-lock "
            f"creation: {destination}"
        )
    return destination, payload


def _require_curriculum_provenance_listing(
    curriculum: Mapping[str, Any],
    *,
    model_path: str,
    model_sha256: str,
    provenance_path: str,
    provenance_sha256: str,
) -> None:
    stages = curriculum.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Curriculum manifest stages must be a list.")
    w2 = [item for item in stages if isinstance(item, dict) and item.get("id") == "W2"]
    if len(w2) != 1:
        raise ValueError("Curriculum manifest must contain exactly one W2 stage.")
    records = w2[0].get("artifact_provenance")
    if not isinstance(records, list):
        raise ValueError("Curriculum W2 must list its training-time artifact provenance.")
    expected = {
        "model_path": model_path,
        "model_sha256": model_sha256,
        "provenance_path": provenance_path,
        "provenance_sha256": provenance_sha256,
    }
    if sum(record == expected for record in records) != 1:
        raise ValueError(
            "Selected model provenance is not uniquely recorded by the W2 curriculum."
        )


def _validate_validation_csv(
    episodes_path: Path,
    *,
    expected_planner: str,
    expected_condition: str,
    split_spec: SplitSpec,
) -> tuple[float, float]:
    with episodes_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"split", "scenario_seed", "planner", "condition_id"}
    if len(rows) != len(split_spec.validation_scenario_seeds):
        raise ValueError("Validation episodes CSV must contain exactly 20 rows.")
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Validation episodes CSV is missing columns: {sorted(required)}.")
    try:
        scenario_seeds = tuple(int(row["scenario_seed"]) for row in rows)
    except ValueError as error:
        raise ValueError("Validation episodes CSV contains a non-integer scenario seed.") from error
    split_spec.require_evaluation_seeds("validation", scenario_seeds)
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("Every validation CSV row must be labelled validation.")
    if any(row["planner"] != expected_planner for row in rows):
        raise ValueError("Validation CSV planner does not match the selected candidate.")
    if any(row["condition_id"] != expected_condition for row in rows):
        raise ValueError("Validation CSV condition does not match the selection contract.")
    outcomes = [row.get("outcome", "").casefold() for row in rows]
    if all(outcome in {"success", "collision", "timeout", "error"} for outcome in outcomes):
        success_count = outcomes.count("success")
        collision_count = outcomes.count("collision")
    elif {"is_success", "collision"}.issubset(rows[0]):
        truthy = {"1", "true", "yes"}
        falsey = {"0", "false", "no"}

        def parsed_boolean(value: str, label: str) -> bool:
            normalized = value.casefold()
            if normalized in truthy:
                return True
            if normalized in falsey:
                return False
            raise ValueError(f"Validation CSV {label} contains an invalid boolean.")

        success_count = sum(
            parsed_boolean(row["is_success"], "is_success") for row in rows
        )
        collision_count = sum(
            parsed_boolean(row["collision"], "collision") for row in rows
        )
    else:
        raise ValueError(
            "Validation CSV must provide outcome or is_success/collision evidence."
        )
    denominator = len(rows)
    return success_count / denominator, collision_count / denominator


def validate_model_manifest(
    manifest: FormalModelManifest,
    *,
    repository_root: Path,
    split_spec: SplitSpec | None = None,
) -> tuple[str, str | None, str, str]:
    """Verify one model and return revision, variant, protocol, and algorithm IDs."""

    spec = split_spec or SplitSpec()
    model_path = _verified_file(
        repository_root, manifest.model_path, manifest.model_sha256, "formal model"
    )
    metadata_path = _verified_file(
        repository_root,
        manifest.training_metadata_path,
        manifest.training_metadata_sha256,
        "training metadata",
    )
    provenance_path = _verified_file(
        repository_root,
        manifest.model_provenance_path,
        manifest.model_provenance_sha256,
        "model provenance",
    )
    curriculum_path = _verified_file(
        repository_root,
        manifest.curriculum_manifest_path,
        manifest.curriculum_manifest_sha256,
        "curriculum manifest",
    )
    selection_path = _verified_file(
        repository_root,
        manifest.selection_manifest_path,
        manifest.selection_manifest_sha256,
        "selection manifest",
    )
    episodes_path = _verified_file(
        repository_root,
        manifest.validation_episodes_path,
        manifest.validation_episodes_sha256,
        "validation episodes",
    )

    expected_provenance = _build_model_provenance_payload(
        repository_root=repository_root,
        training_seed=manifest.training_seed,
        model_path=model_path,
        metadata_path=metadata_path,
        curriculum_manifest_path=curriculum_path,
    )
    provenance = _mapping(
        json.loads(provenance_path.read_text(encoding="utf-8")), "model provenance"
    )
    if provenance != expected_provenance:
        raise ValueError("Model provenance sidecar differs from the verified artifact chain.")
    curriculum = _mapping(
        yaml.safe_load(curriculum_path.read_text(encoding="utf-8")),
        "curriculum manifest",
    )
    _require_curriculum_provenance_listing(
        curriculum,
        model_path=manifest.model_path,
        model_sha256=manifest.model_sha256,
        provenance_path=manifest.model_provenance_path,
        provenance_sha256=manifest.model_provenance_sha256,
    )
    for name, expected_value in (
        ("training_stage", manifest.training_stage),
        ("model_variant", manifest.model_variant),
        ("training_protocol_sha256", manifest.training_protocol_sha256),
        ("algorithm_config_sha256", manifest.algorithm_config_sha256),
    ):
        if provenance.get(name) != expected_value:
            raise ValueError(f"Model provenance {name} differs from the final lock.")

    metadata = _mapping(yaml.safe_load(metadata_path.read_text(encoding="utf-8")), "metadata")
    if _integer(metadata.get("seed"), "metadata.seed") != manifest.training_seed:
        raise ValueError("Training metadata seed does not match the model manifest.")
    git = _mapping(metadata.get("git"), "metadata.git")
    revision = _git_revision(git.get("revision"), "metadata.git.revision")
    if git.get("dirty") is not False:
        raise ValueError("Formal model training metadata must record a clean Git worktree.")

    selection = _mapping(
        json.loads(selection_path.read_text(encoding="utf-8")), "selection manifest"
    )
    if selection.get("schema_version") != 1:
        raise ValueError("Selection manifest schema_version must be 1.")
    if selection.get("selection_status") != "formal_selected":
        raise ValueError("Model manifest requires a gate-passing formal selection.")
    contract = _mapping(selection.get("validation_contract"), "validation contract")
    if contract.get("split") != "validation":
        raise ValueError("Model selection must use the validation split.")
    if _integer(contract.get("seed"), "validation_contract.seed") != manifest.training_seed:
        raise ValueError("Selection training seed does not match the model manifest.")
    if contract.get("condition") != manifest.training_stage:
        raise ValueError("Formal selection must validate the W2 training stage.")
    if contract.get("model_variant") != manifest.model_variant:
        raise ValueError("Selection model_variant differs from the final lock.")
    if _integer(contract.get("episodes"), "validation_contract.episodes") != len(
        spec.validation_scenario_seeds
    ):
        raise ValueError("Formal selection must contain exactly 20 validation episodes.")
    condition = contract.get("condition")
    if not isinstance(condition, str) or not condition:
        raise ValueError("Selection validation contract must identify one condition.")

    gate = _mapping(selection.get("gate"), "selection gate")
    if gate.get("requires_model_file") is not True:
        raise ValueError("Formal selection gate must require the model file.")
    minimum_success = _finite_rate(gate.get("min_success_rate"), "gate.min_success_rate")
    maximum_collision = _finite_rate(
        gate.get("max_collision_rate"), "gate.max_collision_rate"
    )
    if minimum_success < _FORMAL_MIN_SUCCESS_RATE:
        raise ValueError("Formal selection success gate is weaker than 0.75.")
    if maximum_collision > _FORMAL_MAX_COLLISION_RATE:
        raise ValueError("Formal selection collision gate is weaker than 0.15.")

    formal = _mapping(selection.get("formal_selected"), "formal_selected")
    formal_rank = _integer(formal.get("rank"), "formal_selected.rank")
    if formal_rank <= 0:
        raise ValueError("formal_selected.rank must be positive.")
    if formal.get("model_path") != manifest.model_path:
        raise ValueError("Formal selection model path does not match the model manifest.")
    if formal.get("model_sha256") != manifest.model_sha256:
        raise ValueError("Formal selection model digest does not match the model manifest.")
    if formal.get("model_variant") != manifest.model_variant:
        raise ValueError("Formal selected model_variant differs from the final lock.")
    if formal.get("gate_passed") is not True:
        raise ValueError("Formal selected model must have gate_passed=true.")
    if _integer(formal.get("episodes"), "formal_selected.episodes") != len(
        spec.validation_scenario_seeds
    ):
        raise ValueError("Formal selected model must contain exactly 20 episodes.")
    success_rate = _finite_rate(formal.get("success_rate"), "formal_selected.success_rate")
    collision_rate = _finite_rate(
        formal.get("collision_rate"), "formal_selected.collision_rate"
    )
    if success_rate < minimum_success or collision_rate > maximum_collision:
        raise ValueError("Formal selected model metrics do not satisfy its gate.")
    planner = formal.get("planner")
    if not isinstance(planner, str) or not planner:
        raise ValueError("Formal selected model must identify one planner.")

    candidate = _selected_candidate(selection, formal)
    if candidate.get("gate_passed") is not True:
        raise ValueError("Selected candidate record must have gate_passed=true.")
    if candidate.get("model_variant") != manifest.model_variant:
        raise ValueError("Selected candidate model_variant differs from the final lock.")
    for name, expected_value in (
        ("planner", planner),
        ("episodes", len(spec.validation_scenario_seeds)),
        ("success_rate", success_rate),
        ("collision_rate", collision_rate),
    ):
        if candidate.get(name) != expected_value:
            raise ValueError(f"Selected candidate and formal_selected disagree on {name}.")
    summary_path_value = _portable_path(candidate.get("summary_path"), "candidate.summary_path")
    summary_sha256 = _sha256(candidate.get("summary_sha256"), "candidate.summary_sha256")
    summary_path = _verified_file(
        repository_root, summary_path_value, summary_sha256, "validation summary"
    )
    if episodes_path.parent != summary_path.parent:
        raise ValueError(
            "Validation episodes and selected summary must share one result directory."
        )

    summary = _mapping(json.loads(summary_path.read_text(encoding="utf-8")), "summary")
    if summary.get("split") != "validation":
        raise ValueError("Selected summary must be labelled validation.")
    if _integer(summary.get("episode_count"), "summary.episode_count") != len(
        spec.validation_scenario_seeds
    ):
        raise ValueError("Selected summary must aggregate exactly 20 episodes.")
    if summary.get("planners") != [planner] or summary.get("conditions") != [condition]:
        raise ValueError("Selected summary planner or condition differs from its contract.")
    provenance = _mapping(summary.get("evaluation_provenance"), "evaluation_provenance")
    if provenance.get("model_path") != manifest.model_path:
        raise ValueError("Validation summary provenance model path differs from the lock.")
    if provenance.get("model_sha256") != manifest.model_sha256:
        raise ValueError("Validation summary provenance model digest differs from the lock.")
    if _integer(provenance.get("training_seed"), "provenance.training_seed") != (
        manifest.training_seed
    ):
        raise ValueError("Validation summary provenance training seed differs from the lock.")
    if provenance.get("model_variant") != manifest.model_variant:
        raise ValueError("Validation summary provenance model_variant differs from the lock.")
    csv_success_rate, csv_collision_rate = _validate_validation_csv(
        episodes_path,
        expected_planner=planner,
        expected_condition=condition,
        split_spec=spec,
    )
    if csv_success_rate != success_rate or csv_collision_rate != collision_rate:
        raise ValueError("Formal selected metrics do not match the raw validation CSV.")
    if file_sha256(model_path) != manifest.model_sha256:
        raise ValueError("Formal model changed during manifest validation.")
    return (
        revision,
        manifest.model_variant,
        manifest.training_protocol_sha256,
        manifest.algorithm_config_sha256,
    )


@dataclass(frozen=True, slots=True)
class FinalLock:
    """Fully hashed, immutable input contract for exactly one final campaign."""

    campaign_id: str
    source_revision: str
    source_dirty: bool
    evaluation_config_path: str
    evaluation_config_sha256: str
    world_path: str
    world_sha256: str
    evaluator_path: str
    evaluator_sha256: str
    controller_path: str
    controller_sha256: str
    source_bundle: tuple[FormalSourceArtifact, ...]
    source_bundle_sha256: str
    test_consumption_key: str
    output_directory: str
    test_ledger_path: str
    model_manifests: tuple[FormalModelManifest, ...]
    split_spec: SplitSpec = field(default_factory=SplitSpec)
    schema_version: int = _LOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _LOCK_SCHEMA_VERSION:
            raise ValueError(f"FinalLock schema_version must be {_LOCK_SCHEMA_VERSION}.")
        if not isinstance(self.campaign_id, str) or _CAMPAIGN_PATTERN.fullmatch(
            self.campaign_id
        ) is None:
            raise ValueError("campaign_id must be a path-safe identifier up to 128 characters.")
        object.__setattr__(self, "source_revision", _git_revision(self.source_revision))
        if self.source_dirty is not False:
            raise ValueError("A final lock may only be created from a clean source revision.")
        object.__setattr__(
            self,
            "evaluation_config_path",
            _portable_path(self.evaluation_config_path, "evaluation_config_path"),
        )
        object.__setattr__(
            self,
            "evaluation_config_sha256",
            _sha256(self.evaluation_config_sha256, "evaluation_config_sha256"),
        )
        for path_name in ("world_path", "evaluator_path", "controller_path"):
            object.__setattr__(
                self,
                path_name,
                _portable_path(getattr(self, path_name), path_name),
            )
        if self.world_path != _FORMAL_WORLD_PATH:
            raise ValueError(f"world_path must be exactly {_FORMAL_WORLD_PATH}.")
        if self.evaluator_path != _FORMAL_EVALUATOR_PATH:
            raise ValueError(f"evaluator_path must be exactly {_FORMAL_EVALUATOR_PATH}.")
        if self.controller_path != _FORMAL_CONTROLLER_PATH:
            raise ValueError(f"controller_path must be exactly {_FORMAL_CONTROLLER_PATH}.")
        for digest_name in (
            "world_sha256",
            "evaluator_sha256",
            "controller_sha256",
            "source_bundle_sha256",
            "test_consumption_key",
        ):
            object.__setattr__(
                self,
                digest_name,
                _sha256(getattr(self, digest_name), digest_name),
            )
        object.__setattr__(
            self,
            "output_directory",
            _portable_path(self.output_directory, "output_directory"),
        )
        object.__setattr__(
            self,
            "test_ledger_path",
            _portable_path(self.test_ledger_path, "test_ledger_path"),
        )
        _require_prefix(self.output_directory, ("artifacts", "results"), "output_directory")
        _require_prefix(self.test_ledger_path, ("artifacts", "results"), "test_ledger_path")
        output_parts = PurePosixPath(self.output_directory).parts
        ledger_parts = PurePosixPath(self.test_ledger_path).parts
        if ledger_parts[: len(output_parts)] == output_parts:
            raise ValueError("test_ledger_path must be outside the final output directory.")
        if not isinstance(self.split_spec, SplitSpec):
            raise ValueError("split_spec must be a canonical SplitSpec instance.")
        expected_consumption_key = protected_test_consumption_key(self.split_spec)
        if self.test_consumption_key != expected_consumption_key:
            raise ValueError(
                "test_consumption_key must be derived only from the protected test protocol."
            )
        expected_ledger_path = protected_test_ledger_path(self.split_spec)
        if self.test_ledger_path != expected_ledger_path:
            raise ValueError(
                "test_ledger_path must be the repository-global protected-test ledger: "
                f"{expected_ledger_path}."
            )
        bundle = tuple(self.source_bundle)
        if any(not isinstance(item, FormalSourceArtifact) for item in bundle):
            raise ValueError("source_bundle must contain FormalSourceArtifact values.")
        bundle_paths = tuple(item.path for item in bundle)
        if bundle_paths != tuple(sorted(set(bundle_paths))):
            raise ValueError("source_bundle paths must be unique and sorted.")
        bundle_by_path = {item.path: item for item in bundle}
        for path_name, sha_name in (
            (self.evaluator_path, self.evaluator_sha256),
            (self.controller_path, self.controller_sha256),
        ):
            if path_name not in bundle_by_path or bundle_by_path[path_name].sha256 != sha_name:
                raise ValueError("Direct evaluator/controller digests must match source_bundle.")
        if source_bundle_sha256(bundle) != self.source_bundle_sha256:
            raise ValueError("source_bundle_sha256 differs from the canonical source bundle.")
        object.__setattr__(self, "source_bundle", bundle)
        manifests = tuple(self.model_manifests)
        if any(not isinstance(item, FormalModelManifest) for item in manifests):
            raise ValueError("model_manifests must contain FormalModelManifest values.")
        seeds = tuple(item.training_seed for item in manifests)
        self.split_spec.require_training_seeds(seeds)
        if len({item.training_stage for item in manifests}) != 1:
            raise ValueError("All formal models must use the same W2 training stage.")
        if len({item.model_variant for item in manifests}) != 1:
            raise ValueError("All formal models must use the same model_variant.")
        if len({item.training_protocol_sha256 for item in manifests}) != 1:
            raise ValueError(
                "All formal models must share one normalized training protocol fingerprint."
            )
        if len({item.algorithm_config_sha256 for item in manifests}) != 1:
            raise ValueError("All formal models must share one algorithm configuration.")
        object.__setattr__(self, "model_manifests", manifests)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "source_dirty": self.source_dirty,
            "split_spec": self.split_spec.to_mapping(),
            "evaluation_config_path": self.evaluation_config_path,
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "world_path": self.world_path,
            "world_sha256": self.world_sha256,
            "evaluator_path": self.evaluator_path,
            "evaluator_sha256": self.evaluator_sha256,
            "controller_path": self.controller_path,
            "controller_sha256": self.controller_sha256,
            "source_bundle": [item.to_mapping() for item in self.source_bundle],
            "source_bundle_sha256": self.source_bundle_sha256,
            "test_consumption_key": self.test_consumption_key,
            "output_directory": self.output_directory,
            "test_ledger_path": self.test_ledger_path,
            "model_manifests": [item.to_mapping() for item in self.model_manifests],
        }

    @property
    def lock_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, Any]:
        return {**self.payload(), "lock_sha256": self.lock_sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FinalLock:
        raw = _mapping(value, "final lock")
        expected = {
            "schema_version",
            "campaign_id",
            "source_revision",
            "source_dirty",
            "split_spec",
            "evaluation_config_path",
            "evaluation_config_sha256",
            "world_path",
            "world_sha256",
            "evaluator_path",
            "evaluator_sha256",
            "controller_path",
            "controller_sha256",
            "source_bundle",
            "source_bundle_sha256",
            "test_consumption_key",
            "output_directory",
            "test_ledger_path",
            "model_manifests",
            "lock_sha256",
        }
        _exact_keys(raw, expected, "final lock")
        model_values = raw["model_manifests"]
        if not isinstance(model_values, list):
            raise ValueError("final lock model_manifests must be a list.")
        source_values = raw["source_bundle"]
        if not isinstance(source_values, list):
            raise ValueError("final lock source_bundle must be a list.")
        lock = cls(
            schema_version=_integer(raw["schema_version"], "final_lock.schema_version"),
            campaign_id=raw["campaign_id"],
            source_revision=raw["source_revision"],
            source_dirty=raw["source_dirty"],
            split_spec=SplitSpec.from_mapping(_mapping(raw["split_spec"], "split_spec")),
            evaluation_config_path=raw["evaluation_config_path"],
            evaluation_config_sha256=raw["evaluation_config_sha256"],
            world_path=raw["world_path"],
            world_sha256=raw["world_sha256"],
            evaluator_path=raw["evaluator_path"],
            evaluator_sha256=raw["evaluator_sha256"],
            controller_path=raw["controller_path"],
            controller_sha256=raw["controller_sha256"],
            source_bundle=tuple(
                FormalSourceArtifact.from_mapping(_mapping(item, "source artifact"))
                for item in source_values
            ),
            source_bundle_sha256=raw["source_bundle_sha256"],
            test_consumption_key=raw["test_consumption_key"],
            output_directory=raw["output_directory"],
            test_ledger_path=raw["test_ledger_path"],
            model_manifests=tuple(
                FormalModelManifest.from_mapping(_mapping(item, "model manifest"))
                for item in model_values
            ),
        )
        declared = _sha256(raw["lock_sha256"], "final_lock.lock_sha256")
        if declared != lock.lock_sha256:
            raise ValueError("Final lock SHA-256 does not match its canonical payload.")
        return lock

    def validate_files(self, repository_root: Path) -> None:
        """Verify the evaluation config and all three model evidence chains."""

        config_path = _verified_file(
            repository_root,
            self.evaluation_config_path,
            self.evaluation_config_sha256,
            "evaluation config",
        )
        world_path = _verified_file(
            repository_root, self.world_path, self.world_sha256, "formal world"
        )
        evaluator_path = _verified_file(
            repository_root,
            self.evaluator_path,
            self.evaluator_sha256,
            "formal evaluator",
        )
        controller_path = _verified_file(
            repository_root,
            self.controller_path,
            self.controller_sha256,
            "formal controller",
        )
        config_raw = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "formal evaluation config",
        )
        if config_raw.get("world") != self.world_path:
            raise ValueError("Formal evaluation config world differs from the final lock.")
        if not config_path.is_file() or not world_path.is_file():
            raise FileNotFoundError("Formal evaluation config or world is unavailable.")
        if not evaluator_path.is_file() or not controller_path.is_file():
            raise FileNotFoundError("Formal evaluator or controller is unavailable.")
        discovered = discover_formal_source_paths(repository_root)
        if tuple(item.path for item in self.source_bundle) != discovered:
            raise ValueError("Formal source bundle membership drifted after lock creation.")
        for artifact in self.source_bundle:
            _verified_file(
                repository_root,
                artifact.path,
                artifact.sha256,
                f"formal source artifact {artifact.path}",
            )
        identities = {
            validate_model_manifest(
                manifest,
                repository_root=repository_root,
                split_spec=self.split_spec,
            )
            for manifest in self.model_manifests
        }
        revisions = {item[0] for item in identities}
        if revisions != {self.source_revision}:
            raise ValueError(
                "All formal model training metadata must use the final lock source revision."
            )
        expected_identity = {
            (
                self.source_revision,
                self.model_manifests[0].model_variant,
                self.model_manifests[0].training_protocol_sha256,
                self.model_manifests[0].algorithm_config_sha256,
            )
        }
        if identities != expected_identity:
            raise ValueError(
                "All formal models must share source, variant, normalized protocol, "
                "and algorithm configuration."
            )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _exclusive_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _ledger_mutex(ledger_path: Path):
    mutex = ledger_path.with_name(f"{ledger_path.name}.lock")
    mutex.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(mutex, flags, 0o600)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        mutex.unlink(missing_ok=True)


def write_final_lock(
    path: Path,
    final_lock: FinalLock,
    *,
    repository_root: Path,
) -> Path:
    """Validate and create an immutable lock file without overwriting an old lock."""

    final_lock.validate_files(repository_root)
    _exclusive_write(path, final_lock.to_mapping())
    return path


def load_final_lock(
    path: Path,
    *,
    repository_root: Path,
    verify_files: bool = True,
) -> FinalLock:
    """Load a final lock, verify its canonical digest, and optionally rehash inputs."""

    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "final lock")
    final_lock = FinalLock.from_mapping(raw)
    if verify_files:
        final_lock.validate_files(repository_root)
    return final_lock


def _ledger_payload_sha256(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "ledger_sha256"}
    return canonical_sha256(body)


def _with_ledger_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["ledger_sha256"] = _ledger_payload_sha256(result)
    return result


def _validate_ledger(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "test_consumption_key",
        "campaign_id",
        "final_lock_path",
        "final_lock_sha256",
        "status",
        "attempt",
        "success",
        "invalidation",
        "ledger_sha256",
    }
    _exact_keys(raw, expected, "test ledger")
    if _integer(raw["schema_version"], "ledger.schema_version") != _LEDGER_SCHEMA_VERSION:
        raise ValueError(f"Test ledger schema_version must be {_LEDGER_SCHEMA_VERSION}.")
    _sha256(raw["test_consumption_key"], "ledger.test_consumption_key")
    if not isinstance(raw["campaign_id"], str) or _CAMPAIGN_PATTERN.fullmatch(
        raw["campaign_id"]
    ) is None:
        raise ValueError("Test ledger campaign_id is invalid.")
    _portable_path(raw["final_lock_path"], "ledger.final_lock_path")
    _sha256(raw["final_lock_sha256"], "ledger.final_lock_sha256")
    _sha256(raw["ledger_sha256"], "ledger.ledger_sha256")
    if raw["ledger_sha256"] != _ledger_payload_sha256(raw):
        raise ValueError("Test ledger SHA-256 does not match its canonical payload.")
    status = raw["status"]
    if status not in {"attempt", "success", "invalidated"}:
        raise ValueError("Test ledger status is invalid.")
    attempt = _mapping(raw["attempt"], "ledger.attempt")
    _exact_keys(
        attempt,
        {
            "attempt_id",
            "started_at",
            "output_directory",
            "runtime_evidence",
            "runtime_evidence_sha256",
        },
        "attempt",
    )
    if not isinstance(attempt["attempt_id"], str) or _ATTEMPT_PATTERN.fullmatch(
        attempt["attempt_id"]
    ) is None:
        raise ValueError("Test ledger attempt_id is invalid.")
    if not isinstance(attempt["started_at"], str) or not attempt["started_at"]:
        raise ValueError("Test ledger started_at is invalid.")
    _portable_path(attempt["output_directory"], "attempt.output_directory")
    runtime_evidence = _runtime_evidence(
        attempt["runtime_evidence"], "attempt.runtime_evidence"
    )
    runtime_evidence_sha256 = _sha256(
        attempt["runtime_evidence_sha256"], "attempt.runtime_evidence_sha256"
    )
    if canonical_sha256(runtime_evidence) != runtime_evidence_sha256:
        raise ValueError("Attempt runtime evidence SHA-256 differs from its payload.")
    success = raw["success"]
    invalidation = raw["invalidation"]
    if status == "attempt" and (success is not None or invalidation is not None):
        raise ValueError("An attempt ledger cannot already contain a terminal transition.")
    if status == "success":
        if not isinstance(success, dict) or invalidation is not None:
            raise ValueError("A successful ledger requires success evidence only.")
    if status == "invalidated" and not isinstance(invalidation, dict):
        raise ValueError("An invalidated ledger requires invalidation evidence.")
    if isinstance(success, dict):
        _exact_keys(
            success,
            {"finished_at", "result_manifest_path", "result_manifest_sha256"},
            "ledger.success",
        )
        _portable_path(success["result_manifest_path"], "success.result_manifest_path")
        _sha256(success["result_manifest_sha256"], "success.result_manifest_sha256")
    if isinstance(invalidation, dict):
        _exact_keys(
            invalidation,
            {"invalidated_at", "previous_status", "reason"},
            "ledger.invalidation",
        )
        if invalidation["previous_status"] not in {"attempt", "success"}:
            raise ValueError("Invalidation previous_status must be attempt or success.")
        if not isinstance(invalidation["reason"], str) or not invalidation["reason"].strip():
            raise ValueError("Invalidation reason must be non-empty.")
    return dict(raw)


def load_test_ledger(path: Path) -> dict[str, Any]:
    """Load and hash-check one formal test ledger."""

    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "test ledger")
    return _validate_ledger(raw)


def _portable_existing_path(repository_root: Path, path: Path, label: str) -> str:
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside the repository.") from error
    return _portable_path(relative.as_posix(), label)


def _load_lock_for_transition(
    repository_root: Path,
    final_lock_path: Path,
    *,
    verify_files: bool = True,
) -> tuple[FinalLock, str, Path]:
    portable_lock_path = _portable_existing_path(
        repository_root, final_lock_path, "final_lock_path"
    )
    final_lock = load_final_lock(
        final_lock_path,
        repository_root=repository_root,
        verify_files=verify_files,
    )
    ledger_path = _resolve_repository_path(
        repository_root, final_lock.test_ledger_path, "test_ledger_path"
    )
    return final_lock, portable_lock_path, ledger_path


def begin_test_attempt(
    *,
    repository_root: Path,
    final_lock_path: Path,
    confirm_test: bool,
    runtime_evidence: Mapping[str, Any],
    attempt_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Atomically consume the one allowed test attempt for a final lock."""

    if confirm_test is not True:
        raise PermissionError("Beginning the held-out test requires confirm_test=True.")
    final_lock, portable_lock_path, ledger_path = _load_lock_for_transition(
        repository_root, final_lock_path
    )
    final_lock.split_spec.require_evaluation_seeds(
        "test", final_lock.split_spec.test_scenario_seeds, confirm_test=True
    )
    identifier = attempt_id or uuid.uuid4().hex
    if not isinstance(identifier, str) or _ATTEMPT_PATTERN.fullmatch(identifier) is None:
        raise ValueError("attempt_id must be a path-safe identifier up to 128 characters.")
    validated_runtime_evidence = _runtime_evidence(runtime_evidence)
    output_directory = _resolve_repository_path(
        repository_root, final_lock.output_directory, "output_directory"
    )
    with _ledger_mutex(ledger_path):
        if ledger_path.exists():
            raise FileExistsError(
                f"Final test ledger already exists; refusing a second attempt: {ledger_path}"
            )
        if output_directory.exists():
            raise FileExistsError(
                "Final test output directory already exists; refusing to overwrite it: "
                f"{output_directory}"
            )
        ledger = _with_ledger_sha256(
            {
                "schema_version": _LEDGER_SCHEMA_VERSION,
                "test_consumption_key": final_lock.test_consumption_key,
                "campaign_id": final_lock.campaign_id,
                "final_lock_path": portable_lock_path,
                "final_lock_sha256": final_lock.lock_sha256,
                "status": "attempt",
                "attempt": {
                    "attempt_id": identifier,
                    "started_at": _timestamp(now),
                    "output_directory": final_lock.output_directory,
                    "runtime_evidence": validated_runtime_evidence,
                    "runtime_evidence_sha256": canonical_sha256(
                        validated_runtime_evidence
                    ),
                },
                "success": None,
                "invalidation": None,
            }
        )
        _exclusive_write(ledger_path, ledger)
    return ledger


def _assert_ledger_matches_lock(
    ledger: Mapping[str, Any],
    final_lock: FinalLock,
    portable_lock_path: str,
) -> None:
    if ledger["test_consumption_key"] != final_lock.test_consumption_key:
        raise ValueError("Test ledger consumption key differs from the protected protocol.")
    if ledger["campaign_id"] != final_lock.campaign_id:
        raise ValueError("Test ledger campaign differs from the final lock.")
    if ledger["final_lock_path"] != portable_lock_path:
        raise ValueError("Test ledger points to a different final lock path.")
    if ledger["final_lock_sha256"] != final_lock.lock_sha256:
        raise ValueError("Test ledger points to a different final lock digest.")
    if ledger["attempt"]["output_directory"] != final_lock.output_directory:
        raise ValueError("Test ledger output directory differs from the final lock.")


def _validate_result_manifest(
    result_path: Path,
    *,
    repository_root: Path,
    final_lock: FinalLock,
) -> tuple[dict[str, Any], str]:
    result = _mapping(json.loads(result_path.read_text(encoding="utf-8")), "test result")
    expected_fields = {
        "schema_version",
        "campaign_id",
        "final_lock_sha256",
        "test_consumption_key",
        "source_revision",
        "world_path",
        "world_sha256",
        "evaluator_sha256",
        "controller_sha256",
        "source_bundle_sha256",
        "status",
        "split",
        "training_seeds",
        "scenario_seeds",
        "models",
        "evaluation_config_path",
        "evaluation_config_sha256",
        "formal_report_path",
        "formal_report_sha256",
        "webots_executable",
        "webots_executable_evidence_path",
        "webots_executable_evidence_sha256",
        "run_count",
        "episode_count",
        "run_artifacts",
        "completed_at",
    }
    _exact_keys(result, expected_fields, "test result manifest")
    if result["schema_version"] != 1 or result["status"] != "completed":
        raise ValueError("Test result manifest must be schema 1 with status=completed.")
    if result["campaign_id"] != final_lock.campaign_id:
        raise ValueError("Test result campaign differs from the final lock.")
    if result["final_lock_sha256"] != final_lock.lock_sha256:
        raise ValueError("Test result is not bound to the final lock digest.")
    for name in (
        "test_consumption_key",
        "world_path",
        "world_sha256",
        "evaluator_sha256",
        "controller_sha256",
        "source_bundle_sha256",
    ):
        if result[name] != getattr(final_lock, name):
            raise ValueError(f"Test result {name} differs from the final lock.")
    if result["source_revision"] != final_lock.source_revision:
        raise ValueError("Test result source revision differs from the final lock.")
    if result["split"] != "test":
        raise ValueError("Formal test result manifest must be labelled test.")
    final_lock.split_spec.require_training_seeds(result["training_seeds"])
    final_lock.split_spec.require_evaluation_seeds(
        "test", result["scenario_seeds"], confirm_test=True
    )
    if result["evaluation_config_path"] != final_lock.evaluation_config_path:
        raise ValueError("Test result evaluation config path differs from the final lock.")
    if result["evaluation_config_sha256"] != final_lock.evaluation_config_sha256:
        raise ValueError("Test result evaluation config digest differs from the final lock.")
    expected_models = [
        {
            "training_seed": item.training_seed,
            "model_path": item.model_path,
            "model_sha256": item.model_sha256,
        }
        for item in final_lock.model_manifests
    ]
    if result["models"] != expected_models:
        raise ValueError("Test result model list differs from the final lock.")
    if _integer(result["run_count"], "result.run_count") != 20:
        raise ValueError("Formal test result must contain exactly 20 runs.")
    if _integer(result["episode_count"], "result.episode_count") != 400:
        raise ValueError("Formal test result must contain exactly 400 episodes.")
    if not isinstance(result["completed_at"], str) or not result["completed_at"]:
        raise ValueError("Test result completed_at must be non-empty.")

    output_directory = _resolve_repository_path(
        repository_root, final_lock.output_directory, "output_directory"
    )
    executable_evidence = _runtime_evidence(
        result["webots_executable"], "result.webots_executable"
    )
    evidence_portable = _portable_path(
        result["webots_executable_evidence_path"],
        "webots_executable_evidence_path",
    )
    expected_evidence_path = f"{final_lock.output_directory}/webots_executable.json"
    if evidence_portable != expected_evidence_path:
        raise ValueError("Webots executable evidence path is not canonical.")
    evidence_sha256 = _sha256(
        result["webots_executable_evidence_sha256"],
        "webots_executable_evidence_sha256",
    )
    evidence_path = _verified_file(
        repository_root,
        evidence_portable,
        evidence_sha256,
        "Webots executable evidence",
    )
    stored_evidence = _runtime_evidence(
        json.loads(evidence_path.read_text(encoding="utf-8")),
        "stored Webots executable evidence",
    )
    if executable_evidence != stored_evidence:
        raise ValueError("Result manifest Webots evidence differs from its evidence artifact.")
    executable_path = Path(executable_evidence["executable_path"])
    if not executable_path.is_file():
        raise FileNotFoundError(
            f"Locked Webots executable no longer exists: {executable_path}"
        )
    if file_sha256(executable_path) != executable_evidence["executable_sha256"]:
        raise ValueError("Webots executable changed after the formal test.")
    report_portable = _portable_path(result["formal_report_path"], "formal_report_path")
    report_path = _verified_file(
        repository_root,
        report_portable,
        _sha256(result["formal_report_sha256"], "formal_report_sha256"),
        "formal report",
    )
    try:
        report_path.relative_to(output_directory)
    except ValueError as error:
        raise ValueError("Formal report must be inside the locked output directory.") from error

    run_artifacts = result["run_artifacts"]
    if not isinstance(run_artifacts, list) or len(run_artifacts) != 20:
        raise ValueError("Test result run_artifacts must contain exactly 20 records.")
    expected_matrix = [
        (condition, planner)
        for condition in ("G0", "G1", "G2", "G3")
        for planner in ("reactive", "dwa", "ppo_seed11", "ppo_seed22", "ppo_seed33")
    ]
    total_episodes = 0
    seen_paths: set[str] = set()
    model_by_seed = {item.training_seed: item for item in final_lock.model_manifests}
    for index, (artifact_value, (condition, planner)) in enumerate(
        zip(run_artifacts, expected_matrix, strict=True)
    ):
        artifact = _mapping(artifact_value, f"run_artifacts[{index}]")
        _exact_keys(
            artifact,
            {"episodes_path", "episodes_sha256", "summary_path", "summary_sha256"},
            f"run_artifacts[{index}]",
        )
        expected_base = f"{final_lock.output_directory}/runs/{condition}/{planner}"
        expected_episodes = f"{expected_base}/episodes.csv"
        expected_summary = f"{expected_base}/summary.json"
        if artifact["episodes_path"] != expected_episodes:
            raise ValueError(f"Run artifact {index} episodes_path is not canonical.")
        if artifact["summary_path"] != expected_summary:
            raise ValueError(f"Run artifact {index} summary_path is not canonical.")
        for path_name in ("episodes_path", "summary_path"):
            value = _portable_path(artifact[path_name], f"artifact.{path_name}")
            if value in seen_paths:
                raise ValueError("Test result run artifacts contain a duplicate path.")
            seen_paths.add(value)
        episodes_path = _verified_file(
            repository_root,
            artifact["episodes_path"],
            _sha256(artifact["episodes_sha256"], "artifact.episodes_sha256"),
            f"formal episodes {condition}/{planner}",
        )
        summary_path = _verified_file(
            repository_root,
            artifact["summary_path"],
            _sha256(artifact["summary_sha256"], "artifact.summary_sha256"),
            f"formal summary {condition}/{planner}",
        )
        with episodes_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required_columns = {"split", "scenario_seed", "planner", "condition_id"}
        if len(rows) != 20 or not rows or not required_columns.issubset(rows[0]):
            raise ValueError(
                f"Formal episodes {condition}/{planner} must contain 20 complete rows."
            )
        try:
            seeds = tuple(int(row["scenario_seed"]) for row in rows)
        except ValueError as error:
            raise ValueError("Formal result contains a non-integer scenario seed.") from error
        final_lock.split_spec.require_evaluation_seeds(
            "test", seeds, confirm_test=True
        )
        if any(row["split"] != "test" for row in rows):
            raise ValueError("Every formal result episode must be labelled test.")
        if any(row["planner"] != planner for row in rows):
            raise ValueError("Formal result episode planner differs from its matrix slot.")
        if any(row["condition_id"] != condition for row in rows):
            raise ValueError("Formal result episode condition differs from its matrix slot.")

        summary = _mapping(
            json.loads(summary_path.read_text(encoding="utf-8")),
            f"formal summary {condition}/{planner}",
        )
        if summary.get("split") != "test" or summary.get("episode_count") != 20:
            raise ValueError("Each formal summary must aggregate exactly 20 test episodes.")
        if summary.get("planners") != [planner] or summary.get("conditions") != [condition]:
            raise ValueError("Formal summary planner/condition differs from its matrix slot.")
        if summary.get("episodes_sha256") != artifact["episodes_sha256"]:
            raise ValueError("Formal summary is not bound to its episodes CSV digest.")
        provenance = _mapping(
            summary.get("evaluation_provenance"), "formal summary provenance"
        )
        provenance_expected: dict[str, Any] = {
            "split": "test",
            "condition_id": condition,
            "planner_label": planner,
            "scenario_seeds": list(final_lock.split_spec.test_scenario_seeds),
            "formal_campaign_id": final_lock.campaign_id,
            "final_lock_sha256": final_lock.lock_sha256,
            "test_consumption_key": final_lock.test_consumption_key,
            "evaluation_config_sha256": final_lock.evaluation_config_sha256,
            "formal_config_path": final_lock.evaluation_config_path,
            "formal_config_sha256": final_lock.evaluation_config_sha256,
            "world_path": final_lock.world_path,
            "world_sha256": final_lock.world_sha256,
            "evaluator_sha256": final_lock.evaluator_sha256,
            "controller_sha256": final_lock.controller_sha256,
            "source_bundle_sha256": final_lock.source_bundle_sha256,
            "episodes_csv_sha256": artifact["episodes_sha256"],
        }
        if planner.startswith("ppo_seed"):
            seed = int(planner.removeprefix("ppo_seed"))
            model = model_by_seed[seed]
            provenance_expected.update(
                {
                    "training_seed": seed,
                    "model_path": model.model_path,
                    "model_sha256": model.model_sha256,
                    "model_variant": model.model_variant,
                }
            )
        else:
            provenance_expected.update(
                {"training_seed": None, "model_path": None, "model_sha256": None}
            )
        for name, expected_value in provenance_expected.items():
            if provenance.get(name) != expected_value:
                raise ValueError(
                    f"Formal summary provenance {name} differs for {condition}/{planner}."
                )
        total_episodes += len(rows)
    if total_episodes != result["episode_count"]:
        raise ValueError("Formal result episode_count differs from its complete run artifacts.")
    return executable_evidence, canonical_sha256(executable_evidence)


def mark_test_success(
    *,
    repository_root: Path,
    final_lock_path: Path,
    result_manifest_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Atomically transition the sole attempt to success with hashed result evidence."""

    final_lock, portable_lock_path, ledger_path = _load_lock_for_transition(
        repository_root, final_lock_path
    )
    output_directory = _resolve_repository_path(
        repository_root, final_lock.output_directory, "output_directory"
    )
    result_path = result_manifest_path.resolve()
    try:
        result_path.relative_to(output_directory)
    except ValueError as error:
        raise ValueError(
            "Test result manifest must be inside the locked output directory."
        ) from error
    if not result_path.is_file():
        raise FileNotFoundError(f"Test result manifest does not exist: {result_path}")
    result_sha256 = file_sha256(result_path)
    runtime_evidence, runtime_evidence_sha256 = _validate_result_manifest(
        result_path,
        repository_root=repository_root,
        final_lock=final_lock,
    )
    result_portable = _portable_existing_path(
        repository_root, result_path, "result_manifest_path"
    )
    if file_sha256(result_path) != result_sha256:
        raise ValueError("Test result manifest changed while it was being validated.")
    with _ledger_mutex(ledger_path):
        ledger = load_test_ledger(ledger_path)
        _assert_ledger_matches_lock(ledger, final_lock, portable_lock_path)
        if ledger["status"] != "attempt":
            raise RuntimeError(
                f"Only an active attempt can become successful; status={ledger['status']}."
            )
        if ledger["attempt"]["runtime_evidence"] != runtime_evidence:
            raise ValueError("Result Webots evidence differs from the atomic attempt ledger.")
        if ledger["attempt"]["runtime_evidence_sha256"] != runtime_evidence_sha256:
            raise ValueError("Result Webots evidence digest differs from the attempt ledger.")
        updated = dict(ledger)
        updated.update(
            {
                "status": "success",
                "success": {
                    "finished_at": _timestamp(now),
                    "result_manifest_path": result_portable,
                    "result_manifest_sha256": result_sha256,
                },
            }
        )
        updated = _with_ledger_sha256(updated)
        _atomic_replace(ledger_path, updated)
    return updated


def invalidate_test_attempt(
    *,
    repository_root: Path,
    final_lock_path: Path,
    reason: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Invalidate an attempt or result without erasing its prior evidence."""

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Invalidation requires a non-empty reason.")
    final_lock, portable_lock_path, ledger_path = _load_lock_for_transition(
        repository_root, final_lock_path, verify_files=False
    )
    with _ledger_mutex(ledger_path):
        ledger = load_test_ledger(ledger_path)
        _assert_ledger_matches_lock(ledger, final_lock, portable_lock_path)
        previous_status: LedgerStatus = ledger["status"]
        if previous_status == "invalidated":
            raise RuntimeError("An invalidated test ledger is terminal and cannot be overwritten.")
        updated = dict(ledger)
        updated.update(
            {
                "status": "invalidated",
                "invalidation": {
                    "invalidated_at": _timestamp(now),
                    "previous_status": previous_status,
                    "reason": reason.strip(),
                },
            }
        )
        updated = _with_ledger_sha256(updated)
        _atomic_replace(ledger_path, updated)
    return updated


__all__ = [
    "FORMAL_TEST_SEEDS",
    "FORMAL_TRAINING_SEEDS",
    "FORMAL_VALIDATION_SEEDS",
    "FinalLock",
    "FormalModelManifest",
    "FormalSourceArtifact",
    "SplitSpec",
    "begin_test_attempt",
    "build_formal_source_bundle",
    "canonical_sha256",
    "discover_formal_source_paths",
    "file_sha256",
    "invalidate_test_attempt",
    "load_final_lock",
    "load_test_ledger",
    "mark_test_success",
    "protected_test_consumption_key",
    "protected_test_ledger_path",
    "protected_test_protocol_payload",
    "source_bundle_sha256",
    "validate_model_manifest",
    "write_final_lock",
    "write_or_validate_model_provenance",
]
