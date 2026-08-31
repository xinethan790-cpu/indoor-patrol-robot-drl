from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from indoor_patrol_rl.application import deployment
from indoor_patrol_rl.application.deployment import (
    AUDITED_R2A_MODEL_SHA256,
    AUDITED_R2A_PROFILE_SHA256,
    AUDITED_R2A_SHIELD_RESULT_TOKEN,
    AUDITED_R2A_SYSTEM_ID,
    DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH,
    DeploymentIntegrityError,
    DeploymentProfileError,
    load_r2a_system_deployment_profile,
    load_system_deployment_profile,
)

_SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_profile() -> dict[str, Any]:
    path = _SOURCE_ROOT / DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def _bound_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        if "path" in value and ("sha256" in value or "source_sha256" in value):
            paths.add(value["path"])
        for item in value.values():
            paths.update(_bound_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.update(_bound_paths(item))
    return paths


def _copy_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    profile = _source_profile()
    for portable in _bound_paths(profile):
        source = _SOURCE_ROOT / portable
        destination = root / portable
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_profile = _SOURCE_ROOT / DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH
    destination_profile = root / DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH
    destination_profile.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_profile, destination_profile)
    shutil.copy2(source_profile.with_suffix(".sha256"), destination_profile.with_suffix(".sha256"))
    return root


def _read_profile(root: Path) -> dict[str, Any]:
    return json.loads((root / DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH).read_text(encoding="utf-8"))


def _rewrite_profile(root: Path, value: dict[str, Any]) -> Path:
    profile_path = root / DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH
    profile_path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    profile_path.with_suffix(".sha256").write_text(
        f"{_sha256(profile_path)}  {profile_path.name}\n",
        encoding="utf-8",
    )
    return profile_path


def test_loads_repository_audited_r2a_system() -> None:
    profile = load_r2a_system_deployment_profile(repository_root=_SOURCE_ROOT)

    assert profile.system_id == AUDITED_R2A_SYSTEM_ID
    assert profile.profile_sha256 == AUDITED_R2A_PROFILE_SHA256
    assert profile.model_sha256 == AUDITED_R2A_MODEL_SHA256
    assert profile.safety_shield_id == AUDITED_R2A_SHIELD_RESULT_TOKEN
    assert profile.model_path.is_file()
    assert profile.safety_shield_config.front_stop_distance == pytest.approx(0.24)
    assert profile.safety_shield_config.front_slowdown_distance == pytest.approx(0.75)
    assert profile.validation_episode_count == 100
    assert profile.validation_success_rate == pytest.approx(0.88)
    assert profile.validation_collision_rate == pytest.approx(0.12)
    assert len(profile.all_artifacts) == 15
    profile.verify_integrity()


def test_generic_loader_returns_typed_verified_bindings(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    profile = load_system_deployment_profile(
        DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH,
        repository_root=root,
    )

    assert profile.repository_root == root.resolve()
    assert profile.profile_path == (root / DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH).resolve()
    assert {item.name for item in profile.qualification_artifacts} == {
        "qualification.fixed_safety_report",
        "qualification.paired_diagnostic_report",
        "qualification.paired_diagnostic_config",
    }
    assert {item.name for item in profile.validation_artifacts} == {
        "validation_evidence.sweep_manifest",
        "validation_evidence.summary",
        "validation_evidence.episodes_csv",
    }
    assert len(profile.runtime_artifacts) == 7


def test_rejects_changed_model_artifact(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    raw = _read_profile(root)
    model_path = root / raw["system"]["model"]["path"]
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    with pytest.raises(DeploymentIntegrityError, match="system.model SHA-256 mismatch"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)


def test_rejects_changed_safety_shield_source(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    raw = _read_profile(root)
    shield_path = root / raw["system"]["shield_profile"]["path"]
    shield_path.write_text(shield_path.read_text(encoding="utf-8") + "\n# changed\n")

    with pytest.raises(DeploymentIntegrityError, match="Safety-shield source SHA-256"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)


@pytest.mark.parametrize(
    "section,name",
    [
        ("qualification", "fixed_safety_report"),
        ("qualification", "paired_diagnostic_report"),
        ("validation_evidence", "summary"),
        ("validation_evidence", "episodes_csv"),
        ("runtime_inputs", "world"),
    ],
)
def test_rejects_changed_bound_evidence_or_runtime_file(
    tmp_path: Path,
    section: str,
    name: str,
) -> None:
    root = _copy_fixture(tmp_path)
    raw = _read_profile(root)
    artifact_path = root / raw[section][name]["path"]
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\nchanged")

    with pytest.raises(DeploymentIntegrityError, match="SHA-256 mismatch"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)


def test_rejects_profile_with_stale_digest_sidecar(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    profile_path = root / DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH
    profile_path.write_text(profile_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(DeploymentIntegrityError, match="digest sidecar"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)


def test_audited_loader_rejects_coherently_rewritten_profile_and_sidecar(
    tmp_path: Path,
) -> None:
    root = _copy_fixture(tmp_path)
    raw = _read_profile(root)
    raw["qualification"]["preregistration_revision"] = "0" * 40
    _rewrite_profile(root, raw)

    with pytest.raises(DeploymentIntegrityError, match="audited artifact"):
        load_r2a_system_deployment_profile(repository_root=root)


def test_generic_loader_cross_checks_preregistration_revision(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    raw = _read_profile(root)
    raw["qualification"]["preregistration_revision"] = "0" * 40
    _rewrite_profile(root, raw)

    with pytest.raises(DeploymentProfileError, match="preregistration identity"):
        load_system_deployment_profile(
            DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH,
            repository_root=root,
        )


def test_requires_profile_digest_sidecar(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    (root / DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH).with_suffix(".sha256").unlink()

    with pytest.raises(FileNotFoundError, match="profile digest is missing"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)


def test_rejects_policy_only_or_disabled_shield_deployment(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    raw = _read_profile(root)
    raw["restrictions"]["model_and_shield_must_be_deployed_together"] = False
    _rewrite_profile(root, raw)

    with pytest.raises(DeploymentProfileError, match="model-plus-shield"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)

    raw = _source_profile()
    raw["system"]["shield_profile"]["enabled"] = False
    _rewrite_profile(root, raw)

    with pytest.raises(DeploymentProfileError, match="shield must be enabled"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)


def test_rejects_inconsistent_shield_semantic_identity(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    raw = _read_profile(root)
    raw["system"]["shield_profile"]["semantic_sha256"] = "0" * 64
    _rewrite_profile(root, raw)

    with pytest.raises(DeploymentProfileError, match="semantic SHA-256"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)


def test_rejects_inconsistent_metric_identity(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    raw = _read_profile(root)
    raw["validation_evidence"]["metrics"]["success_rate"] = 0.99
    _rewrite_profile(root, raw)

    with pytest.raises(DeploymentProfileError, match="success_rate is inconsistent"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)


def test_rejects_repository_escape_in_bound_path(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    raw = _read_profile(root)
    raw["system"]["model"]["path"] = "../outside-model.zip"
    _rewrite_profile(root, raw)

    with pytest.raises(DeploymentProfileError, match="normalized repository-relative path"):
        load_system_deployment_profile(DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH, repository_root=root)


def test_runtime_reverification_detects_post_load_change(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    profile = load_system_deployment_profile(
        DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH,
        repository_root=root,
    )
    profile.model.path.write_bytes(profile.model.path.read_bytes() + b"changed-after-load")

    with pytest.raises(DeploymentIntegrityError, match="system.model SHA-256 mismatch"):
        profile.verify_integrity()


def test_audited_loader_rejects_another_valid_system_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_system_deployment_profile(
        DEFAULT_R2A_DEPLOYMENT_PROFILE_PATH,
        repository_root=_SOURCE_ROOT,
    )
    another_system = replace(profile, system_id="another_valid_system")
    monkeypatch.setattr(
        deployment,
        "load_system_deployment_profile",
        lambda *_args, **_kwargs: another_system,
    )

    with pytest.raises(DeploymentProfileError, match="not the audited R2a system"):
        load_r2a_system_deployment_profile(repository_root=_SOURCE_ROOT)
