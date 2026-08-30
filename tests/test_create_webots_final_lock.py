from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from indoor_patrol_rl.evaluation.formal_protocol import load_final_lock
from scripts.create_webots_final_lock import (
    create_webots_final_lock,
    parse_model_evidence,
    require_clean_head,
)
from tests.test_formal_protocol import _build_model_manifest

REVISION = "a" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _copy_config(root: Path) -> Path:
    project = Path(__file__).absolute().parents[1]
    source = project / "configs/eval/webots_formal.yaml"
    destination = root / "configs/eval/webots_formal.yaml"
    destination.parent.mkdir(parents=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    for value in (
        "worlds/webots_ppo_evaluate.wbt",
        "scripts/run_webots_formal_test.py",
        "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py",
        "requirements-formal-lock.txt",
    ):
        target = root / value
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / value, target)
    shutil.copytree(
        project / "src/indoor_patrol_rl",
        root / "src/indoor_patrol_rl",
        dirs_exist_ok=True,
    )
    return destination


def _evidence(root: Path, seed: int, *, checkpoint: bool = False) -> tuple[str, ...]:
    manifest = _build_model_manifest(root, seed, checkpoint=checkpoint)
    return (
        str(seed),
        str(root / manifest.model_path),
        str(root / manifest.training_metadata_path),
        str(root / manifest.curriculum_manifest_path),
        str(root / manifest.selection_manifest_path),
        str(root / manifest.validation_episodes_path),
    )


def _git_runner(*, dirty: str = ""):
    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[1:3] == ["rev-parse", "--verify"]:
            return subprocess.CompletedProcess(command, 0, stdout=f"{REVISION}\n", stderr="")
        if command[1:3] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(command, 0, stdout=dirty, stderr="")
        raise AssertionError(f"unexpected git command: {command}")

    return run


def test_create_cli_builds_a_hashed_lock_from_all_three_evidence_sets(
    tmp_path: Path,
) -> None:
    config = _copy_config(tmp_path)
    entries = [_evidence(tmp_path, seed) for seed in (33, 11, 22)]

    final_lock, destination = create_webots_final_lock(
        repository_root=tmp_path,
        config_path=config,
        evidence_entries=entries,
        git_runner=_git_runner(),
    )

    loaded = load_final_lock(destination, repository_root=tmp_path)
    assert loaded == final_lock
    assert final_lock.source_revision == REVISION
    assert [item.training_seed for item in final_lock.model_manifests] == [11, 22, 33]
    assert final_lock.evaluation_config_path == "configs/eval/webots_formal.yaml"
    assert destination == (
        tmp_path
        / "artifacts/results/formal_locks/webots_formal_g0_g3_v1.json"
    )
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["lock_sha256"] == final_lock.lock_sha256


def test_create_cli_requires_a_clean_head_before_writing(tmp_path: Path) -> None:
    config = _copy_config(tmp_path)
    entries = [_evidence(tmp_path, seed) for seed in (11, 22, 33)]

    with pytest.raises(RuntimeError, match="clean Git HEAD"):
        create_webots_final_lock(
            repository_root=tmp_path,
            config_path=config,
            evidence_entries=entries,
            git_runner=_git_runner(dirty=" M src/changed.py\n"),
        )
    assert not (tmp_path / "artifacts/results/formal_locks").exists()


def test_create_cli_rejects_missing_or_duplicate_training_seed() -> None:
    dummy = ("11", "model", "metadata", "curriculum", "selection", "episodes")

    with pytest.raises(ValueError, match="training seeds"):
        parse_model_evidence([dummy, ("22", *dummy[1:])])
    with pytest.raises(ValueError, match="training seeds"):
        parse_model_evidence([dummy, dummy, ("33", *dummy[1:])])


def test_create_cli_never_overwrites_an_existing_final_lock(tmp_path: Path) -> None:
    config = _copy_config(tmp_path)
    entries = [_evidence(tmp_path, seed) for seed in (11, 22, 33)]
    _, destination = create_webots_final_lock(
        repository_root=tmp_path,
        config_path=config,
        evidence_entries=entries,
        git_runner=_git_runner(),
    )
    before = destination.read_bytes()

    with pytest.raises(FileExistsError):
        create_webots_final_lock(
            repository_root=tmp_path,
            config_path=config,
            evidence_entries=entries,
            git_runner=_git_runner(),
        )
    assert destination.read_bytes() == before


def test_require_clean_head_rejects_an_invalid_revision(tmp_path: Path) -> None:
    def invalid_revision(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="HEAD\n", stderr="")

    with pytest.raises(ValueError, match="full 40- or 64-character revision"):
        require_clean_head(tmp_path, git_runner=invalid_revision)


def test_create_cli_accepts_training_recorded_intermediate_checkpoints(
    tmp_path: Path,
) -> None:
    config = _copy_config(tmp_path)
    entries = [_evidence(tmp_path, seed, checkpoint=True) for seed in (11, 22, 33)]

    final_lock, destination = create_webots_final_lock(
        repository_root=tmp_path,
        config_path=config,
        evidence_entries=entries,
        git_runner=_git_runner(),
    )

    loaded = load_final_lock(destination, repository_root=tmp_path)
    assert loaded == final_lock
    assert all("/checkpoints/" in item.model_path for item in loaded.model_manifests)


def test_create_cli_does_not_backfill_missing_training_provenance(
    tmp_path: Path,
) -> None:
    config = _copy_config(tmp_path)
    entries = [_evidence(tmp_path, seed) for seed in (11, 22, 33)]
    missing = Path(entries[0][1] + ".formal-provenance.json")
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="Training-time model provenance"):
        create_webots_final_lock(
            repository_root=tmp_path,
            config_path=config,
            evidence_entries=entries,
            git_runner=_git_runner(),
        )
    assert not missing.exists()
