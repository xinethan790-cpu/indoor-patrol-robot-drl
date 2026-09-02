from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from indoor_patrol_rl.training import (
    WarmStartPreflightError,
    WebotsCurriculumRunner,
    load_webots_curriculum,
    verify_warm_start_requirement,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(rows: list[dict[str, str]]) -> dict[str, int | float]:
    episodes = len(rows)
    success_count = sum(row["outcome"] == "success" for row in rows)
    collision_count = sum(row["outcome"] == "collision" for row in rows)
    timeout_count = sum(row["outcome"] == "timeout" for row in rows)
    error_count = sum(row["outcome"] == "error" for row in rows)
    return {
        "episodes": episodes,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "success_rate": success_count / episodes,
        "collision_rate": collision_count / episodes,
        "timeout_rate": timeout_count / episodes,
    }


def _aggregate(
    planner: str, rows: list[dict[str, str]], condition: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"planner": planner, **_metrics(rows)}
    if condition is not None:
        result["condition_id"] = condition
    return result


def _write_fixture(root: Path) -> tuple[Path, dict[str, Path]]:
    model = root / "artifacts/models/warm-start.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"audited warm-start model")

    rows = []
    outcomes = {
        ("G0", 5000): "success",
        ("G0", 5001): "success",
        ("G1", 5000): "success",
        ("G1", 5001): "collision",
    }
    for (condition, seed), outcome in outcomes.items():
        rows.append(
            {
                "split": "validation",
                "condition_id": condition,
                "scenario_seed": str(seed),
                "planner": "ppo",
                "outcome": outcome,
                "is_success": str(outcome == "success"),
                "collision": str(outcome == "collision"),
                "timeout": str(outcome == "timeout"),
            }
        )
    evidence = root / "artifacts/results/warm-start-validation"
    evidence.mkdir(parents=True)
    episodes = evidence / "episodes.csv"
    with episodes.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "1.0",
        "split": "validation",
        "episode_count": 4,
        "planners": ["ppo"],
        "conditions": ["G0", "G1"],
        "by_planner": [_aggregate("ppo", rows)],
        "by_planner_condition": [
            _aggregate("ppo", [row for row in rows if row["condition_id"] == condition], condition)
            for condition in ("G0", "G1")
        ],
    }
    summary_path = evidence / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    registry_path = root / "configs/models/registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry = {
        "kinematic_seed11": {
            "candidates": [
                {
                    "id": "stage2_nominal_dynamic",
                    "path": "artifacts/models/warm-start.zip",
                    "sha256": _sha256(model),
                    "actual_steps": 100,
                    "validation_episodes": 4,
                    "validation_success_rate": 0.75,
                    "validation_collision_rate": 0.25,
                    "selected": True,
                    "role": "webots_policy_warm_start",
                }
            ]
        }
    }
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    requirement_path = root / "configs/models/warm-start-requirement.yaml"
    requirement = {
        "schema_version": 1,
        "requirement_id": "fixture-warm-start-v1",
        "curriculum": {
            "config_path": "configs/train/curriculum.yaml",
            "transfer_model_path": "artifacts/models/warm-start.zip",
        },
        "model": {"sha256": _sha256(model), "actual_steps": 100},
        "registry": {
            "path": "configs/models/registry.yaml",
            "section": "kinematic_seed11",
            "candidate_id": "stage2_nominal_dynamic",
        },
        "validation": {
            "split": "validation",
            "planner": "ppo",
            "conditions": ["G0", "G1"],
            "seed_start": 5000,
            "seed_count": 2,
            "total_episode_count": 4,
            "planner_episode_count": 4,
            "success_count": 3,
            "collision_count": 1,
            "timeout_count": 0,
            "error_count": 0,
            "success_rate": 0.75,
            "collision_rate": 0.25,
            "timeout_rate": 0.0,
            "summary_path": "artifacts/results/warm-start-validation/summary.json",
            "summary_sha256": _sha256(summary_path),
            "episodes_path": "artifacts/results/warm-start-validation/episodes.csv",
            "episodes_sha256": _sha256(episodes),
        },
    }
    requirement_path.write_text(yaml.safe_dump(requirement, sort_keys=False), encoding="utf-8")

    config_path = root / "configs/train/curriculum.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "transfer_model": "artifacts/models/warm-start.zip",
                "warm_start_requirement": "configs/models/warm-start-requirement.yaml",
                "port_base": 12490,
                "validation": {
                    "split": "validation",
                    "seed_start": 5000,
                    "episodes": 2,
                    "min_success_rate": 0.75,
                    "max_collision_rate": 0.25,
                },
                "stages": [
                    {
                        "id": "W0",
                        "name": "empty_room_transfer",
                        "obstacle_count": 0,
                        "training_steps": 10,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path, {
        "model": model,
        "episodes": episodes,
        "summary": summary_path,
        "registry": registry_path,
        "requirement": requirement_path,
    }


def _load_fixture(root: Path):
    config_path, paths = _write_fixture(root)
    return load_webots_curriculum(config_path), paths


def test_complete_temporary_warm_start_evidence_passes(tmp_path: Path) -> None:
    config, paths = _load_fixture(tmp_path)
    result = verify_warm_start_requirement(repository_root=tmp_path, config=config)

    assert result is not None
    assert result.model_sha256 == _sha256(paths["model"])
    assert result.summary_sha256 == _sha256(paths["summary"])
    assert result.episodes_sha256 == _sha256(paths["episodes"])
    assert result.episode_count == 4
    assert result.success_rate == 0.75
    assert result.collision_rate == 0.25


def test_model_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    config, paths = _load_fixture(tmp_path)
    paths["model"].write_bytes(b"different model bytes")

    with pytest.raises(WarmStartPreflightError, match="model SHA-256 mismatch"):
        verify_warm_start_requirement(repository_root=tmp_path, config=config)


def test_policy_transfer_must_be_the_hash_bound_warm_start(tmp_path: Path) -> None:
    config, _paths = _load_fixture(tmp_path)
    mismatched = replace(
        config,
        policy_transfer_model=Path("artifacts/models/different-policy.zip"),
    )

    with pytest.raises(
        WarmStartPreflightError,
        match="policy_transfer_model differs from the warm-start requirement",
    ):
        verify_warm_start_requirement(repository_root=tmp_path, config=mismatched)


def test_missing_validation_evidence_reports_both_files(tmp_path: Path) -> None:
    config, paths = _load_fixture(tmp_path)
    paths["summary"].unlink()
    paths["episodes"].unlink()

    with pytest.raises(WarmStartPreflightError) as captured:
        verify_warm_start_requirement(repository_root=tmp_path, config=config)
    message = str(captured.value)
    assert "validation summary is missing" in message
    assert "validation episodes is missing" in message


def test_registry_candidate_must_match_requirement(tmp_path: Path) -> None:
    config, paths = _load_fixture(tmp_path)
    registry = yaml.safe_load(paths["registry"].read_text(encoding="utf-8"))
    registry["kinematic_seed11"]["candidates"][0]["sha256"] = "0" * 64
    paths["registry"].write_text(yaml.safe_dump(registry), encoding="utf-8")

    with pytest.raises(WarmStartPreflightError, match=r"registry candidate\.sha256"):
        verify_warm_start_requirement(repository_root=tmp_path, config=config)


def test_summary_metrics_are_recomputed_from_csv(tmp_path: Path) -> None:
    config, paths = _load_fixture(tmp_path)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["by_planner"][0]["success_count"] = 4
    paths["summary"].write_text(json.dumps(summary) + "\n", encoding="utf-8")
    requirement = yaml.safe_load(paths["requirement"].read_text(encoding="utf-8"))
    requirement["validation"]["summary_sha256"] = _sha256(paths["summary"])
    paths["requirement"].write_text(yaml.safe_dump(requirement), encoding="utf-8")

    with pytest.raises(WarmStartPreflightError, match="disagrees with episodes.csv"):
        verify_warm_start_requirement(repository_root=tmp_path, config=config)


def test_requirement_rejects_repository_escape(tmp_path: Path) -> None:
    config, paths = _load_fixture(tmp_path)
    requirement = yaml.safe_load(paths["requirement"].read_text(encoding="utf-8"))
    requirement["curriculum"]["transfer_model_path"] = "../outside.zip"
    paths["requirement"].write_text(yaml.safe_dump(requirement), encoding="utf-8")

    with pytest.raises(WarmStartPreflightError, match="must stay inside the repository"):
        verify_warm_start_requirement(repository_root=tmp_path, config=config)


def test_requirement_rejects_non_string_condition_without_type_error(
    tmp_path: Path,
) -> None:
    config, paths = _load_fixture(tmp_path)
    requirement = yaml.safe_load(paths["requirement"].read_text(encoding="utf-8"))
    requirement["validation"]["conditions"] = ["G0", []]
    paths["requirement"].write_text(yaml.safe_dump(requirement), encoding="utf-8")

    with pytest.raises(
        WarmStartPreflightError, match="conditions must contain unique condition ids"
    ):
        verify_warm_start_requirement(repository_root=tmp_path, config=config)


def test_requirement_rejects_symbolic_linked_model(tmp_path: Path) -> None:
    config, paths = _load_fixture(tmp_path)
    real_model = paths["model"].with_name("real-model.zip")
    paths["model"].replace(real_model)
    paths["model"].symlink_to(real_model.name)

    with pytest.raises(WarmStartPreflightError, match="must not use a symbolic link"):
        verify_warm_start_requirement(repository_root=tmp_path, config=config)


def test_runner_fails_preflight_before_process_runner_can_start(tmp_path: Path) -> None:
    config, paths = _load_fixture(tmp_path)
    paths["model"].write_bytes(b"tampered")
    process_started = False

    def process_runner(*_args: Any, **_kwargs: Any):
        nonlocal process_started
        process_started = True
        raise AssertionError("Webots must not start")

    with pytest.raises(WarmStartPreflightError, match="model SHA-256 mismatch"):
        WebotsCurriculumRunner(
            repository_root=tmp_path,
            config=config,
            webots_executable=tmp_path / "webots",
            seed=22,
            manifest_path=tmp_path / "artifacts/results/curriculum/run_manifest.yaml",
            process_runner=process_runner,
        )
    assert process_started is False


def test_current_repository_is_explicitly_not_ready() -> None:
    repository_root = Path(__file__).absolute().parents[1]
    config = load_webots_curriculum(repository_root / "configs/train/webots_curriculum.yaml")

    with pytest.raises(WarmStartPreflightError) as captured:
        verify_warm_start_requirement(repository_root=repository_root, config=config)
    message = str(captured.value)
    assert (
        "warm-start model SHA-256 mismatch" in message or "warm-start model is missing" in message
    )
    assert "validation summary is missing" in message
    assert "validation episodes is missing" in message
