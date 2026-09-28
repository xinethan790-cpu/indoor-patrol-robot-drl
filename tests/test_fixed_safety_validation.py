from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

import indoor_patrol_rl.evaluation.fixed_safety_validation as fixed_validation
from indoor_patrol_rl.core import SafetyShieldConfig
from indoor_patrol_rl.evaluation.checkpoint_selection import sha256_file
from indoor_patrol_rl.evaluation.checkpoint_sweep import load_safety_shield_profile
from indoor_patrol_rl.evaluation.fixed_safety_validation import (
    analyze_fixed_safety_validation,
)

CSV_FIELDS = (
    "schema_version",
    "episode_id",
    "split",
    "condition_id",
    "scenario_seed",
    "planner",
    "outcome",
    "is_success",
    "collision",
    "timeout",
    "step_count",
    "elapsed_sim_s",
    "completion_time_s",
    "episode_return",
    "path_length_m",
    "reference_path_length_m",
    "path_length_ratio",
    "minimum_clearance_m",
    "near_collision_ratio",
    "action_delta_sq_mean",
    "inference_latency_p50_ms",
    "inference_latency_p95_ms",
    "error_type",
)


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-q",
        "-m",
        message,
    )
    return _git(root, "rev-parse", "HEAD")


def _write_episodes(path: Path, outcomes: list[str], planner: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for offset, outcome in enumerate(outcomes):
            success = outcome == "success"
            seed = 5_000 + offset
            writer.writerow(
                {
                    "schema_version": "1.0",
                    "episode_id": f"validation-R2ASV-{seed}-{planner}",
                    "split": "validation",
                    "condition_id": "R2ASV",
                    "scenario_seed": seed,
                    "planner": planner,
                    "outcome": outcome,
                    "is_success": str(success),
                    "collision": str(outcome == "collision"),
                    "timeout": str(outcome == "timeout"),
                    "step_count": 100,
                    "elapsed_sim_s": 10.0,
                    "completion_time_s": 10.0 if success else "",
                    "episode_return": 1.0,
                    "path_length_m": 2.0,
                    "reference_path_length_m": 2.0,
                    "path_length_ratio": 1.0 if success else "",
                    "minimum_clearance_m": 0.2,
                    "near_collision_ratio": 0.0,
                    "action_delta_sq_mean": 0.1,
                    "inference_latency_p50_ms": 0.1,
                    "inference_latency_p95_ms": 0.2,
                    "error_type": "fixture_error" if outcome == "error" else "",
                }
            )


def _aggregate(outcomes: list[str], planner: str) -> dict[str, Any]:
    total = len(outcomes)
    counts = {name: outcomes.count(name) for name in ("success", "collision", "timeout", "error")}
    return {
        "planner": planner,
        "episodes": total,
        "success_count": counts["success"],
        "collision_count": counts["collision"],
        "timeout_count": counts["timeout"],
        "error_count": counts["error"],
        "success_rate": counts["success"] / total,
        "collision_rate": counts["collision"] / total,
        "timeout_rate": counts["timeout"] / total,
        "completion_time_s_mean": 10.0,
    }


def _build_fixture(
    root: Path,
    *,
    success_count: int = 80,
    collision_count: int = 20,
    timeout_count: int = 0,
    error_count: int = 0,
    diagnostic_passed: bool = True,
    extra_run: bool = False,
) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    # Pin line endings so evidence byte-comparisons match the committed
    # blobs regardless of the developer machine's core.autocrlf setting.
    _git(root, "config", "core.autocrlf", "false")
    model_path = root / "artifacts/models/model.zip"
    source_contents = {
        model_path: b"fixed model\n",
        root / "worlds/evaluate.wbt": b"#VRML_SIM fixed\n",
        root / "controllers/evaluate.py": b"# evaluator\n",
        root / "src/indoor_patrol_rl/envs/webots_patrol.py": b"# adapter\n",
        root / "src/indoor_patrol_rl/evaluation/checkpoint_sweep.py": b"# runner\n",
        root / "src/indoor_patrol_rl/evaluation/fixed_safety_validation.py": (
            b"# fixed validator\n"
        ),
        root / "scripts/analyze_webots_fixed_safety.py": b"# fixed cli\n",
        root / "src/indoor_patrol_rl/evaluation/paired_safety.py": b"# paired validator\n",
        root / "configs/eval/paired.yaml": b"schema_version: 1\n",
    }
    for path, contents in source_contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    profile_path = root / "configs/eval/shield.yaml"
    profile_config = asdict(SafetyShieldConfig())
    profile_config.pop("lidar_max_range")
    _write_yaml(
        profile_path,
        {
            "schema_version": 1,
            "profile_id": "fixture-shield-v1",
            "enabled": True,
            "config": profile_config,
        },
    )
    (root / "README.md").write_text("fixed fixture\n", encoding="utf-8")
    feature_revision = _commit(root, "fixed feature")
    profile = load_safety_shield_profile(profile_path, repository_root=root)

    required_outcome = "retain_fixture_for_fixed_acceptance"
    paired_report_path = root / "artifacts/results/paired/paired_safety_report.json"
    paired_report = {
        "schema_version": 1,
        "kind": "webots_paired_safety_diagnostic",
        "split": "validation",
        "decision": {
            "passed": diagnostic_passed,
            "outcome": required_outcome if diagnostic_passed else "reject_fixture",
            "formal_model_promotion": "forbidden",
            "model_alias_created": False,
        },
        "gate": {
            "require_all": True,
            "checks": {
                "fixture_check": {
                    "operator": ">=",
                    "observed": 1,
                    "threshold": 1,
                    "passed": diagnostic_passed,
                }
            },
            "passed": diagnostic_passed,
        },
        "source_bindings": {
            "git_revision": feature_revision,
            "model": {
                "path": "artifacts/models/model.zip",
                "sha256": sha256_file(model_path),
            },
            "arms": {
                "candidate": {
                    "profile": {
                        "path": "configs/eval/shield.yaml",
                        "source_sha256": profile.source_sha256,
                        "semantic_sha256": profile.semantic_sha256,
                        "result_token": profile.result_token,
                        "enabled": True,
                    }
                }
            },
        },
        "paired_validation": {
            "episode_count": 120,
            "scenario_seed_start": 72_542,
            "scenario_seed_end": 72_661,
        },
    }
    _write_json(paired_report_path, paired_report)
    _commit(root, "record paired qualification")

    run_name = f"webots_validation_r2asv_checkpoint10240_seed11_shield-{profile.result_token}"
    config = {
        "schema_version": 1,
        "experiment_id": "fixture_fixed_safety_validation_v1",
        "purpose": "fixed_system_safety_acceptance",
        "split": "validation",
        "condition_id": "R2ASV",
        "sweep_feature_revision": feature_revision,
        "training_seed": 11,
        "scenario_seed_start": 5_000,
        "episodes": 100,
        "model": {
            "path": "artifacts/models/model.zip",
            "sha256": sha256_file(model_path),
            "variant": "fixture_v1",
            "checkpoint_step": 10_240,
        },
        "scenario": {
            "obstacle_count": 1,
            "obstacle_speed_range": [0.1, 0.2],
            "horizontal_route_half_length": 1.1,
            "vertical_route_half_length": 1.1,
            "deterministic_policy": True,
            "static_layout": "train_a",
            "action_repeat": 4,
            "max_episode_steps": 300,
        },
        "runtime_inputs": {
            "world": {
                "path": "worlds/evaluate.wbt",
                "sha256": sha256_file(root / "worlds/evaluate.wbt"),
            },
            "evaluator": {
                "path": "controllers/evaluate.py",
                "sha256": sha256_file(root / "controllers/evaluate.py"),
            },
            "environment_adapter": {
                "path": "src/indoor_patrol_rl/envs/webots_patrol.py",
                "sha256": sha256_file(root / "src/indoor_patrol_rl/envs/webots_patrol.py"),
            },
            "checkpoint_sweep_runner": {
                "path": "src/indoor_patrol_rl/evaluation/checkpoint_sweep.py",
                "sha256": sha256_file(
                    root / "src/indoor_patrol_rl/evaluation/checkpoint_sweep.py"
                ),
            },
            "fixed_validator": {
                "path": "src/indoor_patrol_rl/evaluation/fixed_safety_validation.py",
                "sha256": sha256_file(
                    root / "src/indoor_patrol_rl/evaluation/fixed_safety_validation.py"
                ),
            },
            "fixed_cli": {
                "path": "scripts/analyze_webots_fixed_safety.py",
                "sha256": sha256_file(root / "scripts/analyze_webots_fixed_safety.py"),
            },
            "paired_validator": {
                "path": "src/indoor_patrol_rl/evaluation/paired_safety.py",
                "sha256": sha256_file(
                    root / "src/indoor_patrol_rl/evaluation/paired_safety.py"
                ),
            },
        },
        "shield_profile": {
            "path": "configs/eval/shield.yaml",
            "source_sha256": profile.source_sha256,
            "semantic_sha256": profile.semantic_sha256,
            "result_token": profile.result_token,
            "enabled": True,
        },
        "evaluation_arm": {
            "id": "fixture_system",
            "role": "system",
            "port_base": 14_020,
            "run_name": run_name,
            "sweep_directory": "artifacts/results/fixed_sweep",
        },
        "diagnostic_qualification": {
            "report_path": "artifacts/results/paired/paired_safety_report.json",
            "report_sha256": sha256_file(paired_report_path),
            "paired_config_path": "configs/eval/paired.yaml",
            "paired_config_sha256": sha256_file(root / "configs/eval/paired.yaml"),
            "required_outcome": required_outcome,
            "require_passed": True,
        },
        "preregistration": {
            "fixed_model": True,
            "fixed_profile": True,
            "candidate_count": 1,
            "parameter_scan": "forbidden",
            "adaptive_extension": "forbidden",
            "held_out_formal_test_access": "forbidden",
            "diagnostic_seed_overlap": "forbidden",
            "selection_rule": "post_selection_fixed_system_acceptance_5000_5099",
        },
        "gate": {
            "minimum_success_rate": 0.75,
            "maximum_collision_rate": 0.25,
            "maximum_timeout_count": 2,
            "maximum_error_count": 0,
        },
        "decision": {
            "pass": "retain_fixture_system",
            "fail": "reject_fixture_system",
            "formal_model_promotion": "forbidden",
            "output_directory": "artifacts/results/fixed_report",
            "deployment_directory": "artifacts/deployments/fixed_system",
        },
    }
    config_path = root / "configs/eval/fixed.yaml"
    _write_yaml(config_path, config)
    preregistration_revision = _commit(root, "preregister fixed system")
    config_blob_oid = _git(
        root,
        "rev-parse",
        f"{preregistration_revision}:configs/eval/fixed.yaml",
    )
    _write_yaml(
        config_path.with_suffix(".binding.yaml"),
        {
            "schema_version": 1,
            "kind": "paired_safety_preregistration_git_binding",
            "config_path": "configs/eval/fixed.yaml",
            "preregistration_revision": preregistration_revision,
            "config_blob_oid": config_blob_oid,
            "config_sha256": sha256_file(config_path),
        },
    )
    _commit(root, "bind fixed preregistration")

    outcomes = (
        ["success"] * success_count
        + ["collision"] * collision_count
        + ["timeout"] * timeout_count
        + ["error"] * error_count
    )
    assert len(outcomes) == 100
    planner = "ppo_r2asv_step10240"
    run_directory = root / "artifacts/results" / run_name
    episodes_path = run_directory / "episodes.csv"
    summary_path = run_directory / "summary.json"
    _write_episodes(episodes_path, outcomes, planner)
    aggregate = _aggregate(outcomes, planner)
    summary = {
        "split": "validation",
        "episode_count": 100,
        "planners": [planner],
        "conditions": ["R2ASV"],
        "by_planner": [aggregate],
        "by_planner_condition": [{**aggregate, "condition_id": "R2ASV"}],
        "evaluation_provenance": {
            "schema_version": 1,
            "model_path": "artifacts/models/model.zip",
            "model_sha256": sha256_file(model_path),
            "model_variant": "fixture_v1",
            "training_seed": 11,
            "split": "validation",
            "condition_id": "R2ASV",
            "planner": "ppo",
            "planner_label": planner,
            "scenario_seed_start": 5_000,
            "scenario_seed_count": 100,
            "scenario_seeds": list(range(5_000, 5_100)),
            "obstacle_count": 1,
            "obstacle_speed_range": [0.1, 0.2],
            "horizontal_route_half_length": 1.1,
            "vertical_route_half_length": 1.1,
            "static_layout": "train_a",
            "safety_shield_enabled": True,
            "safety_shield_config": asdict(SafetyShieldConfig()),
            "reset_health": {
                "attempts": 100,
                "successful": 100,
                "finite_settle_checks": 200,
                "readback_checks": 100,
            },
            "episodes_csv_sha256": sha256_file(episodes_path),
        },
    }
    _write_json(summary_path, summary)
    request = {
        "condition": "R2ASV",
        "episodes": 100,
        "seed": 11,
        "seed_start": 5_000,
        "scenario_seeds": list(range(5_000, 5_100)),
        "model_variant": "fixture_v1",
        "steps": [10_240],
        "obstacle_count": 1,
        "obstacle_speed_range": [0.1, 0.2],
        "horizontal_route_half_length": 1.1,
        "vertical_route_half_length": 1.1,
        "static_layout": "train_a",
        "action_repeat": 4,
        "max_episode_steps": 300,
        "port_base": 14_020,
        "workers": 1,
        "safety_shield_enabled": True,
        "safety_shield_profile": profile.manifest_record(root),
    }
    environment = {
        "WEBOTS_EVAL_MODEL": "artifacts/models/model.zip",
        "WEBOTS_EVAL_MODEL_SHA256": sha256_file(model_path),
        "WEBOTS_EVAL_RUN_NAME": run_name,
        "WEBOTS_EVAL_CONDITION": "R2ASV",
        "WEBOTS_EVAL_SEED_START": "5000",
        "WEBOTS_EVAL_EPISODES": "100",
        "WEBOTS_SAFETY_SHIELD": "true",
        **profile.controller_environment,
    }
    run = {
        "run_name": run_name,
        "status": "completed",
        "exit_code": 0,
        "step": 10_240,
        "port": 14_020,
        "planner": planner,
        "environment": environment,
        "started_at": "2999-01-01T00:00:00+00:00",
        "finished_at": "2999-01-01T00:01:00+00:00",
        "model_path": "artifacts/models/model.zip",
        "model_sha256": sha256_file(model_path),
        "summary_path": summary_path.relative_to(root).as_posix(),
        "summary_sha256": sha256_file(summary_path),
    }
    runs = [run, {**run, "run_name": "unexpected"}] if extra_run else [run]
    manifest = {
        "schema_version": 1,
        "kind": "webots_checkpoint_validation_sweep",
        "split": "validation",
        "status": "completed",
        "request": request,
        "runtime": {
            "evaluation_world": "worlds/evaluate.wbt",
            "evaluation_world_sha256": sha256_file(root / "worlds/evaluate.wbt"),
            "evaluator_controller": "controllers/evaluate.py",
            "evaluator_controller_sha256": sha256_file(root / "controllers/evaluate.py"),
        },
        "runs": runs,
    }
    _write_json(root / "artifacts/results/fixed_sweep/checkpoint_sweep.json", manifest)
    _commit(root, "record fixed validation evidence")
    return config_path


def _stub_paired_recomputation(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    mismatch: bool = False,
) -> None:
    monkeypatch.setattr(
        fixed_validation,
        "load_paired_safety_config",
        lambda *_args, **_kwargs: object(),
    )

    def recompute(_config: object) -> dict[str, Any]:
        report_path = root / "artifacts/results/paired/paired_safety_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["source_bindings"]["git_revision"] = _git(root, "rev-parse", "HEAD")
        if mismatch:
            report["paired_validation"]["episode_count"] += 1
        return report

    monkeypatch.setattr(fixed_validation, "build_paired_safety_report", recompute)


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "known Windows gap: the R2a-audit-frozen write path records LF digests "
        "while text-mode writes emit CRLF; fixing it requires the next rebaseline"
    ),
)
def test_pass_writes_bound_system_deployment_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    config_path = _build_fixture(root)
    _stub_paired_recomputation(monkeypatch, root)

    report = analyze_fixed_safety_validation(config_path, repository_root=root)

    assert report["decision"]["passed"] is True
    assert report["decision"]["system_deployment_profile_eligible"] is True
    assert "system_deployment_profile_created" not in report["decision"]
    assert report["metrics"]["success_count"] == 80
    assert report["metrics"]["collision_count"] == 20
    assert report["source_bindings"]["shield_profile"]["resolved_full_config"][
        "lidar_max_range"
    ] == pytest.approx(SafetyShieldConfig().lidar_max_range)
    report_json = root / "artifacts/results/fixed_report/fixed_safety_report.json"
    deployment_json = (
        root / "artifacts/deployments/fixed_system/system_deployment_profile.json"
    )
    deployment = json.loads(deployment_json.read_text(encoding="utf-8"))
    assert deployment["qualification"]["fixed_safety_report"]["sha256"] == sha256_file(
        report_json
    )
    assert deployment["system"]["model"]["sha256"] == sha256_file(
        root / "artifacts/models/model.zip"
    )
    assert deployment["qualification"]["paired_diagnostic_config"]["sha256"] == sha256_file(
        root / "configs/eval/paired.yaml"
    )
    assert deployment["restrictions"] == {
        "formal_model_promotion": "forbidden",
        "model_and_shield_must_be_deployed_together": True,
        "policy_only_alias_created": False,
    }


def test_gate_failure_writes_report_without_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "failed-repository"
    config_path = _build_fixture(root, success_count=74, collision_count=26)
    _stub_paired_recomputation(monkeypatch, root)

    report = analyze_fixed_safety_validation(config_path, repository_root=root)

    assert report["decision"]["passed"] is False
    assert report["gate"]["checks"]["minimum_success_rate"]["passed"] is False
    assert (root / "artifacts/results/fixed_report/fixed_safety_report.json").is_file()
    assert not (root / "artifacts/deployments/fixed_system").exists()


def test_rejects_unqualified_paired_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "unqualified-repository"
    config_path = _build_fixture(root, diagnostic_passed=False)
    _stub_paired_recomputation(monkeypatch, root)

    with pytest.raises(ValueError, match="did not pass|required passed decision"):
        analyze_fixed_safety_validation(config_path, repository_root=root)


def test_rejects_saved_vs_recomputed_paired_report_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "mismatched-paired-repository"
    config_path = _build_fixture(root)
    _stub_paired_recomputation(monkeypatch, root, mismatch=True)

    with pytest.raises(ValueError, match="does not match the recomputed evidence"):
        analyze_fixed_safety_validation(config_path, repository_root=root)


@pytest.mark.parametrize("dirty_kind", ["validator", "untracked"])
def test_rejects_any_dirty_repository_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dirty_kind: str,
) -> None:
    root = tmp_path / f"dirty-{dirty_kind}-repository"
    config_path = _build_fixture(root)
    _stub_paired_recomputation(monkeypatch, root)
    if dirty_kind == "validator":
        validator = root / "src/indoor_patrol_rl/evaluation/fixed_safety_validation.py"
        validator.write_text("# dirty validator\n", encoding="utf-8")
    else:
        untracked = root / "src/untracked_runtime_source.py"
        untracked.parent.mkdir(parents=True, exist_ok=True)
        untracked.write_text("# untracked\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completely clean repository"):
        analyze_fixed_safety_validation(config_path, repository_root=root)


def test_rejects_tampered_bound_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tampered-repository"
    config_path = _build_fixture(root)
    _stub_paired_recomputation(monkeypatch, root)
    episodes_path = next((root / "artifacts/results").glob("webots_*/*episodes.csv"))
    episodes_path.write_text(episodes_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _commit(root, "tamper fixed episodes")

    with pytest.raises(ValueError, match="episodes.csv SHA-256 mismatch"):
        analyze_fixed_safety_validation(config_path, repository_root=root)


def test_rejects_more_than_one_sweep_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "multiple-run-repository"
    config_path = _build_fixture(root, extra_run=True)
    _stub_paired_recomputation(monkeypatch, root)

    with pytest.raises(ValueError, match="exactly one run"):
        analyze_fixed_safety_validation(config_path, repository_root=root)


def test_refuses_to_overwrite_outputs_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "overwrite-repository"
    config_path = _build_fixture(root)
    _stub_paired_recomputation(monkeypatch, root)
    analyze_fixed_safety_validation(config_path, repository_root=root)
    _commit(root, "record fixed analysis outputs")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_fixed_safety_validation(config_path, repository_root=root)
