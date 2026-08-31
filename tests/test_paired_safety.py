from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from indoor_patrol_rl.core import SafetyShieldConfig
from indoor_patrol_rl.evaluation.checkpoint_selection import sha256_file
from indoor_patrol_rl.evaluation.checkpoint_sweep import load_safety_shield_profile
from indoor_patrol_rl.evaluation.paired_safety import analyze_paired_safety

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


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_profile(root: Path, name: str, *, enabled: bool) -> Path:
    config = asdict(SafetyShieldConfig())
    config.pop("lidar_max_range")
    path = root / "configs/eval" / name
    _write_yaml(
        path,
        {
            "schema_version": 1,
            "profile_id": "fixture-brake-v1",
            "enabled": enabled,
            "config": config,
        },
    )
    return path


def _write_episodes(
    path: Path,
    outcomes: tuple[str, ...],
    completion_times: dict[int, float],
    *,
    seed_start: int,
    planner: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for offset, outcome in enumerate(outcomes):
            seed = seed_start + offset
            success = outcome == "success"
            writer.writerow(
                {
                    "schema_version": "1.0",
                    "episode_id": f"validation-R2AD-{seed}-{planner}",
                    "split": "validation",
                    "condition_id": "R2AD",
                    "scenario_seed": seed,
                    "planner": planner,
                    "outcome": outcome,
                    "is_success": str(success),
                    "collision": str(outcome == "collision"),
                    "timeout": str(outcome == "timeout"),
                    "step_count": 10,
                    "elapsed_sim_s": completion_times.get(seed, 2.0),
                    "completion_time_s": completion_times.get(seed, "") if success else "",
                    "episode_return": 1.0,
                    "path_length_m": 2.0,
                    "reference_path_length_m": 2.0 + offset / 10,
                    "path_length_ratio": 1.0 if success else "",
                    "minimum_clearance_m": 0.2,
                    "near_collision_ratio": 0.0,
                    "action_delta_sq_mean": 0.1,
                    "inference_latency_p50_ms": 0.1,
                    "inference_latency_p95_ms": 0.2,
                    "error_type": "fixture_error" if outcome == "error" else "",
                }
            )


def _aggregate(
    outcomes: tuple[str, ...], completion_times: dict[int, float], seed_start: int
) -> dict[str, Any]:
    total = len(outcomes)
    counts = {name: outcomes.count(name) for name in ("success", "collision", "timeout", "error")}
    successful_times = [
        completion_times[seed_start + index]
        for index, outcome in enumerate(outcomes)
        if outcome == "success"
    ]
    return {
        "planner": "ppo_r2ad_step10",
        "episodes": total,
        "success_count": counts["success"],
        "collision_count": counts["collision"],
        "timeout_count": counts["timeout"],
        "error_count": counts["error"],
        "success_rate": counts["success"] / total,
        "collision_rate": counts["collision"] / total,
        "timeout_rate": counts["timeout"] / total,
        "completion_time_s_mean": sum(successful_times) / len(successful_times),
    }


def _build_fixture(root: Path, *, minimum_success_rate: float = 0.5) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    (root / "README.md").write_text("paired fixture\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-q",
        "-m",
        "fixture revision",
    )
    feature_revision = _git(root, "rev-parse", "HEAD")

    model_path = root / "artifacts/models/model.zip"
    world_path = root / "worlds/evaluate.wbt"
    controller_path = root / "controllers/evaluate.py"
    for path, contents in (
        (model_path, b"model fixture\n"),
        (world_path, b"#VRML_SIM fixture\n"),
        (controller_path, b"# controller fixture\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    reference_profile_path = _write_profile(root, "reference.yaml", enabled=False)
    candidate_profile_path = _write_profile(root, "candidate.yaml", enabled=True)
    reference_profile = load_safety_shield_profile(reference_profile_path, repository_root=root)
    candidate_profile = load_safety_shield_profile(candidate_profile_path, repository_root=root)

    seed_start = 72_542
    episode_count = 120
    planner = "ppo_r2ad_step10"
    reference_outcomes = (
        "collision",
        "success",
        "success",
        "collision",
        *("success" for _ in range(episode_count - 4)),
    )
    candidate_outcomes = (
        "success",
        "collision",
        "success",
        "collision",
        *("success" for _ in range(episode_count - 4)),
    )

    def completion_times(
        outcomes: tuple[str, ...], overrides: dict[int, float]
    ) -> dict[int, float]:
        values = {
            seed_start + index: 10.0
            for index, outcome in enumerate(outcomes)
            if outcome == "success"
        }
        values.update(overrides)
        return values

    arm_inputs = (
        (
            "fixture_reference",
            "reference",
            15000,
            reference_profile,
            reference_outcomes,
            completion_times(
                reference_outcomes,
                {seed_start + 1: 9.0, seed_start + 2: 10.0},
            ),
        ),
        (
            "fixture_candidate",
            "candidate",
            15010,
            candidate_profile,
            candidate_outcomes,
            completion_times(
                candidate_outcomes,
                {seed_start: 8.0, seed_start + 2: 11.0},
            ),
        ),
    )
    arm_configurations: list[dict[str, Any]] = []
    for arm_id, role, port, profile, outcomes, completion_times in arm_inputs:
        run_name = f"webots_validation_r2ad_checkpoint10_seed11_shield-{profile.result_token}"
        run_directory = root / "artifacts/results" / run_name
        episodes_path = run_directory / "episodes.csv"
        summary_path = run_directory / "summary.json"
        _write_episodes(
            episodes_path,
            outcomes,
            completion_times,
            seed_start=seed_start,
            planner=planner,
        )
        aggregate = _aggregate(outcomes, completion_times, seed_start)
        summary = {
            "split": "validation",
            "episode_count": episode_count,
            "planners": [planner],
            "conditions": ["R2AD"],
            "by_planner": [aggregate],
            "by_planner_condition": [{**aggregate, "condition_id": "R2AD"}],
            "evaluation_provenance": {
                "schema_version": 1,
                "model_path": "artifacts/models/model.zip",
                "model_sha256": sha256_file(model_path),
                "model_variant": "fixture_v1",
                "training_seed": 11,
                "split": "validation",
                "condition_id": "R2AD",
                "planner": "ppo",
                "planner_label": planner,
                "scenario_seed_start": seed_start,
                "scenario_seed_count": episode_count,
                "scenario_seeds": list(range(seed_start, seed_start + episode_count)),
                "obstacle_count": 1,
                "obstacle_speed_range": [0.1, 0.2],
                "horizontal_route_half_length": 1.1,
                "vertical_route_half_length": 1.1,
                "static_layout": "train_a",
                "safety_shield_enabled": profile.enabled,
                "safety_shield_config": asdict(SafetyShieldConfig()),
                "reset_health": {
                    "attempts": episode_count,
                    "successful": episode_count,
                    "finite_settle_checks": 2 * episode_count,
                    "readback_checks": episode_count,
                },
                "episodes_csv_sha256": sha256_file(episodes_path),
            },
        }
        _write_json(summary_path, summary)
        sweep_relative = f"artifacts/results/{role}_sweep"
        sweep_path = root / sweep_relative / "checkpoint_sweep.json"
        manifest = {
            "schema_version": 1,
            "kind": "webots_checkpoint_validation_sweep",
            "split": "validation",
            "status": "completed",
            "request": {
                "condition": "R2AD",
                "episodes": episode_count,
                "seed": 11,
                "seed_start": seed_start,
                "scenario_seeds": list(range(seed_start, seed_start + episode_count)),
                "model_variant": "fixture_v1",
                "obstacle_count": 1,
                "obstacle_speed_range": [0.1, 0.2],
                "horizontal_route_half_length": 1.1,
                "vertical_route_half_length": 1.1,
                "static_layout": "train_a",
                "action_repeat": 4,
                "max_episode_steps": 300,
                "safety_shield_enabled": profile.enabled,
                "safety_shield_profile": profile.manifest_record(root),
            },
            "runtime": {
                "evaluation_world": "worlds/evaluate.wbt",
                "evaluation_world_sha256": sha256_file(world_path),
                "evaluator_controller": "controllers/evaluate.py",
                "evaluator_controller_sha256": sha256_file(controller_path),
            },
            "runs": [
                {
                    "run_name": run_name,
                    "status": "completed",
                    "exit_code": 0,
                    "started_at": "2999-01-01T00:00:00+00:00",
                    "finished_at": "2999-01-01T00:01:00+00:00",
                    "model_path": "artifacts/models/model.zip",
                    "model_sha256": sha256_file(model_path),
                    "summary_path": summary_path.relative_to(root).as_posix(),
                    "summary_sha256": sha256_file(summary_path),
                }
            ],
        }
        _write_json(sweep_path, manifest)
        arm_configurations.append(
            {
                "id": arm_id,
                "role": role,
                "port_base": port,
                "shield_profile": {
                    "path": profile.source_path.relative_to(root).as_posix(),
                    "source_sha256": profile.source_sha256,
                    "semantic_sha256": profile.semantic_sha256,
                    "result_token": profile.result_token,
                    "enabled": profile.enabled,
                },
                "run_name": run_name,
                "sweep_directory": sweep_relative,
            }
        )

    config = {
        "schema_version": 1,
        "experiment_id": "fixture_paired_diagnostic_v1",
        "purpose": "exploratory_safety_profile_diagnostic",
        "split": "validation",
        "condition_id": "R2AD",
        "sweep_feature_revision": feature_revision,
        "training_seed": 11,
        "scenario_seed_start": seed_start,
        "episodes": episode_count,
        "model": {
            "path": "artifacts/models/model.zip",
            "sha256": sha256_file(model_path),
            "variant": "fixture_v1",
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
            "world": {"path": "worlds/evaluate.wbt", "sha256": sha256_file(world_path)},
            "evaluator": {
                "path": "controllers/evaluate.py",
                "sha256": sha256_file(controller_path),
            },
        },
        "arms": arm_configurations,
        "preregistration": {
            "paired_same_scenarios": True,
            "same_model_and_controller": True,
            "only_operational_difference": "safety_shield_enabled",
            "candidate_count": 1,
            "parameter_scan": "forbidden",
            "adaptive_extension": "forbidden",
            "held_out_formal_test_access": "forbidden",
            "seed_window_selection": {
                "search_start": 60_000,
                "selected_start": seed_start,
                "rule": "first_contiguous_120_balancing_route_reverse_and_obstacle_direction",
                "covariates_only": [
                    "route_axis",
                    "reverse_route",
                    "initial_obstacle_direction",
                ],
                "outcome_data_used": False,
                "expected_route_strata_counts": {
                    "horizontal_non_reverse": 30,
                    "horizontal_reverse": 30,
                    "vertical_non_reverse": 30,
                    "vertical_reverse": 30,
                },
                "expected_initial_obstacle_direction_counts": {
                    "negative": 60,
                    "positive": 60,
                },
            },
        },
        "candidate_gate": {
            "require_all": True,
            "minimum_success_rate": minimum_success_rate,
            "maximum_collision_rate": 0.5,
            "minimum_relative_collision_reduction": 0.0,
            "exact_one_sided_mcnemar_significance_level": 1.0,
            "maximum_new_timeout_count": 0,
            "maximum_median_completion_time_ratio": 1.2,
            "minimum_common_success_count": 1,
            "maximum_error_count_per_arm": 0,
        },
        "decision": {
            "pass": "retain_fixture_profile",
            "fail": "reject_fixture_profile",
            "formal_model_promotion": "forbidden",
            "output_directory": "artifacts/results/paired_fixture",
        },
    }
    config_path = root / "configs/eval/paired.yaml"
    _write_yaml(config_path, config)
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
        "preregister paired fixture",
    )
    preregistration_revision = _git(root, "rev-parse", "HEAD")
    config_blob_oid = _git(
        root,
        "rev-parse",
        f"{preregistration_revision}:configs/eval/paired.yaml",
    )
    _write_yaml(
        config_path.with_suffix(".binding.yaml"),
        {
            "schema_version": 1,
            "kind": "paired_safety_preregistration_git_binding",
            "config_path": "configs/eval/paired.yaml",
            "preregistration_revision": preregistration_revision,
            "config_blob_oid": config_blob_oid,
            "config_sha256": sha256_file(config_path),
        },
    )
    _git(root, "add", "configs/eval/paired.binding.yaml")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-q",
        "-m",
        "bind paired fixture preregistration",
    )
    return config_path


@pytest.fixture
def paired_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository"
    return root, _build_fixture(root)


def test_paired_analysis_recomputes_metrics_and_writes_bound_reports(
    paired_fixture: tuple[Path, Path],
) -> None:
    root, config_path = paired_fixture
    output = root / "custom/output"
    markdown = root / "custom/report.md"

    report = analyze_paired_safety(
        config_path,
        repository_root=root,
        output_directory=output,
        markdown_path=markdown,
    )

    assert (
        report["paired_outcomes"]
        | {
            "rescue_count": 1,
            "regression_count": 1,
            "common_success_count": 117,
            "common_collision_count": 1,
        }
        == report["paired_outcomes"]
    )
    assert report["collision_analysis"]["exact_one_sided_mcnemar_p"] == 0.75
    assert report["common_success_completion_time"][
        "median_candidate_to_reference_ratio"
    ] == pytest.approx(1.0)
    assert report["decision"] == {
        "passed": True,
        "outcome": "retain_fixture_profile",
        "formal_model_promotion": "forbidden",
        "model_alias_created": False,
    }
    assert report["source_bindings"]["input_config"]["sha256"] == sha256_file(config_path)
    git_binding = report["source_bindings"]["input_config"]["git_binding"]
    assert git_binding["config_clean"] is True
    assert git_binding["committed_before_both_runs"] is True
    assert git_binding["preregistration_revision"] == _git(
        root, "rev-parse", "HEAD^"
    )
    assert report["paired_validation"]["seed_window_rule_recomputed"] is True
    assert report["paired_validation"]["first_balanced_seed_window_start"] == 72_542
    assert (
        report["provenance_limits"][
            "evaluation_runtime_worktree_state_cryptographically_bound"
        ]
        is False
    )
    assert report["provenance_limits"]["webots_executable_binary_sha256_bound"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "paired_safety_report.json",
        "paired_safety_report.sha256",
        "paired_safety_report.yaml",
    ]
    assert markdown.is_file()


def test_paired_analysis_rejects_tampered_bound_csv(
    paired_fixture: tuple[Path, Path],
) -> None:
    root, config_path = paired_fixture
    candidate_csv = next(
        path
        for path in (root / "artifacts/results").glob("webots_*/*episodes.csv")
        if "354" not in path.as_posix()
    )
    candidate_csv.write_text(
        candidate_csv.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="episodes.csv SHA-256 mismatch"):
        analyze_paired_safety(
            config_path,
            repository_root=root,
            output_directory=root / "output",
            markdown_path=root / "report.md",
        )


def test_gate_failure_is_reported_without_becoming_a_validation_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "failed-repository"
    config_path = _build_fixture(root, minimum_success_rate=0.99)

    report = analyze_paired_safety(
        config_path,
        repository_root=root,
        output_directory=root / "failed/output",
        markdown_path=root / "failed/report.md",
    )

    assert report["decision"]["passed"] is False
    assert report["decision"]["outcome"] == "reject_fixture_profile"
    assert report["gate"]["checks"]["minimum_success_rate"]["passed"] is False
    assert (root / "failed/output/paired_safety_report.json").is_file()


def test_report_writer_refuses_to_overwrite_by_default(
    paired_fixture: tuple[Path, Path],
) -> None:
    root, config_path = paired_fixture
    kwargs = {
        "repository_root": root,
        "output_directory": root / "output",
        "markdown_path": root / "report.md",
    }
    analyze_paired_safety(config_path, **kwargs)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_paired_safety(config_path, **kwargs)


def test_analysis_rejects_dirty_preregistered_config(
    paired_fixture: tuple[Path, Path],
) -> None:
    root, config_path = paired_fixture
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["candidate_gate"]["minimum_success_rate"] = 0.99
    _write_yaml(config_path, config)

    with pytest.raises(ValueError, match="pre-registered blob|tracked and clean|must be clean"):
        analyze_paired_safety(
            config_path,
            repository_root=root,
            output_directory=root / "output",
            markdown_path=root / "report.md",
        )


def test_analysis_recomputes_first_balanced_seed_window(tmp_path: Path) -> None:
    root = tmp_path / "wrong-window-repository"
    config_path = _build_fixture(root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["scenario_seed_start"] = 72_543
    config["preregistration"]["seed_window_selection"]["selected_start"] = 72_543
    _write_yaml(config_path, config)
    _git(root, "add", "configs/eval/paired.yaml")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-q",
        "-m",
        "preregister wrong seed window",
    )
    preregistration_revision = _git(root, "rev-parse", "HEAD")
    binding_path = config_path.with_suffix(".binding.yaml")
    binding = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    binding["preregistration_revision"] = preregistration_revision
    binding["config_blob_oid"] = _git(
        root,
        "rev-parse",
        f"{preregistration_revision}:configs/eval/paired.yaml",
    )
    binding["config_sha256"] = sha256_file(config_path)
    _write_yaml(binding_path, binding)
    _git(root, "add", "configs/eval/paired.binding.yaml")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-q",
        "-m",
        "bind wrong seed window",
    )

    with pytest.raises(ValueError, match="not the first covariate-balanced"):
        analyze_paired_safety(
            config_path,
            repository_root=root,
            output_directory=root / "output",
            markdown_path=root / "report.md",
        )
