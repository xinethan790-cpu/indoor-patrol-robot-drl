from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from indoor_patrol_rl.evaluation import (
    RISK_TRACE_COORDINATE_SEMANTICS,
    EpisodeRecord,
    write_benchmark_outputs,
    write_episode_context_jsonl,
)

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/run_webots_diagnostic_audit.py"


def _load_runner() -> Any:
    name = "_test_run_webots_diagnostic_audit"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _sha256(path: Path) -> str:
    return RUNNER._sha256_file(path)


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "repository"
    monitor = (
        root
        / "artifacts/logs/webots/seed_11/w1_horizontal_recovery_v1/training.monitor.csv"
    )
    reference = (
        root
        / "artifacts/models/webots/seed_11/w1_recovery_v1/ppo_webots_s11_step000040960.zip"
    )
    candidate = (
        root
        / "artifacts/models/webots/seed_11/w1_horizontal_recovery_v1/checkpoints/"
        "ppo_webots_s11_w1_horizontal_recovery_v1_30720_steps.zip"
    )
    world = root / "worlds/webots_ppo_evaluate.wbt"
    webots = root / "tools/webots.exe"
    for path, payload in (
        (monitor, b"monitor"),
        (reference, b"reference"),
        (candidate, b"candidate"),
        (world, b"world"),
        (webots, b"webots"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    config: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "webots_h1_checkpoint_audit_v1",
        "split": "diagnostic",
        "condition_id": "DH1",
        "training_seed": 11,
        "scenario_seed_start": 20_300,
        "episodes": 40,
        "scenario": {
            "obstacle_count": 1,
            "obstacle_speed_range": [0.15, 0.30],
            "horizontal_route_half_length": 1.10,
            "vertical_route_half_length": 1.10,
            "safety_shield": False,
            "deterministic_policy": True,
        },
        "selection_basis": {
            "source_monitor": monitor.relative_to(root).as_posix(),
            "source_monitor_sha256": _sha256(monitor),
            "block_size_steps": 10_240,
            "rule": "highest_non_overlapping_training_block_success_rate",
            "selected_checkpoint_step": 30_720,
            "selected_block_success_rate": 0.6333333333333333,
            "audit_window_results_seen_when_selected": False,
        },
        "arms": [
            {
                "id": "reference",
                "role": "reference",
                "model": reference.relative_to(root).as_posix(),
                "model_sha256": _sha256(reference),
                "model_variant": "w1_recovery_v1",
                "run_name": "audit_reference",
                "planner_label": "ppo_reference",
            },
            {
                "id": "candidate",
                "role": "candidate",
                "model": candidate.relative_to(root).as_posix(),
                "model_sha256": _sha256(candidate),
                "model_variant": "w1_horizontal_recovery_v1",
                "run_name": "audit_candidate",
                "planner_label": "ppo_candidate",
            },
        ],
        "candidate_gate": {
            "require_all": True,
            "minimum_success_count": 28,
            "maximum_collision_count": 12,
            "minimum_horizontal_success_count": 13,
            "minimum_vertical_success_count": 14,
            "minimum_horizontal_paired_net_rescues": 2,
            "maximum_vertical_paired_net_regressions": 1,
        },
        "decision": {
            "pass": "use_h1_step30720_as_next_bridge_source",
            "fail": "reject_entire_h1_branch",
            "validation_or_test_access": "forbidden",
        },
    }
    config_path = root / "configs/eval/audit.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, webots, config


def test_dry_run_is_diagnostic_only_and_writes_nothing(tmp_path: Path) -> None:
    root, webots, _raw = _fixture(tmp_path)

    exit_code, plan = RUNNER.execute_audit(
        repository_root=root,
        config_path=Path("configs/eval/audit.yaml"),
        webots_executable=webots,
        dry_run=True,
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        source_verifier=lambda _root: "a" * 40,
    )

    assert exit_code == 0
    assert plan["split"] == "diagnostic"
    assert plan["axis_counts"] == {"horizontal": 20, "vertical": 20}
    assert len(plan["runs"]) == 2
    for run in plan["runs"]:
        environment = run["environment"]
        assert environment["WEBOTS_EVAL_SPLIT"] == "diagnostic"
        assert environment["WEBOTS_EVAL_DETERMINISTIC_POLICY"] == "1"
        assert environment["WEBOTS_SAFETY_SHIELD"] == "0"
        assert "WEBOTS_CONFIRM_TEST" not in environment
    assert not (root / "artifacts/results").exists()


def test_preflight_rejects_reserved_seed_and_sha_drift(tmp_path: Path) -> None:
    root, _webots, raw = _fixture(tmp_path)
    config_path = root / "configs/eval/audit.yaml"
    raw["scenario_seed_start"] = 5_000
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="reserved"):
        RUNNER.load_audit_config(config_path, repository_root=root)

    raw["scenario_seed_start"] = 20_300
    raw["arms"][0]["model_sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        RUNNER.load_audit_config(config_path, repository_root=root)


def _episode(seed: int, planner: str, outcome: str = "success") -> EpisodeRecord:
    success = outcome == "success"
    collision = outcome == "collision"
    timeout = outcome == "timeout"
    return EpisodeRecord(
        schema_version="1.0",
        episode_id=f"diagnostic-DH1-{seed}-{planner}",
        split="diagnostic",
        condition_id="DH1",
        scenario_seed=seed,
        planner=planner,
        outcome=outcome,  # type: ignore[arg-type]
        is_success=success,
        collision=collision,
        timeout=timeout,
        step_count=10,
        elapsed_sim_s=1.0,
        completion_time_s=1.0 if success else None,
        episode_return=1.0,
        path_length_m=1.0,
        reference_path_length_m=1.0,
        path_length_ratio=1.0 if success else None,
        minimum_clearance_m=0.3,
        near_collision_ratio=0.0,
        action_delta_sq_mean=0.0,
        inference_latency_p50_ms=0.1,
        inference_latency_p95_ms=0.2,
        error_type=None,
    )


def _risk_trace() -> dict[str, Any]:
    sample = {
        "step": 10,
        "post_elapsed_s": 1.28,
        "post_robot_root_xy": [1.03, 0.0],
        "post_robot_gps_xy": [1.0, 0.0],
        "post_robot_yaw": 0.0,
        "post_goal_distance": 0.1,
        "post_global_clearance": 0.3,
        "pre_action_front_clearance": 0.4,
        "pre_action_left_clearance": 0.5,
        "pre_action_right_clearance": 0.6,
        "post_lidar_argmin": {"range_m": 0.4, "bearing_rad": 0.0},
        "post_obstacle_xy": [[0.0, 0.2]],
        "post_obstacle_motion": [
            {"motion_axis": "y", "speed_mps": 0.2, "direction": 1.0}
        ],
        "interval_policy_action": [0.1, 0.0],
        "interval_executed_action": [0.05, 0.0],
        "interval_commanded_linear": 0.12,
        "interval_commanded_angular": 0.0,
        "post_contact_collision": False,
        "post_lidar_collision": False,
    }
    return {
        "sample_period_s": 0.128,
        "sample_semantics": (
            "pre_action_sector_and_interval_action_with_post_control_state_v1"
        ),
        "coordinate_semantics": RISK_TRACE_COORDINATE_SEMANTICS,
        "lidar_min_range_m": 0.12,
        "lidar_max_range_m": 3.5,
        "window_steps": 16,
        "terminal_window": [sample],
        "minimum_clearance_window": [sample],
    }


def _fake_webots(repository_root: Path):
    def run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        planner = environment["WEBOTS_EVAL_PLANNER_LABEL"]
        run_name = environment["WEBOTS_EVAL_RUN_NAME"]
        output = repository_root / "artifacts/results" / run_name
        seeds = range(
            int(environment["WEBOTS_EVAL_SEED_START"]),
            int(environment["WEBOTS_EVAL_SEED_START"])
            + int(environment["WEBOTS_EVAL_EPISODES"]),
        )
        episodes = [_episode(seed, planner) for seed in seeds]
        contexts = [
            {
                "schema_version": "1.1",
                "identity": {
                    "episode_id": record.episode_id,
                    "split": "diagnostic",
                    "condition_id": "DH1",
                    "scenario_seed": record.scenario_seed,
                    "planner": planner,
                },
                "route_axis": str(
                    np.random.default_rng(record.scenario_seed).choice(
                        ["horizontal", "vertical"]
                    )
                ),
                "reverse_route": False,
                "start_xy": [-1.1, 0.0],
                "goal_xy": [1.1, 0.0],
                "start_yaw": 0.0,
                "dynamic_obstacles": [
                    {
                        "def_name": "DYNAMIC_OBSTACLE_1",
                        "motion_axis": "y",
                        "initial_xy": [0.0, 0.0],
                        "speed": 0.2,
                        "initial_direction": 1.0,
                        "lower": -1.05,
                        "upper": 1.05,
                    }
                ],
                "termination": {
                    "contact_collision": False,
                    "lidar_collision": False,
                    "terminal_step": 10,
                },
                "risk_trace": _risk_trace(),
            }
            for record in episodes
        ]
        context_binding = write_episode_context_jsonl(
            output / "episode_context.jsonl",
            contexts,
            repository_root=repository_root,
        )
        provenance = {
            "schema_version": 1,
            "model_path": environment["WEBOTS_EVAL_MODEL"],
            "model_sha256": environment["WEBOTS_EVAL_MODEL_SHA256"],
            "model_variant": environment["WEBOTS_MODEL_VARIANT"],
            "training_seed": int(environment["WEBOTS_MODEL_SEED"]),
            "split": "diagnostic",
            "condition_id": "DH1",
            "planner": "ppo",
            "planner_label": planner,
            "scenario_seed_start": 20_300,
            "scenario_seed_count": 40,
            "scenario_seeds": list(range(20_300, 20_340)),
            "obstacle_count": 1,
            "obstacle_speed_range": [0.15, 0.3],
            "horizontal_route_half_length": 1.1,
            "vertical_route_half_length": 1.1,
            "safety_shield_enabled": False,
            "deterministic_policy": True,
            "episode_context": context_binding,
            "reset_health": {
                "attempts": 40,
                "successful": 40,
                "finite_settle_checks": 80,
                "readback_checks": 40,
            },
        }
        write_benchmark_outputs(output, episodes, provenance=provenance)
        provenance["episodes_csv_sha256"] = _sha256(output / "episodes.csv")
        write_benchmark_outputs(output, episodes, provenance=provenance)
        return subprocess.CompletedProcess(command, 0)

    return run


def test_gate_failure_is_nonzero_and_preserves_sha_bound_report(tmp_path: Path) -> None:
    root, webots, _raw = _fixture(tmp_path)

    exit_code, report = RUNNER.execute_audit(
        repository_root=root,
        config_path=Path("configs/eval/audit.yaml"),
        webots_executable=webots,
        dry_run=False,
        process_runner=_fake_webots(root),
        source_verifier=lambda _root: "a" * 40,
    )

    assert exit_code == RUNNER.EXIT_CANDIDATE_GATE_FAILED
    assert report["status"] == "gate_failed"
    assert report["candidate_gate"]["checks"][
        "minimum_horizontal_paired_net_rescues"
    ] == {"actual": 0, "operator": ">=", "threshold": 2, "passed": False}
    output = root / "artifacts/results/webots_h1_checkpoint_audit_v1"
    persisted = json.loads((output / "audit_report.json").read_text(encoding="utf-8"))
    assert persisted["test_split_was_run"] is False
    assert persisted["validation_split_was_run"] is False
    assert (output / "audit_report.yaml").is_file()
    assert "audit_report.json" in (output / "audit_report.sha256").read_text(
        encoding="utf-8"
    )


def test_candidate_gate_uses_all_boundary_conditions() -> None:
    gate = RUNNER.CandidateGate(
        require_all=True,
        minimum_success_count=28,
        maximum_collision_count=12,
        minimum_horizontal_success_count=13,
        minimum_vertical_success_count=14,
        minimum_horizontal_paired_net_rescues=2,
        maximum_vertical_paired_net_regressions=1,
    )
    metrics = {
        "overall": {"success_count": 28, "collision_count": 12},
        "by_route_axis": {
            "horizontal": {"success_count": 13},
            "vertical": {"success_count": 14},
        },
    }
    paired = {
        "paired_success_changes": {
            "horizontal": {"net_rescues": 2},
            "vertical": {"net_regressions": 1},
        }
    }

    report = RUNNER._gate_report(gate, candidate_metrics=metrics, paired=paired)

    assert report["passed"] is True
    assert all(check["passed"] for check in report["checks"].values())
