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
    EPISODE_CONTEXT_SCHEMA_VERSION,
    RISK_TRACE_COORDINATE_SEMANTICS,
    RISK_TRACE_SAMPLE_SEMANTICS,
    RISK_TRACE_WINDOW_STEPS,
    EpisodeRecord,
    write_benchmark_outputs,
    write_episode_context_jsonl,
)

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/run_webots_risk_trajectory.py"


def _load_runner() -> Any:
    name = "_test_run_webots_risk_trajectory"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _sha256(path: Path) -> str:
    return RUNNER._sha256_file(path)


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "repository"
    model = root / RUNNER._FIXED_MODEL
    world = root / RUNNER._WORLD
    controller = root / RUNNER._CONTROLLER
    runner = root / RUNNER._RUNNER
    webots = root / "tools/webots.exe"
    for path, payload in (
        (model, b"locked-recovery-v1-model"),
        (world, b"world"),
        (controller, b"controller"),
        (runner, b"runner"),
        (webots, b"webots"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(RUNNER, "_FIXED_MODEL_SHA256", _sha256(model))
    config: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "webots_recovery_v1_risk_trajectory_v1",
        "split": "diagnostic",
        "condition_id": "DR1",
        "training_seed": 11,
        "scenario_seed_start": 20_494,
        "episodes": 120,
        "model": {
            "path": RUNNER._FIXED_MODEL,
            "sha256": _sha256(model),
            "variant": "w1_recovery_v1",
            "run_name": "risk_trajectory_result",
            "planner_label": "ppo_recovery_v1_risk_trace",
        },
        "scenario": {
            "obstacle_count": 1,
            "obstacle_speed_range": [0.15, 0.30],
            "horizontal_route_half_length": 1.10,
            "vertical_route_half_length": 1.10,
            "safety_shield": False,
            "deterministic_policy": True,
        },
        "collection": {
            "episode_context_schema_version": "1.1",
            "risk_trace_required": True,
            "required_episodes": 120,
            "expected_route_axis_counts": {
                "horizontal": 60,
                "vertical": 60,
            },
            "completion_rule": "complete_artifact_collection_only",
            "performance_gate": "none",
            "adaptive_extension": "forbidden",
        },
        "access": {"validation_or_test": "forbidden"},
    }
    config_path = root / "configs/eval/risk.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, webots, config


def _episode(seed: int, planner: str) -> EpisodeRecord:
    return EpisodeRecord(
        schema_version="1.0",
        episode_id=f"diagnostic-DR1-{seed}-{planner}",
        split="diagnostic",
        condition_id="DR1",
        scenario_seed=seed,
        planner=planner,
        outcome="collision",
        is_success=False,
        collision=True,
        timeout=False,
        step_count=10,
        elapsed_sim_s=1.28,
        completion_time_s=None,
        episode_return=-1.0,
        path_length_m=1.0,
        reference_path_length_m=2.2,
        path_length_ratio=None,
        minimum_clearance_m=0.0,
        near_collision_ratio=0.1,
        action_delta_sq_mean=0.01,
        inference_latency_p50_ms=0.1,
        inference_latency_p95_ms=0.2,
        error_type=None,
    )


def _risk_trace() -> dict[str, Any]:
    sample = {
        "step": 10,
        "post_elapsed_s": 1.28,
        "post_robot_root_xy": [0.0, 0.0],
        "post_robot_gps_xy": [0.0, 0.0],
        "post_robot_yaw": 0.0,
        "post_goal_distance": 1.0,
        "post_global_clearance": 0.0,
        "pre_action_front_clearance": 0.1,
        "pre_action_left_clearance": 0.2,
        "pre_action_right_clearance": 0.2,
        "post_lidar_argmin": {"range_m": 0.05, "bearing_rad": 0.0},
        "post_obstacle_xy": [[0.1, 0.0]],
        "post_obstacle_motion": [
            {"motion_axis": "y", "speed_mps": 0.2, "direction": 1.0}
        ],
        "interval_policy_action": [0.8, 0.1],
        "interval_executed_action": [0.5, 0.05],
        "interval_commanded_linear": 0.2,
        "interval_commanded_angular": 0.1,
        "post_contact_collision": False,
        "post_lidar_collision": True,
    }
    return {
        "sample_period_s": 0.128,
        "sample_semantics": RISK_TRACE_SAMPLE_SEMANTICS,
        "coordinate_semantics": RISK_TRACE_COORDINATE_SEMANTICS,
        "lidar_min_range_m": 0.01,
        "lidar_max_range_m": 3.5,
        "window_steps": RISK_TRACE_WINDOW_STEPS,
        "terminal_window": [sample],
        "minimum_clearance_window": [sample],
    }


def _fake_webots(repository_root: Path, *, corrupt_trace: bool = False):
    def run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        planner = environment["WEBOTS_EVAL_PLANNER_LABEL"]
        seed_start = int(environment["WEBOTS_EVAL_SEED_START"])
        episode_count = int(environment["WEBOTS_EVAL_EPISODES"])
        seeds = range(seed_start, seed_start + episode_count)
        output = (
            repository_root
            / "artifacts/results"
            / environment["WEBOTS_EVAL_RUN_NAME"]
        )
        episodes = [_episode(seed, planner) for seed in seeds]
        contexts: list[dict[str, Any]] = []
        for record in episodes:
            axis = str(
                np.random.default_rng(record.scenario_seed).choice(
                    ["horizontal", "vertical"]
                )
            )
            start_xy = [-1.1, 0.0] if axis == "horizontal" else [0.0, -1.1]
            goal_xy = [1.1, 0.0] if axis == "horizontal" else [0.0, 1.1]
            contexts.append(
                {
                    "schema_version": EPISODE_CONTEXT_SCHEMA_VERSION,
                    "identity": {
                        "episode_id": record.episode_id,
                        "split": "diagnostic",
                        "condition_id": "DR1",
                        "scenario_seed": record.scenario_seed,
                        "planner": planner,
                    },
                    "route_axis": axis,
                    "reverse_route": False,
                    "start_xy": start_xy,
                    "goal_xy": goal_xy,
                    "start_yaw": 0.0,
                    "dynamic_obstacles": [
                        {
                            "def_name": "TRAIN_OBSTACLE_0",
                            "motion_axis": "y" if axis == "horizontal" else "x",
                            "initial_xy": [0.0, 0.0],
                            "speed": 0.2,
                            "initial_direction": 1.0,
                            "lower": -1.05,
                            "upper": 1.05,
                        }
                    ],
                    "termination": {
                        "contact_collision": False,
                        "lidar_collision": True,
                        "terminal_step": 10,
                    },
                    "risk_trace": _risk_trace(),
                }
            )
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
            "condition_id": "DR1",
            "planner": "ppo",
            "planner_label": planner,
            "deterministic_policy": True,
            "scenario_seed_start": seed_start,
            "scenario_seed_count": episode_count,
            "scenario_seeds": list(seeds),
            "obstacle_count": 1,
            "obstacle_speed_range": [0.15, 0.3],
            "horizontal_route_half_length": 1.1,
            "vertical_route_half_length": 1.1,
            "safety_shield_enabled": False,
            "episode_context": context_binding,
            "reset_health": {
                "attempts": episode_count,
                "successful": episode_count,
                "finite_settle_checks": 2 * episode_count,
                "readback_checks": episode_count,
            },
        }
        write_benchmark_outputs(output, episodes, provenance=provenance)
        provenance["episodes_csv_sha256"] = _sha256(output / "episodes.csv")
        write_benchmark_outputs(output, episodes, provenance=provenance)
        if corrupt_trace:
            context_path = output / "episode_context.jsonl"
            lines = context_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["risk_trace"]["window_steps"] = RISK_TRACE_WINDOW_STEPS - 1
            lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
            context_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            summary_path = output / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["evaluation_provenance"]["episode_context"]["sha256"] = _sha256(
                context_path
            )
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    return run


def test_default_protocol_is_fixed_and_axis_balanced() -> None:
    raw = yaml.safe_load(
        (
            REPOSITORY_ROOT
            / "configs/eval/webots_recovery_v1_risk_trajectory_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert raw["scenario_seed_start"] == 20_494
    assert raw["episodes"] == 120
    assert raw["model"]["sha256"] == (
        "c5e7c3de42c7664477097b2734975660fec840ad5b7f45425098158c27cb7275"
    )
    counts = {
        axis: sum(
            RUNNER._axis_for_seed(seed) == axis for seed in range(20_494, 20_614)
        )
        for axis in ("horizontal", "vertical")
    }
    assert counts == {"horizontal": 60, "vertical": 60}


def test_dry_run_writes_nothing_and_never_starts_webots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, _raw = _fixture(tmp_path, monkeypatch)

    exit_code, plan = RUNNER.execute_collection(
        repository_root=root,
        config_path=Path("configs/eval/risk.yaml"),
        webots_executable=webots,
        dry_run=True,
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        source_verifier=lambda _root: "a" * 40,
    )

    assert exit_code == 0
    assert plan["axis_counts"] == {"horizontal": 60, "vertical": 60}
    assert plan["required_episodes"] == 120
    assert plan["adaptive_extension"] == "forbidden"
    assert plan["performance_gate_applied"] is False
    assert plan["validation_or_test_access"] == "forbidden"
    environment = plan["run"]["environment"]
    assert environment["WEBOTS_EVAL_SPLIT"] == "diagnostic"
    assert environment["WEBOTS_EVAL_SEED_START"] == "20494"
    assert environment["WEBOTS_EVAL_EPISODES"] == "120"
    assert environment["WEBOTS_SAFETY_SHIELD"] == "0"
    assert environment["WEBOTS_EVAL_DETERMINISTIC_POLICY"] == "1"
    assert "WEBOTS_CONFIRM_TEST" not in environment
    assert not (root / "artifacts/results").exists()


def test_config_rejects_duplicate_keys_reserved_seeds_and_sha_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _webots, raw = _fixture(tmp_path, monkeypatch)
    config_path = root / "configs/eval/risk.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "episodes: 120\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        RUNNER.load_risk_trajectory_config(config_path, repository_root=root)

    raw["scenario_seed_start"] = 5_000
    raw["episodes"] = 20
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="reserved"):
        RUNNER.load_risk_trajectory_config(config_path, repository_root=root)

    raw["scenario_seed_start"] = 20_494
    raw["episodes"] = 120
    raw["model"]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="locked to the recovery-v1"):
        RUNNER.load_risk_trajectory_config(config_path, repository_root=root)


def test_complete_collection_has_no_performance_gate_and_is_sha_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, _raw = _fixture(tmp_path, monkeypatch)

    exit_code, report = RUNNER.execute_collection(
        repository_root=root,
        config_path=Path("configs/eval/risk.yaml"),
        webots_executable=webots,
        dry_run=False,
        process_runner=_fake_webots(root),
        source_verifier=lambda _root: "a" * 40,
    )

    assert exit_code == 0
    assert report["status"] == "collection_complete"
    assert report["performance_gate_applied"] is False
    assert report["descriptive_metrics_not_used_as_a_gate"]["overall"] == {
        "episodes": 120,
        "success_count": 0,
        "collision_count": 120,
        "timeout_count": 0,
        "error_count": 0,
    }
    assert report["collection"] == {
        "expected_episode_count": 120,
        "episodes_csv_count": 120,
        "episode_context_count": 120,
        "risk_trace_count": 120,
        "episode_context_schema_version": "1.1",
        "expected_route_axis_counts": {"horizontal": 60, "vertical": 60},
        "adaptive_extension": "forbidden",
        "complete": True,
    }
    result_directory = root / "artifacts/results/risk_trajectory_result"
    expected_artifacts = {
        "summary_json": result_directory / "summary.json",
        "episodes_csv": result_directory / "episodes.csv",
        "episode_context_jsonl": result_directory / "episode_context.jsonl",
    }
    assert report["artifacts"]["result_directory"] == (
        "artifacts/results/risk_trajectory_result"
    )
    for name, path in expected_artifacts.items():
        assert report["artifacts"][name] == {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
        }
    assert report["written_reports"] == {
        "json": (
            "artifacts/results/webots_recovery_v1_risk_trajectory_v1/"
            "collection_report.json"
        ),
        "yaml": (
            "artifacts/results/webots_recovery_v1_risk_trajectory_v1/"
            "collection_report.yaml"
        ),
        "sha256_sidecar": (
            "artifacts/results/webots_recovery_v1_risk_trajectory_v1/"
            "collection_report.sha256"
        ),
    }
    assert set(report["source_bindings"]) == {
        "git_revision",
        "config",
        "runner",
        "controller",
        "world",
        "model",
        "webots_executable",
    }
    output = root / "artifacts/results/webots_recovery_v1_risk_trajectory_v1"
    persisted = json.loads((output / "collection_report.json").read_text(encoding="utf-8"))
    assert persisted["validation_split_was_run"] is False
    assert persisted["test_split_was_run"] is False
    assert (output / "collection_report.yaml").is_file()
    sidecar = (output / "collection_report.sha256").read_text(encoding="utf-8")
    assert "collection_report.json" in sidecar
    assert "collection_report.yaml" in sidecar


def test_invalid_risk_trace_is_rejected_even_when_provenance_sha_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, _raw = _fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="window_steps"):
        RUNNER.execute_collection(
            repository_root=root,
            config_path=Path("configs/eval/risk.yaml"),
            webots_executable=webots,
            dry_run=False,
            process_runner=_fake_webots(root, corrupt_trace=True),
            source_verifier=lambda _root: "a" * 40,
        )
    assert not (
        root
        / "artifacts/results/webots_recovery_v1_risk_trajectory_v1/collection_report.json"
    ).exists()


def test_existing_destination_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, _raw = _fixture(tmp_path, monkeypatch)
    destination = root / "artifacts/results/risk_trajectory_result"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        RUNNER.execute_collection(
            repository_root=root,
            config_path=Path("configs/eval/risk.yaml"),
            webots_executable=webots,
            dry_run=True,
            process_runner=lambda *_args, **_kwargs: pytest.fail("started Webots"),
            source_verifier=lambda _root: "a" * 40,
        )
    assert marker.read_text(encoding="utf-8") == "keep"
