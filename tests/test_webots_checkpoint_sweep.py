from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from indoor_patrol_rl.evaluation.checkpoint_selection import sha256_file
from indoor_patrol_rl.evaluation.checkpoint_sweep import (
    WebotsCheckpointSweepRequest,
    run_checkpoint_sweep,
    sweep_exit_code,
)


def _request(
    root: Path,
    *,
    steps: tuple[int, ...] = (10, 20),
    workers: int = 2,
) -> WebotsCheckpointSweepRequest:
    world = root / "worlds/webots_ppo_evaluate.wbt"
    world.parent.mkdir(parents=True, exist_ok=True)
    world.write_text("#VRML_SIM R2025a utf8\n", encoding="utf-8")
    controller = root / "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
    controller.parent.mkdir(parents=True, exist_ok=True)
    controller.write_text("# evaluator fixture\n", encoding="utf-8")
    executable = root / "webots"
    executable.write_bytes(b"fake webots\n")
    checkpoint_directory = (
        root
        / "artifacts/models/webots/seed_11/r2a_targeted_v1/checkpoints"
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (checkpoint_directory / f"ppo_webots_s11_r2a_targeted_v1_{step}_steps.zip").write_bytes(
            f"model-{step}".encode()
        )
    return WebotsCheckpointSweepRequest(
        repository_root=root,
        webots_executable=executable,
        seed=11,
        condition="r2a",
        model_variant="r2a_targeted_v1",
        steps=steps,
        episodes=2,
        seed_start=5000,
        obstacle_count=1,
        obstacle_speed_min=0.10,
        obstacle_speed_max=0.20,
        horizontal_route_half_length=1.10,
        vertical_route_half_length=1.10,
        port_base=14000,
        workers=workers,
        timeout_seconds=30.0,
    )


def _write_summary(
    root: Path,
    environment: Mapping[str, str],
    *,
    speed_range: tuple[float, float] | None = None,
) -> Path:
    output = root / "artifacts/results" / environment["WEBOTS_EVAL_RUN_NAME"]
    output.mkdir(parents=True, exist_ok=True)
    episodes_path = output / "episodes.csv"
    episodes_path.write_text("split,scenario_seed\nvalidation,5000\nvalidation,5001\n")
    episodes = int(environment["WEBOTS_EVAL_EPISODES"])
    seed_start = int(environment["WEBOTS_EVAL_SEED_START"])
    planner = environment["WEBOTS_EVAL_PLANNER_LABEL"]
    condition = environment["WEBOTS_EVAL_CONDITION"]
    model_path = root / environment["WEBOTS_EVAL_MODEL"]
    speeds = speed_range or (
        float(environment["WEBOTS_OBSTACLE_SPEED_MIN"]),
        float(environment["WEBOTS_OBSTACLE_SPEED_MAX"]),
    )
    aggregate = {
        "planner": planner,
        "episodes": episodes,
        "success_count": episodes,
        "collision_count": 0,
        "timeout_count": 0,
        "error_count": 0,
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "path_length_ratio_mean": 1.1,
        "completion_time_s_mean": 4.0,
    }
    summary = {
        "split": "validation",
        "episode_count": episodes,
        "planners": [planner],
        "conditions": [condition],
        "by_planner": [aggregate],
        "by_planner_condition": [{**aggregate, "condition_id": condition}],
        "evaluation_provenance": {
            "schema_version": 1,
            "model_path": environment["WEBOTS_EVAL_MODEL"],
            "model_sha256": sha256_file(model_path),
            "model_variant": environment.get("WEBOTS_MODEL_VARIANT"),
            "training_seed": int(environment["WEBOTS_MODEL_SEED"]),
            "split": "validation",
            "condition_id": condition,
            "scenario_seed_start": seed_start,
            "scenario_seed_count": episodes,
            "scenario_seeds": list(range(seed_start, seed_start + episodes)),
            "obstacle_count": int(environment["WEBOTS_OBSTACLE_COUNT"]),
            "obstacle_speed_range": list(speeds),
            "horizontal_route_half_length": float(
                environment["WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH"]
            ),
            "vertical_route_half_length": float(
                environment["WEBOTS_VERTICAL_ROUTE_HALF_LENGTH"]
            ),
            "safety_shield_enabled": False,
            "static_layout": "train_a",
            "episodes_csv_sha256": sha256_file(episodes_path),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return summary_path


def _sequential_port(preferred: int, reserved: set[int]) -> int:
    port = preferred
    while port in reserved:
        port += 1
    reserved.add(port)
    return port


def test_sweep_runs_checkpoints_with_clean_environment_and_unique_ports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("WEBOTS_STALE_PARENT_VALUE", "must-not-leak")

    def runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert "WEBOTS_STALE_PARENT_VALUE" not in environment
        assert environment["WEBOTS_EVAL_SPLIT"] == "validation"
        assert environment["WEBOTS_EVAL_MODEL_SHA256"] == sha256_file(
            tmp_path / environment["WEBOTS_EVAL_MODEL"]
        )
        calls.append({"command": tuple(command), "environment": dict(environment)})
        _write_summary(tmp_path, environment)
        return subprocess.CompletedProcess(command, 0, "webots ok\n", "")

    manifest = run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/sweep",
        process_runner=runner,
        port_allocator=_sequential_port,
    )

    assert manifest["status"] == "completed"
    assert sweep_exit_code(manifest) == 0
    assert [run["status"] for run in manifest["runs"]] == ["completed", "completed"]
    ports = {call["command"][1] for call in calls}
    assert len(ports) == 2
    on_disk = json.loads(
        (tmp_path / "artifacts/results/sweep/checkpoint_sweep.json").read_text()
    )
    assert on_disk["manifest_paths"] == manifest["manifest_paths"]
    assert all(run["model_sha256"] for run in on_disk["runs"])


def test_sweep_reuses_only_a_strict_scenario_match(tmp_path: Path) -> None:
    request = _request(tmp_path)
    environments = {
        step: {
            "WEBOTS_MODEL_SEED": "11",
            "WEBOTS_MODEL_VARIANT": "r2a_targeted_v1",
            "WEBOTS_EVAL_MODEL": (
                "artifacts/models/webots/seed_11/r2a_targeted_v1/checkpoints/"
                f"ppo_webots_s11_r2a_targeted_v1_{step}_steps.zip"
            ),
            "WEBOTS_EVAL_EPISODES": "2",
            "WEBOTS_EVAL_SEED_START": "5000",
            "WEBOTS_EVAL_SPLIT": "validation",
            "WEBOTS_EVAL_CONDITION": "R2A",
            "WEBOTS_EVAL_PLANNER_LABEL": f"ppo_r2a_step{step}",
            "WEBOTS_EVAL_RUN_NAME": f"webots_validation_r2a_checkpoint{step}_seed11",
            "WEBOTS_OBSTACLE_COUNT": "1",
            "WEBOTS_OBSTACLE_SPEED_MIN": "0.1",
            "WEBOTS_OBSTACLE_SPEED_MAX": "0.2",
            "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": "1.1",
            "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": "1.1",
        }
        for step in request.steps
    }
    _write_summary(tmp_path, environments[10])
    _write_summary(tmp_path, environments[20], speed_range=(0.4, 0.5))
    calls: list[int] = []

    def runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        calls.append(int(environment["WEBOTS_EVAL_PLANNER_LABEL"].split("step")[-1]))
        _write_summary(tmp_path, environment)
        return subprocess.CompletedProcess(command, 0, "", "")

    manifest = run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/sweep",
        process_runner=runner,
        port_allocator=_sequential_port,
    )

    assert calls == [20]
    assert [run["status"] for run in manifest["runs"]] == ["reused", "completed"]
    assert "speed range" in manifest["runs"][1]["prior_summary_rejection"]


def test_sweep_records_process_failure_and_returns_nonzero(tmp_path: Path) -> None:
    request = _request(tmp_path, steps=(10,), workers=1)

    def runner(
        command: Sequence[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, "before failure", "failed")

    manifest = run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/sweep",
        process_runner=runner,
        port_allocator=_sequential_port,
    )

    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["exit_code"] == 9
    assert sweep_exit_code(manifest) == 2
    assert (tmp_path / "artifacts/results/sweep/checkpoint10.stderr.log").read_text() == (
        "failed"
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"condition": "R2/A"}, "condition"),
        ({"model_variant": "../escape"}, "model_variant"),
        ({"steps": (10, 10)}, "duplicates"),
        ({"obstacle_speed_min": 0.3, "obstacle_speed_max": 0.2}, "ordered"),
    ],
)
def test_sweep_request_rejects_ambiguous_identity(
    tmp_path: Path,
    changes: Mapping[str, Any],
    message: str,
) -> None:
    values: dict[str, Any] = {
        "repository_root": tmp_path,
        "webots_executable": tmp_path / "webots",
        "seed": 11,
        "condition": "R2A",
        "model_variant": "r2a_targeted_v1",
        "steps": (10,),
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        WebotsCheckpointSweepRequest(**values)
