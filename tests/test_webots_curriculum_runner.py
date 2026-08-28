from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from indoor_patrol_rl.training import (
    ValidationGate,
    WebotsCurriculumConfig,
    WebotsCurriculumRunner,
    WebotsStage,
    apply_preferred_resume_overrides,
    discover_model_snapshots,
    evaluation_controller_env,
    load_validation_metrics,
    load_webots_curriculum,
    resolve_stage_index,
    resolve_webots_executable,
    snapshot_matches_stage,
)


def make_stage(
    stage_id: str = "W0",
    *,
    obstacles: int = 0,
    steps: int = 10,
    success: float = 0.75,
    controller_env: dict[str, str] | None = None,
    preferred_resume_model: Path | None = None,
) -> WebotsStage:
    return WebotsStage(
        stage_id=stage_id,
        name=f"stage_{stage_id.lower()}",
        obstacle_count=obstacles,
        training_steps=steps,
        gate=ValidationGate(
            seed_start=5000,
            episodes=4,
            min_success_rate=success,
            max_collision_rate=0.25,
        ),
        controller_env=controller_env,
        preferred_resume_model=preferred_resume_model,
    )


def make_config(
    *stages: WebotsStage,
    variant: str | None = None,
    policy_transfer_model: Path | None = None,
) -> WebotsCurriculumConfig:
    return WebotsCurriculumConfig(
        transfer_model=Path("artifacts/models/transfer.zip"),
        port_base=12490,
        stages=stages,
        model_variant=variant,
        policy_transfer_model=policy_transfer_model,
    )


def save_training_result(
    repository_root: Path,
    *,
    seed: int,
    obstacles: int,
    steps: int,
    variant: str | None = None,
    starting_steps: int | None = None,
    additional_steps: int | None = None,
    speed_range: tuple[float, float] | None = None,
    route_half_lengths: tuple[float, float] | None = None,
    run_id: str | None = None,
) -> Path:
    model_directory = repository_root / "artifacts/models/webots" / f"seed_{seed}"
    if variant:
        model_directory /= variant
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / f"ppo_webots_s{seed}_step{steps:09d}.zip"
    model_path.write_bytes(b"model")
    metadata = {
        "obstacle_count": obstacles,
        "final_model_steps": steps,
        "final_model": str(model_path),
        "model_variant": variant,
    }
    if starting_steps is not None:
        metadata["starting_webots_steps"] = starting_steps
    if additional_steps is not None:
        metadata["additional_steps_requested"] = additional_steps
    if speed_range is not None:
        metadata["obstacle_speed_range"] = list(speed_range)
    if route_half_lengths is not None:
        metadata["route_half_lengths"] = {
            "horizontal": route_half_lengths[0],
            "vertical": route_half_lengths[1],
        }
    if run_id is not None:
        metadata["monitor_log"] = str(model_directory / f"training_{run_id}.monitor.csv")
    (model_directory / f"run_metadata_step{steps:09d}.yaml").write_text(
        yaml.safe_dump(metadata), encoding="utf-8"
    )
    (model_directory / "run_metadata.yaml").write_text(
        yaml.safe_dump(metadata), encoding="utf-8"
    )
    return model_path


def save_validation_result(
    repository_root: Path,
    *,
    seed: int,
    obstacles: int,
    episodes: int,
    success_rate: float,
    collision_rate: float,
    run_name: str | None = None,
) -> None:
    result_name = run_name or f"webots_validation_seed{seed}_obstacles{obstacles}"
    output = repository_root / "artifacts/results" / result_name
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": "validation",
        "by_planner": [
            {
                "planner": f"ppo_seed{seed}",
                "episodes": episodes,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
            }
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_repository_webots_curriculum_has_required_w0_w2_schedule() -> None:
    config = load_webots_curriculum(Path("configs/train/webots_curriculum.yaml"))

    assert [stage.stage_id for stage in config.stages] == ["W0", "W1", "W2"]
    assert [stage.obstacle_count for stage in config.stages] == [0, 1, 3]
    assert [stage.training_steps for stage in config.stages] == [50_000, 150_000, 300_000]
    assert resolve_stage_index(config.stages, "w1", 0) == 1
    assert resolve_stage_index(config.stages, "2", 0) == 2


def test_repository_v2_curriculum_records_variant_scenarios_and_gates() -> None:
    config = load_webots_curriculum(Path("configs/train/webots_curriculum_v2.yaml"))

    assert config.model_variant == "curriculum_v2"
    assert config.policy_transfer_model == Path(
        "artifacts/models/webots/seed_11/ppo_webots_s11_step000050176.zip"
    )
    assert [stage.stage_id for stage in config.stages] == ["R1", "R2", "R3"]
    assert [stage.training_steps for stage in config.stages] == [50_000, 75_000, 100_000]
    assert [stage.obstacle_count for stage in config.stages] == [1, 1, 1]
    assert [
        (
            stage.controller_env["WEBOTS_OBSTACLE_SPEED_MIN"],
            stage.controller_env["WEBOTS_OBSTACLE_SPEED_MAX"],
        )
        for stage in config.stages
    ] == [("0.05", "0.15"), ("0.10", "0.25"), ("0.18", "0.35")]
    assert config.stages[0].gate.min_success_rate == 0.75
    assert config.stages[1].gate.max_collision_rate == 0.25
    assert config.stages[2].gate.min_success_rate == 0.70
    assert config.stages[2].gate.max_collision_rate == 0.30
    assert config.stages[1].controller_env["WEBOTS_PPO_N_STEPS"] == "2048"
    assert config.stages[1].controller_env["WEBOTS_REWARD_SAFETY_WEIGHT"] == "-0.60"
    assert config.stages[1].preferred_resume_model == Path(
        "artifacts/models/webots/seed_11/curriculum_v2/checkpoints/"
        "ppo_webots_s11_curriculum_v2_40960_steps.zip"
    )
    assert config.stages[2].preferred_resume_model is None


@pytest.mark.parametrize(
    ("controller_env", "message"),
    [
        ({"PATH": "/tmp"}, "only uppercase WEBOTS"),
        ({"WEBOTS_CONFIRM_TEST": "1"}, "never set WEBOTS_CONFIRM_TEST"),
        ({"WEBOTS_EVAL_SPLIT": "test"}, "may only be validation"),
    ],
)
def test_controller_env_rejects_unsafe_or_non_webots_values(
    controller_env: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_stage(controller_env=controller_env)


def test_evaluation_receives_only_scenario_and_validation_controller_env() -> None:
    shared = evaluation_controller_env(
        {
            "WEBOTS_OBSTACLE_SPEED_MIN": "0.10",
            "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": "1.10",
            "WEBOTS_PPO_N_STEPS": "2048",
            "WEBOTS_REWARD_SAFETY_WEIGHT": "-0.60",
            "WEBOTS_EVAL_PLANNER_LABEL": "ppo_v2",
        }
    )

    assert shared == {
        "WEBOTS_OBSTACLE_SPEED_MIN": "0.10",
        "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": "1.10",
        "WEBOTS_EVAL_PLANNER_LABEL": "ppo_v2",
    }


def test_loader_refuses_to_automatically_run_test_split(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.yaml"
    path.write_text(
        """
schema_version: 1
transfer_model: model.zip
validation:
  split: test
  seed_start: 1
  episodes: 1
  min_success_rate: 0.5
  max_collision_rate: 0.5
stages: []
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation split"):
        load_webots_curriculum(path)


def test_resolve_webots_accepts_macos_app_bundle(tmp_path: Path) -> None:
    app = tmp_path / "Webots.app"
    executable = app / "Contents/MacOS/webots"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")

    assert resolve_webots_executable(app) == executable.absolute()


def test_validation_parser_rejects_non_validation_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "split": "test",
                "by_planner": [
                    {"episodes": 10, "success_rate": 1.0, "collision_rate": 0.0}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-validation"):
        load_validation_metrics(summary)


def test_runner_records_commands_results_and_both_manifest_formats(tmp_path: Path) -> None:
    stage = make_stage()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        environment = kwargs["env"]
        if command[-1].endswith("webots_ppo_train.wbt"):
            save_training_result(
                tmp_path,
                seed=11,
                obstacles=int(environment["WEBOTS_OBSTACLE_COUNT"]),
                steps=int(environment["WEBOTS_TRAIN_STEPS"]),
            )
        else:
            save_validation_result(
                tmp_path,
                seed=11,
                obstacles=int(environment["WEBOTS_OBSTACLE_COUNT"]),
                episodes=int(environment["WEBOTS_EVAL_EPISODES"]),
                success_rate=1.0,
                collision_rate=0.0,
            )
        return subprocess.CompletedProcess(command, 0)

    manifest = tmp_path / "run" / "manifest.yaml"
    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=make_config(stage),
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=manifest,
        process_runner=fake_run,
    )

    assert runner.run(start_stage=0, stop_after_stage=0) == 0
    assert len(calls) == 2
    assert all(isinstance(call[0], list) for call in calls)
    assert all("shell" not in call[1] for call in calls)
    assert calls[0][0][1] != calls[1][0][1]
    assert calls[1][1]["env"]["WEBOTS_EVAL_SPLIT"] == "validation"
    assert "WEBOTS_CONFIRM_TEST" not in calls[1][1]["env"]
    saved = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["test_split_was_run"] is False
    assert saved["stages"][0]["training_attempts"][0]["exit_code"] == 0
    assert saved["stages"][0]["validation"]["passed"] is True
    assert manifest.with_suffix(".json").is_file()


def test_gate_failure_stops_before_next_stage(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        environment = kwargs["env"]
        if command[-1].endswith("webots_ppo_train.wbt"):
            save_training_result(
                tmp_path,
                seed=11,
                obstacles=int(environment["WEBOTS_OBSTACLE_COUNT"]),
                steps=int(environment["WEBOTS_TRAIN_STEPS"]),
            )
        else:
            save_validation_result(
                tmp_path,
                seed=11,
                obstacles=int(environment["WEBOTS_OBSTACLE_COUNT"]),
                episodes=int(environment["WEBOTS_EVAL_EPISODES"]),
                success_rate=0.5,
                collision_rate=0.5,
            )
        return subprocess.CompletedProcess(command, 0)

    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=make_config(make_stage(success=0.8), make_stage("W1", obstacles=1)),
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=fake_run,
    )

    assert runner.run(start_stage=0, stop_after_stage=1) == 3
    assert len(calls) == 2
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["status"] == "gate_failed"
    assert [stage["id"] for stage in manifest["stages"]] == ["W0"]


def test_runner_resumes_partial_checkpoint_with_only_remaining_steps(tmp_path: Path) -> None:
    checkpoint = (
        tmp_path
        / "artifacts/models/webots/seed_11/checkpoints"
        / "ppo_webots_s11_40_steps.zip"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seen_environment: dict[str, str] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        if command[-1].endswith("webots_ppo_train.wbt"):
            seen_environment.update(environment)
            save_training_result(tmp_path, seed=11, obstacles=0, steps=100)
        else:
            save_validation_result(
                tmp_path,
                seed=11,
                obstacles=0,
                episodes=4,
                success_rate=1.0,
                collision_rate=0.0,
            )
        return subprocess.CompletedProcess(command, 0)

    snapshots = discover_model_snapshots(tmp_path, 11)
    assert snapshots[-1].path == checkpoint
    assert snapshots[-1].steps == 40
    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=make_config(make_stage(steps=100)),
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=fake_run,
    )

    assert runner.run(start_stage=0, stop_after_stage=0) == 0
    assert seen_environment["WEBOTS_TRAIN_STEPS"] == "60"
    assert seen_environment["WEBOTS_RESUME_MODEL"].endswith(
        "ppo_webots_s11_40_steps.zip"
    )
    assert "WEBOTS_TRANSFER_MODEL" not in seen_environment


def test_dry_run_plans_all_stages_without_starting_subprocesses(tmp_path: Path) -> None:
    def forbidden_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("dry-run must not invoke subprocess.run")

    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=make_config(make_stage(), make_stage("W1", obstacles=1)),
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest",
        process_runner=forbidden_run,
        dry_run=True,
    )

    assert runner.run(start_stage=0, stop_after_stage=1) == 0
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    assert [stage["status"] for stage in manifest["stages"]] == ["planned", "planned"]
    assert (
        "WEBOTS_RESUME_MODEL"
        in manifest["stages"][1]["training_attempts"][0]["environment"]
    )


def test_snapshot_discovery_is_recursive_but_isolated_by_variant(tmp_path: Path) -> None:
    selected_checkpoint = (
        tmp_path
        / "artifacts/models/webots/seed_11/selected/nested/checkpoints"
        / "ppo_webots_s11_selected_80_steps.zip"
    )
    selected_checkpoint.parent.mkdir(parents=True)
    selected_checkpoint.write_bytes(b"selected")
    other_checkpoint = (
        tmp_path
        / "artifacts/models/webots/seed_11/other/checkpoints"
        / "ppo_webots_s11_other_900_steps.zip"
    )
    other_checkpoint.parent.mkdir(parents=True)
    other_checkpoint.write_bytes(b"other")
    legacy_checkpoint = (
        tmp_path
        / "artifacts/models/webots/seed_11/checkpoints"
        / "ppo_webots_s11_700_steps.zip"
    )
    legacy_checkpoint.parent.mkdir(parents=True)
    legacy_checkpoint.write_bytes(b"legacy")

    selected = discover_model_snapshots(tmp_path, 11, "selected")
    legacy = discover_model_snapshots(tmp_path, 11)

    assert [snapshot.path for snapshot in selected] == [selected_checkpoint]
    assert all(snapshot.model_variant == "selected" for snapshot in selected)
    assert [snapshot.path for snapshot in legacy] == [legacy_checkpoint]
    assert other_checkpoint not in {snapshot.path for snapshot in selected}
    assert other_checkpoint not in {snapshot.path for snapshot in legacy}


def test_variant_runner_uses_policy_transfer_then_same_variant_resume(tmp_path: Path) -> None:
    common = {
        "WEBOTS_OBSTACLE_SPEED_MIN": "0.05",
        "WEBOTS_OBSTACLE_SPEED_MAX": "0.15",
        "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": "1.10",
        "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": "1.10",
        "WEBOTS_PPO_N_STEPS": "2048",
        "WEBOTS_REWARD_SAFETY_WEIGHT": "-0.60",
    }
    preferred_r2 = (
        tmp_path
        / "artifacts/models/webots/seed_11/curriculum_v2/checkpoints"
        / "ppo_webots_s11_curriculum_v2_40_steps.zip"
    )
    preferred_r2.parent.mkdir(parents=True)
    preferred_r2.write_bytes(b"selected")
    config = make_config(
        make_stage("R1", obstacles=1, steps=50, controller_env=common),
        make_stage(
            "R2",
            obstacles=1,
            steps=75,
            controller_env={
                **common,
                "WEBOTS_OBSTACLE_SPEED_MIN": "0.10",
                "WEBOTS_OBSTACLE_SPEED_MAX": "0.25",
            },
            preferred_resume_model=preferred_r2,
        ),
        variant="curriculum_v2",
        policy_transfer_model=Path("artifacts/models/webots/seed_11/base.zip"),
    )
    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        dry_run=True,
    )

    assert runner.run(start_stage=0, stop_after_stage=1) == 0
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    first_env = manifest["stages"][0]["training_attempts"][0]["environment"]
    second_env = manifest["stages"][1]["training_attempts"][0]["environment"]
    first_eval = manifest["stages"][0]["validation"]["environment"]

    assert first_env["WEBOTS_MODEL_VARIANT"] == "curriculum_v2"
    assert first_env["WEBOTS_POLICY_TRANSFER_MODEL"].endswith("seed_11/base.zip")
    assert "WEBOTS_RESUME_MODEL" not in first_env
    assert second_env["WEBOTS_RESUME_MODEL"].endswith(
        "seed_11/curriculum_v2/checkpoints/ppo_webots_s11_curriculum_v2_40_steps.zip"
    )
    assert "WEBOTS_POLICY_TRANSFER_MODEL" not in second_env
    assert second_env["WEBOTS_MODEL_VARIANT"] == "curriculum_v2"
    assert first_eval["WEBOTS_OBSTACLE_SPEED_MIN"] == "0.05"
    assert first_eval["WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH"] == "1.10"
    assert "WEBOTS_PPO_N_STEPS" not in first_eval
    assert "WEBOTS_REWARD_SAFETY_WEIGHT" not in first_eval
    assert first_eval["WEBOTS_EVAL_SPLIT"] == "validation"
    assert first_eval["WEBOTS_EVAL_RUN_NAME"] == (
        "webots_validation_seed11_curriculum_v2_r1"
    )


def test_preferred_resume_ignores_endpoint_then_uses_new_stage_checkpoint(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "artifacts/models/webots/seed_11/curriculum_v2"
    preferred = (
        model_directory
        / "checkpoints/ppo_webots_s11_curriculum_v2_40_steps.zip"
    )
    endpoint = model_directory / "ppo_webots_s11_step000000050.zip"
    preferred.parent.mkdir(parents=True)
    preferred.write_bytes(b"selected")
    endpoint.write_bytes(b"not selected")
    os.utime(endpoint, ns=(946684800_000_000_000, 946684800_000_000_000))
    stage = make_stage(
        "R2",
        obstacles=1,
        steps=75,
        preferred_resume_model=preferred,
    )
    config = make_config(stage, variant="curriculum_v2")
    manifest_path = tmp_path / "manifest.yaml"
    first_runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=manifest_path,
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        dry_run=True,
    )

    assert first_runner.run(start_stage=0, stop_after_stage=0) == 0
    first_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    first_attempt = first_manifest["stages"][0]["training_attempts"][0]
    assert first_attempt["environment"]["WEBOTS_RESUME_MODEL"].endswith(
        "curriculum_v2_40_steps.zip"
    )
    assert first_attempt["additional_steps_requested"] == 75

    first_attempt["status"] = "failed"
    first_attempt["started_at"] = "2001-01-01T00:00:00+00:00"
    manifest_path.write_text(yaml.safe_dump(first_manifest), encoding="utf-8")
    newer = (
        model_directory
        / "checkpoints/ppo_webots_s11_curriculum_v2_80_steps.zip"
    )
    newer.write_bytes(b"current stage")
    os.utime(newer, ns=(1009843200_000_000_000, 1009843200_000_000_000))
    second_runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=manifest_path,
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        dry_run=True,
    )

    assert second_runner.run(start_stage=0, stop_after_stage=0) == 0
    second_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    second_attempt = second_manifest["stages"][0]["training_attempts"][-1]
    assert second_attempt["environment"]["WEBOTS_RESUME_MODEL"].endswith(
        "curriculum_v2_80_steps.zip"
    )
    assert second_attempt["additional_steps_requested"] == 35


def test_variant_later_stage_requires_validation_selected_resume(tmp_path: Path) -> None:
    config = make_config(
        make_stage("R1", steps=10),
        make_stage("R2", steps=10),
        variant="curriculum_v2",
        policy_transfer_model=Path("base.zip"),
    )
    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        dry_run=True,
    )

    with pytest.raises(ValueError, match="variant endpoint is never chosen"):
        runner.run(start_stage=0, stop_after_stage=1)


def test_cli_preferred_resume_override_updates_only_selected_stage() -> None:
    config = make_config(make_stage("R1"), make_stage("R2"), variant="curriculum_v2")

    updated = apply_preferred_resume_overrides(config, ["r2=models/selected_80_steps.zip"])

    assert updated.stages[0].preferred_resume_model is None
    assert updated.stages[1].preferred_resume_model == Path(
        "models/selected_80_steps.zip"
    )
    assert config.stages[1].preferred_resume_model is None


def test_first_stage_may_supply_policy_transfer_in_controller_env(tmp_path: Path) -> None:
    stage = make_stage(
        "R1",
        controller_env={
            "WEBOTS_POLICY_TRANSFER_MODEL": "artifacts/models/selected_policy.zip"
        },
    )
    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=make_config(stage, variant="curriculum_v2"),
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        dry_run=True,
    )

    assert runner.run(start_stage=0, stop_after_stage=0) == 0
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    environment = manifest["stages"][0]["training_attempts"][0]["environment"]
    assert environment["WEBOTS_POLICY_TRANSFER_MODEL"] == (
        "artifacts/models/selected_policy.zip"
    )
    assert "WEBOTS_TRANSFER_MODEL" not in environment


def test_stage_completion_matches_scenario_not_just_obstacle_count(tmp_path: Path) -> None:
    r1_model = save_training_result(
        tmp_path,
        seed=11,
        obstacles=1,
        steps=51,
        variant="curriculum_v2",
        starting_steps=0,
        additional_steps=50,
        speed_range=(0.05, 0.15),
        route_half_lengths=(1.1, 1.1),
        run_id="r1_slow",
    )
    r2_model = save_training_result(
        tmp_path,
        seed=11,
        obstacles=1,
        steps=117,
        variant="curriculum_v2",
        starting_steps=41,
        additional_steps=75,
        speed_range=(0.10, 0.25),
        route_half_lengths=(1.1, 1.1),
        run_id="r2_medium",
    )
    snapshots = discover_model_snapshots(tmp_path, 11, "curriculum_v2")
    r1_stage = make_stage(
        "R1",
        obstacles=1,
        steps=50,
        controller_env={
            "WEBOTS_RUN_ID": "r1_slow",
            "WEBOTS_OBSTACLE_SPEED_MIN": "0.05",
            "WEBOTS_OBSTACLE_SPEED_MAX": "0.15",
            "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": "1.10",
            "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": "1.10",
        },
    )
    r2_stage = make_stage(
        "R2",
        obstacles=1,
        steps=75,
        controller_env={
            "WEBOTS_RUN_ID": "r2_medium",
            "WEBOTS_OBSTACLE_SPEED_MIN": "0.10",
            "WEBOTS_OBSTACLE_SPEED_MAX": "0.25",
            "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": "1.10",
            "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": "1.10",
        },
    )

    r1_matches = [
        snapshot.path
        for snapshot in snapshots
        if snapshot_matches_stage(snapshot, r1_stage)
    ]
    r2_matches = [
        snapshot.path
        for snapshot in snapshots
        if snapshot_matches_stage(snapshot, r2_stage)
    ]
    assert r1_matches == [r1_model]
    assert r2_matches == [r2_model]
