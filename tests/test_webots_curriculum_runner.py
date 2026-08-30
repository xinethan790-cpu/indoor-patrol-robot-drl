from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from indoor_patrol_rl.evaluation.checkpoint_selection import (
    CheckpointSelectionGate,
    make_candidate_record,
)
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
    validate_evaluation_run_name,
)
from tests.test_formal_protocol import _build_model_manifest


def make_stage(
    stage_id: str = "W0",
    *,
    obstacles: int = 0,
    steps: int = 10,
    success: float = 0.75,
    controller_env: dict[str, str] | None = None,
    preferred_resume_model: Path | None = None,
    world: str = "webots_ppo_train.wbt",
    evaluation_split: str = "validation",
    resume_from_previous_stage: bool = False,
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
        world=world,
        evaluation_split=evaluation_split,
        resume_from_previous_stage=resume_from_previous_stage,
    )


def make_config(
    *stages: WebotsStage,
    variant: str | None = None,
    policy_transfer_model: Path | None = None,
    policy_transfer_model_sha256: str | None = None,
) -> WebotsCurriculumConfig:
    return WebotsCurriculumConfig(
        transfer_model=Path("artifacts/models/transfer.zip"),
        port_base=12490,
        stages=stages,
        model_variant=variant,
        policy_transfer_model=policy_transfer_model,
        policy_transfer_model_sha256=policy_transfer_model_sha256,
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
    source_model: Path | None = None,
    source_model_sha256: str | None = None,
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
    if source_model is not None:
        metadata["source_model"] = str(source_model)
        metadata["source_model_sha256"] = source_model_sha256
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
    environment: dict[str, str],
    success_rate: float,
    collision_rate: float,
) -> None:
    seed = int(environment["WEBOTS_MODEL_SEED"])
    episodes = int(environment["WEBOTS_EVAL_EPISODES"])
    seed_start = int(environment["WEBOTS_EVAL_SEED_START"])
    result_name = environment["WEBOTS_EVAL_RUN_NAME"]
    planner = environment["WEBOTS_EVAL_PLANNER_LABEL"]
    condition = environment["WEBOTS_EVAL_CONDITION"]
    model_path = (repository_root / environment["WEBOTS_EVAL_MODEL"]).resolve()
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    output = repository_root / "artifacts/results" / result_name
    output.mkdir(parents=True, exist_ok=True)
    success_count = round(success_rate * episodes)
    collision_count = round(collision_rate * episodes)
    timeout_count = episodes - success_count - collision_count
    aggregate = {
        "planner": planner,
        "episodes": episodes,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "error_count": 0,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_count / episodes,
        "completion_time_s_mean": 1.0,
        "path_length_ratio_mean": 1.0,
    }
    split = environment["WEBOTS_EVAL_SPLIT"]
    summary = {
        "schema_version": "1.0",
        "split": split,
        "episode_count": episodes,
        "planners": [planner],
        "conditions": [condition],
        "by_planner": [aggregate],
        "by_planner_condition": [{**aggregate, "condition_id": condition}],
        "evaluation_provenance": {
            "schema_version": 1,
            "model_path": model_path.relative_to(repository_root).as_posix(),
            "model_sha256": model_sha256,
            "model_variant": environment.get("WEBOTS_MODEL_VARIANT"),
            "training_seed": seed,
            "split": split,
            "condition_id": condition,
            "scenario_seed_start": seed_start,
            "scenario_seed_count": episodes,
            "scenario_seeds": list(range(seed_start, seed_start + episodes)),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def prepare_recovery_resume_fixture(
    tmp_path: Path,
) -> tuple[
    WebotsCurriculumConfig,
    WebotsStage,
    WebotsCurriculumRunner,
    list[dict[str, str]],
    Any,
]:
    base_model = tmp_path / "artifacts/models/base.zip"
    base_model.parent.mkdir(parents=True)
    base_model.write_bytes(b"base")
    e1 = make_stage("E1", obstacles=0, steps=10)
    e2 = make_stage(
        "E2",
        obstacles=1,
        steps=10,
        resume_from_previous_stage=True,
    )
    config = make_config(
        e1,
        e2,
        variant="recovery",
        policy_transfer_model=Path("artifacts/models/base.zip"),
        policy_transfer_model_sha256=hashlib.sha256(b"base").hexdigest(),
    )
    training_environments: list[dict[str, str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        if command[-1].endswith("webots_ppo_train.wbt"):
            training_environments.append(environment)
            obstacles = int(environment["WEBOTS_OBSTACLE_COUNT"])
            if "WEBOTS_POLICY_TRANSFER_MODEL" in environment:
                source_model = (
                    tmp_path / environment["WEBOTS_POLICY_TRANSFER_MODEL"]
                ).resolve()
                starting_steps = 0
            else:
                source_model = (tmp_path / environment["WEBOTS_RESUME_MODEL"]).resolve()
                starting_steps = 15 if "_15_steps.zip" in source_model.name else 10
            save_training_result(
                tmp_path,
                seed=11,
                obstacles=obstacles,
                steps=10 if obstacles == 0 else 20,
                variant="recovery",
                starting_steps=starting_steps,
                additional_steps=10,
                source_model=source_model,
                source_model_sha256=hashlib.sha256(source_model.read_bytes()).hexdigest(),
            )
        else:
            save_validation_result(
                tmp_path,
                environment=environment,
                success_rate=1.0,
                collision_rate=0.0,
            )
        return subprocess.CompletedProcess(command, 0)

    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=fake_run,
    )
    assert runner.run(start_stage=0, stop_after_stage=0) == 0
    return config, e2, runner, training_environments, fake_run


def add_partial_recovery_checkpoint(
    tmp_path: Path,
    runner: WebotsCurriculumRunner,
    stage: WebotsStage,
) -> Path:
    previous = runner.manifest["stages"][0]
    previous_model = previous["training"]["model"]
    previous_steps = previous["training"]["final_model_steps"]
    previous_sha256 = previous["validation"]["model_sha256"]
    record = runner._stage_record(stage)
    record["stage_start_steps"] = 10
    record["status"] = "training_failed"
    record["training_attempts"].append(
        {
            "started_at": "2001-01-01T00:00:00+00:00",
            "status": "failed",
            "environment": {"WEBOTS_RESUME_MODEL": previous_model},
            "resume_source": {
                "kind": "previous_stage_endpoint",
                "model": previous_model,
                "steps": previous_steps,
                "sha256": previous_sha256,
            },
        }
    )
    runner._write_manifest()
    checkpoint = (
        tmp_path
        / "artifacts/models/webots/seed_11/recovery/checkpoints"
        / "ppo_webots_s11_recovery_15_steps.zip"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"partial")
    return checkpoint


def test_repository_webots_curriculum_has_required_w0_w2_schedule() -> None:
    config = load_webots_curriculum(Path("configs/train/webots_curriculum.yaml"))

    assert [stage.stage_id for stage in config.stages] == ["W0", "W1", "W2"]
    assert [stage.obstacle_count for stage in config.stages] == [0, 1, 3]
    assert [stage.training_steps for stage in config.stages] == [50_000, 150_000, 300_000]
    assert {stage.world for stage in config.stages} == {"webots_ppo_train.wbt"}
    assert resolve_stage_index(config.stages, "w1", 0) == 1
    assert resolve_stage_index(config.stages, "2", 0) == 2


def test_repository_v2_curriculum_records_variant_scenarios_and_gates() -> None:
    config = load_webots_curriculum(Path("configs/train/webots_curriculum_v2.yaml"))

    assert config.model_variant == "curriculum_v2"
    assert config.policy_transfer_model == Path(
        "artifacts/models/webots/seed_11/ppo_webots_s11_step000050176.zip"
    )
    assert config.policy_transfer_model_sha256 == (
        "91bc5dffe05566cbfcbf1fb2cab4588cdc54115c779f377e7aeea8428c4c2441"
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


def test_repository_w1_recovery_config_is_predeclared_and_split_safe() -> None:
    config = load_webots_curriculum(
        Path("configs/train/webots_w1_recovery_v1.yaml")
    )

    assert config.model_variant == "w1_recovery_v1"
    assert config.policy_transfer_model == Path(
        "artifacts/models/webots/seed_11/ppo_webots_s11_step000200192.zip"
    )
    assert config.policy_transfer_model_sha256 == (
        "84e0ebb459e0ef800b1e891f5d549237e246c04be33a3bed24f4811b18993cc8"
    )
    assert [stage.stage_id for stage in config.stages] == ["E1", "E2", "F1"]
    assert [stage.training_steps for stage in config.stages] == [40_960, 61_440, 102_400]
    assert [stage.evaluation_split for stage in config.stages] == [
        "diagnostic",
        "diagnostic",
        "validation",
    ]
    assert [stage.resume_from_previous_stage for stage in config.stages] == [
        False,
        True,
        True,
    ]
    assert [(stage.gate.seed_start, stage.gate.episodes) for stage in config.stages] == [
        (20_000, 20),
        (20_020, 20),
        (5_000, 20),
    ]
    assert all(
        "WEBOTS_PPO_RESET_LINEAR_HEAD" not in stage.controller_env
        and "WEBOTS_PPO_INITIAL_ACTION_STD" not in stage.controller_env
        for stage in config.stages
    )


def test_recovery_stage_constraints_reject_unsafe_resume_and_seed_windows() -> None:
    with pytest.raises(ValueError, match="first stage"):
        make_config(make_stage(resume_from_previous_stage=True))
    with pytest.raises(ValueError, match="cannot be combined"):
        make_stage(
            "E2",
            resume_from_previous_stage=True,
            preferred_resume_model=Path("selected_10_steps.zip"),
        )
    with pytest.raises(ValueError, match="must not overlap"):
        make_stage(
            "E1",
            evaluation_split="diagnostic",
        )


@pytest.mark.parametrize(
    ("evaluation_split", "seed_start", "episodes", "should_fail"),
    [
        ("validation", 9_980, 20, False),
        ("validation", 10_000, 1, True),
        ("validation", 10_019, 1, True),
        ("validation", 10_020, 1, False),
        ("diagnostic", 4_980, 20, False),
        ("diagnostic", 4_999, 2, True),
        ("diagnostic", 5_020, 1, False),
        ("diagnostic", 10_000, 1, True),
    ],
)
def test_non_test_gate_seed_window_boundaries(
    evaluation_split: str,
    seed_start: int,
    episodes: int,
    should_fail: bool,
) -> None:
    gate = ValidationGate(seed_start, episodes, 0.5, 0.5)
    if should_fail:
        with pytest.raises(ValueError, match="must not overlap"):
            replace(
                make_stage(),
                gate=gate,
                evaluation_split=evaluation_split,
            )
    else:
        stage = replace(
            make_stage(),
            gate=gate,
            evaluation_split=evaluation_split,
        )
        assert stage.gate == gate


@pytest.mark.parametrize(
    ("evaluation_split", "seed_start_override"),
    [("validation", 10_000), ("diagnostic", 5_000), ("diagnostic", 10_000)],
)
def test_gate_seed_override_cannot_enter_reserved_windows(
    tmp_path: Path,
    evaluation_split: str,
    seed_start_override: int,
) -> None:
    stage = make_stage()
    if evaluation_split == "diagnostic":
        stage = replace(
            stage,
            evaluation_split="diagnostic",
            gate=replace(stage.gate, seed_start=20_000),
        )
    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=make_config(stage),
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=lambda *_args, **_kwargs: pytest.fail("started Webots"),
        dry_run=True,
        seed_start_override=seed_start_override,
    )

    with pytest.raises(ValueError, match="must not overlap"):
        runner.run(start_stage=0, stop_after_stage=0)


def test_policy_transfer_model_and_sha256_are_strictly_paired() -> None:
    with pytest.raises(ValueError, match="configured together"):
        make_config(make_stage(), policy_transfer_model=Path("base.zip"))
    with pytest.raises(ValueError, match="configured together"):
        make_config(make_stage(), policy_transfer_model_sha256="0" * 64)
    with pytest.raises(ValueError, match="64 hexadecimal"):
        make_config(
            make_stage(),
            policy_transfer_model=Path("base.zip"),
            policy_transfer_model_sha256="not-a-sha",
        )


@pytest.mark.parametrize(
    ("controller_env", "message"),
    [
        ({"PATH": "/tmp"}, "only uppercase WEBOTS"),
        ({"WEBOTS_CONFIRM_TEST": "1"}, "never set WEBOTS_CONFIRM_TEST"),
        ({"WEBOTS_EVAL_SPLIT": "validation"}, "evaluation_split field"),
        ({"WEBOTS_EVAL_SPLIT": "diagnostic"}, "evaluation_split field"),
        ({"WEBOTS_EVAL_SPLIT": "test"}, "evaluation_split field"),
        ({"WEBOTS_POLICY_TRANSFER_MODEL": "base.zip"}, "structured curriculum"),
        ({"WEBOTS_RESUME_MODEL": "resume.zip"}, "structured curriculum"),
        ({"WEBOTS_TRANSFER_MODEL": "base.zip"}, "structured curriculum"),
    ],
)
def test_controller_env_rejects_unsafe_or_non_webots_values(
    controller_env: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_stage(controller_env=controller_env)


@pytest.mark.parametrize(
    "variant",
    ["", ".", "..", "nested/variant", "nested\\variant"],
)
def test_curriculum_rejects_unsafe_model_variants(variant: str) -> None:
    with pytest.raises(ValueError, match="model_variant"):
        make_config(make_stage(), variant=variant)


def test_loader_does_not_collapse_empty_model_variant_to_legacy(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.yaml"
    path.write_text(
        """
schema_version: 1
transfer_model: model.zip
model_variant: ""
validation:
  split: validation
  seed_start: 1
  episodes: 1
  min_success_rate: 0.5
  max_collision_rate: 0.5
stages:
  - id: W0
    name: empty
    obstacle_count: 0
    training_steps: 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty path-safe"):
        load_webots_curriculum(path)


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


@pytest.mark.parametrize("run_name", ["", ".", "..", "nested/name", "nested\\name"])
def test_evaluation_run_name_rejects_empty_dot_and_path_escapes(run_name: str) -> None:
    with pytest.raises(ValueError, match="path-safe"):
        validate_evaluation_run_name(run_name)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("evaluation_provenance", "model_path"), "artifacts/models/other.zip", "model path"),
        (("evaluation_provenance", "model_sha256"), "0" * 64, "SHA-256"),
        (("evaluation_provenance", "training_seed"), 22, "training_seed"),
        (("evaluation_provenance", "model_variant"), "other", "model_variant"),
        (("evaluation_provenance", "condition_id"), "W1", "condition"),
        (("evaluation_provenance", "scenario_seeds"), [5000], "scenario_seeds"),
        (("root", "episode_count"), 3, "episode count"),
    ],
)
def test_strict_stage_gate_rejects_wrong_evaluation_provenance(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    model = save_training_result(tmp_path, seed=11, obstacles=0, steps=10)
    environment = {
        "WEBOTS_MODEL_SEED": "11",
        "WEBOTS_EVAL_MODEL": model.relative_to(tmp_path).as_posix(),
        "WEBOTS_EVAL_EPISODES": "4",
        "WEBOTS_EVAL_SEED_START": "5000",
        "WEBOTS_EVAL_SPLIT": "validation",
        "WEBOTS_EVAL_CONDITION": "W0",
        "WEBOTS_EVAL_PLANNER_LABEL": "ppo_w0_step10",
        "WEBOTS_EVAL_RUN_NAME": "webots_validation_w0_final_step10_seed11",
        "WEBOTS_OBSTACLE_COUNT": "0",
    }
    save_validation_result(
        tmp_path,
        environment=environment,
        success_rate=1.0,
        collision_rate=0.0,
    )
    summary_path = (
        tmp_path
        / "artifacts/results/webots_validation_w0_final_step10_seed11/summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if path[0] == "root":
        payload[path[1]] = value
    else:
        payload[path[0]][path[1]] = value
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_validation_metrics(
            summary_path,
            repository_root=tmp_path,
            expected_model_path=model,
            expected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            expected_training_seed=11,
            expected_model_variant=None,
            expected_condition="W0",
            expected_planner="ppo_w0_step10",
            expected_seed_start=5000,
            expected_episodes=4,
        )


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
                environment=environment,
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
    assert calls[1][1]["env"]["WEBOTS_EVAL_RUN_NAME"] == (
        "webots_validation_w0_final_step10_seed11"
    )
    assert calls[1][1]["env"]["WEBOTS_EVAL_PLANNER_LABEL"] == "ppo_w0_step10"
    saved = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert len(saved["curriculum_config_sha256"]) == 64
    int(saved["curriculum_config_sha256"], 16)
    assert saved["curriculum_config"]["transfer_model"] == (
        "artifacts/models/transfer.zip"
    )
    assert saved["curriculum_config"]["stages"][0]["world"] == (
        "webots_ppo_train.wbt"
    )
    assert saved["status"] == "completed"
    assert saved["test_split_was_run"] is False
    assert saved["stages"][0]["training_attempts"][0]["exit_code"] == 0
    assert saved["stages"][0]["validation"]["passed"] is True
    assert len(saved["stages"][0]["validation"]["model_sha256"]) == 64
    assert manifest.with_suffix(".json").is_file()

    summary_path = (
        tmp_path
        / "artifacts/results/webots_validation_w0_final_step10_seed11/summary.json"
    )
    candidate = make_candidate_record(
        summary_path,
        repository_root=tmp_path,
        gate=CheckpointSelectionGate(
            seed=11,
            condition="W0",
            episodes=4,
            min_success_rate=0.75,
            max_collision_rate=0.25,
        ),
    )
    assert candidate["gate_passed"] is True


def test_identical_semantic_config_can_resume_with_normalized_paths_and_json(
    tmp_path: Path,
) -> None:
    stage = make_stage(
        controller_env={
            "WEBOTS_PPO_N_STEPS": "2048",
            "WEBOTS_REWARD_SAFETY_WEIGHT": "-0.60",
        }
    )
    base_model = tmp_path / "artifacts/models/base.zip"
    base_model.parent.mkdir(parents=True)
    base_model.write_bytes(b"base")
    base_sha256 = hashlib.sha256(base_model.read_bytes()).hexdigest()
    config = WebotsCurriculumConfig(
        transfer_model=Path("artifacts/models/transfer.zip"),
        policy_transfer_model=Path("artifacts/models/base.zip"),
        policy_transfer_model_sha256=base_sha256,
        model_variant="curriculum_v2",
        port_base=12490,
        stages=(stage,),
    )
    manifest_path = tmp_path / "manifest.yaml"
    first = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=manifest_path,
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        dry_run=True,
        episodes_override=3,
    )
    assert first.run(start_stage=0, stop_after_stage=0) == 0

    equivalent_stage = replace(
        stage,
        controller_env={
            "WEBOTS_REWARD_SAFETY_WEIGHT": "-0.60",
            "WEBOTS_PPO_N_STEPS": "2048",
        },
    )
    equivalent_config = replace(
        config,
        transfer_model=tmp_path / "artifacts/models/transfer.zip",
        policy_transfer_model=tmp_path / "artifacts/models/base.zip",
        stages=(equivalent_stage,),
    )
    resumed = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=equivalent_config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=manifest_path,
        process_runner=lambda *_args, **_kwargs: pytest.fail("resume started Webots"),
        episodes_override=3,
    )

    assert resumed.curriculum_config_sha256 == first.curriculum_config_sha256
    assert resumed.manifest["curriculum_config_sha256"] == (
        first.curriculum_config_sha256
    )
    assert resumed.manifest["curriculum_config"]["transfer_model"] == (
        "artifacts/models/transfer.zip"
    )


@pytest.mark.parametrize(
    "changed_field",
    [
        "transfer_model",
        "policy_transfer_model",
        "model_variant",
        "port_base",
        "stage_id",
        "stage_name",
        "stage_world",
        "obstacle_count",
        "training_steps",
        "gate",
        "controller_env",
        "preferred_resume_model",
    ],
)
def test_resume_rejects_changed_semantic_config(
    tmp_path: Path, changed_field: str
) -> None:
    stage = make_stage()
    config = make_config(stage)
    manifest_path = tmp_path / "manifest.yaml"
    first = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=manifest_path,
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        dry_run=True,
    )
    assert first.run(start_stage=0, stop_after_stage=0) == 0

    changed_stage = stage
    changed_config = config
    if changed_field == "transfer_model":
        changed_config = replace(config, transfer_model=Path("models/other.zip"))
    elif changed_field == "policy_transfer_model":
        changed_config = replace(
            config,
            policy_transfer_model=Path("models/base.zip"),
            policy_transfer_model_sha256="0" * 64,
        )
    elif changed_field == "model_variant":
        changed_config = replace(config, model_variant="curriculum_v2")
    elif changed_field == "port_base":
        changed_config = replace(config, port_base=config.port_base + 1)
    elif changed_field == "stage_id":
        changed_stage = replace(stage, stage_id="W1")
    elif changed_field == "stage_name":
        changed_stage = replace(stage, name="renamed")
    elif changed_field == "stage_world":
        changed_stage = replace(stage, world="webots_custom_train.wbt")
    elif changed_field == "obstacle_count":
        changed_stage = replace(stage, obstacle_count=1)
    elif changed_field == "training_steps":
        changed_stage = replace(stage, training_steps=stage.training_steps + 1)
    elif changed_field == "gate":
        changed_stage = replace(stage, gate=replace(stage.gate, episodes=5))
    elif changed_field == "controller_env":
        changed_stage = replace(stage, controller_env={"WEBOTS_PPO_N_STEPS": "2048"})
    elif changed_field == "preferred_resume_model":
        changed_stage = replace(
            stage,
            preferred_resume_model=Path(
                "artifacts/models/webots/seed_11/checkpoints/"
                "ppo_webots_s11_5_steps.zip"
            ),
        )
    if changed_stage is not stage:
        changed_config = replace(config, stages=(changed_stage,))

    with pytest.raises(ValueError, match="configuration fingerprint"):
        WebotsCurriculumRunner(
            repository_root=tmp_path,
            config=changed_config,
            webots_executable=tmp_path / "webots",
            seed=11,
            manifest_path=manifest_path,
            process_runner=lambda *_args, **_kwargs: pytest.fail(
                "mismatched resume started Webots"
            ),
            dry_run=True,
        )


def test_resume_rejects_changed_gate_override(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    config = make_config(make_stage())
    first = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=manifest_path,
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        dry_run=True,
    )
    assert first.run(start_stage=0, stop_after_stage=0) == 0

    with pytest.raises(ValueError, match="configuration fingerprint"):
        WebotsCurriculumRunner(
            repository_root=tmp_path,
            config=config,
            webots_executable=tmp_path / "webots",
            seed=11,
            manifest_path=manifest_path,
            process_runner=lambda *_args, **_kwargs: pytest.fail(
                "mismatched resume started Webots"
            ),
            dry_run=True,
            success_rate_override=0.5,
        )


def test_legacy_manifest_without_fingerprint_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "seed": 11,
                "model_variant": None,
                "stages": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="schema_version 2 semantic configuration fingerprint"
    ):
        WebotsCurriculumRunner(
            repository_root=tmp_path,
            config=make_config(make_stage()),
            webots_executable=tmp_path / "webots",
            seed=11,
            manifest_path=manifest_path,
            dry_run=True,
        )


def test_schema_v2_manifest_without_fingerprint_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "seed": 11,
                "model_variant": None,
                "stages": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing a valid curriculum_config_sha256"):
        WebotsCurriculumRunner(
            repository_root=tmp_path,
            config=make_config(make_stage()),
            webots_executable=tmp_path / "webots",
            seed=11,
            manifest_path=manifest_path,
            dry_run=True,
        )


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
                environment=environment,
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
                environment=environment,
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


def test_dry_run_records_diagnostic_split_and_exact_previous_stage_resume(
    tmp_path: Path,
) -> None:
    base_model = tmp_path / "artifacts/models/base.zip"
    base_model.parent.mkdir(parents=True)
    base_model.write_bytes(b"base")
    base_sha256 = hashlib.sha256(base_model.read_bytes()).hexdigest()
    e1 = replace(
        make_stage("E1", obstacles=1),
        evaluation_split="diagnostic",
        gate=ValidationGate(20_000, 4, 0.70, 0.30),
    )
    e2 = make_stage(
        "E2",
        obstacles=1,
        resume_from_previous_stage=True,
    )
    runner = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=make_config(
            e1,
            e2,
            variant="recovery",
            policy_transfer_model=Path("artifacts/models/base.zip"),
            policy_transfer_model_sha256=base_sha256,
        ),
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        dry_run=True,
    )

    assert runner.run(start_stage=0, stop_after_stage=1) == 0
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    e1_record, e2_record = manifest["stages"]
    assert manifest["automatic_split"] == "diagnostic_and_validation"
    assert e1_record["validation"]["environment"]["WEBOTS_EVAL_SPLIT"] == (
        "diagnostic"
    )
    assert e1_record["validation"]["environment"]["WEBOTS_EVAL_RUN_NAME"].startswith(
        "webots_diagnostic_e1_"
    )
    assert e2_record["training_attempts"][0]["resume_source"]["kind"] == (
        "previous_stage_endpoint"
    )
    assert e2_record["resume_from_previous_stage"] is True
    assert manifest["curriculum_config"]["stages"][0]["evaluation_split"] == (
        "diagnostic"
    )
    assert manifest["curriculum_config"]["stages"][1][
        "resume_from_previous_stage"
    ] is True


def test_passed_previous_gate_allows_current_stage_partial_checkpoint_priority(
    tmp_path: Path,
) -> None:
    config, e2, first_runner, environments, fake_run = (
        prepare_recovery_resume_fixture(tmp_path)
    )
    partial = add_partial_recovery_checkpoint(tmp_path, first_runner, e2)
    environments.clear()
    resumed = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=fake_run,
    )

    assert resumed.run(start_stage=1, stop_after_stage=1) == 0
    assert Path(environments[0]["WEBOTS_RESUME_MODEL"]).as_posix() == (
        partial.relative_to(tmp_path).as_posix()
    )
    assert environments[0]["WEBOTS_TRAIN_STEPS"] == "5"
    manifest = yaml.safe_load(
        (tmp_path / "manifest.yaml").read_text(encoding="utf-8")
    )
    attempt = manifest["stages"][1]["training_attempts"][-1]
    assert attempt["resume_source"]["kind"] == "current_stage_checkpoint"
    assert attempt["resume_source"]["steps"] == 15

    environments.clear()
    verified = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=fake_run,
    )
    assert verified.run(start_stage=1, stop_after_stage=1) == 0
    assert environments == []


def test_new_stage_manifest_does_not_attach_residual_partial_or_completed(
    tmp_path: Path,
) -> None:
    config, e2, first_runner, environments, fake_run = (
        prepare_recovery_resume_fixture(tmp_path)
    )
    previous = first_runner.manifest["stages"][0]
    previous_path = (tmp_path / previous["training"]["model"]).resolve()
    checkpoint = (
        tmp_path
        / "artifacts/models/webots/seed_11/recovery/checkpoints"
        / "ppo_webots_s11_recovery_15_steps.zip"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"residual-partial")
    save_training_result(
        tmp_path,
        seed=11,
        obstacles=1,
        steps=20,
        variant="recovery",
        starting_steps=10,
        additional_steps=10,
        source_model=previous_path,
        source_model_sha256=previous["validation"]["model_sha256"],
    )
    environments.clear()
    resumed = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=fake_run,
    )

    assert resumed.run(start_stage=1, stop_after_stage=1) == 0
    assert environments[0]["WEBOTS_RESUME_MODEL"] == previous["training"]["model"]
    assert environments[0]["WEBOTS_TRAIN_STEPS"] == "10"


@pytest.mark.parametrize("lineage_error", ["starting_steps", "source_path", "source_sha"])
def test_completed_recovery_artifact_requires_exact_previous_lineage(
    tmp_path: Path,
    lineage_error: str,
) -> None:
    config, e2, first_runner, environments, fake_run = (
        prepare_recovery_resume_fixture(tmp_path)
    )
    add_partial_recovery_checkpoint(tmp_path, first_runner, e2)
    previous = first_runner.manifest["stages"][0]
    source_path = (tmp_path / previous["training"]["model"]).resolve()
    source_sha256 = previous["validation"]["model_sha256"]
    starting_steps = 10
    if lineage_error == "starting_steps":
        starting_steps = 9
    elif lineage_error == "source_path":
        source_path = tmp_path / "wrong-source.zip"
        source_path.write_bytes(b"wrong")
        source_sha256 = hashlib.sha256(b"wrong").hexdigest()
    else:
        source_sha256 = "0" * 64
    save_training_result(
        tmp_path,
        seed=11,
        obstacles=1,
        steps=20,
        variant="recovery",
        starting_steps=starting_steps,
        additional_steps=10,
        source_model=source_path,
        source_model_sha256=source_sha256,
    )
    environments.clear()
    resumed = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=fake_run,
    )

    with pytest.raises(ValueError, match="completed artifact"):
        resumed.run(start_stage=1, stop_after_stage=1)
    assert environments == []


@pytest.mark.parametrize("gate_state", ["missing", "failed"])
def test_partial_checkpoint_cannot_bypass_previous_stage_gate(
    tmp_path: Path,
    gate_state: str,
) -> None:
    config, e2, first_runner, environments, fake_run = (
        prepare_recovery_resume_fixture(tmp_path)
    )
    add_partial_recovery_checkpoint(tmp_path, first_runner, e2)
    previous_record = first_runner.manifest["stages"][0]
    if gate_state == "missing":
        previous_record.pop("validation")
    else:
        previous_record["status"] = "gate_failed"
        previous_record["validation"]["status"] = "gate_failed"
        previous_record["validation"]["passed"] = False
    first_runner._write_manifest()
    environments.clear()
    resumed = WebotsCurriculumRunner(
        repository_root=tmp_path,
        config=config,
        webots_executable=tmp_path / "webots",
        seed=11,
        manifest_path=tmp_path / "manifest.yaml",
        process_runner=fake_run,
    )

    with pytest.raises(ValueError, match="validation must be a mapping|has not completed"):
        resumed.run(start_stage=1, stop_after_stage=1)
    assert environments == []


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


def test_snapshot_discovery_parses_source_model_lineage(tmp_path: Path) -> None:
    source = tmp_path / "artifacts/models/source.zip"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    model = save_training_result(
        tmp_path,
        seed=11,
        obstacles=1,
        steps=20,
        variant="recovery",
        starting_steps=10,
        additional_steps=10,
        source_model=source,
        source_model_sha256=source_sha256.upper(),
    )

    snapshot = next(
        item for item in discover_model_snapshots(tmp_path, 11, "recovery")
        if item.path == model
    )
    assert snapshot.source_model == source
    assert snapshot.source_model_sha256 == source_sha256


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
    base_model = tmp_path / "artifacts/models/webots/seed_11/base.zip"
    base_model.write_bytes(b"base")
    base_sha256 = hashlib.sha256(base_model.read_bytes()).hexdigest()
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
        policy_transfer_model_sha256=base_sha256,
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
    assert first_eval["WEBOTS_MODEL_VARIANT"] == "curriculum_v2"
    assert Path(first_env["WEBOTS_POLICY_TRANSFER_MODEL"]).as_posix().endswith(
        "seed_11/base.zip"
    )
    assert "WEBOTS_RESUME_MODEL" not in first_env
    assert Path(second_env["WEBOTS_RESUME_MODEL"]).as_posix().endswith(
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
        "webots_validation_r1_final_step50_seed11"
    )
    assert first_eval["WEBOTS_EVAL_PLANNER_LABEL"] == "ppo_r1_step50"


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
    base_model = tmp_path / "base.zip"
    base_model.write_bytes(b"base")
    config = make_config(
        make_stage("R1", steps=10),
        make_stage("R2", steps=10),
        variant="curriculum_v2",
        policy_transfer_model=Path("base.zip"),
        policy_transfer_model_sha256=hashlib.sha256(b"base").hexdigest(),
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


def test_completed_formal_curriculum_records_training_time_checkpoint_sidecars(
    tmp_path: Path,
) -> None:
    selected = _build_model_manifest(tmp_path, 11, checkpoint=True)
    selected_sidecar = tmp_path / selected.model_provenance_path
    selected_sidecar.unlink()
    curriculum_path = tmp_path / selected.curriculum_manifest_path
    manifest = yaml.safe_load(curriculum_path.read_text(encoding="utf-8"))
    w2 = manifest["stages"][2]
    w2.pop("artifact_provenance")
    w2["stage_start_steps"] = 200_000
    w2["training"] = {
        "metadata": selected.training_metadata_path,
        "final_model_steps": 500_000,
    }
    curriculum_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    runner = object.__new__(WebotsCurriculumRunner)
    runner.repository_root = tmp_path.resolve()
    runner.config = make_config(
        make_stage("W0"),
        make_stage("W1", obstacles=1),
        make_stage("W2", obstacles=3),
    )
    runner.dry_run = False
    runner.seed = 11
    runner.manifest_path = curriculum_path
    runner.manifest = manifest

    runner._record_formal_w2_artifact_provenance()

    records = runner.manifest["stages"][2]["artifact_provenance"]
    assert len(records) == 2
    assert {record["model_path"] for record in records} == {
        selected.model_path,
        "artifacts/models/webots/seed_11/ppo_webots_s11_step000500000.zip",
    }
    assert all((tmp_path / record["provenance_path"]).is_file() for record in records)
