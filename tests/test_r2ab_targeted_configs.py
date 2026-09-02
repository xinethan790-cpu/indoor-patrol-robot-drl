from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from indoor_patrol_rl.evaluation.checkpoint_sweep import load_safety_shield_profile
from indoor_patrol_rl.training import (
    load_webots_curriculum,
    verify_warm_start_requirement,
)

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]


def test_r2a_rebaseline_binds_recoverable_source_and_fresh_validation_grid() -> None:
    config = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2a_rebaseline_v1.yaml"
    )

    assert config.model_variant == "r2a_rebaseline_v1"
    assert config.require_clean_git is True
    assert config.transfer_model == config.policy_transfer_model == Path(
        "artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip"
    )
    verification = verify_warm_start_requirement(
        repository_root=REPOSITORY_ROOT,
        config=config,
    )
    assert verification is not None
    assert verification.model_sha256 == (
        "e01bfcfa5dba22db4a8adf8b0ae8b325a6f7ef7eb50c2b1af1840e3d0c079c67"
    )
    assert verification.episode_count == 80
    assert verification.success_rate == 0.6875
    assert verification.collision_rate == 0.3125

    assert len(config.stages) == 1
    stage = config.stages[0]
    assert stage.stage_id == "RB2A"
    assert stage.training_steps == 40960
    assert stage.gate.seed_start == 6000
    assert stage.gate.episodes == 100
    assert stage.gate.min_success_rate == 0.75
    assert stage.gate.max_collision_rate == 0.25
    assert stage.controller_env["WEBOTS_OBSTACLE_SPEED_MIN"] == "0.10"
    assert stage.controller_env["WEBOTS_OBSTACLE_SPEED_MAX"] == "0.20"
    assert stage.controller_env["WEBOTS_SAFETY_SHIELD"] == "true"
    profile = load_safety_shield_profile(
        REPOSITORY_ROOT / "configs/eval/webots_r2a_front_brake_v1.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    for name, value in profile.controller_environment.items():
        assert stage.controller_env[name] == value


def test_r2b_rebaseline_resumes_exact_rb2a_endpoint_and_changes_only_speed() -> None:
    r2a = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2a_rebaseline_v1.yaml"
    )
    r2b = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2b_rebaseline_v1.yaml"
    )

    source = Path(
        "artifacts/models/webots/seed_11/r2b_rebaseline_v1/sources/"
        "ppo_webots_s11_r2b_rebaseline_v1_source_r2a_rebaseline_v1_40960_steps.zip"
    )
    source_path = REPOSITORY_ROOT / source
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    assert r2b.model_variant == "r2b_rebaseline_v1"
    assert r2b.require_clean_git is True
    assert r2b.transfer_model == source
    assert r2b.policy_transfer_model is None
    assert r2b.warm_start_requirement is None
    assert source_sha256 == (
        "f52a4691dbe7d787f6a9dfdb02383d206f597999e15f991063bc812a66706c2c"
    )

    assert len(r2a.stages) == len(r2b.stages) == 1
    r2a_stage = r2a.stages[0]
    r2b_stage = r2b.stages[0]
    assert r2b_stage.stage_id == "RB2B"
    assert r2b_stage.training_steps == 40960
    assert r2b_stage.preferred_resume_model == source
    assert r2b_stage.preferred_resume_sha256 == source_sha256
    assert r2b_stage.gate == r2a_stage.gate
    assert r2b_stage.gate.seed_start == 6000
    assert r2b_stage.gate.episodes == 100
    assert r2b_stage.gate.min_success_rate == 0.75
    assert r2b_stage.gate.max_collision_rate == 0.25

    r2a_environment = dict(r2a_stage.controller_env)
    r2b_environment = dict(r2b_stage.controller_env)
    assert r2a_environment.pop("WEBOTS_RUN_ID") == "r2a_rebaseline_v1"
    assert r2b_environment.pop("WEBOTS_RUN_ID") == "r2b_rebaseline_v1"
    assert r2a_environment.pop("WEBOTS_OBSTACLE_SPEED_MIN") == "0.10"
    assert r2a_environment.pop("WEBOTS_OBSTACLE_SPEED_MAX") == "0.20"
    assert r2b_environment.pop("WEBOTS_OBSTACLE_SPEED_MIN") == "0.18"
    assert r2b_environment.pop("WEBOTS_OBSTACLE_SPEED_MAX") == "0.25"
    assert r2a_environment.pop("WEBOTS_PPO_INITIAL_ACTION_STD") == "0.45"
    assert "WEBOTS_PPO_INITIAL_ACTION_STD" not in r2b_environment
    assert r2b_environment == r2a_environment

    with ZipFile(source_path) as archive:
        assert "policy.optimizer.pth" in archive.namelist()
        model_data = json.loads(archive.read("data"))
    assert model_data["num_timesteps"] == 40960
    assert model_data["n_steps"] == 2048
    assert model_data["batch_size"] == 128
    assert model_data["n_epochs"] == 5
    assert model_data["learning_rate"] == 3e-05
    assert model_data["ent_coef"] == 0.001
    assert model_data["gamma"] == 0.995
    assert model_data["target_kl"] == 0.02
    assert model_data["observation_space"]["_shape"] == [79]
    assert model_data["action_space"]["_shape"] == [2]


def test_r2c_density2_resumes_exact_rb2b_endpoint_and_changes_only_density() -> None:
    r2b = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2b_rebaseline_v1.yaml"
    )
    r2c = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2c_density2_v1.yaml"
    )

    source = Path(
        "artifacts/models/webots/seed_11/r2c_density2_v1/sources/"
        "ppo_webots_s11_r2c_density2_v1_source_r2b_rebaseline_v1_81920_steps.zip"
    )
    source_path = REPOSITORY_ROOT / source
    rb2b_endpoint = (
        REPOSITORY_ROOT
        / "artifacts/models/webots/seed_11/r2b_rebaseline_v1/"
        "ppo_webots_s11_step000081920.zip"
    )
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    assert r2c.model_variant == "r2c_density2_v1"
    assert r2c.require_clean_git is True
    assert r2c.transfer_model == source
    assert r2c.policy_transfer_model is None
    assert r2c.warm_start_requirement is None
    assert source_sha256 == hashlib.sha256(rb2b_endpoint.read_bytes()).hexdigest()
    assert source_sha256 == (
        "498d970cfcf539d55943ace88dcbaad83445713ad852291098b391fd38a86cfa"
    )

    assert len(r2c.stages) == 1
    rb2b_stage = r2b.stages[0]
    r2c_stage = r2c.stages[0]
    assert r2c_stage.stage_id == "R2C1"
    assert r2c_stage.obstacle_count == 2
    assert r2c_stage.training_steps == 40960
    assert r2c_stage.training_steps % 2048 == 0
    assert r2c_stage.training_steps % 10240 == 0
    assert 81920 + r2c_stage.training_steps == 122880
    assert r2c_stage.preferred_resume_model == source
    assert r2c_stage.preferred_resume_sha256 == source_sha256
    assert r2c_stage.gate.seed_start == 8200
    assert r2c_stage.gate.episodes == 100
    assert r2c_stage.gate.min_success_rate == 0.75
    assert r2c_stage.gate.max_collision_rate == 0.25
    validation_seeds = set(range(8200, 8300))
    assert validation_seeds.isdisjoint(range(6000, 6100))
    assert validation_seeds.isdisjoint(range(8000, 8020))
    assert validation_seeds.isdisjoint(range(10000, 10020))

    rb2b_environment = dict(rb2b_stage.controller_env)
    r2c_environment = dict(r2c_stage.controller_env)
    assert rb2b_environment.pop("WEBOTS_RUN_ID") == "r2b_rebaseline_v1"
    assert r2c_environment.pop("WEBOTS_RUN_ID") == "r2c_density2_v1"
    assert r2c_environment == rb2b_environment
    assert "WEBOTS_PPO_INITIAL_ACTION_STD" not in r2c_environment
    assert "WEBOTS_PPO_RESET_LINEAR_HEAD" not in r2c_environment

    profile = load_safety_shield_profile(
        REPOSITORY_ROOT / "configs/eval/webots_r2a_front_brake_v1.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    for name, value in profile.controller_environment.items():
        assert r2c_stage.controller_env[name] == value

    with ZipFile(source_path) as archive:
        assert "policy.optimizer.pth" in archive.namelist()
        model_data = json.loads(archive.read("data"))
    assert model_data["num_timesteps"] == 81920
    assert model_data["n_steps"] == 2048
    assert model_data["batch_size"] == 128
    assert model_data["n_epochs"] == 5
    assert model_data["learning_rate"] == 3e-05
    assert model_data["ent_coef"] == 0.001
    assert model_data["gamma"] == 0.995
    assert model_data["target_kl"] == 0.02
    assert model_data["observation_space"]["_shape"] == [79]
    assert model_data["action_space"]["_shape"] == [2]


def test_r2a_targeted_config_locks_policy_transfer_and_validation_gate() -> None:
    config = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2a_targeted_v1.yaml"
    )

    assert config.model_variant == "r2a_targeted_v1"
    assert config.policy_transfer_model == Path(
        "artifacts/models/webots/seed_11/curriculum_v2/checkpoints/"
        "ppo_webots_s11_curriculum_v2_40960_steps.zip"
    )
    assert len(config.stages) == 1
    stage = config.stages[0]
    assert stage.stage_id == "R2A"
    assert stage.training_steps == 40960
    assert stage.gate.seed_start == 5000
    assert stage.gate.episodes == 100
    assert stage.gate.min_success_rate == 0.75
    assert stage.gate.max_collision_rate == 0.25
    assert stage.controller_env["WEBOTS_OBSTACLE_SPEED_MIN"] == "0.10"
    assert stage.controller_env["WEBOTS_OBSTACLE_SPEED_MAX"] == "0.20"
    assert stage.controller_env["WEBOTS_REWARD_PROXIMITY_SPEED_WEIGHT"] == "0.80"
    assert stage.controller_env["WEBOTS_PPO_INITIAL_ACTION_STD"] == "0.45"
    assert stage.controller_env["WEBOTS_SAFETY_SHIELD"] == "false"


def test_r2b_targeted_config_requires_selected_resume_without_resetting_std() -> None:
    config = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2b_targeted_v1.yaml"
    )

    assert config.model_variant == "r2b_targeted_v1"
    assert config.policy_transfer_model is None
    assert len(config.stages) == 1
    stage = config.stages[0]
    assert stage.stage_id == "R2B"
    assert stage.training_steps == 40960
    expected_resume = Path(
        "artifacts/models/webots/seed_11/r2b_targeted_v1/sources/"
        "ppo_webots_s11_r2b_targeted_v1_source_r2a_v2_10240_steps.zip"
    )
    assert stage.preferred_resume_model == expected_resume
    assert stage.preferred_resume_sha256 == (
        "051d1e2f07bb147d247c91ded821e58af5b87987bedc2cee4e287fdbdfcdcbe0"
    )
    assert stage.gate.seed_start == 5000
    assert stage.gate.episodes == 100
    assert stage.controller_env["WEBOTS_OBSTACLE_SPEED_MIN"] == "0.18"
    assert stage.controller_env["WEBOTS_OBSTACLE_SPEED_MAX"] == "0.25"
    assert "WEBOTS_PPO_INITIAL_ACTION_STD" not in stage.controller_env
    assert stage.controller_env["WEBOTS_REWARD_PROXIMITY_SPEED_WEIGHT"] == "0.80"
    assert stage.controller_env["WEBOTS_REWARD_PROXIMITY_DISTANCE"] == "0.70"
    assert stage.controller_env["WEBOTS_SAFETY_SHIELD"] == "true"
    profile = load_safety_shield_profile(
        REPOSITORY_ROOT / "configs/eval/webots_r2a_front_brake_v1.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    for name, value in profile.controller_environment.items():
        assert stage.controller_env[name] == value


def test_r2a_v2_changes_only_the_early_warning_distance() -> None:
    v1 = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2a_targeted_v1.yaml"
    )
    v2 = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2a_targeted_v2.yaml"
    )

    assert v2.model_variant == "r2a_targeted_v2"
    assert v2.policy_transfer_model == v1.policy_transfer_model
    assert len(v2.stages) == 1
    v1_stage = v1.stages[0]
    v2_stage = v2.stages[0]
    assert v2_stage.stage_id == "R2A2"
    assert v2_stage.training_steps == v1_stage.training_steps == 40960
    assert v2_stage.gate == v1_stage.gate

    v1_environment = dict(v1_stage.controller_env)
    v2_environment = dict(v2_stage.controller_env)
    assert v1_environment.pop("WEBOTS_RUN_ID") == "r2a_targeted"
    assert v2_environment.pop("WEBOTS_RUN_ID") == "r2a_targeted_v2"
    assert v1_environment.pop("WEBOTS_REWARD_PROXIMITY_DISTANCE") == "0.60"
    assert v2_environment.pop("WEBOTS_REWARD_PROXIMITY_DISTANCE") == "0.70"
    assert v2_environment == v1_environment


def test_r2a_v3_changes_only_the_terminal_collision_penalty() -> None:
    v2 = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2a_targeted_v2.yaml"
    )
    v3 = load_webots_curriculum(
        REPOSITORY_ROOT / "configs/train/webots_r2a_targeted_v3.yaml"
    )

    assert v3.model_variant == "r2a_targeted_v3"
    assert v3.policy_transfer_model == v2.policy_transfer_model
    assert len(v3.stages) == 1
    v2_stage = v2.stages[0]
    v3_stage = v3.stages[0]
    assert v3_stage.stage_id == "R2A3"
    assert v3_stage.training_steps == v2_stage.training_steps == 40960
    assert v3_stage.gate == v2_stage.gate

    v2_environment = dict(v2_stage.controller_env)
    v3_environment = dict(v3_stage.controller_env)
    assert v2_environment.pop("WEBOTS_RUN_ID") == "r2a_targeted_v2"
    assert v3_environment.pop("WEBOTS_RUN_ID") == "r2a_targeted_v3"
    assert v2_environment.pop("WEBOTS_REWARD_COLLISION_PENALTY") == "-30.0"
    assert v3_environment.pop("WEBOTS_REWARD_COLLISION_PENALTY") == "-40.0"
    assert v3_environment == v2_environment
