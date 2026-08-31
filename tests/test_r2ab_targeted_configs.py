from __future__ import annotations

from pathlib import Path

from indoor_patrol_rl.training import load_webots_curriculum

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]


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
    assert stage.preferred_resume_model is None
    assert stage.gate.seed_start == 5000
    assert stage.gate.episodes == 100
    assert stage.controller_env["WEBOTS_OBSTACLE_SPEED_MIN"] == "0.18"
    assert stage.controller_env["WEBOTS_OBSTACLE_SPEED_MAX"] == "0.25"
    assert "WEBOTS_PPO_INITIAL_ACTION_STD" not in stage.controller_env
    assert stage.controller_env["WEBOTS_REWARD_PROXIMITY_SPEED_WEIGHT"] == "0.80"
    assert stage.controller_env["WEBOTS_SAFETY_SHIELD"] == "false"
