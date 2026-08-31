"""Formal Webots PPO transfer/resume training with checkpoints and metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import platform
import re
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import yaml
from controller import Supervisor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from indoor_patrol_rl.core import (
    LidarSafetyShield,
    RewardConfig,
    safety_shield_config_from_env,
)
from indoor_patrol_rl.envs import WebotsPatrolEnv

ALL_OBSTACLE_DEFS = ("TRAIN_OBSTACLE_0", "TRAIN_OBSTACLE_1", "TRAIN_OBSTACLE_2")


def environment_flag(name: str, *, default: bool = False) -> bool:
    """Parse an explicit environment boolean and reject ambiguous values."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution: str) -> str:
    """Return a dependency version without making metadata collection fatal."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def git_metadata(repository_root: Path) -> dict[str, object]:
    """Collect the exact source revision used by a training segment."""

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    revision = run("rev-parse", "HEAD") or "uncommitted"
    return {
        "revision": revision,
        "dirty": bool(run("status", "--porcelain")),
    }


def main() -> None:
    repository_root = Path(__file__).absolute().parents[2]
    supervisor = Supervisor()
    exit_code = 1
    env = None
    raw_env = None
    model = None
    seed: int | None = None
    model_variant = ""
    segment_id = time.strftime("failed_%Y%m%dT%H%M%S")
    training_steps: int | None = None
    obstacle_count: int | None = None
    source_path: Path | None = None
    monitor_path: Path | None = None
    try:
        seed = int(os.environ.get("WEBOTS_TRAIN_SEED", "11"))
        training_steps = int(os.environ.get("WEBOTS_TRAIN_STEPS", "50000"))
        obstacle_count = int(os.environ.get("WEBOTS_OBSTACLE_COUNT", "1"))
        safety_shield_enabled = environment_flag("WEBOTS_SAFETY_SHIELD")
        safety_shield_config = safety_shield_config_from_env(
            os.environ,
            lidar_max_range=float(supervisor.getDevice("LDS-01").getMaxRange()),
        )
        model_variant_value = os.environ.get("WEBOTS_MODEL_VARIANT", "").strip()
        if model_variant_value in {".", ".."} or (
            model_variant_value
            and re.fullmatch(r"[A-Za-z0-9_.-]+", model_variant_value) is None
        ):
            raise ValueError(
                "WEBOTS_MODEL_VARIANT must be a path-safe single name containing only "
                "letters, digits, '.', '_' and '-'."
            )
        model_variant = model_variant_value
        segment_id_value = os.environ.get(
            "WEBOTS_RUN_ID",
            time.strftime(f"o{obstacle_count}_%Y%m%dT%H%M%S"),
        )
        if segment_id_value in {"", ".", ".."} or re.fullmatch(
            r"[A-Za-z0-9_.-]+", segment_id_value
        ) is None:
            raise ValueError(
                "WEBOTS_RUN_ID must be a path-safe single name containing only "
                "letters, digits, '.', '_' and '-'."
            )
        segment_id = segment_id_value
        obstacle_speed_range = (
            float(os.environ.get("WEBOTS_OBSTACLE_SPEED_MIN", "0.20")),
            float(os.environ.get("WEBOTS_OBSTACLE_SPEED_MAX", "0.42")),
        )
        reward_safety_distance = float(os.environ.get("WEBOTS_REWARD_SAFETY_DISTANCE", "0.20"))
        proximity_speed_distance = float(os.environ.get("WEBOTS_REWARD_PROXIMITY_DISTANCE", "0.60"))
        proximity_speed_weight = float(
            os.environ.get("WEBOTS_REWARD_PROXIMITY_SPEED_WEIGHT", "0.0")
        )
        reward_config = RewardConfig(
            progress_weight=float(os.environ.get("WEBOTS_REWARD_PROGRESS_WEIGHT", "3.0")),
            time_penalty=float(os.environ.get("WEBOTS_REWARD_TIME_PENALTY", "-0.01")),
            safety_weight=float(os.environ.get("WEBOTS_REWARD_SAFETY_WEIGHT", "-0.20")),
            smoothness_weight=float(os.environ.get("WEBOTS_REWARD_SMOOTHNESS_WEIGHT", "-0.02")),
            goal_reward=float(os.environ.get("WEBOTS_REWARD_GOAL", "30.0")),
            collision_penalty=float(os.environ.get("WEBOTS_REWARD_COLLISION_PENALTY", "-30.0")),
            timeout_penalty=float(os.environ.get("WEBOTS_REWARD_TIMEOUT_PENALTY", "-5.0")),
        )
        n_steps = int(os.environ.get("WEBOTS_PPO_N_STEPS", "512"))
        batch_size = int(os.environ.get("WEBOTS_PPO_BATCH_SIZE", "64"))
        n_epochs = int(os.environ.get("WEBOTS_PPO_N_EPOCHS", "10"))
        learning_rate = float(os.environ.get("WEBOTS_PPO_LEARNING_RATE", "0.0001"))
        ent_coef = float(os.environ.get("WEBOTS_PPO_ENT_COEF", "0.002"))
        gamma = float(os.environ.get("WEBOTS_PPO_GAMMA", "0.99"))
        horizontal_route_half_length = float(
            os.environ.get("WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH", "1.55")
        )
        vertical_route_half_length = float(
            os.environ.get("WEBOTS_VERTICAL_ROUTE_HALF_LENGTH", "1.10")
        )
        target_kl_value = os.environ.get("WEBOTS_PPO_TARGET_KL")
        target_kl = float(target_kl_value) if target_kl_value is not None else None
        if not 0 <= obstacle_count <= len(ALL_OBSTACLE_DEFS):
            raise ValueError("WEBOTS_OBSTACLE_COUNT must be between 0 and 3.")
        for def_name in ALL_OBSTACLE_DEFS[obstacle_count:]:
            supervisor.getFromDef(def_name).getField("translation").setSFVec3f([0.0, 0.0, -5.0])

        raw_env = WebotsPatrolEnv(
            supervisor,
            obstacle_defs=ALL_OBSTACLE_DEFS[:obstacle_count],
            action_repeat=4,
            max_episode_steps=300,
            enable_safety_shield=safety_shield_enabled,
            safety_shield=LidarSafetyShield(safety_shield_config),
            obstacle_speed_range=obstacle_speed_range,
            safety_distance=reward_safety_distance,
            proximity_speed_distance=proximity_speed_distance,
            proximity_speed_weight=proximity_speed_weight,
            reward_config=reward_config,
            horizontal_route_half_length=horizontal_route_half_length,
            vertical_route_half_length=vertical_route_half_length,
        )
        log_directory = repository_root / "artifacts/logs/webots" / f"seed_{seed}"
        if model_variant:
            log_directory /= model_variant
        log_directory.mkdir(parents=True, exist_ok=True)
        monitor_path = log_directory / f"training_{segment_id}.monitor.csv"
        env = Monitor(
            raw_env,
            filename=str(monitor_path),
            info_keywords=(
                "is_success",
                "collision",
                "timeout",
                "episode_min_clearance",
                "shield_active",
                "shield_intervention",
                "episode_shield_intervention_ratio",
                "episode_shield_intervention_mean",
                "scenario_seed",
                "route_axis",
                "reverse_route",
                "obstacle_speed_mean",
                "contact_collision",
                "lidar_collision",
            ),
            override_existing=True,
        )

        resume_value = os.environ.get("WEBOTS_RESUME_MODEL")
        policy_transfer_value = os.environ.get("WEBOTS_POLICY_TRANSFER_MODEL")
        if resume_value and policy_transfer_value:
            raise ValueError(
                "Set only one of WEBOTS_RESUME_MODEL and WEBOTS_POLICY_TRANSFER_MODEL."
            )
        transfer_value = os.environ.get(
            "WEBOTS_TRANSFER_MODEL",
            "artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip",
        )
        if resume_value:
            source_path = repository_root / resume_value
            model = PPO.load(source_path, env=env, device="cpu")
            if n_steps != model.n_steps or batch_size != model.batch_size:
                raise ValueError(
                    "A resume must keep the checkpoint rollout and batch sizes. "
                    f"Expected n_steps={model.n_steps}, batch_size={model.batch_size}."
                )
            model.ent_coef = ent_coef
            model.learning_rate = learning_rate
            model.lr_schedule = lambda _: learning_rate
            for parameter_group in model.policy.optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            # These deques are display-only rolling episode statistics. Carrying
            # them across curriculum stages makes the first logs of a harder
            # stage look artificially easy, even though policy/optimizer state
            # should be resumed.
            if model.ep_info_buffer is not None:
                model.ep_info_buffer.clear()
            if model.ep_success_buffer is not None:
                model.ep_success_buffer.clear()
            mode = "webots_resume"
            reset_num_timesteps = False
        else:
            model = PPO(
                "MlpPolicy",
                env,
                device="cpu",
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=ent_coef,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=target_kl,
                policy_kwargs={"net_arch": [128, 128]},
                tensorboard_log=str(repository_root / "artifacts/logs/tensorboard"),
                seed=seed,
                verbose=1,
            )
            source_path = repository_root / (policy_transfer_value or transfer_value)
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Transfer model not found: {source_path}. Finish kinematic training first."
                )
            source_model = PPO.load(source_path, device="cpu")
            model.policy.load_state_dict(source_model.policy.state_dict())
            mode = "webots_policy_transfer" if policy_transfer_value else "kinematic_transfer"
            reset_num_timesteps = True

        initial_action_std = os.environ.get("WEBOTS_PPO_INITIAL_ACTION_STD")
        if initial_action_std is not None:
            action_std = float(initial_action_std)
            if action_std <= 0.0:
                raise ValueError("WEBOTS_PPO_INITIAL_ACTION_STD must be positive.")
            model.policy.log_std.data.fill_(math.log(action_std))
        reset_linear_head = os.environ.get("WEBOTS_PPO_RESET_LINEAR_HEAD")
        if reset_linear_head is not None:
            linear_mean = float(reset_linear_head)
            if not -1.0 <= linear_mean <= 1.0:
                raise ValueError("WEBOTS_PPO_RESET_LINEAR_HEAD must be in [-1, 1].")
            model.policy.action_net.weight.data[0].zero_()
            model.policy.action_net.bias.data[0].fill_(linear_mean)

        # Compute this before saving ``ppo_webots_latest.zip``. During a resume,
        # that destination may be the same file as ``source_path``.
        source_model_sha256 = sha256(source_path) if source_path else None

        output_directory = repository_root / "artifacts/models/webots" / f"seed_{seed}"
        if model_variant:
            output_directory /= model_variant
        checkpoint_directory = output_directory / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = CheckpointCallback(
            save_freq=10240,
            save_path=str(checkpoint_directory),
            name_prefix=(
                f"ppo_webots_s{seed}_{model_variant}" if model_variant else f"ppo_webots_s{seed}"
            ),
        )
        starting_steps = model.num_timesteps
        print(
            f"WEBOTS_TRAINING_START mode={mode} seed={seed} obstacles={obstacle_count} "
            f"shield={int(safety_shield_enabled)} start_steps={starting_steps} "
            f"additional_steps={training_steps}",
            flush=True,
        )
        model.learn(
            total_timesteps=training_steps,
            reset_num_timesteps=reset_num_timesteps,
            callback=checkpoint,
            progress_bar=False,
            tb_log_name=f"ppo_webots_seed{seed}",
        )
        reset_health = raw_env.reset_health_diagnostics()
        if (
            reset_health["attempts"] != reset_health["successful"]
            or reset_health["finite_settle_checks"] != 2 * reset_health["successful"]
            or reset_health["readback_checks"] != reset_health["successful"]
        ):
            raise RuntimeError(
                f"Webots training reset-health counters are inconsistent: {reset_health}."
            )
        final_path = output_directory / f"ppo_webots_s{seed}_step{model.num_timesteps:09d}"
        latest_path = output_directory / "ppo_webots_latest"
        model.save(final_path)
        model.save(latest_path)
        metadata = {
            "mode": mode,
            "seed": seed,
            "model_variant": model_variant or None,
            "obstacle_count": obstacle_count,
            "obstacle_speed_range": list(obstacle_speed_range),
            "reward_config": asdict(reward_config),
            "reward_safety_distance": reward_safety_distance,
            "proximity_speed_distance": proximity_speed_distance,
            "proximity_speed_weight": proximity_speed_weight,
            "route_half_lengths": {
                "horizontal": horizontal_route_half_length,
                "vertical": vertical_route_half_length,
            },
            "ppo": {
                "n_steps": model.n_steps,
                "batch_size": model.batch_size,
                "n_epochs": model.n_epochs,
                "learning_rate": learning_rate,
                "ent_coef": ent_coef,
                "gamma": model.gamma,
                "target_kl": model.target_kl,
                "initial_action_std_override": (
                    float(initial_action_std) if initial_action_std is not None else None
                ),
                "reset_linear_head_mean": (
                    float(reset_linear_head) if reset_linear_head is not None else None
                ),
            },
            "safety_shield_enabled": safety_shield_enabled,
            "safety_shield_intervention_penalty": (raw_env.safety_shield_intervention_penalty),
            "safety_shield_config": asdict(raw_env.safety_shield.config),
            "reset_health": reset_health,
            "starting_webots_steps": starting_steps,
            "additional_steps_requested": training_steps,
            "final_model_steps": model.num_timesteps,
            "source_model": str(source_path) if source_path else None,
            "source_model_sha256": source_model_sha256,
            "final_model": f"{final_path}.zip",
            "final_model_sha256": sha256(Path(f"{final_path}.zip")),
            "monitor_log": str(monitor_path),
            "git": git_metadata(repository_root),
            "runtime": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "stable_baselines3": package_version("stable-baselines3"),
                "torch": package_version("torch"),
                "gymnasium": package_version("gymnasium"),
                "numpy": package_version("numpy"),
            },
        }
        metadata_text = yaml.safe_dump(metadata, sort_keys=False)
        (output_directory / f"run_metadata_step{model.num_timesteps:09d}.yaml").write_text(
            metadata_text, encoding="utf-8"
        )
        (output_directory / "run_metadata.yaml").write_text(metadata_text, encoding="utf-8")
        print(
            f"WEBOTS_TRAINING_OK steps={model.num_timesteps} model={final_path}.zip",
            flush=True,
        )
        exit_code = 0
    except Exception as error:
        failure_traceback = traceback.format_exc()
        print(failure_traceback, end="", flush=True)
        try:
            failure_directory = repository_root / "artifacts/logs/webots"
            if seed is not None:
                failure_directory /= f"seed_{seed}"
            else:
                failure_directory /= "seed_unknown"
            if model_variant:
                failure_directory /= model_variant
            failure_directory.mkdir(parents=True, exist_ok=True)
            runtime_diagnostics: object = None
            if raw_env is not None:
                try:
                    runtime_diagnostics = raw_env.runtime_diagnostics()
                except Exception as diagnostic_error:  # pragma: no cover - live API failure
                    runtime_diagnostics = (
                        f"unavailable: {type(diagnostic_error).__name__}: {diagnostic_error}"
                    )
            failure_metadata = {
                "status": "failed",
                "seed": seed,
                "model_variant": model_variant or None,
                "segment_id": segment_id,
                "obstacle_count": obstacle_count,
                "additional_steps_requested": training_steps,
                "current_model_steps": (int(model.num_timesteps) if model is not None else None),
                "source_model": str(source_path) if source_path is not None else None,
                "monitor_log": str(monitor_path) if monitor_path is not None else None,
                "exception": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": failure_traceback,
                },
                "runtime_diagnostics": runtime_diagnostics,
                "git": git_metadata(repository_root),
                "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            failure_path = failure_directory / f"failure_{segment_id}_{time.time_ns()}.yaml"
            temporary_path = failure_path.with_suffix(".yaml.tmp")
            temporary_path.write_text(
                yaml.safe_dump(failure_metadata, sort_keys=False), encoding="utf-8"
            )
            temporary_path.replace(failure_path)
            print(f"WEBOTS_TRAINING_FAILURE_METADATA path={failure_path}", flush=True)
        except Exception:
            traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        supervisor.step(int(supervisor.getBasicTimeStep()))
        supervisor.simulationQuit(exit_code)


if __name__ == "__main__":
    main()
