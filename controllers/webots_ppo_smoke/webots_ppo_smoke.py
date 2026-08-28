"""Run a short PPO learning loop against the real Webots Gymnasium adapter."""

from __future__ import annotations

import os
from pathlib import Path

from controller import Supervisor
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from indoor_patrol_rl.envs import WebotsPatrolEnv


def main() -> None:
    supervisor = Supervisor()
    env = WebotsPatrolEnv(supervisor, action_repeat=4, max_episode_steps=160)
    check_env(env, warn=True, skip_render_check=True)

    training_steps = int(os.environ.get("WEBOTS_PPO_STEPS", "512"))
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        learning_rate=3e-4,
        seed=7,
        verbose=1,
    )
    model.learn(total_timesteps=training_steps, progress_bar=False)

    repository_root = Path(__file__).resolve().parents[2]
    output = repository_root / "artifacts" / "models" / "ppo_webots_smoke"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(
        f"WEBOTS_PPO_SMOKE_OK steps={training_steps} model={output}.zip",
        flush=True,
    )
    env.close()
    supervisor.step(int(supervisor.getBasicTimeStep()))
    supervisor.simulationQuit(0)


if __name__ == "__main__":
    main()
