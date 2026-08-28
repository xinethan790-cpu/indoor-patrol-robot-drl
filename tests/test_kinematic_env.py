from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from indoor_patrol_rl.envs import DynamicObstacleAvoidanceEnv


def test_gymnasium_contract() -> None:
    env = DynamicObstacleAvoidanceEnv(obstacle_count=3, max_episode_steps=20)
    check_env(env, skip_render_check=True)


def test_reset_is_reproducible() -> None:
    first = DynamicObstacleAvoidanceEnv()
    second = DynamicObstacleAvoidanceEnv()
    first_observation, _ = first.reset(seed=123)
    second_observation, _ = second.reset(seed=123)
    np.testing.assert_allclose(first_observation, second_observation)
    np.testing.assert_allclose(first.robot_position, second.robot_position)
    np.testing.assert_allclose(first.goal_position, second.goal_position)
    np.testing.assert_array_equal(first_observation[-2:], [-1.0, 0.0])


def test_step_returns_bounded_observation() -> None:
    env = DynamicObstacleAvoidanceEnv(obstacle_count=1)
    observation, _ = env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step(
        np.array([0.25, -0.5], dtype=np.float32)
    )
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert {
        "is_success",
        "collision",
        "timeout",
        "clearance",
        "episode_min_clearance",
        "path_length",
        "elapsed_sim_time",
        "executed_action",
    } <= info.keys()


def test_episode_truncates_at_configured_limit() -> None:
    env = DynamicObstacleAvoidanceEnv(obstacle_count=0, max_episode_steps=3)
    env.reset(seed=11)
    terminated = truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert not terminated
    assert truncated
