from __future__ import annotations

from indoor_patrol_rl.envs import DynamicObstacleAvoidanceEnv
from indoor_patrol_rl.policies import ReactiveAvoidancePolicy


def test_reactive_policy_action_is_valid() -> None:
    env = DynamicObstacleAvoidanceEnv()
    observation, _ = env.reset(seed=5)
    action = ReactiveAvoidancePolicy(env.lidar_beams).predict(observation)
    assert env.action_space.contains(action)
