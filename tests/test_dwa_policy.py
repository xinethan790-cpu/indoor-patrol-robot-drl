from __future__ import annotations

import numpy as np

from indoor_patrol_rl.policies import DWAConfig, DWAPlanner


def make_observation(
    *,
    lidar: np.ndarray | None = None,
    goal_bearing: float = 0.0,
    current_linear: float = 0.0,
    current_angular: float = 0.0,
) -> np.ndarray:
    config = DWAConfig()
    lidar_values = np.ones(config.lidar_beams, dtype=np.float32) if lidar is None else lidar
    tail = np.array(
        [
            0.5,
            np.sin(goal_bearing),
            np.cos(goal_bearing),
            current_linear / config.max_linear_speed,
            current_angular / config.max_angular_speed,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([lidar_values, np.zeros(config.lidar_beams), tail])


def test_dwa_accelerates_toward_visible_goal() -> None:
    planner = DWAPlanner()

    result = planner.plan(make_observation())

    assert result.linear_velocity > 0.0
    assert abs(result.angular_velocity) < 1e-9
    assert result.feasible_trajectories == result.evaluated_trajectories
    assert np.all((-1.0 <= result.action) & (result.action <= 1.0))
    left = (
        result.linear_velocity - 0.5 * planner.config.axle_length * result.angular_velocity
    ) / planner.config.wheel_radius
    right = (
        result.linear_velocity + 0.5 * planner.config.axle_length * result.angular_velocity
    ) / planner.config.wheel_radius
    assert max(abs(left), abs(right)) <= planner.config.max_wheel_speed


def test_dwa_slows_for_close_front_left_obstacle() -> None:
    lidar = np.ones(36, dtype=np.float32)
    lidar[0:4] = 0.08
    planner = DWAPlanner()

    open_result = planner.plan(make_observation())
    result = planner.plan(make_observation(lidar=lidar))

    assert result.linear_velocity < open_result.linear_velocity
    assert result.feasible_trajectories > 0


def test_dwa_uses_emergency_command_when_surrounded() -> None:
    planner = DWAPlanner()
    lidar = np.full(36, 0.02, dtype=np.float32)

    result = planner.plan(make_observation(lidar=lidar, goal_bearing=0.7))

    assert result.feasible_trajectories == 0
    assert result.linear_velocity == 0.0
    assert result.minimum_clearance == 0.0


def test_dwa_is_deterministic() -> None:
    planner = DWAPlanner()
    observation = make_observation(goal_bearing=-0.4)

    first = planner.plan(observation)
    second = planner.plan(observation)

    np.testing.assert_array_equal(first.action, second.action)
    assert first.cost == second.cost
