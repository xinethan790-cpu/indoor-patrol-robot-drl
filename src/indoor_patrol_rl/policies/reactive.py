"""A deterministic rule-based baseline for smoke tests and comparisons."""

from __future__ import annotations

import numpy as np


class ReactiveAvoidancePolicy:
    """Turn toward free space when blocked; otherwise steer toward the goal."""

    def __init__(self, lidar_beams: int, obstacle_threshold: float = 0.18) -> None:
        if lidar_beams < 8 or lidar_beams % 4 != 0:
            raise ValueError("lidar_beams must be divisible by four and at least eight.")
        self.lidar_beams = lidar_beams
        self.obstacle_threshold = obstacle_threshold

    def predict(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        lidar = observation[: self.lidar_beams]
        goal_offset = 2 * self.lidar_beams
        goal_sin = float(observation[goal_offset + 1])
        goal_cos = float(observation[goal_offset + 2])
        bearing = float(np.arctan2(goal_sin, goal_cos))

        front_window = np.concatenate([lidar[:2], lidar[-2:]])
        if float(front_window.min()) < self.obstacle_threshold:
            left = float(lidar[1 : self.lidar_beams // 2].mean())
            right = float(lidar[self.lidar_beams // 2 + 1 :].mean())
            turn = 1.0 if left >= right else -1.0
            return np.array([-0.35, turn], dtype=np.float32)

        turn = float(np.clip(bearing / (np.pi / 2.0), -1.0, 1.0))
        speed = float(np.clip(0.75 - 0.35 * abs(turn), -1.0, 1.0))
        return np.array([speed, turn], dtype=np.float32)
