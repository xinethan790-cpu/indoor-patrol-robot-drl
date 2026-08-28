"""Reward calculation shared by prototype and Webots environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 3.0
    time_penalty: float = -0.01
    safety_weight: float = -0.20
    smoothness_weight: float = -0.02
    goal_reward: float = 30.0
    collision_penalty: float = -30.0
    timeout_penalty: float = -5.0


def navigation_reward(
    *,
    progress: float,
    min_clearance: float,
    safety_distance: float,
    action: np.ndarray,
    previous_action: np.ndarray,
    success: bool,
    collision: bool,
    timeout: bool,
    config: RewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    config = config or RewardConfig()
    if safety_distance <= 0.0:
        raise ValueError("safety_distance must be positive.")
    action = np.asarray(action, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)
    if action.shape != (2,) or previous_action.shape != (2,):
        raise ValueError("action and previous_action must both have shape (2,).")

    safety_ratio = max(0.0, (safety_distance - min_clearance) / safety_distance)
    terms = {
        "progress": config.progress_weight * float(progress),
        "time": config.time_penalty,
        "safety": config.safety_weight * safety_ratio**2,
        "smoothness": config.smoothness_weight
        * float(np.linalg.norm(action - previous_action) ** 2),
        "terminal": 0.0,
    }
    if collision:
        terms["terminal"] = config.collision_penalty
    elif success:
        terms["terminal"] = config.goal_reward
    elif timeout:
        terms["terminal"] = config.timeout_penalty
    return float(sum(terms.values())), terms
