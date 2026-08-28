"""Simulator-independent observation, control, and reward components."""

from indoor_patrol_rl.core.control import DifferentialDriveMapper, VelocityRateLimiter
from indoor_patrol_rl.core.observation import ObservationBuilder
from indoor_patrol_rl.core.reward import RewardConfig, navigation_reward
from indoor_patrol_rl.core.safety_shield import (
    LidarSafetyShield,
    SafetyShieldConfig,
    SafetyShieldResult,
)

__all__ = [
    "DifferentialDriveMapper",
    "LidarSafetyShield",
    "ObservationBuilder",
    "RewardConfig",
    "SafetyShieldConfig",
    "SafetyShieldResult",
    "VelocityRateLimiter",
    "navigation_reward",
]
