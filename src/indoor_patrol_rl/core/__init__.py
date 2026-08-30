"""Simulator-independent observation, control, and reward components."""

from indoor_patrol_rl.core.control import DifferentialDriveMapper, VelocityRateLimiter
from indoor_patrol_rl.core.observation import ObservationBuilder
from indoor_patrol_rl.core.reward import RewardConfig, navigation_reward
from indoor_patrol_rl.core.safety_shield import (
    SAFETY_SHIELD_CONFIG_ENV_FIELDS,
    SAFETY_SHIELD_CONFIG_ENV_PREFIX,
    LidarSafetyShield,
    SafetyShieldConfig,
    SafetyShieldResult,
    safety_shield_config_from_env,
)

__all__ = [
    "DifferentialDriveMapper",
    "LidarSafetyShield",
    "ObservationBuilder",
    "RewardConfig",
    "SAFETY_SHIELD_CONFIG_ENV_FIELDS",
    "SAFETY_SHIELD_CONFIG_ENV_PREFIX",
    "SafetyShieldConfig",
    "SafetyShieldResult",
    "VelocityRateLimiter",
    "navigation_reward",
    "safety_shield_config_from_env",
]
