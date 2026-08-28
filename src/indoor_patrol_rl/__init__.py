"""Indoor patrol robot reinforcement-learning environments."""

from gymnasium.envs.registration import register, registry

ENV_ID = "IndoorPatrolKinematic-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="indoor_patrol_rl.envs:DynamicObstacleAvoidanceEnv",
    )

__all__ = ["ENV_ID"]
