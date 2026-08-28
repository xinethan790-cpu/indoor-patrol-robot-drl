"""Validated curriculum configuration shared by training scripts and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    timesteps: int
    obstacle_count: int
    obstacle_speed_range: tuple[float, float]
    max_episode_steps: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Curriculum stage name cannot be empty.")
        if self.timesteps <= 0 or self.max_episode_steps <= 0:
            raise ValueError("Curriculum timesteps and episode limit must be positive.")
        if self.obstacle_count < 0:
            raise ValueError("Curriculum obstacle_count cannot be negative.")
        minimum, maximum = self.obstacle_speed_range
        if minimum < 0.0 or maximum < minimum:
            raise ValueError("Invalid curriculum obstacle speed range.")

    @property
    def env_kwargs(self) -> dict[str, Any]:
        return {
            "obstacle_count": self.obstacle_count,
            "obstacle_speed_range": self.obstacle_speed_range,
            "max_episode_steps": self.max_episode_steps,
        }


@dataclass(frozen=True)
class CurriculumConfig:
    n_envs: int
    checkpoint_freq: int
    stages: tuple[CurriculumStage, ...]

    def __post_init__(self) -> None:
        if self.n_envs <= 0 or self.checkpoint_freq <= 0:
            raise ValueError("n_envs and checkpoint_freq must be positive.")
        if not self.stages:
            raise ValueError("A curriculum must contain at least one stage.")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Curriculum stage names must be unique.")

    @property
    def total_timesteps(self) -> int:
        return sum(stage.timesteps for stage in self.stages)


def load_curriculum(path: Path) -> CurriculumConfig:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get("stages"), list):
        raise ValueError(f"Expected a curriculum mapping with stages in {path}.")
    stages = tuple(
        CurriculumStage(
            name=str(item["name"]),
            timesteps=int(item["timesteps"]),
            obstacle_count=int(item["env"]["obstacle_count"]),
            obstacle_speed_range=tuple(
                float(value) for value in item["env"]["obstacle_speed_range"]
            ),
            max_episode_steps=int(item["env"]["max_episode_steps"]),
        )
        for item in raw["stages"]
    )
    return CurriculumConfig(
        n_envs=int(raw["n_envs"]),
        checkpoint_freq=int(raw["checkpoint_freq"]),
        stages=stages,
    )
