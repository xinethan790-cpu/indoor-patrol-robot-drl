from __future__ import annotations

from pathlib import Path

import pytest

from indoor_patrol_rl.training import CurriculumConfig, CurriculumStage, load_curriculum


def test_repository_curriculum_has_four_stages_and_800k_steps() -> None:
    config = load_curriculum(Path("configs/train/curriculum.yaml"))

    assert len(config.stages) == 4
    assert config.total_timesteps == 800_000
    assert config.stages[0].obstacle_count == 0
    assert config.stages[-1].obstacle_count == 7


def test_curriculum_rejects_duplicate_stage_names() -> None:
    stage = CurriculumStage("same", 10, 0, (0.0, 0.0), 10)

    with pytest.raises(ValueError, match="unique"):
        CurriculumConfig(n_envs=1, checkpoint_freq=10, stages=(stage, stage))
