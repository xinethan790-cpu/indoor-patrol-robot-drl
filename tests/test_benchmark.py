from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from indoor_patrol_rl.evaluation import (
    load_benchmark_config,
    require_test_confirmation,
    run_kinematic_benchmark,
    write_benchmark_outputs,
)


class StationaryPolicy:
    def predict(self, _observation: np.ndarray) -> np.ndarray:
        return np.array([-1.0, 0.0], dtype=np.float32)


def write_config(path: Path) -> None:
    path.write_text(
        """
seed_sets:
  validation: {start: 50, count: 2}
  test: {start: 80, count: 1}
planners: [first, second]
models: {}
backend:
  kinematic:
    defaults:
      width: 6.0
      height: 6.0
      obstacle_count: 0
      max_episode_steps: 2
conditions:
  - id: G0
    kinematic: {}
metrics: [success]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_config_merges_kinematic_defaults(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"
    write_config(path)

    config = load_benchmark_config(path)

    assert config.seed_sets["validation"].seeds == (50, 51)
    assert config.planners == ("first", "second")
    assert config.conditions[0].env_kwargs["obstacle_count"] == 0


def test_held_out_test_split_requires_explicit_confirmation() -> None:
    require_test_confirmation("validation", confirmed=False)
    with pytest.raises(PermissionError, match="confirm-test"):
        require_test_confirmation("test", confirmed=False)
    require_test_confirmation("test", confirmed=True)


def test_benchmark_matches_seeds_and_writes_both_outputs(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"
    write_config(path)
    config = load_benchmark_config(path)
    builders = {
        "first": lambda _env: StationaryPolicy(),
        "second": lambda _env: StationaryPolicy(),
    }

    records = run_kinematic_benchmark(config, policy_builders=builders)
    provenance = {
        "schema_version": 1,
        "model_path": "artifacts/models/example.zip",
        "model_sha256": "0" * 64,
        "model_variant": None,
        "training_seed": 11,
    }
    summary = write_benchmark_outputs(
        tmp_path / "results",
        records,
        provenance=provenance,
    )

    assert len(records) == 4
    for seed in (50, 51):
        matching = [record for record in records if record.scenario_seed == seed]
        assert {record.planner for record in matching} == {"first", "second"}
        assert len({record.reference_path_length_m for record in matching}) == 1
        assert len({record.outcome for record in matching}) == 1

    with (tmp_path / "results" / "episodes.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    saved_summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert len(rows) == 4
    assert summary == saved_summary
    assert saved_summary["episode_count"] == 4
    assert saved_summary["evaluation_provenance"] == provenance
    assert len(saved_summary["by_planner_condition"]) == 2
