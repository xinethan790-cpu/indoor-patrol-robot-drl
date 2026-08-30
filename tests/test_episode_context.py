from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from indoor_patrol_rl.envs.webots_patrol import WebotsPatrolEnv
from indoor_patrol_rl.evaluation import (
    EPISODE_CONTEXT_FILENAME,
    EPISODE_CONTEXT_SCHEMA_VERSION,
    write_episode_context_jsonl,
)

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
CONTROLLER_PATH = (
    REPOSITORY_ROOT / "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
)


def _load_evaluator_module() -> Any:
    module_name = "_test_episode_context_evaluator"
    fake_controller = types.ModuleType("controller")
    fake_controller.Supervisor = object  # type: ignore[attr-defined]
    previous_controller = sys.modules.get("controller")
    sys.modules["controller"] = fake_controller
    spec = importlib.util.spec_from_file_location(module_name, CONTROLLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_controller is None:
            sys.modules.pop("controller", None)
        else:
            sys.modules["controller"] = previous_controller
    return module


EVALUATOR = _load_evaluator_module()


def _context_record(episode_id: str = "diagnostic-D0-20000-reactive") -> dict[str, Any]:
    return {
        "schema_version": EPISODE_CONTEXT_SCHEMA_VERSION,
        "identity": {
            "episode_id": episode_id,
            "split": "diagnostic",
            "condition_id": "D0",
            "scenario_seed": 20_000,
            "planner": "reactive",
        },
        "route_axis": "horizontal",
        "reverse_route": False,
        "start_xy": [-1.55, 0.1],
        "goal_xy": [1.55, -0.1],
        "start_yaw": 0.2,
        "dynamic_obstacles": [],
        "termination": {
            "contact_collision": False,
            "lidar_collision": False,
            "terminal_step": 12,
            "terminal_xy": [1.4, -0.1],
            "terminal_yaw": 0.0,
            "goal_distance": 0.15,
            "policy_action": [0.1, -0.2],
            "executed_action": [0.05, -0.1],
        },
    }


def test_episode_context_writer_is_stable_and_returns_complete_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results" / EPISODE_CONTEXT_FILENAME

    first = write_episode_context_jsonl(
        path,
        [_context_record()],
        repository_root=tmp_path,
    )
    second = write_episode_context_jsonl(
        path,
        [_context_record()],
        repository_root=tmp_path,
    )

    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    assert first == second == {
        "path": "results/episode_context.jsonl",
        "schema_version": "1.0",
        "count": 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert rows == [_context_record()]


def test_episode_context_writer_rejects_non_diagnostic_and_non_finite_rows(
    tmp_path: Path,
) -> None:
    validation = _context_record()
    validation["identity"]["split"] = "validation"
    with pytest.raises(ValueError, match="diagnostic split"):
        write_episode_context_jsonl(
            tmp_path / EPISODE_CONTEXT_FILENAME,
            [validation],
            repository_root=tmp_path,
        )

    non_finite = _context_record()
    non_finite["start_yaw"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        write_episode_context_jsonl(
            tmp_path / EPISODE_CONTEXT_FILENAME,
            [non_finite],
            repository_root=tmp_path,
        )


class _VectorField:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def getSFVec3f(self) -> list[float]:
        return list(self.values)


def test_webots_environment_exposes_reset_and_terminal_context() -> None:
    env = object.__new__(WebotsPatrolEnv)
    env.current_scenario_seed = 20_000
    env.current_route_axis = "horizontal"
    env.current_reverse_route = True
    env.current_start_position = np.array([1.55, -0.1])
    env.goal_position = np.array([-1.55, 0.1])
    env.current_start_yaw = 0.25
    env.obstacle_defs = ("TRAIN_OBSTACLE_0", "TRAIN_OBSTACLE_1")
    env.obstacle_translations = [
        _VectorField([0.2, -0.8, 0.36]),
        _VectorField([-0.3, 0.4, 0.36]),
    ]
    env.obstacle_axes = np.array([1, 0], dtype=np.int8)
    env.obstacle_speeds = np.array([0.24, 0.31])
    env.obstacle_directions = np.array([-1.0, 1.0])
    env.step_count = 15
    env._position = lambda: np.array([-1.4, 0.08])  # type: ignore[method-assign]
    env._roll_pitch_yaw = lambda: np.array([0.0, 0.0, -0.2])  # type: ignore[method-assign]

    initial = env.episode_context_at_reset()
    terminal = env.episode_context_at_terminal(
        {
            "contact_collision": False,
            "lidar_collision": True,
            "step_count": 15,
            "distance_to_goal": 0.16,
            "policy_action": np.array([0.3, -0.4]),
            "executed_action": np.array([0.2, -0.1]),
        }
    )

    assert initial == {
        "route_axis": "horizontal",
        "reverse_route": True,
        "start_xy": [1.55, -0.1],
        "goal_xy": [-1.55, 0.1],
        "start_yaw": 0.25,
        "dynamic_obstacles": [
            {
                "def_name": "TRAIN_OBSTACLE_0",
                "motion_axis": "y",
                "initial_xy": [0.2, -0.8],
                "speed": 0.24,
                "initial_direction": -1.0,
                "lower": -1.05,
                "upper": 1.05,
            },
            {
                "def_name": "TRAIN_OBSTACLE_1",
                "motion_axis": "x",
                "initial_xy": [-0.3, 0.4],
                "speed": 0.31,
                "initial_direction": 1.0,
                "lower": -1.55,
                "upper": 1.55,
            },
        ],
    }
    assert terminal == {
        "contact_collision": False,
        "lidar_collision": True,
        "terminal_step": 15,
        "terminal_xy": [-1.4, 0.08],
        "terminal_yaw": -0.2,
        "goal_distance": 0.16,
        "policy_action": [0.3, -0.4],
        "executed_action": [0.2, -0.1],
    }


def test_evaluator_binds_context_only_for_diagnostic_split(tmp_path: Path) -> None:
    diagnostic_plan = SimpleNamespace(split="diagnostic")
    provenance: dict[str, Any] = {"split": "diagnostic"}
    EVALUATOR._bind_episode_context_artifact(
        tmp_path,
        diagnostic_plan,
        tmp_path / "results",
        [_context_record()],
        provenance,
    )
    assert provenance["episode_context"]["count"] == 1
    assert (tmp_path / "results" / EPISODE_CONTEXT_FILENAME).is_file()

    for split in ("validation", "test"):
        output = tmp_path / split
        split_provenance: dict[str, Any] = {"split": split}
        EVALUATOR._bind_episode_context_artifact(
            tmp_path,
            SimpleNamespace(split=split),
            output,
            [],
            split_provenance,
        )
        assert "episode_context" not in split_provenance
        assert not (output / EPISODE_CONTEXT_FILENAME).exists()


def test_formal_plan_fails_closed_on_explicit_context_enable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBOTS_EVAL_EPISODE_CONTEXT", "1")

    with pytest.raises(PermissionError, match="forbids diagnostic"):
        EVALUATOR._resolve_formal_plan(tmp_path)


def test_evaluator_rejects_nondeterministic_policy_and_records_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBOTS_EVAL_SPLIT", "diagnostic")
    monkeypatch.setenv("WEBOTS_EVAL_PLANNER", "reactive")
    monkeypatch.setenv("WEBOTS_EVAL_DETERMINISTIC_POLICY", "0")
    with pytest.raises(ValueError, match="requires deterministic"):
        EVALUATOR._resolve_plan(REPOSITORY_ROOT)

    monkeypatch.setenv("WEBOTS_EVAL_DETERMINISTIC_POLICY", "1")
    plan = EVALUATOR._resolve_plan(REPOSITORY_ROOT)
    provenance = EVALUATOR._provenance(
        REPOSITORY_ROOT,
        plan,
        model_path=None,
        model_variant=None,
    )
    assert provenance["deterministic_policy"] is True
