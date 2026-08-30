from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from indoor_patrol_rl.evaluation.webots_formal_config import load_webots_formal_config

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
CONTROLLER_PATH = (
    REPOSITORY_ROOT
    / "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
)


def _load_evaluator_module() -> Any:
    module_name = "_test_webots_ppo_evaluate_controller"
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


def _clear_webots_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("WEBOTS_"):
            monkeypatch.delenv(name, raising=False)


def _formal_root(tmp_path: Path) -> tuple[Path, str]:
    config_path = tmp_path / "configs/eval/webots_formal.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        (REPOSITORY_ROOT / "configs/eval/webots_formal.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    config = load_webots_formal_config(config_path)
    return config_path, config.source_sha256


def _set_formal_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    condition: str = "G2",
    planner: str = "reactive",
) -> None:
    _clear_webots_environment(monkeypatch)
    _config_path, digest = _formal_root(tmp_path)
    values = {
        "WEBOTS_FORMAL_CONFIG": "configs/eval/webots_formal.yaml",
        "WEBOTS_EVAL_CONFIG_SHA256": digest,
        "WEBOTS_FORMAL_CAMPAIGN_ID": "campaign_v1",
        "WEBOTS_FINAL_LOCK_SHA256": "a" * 64,
        "WEBOTS_EVAL_RUN_NAME": f"formal_{condition.lower()}_{planner}",
        "WEBOTS_EVAL_SPLIT": "test",
        "WEBOTS_CONFIRM_TEST": "1",
        "WEBOTS_EVAL_CONDITION": condition,
        "WEBOTS_EVAL_PLANNER": planner,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_locked_formal_plan_resolves_g2_five_lane_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_formal_environment(monkeypatch, tmp_path)

    plan = EVALUATOR._resolve_formal_plan(tmp_path)

    assert plan.split == "test"
    assert plan.condition_id == "G2"
    assert plan.planner_label == "reactive"
    assert plan.training_seed is None
    assert plan.scenario_seeds == tuple(range(10_000, 10_020))
    assert plan.obstacle_cross_lanes == (0.00, -0.39, 0.39, 0.75, -0.77)


def test_formal_plan_refuses_direct_controller_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_formal_environment(monkeypatch, tmp_path)
    monkeypatch.delenv("WEBOTS_FINAL_LOCK_SHA256")

    with pytest.raises(PermissionError, match="locked matrix runner"):
        EVALUATOR._resolve_formal_plan(tmp_path)


def test_formal_plan_rejects_noncanonical_seed_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_formal_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("WEBOTS_EVAL_EPISODES", "19")

    with pytest.raises(ValueError, match="10000..10019"):
        EVALUATOR._resolve_formal_plan(tmp_path)


def test_formal_ppo_model_digest_is_checked_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_formal_environment(monkeypatch, tmp_path)
    baseline_plan = EVALUATOR._resolve_formal_plan(tmp_path)
    model = tmp_path / "artifacts/models/webots/seed_11/selected_step100.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"not-an-sb3-model")
    monkeypatch.setenv("WEBOTS_EVAL_MODEL", model.relative_to(tmp_path).as_posix())
    monkeypatch.setenv("WEBOTS_EVAL_MODEL_SHA256", "0" * 64)
    ppo_plan = replace(
        baseline_plan,
        planner="ppo",
        planner_label="ppo_seed11",
        training_seed=11,
    )

    with pytest.raises(ValueError, match="SHA256 check"):
        EVALUATOR._model_and_variant(tmp_path, ppo_plan)


def test_curriculum_validation_label_contract_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    values = {
        "WEBOTS_EVAL_SPLIT": "validation",
        "WEBOTS_EVAL_PLANNER": "ppo",
        "WEBOTS_MODEL_SEED": "11",
        "WEBOTS_EVAL_SEED_START": "5000",
        "WEBOTS_EVAL_EPISODES": "20",
        "WEBOTS_EVAL_CONDITION": "W0",
        "WEBOTS_EVAL_PLANNER_LABEL": "ppo_w0_step10",
        "WEBOTS_OBSTACLE_COUNT": "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    plan = EVALUATOR._resolve_validation_plan()

    assert plan.condition_id == "W0"
    assert plan.planner == "ppo"
    assert plan.planner_label == "ppo_w0_step10"
    assert plan.training_seed == 11
    assert plan.scenario_seeds == tuple(range(5000, 5020))


@pytest.mark.parametrize(
    ("seed_start", "episode_count"),
    [(9_999, 2), (10_000, 1), (10_019, 1)],
)
def test_validation_plan_rejects_any_held_out_test_seed_overlap(
    monkeypatch: pytest.MonkeyPatch,
    seed_start: int,
    episode_count: int,
) -> None:
    _clear_webots_environment(monkeypatch)
    monkeypatch.setenv("WEBOTS_EVAL_SPLIT", "validation")
    monkeypatch.setenv("WEBOTS_EVAL_SEED_START", str(seed_start))
    monkeypatch.setenv("WEBOTS_EVAL_EPISODES", str(episode_count))

    with pytest.raises(ValueError, match="must not overlap held-out test"):
        EVALUATOR._resolve_plan(REPOSITORY_ROOT)


@pytest.mark.parametrize("seed_start", [9_999, 10_020])
def test_validation_plan_allows_windows_adjacent_to_held_out_test(
    monkeypatch: pytest.MonkeyPatch,
    seed_start: int,
) -> None:
    _clear_webots_environment(monkeypatch)
    monkeypatch.setenv("WEBOTS_EVAL_SPLIT", "validation")
    monkeypatch.setenv("WEBOTS_EVAL_SEED_START", str(seed_start))
    monkeypatch.setenv("WEBOTS_EVAL_EPISODES", "1")

    assert EVALUATOR._resolve_plan(REPOSITORY_ROOT).scenario_seeds == (seed_start,)


def test_diagnostic_plan_defaults_to_unreserved_seed_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    monkeypatch.setenv("WEBOTS_EVAL_SPLIT", "diagnostic")

    plan = EVALUATOR._resolve_plan(REPOSITORY_ROOT)

    reserved = set(EVALUATOR.FORMAL_VALIDATION_SEEDS) | set(
        EVALUATOR.FORMAL_TEST_SEEDS
    )
    assert plan.split == "diagnostic"
    assert plan.scenario_seeds == tuple(range(20_000, 20_020))
    assert reserved.isdisjoint(plan.scenario_seeds)


@pytest.mark.parametrize(
    ("seed_start", "episode_count"),
    [
        (4_999, 2),
        (10_019, 2),
    ],
)
def test_diagnostic_plan_rejects_any_reserved_seed_overlap(
    monkeypatch: pytest.MonkeyPatch,
    seed_start: int,
    episode_count: int,
) -> None:
    _clear_webots_environment(monkeypatch)
    monkeypatch.setenv("WEBOTS_EVAL_SPLIT", "diagnostic")
    monkeypatch.setenv("WEBOTS_EVAL_SEED_START", str(seed_start))
    monkeypatch.setenv("WEBOTS_EVAL_EPISODES", str(episode_count))

    with pytest.raises(ValueError, match="must not overlap"):
        EVALUATOR._resolve_plan(REPOSITORY_ROOT)


def test_diagnostic_outputs_keep_diagnostic_split_and_custom_safe_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    values = {
        "WEBOTS_EVAL_SPLIT": "diagnostic",
        "WEBOTS_EVAL_PLANNER": "reactive",
        "WEBOTS_EVAL_SEED_START": "7000",
        "WEBOTS_EVAL_EPISODES": "3",
        "WEBOTS_EVAL_CONDITION": "D0",
        "WEBOTS_OBSTACLE_COUNT": "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    plan = EVALUATOR._resolve_plan(REPOSITORY_ROOT)
    provenance = EVALUATOR._provenance(
        REPOSITORY_ROOT,
        plan,
        model_path=None,
        model_variant=None,
    )
    accumulator = EVALUATOR.EpisodeAccumulator(
        episode_id="diagnostic-D0-7000-reactive",
        split=plan.split,
        condition_id=plan.condition_id,
        scenario_seed=plan.scenario_seeds[0],
        planner=plan.planner_label,
        reference_path_length=1.0,
    )
    accumulator.update(
        reward=1.0,
        info={
            "is_success": True,
            "collision": False,
            "timeout": False,
            "clearance": 0.5,
            "path_length": 1.0,
            "step_count": 1,
            "elapsed_sim_time": 0.128,
            "executed_action": [0.0, 0.0],
        },
        inference_latency_s=0.001,
    )

    output_directory = tmp_path / "diagnostic_results"
    EVALUATOR.write_benchmark_outputs(
        output_directory,
        [accumulator.finalize()],
        provenance=provenance,
    )
    summary = json.loads(
        (output_directory / "summary.json").read_text(encoding="utf-8")
    )

    assert plan.scenario_seeds == (7000, 7001, 7002)
    assert summary["split"] == "diagnostic"
    assert summary["evaluation_provenance"]["split"] == "diagnostic"
