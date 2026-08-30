from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indoor_patrol_rl.evaluation.episode_context import (
    EPISODE_CONTEXT_SCHEMA_VERSION,
    RISK_TRACE_COORDINATE_SEMANTICS,
    RISK_TRACE_SAMPLE_SEMANTICS,
    write_episode_context_jsonl,
)
from indoor_patrol_rl.evaluation.metrics import EpisodeRecord, write_episode_records

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/analyze_webots_risk_trajectory.py"


def _load_analyzer() -> Any:
    name = "_test_analyze_webots_risk_trajectory"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = _load_analyzer()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode(seed: int) -> EpisodeRecord:
    episode_id = (
        f"diagnostic-{ANALYZER.RECOVERY_CONDITION_ID}-{seed}-"
        f"{ANALYZER.RECOVERY_PLANNER}"
    )
    return EpisodeRecord(
        schema_version="1.0",
        episode_id=episode_id,
        split="diagnostic",
        condition_id=ANALYZER.RECOVERY_CONDITION_ID,
        scenario_seed=seed,
        planner=ANALYZER.RECOVERY_PLANNER,
        outcome="collision",
        is_success=False,
        collision=True,
        timeout=False,
        step_count=3,
        elapsed_sim_s=0.384,
        completion_time_s=None,
        episode_return=-1.0,
        path_length_m=0.67,
        reference_path_length_m=2.2,
        path_length_ratio=None,
        minimum_clearance_m=0.0,
        near_collision_ratio=0.5,
        action_delta_sq_mean=0.0,
        inference_latency_p50_ms=0.1,
        inference_latency_p95_ms=0.2,
        error_type=None,
    )


def _sample(
    step: int,
    *,
    robot_root_x: float,
    obstacle_x: float = 1.0,
    lidar_range: float,
    lidar_bearing: float = 0.0,
    terminal: bool = False,
) -> dict[str, Any]:
    return {
        "step": step,
        "post_elapsed_s": step * 0.128,
        "post_robot_root_xy": [robot_root_x, 0.0],
        # Deliberately offset GPS so tests fail if geometry silently uses it.
        "post_robot_gps_xy": [robot_root_x + 0.01, 0.0],
        "post_robot_yaw": 0.0,
        "post_goal_distance": 2.0 - robot_root_x,
        "post_global_clearance": 0.0 if terminal else max(lidar_range - 0.1, 0.0),
        "pre_action_front_clearance": 0.25,
        "pre_action_left_clearance": 0.4,
        "pre_action_right_clearance": 0.3,
        "post_lidar_argmin": {"range_m": lidar_range, "bearing_rad": lidar_bearing},
        "post_obstacle_xy": [[obstacle_x, 0.0]],
        "post_obstacle_motion": [
            {"motion_axis": "x", "speed_mps": 0.0, "direction": 1.0}
        ],
        "interval_policy_action": [0.2, 0.0],
        "interval_executed_action": [0.1, 0.0],
        "interval_commanded_linear": 0.12,
        "interval_commanded_angular": 0.0,
        "post_contact_collision": False,
        "post_lidar_collision": terminal,
    }


def _context(seed: int, *, kind: str) -> dict[str, Any]:
    if kind == "physical":
        samples = [
            _sample(1, robot_root_x=0.20, lidar_range=0.60),
            _sample(2, robot_root_x=0.45, lidar_range=0.35),
            _sample(3, robot_root_x=0.67, lidar_range=0.13, terminal=True),
        ]
        start_xy = [0.0, 0.0]
    else:
        bearing = 0.0 if kind == "proxy" else np.pi / 2.0
        samples = [
            _sample(
                1,
                robot_root_x=0.65,
                lidar_range=0.20,
                lidar_bearing=bearing,
            ),
            _sample(
                2,
                robot_root_x=0.65,
                lidar_range=0.20,
                lidar_bearing=bearing,
            ),
            _sample(
                3,
                robot_root_x=0.65,
                lidar_range=0.13,
                lidar_bearing=bearing,
                terminal=True,
            ),
        ]
        start_xy = [0.65, 0.0]
    terminal = samples[-1]
    episode = _episode(seed)
    return {
        "schema_version": EPISODE_CONTEXT_SCHEMA_VERSION,
        "identity": {
            "episode_id": episode.episode_id,
            "split": "diagnostic",
            "condition_id": episode.condition_id,
            "scenario_seed": seed,
            "planner": episode.planner,
        },
        "route_axis": ANALYZER._expected_axis(seed),
        "reverse_route": False,
        "start_xy": start_xy,
        "goal_xy": [2.2, 0.0],
        "start_yaw": 0.0,
        "dynamic_obstacles": [
            {
                "def_name": "TRAIN_OBSTACLE_0",
                "motion_axis": "x",
                "initial_xy": [1.0, 0.0],
                "speed": 0.0,
                "initial_direction": 1.0,
                "lower": -1.55,
                "upper": 1.55,
            }
        ],
        "termination": {
            "contact_collision": False,
            "lidar_collision": True,
            "terminal_step": 3,
            "terminal_xy": terminal["post_robot_gps_xy"],
            "terminal_yaw": terminal["post_robot_yaw"],
            "goal_distance": terminal["post_goal_distance"],
            "policy_action": terminal["interval_policy_action"],
            "executed_action": terminal["interval_executed_action"],
        },
        "risk_trace": {
            "sample_period_s": 0.128,
            "sample_semantics": RISK_TRACE_SAMPLE_SEMANTICS,
            "coordinate_semantics": RISK_TRACE_COORDINATE_SEMANTICS,
            "lidar_min_range_m": 0.12,
            "lidar_max_range_m": 3.5,
            "window_steps": 16,
            "terminal_window": copy.deepcopy(samples),
            "minimum_clearance_window": copy.deepcopy(samples),
        },
    }


def _write_inputs(
    root: Path,
    *,
    seeds: list[int] | None = None,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    selected_seeds = seeds or list(ANALYZER.DIAGNOSTIC_SEEDS)
    kinds = ["physical", "proxy", "ambiguous"] + ["proxy"] * (
        len(selected_seeds) - 3
    )
    contexts = [
        _context(seed, kind=kind)
        for seed, kind in zip(selected_seeds, kinds, strict=True)
    ]
    episodes = [_episode(seed) for seed in selected_seeds]
    context_path = root / "inputs/episode_context.jsonl"
    episodes_path = root / "inputs/episodes.csv"
    write_episode_context_jsonl(context_path, contexts, repository_root=root)
    write_episode_records(episodes_path, episodes)
    return context_path, episodes_path, contexts


def _binding(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
    }


def _lock_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    yaml_path = report_path.with_suffix(".yaml")
    yaml_path.write_text("locked_collection: true\n", encoding="utf-8")
    report_path.with_suffix(".sha256").write_text(
        f"{_sha256(report_path)}  {report_path.name}\n"
        f"{_sha256(yaml_path)}  {yaml_path.name}\n",
        encoding="utf-8",
    )


def _write_collection_report(
    root: Path,
    *,
    context_path: Path,
    episodes_path: Path,
) -> tuple[Path, dict[str, Any]]:
    summary_path = context_path.with_name("summary.json")
    summary_path.write_text("{}\n", encoding="utf-8")
    report_path = (
        root
        / "artifacts/results"
        / ANALYZER.COLLECTION_EXPERIMENT_ID
        / "collection_report.json"
    )
    relative_report = report_path.relative_to(root).as_posix()
    environment = {
        "WEBOTS_MODEL_SEED": str(ANALYZER.RECOVERY_TRAINING_SEED),
        "WEBOTS_EVAL_MODEL": ANALYZER.RECOVERY_MODEL_PATH,
        "WEBOTS_EVAL_MODEL_SHA256": ANALYZER.RECOVERY_MODEL_SHA256,
        "WEBOTS_MODEL_VARIANT": ANALYZER.RECOVERY_MODEL_VARIANT,
        "WEBOTS_EVAL_SPLIT": "diagnostic",
        "WEBOTS_EVAL_SEED_START": str(ANALYZER.DIAGNOSTIC_SEED_START),
        "WEBOTS_EVAL_EPISODES": str(ANALYZER.DIAGNOSTIC_EPISODES),
        "WEBOTS_EVAL_CONDITION": ANALYZER.RECOVERY_CONDITION_ID,
        "WEBOTS_EVAL_PLANNER_LABEL": ANALYZER.RECOVERY_PLANNER,
        "WEBOTS_SAFETY_SHIELD": "0",
        "WEBOTS_EVAL_DETERMINISTIC_POLICY": "1",
        "WEBOTS_EVAL_EPISODE_CONTEXT": "1",
        "WEBOTS_RISK_TRAJECTORY_ID": ANALYZER.COLLECTION_EXPERIMENT_ID,
    }
    source_sha = "a" * 64
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": ANALYZER.COLLECTION_EXPERIMENT_ID,
        "status": ANALYZER.COLLECTION_STATUS,
        "generated_at_utc": "2026-08-30T00:00:00+00:00",
        "split": "diagnostic",
        "validation_split_was_run": False,
        "test_split_was_run": False,
        "performance_gate_applied": False,
        "source_bindings": {
            "git_revision": "b" * 40,
            "config": {"path": "configs/fake.yaml", "sha256": source_sha},
            "runner": {"path": "scripts/fake.py", "sha256": source_sha},
            "controller": {"path": "controllers/fake.py", "sha256": source_sha},
            "world": {"path": "worlds/fake.wbt", "sha256": source_sha},
            "model": {
                "path": ANALYZER.RECOVERY_MODEL_PATH,
                "sha256": ANALYZER.RECOVERY_MODEL_SHA256,
            },
            "webots_executable": {
                "path": "C:/Webots/webots.exe",
                "sha256": source_sha,
            },
        },
        "scenario": {
            "condition_id": ANALYZER.RECOVERY_CONDITION_ID,
            "scenario_seed_start": ANALYZER.DIAGNOSTIC_SEED_START,
            "episodes": ANALYZER.DIAGNOSTIC_EPISODES,
            "scenario_seeds": list(ANALYZER.DIAGNOSTIC_SEEDS),
            "axis_counts": ANALYZER.EXPECTED_AXIS_COUNTS,
            "obstacle_count": 1,
            "obstacle_speed_range": [0.15, 0.30],
            "horizontal_route_half_length": 1.10,
            "vertical_route_half_length": 1.10,
            "safety_shield": False,
            "deterministic_policy": True,
        },
        "collection": {
            "expected_episode_count": ANALYZER.DIAGNOSTIC_EPISODES,
            "episodes_csv_count": ANALYZER.DIAGNOSTIC_EPISODES,
            "episode_context_count": ANALYZER.DIAGNOSTIC_EPISODES,
            "risk_trace_count": ANALYZER.DIAGNOSTIC_EPISODES,
            "episode_context_schema_version": EPISODE_CONTEXT_SCHEMA_VERSION,
            "expected_route_axis_counts": ANALYZER.EXPECTED_AXIS_COUNTS,
            "adaptive_extension": "forbidden",
            "complete": True,
        },
        "model": {
            "path": ANALYZER.RECOVERY_MODEL_PATH,
            "sha256": ANALYZER.RECOVERY_MODEL_SHA256,
            "variant": ANALYZER.RECOVERY_MODEL_VARIANT,
            "training_seed": ANALYZER.RECOVERY_TRAINING_SEED,
        },
        "execution": {
            "command": ["webots"],
            "environment": environment,
            "port": 13_640,
            "exit_code": 0,
        },
        "descriptive_metrics_not_used_as_a_gate": {},
        "artifacts": {
            "result_directory": context_path.parent.relative_to(root).as_posix(),
            "summary_json": _binding(summary_path, root),
            "episodes_csv": _binding(episodes_path, root),
            "episode_context_jsonl": _binding(context_path, root),
        },
        "access": {"validation_or_test": "forbidden"},
        "written_reports": {
            "json": relative_report,
            "yaml": report_path.with_suffix(".yaml").relative_to(root).as_posix(),
            "sha256_sidecar": (
                report_path.with_suffix(".sha256").relative_to(root).as_posix()
            ),
        },
    }
    _lock_report(report_path, report)
    return report_path, report


def _write_locked_collection(root: Path) -> tuple[Path, Path, Path]:
    context_path, episodes_path, _contexts = _write_inputs(root)
    report_path, _report = _write_collection_report(
        root,
        context_path=context_path,
        episodes_path=episodes_path,
    )
    return report_path, context_path, episodes_path


def test_analyzer_requires_locked_collection_and_uses_root_gps_semantics(
    tmp_path: Path,
) -> None:
    report_path, context_path, _episodes_path = _write_locked_collection(tmp_path)
    output = tmp_path / "analysis"

    report = ANALYZER.analyze_risk_trajectory(
        collection_report_path=report_path,
        output_directory=output,
        repository_root=tmp_path,
    )

    metrics = json.loads(
        (output / "risk_episode_metrics.json").read_text(encoding="utf-8")
    )
    by_seed = {value["scenario_seed"]: value for value in metrics}
    physical = by_seed[20_494]
    proxy = by_seed[20_495]
    ambiguous = by_seed[20_496]
    assert physical["mechanism_classification"] == "physical_imminent_dynamic"
    assert physical["minimum_dynamic_ttc_defined"] is True
    assert physical["deceleration_reaction_evaluable"] is False
    assert proxy["mechanism_classification"] == "lidar_proxy_compatible"
    assert proxy["minimum_dynamic_ttc_s"] is None
    assert proxy["minimum_dynamic_ttc_defined"] is False
    assert ambiguous["mechanism_classification"] == "ambiguous"
    assert "not_uniquely_attributed" in ambiguous["ambiguity_reason"]
    assert report["mechanism_decision"]["decision"] == "lidar_proxy_dominant"
    assert report["inputs"]["collection_report"]["sha256"] == _sha256(report_path)
    assert report["inputs"]["episode_context"]["sha256"] == _sha256(context_path)
    assert report["analysis_protocol"]["geometry_robot_position_field"] == (
        "post_robot_root_xy"
    )
    assert report["analysis_protocol"]["terminal_position_validation_field"] == (
        "post_robot_gps_xy"
    )
    assert report["scope"]["exact_scenario_seed_count"] == 120
    assert (output / "risk_episode_metrics.csv").is_file()


def test_locked_collection_rejects_model_drift_and_artifact_sha_drift(
    tmp_path: Path,
) -> None:
    context_path, episodes_path, _contexts = _write_inputs(tmp_path)
    report_path, report = _write_collection_report(
        tmp_path,
        context_path=context_path,
        episodes_path=episodes_path,
    )
    report["model"]["sha256"] = "0" * 64
    _lock_report(report_path, report)
    with pytest.raises(PermissionError, match="locked recovery-v1 model"):
        ANALYZER.analyze_risk_trajectory(
            collection_report_path=report_path,
            output_directory=tmp_path / "analysis-model",
            repository_root=tmp_path,
        )

    report["model"]["sha256"] = ANALYZER.RECOVERY_MODEL_SHA256
    _lock_report(report_path, report)
    episodes_path.write_text(
        episodes_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="episodes_csv SHA-256 mismatch"):
        ANALYZER.analyze_risk_trajectory(
            collection_report_path=report_path,
            output_directory=tmp_path / "analysis-artifact",
            repository_root=tmp_path,
        )


def test_formal_scope_rejects_reserved_or_arbitrary_diagnostic_seeds(
    tmp_path: Path,
) -> None:
    seeds = [5_000, *list(ANALYZER.DIAGNOSTIC_SEEDS)[1:]]
    context_path, episodes_path, _contexts = _write_inputs(tmp_path, seeds=seeds)
    report_path, _report = _write_collection_report(
        tmp_path,
        context_path=context_path,
        episodes_path=episodes_path,
    )
    with pytest.raises(PermissionError, match="reserved validation/test seeds"):
        ANALYZER.analyze_risk_trajectory(
            collection_report_path=report_path,
            output_directory=tmp_path / "analysis",
            repository_root=tmp_path,
        )

    arbitrary = {
        f"diagnostic-DR1-{seed}-{ANALYZER.RECOVERY_PLANNER}": ANALYZER.EpisodeRow(
            episode_id=f"diagnostic-DR1-{seed}-{ANALYZER.RECOVERY_PLANNER}",
            split="diagnostic",
            condition_id="DR1",
            scenario_seed=seed,
            planner=ANALYZER.RECOVERY_PLANNER,
            outcome="collision",
        )
        for seed in range(30_000, 30_024)
    }
    with pytest.raises(PermissionError, match="exact diagnostic seeds"):
        ANALYZER._validate_locked_episode_scope(arbitrary, [])


def test_reader_rejects_window_drift_and_uses_gps_for_terminal_alignment(
    tmp_path: Path,
) -> None:
    seed = ANALYZER.DIAGNOSTIC_SEEDS[0]
    episode = _episode(seed)
    episodes_path = tmp_path / "episodes.csv"
    write_episode_records(episodes_path, [episode])
    rows = ANALYZER.read_episode_rows(episodes_path)
    context = _context(seed, kind="physical")
    context["risk_trace"]["minimum_clearance_window"][1][
        "interval_commanded_linear"
    ] = 0.01
    context_path = tmp_path / "episode_context.jsonl"
    context_path.write_text(json.dumps(context) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="disagree for duplicated control step"):
        ANALYZER.read_context_records(context_path, episodes=rows)

    context = _context(seed, kind="physical")
    context["termination"]["terminal_xy"] = context["risk_trace"][
        "terminal_window"
    ][-1]["post_robot_root_xy"]
    context_path.write_text(json.dumps(context) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="post_robot_gps_xy disagrees"):
        ANALYZER.read_context_records(context_path, episodes=rows)


def _mechanism_episode(
    classification: str,
    *,
    evaluable: bool = True,
    timely: bool = False,
    source: str = "lidar_only",
) -> dict[str, Any]:
    return {
        "terminal_collision_source": source,
        "mechanism_classification": classification,
        "deceleration_reaction_evaluable": evaluable,
        "timely_deceleration_response": timely,
    }


def test_mechanism_decision_separates_geometry_and_response_denominators() -> None:
    dynamic = [
        _mechanism_episode("physical_imminent_dynamic") for _ in range(40)
    ] + [_mechanism_episode("lidar_proxy_compatible") for _ in range(10)]
    proxy = [_mechanism_episode("lidar_proxy_compatible") for _ in range(40)] + [
        _mechanism_episode("physical_imminent_dynamic") for _ in range(10)
    ]
    mixed = [_mechanism_episode("lidar_proxy_compatible") for _ in range(25)] + [
        _mechanism_episode("physical_imminent_dynamic") for _ in range(25)
    ]

    dynamic_result = ANALYZER.mechanism_decision(dynamic)
    assert dynamic_result["decision"] == (
        "dynamic_short_horizon_response_insufficient"
    )
    assert dynamic_result["geometry"]["classifiable_lidar_only_episodes"] == 50
    assert dynamic_result["response"]["reaction_evaluable_physical_episodes"] == 40
    assert ANALYZER.mechanism_decision(proxy)["decision"] == "lidar_proxy_dominant"
    assert ANALYZER.mechanism_decision(mixed)["decision"] == "mixed"

    missing_onset_or_baseline = [
        _mechanism_episode("physical_imminent_dynamic", evaluable=False)
        for _ in range(40)
    ] + [_mechanism_episode("lidar_proxy_compatible") for _ in range(10)]
    unresolved = ANALYZER.mechanism_decision(missing_onset_or_baseline)
    assert unresolved["decision"] == "indeterminate"
    assert unresolved["response"]["reaction_evaluable_physical_episodes"] == 0
    assert unresolved["response"]["short_horizon_response_insufficient"]["total"] == 0


def test_reaction_missing_onset_or_baseline_is_indeterminate() -> None:
    no_onset = ANALYZER._reaction_metrics([], None)
    assert no_onset["deceleration_reaction_evaluable"] is False
    assert no_onset["deceleration_reaction_indeterminate_reason"] == (
        "risk_onset_not_detected"
    )
    onset = {
        "step": 3,
        "post_elapsed_s": 0.384,
        "interval_commanded_linear_mps": 0.12,
    }
    missing_baseline = ANALYZER._reaction_metrics([onset], onset)
    assert missing_baseline["deceleration_reaction_evaluable"] is False
    assert missing_baseline["deceleration_reaction_indeterminate_reason"] == (
        "missing_contiguous_pre_reaction_baseline"
    )


def test_exact_disc_ttc_rejects_radially_closing_near_miss() -> None:
    head_on = ANALYZER._relative_metrics(
        np.asarray([1.0, 0.0]),
        np.asarray([-1.0, 0.0]),
        horizon_s=1.0,
    )
    assert head_on["ttc_defined"] is True
    assert head_on["ttc_s"] == pytest.approx(1.0 - 0.122 - 0.20)

    near_miss = ANALYZER._relative_metrics(
        np.asarray([1.0, 0.4]),
        np.asarray([-1.0, 0.0]),
        horizon_s=1.0,
    )
    assert near_miss["closing_rate_mps"] > 0.0
    assert near_miss["ttc_defined"] is False
    assert near_miss["ttc_s"] is None


def test_wilson_interval_has_exact_empty_encoding_and_valid_bounds() -> None:
    assert ANALYZER.wilson_interval(0, 0) == {
        "successes": 0,
        "total": 0,
        "proportion": None,
        "low": None,
        "high": None,
    }
    interval = ANALYZER.wilson_interval(40, 50)
    assert interval["proportion"] == pytest.approx(0.8)
    assert 0.5 < interval["low"] < interval["proportion"] < interval["high"] < 1.0
