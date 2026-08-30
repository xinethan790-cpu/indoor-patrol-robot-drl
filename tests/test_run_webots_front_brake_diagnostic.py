from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from indoor_patrol_rl.evaluation import (
    RISK_TRACE_COORDINATE_SEMANTICS,
    EpisodeRecord,
    build_shield_evidence_record,
    write_benchmark_outputs,
    write_episode_context_jsonl,
    write_shield_evidence_jsonl,
)

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/run_webots_front_brake_diagnostic.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs/eval/webots_front_brake_paired_confirmation_v2.yaml"
V1_CONFIG_PATH = REPOSITORY_ROOT / "configs/eval/webots_front_brake_paired_diagnostic_v1.yaml"


def _load_runner() -> Any:
    name = "_test_run_webots_front_brake_diagnostic"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository"
    for relative, payload in (
        ("scripts/run_webots_front_brake_diagnostic.py", b"runner"),
        ("controllers/webots_ppo_evaluate/webots_ppo_evaluate.py", b"controller"),
        ("worlds/webots_ppo_evaluate.wbt", b"world"),
        ("src/indoor_patrol_rl/core/safety_shield.py", b"shield"),
        ("src/indoor_patrol_rl/core/control.py", b"control"),
        ("src/indoor_patrol_rl/envs/webots_patrol.py", b"environment"),
        ("src/indoor_patrol_rl/evaluation/__init__.py", b"evaluation exports"),
        ("src/indoor_patrol_rl/evaluation/metrics.py", b"metrics"),
        (
            "src/indoor_patrol_rl/evaluation/webots_formal_config.py",
            b"formal config",
        ),
        ("src/indoor_patrol_rl/evaluation/episode_context.py", b"context"),
        ("src/indoor_patrol_rl/evaluation/shield_evidence.py", b"shield evidence"),
        ("src/indoor_patrol_rl/training/webots_curriculum.py", b"curriculum"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    config = root / "configs/eval/front_brake.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_PATH, config)
    model = root / RUNNER._FIXED_MODEL
    model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPOSITORY_ROOT / RUNNER._FIXED_MODEL, model)
    for relative in (RUNNER._V1_REPORT, RUNNER._MANIPULATION_REAUDIT):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, target)
    webots = root / "tools/webots.exe"
    webots.parent.mkdir(parents=True, exist_ok=True)
    webots.write_bytes(b"webots")
    return root, webots


def test_locked_config_uses_first_balanced_covariate_window(tmp_path: Path) -> None:
    root, _webots = _fixture(tmp_path)

    config = RUNNER.load_diagnostic_config(
        Path("configs/eval/front_brake.yaml"), repository_root=root
    )

    assert config.seeds == tuple(range(54_775, 54_895))
    assert RUNNER._strata_counts(config.seeds) == {
        "horizontal": {"non_reverse": 30, "reverse": 30},
        "vertical": {"non_reverse": 30, "reverse": 30},
    }
    assert RUNNER._obstacle_direction_counts(config.seeds) == {
        "negative": 60,
        "positive": 60,
    }
    assert RUNNER._first_balanced_seed_window(25_848, 120) == 54_775
    assert RUNNER._seed_window_is_balanced(54_775, 120)
    assert not RUNNER._seed_window_is_balanced(54_774, 120)
    assert RUNNER._strata_counts(tuple(range(54_774, 54_894))) == {
        "horizontal": {"non_reverse": 29, "reverse": 31},
        "vertical": {"non_reverse": 30, "reverse": 30},
    }
    assert set(config.seeds).isdisjoint(RUNNER._V1_SEEDS)
    assert set(config.seeds).isdisjoint(RUNNER._VALIDATION_SEEDS)
    assert set(config.seeds).isdisjoint(RUNNER._TEST_SEEDS)
    reference = config.arm_for_role("reference")
    candidate = config.arm_for_role("candidate")
    assert set(reference.controller_env) == set(candidate.controller_env)
    assert {
        name: value
        for name, value in reference.controller_env.items()
        if name != "WEBOTS_SAFETY_SHIELD"
    } == {
        name: value
        for name, value in candidate.controller_env.items()
        if name != "WEBOTS_SAFETY_SHIELD"
    }
    assert reference.controller_env["WEBOTS_SAFETY_SHIELD"] == "0"
    assert candidate.controller_env["WEBOTS_SAFETY_SHIELD"] == "1"
    assert dict(reference.controller_env) == RUNNER._REFERENCE_ENV
    assert dict(candidate.controller_env) == RUNNER._CANDIDATE_ENV
    assert candidate.controller_env["WEBOTS_SAFETY_SHIELD_FRONT_HALF_ANGLE"] == (
        "1.0471975511965976"
    )


def test_config_rejects_seed_parameter_and_access_drift(tmp_path: Path) -> None:
    root, _webots = _fixture(tmp_path)
    config_path = root / "configs/eval/front_brake.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["scenario_seed_start"] = 25_848
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="54775"):
        RUNNER.load_diagnostic_config(config_path, repository_root=root)

    shutil.copyfile(CONFIG_PATH, config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["arms"][1]["controller_env"]["WEBOTS_SAFETY_SHIELD_FRONT_HALF_ANGLE"] = "1.0"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="single pre-registered definition"):
        RUNNER.load_diagnostic_config(config_path, repository_root=root)

    shutil.copyfile(CONFIG_PATH, config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["preregistration"]["validation_or_test"] = "allowed"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="forbid"):
        RUNNER.load_diagnostic_config(config_path, repository_root=root)


def test_runner_rejects_sealed_v1_config(tmp_path: Path) -> None:
    root, _webots = _fixture(tmp_path)
    config_path = root / "configs/eval/front_brake.yaml"
    shutil.copyfile(V1_CONFIG_PATH, config_path)

    with pytest.raises(PermissionError, match="only the locked confirmation-v2"):
        RUNNER.load_diagnostic_config(config_path, repository_root=root)


def test_prior_evidence_hashes_restrictions_and_gate_are_locked(tmp_path: Path) -> None:
    root, _webots = _fixture(tmp_path)
    config_path = root / "configs/eval/front_brake.yaml"
    config = RUNNER.load_diagnostic_config(config_path, repository_root=root)

    assert config.v1_report.portable_path == RUNNER._V1_REPORT
    assert config.v1_report.sha256 == RUNNER._V1_REPORT_SHA256
    assert config.manipulation_reaudit.portable_path == RUNNER._MANIPULATION_REAUDIT
    assert config.manipulation_reaudit.sha256 == RUNNER._MANIPULATION_REAUDIT_SHA256
    assert config.prior_evidence_restrictions == RUNNER._PRIOR_EVIDENCE_RESTRICTIONS
    assert config.gate == RUNNER.CandidateGate(True, 0.25, 0.05, 2, 1.15, 24, 0)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["prior_evidence"]["v1_paired_report"]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="sealed v1 evidence binding"):
        RUNNER.load_diagnostic_config(config_path, repository_root=root)

    shutil.copyfile(CONFIG_PATH, config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["prior_evidence"]["restrictions"]["v1_v2_pooled_fallback"] = "allowed"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="pooled fallback"):
        RUNNER.load_diagnostic_config(config_path, repository_root=root)


def test_prior_evidence_file_drift_is_rejected(tmp_path: Path) -> None:
    root, _webots = _fixture(tmp_path)
    reaudit = root / RUNNER._MANIPULATION_REAUDIT
    reaudit.write_bytes(reaudit.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        RUNNER.load_diagnostic_config(Path("configs/eval/front_brake.yaml"), repository_root=root)


def test_dry_run_is_paired_diagnostic_only_and_writes_nothing(tmp_path: Path) -> None:
    root, webots = _fixture(tmp_path)

    exit_code, plan = RUNNER.execute_diagnostic(
        repository_root=root,
        config_path=Path("configs/eval/front_brake.yaml"),
        webots_executable=webots,
        dry_run=True,
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
        source_verifier=lambda _root: "a" * 40,
    )

    assert exit_code == 0
    assert plan["split"] == "diagnostic"
    assert plan["validation_or_test_access"] == "forbidden"
    assert plan["candidate_count"] == 1
    assert plan["parameter_scan"] == "forbidden"
    assert plan["route_strata_counts"]["vertical"]["reverse"] == 30
    assert plan["initial_obstacle_direction_counts"] == {
        "negative": 60,
        "positive": 60,
    }
    assert len(plan["runs"]) == 2
    first, second = plan["runs"]
    first_env = first["environment"]
    second_env = second["environment"]
    assert first_env["WEBOTS_EVAL_SPLIT"] == second_env["WEBOTS_EVAL_SPLIT"] == ("diagnostic")
    assert first_env["WEBOTS_EVAL_SEED_START"] == second_env["WEBOTS_EVAL_SEED_START"] == "54775"
    differences = {name for name in first_env if first_env.get(name) != second_env.get(name)}
    assert differences == {
        "WEBOTS_PORT",
        "WEBOTS_EVAL_PLANNER_LABEL",
        "WEBOTS_EVAL_RUN_NAME",
        "WEBOTS_SAFETY_SHIELD",
    }
    assert "WEBOTS_CONFIRM_TEST" not in first_env
    assert not (root / f"artifacts/results/{RUNNER._FIXED_EXPERIMENT_ID}").exists()
    for run in plan["runs"]:
        assert not (root / "artifacts/results" / run["run_name"]).exists()
    assert plan["prior_evidence"]["restrictions"] == RUNNER._PRIOR_EVIDENCE_RESTRICTIONS
    assert plan["source_bindings"]["v1_paired_report"]["sha256"] == (RUNNER._V1_REPORT_SHA256)
    assert plan["source_bindings"]["manipulation_boundary_reaudit"]["sha256"] == (
        RUNNER._MANIPULATION_REAUDIT_SHA256
    )


def test_cli_defaults_to_locked_confirmation_v2_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_execute(**kwargs: Any) -> tuple[int, dict[str, Any]]:
        captured.update(kwargs)
        return 0, {"dry_run": True}

    monkeypatch.setattr(RUNNER, "execute_diagnostic", fake_execute)
    exit_code = RUNNER.main(["--webots", str(tmp_path / "webots.exe"), "--dry-run"])

    assert exit_code == 0
    assert captured["config_path"] == Path(
        "configs/eval/webots_front_brake_paired_confirmation_v2.yaml"
    )
    assert captured["dry_run"] is True


def test_exact_one_sided_mcnemar_uses_success_discordances() -> None:
    assert RUNNER._exact_one_sided_mcnemar_p(0, 0) == 1.0
    assert RUNNER._exact_one_sided_mcnemar_p(6, 0) == pytest.approx(1 / 64)
    assert RUNNER._exact_one_sided_mcnemar_p(5, 1) == pytest.approx(7 / 64)
    assert RUNNER._exact_one_sided_mcnemar_p(0, 6) == 1.0
    with pytest.raises(ValueError, match="cannot be negative"):
        RUNNER._exact_one_sided_mcnemar_p(-1, 2)


def test_active_float32_boundary_step_still_binds_the_declared_speed_cap() -> None:
    step = {
        "step": 1,
        "shield_active": True,
        "policy_action": [1.0, 0.0],
        "shielded_action": [0.9999989867210388, 0.0],
        "shield_normalized_speed_limit": 0.9999990014894453,
        "shield_intervention": 1.0132789611816406e-6,
    }
    intervention = 1.0132789611816406e-6
    metrics = RUNNER._shield_evidence_metrics(
        {
            25_737: {
                "steps": [step],
                "summary": {
                    "shield_enabled": True,
                    "active_step_count": 1,
                    "intervention_sum": intervention,
                    "intervention_epsilon": 1e-6,
                },
            }
        }
    )

    assert metrics["active_step_count"] == 1
    assert metrics["braking_step_count"] == 1
    assert metrics["speed_cap_binding_step_count"] == 1
    assert metrics["active_operational_semantics_step_count"] == 1
    assert metrics["recorded_active_matches_float32_norm_step_count"] == 1
    assert metrics["recorded_intervention_matches_float32_norm_step_count"] == 1
    assert metrics["active_angular_action_float32_bits_unchanged_step_count"] == 1
    assert metrics["all_angular_action_float32_bits_unchanged_step_count"] == 1


def _single_candidate_step_metrics(step: dict[str, Any]) -> dict[str, Any]:
    return RUNNER._shield_evidence_metrics(
        {
            54_775: {
                "steps": [step],
                "summary": {
                    "shield_enabled": True,
                    "active_step_count": int(bool(step["shield_active"])),
                    "intervention_sum": float(step["shield_intervention"]),
                    "intervention_epsilon": 1e-6,
                },
            }
        }
    )


def test_cap_binding_requires_exact_float32_bits_not_a_numeric_tolerance() -> None:
    metrics = _single_candidate_step_metrics(
        {
            "step": 1,
            "shield_active": True,
            "policy_action": [1.0, 0.0],
            "shielded_action": [0.5, 0.0],
            "shield_normalized_speed_limit": float(np.nextafter(np.float32(0.5), np.float32(1.0))),
            "shield_intervention": 0.5,
        }
    )

    assert metrics["braking_step_count"] == 1
    assert metrics["speed_cap_binding_step_count"] == 0
    assert metrics["active_operational_semantics_step_count"] == 0


def test_operational_semantics_reject_intervention_or_angular_bit_drift() -> None:
    intervention_drift = _single_candidate_step_metrics(
        {
            "step": 1,
            "shield_active": True,
            "policy_action": [1.0, 0.0],
            "shielded_action": [0.5, 0.0],
            "shield_normalized_speed_limit": 0.5,
            "shield_intervention": 0.5000001,
        }
    )
    assert intervention_drift["recorded_intervention_matches_float32_norm_step_count"] == 0
    assert intervention_drift["active_operational_semantics_step_count"] == 0

    one_ulp = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    angular_drift = _single_candidate_step_metrics(
        {
            "step": 1,
            "shield_active": True,
            "policy_action": [1.0, 0.0],
            "shielded_action": [0.5, one_ulp],
            "shield_normalized_speed_limit": 0.5,
            "shield_intervention": 0.5,
        }
    )
    assert angular_drift["maximum_absolute_angular_action_delta"] < 1e-6
    assert angular_drift["active_angular_action_float32_bits_unchanged_step_count"] == 0
    assert angular_drift["active_operational_semantics_step_count"] == 0


def test_inactive_angular_one_ulp_drift_invalidates_turn_blend_zero_profile() -> None:
    one_ulp = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    candidate_metrics = RUNNER._shield_evidence_metrics(
        {
            54_775: {
                "steps": [
                    {
                        "step": 1,
                        "shield_active": True,
                        "policy_action": [1.0, 0.0],
                        "shielded_action": [0.5, 0.0],
                        "shield_normalized_speed_limit": 0.5,
                        "shield_intervention": 0.5,
                    },
                    {
                        "step": 2,
                        "shield_active": False,
                        "policy_action": [0.0, 0.0],
                        "shielded_action": [0.0, one_ulp],
                        "shield_normalized_speed_limit": 1.0,
                        "shield_intervention": 0.0,
                    },
                ],
                "summary": {
                    "shield_enabled": True,
                    "active_step_count": 1,
                    "intervention_sum": 0.5,
                    "intervention_epsilon": 1e-6,
                },
            }
        }
    )
    reference_metrics = RUNNER._shield_evidence_metrics(
        {
            54_775: {
                "steps": [
                    {
                        "step": 1,
                        "shield_active": False,
                        "policy_action": [0.0, 0.0],
                        "shielded_action": [0.0, 0.0],
                        "shield_normalized_speed_limit": 1.0,
                        "shield_intervention": 0.0,
                    }
                ],
                "summary": {
                    "shield_enabled": False,
                    "active_step_count": 0,
                    "intervention_sum": 0.0,
                    "intervention_epsilon": 1e-6,
                },
            }
        }
    )

    def result(role: str, metrics: dict[str, Any]) -> Any:
        arm = RUNNER.Arm(role, role, role, role, {})
        return RUNNER.ParsedArmResult(arm, {}, {}, {}, metrics, {}, {})

    validity = RUNNER._secondary_manipulation_check(
        result("reference", reference_metrics),
        result("candidate", candidate_metrics),
        expected_episodes=1,
    )

    assert candidate_metrics["maximum_absolute_angular_action_delta"] < 1e-6
    assert candidate_metrics["active_angular_action_float32_bits_unchanged_step_count"] == 1
    assert candidate_metrics["all_angular_action_float32_bits_unchanged_step_count"] == 1
    assert validity["checks"]["candidate_preserves_angular_action"] is True
    assert validity["checks"]["candidate_all_steps_preserve_angular_action_float32_bits"] is False
    assert validity["passed"] is False


def _gate_input(
    *,
    reduction: float | None = 0.25,
    p_value: float = 0.049,
    new_timeouts: int = 2,
    median_ratio: float | None = 1.15,
    common_success: int = 24,
    reference_errors: int = 0,
    candidate_errors: int = 0,
) -> dict[str, Any]:
    return {
        "overall": {
            "collision": {"relative_reduction": reduction},
            "paired_success_endpoint": {"exact_one_sided_mcnemar_p": p_value},
            "timeouts": {"new_timeout_count": new_timeouts},
            "completion_delay_on_reference_success": {
                "median_ratio": median_ratio,
                "paired_success_count": common_success,
            },
            "errors": {
                "reference_count": reference_errors,
                "candidate_count": candidate_errors,
            },
        }
    }


def test_gate_requires_every_boundary_and_strict_significance() -> None:
    gate = RUNNER.CandidateGate(True, 0.25, 0.05, 2, 1.15, 24, 0)
    assert RUNNER._gate_report(gate, paired=_gate_input())["passed"] is True
    assert RUNNER._gate_report(gate, paired=_gate_input(p_value=0.05))["passed"] is False
    assert RUNNER._gate_report(gate, paired=_gate_input(new_timeouts=3))["passed"] is False
    assert RUNNER._gate_report(gate, paired=_gate_input(median_ratio=None))["passed"] is False
    assert RUNNER._gate_report(gate, paired=_gate_input(common_success=23))["passed"] is False
    assert RUNNER._gate_report(gate, paired=_gate_input(reference_errors=1))["passed"] is False
    assert RUNNER._gate_report(gate, paired=_gate_input(candidate_errors=1))["passed"] is False


def test_report_is_staged_verified_and_atomically_renamed(tmp_path: Path) -> None:
    output = tmp_path / "paired_report"
    report = {"schema_version": 1, "status": "gate_failed", "value": [1, 2, 3]}

    RUNNER._write_report(report, output_directory=output)

    assert {path.name for path in output.iterdir()} == {
        "paired_diagnostic_report.json",
        "paired_diagnostic_report.yaml",
        "paired_diagnostic_report.sha256",
    }
    assert not output.with_name(output.name + ".staging").exists()
    with pytest.raises(FileExistsError, match="final or staging"):
        RUNNER._write_report(report, output_directory=output)

    blocked = tmp_path / "blocked"
    staging = blocked.with_name(blocked.name + ".staging")
    staging.mkdir()
    marker = staging / "failure_evidence.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="final or staging"):
        RUNNER._write_report(report, output_directory=blocked)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_preseal_artifact_rehash_detects_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "episodes.csv"
    artifact.write_text("locked\n", encoding="utf-8")
    arm = RUNNER.Arm("reference", "reference", "run", "planner", {})
    binding = {
        "path": "episodes.csv",
        "sha256": RUNNER._sha256_file(artifact),
    }
    result = RUNNER.ParsedArmResult(
        arm=arm,
        rows_by_seed={},
        contexts_by_seed={},
        shield_evidence_by_seed={},
        shield_evidence_metrics={},
        metrics={},
        artifacts={
            "summary_json": dict(binding),
            "episodes_csv": dict(binding),
            "episode_context_jsonl": dict(binding),
            "shield_evidence_jsonl": dict(binding),
        },
    )
    RUNNER._verify_parsed_artifacts([result], repository_root=tmp_path)

    artifact.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        RUNNER._verify_parsed_artifacts([result], repository_root=tmp_path)


def _record(seed: int, planner: str, outcome: str = "success") -> EpisodeRecord:
    success = outcome == "success"
    return EpisodeRecord(
        schema_version="1.0",
        episode_id=f"diagnostic-DB1-{seed}-{planner}",
        split="diagnostic",
        condition_id="DB1",
        scenario_seed=seed,
        planner=planner,
        outcome=outcome,  # type: ignore[arg-type]
        is_success=success,
        collision=outcome == "collision",
        timeout=outcome == "timeout",
        step_count=10,
        elapsed_sim_s=1.28,
        completion_time_s=1.28 if success else None,
        episode_return=1.0,
        path_length_m=1.0,
        reference_path_length_m=2.2,
        path_length_ratio=1 / 2.2 if success else None,
        minimum_clearance_m=0.3,
        near_collision_ratio=0.0,
        action_delta_sq_mean=0.0,
        inference_latency_p50_ms=0.1,
        inference_latency_p95_ms=0.2,
        error_type=None,
    )


def _risk_trace() -> dict[str, Any]:
    sample = {
        "step": 10,
        "post_elapsed_s": 1.28,
        "post_robot_root_xy": [1.03, 0.0],
        "post_robot_gps_xy": [1.0, 0.0],
        "post_robot_yaw": 0.0,
        "post_goal_distance": 0.1,
        "post_global_clearance": 0.3,
        "pre_action_front_clearance": 0.4,
        "pre_action_left_clearance": 0.5,
        "pre_action_right_clearance": 0.6,
        "post_lidar_argmin": {"range_m": 0.4, "bearing_rad": 0.0},
        "post_obstacle_xy": [[0.0, 0.2]],
        "post_obstacle_motion": [{"motion_axis": "y", "speed_mps": 0.2, "direction": 1.0}],
        "interval_policy_action": [0.1, 0.0],
        "interval_executed_action": [0.05, 0.0],
        "interval_commanded_linear": 0.12,
        "interval_commanded_angular": 0.0,
        "post_contact_collision": False,
        "post_lidar_collision": False,
    }
    return {
        "sample_period_s": 0.128,
        "sample_semantics": ("pre_action_sector_and_interval_action_with_post_control_state_v1"),
        "coordinate_semantics": RISK_TRACE_COORDINATE_SEMANTICS,
        "lidar_min_range_m": 0.12,
        "lidar_max_range_m": 3.5,
        "window_steps": 16,
        "terminal_window": [sample],
        "minimum_clearance_window": [sample],
    }


def _context(record: EpisodeRecord) -> dict[str, Any]:
    axis, reverse, obstacle_direction = RUNNER._scenario_covariates_for_seed(record.scenario_seed)
    if axis == "horizontal":
        start = [1.1 if reverse else -1.1, 0.0]
        goal = [-start[0], 0.0]
        motion_axis = "y"
    else:
        start = [0.0, 1.1 if reverse else -1.1]
        goal = [0.0, -start[1]]
        motion_axis = "x"
    return {
        "schema_version": "1.1",
        "identity": {
            "episode_id": record.episode_id,
            "split": "diagnostic",
            "condition_id": "DB1",
            "scenario_seed": record.scenario_seed,
            "planner": record.planner,
        },
        "route_axis": axis,
        "reverse_route": reverse,
        "start_xy": start,
        "goal_xy": goal,
        "start_yaw": 0.0,
        "dynamic_obstacles": [
            {
                "def_name": "DYNAMIC_OBSTACLE_1",
                "motion_axis": motion_axis,
                "initial_xy": [0.0, 0.0],
                "speed": 0.2,
                "initial_direction": obstacle_direction,
                "lower": -1.05,
                "upper": 1.05,
            }
        ],
        "termination": {
            "contact_collision": False,
            "lidar_collision": False,
            "terminal_step": 10,
            "terminal_xy": goal,
            "terminal_yaw": 0.0,
            "goal_distance": 0.1,
            "policy_action": [0.1, 0.0],
            "executed_action": [0.05, 0.0],
        },
        "risk_trace": _risk_trace(),
    }


def _shield_evidence_record(
    record: EpisodeRecord,
    *,
    enabled: bool,
    intervene: bool = True,
) -> dict[str, Any]:
    policy_f32 = np.asarray([0.5, 0.1], dtype=np.float32)
    active = enabled and intervene
    shielded_f32 = np.asarray([-0.2, 0.1], dtype=np.float32) if active else policy_f32.copy()
    policy = policy_f32.tolist()
    shielded = shielded_f32.tolist()
    intervention = float(np.linalg.norm(np.subtract(shielded_f32, policy_f32, dtype=np.float32)))
    steps = [
        {
            "step_count": step,
            "policy_action": policy,
            "shielded_action": shielded,
            "executed_action": [-0.25, 0.1] if active else [0.4, 0.1],
            "shield_enabled": enabled,
            "shield_active": active,
            "shield_intervention": intervention if active else 0.0,
            "shield_front_clearance": 0.2 if active else 1.0,
            "shield_left_clearance": 1.0,
            "shield_right_clearance": 1.0,
            "shield_speed_fraction_limit": 0.4 if active else 1.0,
            "shield_normalized_speed_limit": -0.2 if active else 1.0,
        }
        for step in range(1, 11)
    ]
    return build_shield_evidence_record(
        identity={
            "episode_id": record.episode_id,
            "split": "diagnostic",
            "condition_id": "DB1",
            "scenario_seed": record.scenario_seed,
            "planner": record.planner,
        },
        steps=steps,
        intervention_epsilon=1e-6,
        terminal_info={
            "step_count": 10,
            "shield_intervention_epsilon": 1e-6,
            "episode_shield_intervention_ratio": 1.0 if active else 0.0,
            "episode_shield_intervention_mean": intervention if active else 0.0,
        },
    )


def _fake_webots(repository_root: Path, *, candidate_intervenes: bool = True):
    def run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        planner = environment["WEBOTS_EVAL_PLANNER_LABEL"]
        run_name = environment["WEBOTS_EVAL_RUN_NAME"]
        output = repository_root / "artifacts/results" / run_name
        seeds = range(54_775, 54_895)
        episodes = [_record(seed, planner) for seed in seeds]
        contexts = [_context(record) for record in episodes]
        context_binding = write_episode_context_jsonl(
            output / "episode_context.jsonl",
            contexts,
            repository_root=repository_root,
        )
        enabled = environment["WEBOTS_SAFETY_SHIELD"] == "1"
        evidence_binding = write_shield_evidence_jsonl(
            output / "shield_evidence.jsonl",
            [
                _shield_evidence_record(
                    record,
                    enabled=enabled,
                    intervene=(candidate_intervenes if enabled else False),
                )
                for record in episodes
            ],
            repository_root=repository_root,
        )
        arm = RUNNER.Arm(
            arm_id="fake",
            role=("candidate" if environment["WEBOTS_SAFETY_SHIELD"] == "1" else "reference"),
            run_name=run_name,
            planner_label=planner,
            controller_env={
                name: value
                for name, value in environment.items()
                if name == "WEBOTS_SAFETY_SHIELD" or name.startswith("WEBOTS_SAFETY_SHIELD_")
            },
        )
        provenance = {
            "schema_version": 1,
            "model_path": RUNNER._FIXED_MODEL,
            "model_sha256": RUNNER._FIXED_MODEL_SHA256,
            "model_variant": RUNNER._FIXED_MODEL_VARIANT,
            "training_seed": 11,
            "split": "diagnostic",
            "condition_id": "DB1",
            "planner": "ppo",
            "planner_label": planner,
            "deterministic_policy": True,
            "scenario_seed_start": 54_775,
            "scenario_seed_count": 120,
            "scenario_seeds": list(seeds),
            "obstacle_count": 1,
            "obstacle_cross_lanes": None,
            "obstacle_spawn_protocol": "training_random_cross_axis",
            "obstacle_lane_seed_transform": None,
            "obstacle_speed_range": [0.15, 0.30],
            "horizontal_route_half_length": 1.10,
            "vertical_route_half_length": 1.10,
            "static_layout": "train_a",
            "safety_shield_enabled": arm.role == "candidate",
            "safety_shield_config": RUNNER._expected_shield_config(arm),
            "reference_path_definition": "euclidean_start_to_goal",
            "robot_bounding_radius": 0.122,
            "reset_health": {
                "attempts": 120,
                "successful": 120,
                "finite_settle_checks": 240,
                "readback_checks": 120,
                "maximum_translation_error": 0.0,
                "maximum_start_xy_error": 0.0,
                "maximum_gps_root_xy_distance": 0.03,
                "maximum_yaw_error": 0.0,
                "maximum_linear_speed": 0.0,
                "maximum_angular_speed": 0.0,
            },
            "episode_context": context_binding,
            "shield_evidence": evidence_binding,
        }
        write_benchmark_outputs(output, episodes, provenance=provenance)
        provenance["episodes_csv_sha256"] = RUNNER._sha256_file(output / "episodes.csv")
        write_benchmark_outputs(output, episodes, provenance=provenance)
        return subprocess.CompletedProcess(command, 0)

    return run


def test_full_fake_run_validates_provenance_and_reports_itt_scope(tmp_path: Path) -> None:
    root, webots = _fixture(tmp_path)

    exit_code, report = RUNNER.execute_diagnostic(
        repository_root=root,
        config_path=Path("configs/eval/front_brake.yaml"),
        webots_executable=webots,
        dry_run=False,
        process_runner=_fake_webots(root),
        source_verifier=lambda _root: "a" * 40,
    )

    assert exit_code == RUNNER.EXIT_CANDIDATE_GATE_FAILED
    assert report["validation_split_was_run"] is False
    assert report["test_split_was_run"] is False
    assert report["source_bindings"]["safety_shield"]["sha256"]
    assert report["source_bindings"]["control"]["sha256"]
    assert report["source_bindings"]["webots_patrol_environment"]["sha256"]
    assert report["source_bindings"]["shield_evidence"]["sha256"]
    assert report["source_bindings"]["v1_paired_report"]["sha256"] == (RUNNER._V1_REPORT_SHA256)
    assert report["source_bindings"]["manipulation_boundary_reaudit"]["sha256"] == (
        RUNNER._MANIPULATION_REAUDIT_SHA256
    )
    assert report["prior_evidence"]["restrictions"] == RUNNER._PRIOR_EVIDENCE_RESTRICTIONS
    assert report["prior_evidence"]["v1_status_remains"] == "invalid_manipulation"
    assert (
        report["arms"][0]["resolved_safety_shield_config"]
        == report["arms"][1]["resolved_safety_shield_config"]
    )
    overall = report["paired_analysis"]["overall"]
    assert overall["paired_success_endpoint"] == {
        "rescues_reference_non_success_to_candidate_success": 0,
        "regressions_reference_success_to_candidate_non_success": 0,
        "discordant_total": 0,
        "exact_one_sided_mcnemar_p": 1.0,
        "alternative": "candidate_success_probability_is_higher",
    }
    completion = overall["completion_delay_on_reference_success"]
    assert completion["paired_success_count"] == 120
    assert completion["median_ratio"] == pytest.approx(1.0)
    assert completion["ratio_iqr"] == pytest.approx(0.0)
    assert completion["ratio_p90"] == pytest.approx(1.0)
    assert report["interpretation_scope"]["estimand"] == (
        "paired_intent_to_treat_whole_closed_loop_profile_effect"
    )
    assert (
        report["interpretation_scope"]["step_level_action_transformation_evidence_available"]
        is True
    )
    manipulation = report["manipulation_integrity_check"]
    assert manipulation["passed"] is True
    assert manipulation["reference_metrics"]["active_step_count"] == 0
    assert manipulation["candidate_metrics"]["active_step_count"] == 1_200
    assert manipulation["candidate_metrics"]["braking_step_count"] == 1_200
    assert manipulation["candidate_metrics"]["speed_cap_binding_step_count"] == 1_200
    assert manipulation["candidate_metrics"][
        "maximum_absolute_angular_action_delta"
    ] == pytest.approx(0.0)
    assert report["experiment_validity"] == {
        "manipulation_integrity_required": True,
        "manipulation_integrity_passed": True,
        "primary_efficacy_gate_passed": False,
        "overall_passed": False,
    }
    output = root / f"artifacts/results/{RUNNER._FIXED_EXPERIMENT_ID}"
    persisted = json.loads((output / "paired_diagnostic_report.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "gate_failed"
    assert (output / "paired_diagnostic_report.yaml").is_file()
    assert "paired_diagnostic_report.json" in (
        output / "paired_diagnostic_report.sha256"
    ).read_text(encoding="utf-8")
    assert not output.with_name(output.name + ".staging").exists()


def test_failed_manipulation_integrity_invalidates_the_experiment(tmp_path: Path) -> None:
    root, webots = _fixture(tmp_path)

    exit_code, report = RUNNER.execute_diagnostic(
        repository_root=root,
        config_path=Path("configs/eval/front_brake.yaml"),
        webots_executable=webots,
        dry_run=False,
        process_runner=_fake_webots(root, candidate_intervenes=False),
        source_verifier=lambda _root: "a" * 40,
    )

    assert exit_code == RUNNER.EXIT_INVALID_MANIPULATION
    assert report["status"] == "invalid_manipulation"
    assert report["manipulation_integrity_check"]["passed"] is False
    assert report["experiment_validity"]["overall_passed"] is False
    assert report["decision"]["selected"] == "invalidate_front_brake_confirmation_v2"


def test_paired_analysis_rejects_initial_context_mismatch() -> None:
    arm_reference = RUNNER.Arm("r", "reference", "r", "r", {})
    arm_candidate = RUNNER.Arm("c", "candidate", "c", "c", {})
    rows = {
        1: {
            "outcome": "success",
            "is_success": True,
            "collision": False,
            "timeout": False,
            "completion_time_s": 1.0,
        }
    }
    reference = RUNNER.ParsedArmResult(
        arm=arm_reference,
        rows_by_seed=rows,
        contexts_by_seed={
            1: {"route_axis": "horizontal", "reverse_route": False, "start_xy": [0, 0]}
        },
        shield_evidence_by_seed={},
        shield_evidence_metrics={},
        metrics={},
        artifacts={},
    )
    candidate = RUNNER.ParsedArmResult(
        arm=arm_candidate,
        rows_by_seed=rows,
        contexts_by_seed={
            1: {"route_axis": "horizontal", "reverse_route": False, "start_xy": [1, 0]}
        },
        shield_evidence_by_seed={},
        shield_evidence_metrics={},
        metrics={},
        artifacts={},
    )

    with pytest.raises(ValueError, match="initial context start_xy"):
        RUNNER._paired_analysis(reference, candidate, seeds=[1])
