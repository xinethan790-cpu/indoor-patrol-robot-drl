from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
RUNNER_PATH = REPOSITORY_ROOT / "scripts/run_webots_patrol_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_webots_patrol_campaign", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CAMPAIGN
SPEC.loader.exec_module(CAMPAIGN)

EXIT_CODE_BY_OUTCOME = CAMPAIGN.EXIT_CODE_BY_OUTCOME
PatrolCampaignError = CAMPAIGN.PatrolCampaignError
PatrolCampaignRequest = CAMPAIGN.PatrolCampaignRequest
run_patrol_campaign = CAMPAIGN.run_patrol_campaign

_SCENARIO_SCOPE = "validated_component_outside_patrol_scenario"
_DEPLOYMENT_PROFILE_SHA256 = "1" * 64
_SHIELD_SHA256 = "2" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_scenario(path: Path, *, mode: str) -> None:
    if mode == "operational":
        scenario_id = "patrol_operational_v1"
        speeds = (0.18, 0.16)
        bounds = ((-1.20, 0.80), (-1.30, 1.30))
    else:
        scenario_id = "patrol_stress_v1"
        speeds = (0.31, 0.27)
        bounds = ((-1.20, 1.20), (-1.55, 1.55))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"scenario_id: {scenario_id}",
                f"mode: {mode}",
                "inherits_r2a_system_qualification: false",
                "dynamic_obstacles:",
                "  - def_name: DYNAMIC_OBSTACLE_0",
                "    axis: 1",
                f"    bounds_m: [{bounds[0][0]}, {bounds[0][1]}]",
                f"    speed_mps: {speeds[0]}",
                "  - def_name: DYNAMIC_OBSTACLE_1",
                "    axis: 0",
                f"    bounds_m: [{bounds[1][0]}, {bounds[1][1]}]",
                f"    speed_mps: {speeds[1]}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for label, relative in CAMPAIGN.CRITICAL_SOURCE_PATHS.items():
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"fixture critical source: {label}\n", encoding="utf-8")
    _write_scenario(
        root / "configs/patrol/scenario_operational_v1.yaml",
        mode="operational",
    )
    _write_scenario(
        root / "configs/patrol/scenario_stress_v1.yaml",
        mode="stress",
    )
    model = root / "artifacts/models/patrol-model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixed campaign model")
    return root


def _request(
    root: Path,
    output: Path,
    *,
    scenario: str = "operational",
    seed_start: int = 7000,
    episodes: int = 4,
    port: int = 1262,
    episode_wall_timeout_s: float = 600.0,
) -> PatrolCampaignRequest:
    return PatrolCampaignRequest(
        repository_root=root,
        scenario=scenario,
        seed_start=seed_start,
        episodes=episodes,
        output_directory=output,
        port=port,
        episode_wall_timeout_s=episode_wall_timeout_s,
    )


def _argument(command: Sequence[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_csv(path: Path, row_count: int) -> None:
    path.write_text(
        "value\n" + "".join(f"{index}\n" for index in range(row_count)),
        encoding="utf-8",
    )


def _write_run_summary(
    root: Path,
    command: Sequence[str],
    *,
    outcome: str,
    model_path: str = "artifacts/models/patrol-model.zip",
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    seed = int(_argument(command, "--seed"))
    run_id = _argument(command, "--run-id")
    scenario = _argument(command, "--scenario")
    output = Path(_argument(command, "--output-dir"))
    output.mkdir(parents=True)
    scenario_path = root / f"configs/patrol/scenario_{scenario}_v1.yaml"
    model = root / model_path
    visits = {
        "completed": 4,
        "collision": 1,
        "safety_stop": 2,
        "timeout": 3,
    }[outcome]
    elapsed = {
        "completed": 10.0,
        "collision": 2.0,
        "safety_stop": 3.0,
        "timeout": 8.0,
    }[outcome]
    path_length = {
        "completed": 5.0,
        "collision": 1.0,
        "safety_stop": 1.5,
        "timeout": 3.0,
    }[outcome]
    clearance = {
        "completed": 0.40,
        "collision": 0.0,
        "safety_stop": 0.02,
        "timeout": 0.10,
    }[outcome]
    control_steps = {
        "completed": 5,
        "collision": 2,
        "safety_stop": 3,
        "timeout": 4,
    }[outcome]
    collision_count = int(outcome == "collision")
    proximity_count = int(outcome == "safety_stop")
    artifact_rows = {
        "trajectory.csv": control_steps,
        "checkpoints.csv": 4,
        "collisions.csv": collision_count,
        "proximity_events.csv": proximity_count,
        "ttc_escape_guard_events.csv": 2,
    }
    for basename, row_count in artifact_rows.items():
        _write_csv(output / basename, row_count)
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "backend": "Webots R2025a",
        "planner": "astar_local_waypoint_ppo",
        "scenario_seed": seed,
        "outcome": outcome,
        "failure_reason": None if outcome == "completed" else f"fixture {outcome}",
        "checkpoints_total": 4,
        "checkpoints_visited": visits,
        "completion_ratio": visits / 4,
        "collision_count": collision_count,
        "critical_proximity_count": proximity_count,
        "control_step_count": control_steps,
        "elapsed_simulation_time_s": elapsed,
        "actual_path_length_m": path_length,
        "minimum_clearance_m": clearance,
        "trajectory_file": "trajectory.csv",
        "checkpoints_file": "checkpoints.csv",
        "collisions_file": "collisions.csv",
        "proximity_events_file": "proximity_events.csv",
        "ttc_escape_guard_events_file": "ttc_escape_guard_events.csv",
        "patrol_scenario_id": f"patrol_{scenario}_v1",
        "patrol_scenario_mode": scenario,
        "patrol_scenario_config_sha256": _sha256(scenario_path),
        "patrol_scenario_qualified": False,
        "deployment_evidence_scope": _SCENARIO_SCOPE,
        "model_path": model_path,
        "model_sha256": _sha256(model),
        "model_training_domain": "webots",
        "system_deployment_id": "r2a-system-fixture",
        "system_deployment_profile_sha256": _DEPLOYMENT_PROFILE_SHA256,
        "safety_shield_enabled": True,
        "safety_shield_id": "r2a-shield-fixture",
        "safety_shield_sha256": _SHIELD_SHA256,
        "ttc_escape_guard_enabled": True,
        "ttc_escape_guard_id": CAMPAIGN.PATROL_TTC_ESCAPE_GUARD_ID,
        "ttc_escape_guard_evidence_scope": (CAMPAIGN.UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE),
        "dynamic_obstacle_count": 2,
        "dynamic_obstacle_speed_max_mps": (0.18 if scenario == "operational" else 0.31),
    }
    if overrides is not None:
        payload.update(overrides)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(payload, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _fixture_runner(
    root: Path,
    outcomes: Sequence[str],
    calls: list[tuple[str, ...]],
) -> Any:
    def runner(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"cwd": root, "check": False, "timeout": 600.0}
        calls.append(tuple(command))
        outcome = outcomes[len(calls) - 1]
        _write_run_summary(root, command, outcome=outcome)
        return subprocess.CompletedProcess(command, EXIT_CODE_BY_OUTCOME[outcome])

    return runner


def test_campaign_binds_identity_sources_and_artifact_hashes(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "campaign"
    outcomes = ("completed", "collision", "safety_stop", "timeout")
    calls: list[tuple[str, ...]] = []

    summary = run_patrol_campaign(
        _request(root, output),
        process_runner=_fixture_runner(root, outcomes, calls),
    )

    assert len(calls) == 4
    for offset, command in enumerate(calls):
        seed = 7000 + offset
        assert command[:2] == (
            sys.executable,
            str(root / "scripts/run_webots_patrol.py"),
        )
        assert "--headless" in command
        assert _argument(command, "--scenario") == "operational"
        assert _argument(command, "--seed") == str(seed)
        assert _argument(command, "--port") == "1262"
        assert Path(_argument(command, "--output-dir")) == (output / f"runs/seed-{seed}")

    assert summary["execution_status"] == "completed"
    assert summary["acceptance_status"] == "insufficient_evidence"
    assert summary["acceptance_gate"]["evaluated"] is False
    assert summary["patrol_scenario_qualified"] is False
    assert summary["scenario_seeds"] == [7000, 7001, 7002, 7003]
    for outcome in outcomes:
        assert summary[f"{outcome}_count"] == 1
        assert summary[f"{outcome}_rate"] == pytest.approx(0.25)
    assert summary["error_count"] == 0
    assert summary["completion_ratio_mean"] == pytest.approx(0.625)
    assert summary["elapsed_simulation_time_s_mean"] == pytest.approx(5.75)
    assert summary["actual_path_length_m_mean"] == pytest.approx(2.625)
    assert summary["minimum_clearance_m_global"] == pytest.approx(0.0)
    identity = summary["execution_identity"]
    assert identity["schema_version"] == "1.0"
    assert identity["backend"] == "Webots R2025a"
    assert identity["planner"] == "astar_local_waypoint_ppo"
    assert identity["model_training_domain"] == "webots"
    assert identity["patrol_scenario_qualified"] is False
    assert identity["safety_shield_enabled"] is True
    assert identity["ttc_escape_guard_enabled"] is True
    assert identity["dynamic_obstacle_count"] == 2
    assert identity["dynamic_obstacle_speed_max_mps"] == pytest.approx(0.18)

    assert set(summary["critical_sources"]) == {
        *CAMPAIGN.CRITICAL_SOURCE_PATHS,
        "scenario_config",
    }
    for record in summary["critical_sources"].values():
        assert record["sha256"] == _sha256(root / record["path"])

    with (output / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["outcome"] for row in rows] == list(outcomes)
    assert rows[0]["model_path"] == "artifacts/models/patrol-model.zip"
    assert rows[0]["safety_shield_enabled"] == "True"
    assert rows[0]["ttc_escape_guard_id"] == CAMPAIGN.PATROL_TTC_ESCAPE_GUARD_ID
    assert int(rows[0]["trajectory_row_count"]) == 5
    assert int(rows[0]["ttc_escape_guard_events_row_count"]) == 2
    assert rows[0]["trajectory_sha256"] == _sha256(output / "runs/seed-7000/trajectory.csv")
    assert rows[0]["ttc_escape_guard_events_sha256"] == _sha256(
        output / "runs/seed-7000/ttc_escape_guard_events.csv"
    )
    assert summary["episodes_sha256"] == _sha256(output / "episodes.csv")
    assert json.loads((output / "summary.json").read_text()) == summary


def test_campaign_accepts_completed_run_with_recovered_proximity_event(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "recovered-proximity"

    def runner(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        summary_path = _write_run_summary(
            root,
            command,
            outcome="completed",
            overrides={"critical_proximity_count": 1},
        )
        _write_csv(summary_path.parent / "proximity_events.csv", 1)
        return subprocess.CompletedProcess(command, 0)

    summary = run_patrol_campaign(
        _request(root, output, episodes=1),
        process_runner=runner,
    )

    assert summary["completed_count"] == 1
    with (output / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        episode = next(csv.DictReader(stream))
    assert int(episode["critical_proximity_count"]) == 1


def test_stress_scenario_is_forwarded_and_bound(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "stress-campaign"
    calls: list[tuple[str, ...]] = []

    summary = run_patrol_campaign(
        _request(root, output, scenario="stress", episodes=1),
        process_runner=_fixture_runner(root, ("completed",), calls),
    )

    assert _argument(calls[0], "--scenario") == "stress"
    assert summary["patrol_scenario_id"] == "patrol_stress_v1"
    assert summary["patrol_scenario_mode"] == "stress"
    assert summary["execution_identity"]["dynamic_obstacle_speed_max_mps"] == pytest.approx(0.31)


def test_existing_output_directory_is_rejected_before_launch(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    launched = False

    def runner(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        nonlocal launched
        launched = True
        raise AssertionError("runner must not be called")

    with pytest.raises(FileExistsError, match="already exists"):
        run_patrol_campaign(_request(root, output), process_runner=runner)

    assert not launched


def test_exit_one_is_a_campaign_error_even_if_summary_exists(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "exit-one"

    def runner(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        _write_run_summary(root, command, outcome="completed")
        return subprocess.CompletedProcess(command, 1)

    with pytest.raises(PatrolCampaignError, match="exit code 1"):
        run_patrol_campaign(
            _request(root, output, episodes=2),
            process_runner=runner,
        )

    assert not (output / "episodes.csv").exists()
    assert not (output / "summary.json").exists()


def test_missing_subrun_summary_is_a_campaign_error(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "missing-summary"

    def runner(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(PatrolCampaignError, match="without summary.json"):
        run_patrol_campaign(
            _request(root, output, episodes=1),
            process_runner=runner,
        )


def test_subrun_wall_timeout_is_a_campaign_error(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "wall-timeout"

    def runner(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 12.5
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(PatrolCampaignError, match="12.5s wall-clock timeout"):
        run_patrol_campaign(
            _request(
                root,
                output,
                episodes=1,
                episode_wall_timeout_s=12.5,
            ),
            process_runner=runner,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": "2.0"}, "schema_version"),
        ({"backend": "Kinematic"}, "backend"),
        ({"planner": "dwa"}, "planner"),
        ({"model_training_domain": "kinematic"}, "model_training_domain"),
        ({"deployment_evidence_scope": "qualified"}, "deployment_evidence_scope"),
        ({"patrol_scenario_qualified": True}, "patrol_scenario_qualified"),
        ({"safety_shield_enabled": False}, "safety_shield_enabled"),
        ({"ttc_escape_guard_enabled": False}, "ttc_escape_guard_enabled"),
        ({"ttc_escape_guard_id": "wrong-guard"}, "ttc_escape_guard_id"),
        ({"dynamic_obstacle_count": 3}, "dynamic_obstacle_count"),
        ({"dynamic_obstacle_speed_max_mps": 0.19}, "dynamic_obstacle_speed_max_mps"),
    ],
)
def test_subrun_execution_identity_is_strict(
    tmp_path: Path,
    overrides: Mapping[str, Any],
    message: str,
) -> None:
    root = _repository(tmp_path)
    output = tmp_path / f"invalid-{message}"

    def runner(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        _write_run_summary(
            root,
            command,
            outcome="completed",
            overrides=overrides,
        )
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(PatrolCampaignError, match=message):
        run_patrol_campaign(
            _request(root, output, episodes=1),
            process_runner=runner,
        )


@pytest.mark.parametrize(
    ("second_overrides", "message"),
    [
        ({"system_deployment_id": "r2a-system-changed"}, "system_deployment_id"),
        ({"safety_shield_id": "r2a-shield-changed"}, "safety_shield_id"),
    ],
)
def test_execution_identity_must_remain_constant_across_runs(
    tmp_path: Path,
    second_overrides: Mapping[str, Any],
    message: str,
) -> None:
    root = _repository(tmp_path)
    output = tmp_path / f"identity-drift-{message}"
    calls = 0

    def runner(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        overrides = None if calls == 0 else second_overrides
        calls += 1
        _write_run_summary(
            root,
            command,
            outcome="completed",
            overrides=overrides,
        )
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(PatrolCampaignError, match=message):
        run_patrol_campaign(
            _request(root, output, episodes=2),
            process_runner=runner,
        )


def test_model_path_is_fixed_even_when_bytes_match(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    copied_model = root / "artifacts/models/copied-model.zip"
    copied_model.write_bytes((root / "artifacts/models/patrol-model.zip").read_bytes())
    output = tmp_path / "model-path-drift"
    calls = 0

    def runner(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        model_path = (
            "artifacts/models/patrol-model.zip"
            if calls == 0
            else "artifacts/models/copied-model.zip"
        )
        calls += 1
        _write_run_summary(
            root,
            command,
            outcome="completed",
            model_path=model_path,
        )
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(PatrolCampaignError, match="model_path"):
        run_patrol_campaign(
            _request(root, output, episodes=2),
            process_runner=runner,
        )


def test_critical_source_drift_aborts_campaign(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "source-drift"

    def runner(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        _write_run_summary(root, command, outcome="completed")
        controller = root / CAMPAIGN.CRITICAL_SOURCE_PATHS["patrol_controller"]
        controller.write_text("changed while running\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(PatrolCampaignError, match="patrol_controller.*changed"):
        run_patrol_campaign(
            _request(root, output, episodes=1),
            process_runner=runner,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload, run: payload.update(trajectory_file="../trajectory.csv"),
            "portable CSV basename",
        ),
        (
            lambda payload, run: payload.pop("ttc_escape_guard_events_file"),
            "ttc_escape_guard_events_file",
        ),
        (
            lambda payload, run: _write_csv(run / "trajectory.csv", 1),
            "trajectory_file row count",
        ),
        (
            lambda payload, run: (run / "collisions.csv").unlink(),
            "collisions_file does not exist",
        ),
    ],
)
def test_csv_artifact_contract_is_strict(
    tmp_path: Path,
    mutator: Any,
    message: str,
) -> None:
    root = _repository(tmp_path)
    output = tmp_path / f"artifact-{message.split()[0]}"

    def runner(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        summary_path = _write_run_summary(root, command, outcome="completed")
        payload = json.loads(summary_path.read_text())
        mutator(payload, summary_path.parent)
        summary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(PatrolCampaignError, match=message):
        run_patrol_campaign(
            _request(root, output, episodes=1),
            process_runner=runner,
        )


def test_acceptance_gate_passes_at_exact_thresholds(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "acceptance-pass"
    outcomes = ("completed",) * 15 + ("collision",) * 3 + ("timeout",) * 2
    calls: list[tuple[str, ...]] = []

    summary = run_patrol_campaign(
        _request(root, output, episodes=20),
        process_runner=_fixture_runner(root, outcomes, calls),
    )

    assert summary["execution_status"] == "completed"
    assert summary["acceptance_status"] == "pass"
    assert summary["acceptance_gate"]["evaluated"] is True
    assert all(summary["acceptance_gate"]["checks"].values())
    assert summary["acceptance_failures"] == []
    assert summary["completed_rate"] == pytest.approx(0.75)
    assert summary["collision_rate"] == pytest.approx(0.15)
    assert summary["patrol_scenario_qualified"] is False
    assert summary["execution_identity"]["patrol_scenario_qualified"] is False


def test_acceptance_gate_records_evaluated_failures(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "acceptance-fail"
    outcomes = ("completed",) * 14 + ("collision",) * 4 + ("timeout",) * 2
    calls: list[tuple[str, ...]] = []

    summary = run_patrol_campaign(
        _request(root, output, episodes=20),
        process_runner=_fixture_runner(root, outcomes, calls),
    )

    assert summary["execution_status"] == "completed"
    assert summary["acceptance_status"] == "fail"
    assert summary["acceptance_gate"]["evaluated"] is True
    assert summary["acceptance_gate"]["checks"]["completed_rate"] is False
    assert summary["acceptance_gate"]["checks"]["collision_rate"] is False
    assert len(summary["acceptance_failures"]) == 2
    assert any("completed_rate" in item for item in summary["acceptance_failures"])
    assert any("collision_rate" in item for item in summary["acceptance_failures"])


def test_cli_returns_five_for_an_evaluated_acceptance_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "episode_count": 20,
        "completed_count": 14,
        "collision_count": 4,
        "safety_stop_count": 0,
        "timeout_count": 2,
        "acceptance_status": "fail",
    }
    monkeypatch.setattr(CAMPAIGN, "run_patrol_campaign", lambda request: summary)

    exit_code = CAMPAIGN.main(
        [
            "--output-dir",
            str(tmp_path / "unused"),
            "--episodes",
            "20",
        ]
    )

    assert exit_code == 5


def test_timeout_request_and_parser_defaults_are_strict(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    with pytest.raises(ValueError, match="must be positive"):
        _request(root, tmp_path / "invalid-timeout", episode_wall_timeout_s=0.0)
    args = CAMPAIGN.build_parser().parse_args(["--output-dir", "new-campaign"])
    assert args.episode_wall_timeout_s == 600.0
