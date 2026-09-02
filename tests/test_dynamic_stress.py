from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

import indoor_patrol_rl.evaluation.checkpoint_sweep as checkpoint_sweep
import indoor_patrol_rl.evaluation.dynamic_stress as dynamic_stress
import scripts.run_webots_dynamic_stress as dynamic_stress_cli
from indoor_patrol_rl.evaluation.dynamic_stress import (
    DYNAMIC_STRESS_SEEDS,
    RB2B_ENDPOINT_PATH,
    RB2B_ENDPOINT_SHA256,
    build_dynamic_stress_requests,
    finalize_existing_dynamic_stress,
    load_dynamic_stress_config,
    run_dynamic_stress,
)
from scripts.run_webots_dynamic_stress import build_parser

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
CONFIG_PATH = Path("configs/eval/webots_rb2b_posthoc_dynamic_stress_v1.yaml")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _copy_repository_file(root: Path, relative: Path) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPOSITORY_ROOT / relative, destination)


def _fixture_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    for relative in (
        CONFIG_PATH,
        RB2B_ENDPOINT_PATH,
        Path("configs/eval/webots_r2a_front_brake_v1.yaml"),
        Path("worlds/webots_ppo_evaluate.wbt"),
        Path("controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"),
        Path("src/indoor_patrol_rl/envs/webots_patrol.py"),
    ):
        _copy_repository_file(root, relative)
    webots = root / "webots"
    webots.write_bytes(b"fake Webots executable\n")
    (root / ".gitignore").write_text("artifacts/results/\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "lock dynamic stress fixture")
    return root


def _commit_config_change(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = root / CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutation(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    _git(root, "add", CONFIG_PATH.as_posix())
    _git(root, "commit", "-q", "-m", "mutate fixture config")


def test_config_and_requests_lock_exact_endpoint_matrix_without_formal_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture_repository(tmp_path)
    config = load_dynamic_stress_config(CONFIG_PATH, repository_root=root)
    requests = build_dynamic_stress_requests(
        config,
        webots_executable=root / "webots",
    )

    assert config.source_git_revision == _git(root, "rev-parse", "HEAD")
    assert config.model_path == root / RB2B_ENDPOINT_PATH
    assert hashlib.sha256(config.model_path.read_bytes()).hexdigest() == (
        RB2B_ENDPOINT_SHA256
    )
    assert [condition.condition_id for condition in config.conditions] == [
        "BASE1",
        "DENS3",
        "FAST3",
    ]
    assert [
        (
            condition.obstacle_count,
            condition.obstacle_speed_min,
            condition.obstacle_speed_max,
        )
        for condition in config.conditions
    ] == [(1, 0.18, 0.25), (3, 0.18, 0.25), (3, 0.26, 0.42)]
    assert len(config.expected_shield_config) == 15
    assert config.expected_shield_config["lidar_max_range"] == pytest.approx(3.5)

    monkeypatch.setenv("WEBOTS_EVAL_SPLIT", "test")
    monkeypatch.setenv("WEBOTS_CONFIRM_TEST", "1")
    monkeypatch.setenv("WEBOTS_FORMAL_CONFIG", "configs/eval/webots_formal.yaml")
    monkeypatch.setenv("WEBOTS_FINAL_LOCK_SHA256", "f" * 64)
    for request, condition in zip(requests, config.conditions, strict=True):
        assert request.steps == (81920,)
        assert request.seed_start == 8000
        assert request.episodes == 20
        assert request.exact_model_path == config.model_path
        assert request.exact_model_sha256 == RB2B_ENDPOINT_SHA256
        assert request.run_name(81920) == condition.run_name
        environment = checkpoint_sweep._environment(  # noqa: SLF001
            request,
            step=81920,
            model_path=config.model_path,
            model_sha256=RB2B_ENDPOINT_SHA256,
        )
        assert environment["WEBOTS_EVAL_SPLIT"] == "validation"
        assert environment["WEBOTS_EVAL_SEED_START"] == "8000"
        assert environment["WEBOTS_EVAL_EPISODES"] == "20"
        assert environment["WEBOTS_EVAL_MODEL_SHA256"] == RB2B_ENDPOINT_SHA256
        assert "WEBOTS_CONFIRM_TEST" not in environment
        assert "WEBOTS_FORMAL_CONFIG" not in environment
        assert "WEBOTS_FINAL_LOCK_SHA256" not in environment

    parser = build_parser()
    destinations = {action.dest for action in parser._actions}  # noqa: SLF001
    assert not {"split", "confirm_test", "formal_config"} & destinations
    with pytest.raises(SystemExit):
        parser.parse_args(["--split", "test"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.__setitem__("split", "test"), "split"),
        (
            lambda raw: raw.__setitem__("scenario_seed_start", 10000),
            "scenario_seed_start",
        ),
        (
            lambda raw: raw.__setitem__("test_split_was_run", True),
            "test_split_was_run",
        ),
        (
            lambda raw: raw["model"].__setitem__("sha256", "0" * 64),
            "model.sha256",
        ),
        (
            lambda raw: raw["shield_profile"]["expected_resolved_config"].__setitem__(
                "lidar_max_range", 4.0
            ),
            "lidar_max_range",
        ),
        (
            lambda raw: raw["conditions"][1].__setitem__(
                "obstacle_speed_range", [0.26, 0.42]
            ),
            "obstacle_speed_range",
        ),
        (
            lambda raw: raw.__setitem__("formal_config", "forbidden-even-here"),
            "extra",
        ),
    ],
)
def test_loader_rejects_split_identity_shield_and_condition_drift(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    root = _fixture_repository(tmp_path)
    _commit_config_change(root, mutation)

    with pytest.raises(ValueError, match=message):
        load_dynamic_stress_config(CONFIG_PATH, repository_root=root)


def test_dirty_worktree_stops_before_output_or_sweep(
    tmp_path: Path,
) -> None:
    root = _fixture_repository(tmp_path)
    config = load_dynamic_stress_config(CONFIG_PATH, repository_root=root)
    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    calls: list[str] = []

    def sweep_runner(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        calls.append("called")
        raise AssertionError("Sweep must not start from a dirty worktree.")

    with pytest.raises(ValueError, match="completely clean Git worktree"):
        run_dynamic_stress(
            config,
            webots_executable=root / "webots",
            sweep_runner=sweep_runner,
        )

    assert calls == []
    assert not config.output_directory.exists()
    assert not any(condition.sweep_directory.exists() for condition in config.conditions)


@pytest.mark.parametrize("existing_kind", ["aggregate", "sweep", "result"])
def test_all_output_destinations_are_preflighted_before_first_sweep(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    root = _fixture_repository(tmp_path)
    config = load_dynamic_stress_config(CONFIG_PATH, repository_root=root)
    requests = build_dynamic_stress_requests(
        config,
        webots_executable=root / "webots",
    )
    if existing_kind == "aggregate":
        existing = config.output_directory
    elif existing_kind == "sweep":
        existing = config.conditions[1].sweep_directory
    else:
        existing = root / "artifacts/results" / requests[2].run_name(81920)
    existing.mkdir(parents=True)
    (existing / "sentinel.txt").write_text("preserve\n", encoding="utf-8")
    calls: list[str] = []

    def sweep_runner(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        calls.append("called")
        raise AssertionError("Sweep must not start when any output exists.")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_dynamic_stress(
            config,
            webots_executable=root / "webots",
            sweep_runner=sweep_runner,
        )

    assert calls == []
    assert (existing / "sentinel.txt").read_text(encoding="utf-8") == "preserve\n"


def _write_fake_sweep(
    root: Path,
    request: Any,
    output_directory: Path,
    *,
    success_count: int,
) -> Mapping[str, Any]:
    condition = request.condition
    planner = request.planner(81920)
    run_name = request.run_name(81920)
    result_directory = root / "artifacts/results" / run_name
    result_directory.mkdir(parents=True)
    episodes_path = result_directory / "episodes.csv"
    fieldnames = [
        "schema_version",
        "episode_id",
        "split",
        "scenario_seed",
        "planner",
        "condition_id",
        "outcome",
        "is_success",
        "collision",
        "timeout",
        "error_type",
    ]
    with episodes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, scenario_seed in enumerate(DYNAMIC_STRESS_SEEDS):
            success = index < success_count
            writer.writerow(
                {
                    "schema_version": 1.0,
                    "episode_id": f"validation-{condition}-{scenario_seed}-{planner}",
                    "split": "validation",
                    "scenario_seed": scenario_seed,
                    "planner": planner,
                    "condition_id": condition,
                    "outcome": "success" if success else "collision",
                    "is_success": success,
                    "collision": not success,
                    "timeout": False,
                    "error_type": "",
                }
            )
    episodes_sha256 = hashlib.sha256(episodes_path.read_bytes()).hexdigest()
    collision_count = 20 - success_count
    aggregate = {
        "planner": planner,
        "condition_id": condition,
        "episodes": 20,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": 0,
        "error_count": 0,
        "success_rate": success_count / 20,
        "collision_rate": collision_count / 20,
        "path_length_ratio_mean": 1.0 if success_count else None,
        "completion_time_s_mean": 4.0 if success_count else None,
    }
    summary = {
        "split": "validation",
        "episode_count": 20,
        "planners": [planner],
        "conditions": [condition],
        "by_planner": [
            {
                name: value
                for name, value in aggregate.items()
                if name != "condition_id"
            }
        ],
        "by_planner_condition": [aggregate],
        "evaluation_provenance": {
            "schema_version": 1,
            "model_path": RB2B_ENDPOINT_PATH.as_posix(),
            "model_sha256": RB2B_ENDPOINT_SHA256,
            "model_variant": "r2b_rebaseline_v1",
            "training_seed": 11,
            "split": "validation",
            "condition_id": condition,
            "scenario_seed_start": 8000,
            "scenario_seed_count": 20,
            "scenario_seeds": list(DYNAMIC_STRESS_SEEDS),
            "obstacle_count": request.obstacle_count,
            "obstacle_speed_range": [
                request.obstacle_speed_min,
                request.obstacle_speed_max,
            ],
            "horizontal_route_half_length": 1.1,
            "vertical_route_half_length": 1.1,
            "static_layout": "train_a",
            "safety_shield_enabled": True,
            "safety_shield_config": asdict(request.shield_profile.config),
            "reset_health": {
                "attempts": 20,
                "successful": 20,
                "finite_settle_checks": 40,
                "readback_checks": 20,
            },
            "episodes_csv_sha256": episodes_sha256,
        },
    }
    summary_path = result_directory / "summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    request_record = {
        "condition": condition,
        "seed": 11,
        "steps": [81920],
        "seed_start": 8000,
        "episodes": 20,
        "scenario_seeds": list(DYNAMIC_STRESS_SEEDS),
        "model_variant": "r2b_rebaseline_v1",
        "exact_model_path": RB2B_ENDPOINT_PATH.as_posix(),
        "exact_model_sha256": RB2B_ENDPOINT_SHA256,
        "obstacle_count": request.obstacle_count,
        "obstacle_speed_range": [
            request.obstacle_speed_min,
            request.obstacle_speed_max,
        ],
        "horizontal_route_half_length": 1.1,
        "vertical_route_half_length": 1.1,
        "static_layout": "train_a",
        "action_repeat": 4,
        "max_episode_steps": 300,
        "port_base": request.port_base,
        "workers": 1,
        "result_kind": "final",
        "safety_shield_enabled": True,
        "safety_shield_profile": request.shield_profile.manifest_record(root),
    }
    manifest = {
        "schema_version": 1,
        "kind": "webots_checkpoint_validation_sweep",
        "status": "completed",
        "split": "validation",
        "request": request_record,
        "runtime": {
            "evaluation_world": "worlds/webots_ppo_evaluate.wbt",
            "evaluation_world_sha256": hashlib.sha256(
                (root / "worlds/webots_ppo_evaluate.wbt").read_bytes()
            ).hexdigest(),
            "evaluator_controller": (
                "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
            ),
            "evaluator_controller_sha256": hashlib.sha256(
                (
                    root
                    / "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
                ).read_bytes()
            ).hexdigest(),
            "webots_executable": str(root / "webots"),
        },
        "runs": [
            {
                "status": "completed",
                "exit_code": 0,
                "step": 81920,
                "port": request.port_base,
                "run_name": run_name,
                "planner": planner,
                "model_path": RB2B_ENDPOINT_PATH.as_posix(),
                "model_sha256": RB2B_ENDPOINT_SHA256,
                "summary_path": summary_path.relative_to(root).as_posix(),
                "summary_sha256": summary_sha256,
                "environment": {
                    "WEBOTS_EVAL_SPLIT": "validation",
                    "WEBOTS_EVAL_CONDITION": condition,
                    "WEBOTS_EVAL_SEED_START": "8000",
                    "WEBOTS_EVAL_EPISODES": "20",
                    "WEBOTS_EVAL_MODEL": RB2B_ENDPOINT_PATH.as_posix(),
                    "WEBOTS_EVAL_MODEL_SHA256": RB2B_ENDPOINT_SHA256,
                    "WEBOTS_EVAL_PLANNER_LABEL": planner,
                    "WEBOTS_SAFETY_SHIELD": "true",
                },
            }
        ],
    }
    output_directory.mkdir(parents=True)
    (output_directory / "checkpoint_sweep.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    (output_directory / "checkpoint_sweep.yaml").write_text(
        yaml.safe_dump(manifest),
        encoding="utf-8",
    )
    return manifest


def test_all_conditions_run_even_when_first_gate_fails_and_evidence_is_bound(
    tmp_path: Path,
) -> None:
    root = _fixture_repository(tmp_path)
    config = load_dynamic_stress_config(CONFIG_PATH, repository_root=root)
    calls: list[str] = []
    successes = {"BASE1": 14, "DENS3": 16, "FAST3": 15}

    def sweep_runner(
        request: Any,
        *,
        output_directory: Path,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        calls.append(request.condition)
        return _write_fake_sweep(
            root,
            request,
            output_directory,
            success_count=successes[request.condition],
        )

    report = run_dynamic_stress(
        config,
        webots_executable=root / "webots",
        sweep_runner=sweep_runner,
    )

    assert calls == ["BASE1", "DENS3", "FAST3"]
    assert [item["gate"]["passed"] for item in report["conditions"]] == [
        False,
        True,
        True,
    ]
    assert report["decision"] == {
        "passed": False,
        "outcome": "record_observed_development_stress_boundary_without_tuning_or_retry",
        "formal_model_promotion": "forbidden",
    }
    assert report["test_split_was_run"] is False
    assert report["restrictions"]["adaptive_retries"] == "forbidden"
    assert report["source_git_revision"] == _git(root, "rev-parse", "HEAD")
    for item in report["conditions"]:
        assert item["model_sha256"] == RB2B_ENDPOINT_SHA256
        assert len(item["shield_profile"]["resolved_config"]) == 15
        assert len(item["episodes_csv"]["sha256"]) == 64
        assert len(item["summary"]["sha256"]) == 64
    assert (config.output_directory / "dynamic_stress_report.json").is_file()
    assert (config.output_directory / "dynamic_stress_report.yaml").is_file()
    assert (config.output_directory / "dynamic_stress_report.sha256").is_file()
    on_disk = json.loads(
        (config.output_directory / "dynamic_stress_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert on_disk == report


def test_finalize_existing_is_read_only_and_preserves_all_nine_evidence_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture_repository(tmp_path)
    config = load_dynamic_stress_config(CONFIG_PATH, repository_root=root)
    requests = build_dynamic_stress_requests(
        config,
        webots_executable=root / "webots",
    )
    manifests: dict[str, Mapping[str, Any]] = {}
    for request, condition, success_count in zip(
        requests,
        config.conditions,
        (14, 8, 0),
        strict=True,
    ):
        manifests[condition.condition_id] = _write_fake_sweep(
            root,
            request,
            condition.sweep_directory,
            success_count=success_count,
        )

    fast_manifest = json.loads(json.dumps(manifests["FAST3"]))
    fast_manifest["status"] = "failed"
    fast_run = fast_manifest["runs"][0]
    fast_run["status"] = "failed"
    fast_run["error"] = (
        "summary.by_planner[0].path_length_ratio_mean must be a number."
    )
    fast_run.pop("summary_sha256")
    fast_directory = config.conditions[2].sweep_directory
    (fast_directory / "checkpoint_sweep.json").write_text(
        json.dumps(fast_manifest) + "\n",
        encoding="utf-8",
    )
    (fast_directory / "checkpoint_sweep.yaml").write_text(
        yaml.safe_dump(fast_manifest),
        encoding="utf-8",
    )

    nine_paths: list[Path] = []
    sweep_yaml_paths: list[Path] = []
    for condition in config.conditions:
        result = root / "artifacts/results" / condition.run_name
        nine_paths.extend(
            (
                condition.sweep_directory / "checkpoint_sweep.json",
                result / "summary.json",
                result / "episodes.csv",
            )
        )
        sweep_yaml_paths.append(condition.sweep_directory / "checkpoint_sweep.yaml")
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in nine_paths}
    yaml_before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sweep_yaml_paths
    }

    monkeypatch.setattr(
        dynamic_stress,
        "DYNAMIC_STRESS_EXECUTION_REVISION",
        config.source_git_revision,
    )

    def forbidden_sweep(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        raise AssertionError("Finalization must never run a sweep or Webots process.")

    monkeypatch.setattr(dynamic_stress, "run_checkpoint_sweep", forbidden_sweep)
    report = finalize_existing_dynamic_stress(config)

    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in nine_paths}
    yaml_after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sweep_yaml_paths
    }
    assert before == after
    assert yaml_before == yaml_after
    assert report["webots_processes_launched"] == 0
    assert report["posthoc_revalidation"] is True
    assert report["no_rerun"] is True
    assert report["original_sweep_manifests_rewritten"] is False
    assert report["preregistration_revision"] == config.source_git_revision
    assert report["execution_revision"] == config.source_git_revision
    assert report["finalization_revision"] == config.source_git_revision
    assert report["protocol_git_blob"]["byte_identical"] is True
    assert report["software_erratum"] == {
        "code": "zero_success_nullable_metrics_rejected_v1",
        "affected_condition": "FAST3",
        "original_error": (
            "summary.by_planner[0].path_length_ratio_mean must be a number."
        ),
        "disposition": "read_only_revalidation_no_rerun",
        "execution_outputs_complete": True,
    }
    assert report["evidence_artifact_integrity"]["artifact_count"] == 9
    assert report["evidence_artifact_integrity"]["all_unchanged"] is True
    assert [item["original_manifest_status"] for item in report["conditions"]] == [
        "completed",
        "completed",
        "failed",
    ]
    assert [item["original_run_status"] for item in report["conditions"]] == [
        "completed",
        "completed",
        "failed",
    ]
    assert report["conditions"][2]["validator_metrics"][
        "path_length_ratio_mean"
    ] is None
    assert report["conditions"][2]["validator_metrics"][
        "completion_time_s_mean"
    ] is None
    assert not (root / ".finalization-never-launches-webots").exists()
    on_disk = json.loads(
        (config.output_directory / "dynamic_stress_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert on_disk == report


def test_finalize_cli_does_not_resolve_or_launch_webots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_config = object()
    report = {
        "status": "completed",
        "conditions": [],
        "decision": {"passed": False, "outcome": "bounded_failure"},
    }
    monkeypatch.setattr(
        dynamic_stress_cli,
        "load_dynamic_stress_config",
        lambda *_args, **_kwargs: sentinel_config,
    )
    monkeypatch.setattr(
        dynamic_stress_cli,
        "finalize_existing_dynamic_stress",
        lambda config: report if config is sentinel_config else None,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Finalize CLI must not resolve Webots or execute the matrix.")

    monkeypatch.setattr(dynamic_stress_cli, "resolve_webots_executable", forbidden)
    monkeypatch.setattr(dynamic_stress_cli, "run_dynamic_stress", forbidden)

    assert dynamic_stress_cli.main(["--finalize-existing"]) == 1
