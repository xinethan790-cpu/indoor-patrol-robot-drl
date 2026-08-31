from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from indoor_patrol_rl.core import (
    SAFETY_SHIELD_CONFIG_ENV_FIELDS,
    SafetyShieldConfig,
    safety_shield_config_from_env,
)
from indoor_patrol_rl.evaluation.checkpoint_selection import sha256_file
from indoor_patrol_rl.evaluation.checkpoint_sweep import (
    WebotsCheckpointSweepRequest,
    WebotsSafetyShieldProfile,
    load_safety_shield_profile,
    run_checkpoint_sweep,
    sweep_exit_code,
)
from scripts.run_webots_checkpoint_sweep import _default_output_directory

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]


def _request(
    root: Path,
    *,
    steps: tuple[int, ...] = (10, 20),
    workers: int = 2,
    shield_profile: WebotsSafetyShieldProfile | None = None,
) -> WebotsCheckpointSweepRequest:
    world = root / "worlds/webots_ppo_evaluate.wbt"
    world.parent.mkdir(parents=True, exist_ok=True)
    world.write_text("#VRML_SIM R2025a utf8\n", encoding="utf-8")
    controller = root / "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
    controller.parent.mkdir(parents=True, exist_ok=True)
    controller.write_text("# evaluator fixture\n", encoding="utf-8")
    executable = root / "webots"
    executable.write_bytes(b"fake webots\n")
    checkpoint_directory = (
        root
        / "artifacts/models/webots/seed_11/r2a_targeted_v1/checkpoints"
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (checkpoint_directory / f"ppo_webots_s11_r2a_targeted_v1_{step}_steps.zip").write_bytes(
            f"model-{step}".encode()
        )
    return WebotsCheckpointSweepRequest(
        repository_root=root,
        webots_executable=executable,
        seed=11,
        condition="r2a",
        model_variant="r2a_targeted_v1",
        steps=steps,
        episodes=2,
        seed_start=5000,
        obstacle_count=1,
        obstacle_speed_min=0.10,
        obstacle_speed_max=0.20,
        horizontal_route_half_length=1.10,
        vertical_route_half_length=1.10,
        port_base=14000,
        workers=workers,
        timeout_seconds=30.0,
        shield_profile=shield_profile,
    )


def _profile(
    root: Path,
    *,
    profile_id: str = "front-brake-v1",
    enabled: bool = True,
    filename: str = "front_brake_v1.yaml",
    changes: Mapping[str, float] | None = None,
) -> WebotsSafetyShieldProfile:
    config = asdict(SafetyShieldConfig())
    config.pop("lidar_max_range")
    config.update(changes or {})
    path = root / "configs/eval" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile_id": profile_id,
                "enabled": enabled,
                "config": config,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return load_safety_shield_profile(path, repository_root=root)


def _write_summary(
    root: Path,
    environment: Mapping[str, str],
    *,
    speed_range: tuple[float, float] | None = None,
) -> Path:
    output = root / "artifacts/results" / environment["WEBOTS_EVAL_RUN_NAME"]
    output.mkdir(parents=True, exist_ok=True)
    episodes_path = output / "episodes.csv"
    episodes_path.write_text("split,scenario_seed\nvalidation,5000\nvalidation,5001\n")
    episodes = int(environment["WEBOTS_EVAL_EPISODES"])
    seed_start = int(environment["WEBOTS_EVAL_SEED_START"])
    planner = environment["WEBOTS_EVAL_PLANNER_LABEL"]
    condition = environment["WEBOTS_EVAL_CONDITION"]
    model_path = root / environment["WEBOTS_EVAL_MODEL"]
    speeds = speed_range or (
        float(environment["WEBOTS_OBSTACLE_SPEED_MIN"]),
        float(environment["WEBOTS_OBSTACLE_SPEED_MAX"]),
    )
    aggregate = {
        "planner": planner,
        "episodes": episodes,
        "success_count": episodes,
        "collision_count": 0,
        "timeout_count": 0,
        "error_count": 0,
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "path_length_ratio_mean": 1.1,
        "completion_time_s_mean": 4.0,
    }
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "model_path": environment["WEBOTS_EVAL_MODEL"],
        "model_sha256": sha256_file(model_path),
        "model_variant": environment.get("WEBOTS_MODEL_VARIANT"),
        "training_seed": int(environment["WEBOTS_MODEL_SEED"]),
        "split": "validation",
        "condition_id": condition,
        "scenario_seed_start": seed_start,
        "scenario_seed_count": episodes,
        "scenario_seeds": list(range(seed_start, seed_start + episodes)),
        "obstacle_count": int(environment["WEBOTS_OBSTACLE_COUNT"]),
        "obstacle_speed_range": list(speeds),
        "horizontal_route_half_length": float(
            environment["WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH"]
        ),
        "vertical_route_half_length": float(
            environment["WEBOTS_VERTICAL_ROUTE_HALF_LENGTH"]
        ),
        "safety_shield_enabled": environment.get(
            "WEBOTS_SAFETY_SHIELD", "false"
        ).casefold()
        in {"1", "true", "yes", "on"},
        "static_layout": "train_a",
        "episodes_csv_sha256": sha256_file(episodes_path),
    }
    if any(name in environment for name in SAFETY_SHIELD_CONFIG_ENV_FIELDS):
        provenance["safety_shield_config"] = asdict(
            safety_shield_config_from_env(environment)
        )
    summary = {
        "split": "validation",
        "episode_count": episodes,
        "planners": [planner],
        "conditions": [condition],
        "by_planner": [aggregate],
        "by_planner_condition": [{**aggregate, "condition_id": condition}],
        "evaluation_provenance": provenance,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return summary_path


def _sequential_port(preferred: int, reserved: set[int]) -> int:
    port = preferred
    while port in reserved:
        port += 1
    reserved.add(port)
    return port


def test_sweep_runs_checkpoints_with_clean_environment_and_unique_ports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    assert request.run_name(10) == "webots_validation_r2a_checkpoint10_seed11"
    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("WEBOTS_STALE_PARENT_VALUE", "must-not-leak")

    def runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert "WEBOTS_STALE_PARENT_VALUE" not in environment
        assert environment["WEBOTS_EVAL_SPLIT"] == "validation"
        assert environment["WEBOTS_EVAL_MODEL_SHA256"] == sha256_file(
            tmp_path / environment["WEBOTS_EVAL_MODEL"]
        )
        assert environment["WEBOTS_SAFETY_SHIELD"] == "false"
        assert not any(
            name.startswith("WEBOTS_SAFETY_SHIELD_") for name in environment
        )
        calls.append({"command": tuple(command), "environment": dict(environment)})
        _write_summary(tmp_path, environment)
        return subprocess.CompletedProcess(command, 0, "webots ok\n", "")

    manifest = run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/sweep",
        process_runner=runner,
        port_allocator=_sequential_port,
    )

    assert manifest["status"] == "completed"
    assert sweep_exit_code(manifest) == 0
    assert [run["status"] for run in manifest["runs"]] == ["completed", "completed"]
    ports = {call["command"][1] for call in calls}
    assert len(ports) == 2
    on_disk = json.loads(
        (tmp_path / "artifacts/results/sweep/checkpoint_sweep.json").read_text()
    )
    assert on_disk["manifest_paths"] == manifest["manifest_paths"]
    assert all(run["model_sha256"] for run in on_disk["runs"])
    assert "safety_shield_profile" not in on_disk["request"]


def test_sweep_reuses_only_a_strict_scenario_match(tmp_path: Path) -> None:
    request = _request(tmp_path)
    environments = {
        step: {
            "WEBOTS_MODEL_SEED": "11",
            "WEBOTS_MODEL_VARIANT": "r2a_targeted_v1",
            "WEBOTS_EVAL_MODEL": (
                "artifacts/models/webots/seed_11/r2a_targeted_v1/checkpoints/"
                f"ppo_webots_s11_r2a_targeted_v1_{step}_steps.zip"
            ),
            "WEBOTS_EVAL_EPISODES": "2",
            "WEBOTS_EVAL_SEED_START": "5000",
            "WEBOTS_EVAL_SPLIT": "validation",
            "WEBOTS_EVAL_CONDITION": "R2A",
            "WEBOTS_EVAL_PLANNER_LABEL": f"ppo_r2a_step{step}",
            "WEBOTS_EVAL_RUN_NAME": f"webots_validation_r2a_checkpoint{step}_seed11",
            "WEBOTS_OBSTACLE_COUNT": "1",
            "WEBOTS_OBSTACLE_SPEED_MIN": "0.1",
            "WEBOTS_OBSTACLE_SPEED_MAX": "0.2",
            "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": "1.1",
            "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": "1.1",
        }
        for step in request.steps
    }
    _write_summary(tmp_path, environments[10])
    _write_summary(tmp_path, environments[20], speed_range=(0.4, 0.5))
    calls: list[int] = []

    def runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        calls.append(int(environment["WEBOTS_EVAL_PLANNER_LABEL"].split("step")[-1]))
        _write_summary(tmp_path, environment)
        return subprocess.CompletedProcess(command, 0, "", "")

    manifest = run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/sweep",
        process_runner=runner,
        port_allocator=_sequential_port,
    )

    assert calls == [20]
    assert [run["status"] for run in manifest["runs"]] == ["reused", "completed"]
    assert "speed range" in manifest["runs"][1]["prior_summary_rejection"]


def test_sweep_records_process_failure_and_returns_nonzero(tmp_path: Path) -> None:
    request = _request(tmp_path, steps=(10,), workers=1)

    def runner(
        command: Sequence[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, "before failure", "failed")

    manifest = run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/sweep",
        process_runner=runner,
        port_allocator=_sequential_port,
    )

    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["exit_code"] == 9
    assert sweep_exit_code(manifest) == 2
    assert (tmp_path / "artifacts/results/sweep/checkpoint10.stderr.log").read_text() == (
        "failed"
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"condition": "R2/A"}, "condition"),
        ({"model_variant": "../escape"}, "model_variant"),
        ({"steps": (10, 10)}, "duplicates"),
        ({"obstacle_speed_min": 0.3, "obstacle_speed_max": 0.2}, "ordered"),
    ],
)
def test_sweep_request_rejects_ambiguous_identity(
    tmp_path: Path,
    changes: Mapping[str, Any],
    message: str,
) -> None:
    values: dict[str, Any] = {
        "repository_root": tmp_path,
        "webots_executable": tmp_path / "webots",
        "seed": 11,
        "condition": "R2A",
        "model_variant": "r2a_targeted_v1",
        "steps": (10,),
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        WebotsCheckpointSweepRequest(**values)


def test_repository_front_brake_profile_is_complete_and_source_bound() -> None:
    path = REPOSITORY_ROOT / "configs/eval/webots_r2a_front_brake_v1.yaml"

    profile = load_safety_shield_profile(path, repository_root=REPOSITORY_ROOT)

    assert profile.profile_id == "r2a-front-brake-v1"
    assert profile.enabled is True
    assert profile.source_sha256 == sha256_file(path)
    assert len(profile.config_values) == len(SAFETY_SHIELD_CONFIG_ENV_FIELDS) == 14
    assert profile.config.clearance_quantile == pytest.approx(0.0)
    assert profile.config.front_stop_distance == pytest.approx(0.24)
    assert profile.config.front_slowdown_distance == pytest.approx(0.75)
    assert profile.config.front_half_angle == pytest.approx(1.0471975511965976)
    assert profile.config.side_inner_angle == pytest.approx(1.0471975511965976)
    assert profile.config.side_speed_floor == pytest.approx(1.0)
    assert profile.config.turn_blend == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["config"].pop("turn_gain"), "missing"),
        (
            lambda raw: raw["config"].__setitem__("lidar_max_range", 8.0),
            "extra",
        ),
        (lambda raw: raw["config"].__setitem__("turn_gain", float("nan")), "finite"),
        (
            lambda raw: raw["config"].__setitem__("front_stop_distance", 0.80),
            "front_stop_distance",
        ),
    ],
)
def test_shield_profile_rejects_incomplete_hardware_nonfinite_and_invalid_values(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    profile = _profile(tmp_path)
    raw = yaml.safe_load(profile.source_path.read_text(encoding="utf-8"))
    mutation(raw)
    profile.source_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_safety_shield_profile(profile.source_path, repository_root=tmp_path)


def test_shield_profile_source_must_stay_bound_to_the_request(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile.source_path.write_text(
        profile.source_path.read_text(encoding="utf-8") + "# drift\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed after it was loaded"):
        _request(tmp_path, shield_profile=profile)


def test_shield_profile_path_must_stay_inside_repository(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    repository_root = tmp_path / "nested-repository"
    repository_root.mkdir()

    with pytest.raises(ValueError, match="stay inside the repository"):
        load_safety_shield_profile(
            profile.source_path,
            repository_root=repository_root,
        )


def test_direct_profile_construction_cannot_bypass_identity_validation(
    tmp_path: Path,
) -> None:
    values: dict[str, Any] = {
        "source_path": tmp_path / "profile.yaml",
        "source_sha256": "a" * 64,
        "profile_id": "valid-profile",
        "enabled": True,
        "config": SafetyShieldConfig(),
    }
    with pytest.raises(ValueError, match="profile_id"):
        WebotsSafetyShieldProfile(**{**values, "profile_id": "../escape"})
    with pytest.raises(ValueError, match="enabled"):
        WebotsSafetyShieldProfile(**{**values, "enabled": 1})
    with pytest.raises(ValueError, match="source_sha256"):
        WebotsSafetyShieldProfile(**{**values, "source_sha256": "not-a-digest"})


def test_explicit_shield_profile_runs_with_isolated_identity_and_manifest(
    tmp_path: Path,
) -> None:
    profile = _profile(
        tmp_path,
        changes={
            "clearance_quantile": 0.0,
            "front_stop_distance": 0.24,
            "front_slowdown_distance": 0.75,
            "front_half_angle": 1.0471975511965976,
            "side_inner_angle": 1.0471975511965976,
            "side_speed_floor": 1.0,
            "turn_blend": 0.0,
        },
    )
    request = _request(
        tmp_path,
        steps=(10,),
        workers=1,
        shield_profile=profile,
    )
    calls: list[dict[str, str]] = []

    def runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        calls.append(dict(environment))
        assert environment["WEBOTS_SAFETY_SHIELD"] == "true"
        assert {
            name for name in environment if name.startswith("WEBOTS_SAFETY_SHIELD_")
        } == set(SAFETY_SHIELD_CONFIG_ENV_FIELDS)
        _write_summary(tmp_path, environment)
        return subprocess.CompletedProcess(command, 0, "", "")

    manifest = run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/profile-sweep",
        process_runner=runner,
        port_allocator=_sequential_port,
    )

    assert len(calls) == 1
    assert request.run_name(10).endswith(f"_shield-{profile.result_token}")
    assert manifest["runs"][0]["run_name"] == request.run_name(10)
    recorded = manifest["request"]["safety_shield_profile"]
    assert manifest["request"]["safety_shield_enabled"] is True
    assert recorded["profile_id"] == profile.profile_id
    assert recorded["result_token"] == profile.result_token
    assert recorded["semantic_sha256"] == profile.semantic_sha256
    assert recorded["source_sha256"] == profile.source_sha256
    assert recorded["source_path"] == "configs/eval/front_brake_v1.yaml"
    assert recorded["config"] == profile.config_values
    assert recorded["lidar_max_range_source"] == "LDS-01.getMaxRange()"

    def unexpected_runner(
        _command: Sequence[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("A matching profile summary should be reused.")

    reused = run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/profile-sweep-reused",
        process_runner=unexpected_runner,
        port_allocator=_sequential_port,
    )
    assert reused["runs"][0]["status"] == "reused"


def test_default_sweep_output_directory_uses_the_profile_identity(tmp_path: Path) -> None:
    profile = _profile(tmp_path)

    legacy = _default_output_directory(
        tmp_path,
        condition="R2A",
        seed=11,
        model_variant="r2a_targeted_v1",
        shield_profile=None,
    )
    configured = _default_output_directory(
        tmp_path,
        condition="R2A",
        seed=11,
        model_variant="r2a_targeted_v1",
        shield_profile=profile,
    )

    assert legacy.name == (
        "webots_validation_r2a_checkpoint_sweep_seed11_r2a_targeted_v1"
    )
    assert configured.name == f"{legacy.name}_shield-{profile.result_token}"


def test_profile_semantics_prevent_same_id_result_reuse(tmp_path: Path) -> None:
    first = _profile(tmp_path, filename="first.yaml")
    second = _profile(
        tmp_path,
        filename="second.yaml",
        changes={"front_stop_distance": 0.25},
    )

    assert first.profile_id == second.profile_id
    assert first.semantic_sha256 != second.semantic_sha256
    assert _request(tmp_path, shield_profile=first).run_name(10) != _request(
        tmp_path,
        shield_profile=second,
    ).run_name(10)


def test_profile_summary_with_wrong_config_is_not_reused(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    request = _request(
        tmp_path,
        steps=(10,),
        workers=1,
        shield_profile=profile,
    )
    calls: list[int] = []

    def runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(1)
        _write_summary(tmp_path, kwargs["env"])
        return subprocess.CompletedProcess(command, 0, "", "")

    run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/first-sweep",
        process_runner=runner,
        port_allocator=_sequential_port,
    )
    summary_path = (
        tmp_path / "artifacts/results" / request.run_name(10) / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evaluation_provenance"]["safety_shield_config"]["turn_gain"] = 0.1
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    manifest = run_checkpoint_sweep(
        request,
        output_directory=tmp_path / "artifacts/results/second-sweep",
        process_runner=runner,
        port_allocator=_sequential_port,
    )

    assert calls == [1, 1]
    assert manifest["runs"][0]["status"] == "completed"
    assert "turn_gain" in manifest["runs"][0]["prior_summary_rejection"]
