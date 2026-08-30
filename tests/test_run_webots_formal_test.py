from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from indoor_patrol_rl.evaluation.formal_protocol import (
    FinalLock,
    FormalModelManifest,
    FormalSourceArtifact,
    file_sha256,
    protected_test_consumption_key,
    protected_test_ledger_path,
    source_bundle_sha256,
)
from indoor_patrol_rl.evaluation.formal_reporting import FormalResultInput
from indoor_patrol_rl.evaluation.webots_formal_config import (
    load_webots_formal_config,
)
from scripts.run_webots_formal_test import execute_formal_test


def _formal_fixture(root: Path) -> tuple[FinalLock, Path]:
    project = Path(__file__).absolute().parents[1]
    config = root / "configs/eval/webots_formal.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        (project / "configs/eval/webots_formal.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    world = root / "worlds/webots_ppo_evaluate.wbt"
    world.parent.mkdir(parents=True)
    world.write_text("#VRML_SIM R2025a utf8\n", encoding="utf-8")

    evaluator_path = "scripts/run_webots_formal_test.py"
    controller_path = "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
    source_bundle = (
        FormalSourceArtifact(path=controller_path, sha256="3" * 64),
        FormalSourceArtifact(path=evaluator_path, sha256="4" * 64),
    )

    manifests: list[FormalModelManifest] = []
    for seed in (11, 22, 33):
        model = root / f"artifacts/models/webots/seed_{seed}/formal.zip"
        model.parent.mkdir(parents=True)
        model.write_bytes(f"formal-model-{seed}".encode())
        manifests.append(
            FormalModelManifest(
                training_seed=seed,
                model_path=model.relative_to(root).as_posix(),
                model_sha256=file_sha256(model),
                training_metadata_path=(f"artifacts/models/webots/seed_{seed}/metadata.yaml"),
                training_metadata_sha256="0" * 64,
                model_provenance_path=(
                    f"artifacts/models/webots/seed_{seed}/formal.provenance.json"
                ),
                model_provenance_sha256="5" * 64,
                curriculum_manifest_path=(
                    f"artifacts/results/curriculum_seed{seed}/run_manifest.json"
                ),
                curriculum_manifest_sha256="6" * 64,
                selection_manifest_path=(f"artifacts/results/selection_seed{seed}/selection.json"),
                selection_manifest_sha256="1" * 64,
                validation_episodes_path=(f"artifacts/results/selection_seed{seed}/episodes.csv"),
                validation_episodes_sha256="2" * 64,
                training_stage="W2",
                model_variant=None,
                training_protocol_sha256="7" * 64,
                algorithm_config_sha256="8" * 64,
            )
        )
    final_lock = FinalLock(
        campaign_id="test_campaign",
        source_revision="c" * 40,
        source_dirty=False,
        evaluation_config_path=config.relative_to(root).as_posix(),
        evaluation_config_sha256=file_sha256(config),
        world_path=world.relative_to(root).as_posix(),
        world_sha256=file_sha256(world),
        evaluator_path=evaluator_path,
        evaluator_sha256="4" * 64,
        controller_path=controller_path,
        controller_sha256="3" * 64,
        source_bundle=source_bundle,
        source_bundle_sha256=source_bundle_sha256(source_bundle),
        test_consumption_key=protected_test_consumption_key(),
        output_directory="artifacts/results/formal_runs/test_campaign",
        test_ledger_path=protected_test_ledger_path(),
        model_manifests=tuple(manifests),
    )
    lock_path = root / "artifacts/results/formal_locks/test_campaign.json"
    return final_lock, lock_path


def _webots_executable(root: Path) -> Path:
    executable = root / "fake-webots.exe"
    executable.write_bytes(b"fake Webots executable R2025a\n")
    return executable


def _version_probe(version: str = "R2025a", calls: list[tuple[str, ...]] | None = None):
    def run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] > 0
        if calls is not None:
            calls.append(tuple(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"Webots {version}\n",
            stderr="",
        )

    return run


def _windows_metadata(_executable: Path) -> Mapping[str, Any]:
    return {
        "windows_file_version": "2025.1.0.0",
        "windows_product_version": "R2025a",
        "signature_status": "NotSigned",
        "signature_status_message": "The file is not digitally signed.",
        "signer_subject": None,
        "signer_thumbprint": None,
    }


def _sequential_test_port(preferred: int, reserved: set[int]) -> int:
    port = preferred
    while port in reserved:
        port += 1
    reserved.add(port)
    return port


def _lock_loader(final_lock: FinalLock, calls: list[tuple[Path, bool]]):
    def load(path: Path, *, repository_root: Path, verify_files: bool) -> FinalLock:
        assert repository_root.is_absolute()
        calls.append((path, verify_files))
        return final_lock

    return load


def _simulator_runner(
    root: Path,
    calls: list[dict[str, Any]],
    *,
    fail_at: int | None = None,
):
    def run(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, Mapping)
        calls.append({"command": tuple(command), "environment": dict(environment)})
        call_number = len(calls)
        if call_number == fail_at:
            return subprocess.CompletedProcess(
                command,
                9,
                stdout="simulator stdout before failure\n",
                stderr="simulator failed\n",
            )
        staging = root / "artifacts/results" / environment["WEBOTS_EVAL_RUN_NAME"]
        staging.mkdir(parents=True)
        episodes_path = staging / "episodes.csv"
        episodes_path.write_text(
            "split,scenario_seed,planner,condition_id\n",
            encoding="utf-8",
        )
        config = load_webots_formal_config(root / environment["WEBOTS_FORMAL_CONFIG"])
        condition = config.condition(environment["WEBOTS_EVAL_CONDITION"])
        training_seed = environment.get("WEBOTS_MODEL_SEED")
        provenance = {
            "schema_version": 1,
            "split": "test",
            "condition_id": condition.condition_id,
            "planner": environment["WEBOTS_EVAL_PLANNER"],
            "planner_label": environment["WEBOTS_EVAL_PLANNER_LABEL"],
            "training_seed": int(training_seed) if training_seed is not None else None,
            "model_path": environment.get("WEBOTS_EVAL_MODEL"),
            "model_sha256": environment.get("WEBOTS_EVAL_MODEL_SHA256"),
            "scenario_seed_start": config.scenario_seed_start,
            "scenario_seed_count": config.scenario_seed_count,
            "scenario_seeds": list(config.scenario_seeds),
            "obstacle_count": condition.obstacle_count,
            "obstacle_speed_range": list(condition.obstacle_speed_range),
            "horizontal_route_half_length": condition.horizontal_route_half_length,
            "vertical_route_half_length": condition.vertical_route_half_length,
            "static_layout": condition.static_layout,
            "safety_shield_enabled": False,
            "experiment_id": config.experiment_id,
            "webots_version": config.webots_version,
            "formal_config_path": environment["WEBOTS_FORMAL_CONFIG"],
            "formal_config_sha256": config.source_sha256,
            "world_path": config.world,
            "world_sha256": file_sha256(root / config.world),
            "condition_semantic_sha256": condition.semantic_sha256,
            "episodes_csv_sha256": file_sha256(episodes_path),
        }
        (staging / "summary.json").write_text(
            json.dumps(
                {
                    "split": "test",
                    "episode_count": 20,
                    "planners": [environment["WEBOTS_EVAL_PLANNER_LABEL"]],
                    "conditions": [environment["WEBOTS_EVAL_CONDITION"]],
                    "evaluation_provenance": provenance,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="simulator stdout\n",
            stderr="",
        )

    return run


def _report_builder(
    calls: list[tuple[FormalResultInput, ...]],
):
    def build(
        inputs: Sequence[FormalResultInput],
        *,
        repository_root: Path,
        final_lock_path: Path,
    ) -> dict[str, Any]:
        assert repository_root.is_absolute()
        assert final_lock_path == (
            repository_root / "artifacts/results/formal_locks/test_campaign.json"
        )
        result_inputs = tuple(inputs)
        assert len(result_inputs) == 20
        assert all(item.episodes_path.is_file() for item in result_inputs)
        assert all(item.summary_path.is_file() for item in result_inputs)
        calls.append(result_inputs)
        return {
            "schema_version": "1.0",
            "matrix_validation": {"status": "passed", "episode_count": 400},
        }

    return build


def _report_writer(report: Mapping[str, Any], output_path: Path) -> Path:
    output_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return output_path


def test_dry_run_resolves_the_exact_twenty_run_matrix_without_consuming_test(
    tmp_path: Path,
) -> None:
    final_lock, lock_path = _formal_fixture(tmp_path)
    load_calls: list[tuple[Path, bool]] = []
    probe_calls: list[tuple[str, ...]] = []
    executable = _webots_executable(tmp_path)

    execution = execute_formal_test(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        webots_executable=executable,
        confirm_test=True,
        dry_run=True,
        port_base=20_000,
        executable_probe_runner=_version_probe(calls=probe_calls),
        windows_metadata_probe=_windows_metadata,
        lock_loader=_lock_loader(final_lock, load_calls),
        source_verifier=lambda _root: final_lock.source_revision,
        begin_attempt=lambda **_kwargs: pytest.fail("dry-run consumed the ledger"),
        process_runner=lambda *_args, **_kwargs: pytest.fail("dry-run started Webots"),
    )

    assert execution.dry_run is True
    assert probe_calls == [(str(executable.resolve()), "--version")]
    assert execution.executable_evidence is not None
    assert execution.executable_evidence.detected_version == "R2025a"
    assert execution.executable_evidence.executable_sha256 == file_sha256(executable)
    assert len(execution.runs) == 20
    assert load_calls == [(lock_path.resolve(), True), (lock_path.resolve(), True)]
    assert [(run.condition_id, run.planner_label) for run in execution.runs] == [
        (condition, planner)
        for condition in ("G0", "G1", "G2", "G3")
        for planner in ("reactive", "dwa", "ppo_seed11", "ppo_seed22", "ppo_seed33")
    ]
    assert [run.command[1] for run in execution.runs] == [
        f"--port={port}" for port in range(20_000, 20_020)
    ]
    assert all(run.environment["WEBOTS_CONFIRM_TEST"] == "1" for run in execution.runs)
    assert all(
        run.environment["WEBOTS_FORMAL_CONFIG"] == final_lock.evaluation_config_path
        for run in execution.runs
    )
    assert all(
        run.environment["WEBOTS_EVAL_SEED_START"] == "10000"
        and run.environment["WEBOTS_EVAL_EPISODES"] == "20"
        for run in execution.runs
    )
    assert "WEBOTS_EVAL_MODEL" not in execution.runs[0].environment
    ppo_run = execution.runs[2]
    assert ppo_run.environment["WEBOTS_MODEL_SEED"] == "11"
    assert ppo_run.environment["WEBOTS_EVAL_MODEL_SHA256"] == (
        final_lock.model_manifests[0].model_sha256
    )
    assert execution.runs[10].condition_id == "G2"
    assert execution.runs[10].environment["WEBOTS_OBSTACLE_COUNT"] == "5"
    assert execution.runs[15].condition_id == "G3"
    assert execution.runs[15].environment["WEBOTS_OBSTACLE_SPEED_MIN"] == "0.45"
    assert execution.runs[15].environment["WEBOTS_OBSTACLE_SPEED_MAX"] == "0.85"
    assert not (tmp_path / final_lock.output_directory).exists()
    assert not (tmp_path / final_lock.test_ledger_path).exists()


def test_explicit_confirmation_is_required_before_even_loading_the_lock(
    tmp_path: Path,
) -> None:
    _, lock_path = _formal_fixture(tmp_path)

    with pytest.raises(PermissionError, match="explicit --confirm-test"):
        execute_formal_test(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            webots_executable=Path("webots"),
            confirm_test=False,
            lock_loader=lambda *_args, **_kwargs: pytest.fail("lock was loaded"),
        )


def test_stale_staging_is_rejected_before_the_one_shot_ledger_is_consumed(
    tmp_path: Path,
) -> None:
    final_lock, lock_path = _formal_fixture(tmp_path)
    stale = tmp_path / "artifacts/results/formal_stage_test_campaign_G0_reactive"
    stale.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="staging paths"):
        execute_formal_test(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            webots_executable=Path("webots"),
            confirm_test=True,
            lock_loader=_lock_loader(final_lock, []),
            source_verifier=lambda _root: final_lock.source_revision,
            begin_attempt=lambda **_kwargs: pytest.fail("ledger was consumed"),
        )


def test_source_revision_drift_is_rejected_before_the_ledger_is_consumed(
    tmp_path: Path,
) -> None:
    final_lock, lock_path = _formal_fixture(tmp_path)

    with pytest.raises(ValueError, match="Current Git HEAD differs"):
        execute_formal_test(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            webots_executable=Path("webots"),
            confirm_test=True,
            lock_loader=_lock_loader(final_lock, []),
            source_verifier=lambda _root: "d" * 40,
            begin_attempt=lambda **_kwargs: pytest.fail("ledger was consumed"),
        )


@pytest.mark.parametrize("timeout_seconds", [-1.0, 0.0, float("inf"), float("nan"), None])
def test_invalid_timeout_is_rejected_before_the_ledger_is_consumed(
    tmp_path: Path, timeout_seconds: float | None
) -> None:
    _, lock_path = _formal_fixture(tmp_path)

    with pytest.raises(ValueError, match="finite"):
        execute_formal_test(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            webots_executable=Path("webots"),
            confirm_test=True,
            timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
            lock_loader=lambda *_args, **_kwargs: pytest.fail("lock was loaded"),
            begin_attempt=lambda **_kwargs: pytest.fail("ledger was consumed"),
        )


def test_dry_run_rejects_wrong_webots_version_without_consuming_the_ledger(
    tmp_path: Path,
) -> None:
    final_lock, lock_path = _formal_fixture(tmp_path)
    executable = _webots_executable(tmp_path)

    with pytest.raises(ValueError, match=r"R2024b.*locked R2025a"):
        execute_formal_test(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            webots_executable=executable,
            confirm_test=True,
            dry_run=True,
            executable_probe_runner=_version_probe("R2024b"),
            windows_metadata_probe=lambda _path: pytest.fail(
                "metadata was inspected for the wrong release"
            ),
            port_allocator=_sequential_test_port,
            lock_loader=_lock_loader(final_lock, []),
            source_verifier=lambda _root: final_lock.source_revision,
            begin_attempt=lambda **_kwargs: pytest.fail("ledger was consumed"),
        )

    assert not (tmp_path / final_lock.test_ledger_path).exists()
    assert not (tmp_path / final_lock.output_directory).exists()


def test_successful_execution_binds_report_manifest_and_ledger_to_the_lock(
    tmp_path: Path,
) -> None:
    final_lock, lock_path = _formal_fixture(tmp_path)
    simulator_calls: list[dict[str, Any]] = []
    report_calls: list[tuple[FormalResultInput, ...]] = []
    ledger_events: list[tuple[str, dict[str, Any]]] = []
    load_calls: list[tuple[Path, bool]] = []
    source_checks: list[Path] = []
    version_probe_calls: list[tuple[str, ...]] = []
    executable = _webots_executable(tmp_path)

    def begin(**kwargs: Any) -> Mapping[str, Any]:
        ledger_events.append(("begin", kwargs))
        return {"status": "attempt"}

    def mark(**kwargs: Any) -> Mapping[str, Any]:
        ledger_events.append(("success", kwargs))
        assert Path(kwargs["result_manifest_path"]).is_file()
        return {"status": "success"}

    def verify_source(repository_root: Path) -> str:
        source_checks.append(repository_root)
        return final_lock.source_revision

    execution = execute_formal_test(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        webots_executable=executable,
        confirm_test=True,
        process_runner=_simulator_runner(tmp_path, simulator_calls),
        executable_probe_runner=_version_probe(calls=version_probe_calls),
        windows_metadata_probe=_windows_metadata,
        port_allocator=lambda preferred, reserved: (
            reserved.add(preferred + len(reserved)) or preferred + len(reserved) - 1
        ),
        lock_loader=_lock_loader(final_lock, load_calls),
        source_verifier=verify_source,
        port_verifier=lambda _port: None,
        begin_attempt=begin,
        mark_success=mark,
        invalidate_attempt=lambda **_kwargs: pytest.fail("success was invalidated"),
        report_builder=_report_builder(report_calls),
        report_writer=_report_writer,
    )

    assert execution.dry_run is False
    assert len(simulator_calls) == 20
    assert version_probe_calls == [(str(executable.resolve()), "--version")]
    assert len(load_calls) == 44
    assert all(verify_files is True for _path, verify_files in load_calls)
    assert len(source_checks) == 44
    assert len(report_calls) == 1
    assert [event for event, _kwargs in ledger_events] == ["begin", "success"]
    assert ledger_events[0][1]["runtime_evidence"]["detected_version"] == "R2025a"
    assert ledger_events[0][1]["runtime_evidence"]["signature_status"] == "NotSigned"
    assert execution.ledger == {"status": "success"}
    assert execution.report_path is not None and execution.report_path.is_file()
    assert execution.result_manifest_path is not None
    manifest = json.loads(execution.result_manifest_path.read_text(encoding="utf-8"))
    assert manifest["campaign_id"] == final_lock.campaign_id
    assert manifest["final_lock_sha256"] == final_lock.lock_sha256
    assert manifest["source_revision"] == final_lock.source_revision
    assert manifest["evaluation_config_sha256"] == final_lock.evaluation_config_sha256
    assert manifest["world_sha256"] == final_lock.world_sha256
    assert manifest["source_bundle_sha256"] == final_lock.source_bundle_sha256
    assert manifest["test_consumption_key"] == final_lock.test_consumption_key
    assert manifest["training_seeds"] == [11, 22, 33]
    assert [item["model_sha256"] for item in manifest["models"]] == [
        item.model_sha256 for item in final_lock.model_manifests
    ]
    assert manifest["scenario_seeds"] == list(range(10_000, 10_020))
    assert manifest["run_count"] == 20
    assert manifest["episode_count"] == 400
    assert len(manifest["run_artifacts"]) == 20
    assert all(len(item["episodes_sha256"]) == 64 for item in manifest["run_artifacts"])
    executable_evidence_path = tmp_path / final_lock.output_directory / ("webots_executable.json")
    executable_evidence = json.loads(executable_evidence_path.read_text(encoding="utf-8"))
    assert executable_evidence["detected_version"] == "R2025a"
    assert executable_evidence["executable_sha256"] == file_sha256(executable)
    assert executable_evidence["windows_file_version"] == "2025.1.0.0"
    assert executable_evidence["signature_status"] == "NotSigned"
    assert manifest["webots_executable"] == executable_evidence
    assert manifest["webots_executable_evidence_sha256"] == file_sha256(executable_evidence_path)
    config = load_webots_formal_config(tmp_path / final_lock.evaluation_config_path)
    for call, run in zip(simulator_calls, execution.runs, strict=True):
        assert call["environment"]["WEBOTS_FINAL_LOCK_SHA256"] == final_lock.lock_sha256
        assert call["environment"]["WEBOTS_EVAL_CONFIG_SHA256"] == config.source_sha256
        assert call["environment"]["WEBOTS_EVAL_CONDITION"] == run.condition_id
        assert call["environment"]["WEBOTS_EVAL_PLANNER_LABEL"] == run.planner_label


def test_runtime_world_fingerprint_drift_is_detected_after_the_subprocess(
    tmp_path: Path,
) -> None:
    final_lock, lock_path = _formal_fixture(tmp_path)
    executable = _webots_executable(tmp_path)
    world = tmp_path / final_lock.world_path
    simulator_calls: list[dict[str, Any]] = []
    invalidations: list[dict[str, Any]] = []
    simulator = _simulator_runner(tmp_path, simulator_calls)

    def checking_loader(path: Path, *, repository_root: Path, verify_files: bool) -> FinalLock:
        assert path == lock_path.resolve()
        assert repository_root == tmp_path.resolve()
        assert verify_files is True
        if file_sha256(world) != final_lock.world_sha256:
            raise RuntimeError("formal world fingerprint drifted")
        return final_lock

    def mutate_world(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        completed = simulator(command, **kwargs)
        world.write_text("#VRML_SIM R2025a utf8\n# runtime drift\n", encoding="utf-8")
        return completed

    with pytest.raises(RuntimeError, match="formal world fingerprint drifted"):
        execute_formal_test(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            webots_executable=executable,
            confirm_test=True,
            process_runner=mutate_world,
            executable_probe_runner=_version_probe(),
            windows_metadata_probe=_windows_metadata,
            port_allocator=_sequential_test_port,
            port_verifier=lambda _port: None,
            lock_loader=checking_loader,
            source_verifier=lambda _root: final_lock.source_revision,
            begin_attempt=lambda **_kwargs: {"status": "attempt"},
            mark_success=lambda **_kwargs: pytest.fail("drift was marked successful"),
            invalidate_attempt=lambda **kwargs: (
                invalidations.append(kwargs) or {"status": "invalidated"}
            ),
            report_builder=lambda *_args, **_kwargs: pytest.fail("report was built"),
        )

    assert len(simulator_calls) == 1
    assert len(invalidations) == 1
    assert "formal world fingerprint drifted" in invalidations[0]["reason"]
    integrity_error = (
        tmp_path
        / final_lock.output_directory
        / ("logs/G0/reactive/post_process_integrity_error.json")
    )
    assert integrity_error.is_file()


def test_port_taken_after_planning_is_replaced_immediately_before_launch(
    tmp_path: Path,
) -> None:
    final_lock, lock_path = _formal_fixture(tmp_path)
    executable = _webots_executable(tmp_path)
    simulator_calls: list[dict[str, Any]] = []
    report_calls: list[tuple[FormalResultInput, ...]] = []
    held_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        held_socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    port_base = 65_535
    while port_base > 65_516:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as discovery:
            discovery.bind(("127.0.0.1", 0))
            port_base = discovery.getsockname()[1]

    def occupy_planned_port(
        command: Sequence[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        held_socket.bind(("127.0.0.1", port_base))
        held_socket.listen(1)
        return subprocess.CompletedProcess(command, 0, stdout="Webots R2025a\n", stderr="")

    try:
        execution = execute_formal_test(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            webots_executable=executable,
            confirm_test=True,
            port_base=port_base,
            process_runner=_simulator_runner(tmp_path, simulator_calls),
            executable_probe_runner=occupy_planned_port,
            windows_metadata_probe=_windows_metadata,
            lock_loader=_lock_loader(final_lock, []),
            source_verifier=lambda _root: final_lock.source_revision,
            begin_attempt=lambda **_kwargs: {"status": "attempt"},
            mark_success=lambda **_kwargs: {"status": "success"},
            invalidate_attempt=lambda **_kwargs: pytest.fail(
                "a newly selected port was invalidated"
            ),
            report_builder=_report_builder(report_calls),
            report_writer=_report_writer,
        )
    finally:
        held_socket.close()

    assert execution.ledger == {"status": "success"}
    assert len(simulator_calls) == 20
    assert execution.runs[0].port != port_base
    assert simulator_calls[0]["command"][1] == f"--port={execution.runs[0].port}"
    assert simulator_calls[0]["environment"]["WEBOTS_PORT"] == str(execution.runs[0].port)


def test_any_post_attempt_simulator_failure_invalidates_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    final_lock, lock_path = _formal_fixture(tmp_path)
    simulator_calls: list[dict[str, Any]] = []
    invalidations: list[dict[str, Any]] = []

    with pytest.raises(RuntimeError, match="exit code 9"):
        execute_formal_test(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            webots_executable=_webots_executable(tmp_path),
            confirm_test=True,
            process_runner=_simulator_runner(tmp_path, simulator_calls, fail_at=2),
            executable_probe_runner=_version_probe(),
            windows_metadata_probe=_windows_metadata,
            port_allocator=lambda preferred, reserved: (
                reserved.add(preferred + len(reserved)) or preferred + len(reserved) - 1
            ),
            lock_loader=_lock_loader(final_lock, []),
            source_verifier=lambda _root: final_lock.source_revision,
            port_verifier=lambda _port: None,
            begin_attempt=lambda **_kwargs: {"status": "attempt"},
            mark_success=lambda **_kwargs: pytest.fail("failed run was marked successful"),
            invalidate_attempt=lambda **kwargs: (
                invalidations.append(kwargs) or {"status": "invalidated"}
            ),
            report_builder=lambda *_args, **_kwargs: pytest.fail("report was built"),
        )

    assert len(simulator_calls) == 2
    assert len(invalidations) == 1
    assert "exit code 9" in invalidations[0]["reason"]
    output_root = tmp_path / final_lock.output_directory
    assert (output_root / "runs/G0/reactive/episodes.csv").is_file()
    assert (output_root / "logs/G0/reactive/stdout.log").is_file()
    assert (output_root / "logs/G0/dwa/stderr.log").read_text(
        encoding="utf-8"
    ) == "simulator failed\n"
    assert not (output_root / "result_manifest.json").exists()


def test_timeout_invalidates_and_preserves_partial_process_output(tmp_path: Path) -> None:
    final_lock, lock_path = _formal_fixture(tmp_path)
    invalidations: list[dict[str, Any]] = []

    def timeout(command: Sequence[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            command,
            timeout=30,
            output=b"partial stdout\n",
            stderr=b"partial stderr\n",
        )

    with pytest.raises(subprocess.TimeoutExpired):
        execute_formal_test(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            webots_executable=_webots_executable(tmp_path),
            confirm_test=True,
            process_runner=timeout,
            executable_probe_runner=_version_probe(),
            windows_metadata_probe=_windows_metadata,
            port_allocator=lambda preferred, reserved: (
                reserved.add(preferred + len(reserved)) or preferred + len(reserved) - 1
            ),
            lock_loader=_lock_loader(final_lock, []),
            source_verifier=lambda _root: final_lock.source_revision,
            port_verifier=lambda _port: None,
            begin_attempt=lambda **_kwargs: {"status": "attempt"},
            mark_success=lambda **_kwargs: pytest.fail("timeout was marked successful"),
            invalidate_attempt=lambda **kwargs: (
                invalidations.append(kwargs) or {"status": "invalidated"}
            ),
        )

    assert len(invalidations) == 1
    assert "TimeoutExpired" in invalidations[0]["reason"]
    logs = tmp_path / final_lock.output_directory / "logs/G0/reactive"
    assert (logs / "invocation.json").is_file()
    assert (logs / "stdout.log").read_text(encoding="utf-8") == "partial stdout\n"
    assert (logs / "stderr.log").read_text(encoding="utf-8") == "partial stderr\n"
    error = json.loads((logs / "invocation_error.json").read_text(encoding="utf-8"))
    assert error["error_type"] == "TimeoutExpired"
