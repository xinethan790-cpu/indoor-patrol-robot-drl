from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from indoor_patrol_rl.evaluation.formal_protocol import (
    FORMAL_TEST_SEEDS,
    FORMAL_TRAINING_SEEDS,
    FORMAL_VALIDATION_SEEDS,
    FinalLock,
    FormalModelManifest,
    SplitSpec,
    begin_test_attempt,
    build_formal_source_bundle,
    canonical_sha256,
    file_sha256,
    invalidate_test_attempt,
    load_final_lock,
    load_test_ledger,
    mark_test_success,
    protected_test_consumption_key,
    protected_test_ledger_path,
    source_bundle_sha256,
    write_final_lock,
    write_or_validate_model_provenance,
)

REVISION = "a" * 40
STARTED_AT = datetime(2026, 8, 30, 1, 2, 3, tzinfo=UTC)
FINISHED_AT = datetime(2026, 8, 30, 2, 3, 4, tzinfo=UTC)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _runtime_evidence(root: Path) -> dict[str, object]:
    executable = root / "tools" / "webots.exe"
    return {
        "schema_version": 1,
        "executable_path": str(executable.resolve()),
        "executable_sha256": file_sha256(executable),
        "expected_version": "R2025a",
        "detected_version": "R2025a",
        "version_probe_stdout": "Webots R2025a\n",
        "version_probe_stderr": "",
        "windows_file_version": "2025.1.0.0",
        "windows_product_version": "2025.1.0.0",
        "signature_status": "NotSigned",
        "signature_status_message": None,
        "signer_subject": None,
        "signer_thumbprint": None,
        "probed_at": "2026-08-30T01:00:00+00:00",
    }


def _build_model_manifest(
    root: Path,
    seed: int,
    *,
    validation_seeds: tuple[int, ...] = FORMAL_VALIDATION_SEEDS,
    checkpoint: bool = False,
    learning_rate: float = 0.0003,
    w2_training_steps: int = 300_000,
) -> FormalModelManifest:
    model_root = root / "artifacts" / "models" / "webots" / f"seed_{seed}"
    final_model = model_root / f"ppo_webots_s{seed}_step000500000.zip"
    final_model.parent.mkdir(parents=True, exist_ok=True)
    final_model.write_bytes(f"final model seed {seed}".encode())
    model_path = (
        model_root / "checkpoints" / f"ppo_webots_s{seed}_400000_steps.zip"
        if checkpoint
        else final_model
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        model_path.write_bytes(f"checkpoint model seed {seed}".encode())
    model_digest = file_sha256(model_path)

    metadata_path = model_root / "run_metadata_step000500000.yaml"
    algorithm = {
        "mode": "resume",
        "model_variant": None,
        "obstacle_count": 3,
        "obstacle_speed_range": [0.2, 0.42],
        "reward_config": {"progress_weight": 1.0, "collision_penalty": -10.0},
        "reward_safety_distance": 0.3,
        "proximity_speed_distance": 0.5,
        "proximity_speed_weight": 0.4,
        "route_half_lengths": {"horizontal": 1.55, "vertical": 1.1},
        "ppo": {
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "learning_rate": learning_rate,
            "ent_coef": 0.0,
            "gamma": 0.99,
            "target_kl": None,
            "initial_action_std_override": None,
            "reset_linear_head_mean": None,
        },
        "safety_shield_enabled": False,
        "safety_shield_intervention_penalty": 0.0,
        "safety_shield_config": {"stop_distance": 0.16},
        "starting_webots_steps": 200_000,
        "additional_steps_requested": 300_000,
        "final_model_steps": 500_000,
    }
    metadata_path.write_text(
        yaml.safe_dump(
            {
                **algorithm,
                "seed": seed,
                "final_model": _relative(final_model, root),
                "final_model_sha256": file_sha256(final_model),
                "git": {"revision": REVISION, "dirty": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    curriculum_config = {
        "semantic_schema_version": 1,
        "seed": seed,
        "transfer_model": f"artifacts/models/transfer_seed{seed}.zip",
        "policy_transfer_model": None,
        "model_variant": None,
        "port_base": 12_490,
        "automatic_split": "validation",
        "evaluation_world": "webots_ppo_evaluate.wbt",
        "gate_overrides": {
            "seed_start": None,
            "episodes": None,
            "min_success_rate": None,
            "max_collision_rate": None,
        },
        "stages": [
            {
                "id": stage,
                "name": name,
                "world": "webots_ppo_train.wbt",
                "obstacle_count": obstacles,
                "training_steps": steps,
                "gate": {
                    "seed_start": 5000,
                    "episodes": 20,
                    "min_success_rate": 0.75,
                    "max_collision_rate": 0.15,
                },
                "controller_env": {},
                "preferred_resume_model": (
                    None if stage == "W0" else f"artifacts/models/seed_{seed}/{stage}.zip"
                ),
            }
            for stage, name, obstacles, steps in (
                ("W0", "empty", 0, 50_000),
                ("W1", "one", 1, 150_000),
                ("W2", "three", 3, w2_training_steps),
            )
        ],
    }
    curriculum_path = (
        root / "artifacts" / "results" / f"webots_curriculum_seed{seed}" / "run_manifest.yaml"
    )
    curriculum_path.parent.mkdir(parents=True, exist_ok=True)
    curriculum_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "seed": seed,
                "model_variant": None,
                "curriculum_config_sha256": canonical_sha256(curriculum_config),
                "curriculum_config": curriculum_config,
                "status": "completed",
                "automatic_split": "validation",
                "test_split_was_run": False,
                "created_at": f"2026-01-{seed % 10 + 1:02d}T00:00:00Z",
                "updated_at": f"2026-02-{seed % 10 + 1:02d}T00:00:00Z",
                "stages": [
                    {
                        "id": stage,
                        "status": "completed",
                        "validation": {"passed": True, "finished_at": "2026-01-01T00:00:00Z"},
                    }
                    for stage in ("W0", "W1", "W2")
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    selected_step = 400_000 if checkpoint else 500_000
    planner = f"ppo_w2_step{selected_step}_seed{seed}"
    condition = "W2"
    validation_directory = root / "artifacts" / "results" / f"validation_seed{seed}"
    episodes_path = validation_directory / "episodes.csv"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    with episodes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["split", "scenario_seed", "planner", "condition_id", "outcome"],
        )
        writer.writeheader()
        for index, scenario_seed in enumerate(validation_seeds):
            outcome = "success" if index < 16 else "collision" if index < 18 else "timeout"
            writer.writerow(
                {
                    "split": "validation",
                    "scenario_seed": scenario_seed,
                    "planner": planner,
                    "condition_id": condition,
                    "outcome": outcome,
                }
            )

    summary_path = validation_directory / "summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": "1.0",
            "split": "validation",
            "episode_count": 20,
            "planners": [planner],
            "conditions": [condition],
            "by_planner": [
                {
                    "planner": planner,
                    "episodes": 20,
                    "success_rate": 0.80,
                    "collision_rate": 0.10,
                }
            ],
            "by_planner_condition": [
                {
                    "planner": planner,
                    "condition_id": condition,
                    "episodes": 20,
                    "success_rate": 0.80,
                    "collision_rate": 0.10,
                }
            ],
            "evaluation_provenance": {
                "schema_version": 1,
                "model_path": _relative(model_path, root),
                "model_sha256": model_digest,
                "training_seed": seed,
                "model_variant": None,
            },
        },
    )

    selected = {
        "rank": 1,
        "kind": "checkpoint" if checkpoint else "final",
        "step": selected_step,
        "planner": planner,
        "model_variant": None,
        "model_path": _relative(model_path, root),
        "model_sha256": model_digest,
        "episodes": 20,
        "success_rate": 0.80,
        "collision_rate": 0.10,
        "gate_passed": True,
    }
    candidate = {
        **selected,
        "summary_path": _relative(summary_path, root),
        "summary_sha256": file_sha256(summary_path),
    }
    selection_path = (
        root / "artifacts" / "results" / f"selection_seed{seed}" / "selection.json"
    )
    _write_json(
        selection_path,
        {
            "schema_version": 1,
            "selection_status": "formal_selected",
            "model_variant": None,
            "validation_contract": {
                "split": "validation",
                "seed": seed,
                "condition": condition,
                "episodes": 20,
                "model_variant": None,
            },
            "gate": {
                "min_success_rate": 0.75,
                "max_collision_rate": 0.15,
                "requires_model_file": True,
            },
            "formal_selected": selected,
            "candidates": [candidate],
        },
    )
    provenance_path, provenance = write_or_validate_model_provenance(
        repository_root=root,
        training_seed=seed,
        model_path=model_path,
        metadata_path=metadata_path,
        curriculum_manifest_path=curriculum_path,
        allow_create=True,
    )
    curriculum = yaml.safe_load(curriculum_path.read_text(encoding="utf-8"))
    curriculum["stages"][2]["artifact_provenance"] = [
        {
            "model_path": _relative(model_path, root),
            "model_sha256": model_digest,
            "provenance_path": _relative(provenance_path, root),
            "provenance_sha256": file_sha256(provenance_path),
        }
    ]
    curriculum_path.write_text(
        yaml.safe_dump(curriculum, sort_keys=False), encoding="utf-8"
    )
    return FormalModelManifest(
        training_seed=seed,
        model_path=_relative(model_path, root),
        model_sha256=model_digest,
        training_metadata_path=_relative(metadata_path, root),
        training_metadata_sha256=file_sha256(metadata_path),
        model_provenance_path=_relative(provenance_path, root),
        model_provenance_sha256=file_sha256(provenance_path),
        curriculum_manifest_path=_relative(curriculum_path, root),
        curriculum_manifest_sha256=file_sha256(curriculum_path),
        selection_manifest_path=_relative(selection_path, root),
        selection_manifest_sha256=file_sha256(selection_path),
        validation_episodes_path=_relative(episodes_path, root),
        validation_episodes_sha256=file_sha256(episodes_path),
        training_stage="W2",
        model_variant=None,
        training_protocol_sha256=provenance["training_protocol_sha256"],
        algorithm_config_sha256=provenance["algorithm_config_sha256"],
    )


def _build_final_lock(
    root: Path,
    *,
    campaign_id: str = "thesis-final-v1",
    validation_seeds: tuple[int, ...] = FORMAL_VALIDATION_SEEDS,
    checkpoint: bool = False,
    learning_rates: dict[int, float] | None = None,
    w2_training_steps: dict[int, int] | None = None,
) -> FinalLock:
    evaluation_config = root / "configs" / "eval" / "formal.yaml"
    evaluation_config.parent.mkdir(parents=True, exist_ok=True)
    evaluation_config.write_text(
        "world: worlds/webots_ppo_evaluate.wbt\nconditions: [G0, G1, G2, G3]\n",
        encoding="utf-8",
    )
    world = root / "worlds" / "webots_ppo_evaluate.wbt"
    evaluator = root / "scripts" / "run_webots_formal_test.py"
    controller = root / "controllers" / "webots_ppo_evaluate" / "webots_ppo_evaluate.py"
    package = root / "src" / "indoor_patrol_rl" / "evaluation" / "fixture.py"
    requirements = root / "requirements-formal-lock.txt"
    executable = root / "tools" / "webots.exe"
    for path, content in (
        (world, "#VRML_SIM R2025a utf8\n"),
        (evaluator, "# formal runner fixture\n"),
        (controller, "# formal controller fixture\n"),
        (package, "# source fixture\n"),
        (requirements, "Webots==R2025a\n"),
        (executable, "synthetic Webots R2025a executable\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    manifests = tuple(
        _build_model_manifest(
            root,
            seed,
            validation_seeds=validation_seeds,
            checkpoint=checkpoint,
            learning_rate=(learning_rates or {}).get(seed, 0.0003),
            w2_training_steps=(w2_training_steps or {}).get(seed, 300_000),
        )
        for seed in FORMAL_TRAINING_SEEDS
    )
    split_spec = SplitSpec()
    source_bundle = build_formal_source_bundle(root)
    return FinalLock(
        campaign_id=campaign_id,
        source_revision=REVISION,
        source_dirty=False,
        evaluation_config_path=_relative(evaluation_config, root),
        evaluation_config_sha256=file_sha256(evaluation_config),
        world_path=_relative(world, root),
        world_sha256=file_sha256(world),
        evaluator_path=_relative(evaluator, root),
        evaluator_sha256=file_sha256(evaluator),
        controller_path=_relative(controller, root),
        controller_sha256=file_sha256(controller),
        source_bundle=source_bundle,
        source_bundle_sha256=source_bundle_sha256(source_bundle),
        test_consumption_key=protected_test_consumption_key(split_spec),
        output_directory=f"artifacts/results/formal_runs/{campaign_id}",
        test_ledger_path=protected_test_ledger_path(split_spec),
        model_manifests=manifests,
        split_spec=split_spec,
    )


def _write_lock(root: Path, final_lock: FinalLock) -> Path:
    path = root / "artifacts" / "results" / "formal_locks" / f"{final_lock.campaign_id}.json"
    return write_final_lock(path, final_lock, repository_root=root)


def _write_result_manifest(root: Path, final_lock: FinalLock) -> Path:
    path = root / final_lock.output_directory / "result_manifest.json"
    report = path.parent / "formal_report.json"
    _write_json(report, {"schema_version": "1.0", "matrix_validation": "passed"})
    executable_evidence = _runtime_evidence(root)
    executable_evidence_path = path.parent / "webots_executable.json"
    _write_json(executable_evidence_path, executable_evidence)
    artifacts = []
    for condition in ("G0", "G1", "G2", "G3"):
        for planner in ("reactive", "dwa", "ppo_seed11", "ppo_seed22", "ppo_seed33"):
            run = path.parent / "runs" / condition / planner
            episodes = run / "episodes.csv"
            episodes.parent.mkdir(parents=True, exist_ok=True)
            with episodes.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["split", "scenario_seed", "planner", "condition_id"],
                )
                writer.writeheader()
                for scenario_seed in FORMAL_TEST_SEEDS:
                    writer.writerow(
                        {
                            "split": "test",
                            "scenario_seed": scenario_seed,
                            "planner": planner,
                            "condition_id": condition,
                        }
                    )
            episodes_digest = file_sha256(episodes)
            training_seed = (
                int(planner.removeprefix("ppo_seed"))
                if planner.startswith("ppo_seed")
                else None
            )
            model = (
                next(
                    item
                    for item in final_lock.model_manifests
                    if item.training_seed == training_seed
                )
                if training_seed is not None
                else None
            )
            provenance = {
                "schema_version": 1,
                "split": "test",
                "condition_id": condition,
                "planner": "ppo" if training_seed is not None else planner,
                "planner_label": planner,
                "training_seed": training_seed,
                "model_path": model.model_path if model is not None else None,
                "model_sha256": model.model_sha256 if model is not None else None,
                "scenario_seeds": list(FORMAL_TEST_SEEDS),
                "formal_campaign_id": final_lock.campaign_id,
                "final_lock_sha256": final_lock.lock_sha256,
                "test_consumption_key": final_lock.test_consumption_key,
                "evaluation_config_sha256": final_lock.evaluation_config_sha256,
                "formal_config_path": final_lock.evaluation_config_path,
                "formal_config_sha256": final_lock.evaluation_config_sha256,
                "world_path": final_lock.world_path,
                "world_sha256": final_lock.world_sha256,
                "evaluator_sha256": final_lock.evaluator_sha256,
                "controller_sha256": final_lock.controller_sha256,
                "source_bundle_sha256": final_lock.source_bundle_sha256,
                "episodes_csv_sha256": episodes_digest,
            }
            if model is not None:
                provenance["model_variant"] = model.model_variant
            summary = run / "summary.json"
            _write_json(
                summary,
                {
                    "split": "test",
                    "episode_count": 20,
                    "planners": [planner],
                    "conditions": [condition],
                    "episodes_sha256": episodes_digest,
                    "evaluation_provenance": provenance,
                },
            )
            artifacts.append(
                {
                    "episodes_path": _relative(episodes, root),
                    "episodes_sha256": episodes_digest,
                    "summary_path": _relative(summary, root),
                    "summary_sha256": file_sha256(summary),
                }
            )
    _write_json(
        path,
        {
            "schema_version": 1,
            "campaign_id": final_lock.campaign_id,
            "final_lock_sha256": final_lock.lock_sha256,
            "test_consumption_key": final_lock.test_consumption_key,
            "source_revision": final_lock.source_revision,
            "world_path": final_lock.world_path,
            "world_sha256": final_lock.world_sha256,
            "evaluator_sha256": final_lock.evaluator_sha256,
            "controller_sha256": final_lock.controller_sha256,
            "source_bundle_sha256": final_lock.source_bundle_sha256,
            "status": "completed",
            "split": "test",
            "training_seeds": list(FORMAL_TRAINING_SEEDS),
            "scenario_seeds": list(FORMAL_TEST_SEEDS),
            "models": [
                {
                    "training_seed": item.training_seed,
                    "model_path": item.model_path,
                    "model_sha256": item.model_sha256,
                }
                for item in final_lock.model_manifests
            ],
            "evaluation_config_path": final_lock.evaluation_config_path,
            "evaluation_config_sha256": final_lock.evaluation_config_sha256,
            "formal_report_path": _relative(report, root),
            "formal_report_sha256": file_sha256(report),
            "webots_executable": executable_evidence,
            "webots_executable_evidence_path": _relative(
                executable_evidence_path, root
            ),
            "webots_executable_evidence_sha256": file_sha256(
                executable_evidence_path
            ),
            "run_count": 20,
            "episode_count": 400,
            "run_artifacts": artifacts,
            "completed_at": "2026-08-30T02:03:04+00:00",
        },
    )
    return path


def test_split_spec_is_fixed_and_requires_explicit_test_confirmation() -> None:
    split_spec = SplitSpec()

    assert split_spec.training_seeds == (11, 22, 33)
    assert split_spec.validation_scenario_seeds == tuple(range(5_000, 5_020))
    assert split_spec.test_scenario_seeds == tuple(range(10_000, 10_020))
    assert split_spec.require_training_seeds([11, 22, 33]) == (11, 22, 33)
    assert split_spec.require_evaluation_seeds(
        "validation", list(FORMAL_VALIDATION_SEEDS)
    ) == FORMAL_VALIDATION_SEEDS
    with pytest.raises(PermissionError, match="confirm_test"):
        split_spec.require_evaluation_seeds("test", FORMAL_TEST_SEEDS)
    assert split_spec.require_evaluation_seeds(
        "test", FORMAL_TEST_SEEDS, confirm_test=True
    ) == FORMAL_TEST_SEEDS


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("training_seeds", (11, 22), "training seeds"),
        ("training_seeds", (11, 33, 22), "training seeds"),
        ("validation_scenario_seeds", FORMAL_TEST_SEEDS, "validation scenario seeds"),
        ("test_scenario_seeds", FORMAL_VALIDATION_SEEDS, "test scenario seeds"),
    ],
)
def test_split_spec_rejects_every_noncanonical_sequence(
    field_name: str, value: tuple[int, ...], message: str
) -> None:
    kwargs = {field_name: value}
    with pytest.raises(ValueError, match=message):
        SplitSpec(**kwargs)


def test_split_spec_round_trip_rejects_unknown_fields() -> None:
    split_spec = SplitSpec()

    assert SplitSpec.from_mapping(split_spec.to_mapping()) == split_spec
    tampered = {**split_spec.to_mapping(), "test_seed_override": 42}
    with pytest.raises(ValueError, match="extra"):
        SplitSpec.from_mapping(tampered)


def test_final_lock_validates_three_complete_models_and_is_immutable(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)

    loaded = load_final_lock(lock_path, repository_root=tmp_path)

    assert loaded == final_lock
    assert loaded.lock_sha256 == final_lock.lock_sha256
    assert [item.training_seed for item in loaded.model_manifests] == [11, 22, 33]
    with pytest.raises(FileExistsError):
        write_final_lock(lock_path, final_lock, repository_root=tmp_path)


def test_final_lock_rejects_missing_or_reordered_training_seed(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path)

    with pytest.raises(ValueError, match="training seeds"):
        replace(final_lock, model_manifests=final_lock.model_manifests[:2])
    with pytest.raises(ValueError, match="training seeds"):
        replace(final_lock, model_manifests=tuple(reversed(final_lock.model_manifests)))


def test_final_lock_hash_detects_payload_tampering(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["campaign_id"] = "changed-campaign"
    _write_json(lock_path, payload)

    with pytest.raises(ValueError, match="SHA-256"):
        load_final_lock(lock_path, repository_root=tmp_path, verify_files=False)


def test_final_lock_rehashes_every_referenced_file(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    model_path = tmp_path / final_lock.model_manifests[0].model_path
    model_path.write_bytes(b"model changed after lock")

    with pytest.raises(ValueError, match="formal model SHA-256 mismatch"):
        load_final_lock(lock_path, repository_root=tmp_path)


def test_model_manifest_rejects_test_seeds_disguised_as_validation(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path, validation_seeds=FORMAL_TEST_SEEDS)

    with pytest.raises(ValueError, match="validation scenario seeds"):
        final_lock.validate_files(tmp_path)


def test_model_manifest_recomputes_selected_rates_from_raw_csv(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path)
    first = final_lock.model_manifests[0]
    episodes_path = tmp_path / first.validation_episodes_path
    rows = list(csv.DictReader(episodes_path.read_text(encoding="utf-8").splitlines()))
    rows[0]["outcome"] = "collision"
    with episodes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changed = replace(first, validation_episodes_sha256=file_sha256(episodes_path))
    changed_lock = replace(
        final_lock,
        model_manifests=(changed, *final_lock.model_manifests[1:]),
    )

    with pytest.raises(ValueError, match="raw validation CSV"):
        changed_lock.validate_files(tmp_path)


def test_test_ledger_enforces_one_attempt_success_and_terminal_invalidation(
    tmp_path: Path,
) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)

    with pytest.raises(PermissionError, match="confirm_test"):
        begin_test_attempt(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            confirm_test=False,
            runtime_evidence=_runtime_evidence(tmp_path),
        )

    attempt = begin_test_attempt(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        confirm_test=True,
        runtime_evidence=_runtime_evidence(tmp_path),
        attempt_id="formal-attempt-001",
        now=STARTED_AT,
    )
    ledger_path = tmp_path / final_lock.test_ledger_path
    assert attempt["status"] == "attempt"
    assert attempt["attempt"]["started_at"] == "2026-08-30T01:02:03+00:00"
    assert load_test_ledger(ledger_path) == attempt
    with pytest.raises(FileExistsError, match="second attempt"):
        begin_test_attempt(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            confirm_test=True,
            runtime_evidence=_runtime_evidence(tmp_path),
        )

    result_path = _write_result_manifest(tmp_path, final_lock)
    success = mark_test_success(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        result_manifest_path=result_path,
        now=FINISHED_AT,
    )
    assert success["status"] == "success"
    assert success["success"]["result_manifest_sha256"] == file_sha256(result_path)
    with pytest.raises(RuntimeError, match="active attempt"):
        mark_test_success(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            result_manifest_path=result_path,
        )

    invalidated = invalidate_test_attempt(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        reason="A simulator implementation defect invalidated the published result.",
        now=FINISHED_AT,
    )
    assert invalidated["status"] == "invalidated"
    assert invalidated["success"] == success["success"]
    assert invalidated["invalidation"]["previous_status"] == "success"
    with pytest.raises(RuntimeError, match="terminal"):
        invalidate_test_attempt(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            reason="Trying to overwrite the first invalidation.",
        )


def test_test_attempt_refuses_to_overwrite_an_existing_output_directory(
    tmp_path: Path,
) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    (tmp_path / final_lock.output_directory).mkdir(parents=True)

    with pytest.raises(FileExistsError, match="output directory already exists"):
        begin_test_attempt(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            confirm_test=True,
            runtime_evidence=_runtime_evidence(tmp_path),
        )
    assert not (tmp_path / final_lock.test_ledger_path).exists()


def test_invalid_result_manifest_does_not_consume_the_active_attempt(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    begin_test_attempt(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        confirm_test=True,
        runtime_evidence=_runtime_evidence(tmp_path),
    )
    result_path = _write_result_manifest(tmp_path, final_lock)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["scenario_seeds"] = list(FORMAL_VALIDATION_SEEDS)
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="test scenario seeds"):
        mark_test_success(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            result_manifest_path=result_path,
        )
    ledger = load_test_ledger(tmp_path / final_lock.test_ledger_path)
    assert ledger["status"] == "attempt"


def test_invalidation_stays_available_when_a_locked_artifact_was_modified(
    tmp_path: Path,
) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    begin_test_attempt(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        confirm_test=True,
        runtime_evidence=_runtime_evidence(tmp_path),
    )
    (tmp_path / final_lock.model_manifests[0].model_path).write_bytes(b"corrupted")

    invalidated = invalidate_test_attempt(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        reason="The locked model artifact changed after the test attempt began.",
    )

    assert invalidated["status"] == "invalidated"
    assert invalidated["invalidation"]["previous_status"] == "attempt"


def test_ledger_hash_detects_manual_status_overwrite(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    begin_test_attempt(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        confirm_test=True,
        runtime_evidence=_runtime_evidence(tmp_path),
    )
    ledger_path = tmp_path / final_lock.test_ledger_path
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["status"] = "success"
    _write_json(ledger_path, ledger)

    with pytest.raises(ValueError, match="SHA-256"):
        load_test_ledger(ledger_path)


def test_protected_test_consumption_is_global_across_campaign_names(
    tmp_path: Path,
) -> None:
    first = _build_final_lock(tmp_path, campaign_id="campaign-a")
    first_path = _write_lock(tmp_path, first)
    begin_test_attempt(
        repository_root=tmp_path,
        final_lock_path=first_path,
        confirm_test=True,
        runtime_evidence=_runtime_evidence(tmp_path),
    )
    second = _build_final_lock(tmp_path, campaign_id="campaign-b")
    second_path = _write_lock(tmp_path, second)

    assert first.test_consumption_key == second.test_consumption_key
    assert first.test_ledger_path == second.test_ledger_path
    with pytest.raises(FileExistsError, match="second attempt"):
        begin_test_attempt(
            repository_root=tmp_path,
            final_lock_path=second_path,
            confirm_test=True,
            runtime_evidence=_runtime_evidence(tmp_path),
        )


def test_final_lock_rejects_a_campaign_scoped_or_custom_test_ledger(
    tmp_path: Path,
) -> None:
    final_lock = _build_final_lock(tmp_path)

    with pytest.raises(ValueError, match="repository-global"):
        replace(
            final_lock,
            test_ledger_path="artifacts/results/formal_ledgers/another-campaign.json",
        )


def test_final_lock_accepts_selected_intermediate_checkpoint_sidecars(
    tmp_path: Path,
) -> None:
    final_lock = _build_final_lock(tmp_path, checkpoint=True)
    lock_path = _write_lock(tmp_path, final_lock)

    loaded = load_final_lock(lock_path, repository_root=tmp_path)

    assert all("/checkpoints/" in item.model_path for item in loaded.model_manifests)
    for item in loaded.model_manifests:
        metadata = yaml.safe_load(
            (tmp_path / item.training_metadata_path).read_text(encoding="utf-8")
        )
        assert metadata["final_model_sha256"] != item.model_sha256
        provenance = json.loads(
            (tmp_path / item.model_provenance_path).read_text(encoding="utf-8")
        )
        assert provenance["artifact_kind"] == "checkpoint"
        assert provenance["model_sha256"] == item.model_sha256


def test_three_models_require_identical_normalized_protocol_and_algorithm(
    tmp_path: Path,
) -> None:
    final_lock = _build_final_lock(tmp_path)
    assert len({item.training_protocol_sha256 for item in final_lock.model_manifests}) == 1
    assert len({item.algorithm_config_sha256 for item in final_lock.model_manifests}) == 1

    with pytest.raises(ValueError, match="algorithm configuration"):
        _build_final_lock(
            tmp_path / "mismatch",
            learning_rates={33: 0.0001},
        )
    with pytest.raises(ValueError, match="training protocol fingerprint"):
        _build_final_lock(
            tmp_path / "protocol-mismatch",
            w2_training_steps={22: 299_999},
        )


def test_result_manifest_must_keep_all_twenty_hashed_run_artifacts(
    tmp_path: Path,
) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    begin_test_attempt(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        confirm_test=True,
        runtime_evidence=_runtime_evidence(tmp_path),
    )
    result_path = _write_result_manifest(tmp_path, final_lock)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["run_artifacts"] = result["run_artifacts"][:-1]
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="exactly 20"):
        mark_test_success(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            result_manifest_path=result_path,
        )
    assert load_test_ledger(tmp_path / final_lock.test_ledger_path)["status"] == "attempt"


def test_result_manifest_rehashes_every_run_artifact_before_success(
    tmp_path: Path,
) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    begin_test_attempt(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        confirm_test=True,
        runtime_evidence=_runtime_evidence(tmp_path),
    )
    result_path = _write_result_manifest(tmp_path, final_lock)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    episodes = tmp_path / result["run_artifacts"][7]["episodes_path"]
    episodes.write_text("tampered after result manifest\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        mark_test_success(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            result_manifest_path=result_path,
        )
    assert load_test_ledger(tmp_path / final_lock.test_ledger_path)["status"] == "attempt"


def test_result_runtime_evidence_must_match_atomic_attempt_ledger(
    tmp_path: Path,
) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    begin_test_attempt(
        repository_root=tmp_path,
        final_lock_path=lock_path,
        confirm_test=True,
        runtime_evidence=_runtime_evidence(tmp_path),
    )
    result_path = _write_result_manifest(tmp_path, final_lock)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    changed = dict(result["webots_executable"])
    changed["probed_at"] = "2026-08-30T01:00:01+00:00"
    evidence_path = tmp_path / result["webots_executable_evidence_path"]
    _write_json(evidence_path, changed)
    result["webots_executable"] = changed
    result["webots_executable_evidence_sha256"] = file_sha256(evidence_path)
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="atomic attempt ledger"):
        mark_test_success(
            repository_root=tmp_path,
            final_lock_path=lock_path,
            result_manifest_path=result_path,
        )


def test_final_lock_rejects_world_and_source_bundle_drift(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    (tmp_path / final_lock.controller_path).write_text(
        "# changed after lock\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_final_lock(lock_path, repository_root=tmp_path)


def test_final_lock_rejects_new_unlocked_source_bundle_members(tmp_path: Path) -> None:
    final_lock = _build_final_lock(tmp_path)
    lock_path = _write_lock(tmp_path, final_lock)
    injected = tmp_path / "src" / "indoor_patrol_rl" / "unlocked.py"
    injected.write_text("# source added after final lock\n", encoding="utf-8")

    with pytest.raises(ValueError, match="membership drifted"):
        load_final_lock(lock_path, repository_root=tmp_path)
