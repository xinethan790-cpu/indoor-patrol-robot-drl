from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from indoor_patrol_rl.evaluation.checkpoint_selection import (
    CheckpointSelectionGate,
    build_checkpoint_selection,
    make_candidate_record,
    parse_result_identity,
    sha256_file,
    write_checkpoint_selection,
)
from scripts.select_webots_checkpoint import selection_exit_code


def gate(
    *,
    success: float = 0.75,
    collision: float = 0.15,
    model_variant: str | None = None,
) -> CheckpointSelectionGate:
    return CheckpointSelectionGate(
        seed=11,
        condition="W1",
        episodes=20,
        min_success_rate=success,
        max_collision_rate=collision,
        model_variant=model_variant,
    )


def save_candidate(
    root: Path,
    *,
    step: int,
    kind: str = "checkpoint",
    success_count: int = 16,
    collision_count: int = 2,
    path_ratio: float = 1.1,
    completion_time: float = 16.0,
    create_model: bool = True,
    model_variant: str | None = None,
) -> tuple[Path, Path]:
    planner = f"ppo_w1_step{step}"
    suffix = f"checkpoint{step}" if kind == "checkpoint" else f"final_step{step}"
    result_directory = (
        root / "artifacts/results" / f"webots_validation_w1_{suffix}_seed11"
    )
    result_directory.mkdir(parents=True)
    model_directory = root / "artifacts/models/webots/seed_11"
    if model_variant is not None:
        model_directory /= model_variant
    if kind == "checkpoint":
        variant_token = f"_{model_variant}" if model_variant is not None else ""
        model_path = (
            model_directory
            / "checkpoints"
            / f"ppo_webots_s11{variant_token}_{step}_steps.zip"
        )
    else:
        model_path = model_directory / f"ppo_webots_s11_step{step:09d}.zip"
    model_bytes = f"model-{step}".encode()
    if create_model:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(model_bytes)

    timeout_count = 20 - success_count - collision_count
    row = {
        "planner": planner,
        "episodes": 20,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "error_count": 0,
        "success_rate": success_count / 20,
        "collision_rate": collision_count / 20,
        "timeout_rate": timeout_count / 20,
        "completion_time_s_mean": completion_time,
        "path_length_ratio_mean": path_ratio,
    }
    condition_row = {**row, "condition_id": "W1"}
    summary = {
        "schema_version": "1.0",
        "split": "validation",
        "episode_count": 20,
        "planners": [planner],
        "conditions": ["W1"],
        "by_planner_condition": [condition_row],
        "by_planner": [row],
        "evaluation_provenance": {
            "schema_version": 1,
            "model_path": model_path.relative_to(root).as_posix(),
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "model_variant": model_variant,
            "training_seed": 11,
        },
    }
    summary_path = result_directory / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, model_path


def bind_summary_to_model(
    summary_path: Path,
    *,
    repository_root: Path,
    model_path: Path,
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evaluation_provenance"]["model_path"] = model_path.relative_to(
        repository_root
    ).as_posix()
    summary["evaluation_provenance"]["model_sha256"] = sha256_file(model_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_identity_supports_checkpoint_and_final_result_names(tmp_path: Path) -> None:
    checkpoint, _ = save_candidate(tmp_path, step=123)
    final, _ = save_candidate(tmp_path, step=456, kind="final")

    assert parse_result_identity(checkpoint).kind == "checkpoint"
    assert parse_result_identity(checkpoint).step == 123
    assert parse_result_identity(final).kind == "final"
    assert parse_result_identity(final).step == 456


def test_legacy_checkpoint_path_remains_unchanged(tmp_path: Path) -> None:
    summary_path, model_path = save_candidate(tmp_path, step=20480)

    candidate = make_candidate_record(
        summary_path,
        repository_root=tmp_path,
        gate=gate(),
    )

    expected = (
        tmp_path
        / "artifacts/models/webots/seed_11/checkpoints"
        / "ppo_webots_s11_20480_steps.zip"
    )
    assert model_path == expected
    assert candidate["model_variant"] is None
    assert candidate["model_path"] == expected.relative_to(tmp_path).as_posix()
    assert candidate["model_resolution"] == "checkpoint_filename"


def test_variant_checkpoint_uses_variant_directory_and_filename(tmp_path: Path) -> None:
    summary_path, model_path = save_candidate(
        tmp_path,
        step=40960,
        model_variant="curriculum_v2",
    )

    selection = build_checkpoint_selection(
        [summary_path],
        repository_root=tmp_path,
        gate=gate(model_variant="curriculum_v2"),
    )

    expected = (
        tmp_path
        / "artifacts/models/webots/seed_11/curriculum_v2/checkpoints"
        / "ppo_webots_s11_curriculum_v2_40960_steps.zip"
    )
    assert model_path == expected
    assert selection["selection_status"] == "formal_selected"
    assert selection["model_variant"] == "curriculum_v2"
    assert selection["validation_contract"]["model_variant"] == "curriculum_v2"
    assert selection["formal_selected"]["model_variant"] == "curriculum_v2"
    assert (
        selection["candidates"][0]["model_path"]
        == expected.relative_to(tmp_path).as_posix()
    )


def test_variant_final_model_uses_variant_directory(tmp_path: Path) -> None:
    summary_path, model_path = save_candidate(
        tmp_path,
        step=50176,
        kind="final",
        model_variant="curriculum_v2",
    )

    candidate = make_candidate_record(
        summary_path,
        repository_root=tmp_path,
        gate=gate(model_variant="curriculum_v2"),
    )

    assert candidate["model_variant"] == "curriculum_v2"
    assert candidate["model_path"] == model_path.relative_to(tmp_path).as_posix()
    assert candidate["model_resolution"] == "zero_padded_final_filename"


@pytest.mark.parametrize(
    "model_variant",
    ["", ".", "..", "nested/variant", r"nested\variant"],
)
def test_model_variant_rejects_empty_dot_and_path_separators(model_variant: str) -> None:
    with pytest.raises(ValueError, match="model_variant"):
        gate(model_variant=model_variant)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("split", "test", "expected validation"),
        ("episode_count", 19, "episode_count"),
        ("planners", ["ppo_wrong_step123"], "expected planner"),
        ("conditions", ["W2"], "condition"),
    ],
)
def test_strict_validation_rejects_wrong_contract(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    summary_path, _ = save_candidate(tmp_path, step=123)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = bad_value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        make_candidate_record(summary_path, repository_root=tmp_path, gate=gate())


def test_selection_uses_all_four_metrics_in_declared_order(tmp_path: Path) -> None:
    paths = [
        save_candidate(
            tmp_path,
            step=100,
            success_count=15,
            collision_count=1,
            path_ratio=0.9,
            completion_time=10.0,
        )[0],
        save_candidate(
            tmp_path,
            step=200,
            success_count=16,
            collision_count=3,
            path_ratio=1.2,
            completion_time=18.0,
        )[0],
        save_candidate(
            tmp_path,
            step=300,
            success_count=16,
            collision_count=2,
            path_ratio=1.3,
            completion_time=17.0,
        )[0],
        save_candidate(
            tmp_path,
            step=400,
            success_count=16,
            collision_count=2,
            path_ratio=1.1,
            completion_time=17.0,
        )[0],
        save_candidate(
            tmp_path,
            step=500,
            success_count=16,
            collision_count=2,
            path_ratio=1.1,
            completion_time=16.0,
        )[0],
    ]

    selection = build_checkpoint_selection(
        paths, repository_root=tmp_path, gate=gate(success=0.0, collision=1.0)
    )

    assert [candidate["step"] for candidate in selection["candidates"]] == [
        500,
        400,
        300,
        200,
        100,
    ]
    assert selection["formal_selected"]["step"] == 500


def test_missing_model_is_reported_and_never_formally_selected(tmp_path: Path) -> None:
    missing_summary, missing_model = save_candidate(
        tmp_path,
        step=100,
        success_count=20,
        collision_count=0,
        create_model=False,
    )
    available_summary, _ = save_candidate(
        tmp_path,
        step=200,
        success_count=10,
        collision_count=10,
    )

    selection = build_checkpoint_selection(
        [missing_summary, available_summary], repository_root=tmp_path, gate=gate()
    )

    assert selection["selection_status"] == "no_candidate_passed_gate"
    assert selection["formal_selected"] is None
    assert selection["ranking_winner"]["step"] == 200
    assert (
        selection["candidates"][0]["model_path"]
        == missing_model.relative_to(tmp_path).as_posix()
    )
    assert selection["candidates"][0]["metric_gate_passed"] is True
    assert selection["candidates"][0]["gate_passed"] is False
    assert "model_file_missing" in selection["candidates"][0]["gate_failures"]


def test_final_model_can_be_mapped_through_metadata(tmp_path: Path) -> None:
    summary_path, expected_model = save_candidate(
        tmp_path, step=1234, kind="final", create_model=False
    )
    mapped_model = (
        tmp_path
        / "artifacts/models/webots/seed_11"
        / "ppo_webots_selected_step000001234.zip"
    )
    mapped_model.parent.mkdir(parents=True)
    mapped_model.write_bytes(b"mapped")
    metadata = tmp_path / "artifacts/models/webots/seed_11/run_metadata_step000001234.yaml"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        yaml.safe_dump({"final_model_steps": 1234, "final_model": str(mapped_model)}),
        encoding="utf-8",
    )
    bind_summary_to_model(
        summary_path,
        repository_root=tmp_path,
        model_path=mapped_model,
    )

    candidate = make_candidate_record(summary_path, repository_root=tmp_path, gate=gate())

    assert not expected_model.exists()
    assert candidate["model_path"] == mapped_model.relative_to(tmp_path).as_posix()
    assert candidate["model_resolution"].startswith("metadata:")
    assert candidate["model_sha256"] == sha256_file(mapped_model)


def test_variant_final_model_can_be_mapped_through_variant_metadata(
    tmp_path: Path,
) -> None:
    summary_path, expected_model = save_candidate(
        tmp_path,
        step=61440,
        kind="final",
        create_model=False,
        model_variant="curriculum_v2",
    )
    model_directory = tmp_path / "artifacts/models/webots/seed_11/curriculum_v2"
    mapped_model = model_directory / "ppo_webots_selected_step000061440.zip"
    mapped_model.parent.mkdir(parents=True, exist_ok=True)
    mapped_model.write_bytes(b"variant-mapped")
    metadata = model_directory / "run_metadata_step000061440.yaml"
    metadata.write_text(
        yaml.safe_dump(
            {
                "final_model_steps": 61440,
                "final_model": mapped_model.relative_to(tmp_path).as_posix(),
            }
        ),
        encoding="utf-8",
    )
    bind_summary_to_model(
        summary_path,
        repository_root=tmp_path,
        model_path=mapped_model,
    )

    candidate = make_candidate_record(
        summary_path,
        repository_root=tmp_path,
        gate=gate(model_variant="curriculum_v2"),
    )

    assert not expected_model.exists()
    assert candidate["model_variant"] == "curriculum_v2"
    assert candidate["model_path"] == mapped_model.relative_to(tmp_path).as_posix()
    assert candidate["model_resolution"] == "metadata:run_metadata_step000061440.yaml"
    assert candidate["model_sha256"] == sha256_file(mapped_model)


def test_metadata_cannot_map_final_model_outside_seed_directory(tmp_path: Path) -> None:
    summary_path, expected_model = save_candidate(
        tmp_path,
        step=1234,
        kind="final",
        create_model=False,
    )
    outside_model = tmp_path / "custom/final-model.zip"
    outside_model.parent.mkdir(parents=True)
    outside_model.write_bytes(b"outside")
    metadata = tmp_path / "artifacts/models/webots/seed_11/run_metadata_step000001234.yaml"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        yaml.safe_dump(
            {"final_model_steps": 1234, "final_model": str(outside_model)}
        ),
        encoding="utf-8",
    )

    candidate = make_candidate_record(
        summary_path,
        repository_root=tmp_path,
        gate=gate(),
    )

    assert candidate["model_path"] == expected_model.relative_to(tmp_path).as_posix()
    assert candidate["model_exists"] is False
    assert "model_file_missing" in candidate["gate_failures"]


def test_selection_rejects_summary_bound_to_different_variant(tmp_path: Path) -> None:
    summary_path, _ = save_candidate(
        tmp_path,
        step=40960,
        model_variant="curriculum_v2",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evaluation_provenance"]["model_variant"] = None
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="model_variant"):
        make_candidate_record(
            summary_path,
            repository_root=tmp_path,
            gate=gate(model_variant="curriculum_v2"),
        )


def test_selection_rejects_summary_model_hash_mismatch(tmp_path: Path) -> None:
    summary_path, _ = save_candidate(tmp_path, step=40960)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evaluation_provenance"]["model_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="model_sha256"):
        make_candidate_record(
            summary_path,
            repository_root=tmp_path,
            gate=gate(),
        )


def test_manifests_and_copy_alias_require_a_formal_selection(tmp_path: Path) -> None:
    passing_summary, source_model = save_candidate(
        tmp_path, step=123, success_count=18, collision_count=1
    )
    selection = build_checkpoint_selection(
        [passing_summary], repository_root=tmp_path, gate=gate()
    )
    alias = tmp_path / "aliases/selected.zip"

    json_path, yaml_path = write_checkpoint_selection(
        selection,
        output_directory=tmp_path / "selection",
        repository_root=tmp_path,
        copy_selected=True,
        alias_path=alias,
    )

    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    saved_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert saved_json == saved_yaml
    assert saved_json["selection_status"] == "formal_selected"
    assert saved_json["copy_selected"]["performed"] is True
    assert alias.read_bytes() == source_model.read_bytes()
    assert saved_json["copy_selected"]["alias_sha256"] == sha256_file(source_model)


def test_copy_request_does_not_create_alias_when_gate_fails(tmp_path: Path) -> None:
    summary_path, _ = save_candidate(
        tmp_path, step=123, success_count=10, collision_count=10
    )
    selection = build_checkpoint_selection(
        [summary_path], repository_root=tmp_path, gate=gate()
    )
    alias = tmp_path / "aliases/not-formal.zip"

    write_checkpoint_selection(
        selection,
        output_directory=tmp_path / "selection",
        repository_root=tmp_path,
        copy_selected=True,
        alias_path=alias,
    )

    assert not alias.exists()
    assert selection["formal_selected"] is None
    assert selection["copy_selected"]["reason"] == "no_gate_passing_formal_selection"


def test_checkpoint_selection_cli_status_fails_closed_without_formal_model() -> None:
    assert selection_exit_code({"formal_selected": {"model_path": "selected.zip"}}) == 0
    assert selection_exit_code({"formal_selected": None}) != 0
