from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from indoor_patrol_rl.evaluation.benchmark import write_benchmark_outputs
from indoor_patrol_rl.evaluation.formal_protocol import (
    file_sha256,
    write_final_lock,
)
from indoor_patrol_rl.evaluation.formal_reporting import (
    FORMAL_CONDITIONS,
    FORMAL_SCENARIO_SEEDS,
    FORMAL_SERIES,
    FormalResultInput,
    build_formal_report,
    write_formal_report,
)
from indoor_patrol_rl.evaluation.metrics import EpisodeRecord
from indoor_patrol_rl.evaluation.webots_formal_config import (
    load_webots_formal_config,
)
from tests.test_formal_protocol import _build_final_lock


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ppo_seed(series: str) -> int | None:
    return int(series.removeprefix("ppo_seed")) if series.startswith("ppo_seed") else None


def _success_count(series: str) -> int:
    if series == "reactive":
        return 14
    if series == "dwa":
        return 10
    return {11: 11, 22: 12, 33: 13}[_ppo_seed(series)]


def _episode(
    *,
    condition_id: str,
    series: str,
    scenario_seed: int,
) -> EpisodeRecord:
    condition_index = FORMAL_CONDITIONS.index(condition_id)
    seed_index = FORMAL_SCENARIO_SEEDS.index(scenario_seed)
    success_count = _success_count(series)
    if seed_index < success_count:
        outcome = "success"
        is_success, collision, timeout = True, False, False
    elif seed_index < success_count + 3:
        outcome = "collision"
        is_success, collision, timeout = False, True, False
    else:
        outcome = "timeout"
        is_success, collision, timeout = False, False, True
    reference_length = 2.0 + condition_index + 0.001 * seed_index
    path_length = reference_length * (1.0 + 0.01 * condition_index)
    elapsed = 8.0 + 0.1 * seed_index + condition_index
    return EpisodeRecord(
        schema_version="1.0",
        episode_id=f"test-{condition_id}-{scenario_seed}-{series}",
        split="test",
        condition_id=condition_id,
        scenario_seed=scenario_seed,
        planner=series,
        outcome=outcome,
        is_success=is_success,
        collision=collision,
        timeout=timeout,
        step_count=100 + seed_index,
        elapsed_sim_s=elapsed,
        completion_time_s=elapsed if is_success else None,
        episode_return=20.0 if is_success else -20.0,
        path_length_m=path_length,
        reference_path_length_m=reference_length,
        path_length_ratio=path_length / reference_length if is_success else None,
        minimum_clearance_m=0.2 + 0.001 * seed_index,
        near_collision_ratio=0.05 + 0.001 * seed_index,
        action_delta_sq_mean=0.01 + 0.001 * seed_index,
        inference_latency_p50_ms=0.10 + 0.001 * seed_index,
        inference_latency_p95_ms=0.20 + 0.001 * seed_index,
        error_type=None,
    )


def _make_formal_inputs(
    root: Path,
) -> tuple[
    list[FormalResultInput],
    dict[tuple[str, str], FormalResultInput],
    Path,
]:
    project = Path(__file__).absolute().parents[1]
    final_lock = _build_final_lock(
        root,
        campaign_id="reporting-test-campaign",
    )
    config_path = root / final_lock.evaluation_config_path
    config_path.write_text(
        (project / "configs/eval/webots_formal.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = load_webots_formal_config(config_path)
    final_lock = replace(
        final_lock,
        evaluation_config_sha256=file_sha256(config_path),
    )
    world_path = root / final_lock.world_path
    model_bindings = {
        manifest.training_seed: (manifest.model_path, manifest.model_sha256)
        for manifest in final_lock.model_manifests
    }
    final_lock_path = root / "artifacts/results/formal_locks/reporting-test-campaign.json"
    write_final_lock(final_lock_path, final_lock, repository_root=root)

    inputs: list[FormalResultInput] = []
    by_key: dict[tuple[str, str], FormalResultInput] = {}
    for condition_id in FORMAL_CONDITIONS:
        for series in FORMAL_SERIES:
            records = [
                _episode(
                    condition_id=condition_id,
                    series=series,
                    scenario_seed=scenario_seed,
                )
                for scenario_seed in FORMAL_SCENARIO_SEEDS
            ]
            provenance: dict[str, object] = {
                "schema_version": 1,
                "split": "test",
                "condition_id": condition_id,
                "planner": "ppo" if series.startswith("ppo_seed") else series,
                "planner_label": series,
                "training_seed": _ppo_seed(series),
                "model_path": None,
                "model_sha256": None,
                "scenario_seed_start": 10000,
                "scenario_seed_count": 20,
                "scenario_seeds": list(FORMAL_SCENARIO_SEEDS),
                "formal_campaign_id": final_lock.campaign_id,
                "final_lock_sha256": final_lock.lock_sha256,
                "test_consumption_key": final_lock.test_consumption_key,
                "source_revision": final_lock.source_revision,
                "source_bundle_sha256": final_lock.source_bundle_sha256,
                "evaluator_sha256": final_lock.evaluator_sha256,
                "controller_sha256": final_lock.controller_sha256,
                "experiment_id": config.experiment_id,
                "webots_version": config.webots_version,
                "formal_config_path": final_lock.evaluation_config_path,
                "formal_config_sha256": config.source_sha256,
                "evaluation_config_sha256": config.source_sha256,
                "world_path": config.world,
                "world_sha256": file_sha256(world_path),
                "condition_semantic_sha256": config.condition(condition_id).semantic_sha256,
            }
            training_seed = _ppo_seed(series)
            if training_seed is not None:
                model_path, model_sha256 = model_bindings[training_seed]
                provenance.update(
                    {
                        "training_seed": training_seed,
                        "model_path": model_path,
                        "model_sha256": model_sha256,
                        "model_variant": None,
                    }
                )
            result_dir = root / "results" / condition_id / series
            write_benchmark_outputs(result_dir, records, provenance=provenance)
            episodes_path = result_dir / "episodes.csv"
            summary_path = result_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["episodes_sha256"] = _sha256(episodes_path)
            summary["evaluation_provenance"]["episodes_csv_sha256"] = _sha256(episodes_path)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            specification = FormalResultInput(episodes_path, summary_path)
            inputs.append(specification)
            by_key[(condition_id, series)] = specification
    return inputs, by_key, final_lock_path


def _replace_declared_episodes_hash(specification: FormalResultInput) -> None:
    summary = json.loads(specification.summary_path.read_text(encoding="utf-8"))
    digest = _sha256(specification.episodes_path)
    summary["episodes_sha256"] = digest
    summary["evaluation_provenance"]["episodes_csv_sha256"] = digest
    specification.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def test_complete_matrix_reports_group_and_cross_training_seed_statistics(
    tmp_path: Path,
) -> None:
    inputs, _, final_lock_path = _make_formal_inputs(tmp_path)

    report = build_formal_report(
        inputs,
        repository_root=tmp_path,
        final_lock_path=final_lock_path,
    )

    assert report["matrix_validation"] == {
        "status": "passed",
        "input_pair_count": 20,
        "episode_count": 400,
        "unique_episode_ids": 400,
        "unique_episode_keys": 400,
        "paired_scenario_count": 80,
        "series_per_scenario": 5,
    }
    locked = json.loads(final_lock_path.read_text(encoding="utf-8"))
    provenance = report["formal_provenance"]
    assert provenance["campaign_id"] == "reporting-test-campaign"
    assert provenance["final_lock_path"] == (
        "artifacts/results/formal_locks/reporting-test-campaign.json"
    )
    assert provenance["final_lock_sha256"] == locked["lock_sha256"]
    for name in (
        "test_consumption_key",
        "source_revision",
        "source_bundle_sha256",
        "evaluation_config_path",
        "evaluation_config_sha256",
        "world_path",
        "world_sha256",
        "evaluator_path",
        "evaluator_sha256",
        "controller_path",
        "controller_sha256",
    ):
        assert provenance[name] == locked[name]
    config = load_webots_formal_config(tmp_path / locked["evaluation_config_path"])
    assert provenance["condition_semantic_sha256"] == {
        condition: config.condition(condition).semantic_sha256 for condition in FORMAL_CONDITIONS
    }
    assert len(report["input_artifacts"]) == 20
    assert len(report["by_series_condition"]) == 20
    assert len(report["by_method_condition"]) == 12
    assert len(report["ppo_models"]) == 3
    method_totals = {row["method"]: row for row in report["by_method"]}
    assert method_totals["reactive"]["episodes"] == 80
    assert method_totals["dwa"]["episodes"] == 80
    assert method_totals["ppo"]["episodes"] == 240

    g0 = next(row for row in report["ppo_cross_training_seed"] if row["condition_id"] == "G0")
    success = g0["statistics"]["success_rate"]
    assert success["n"] == 3
    assert success["values_by_training_seed"] == {
        "11": 0.55,
        "22": 0.60,
        "33": 0.65,
    }
    assert success["mean"] == pytest.approx(0.60)
    assert success["sample_sd"] == pytest.approx(0.05)
    expected_half_width = 4.302652729911275 * 0.05 / math.sqrt(3)
    assert success["ci95"]["half_width"] == pytest.approx(expected_half_width)
    assert success["ci95"]["lower"] == pytest.approx(0.60 - expected_half_width)
    assert success["ci95"]["upper"] == pytest.approx(0.60 + expected_half_width)

    output = write_formal_report(report, tmp_path / "formal/summary.json")
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_missing_formal_result_pair_is_rejected(tmp_path: Path) -> None:
    inputs, _, final_lock_path = _make_formal_inputs(tmp_path)

    with pytest.raises(ValueError, match="input matrix mismatch"):
        build_formal_report(
            inputs[:-1],
            repository_root=tmp_path,
            final_lock_path=final_lock_path,
        )


def test_duplicate_formal_result_pair_is_rejected(tmp_path: Path) -> None:
    inputs, _, final_lock_path = _make_formal_inputs(tmp_path)

    with pytest.raises(ValueError, match="Duplicate formal result inputs"):
        build_formal_report(
            [*inputs, inputs[0]],
            repository_root=tmp_path,
            final_lock_path=final_lock_path,
        )


def test_declared_episodes_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    inputs, by_key, final_lock_path = _make_formal_inputs(tmp_path)
    specification = by_key[("G0", "reactive")]
    summary = json.loads(specification.summary_path.read_text(encoding="utf-8"))
    summary["episodes_sha256"] = "0" * 64
    specification.summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="episodes.csv SHA-256 mismatch"):
        build_formal_report(
            inputs,
            repository_root=tmp_path,
            final_lock_path=final_lock_path,
        )


def test_ppo_model_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    inputs, _, final_lock_path = _make_formal_inputs(tmp_path)
    locked = json.loads(final_lock_path.read_text(encoding="utf-8"))
    model = tmp_path / locked["model_manifests"][0]["model_path"]
    model.write_bytes(b"changed-after-evaluation")

    with pytest.raises(ValueError, match="formal model SHA-256 mismatch"):
        build_formal_report(
            inputs,
            repository_root=tmp_path,
            final_lock_path=final_lock_path,
        )


def test_summary_counts_are_recomputed_from_csv(tmp_path: Path) -> None:
    inputs, by_key, final_lock_path = _make_formal_inputs(tmp_path)
    specification = by_key[("G1", "dwa")]
    summary = json.loads(specification.summary_path.read_text(encoding="utf-8"))
    summary["by_planner"][0]["success_count"] += 1
    specification.summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="success_count.*recomputed from CSV"):
        build_formal_report(
            inputs,
            repository_root=tmp_path,
            final_lock_path=final_lock_path,
        )


def test_paired_planners_must_share_reference_path(tmp_path: Path) -> None:
    inputs, by_key, final_lock_path = _make_formal_inputs(tmp_path)
    specification = by_key[("G2", "dwa")]
    with specification.episodes_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows[-1]["reference_path_length_m"] = str(float(rows[-1]["reference_path_length_m"]) + 0.5)
    with specification.episodes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _replace_declared_episodes_hash(specification)

    with pytest.raises(ValueError, match="do not share the same reference path"):
        build_formal_report(
            inputs,
            repository_root=tmp_path,
            final_lock_path=final_lock_path,
        )


def test_ppo_training_seed_must_match_planner_label(tmp_path: Path) -> None:
    inputs, by_key, final_lock_path = _make_formal_inputs(tmp_path)
    specification = by_key[("G3", "ppo_seed22")]
    summary = json.loads(specification.summary_path.read_text(encoding="utf-8"))
    summary["evaluation_provenance"]["training_seed"] = 11
    specification.summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="training_seed disagree"):
        build_formal_report(
            inputs,
            repository_root=tmp_path,
            final_lock_path=final_lock_path,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("formal_campaign_id", "different-campaign"),
        ("final_lock_sha256", "e" * 64),
        ("evaluation_config_sha256", "d" * 64),
        ("world_sha256", "f" * 64),
        ("source_revision", "b" * 40),
        ("condition_semantic_sha256", "c" * 64),
    ],
)
def test_mixed_formal_provenance_input_is_rejected(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    inputs, by_key, final_lock_path = _make_formal_inputs(tmp_path)
    specification = by_key[("G1", "reactive")]
    summary = json.loads(specification.summary_path.read_text(encoding="utf-8"))
    summary["evaluation_provenance"][field] = replacement
    specification.summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        build_formal_report(
            inputs,
            repository_root=tmp_path,
            final_lock_path=final_lock_path,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("path_length_ratio", "9.0"),
        ("completion_time_s", "999.0"),
    ],
)
def test_tampered_per_episode_derived_metric_is_rejected(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    inputs, by_key, final_lock_path = _make_formal_inputs(tmp_path)
    specification = by_key[("G0", "reactive")]
    with specification.episodes_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    assert rows[0]["is_success"] == "True"
    rows[0][field] = replacement
    with specification.episodes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _replace_declared_episodes_hash(specification)

    with pytest.raises(ValueError, match=field):
        build_formal_report(
            inputs,
            repository_root=tmp_path,
            final_lock_path=final_lock_path,
        )
