from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/audit_webots_lidar_origin_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("audit_webots_lidar_origin_sensitivity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

COLLECTION_REPORT = (
    REPOSITORY_ROOT
    / "artifacts/results/webots_recovery_v1_risk_trajectory_v1/collection_report.json"
)
V1_REPORT = (
    REPOSITORY_ROOT
    / "artifacts/results/webots_recovery_v1_risk_analysis_v1/risk_analysis_report.json"
)
V1_METRICS = (
    REPOSITORY_ROOT
    / "artifacts/results/webots_recovery_v1_risk_analysis_v1/risk_episode_metrics.json"
)


def _run(tmp_path: Path) -> dict[str, object]:
    return AUDIT.audit_lidar_origin_sensitivity(
        collection_report_path=COLLECTION_REPORT,
        v1_risk_analysis_report_path=V1_REPORT,
        output_directory=tmp_path / "audit",
        repository_root=REPOSITORY_ROOT,
    )


def test_locked_actual_audit_captures_origin_offset_and_25_of_26_improvements(
    tmp_path: Path,
) -> None:
    report = _run(tmp_path)
    counts = report["attribution_response_counts"]
    corrected = report["corrected_physical_candidate_set"]
    assert counts["lidar_only_terminal_episodes"] == 47
    assert counts["root_unique_attribution"] == 19
    assert counts["gps_proxy_unique_attribution"] == 44
    assert counts["root_unresolved_with_circle_intersection_candidate"] == 26
    assert counts["resolved_by_gps_lidar_origin_proxy"] == 25
    assert counts["unresolved_under_both_origins"] == 3
    assert corrected["physical_lidar_only"] == 43
    assert corrected["ambiguous_lidar_only"] == 4
    assert corrected["contact_confirmed"] == 3
    assert corrected["total"] == 46
    assert corrected["reaction_evaluable"] == 46
    assert corrected["reaction_indeterminate"] == 0
    assert corrected["timely_deceleration_response"] == 0
    assert corrected["short_horizon_response_insufficient"] == 46
    assert report["post_hoc_sensitivity_decision"]["outcome"] == (
        "dynamic_short_horizon_response_insufficient_under_sensitivity_only_rule"
    )
    assert report["scope"]["formal_mechanism_inference"] is False
    assert report["scope"]["does_not_replace_v1"] is True
    assert report["scope"]["validation_or_test_access"] == "forbidden"
    rows = json.loads(
        (tmp_path / "audit/lidar_origin_sensitivity_per_episode.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 47
    assert all(value["observed_origin_offset_m"] > 0.02 for value in rows)
    seed_20523 = next(value for value in rows if value["scenario_seed"] == 20523)
    assert seed_20523["gps_unique_attributed_obstacle_index"] == 0
    assert seed_20523["post_hoc_sensitivity_mechanism_classification"] == "ambiguous"
    assert seed_20523["post_hoc_sensitivity_mechanism_reason"] == (
        "geometry_falls_inside_proxy_tolerance_band"
    )
    assert seed_20523["included_in_corrected_physical_candidate_set"] is False
    unresolved = [
        value
        for value in rows
        if value["post_hoc_sensitivity_classification"] == "unresolved_under_both_origins"
    ]
    assert {value["scenario_seed"] for value in unresolved} == {20503, 20598, 20607}
    report_path = tmp_path / "audit/lidar_origin_sensitivity_report.json"
    sidecar = report_path.with_suffix(".sha256").read_text(encoding="utf-8")
    assert sidecar == f"{AUDIT._sha256_file(report_path)}  {report_path.name}\n"
    assert all(
        not Path(binding["path"]).is_absolute()
        for binding in report["inputs"].values()
    )
    assert not Path(report["artifacts"]["per_episode_json"]["path"]).is_absolute()
    assert not Path(report["artifacts"]["per_episode_csv"]["path"]).is_absolute()


@pytest.mark.parametrize(
    ("yaw", "bearing"),
    [(0.0, 0.0), (math.pi / 2.0, 0.0), (0.0, 0.35), (0.0, -0.35)],
)
def test_synthetic_gps_proxy_resolves_fixed_origin_offset(yaw: float, bearing: float) -> None:
    direction = np.asarray([math.cos(yaw + bearing), math.sin(yaw + bearing)])
    gps = np.asarray([-0.03 * math.cos(yaw), -0.03 * math.sin(yaw)])
    obstacle = gps + direction * 0.32
    sample = {
        "post_robot_root_xy": [0.0, 0.0],
        "post_robot_gps_xy": gps.tolist(),
        "post_robot_yaw": yaw,
        "post_lidar_argmin": {"range_m": 0.12, "bearing_rad": bearing},
        "post_obstacle_xy": [obstacle.tolist()],
    }
    root_values = AUDIT._intersection_residuals(sample, "post_robot_root_xy")
    gps_values = AUDIT._intersection_residuals(sample, "post_robot_gps_xy")
    assert AUDIT._unique_match(gps_values) == 0
    assert AUDIT._unique_match(root_values) is None
    if yaw == 0.0 and bearing == 0.0:
        assert obstacle.tolist() == pytest.approx([0.29, 0.0])
        assert root_values[0]["absolute_residual_m"] == pytest.approx(0.03)


def test_origin_response_classifier_preserves_unresolved_and_detects_offset_response() -> None:
    assert AUDIT.classify_origin_response(None, 0) == ("resolved_by_gps_lidar_origin_proxy")
    assert AUDIT.classify_origin_response(None, None) == ("unresolved_under_both_origins")
    assert AUDIT.classify_origin_response(0, 0) == "attribution_retained"
    assert AUDIT.classify_origin_response(0, None) == ("attribution_lost_under_gps_proxy")


def test_collection_and_v1_report_sha_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection_copy = tmp_path / "collection_report.json"
    shutil.copy2(COLLECTION_REPORT, collection_copy)
    collection_copy.write_text(collection_copy.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(PermissionError, match="collection_report.json is not the locked"):
        AUDIT.audit_lidar_origin_sensitivity(
            collection_report_path=collection_copy,
            v1_risk_analysis_report_path=V1_REPORT,
            output_directory=tmp_path / "out-collection",
            repository_root=REPOSITORY_ROOT,
        )

    report_copy = tmp_path / "risk_analysis_report.json"
    shutil.copy2(V1_REPORT, report_copy)
    report_copy.write_text(report_copy.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(PermissionError, match="v1 risk analysis report is not the locked"):
        AUDIT.audit_lidar_origin_sensitivity(
            collection_report_path=COLLECTION_REPORT,
            v1_risk_analysis_report_path=report_copy,
            output_directory=tmp_path / "out-v1",
            repository_root=REPOSITORY_ROOT,
        )


def test_v1_report_schema_and_artifact_binding_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked = AUDIT.v1.load_locked_collection(COLLECTION_REPORT, repository_root=REPOSITORY_ROOT)
    report = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    report.pop("scope")
    schema_path = tmp_path / "schema/risk_analysis_report.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(AUDIT, "V1_RISK_ANALYSIS_REPORT_SHA256", AUDIT._sha256_file(schema_path))
    with pytest.raises(ValueError, match="schema mismatch"):
        AUDIT._load_v1_report(schema_path, locked_collection=locked)

    report = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    report["artifacts"]["risk_episode_metrics_json"]["sha256"] = "0" * 64
    binding_path = tmp_path / "binding/risk_analysis_report.json"
    binding_path.parent.mkdir(parents=True)
    binding_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(AUDIT, "V1_RISK_ANALYSIS_REPORT_SHA256", AUDIT._sha256_file(binding_path))
    with pytest.raises(PermissionError, match="not the locked v1 artifact"):
        AUDIT._load_v1_report(binding_path, locked_collection=locked)


def test_missing_absolute_v1_artifact_path_uses_same_directory_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    portable = tmp_path / "portable"
    portable.mkdir()
    metrics_json = portable / "risk_episode_metrics.json"
    metrics_csv = portable / "risk_episode_metrics.csv"
    shutil.copy2(V1_METRICS, metrics_json)
    shutil.copy2(
        REPOSITORY_ROOT
        / "artifacts/results/webots_recovery_v1_risk_analysis_v1/risk_episode_metrics.csv",
        metrics_csv,
    )
    report = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    report["artifacts"]["risk_episode_metrics_json"]["path"] = (
        "Z:/old-machine/risk_episode_metrics.json"
    )
    report["artifacts"]["risk_episode_metrics_csv"]["path"] = (
        "Z:/old-machine/risk_episode_metrics.csv"
    )
    report_path = portable / "risk_analysis_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(AUDIT, "V1_RISK_ANALYSIS_REPORT_SHA256", AUDIT._sha256_file(report_path))
    locked = AUDIT.v1.load_locked_collection(COLLECTION_REPORT, repository_root=REPOSITORY_ROOT)
    _report, actual_json, actual_csv, _sha = AUDIT._load_v1_report(
        report_path, locked_collection=locked
    )
    assert actual_json == metrics_json
    assert actual_csv == metrics_csv


def test_metric_seed_and_schema_fail_closed(tmp_path: Path) -> None:
    metrics = json.loads(V1_METRICS.read_text(encoding="utf-8"))
    metrics[0]["scenario_seed"] = 5000
    seed_path = tmp_path / "bad-seed.json"
    seed_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(PermissionError, match="exact ordered diagnostic seeds"):
        AUDIT._load_metrics(seed_path)

    metrics = json.loads(V1_METRICS.read_text(encoding="utf-8"))
    metrics[0].pop("terminal_collision_source")
    schema_path = tmp_path / "bad-schema.json"
    schema_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        AUDIT._load_metrics(schema_path)


def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        AUDIT.audit_lidar_origin_sensitivity(
            collection_report_path=COLLECTION_REPORT,
            v1_risk_analysis_report_path=V1_REPORT,
            output_directory=output,
            repository_root=REPOSITORY_ROOT,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
