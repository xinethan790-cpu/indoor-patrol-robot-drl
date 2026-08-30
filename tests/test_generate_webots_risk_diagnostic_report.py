from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/generate_webots_risk_diagnostic_report.py"
SPEC = importlib.util.spec_from_file_location("generate_webots_risk_diagnostic_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)

ANALYSIS_DIRECTORY = REPOSITORY_ROOT / "artifacts/results/webots_recovery_v1_risk_analysis_v1"
SENSITIVITY_DIRECTORY = (
    REPOSITORY_ROOT / "artifacts/results/webots_recovery_v1_lidar_origin_sensitivity_v1"
)
V1_REPORT = ANALYSIS_DIRECTORY / "risk_analysis_report.json"
METRICS_JSON = ANALYSIS_DIRECTORY / "risk_episode_metrics.json"
METRICS_CSV = ANALYSIS_DIRECTORY / "risk_episode_metrics.csv"
SENSITIVITY_REPORT = SENSITIVITY_DIRECTORY / "lidar_origin_sensitivity_report.json"
SENSITIVITY_EPISODES = SENSITIVITY_DIRECTORY / "lidar_origin_sensitivity_per_episode.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate(output: Path) -> dict[str, object]:
    return REPORT.generate_report(
        v1_report_path=V1_REPORT,
        metrics_json_path=METRICS_JSON,
        metrics_csv_path=METRICS_CSV,
        sensitivity_report_path=SENSITIVITY_REPORT,
        output_directory=output,
    )


def test_locked_diagnostic_evidence_generates_public_report_and_three_svgs(
    tmp_path: Path,
) -> None:
    before = {
        path: _sha256(path)
        for path in (
            V1_REPORT,
            METRICS_JSON,
            METRICS_CSV,
            SENSITIVITY_REPORT,
            SENSITIVITY_EPISODES,
        )
    }
    output = tmp_path / "reports/webots_risk_diagnostic"
    summary = _generate(output)

    assert set(path.name for path in output.iterdir()) == set(REPORT.OUTPUT_FILENAMES)
    assert summary["formal_v1_decision"] == "indeterminate"
    assert summary["sensitivity_formal"] is False
    assert summary["split"] == "diagnostic"
    assert summary["validation_or_test_access"] == "forbidden"
    assert summary["exemplar_seed"] == 20500
    assert all(summary["files"][name] == _sha256(output / name) for name in REPORT.OUTPUT_FILENAMES)
    assert before == {path: _sha256(path) for path in before}

    markdown = (output / "diagnostic_report.md").read_text(encoding="utf-8")
    assert "正式 v1\n> 机制判定是 **`indeterminate`" in markdown
    assert "**post-hoc / nonformal**" in markdown
    assert "**不替代 v1 正式判定**" in markdown
    assert "未访问 validation seeds 或 held-out test seeds" in markdown
    assert "成功 70/120（58.3%），碰撞 50/120（41.7%）" in markdown
    assert (
        "seed `20500` 是 v1 中被归为 `physical_imminent_dynamic` 的**最小 scenario seed**"
        in markdown
    )
    assert REPORT.LOCKED_V1_REPORT_SHA256 in markdown
    assert REPORT.LOCKED_METRICS_JSON_SHA256 in markdown
    assert REPORT.LOCKED_METRICS_CSV_SHA256 in markdown
    assert REPORT.LOCKED_SENSITIVITY_REPORT_SHA256 in markdown
    assert REPORT.LOCKED_SENSITIVITY_EPISODES_SHA256 in markdown
    assert "v1 的 28 个 ambiguous LiDAR-only 回合中" in markdown
    assert "25 个因 GPS/LiDAR 代理原点获得唯一归因" in markdown
    assert "另 3 个在两种原点下都未解析" in markdown
    assert "ambiguous 是 seed `20523`" in markdown
    assert "几何量落在 proxy tolerance band 内" in markdown
    assert "trajectory.py \\\n  --collection-report" in markdown
    assert str(REPOSITORY_ROOT) not in markdown
    for target in re.findall(r"\]\(([^)]+)\)", markdown):
        assert not Path(target).is_absolute()
        assert not re.match(r"^[A-Za-z]:", target)

    expected_svg_text = {
        "outcomes_by_axis.svg": ("29 (48.3%)", "41 (68.3%)"),
        "mechanism_evidence_v1_vs_sensitivity.svg": (
            "The sensitivity audit is post-hoc and nonformal",
            "timely=0",
        ),
        "risk_timeline_seed20500.svg": (
            "risk onset: step 52, 6.656 s",
            "LiDAR collision: step 61, 7.808 s",
        ),
    }
    for filename, markers in expected_svg_text.items():
        svg_path = output / filename
        root = ET.parse(svg_path).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        text = svg_path.read_text(encoding="utf-8")
        assert all(marker in text for marker in markers)


def test_generation_is_deterministic_when_refreshing_known_outputs(tmp_path: Path) -> None:
    output = tmp_path / "public"
    first = _generate(output)
    first_hashes = first["files"]
    second = _generate(output)
    assert second["files"] == first_hashes


def test_tampered_direct_input_fails_before_creating_outputs(tmp_path: Path) -> None:
    tampered = tmp_path / "risk_analysis_report.json"
    shutil.copy2(V1_REPORT, tampered)
    tampered.write_text(tampered.read_text(encoding="utf-8") + " ", encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(PermissionError, match="not the locked published artifact"):
        REPORT.generate_report(
            v1_report_path=tampered,
            metrics_json_path=METRICS_JSON,
            metrics_csv_path=METRICS_CSV,
            sensitivity_report_path=SENSITIVITY_REPORT,
            output_directory=output,
        )
    assert not output.exists()


def test_formal_and_nonformal_scope_validators_fail_closed() -> None:
    v1 = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    v1["mechanism_decision"]["decision"] = "dynamic_short_horizon_response_insufficient"
    with pytest.raises(PermissionError, match="must remain indeterminate"):
        REPORT._validate_v1_report(v1)

    sensitivity = json.loads(SENSITIVITY_REPORT.read_text(encoding="utf-8"))
    sensitivity["scope"]["formal_mechanism_inference"] = True
    with pytest.raises(PermissionError, match="nonformal diagnostic audit"):
        REPORT._validate_sensitivity_report(sensitivity)


def test_metrics_csv_json_crosscheck_rejects_route_axis_drift(tmp_path: Path) -> None:
    rows: list[dict[str, str]]
    with METRICS_CSV.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows[0]["route_axis"] = "vertical"
    changed_csv = tmp_path / "changed.csv"
    with changed_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="CSV/JSON mismatch at row 0"):
        REPORT._load_and_crosscheck_metrics(METRICS_JSON, changed_csv)


def test_exemplar_is_fixed_lowest_v1_physical_seed() -> None:
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    physical = [
        value
        for value in metrics
        if value["mechanism_classification"] == "physical_imminent_dynamic"
    ]
    assert min(value["scenario_seed"] for value in physical) == REPORT.EXEMPLAR_SEED == 20500
    exemplar = next(value for value in physical if value["scenario_seed"] == 20500)
    assert exemplar["risk_onset_step"] == 52
    assert exemplar["risk_onset_to_terminal_s"] == pytest.approx(1.152)
