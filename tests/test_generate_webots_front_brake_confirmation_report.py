from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/generate_webots_front_brake_confirmation_report.py"
SPEC = importlib.util.spec_from_file_location(
    "generate_webots_front_brake_confirmation_report", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)

V1_REPORT = (
    REPOSITORY_ROOT
    / "artifacts/results/webots_front_brake_paired_diagnostic_v1"
    / "paired_diagnostic_report.json"
)
AUDIT_REPORT = (
    REPOSITORY_ROOT
    / "artifacts/results/webots_front_brake_v1_manipulation_reaudit_v1"
    / "manipulation_boundary_reaudit.json"
)
V2_REPORT = (
    REPOSITORY_ROOT
    / "artifacts/results/webots_front_brake_paired_confirmation_v2"
    / "paired_diagnostic_report.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_inputs() -> list[Path]:
    paths = [
        *(V1_REPORT.parent / name for name in REPORT.LOCKED_V1_FILES),
        *(AUDIT_REPORT.parent / name for name in REPORT.LOCKED_AUDIT_FILES),
        *(V2_REPORT.parent / name for name in REPORT.LOCKED_V2_FILES),
        SCRIPT_PATH,
    ]
    paths.extend(
        REPOSITORY_ROOT / value[0]
        for arm in REPORT.LOCKED_V2_RAW_FILES.values()
        for value in arm.values()
    )
    return paths


def _generate(output: Path, *, force: bool = False) -> dict[str, object]:
    return REPORT.generate_report(
        repository_root=REPOSITORY_ROOT,
        v1_report_path=V1_REPORT,
        audit_report_path=AUDIT_REPORT,
        v2_report_path=V2_REPORT,
        output_directory=output,
        generator_path=SCRIPT_PATH,
        force=force,
    )


def test_locked_evidence_generates_raw_derived_public_report_and_three_svgs(
    tmp_path: Path,
) -> None:
    before = {path: _sha256(path) for path in _all_inputs()}
    output = tmp_path / "webots_front_brake_confirmation"
    summary = _generate(output)

    assert {path.name for path in output.iterdir()} == set(REPORT.OUTPUT_FILENAMES)
    assert summary["v1_status"] == "invalid_manipulation"
    assert summary["audit_promotes_v1"] is False
    assert summary["v2_status"] == "passed"
    assert summary["split"] == "diagnostic"
    assert summary["validation_or_test_access"] == "forbidden"
    assert summary["pooling"] == "forbidden"
    assert (
        summary["generator_sha256"]
        == summary["input_snapshot"]["scripts/generate_webots_front_brake_confirmation_report.py"]
    )
    metrics = summary["metrics"]
    assert metrics["outcomes"]["reference"] == {"success": 66, "collision": 54}
    assert metrics["outcomes"]["candidate"] == {"success": 95, "collision": 25}
    assert metrics["transitions"] == {
        "collision_to_collision": 19,
        "collision_to_success": 35,
        "success_to_collision": 6,
        "success_to_success": 60,
    }
    assert metrics["collision_relative_reduction"] == pytest.approx(0.5370370370370371)
    assert metrics["mcnemar_p"] == pytest.approx(2.436799150018487e-06)
    assert metrics["common_success"] == 60
    assert metrics["median_completion_ratio"] == pytest.approx(1.072172619047619)
    assert metrics["p90_completion_ratio"] == pytest.approx(1.1896752519596865)
    assert metrics["candidate_shield"] == {
        "total_steps": 12741,
        "active_steps": 3740,
        "active_episodes": 108,
        "braking_steps": 3740,
        "cap_binding_steps": 3740,
        "angular_unchanged_steps": 3740,
    }
    assert all(summary["files"][name] == _sha256(output / name) for name in REPORT.OUTPUT_FILENAMES)
    assert before == {path: _sha256(path) for path in before}

    for path in output.iterdir():
        assert b"\r\n" not in path.read_bytes()
    markdown = (output / "confirmation_report.md").read_text(encoding="utf-8")
    assert "v1 结果仍是 **`invalid_manipulation`**" in markdown
    assert "**没有改写 v1、也没有把 v1 的效果结果升级为有效结论**" in markdown
    assert "seed `25737`、step `21`" in markdown
    assert "成功 **66/120（55.00%）**，碰撞 **54/120（45.00%）**" in markdown
    assert "成功 **95/120（79.17%）**，碰撞 **25/120（20.83%）**" in markdown
    assert "碰撞数相对下降 **53.70%**" in markdown
    assert "**35 个 rescue**" in markdown
    assert "**6 个 regression**" in markdown
    assert "`p = 2.43679915e-06`" in markdown
    assert "**1.0722**" in markdown
    assert "**1.1897**（约增加 18.97%）" in markdown
    assert "**3740/12741**" in markdown
    assert "validation、held-out test 或跨场景泛化结论" in markdown
    assert "没有合并 v1/v2" in markdown
    assert REPORT.LOCKED_V1_FILES["paired_diagnostic_report.json"] in markdown
    assert REPORT.LOCKED_AUDIT_FILES["manipulation_boundary_reaudit.json"] in markdown
    assert REPORT.LOCKED_V2_FILES["paired_diagnostic_report.json"] in markdown
    assert _sha256(SCRIPT_PATH) in markdown
    assert str(REPOSITORY_ROOT) not in markdown
    for target in re.findall(r"\]\(([^)]+)\)", markdown):
        assert not Path(target).is_absolute()
        assert not re.match(r"^[A-Za-z]:", target)

    expected_svg_markers = {
        "outcome_rates.svg": ("66/120 (55.00%)", "95/120 (79.17%)", "-53.70%"),
        "paired_transitions.svg": ("Rescue: collision -&gt; success", ">35<", "2.4368e-6"),
        "safety_tradeoff.svg": (
            "53.70%",
            "1.0722",
            "p90 1.1897 (+18.97%)",
            "3740/12741",
            "Diagnostic confirmation only",
        ),
    }
    for filename, markers in expected_svg_markers.items():
        svg = output / filename
        root = ET.parse(svg).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        text = svg.read_text(encoding="utf-8")
        assert all(marker in text for marker in markers)

    sidecar = (output / "confirmation_report.sha256").read_text(encoding="utf-8")
    for filename in REPORT.OUTPUT_FILENAMES[:-1]:
        assert f"{_sha256(output / filename)}  {filename}\n" in sidecar
    assert f"{_sha256(SCRIPT_PATH)}  scripts/{SCRIPT_PATH.name}\n" in sidecar
    for arm in REPORT.LOCKED_V2_RAW_FILES.values():
        for relative, digest in arm.values():
            assert f"{digest}  {relative}\n" in sidecar


def test_existing_output_requires_force_and_force_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "webots_front_brake_confirmation"
    first = _generate(output)
    with pytest.raises(FileExistsError, match="Output directory already exists"):
        _generate(output)
    refreshed = _generate(output, force=True)
    assert refreshed["files"] == first["files"]

    (output / "unrelated.txt").write_text("do not delete", encoding="utf-8")
    with pytest.raises(PermissionError, match="exact known manifest"):
        _generate(output, force=True)
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "do not delete"


def test_publish_rolls_back_when_post_publish_input_check_fails(tmp_path: Path) -> None:
    output = tmp_path / "webots_front_brake_confirmation"
    _generate(output)
    original = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(RuntimeError, match="simulated concurrent change"):
        REPORT._publish_outputs(
            output_directory=output,
            outputs={name: "replacement\n" for name in REPORT.OUTPUT_FILENAMES},
            force=True,
            post_publish_check=lambda: (_ for _ in ()).throw(
                RuntimeError("simulated concurrent change")
            ),
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == original
    assert not (output.parent / f".{output.name}.staging").exists()
    assert not (output.parent / f".{output.name}.backup").exists()


def test_tampered_direct_report_and_raw_artifact_fail_closed(tmp_path: Path) -> None:
    tampered_report = tmp_path / "paired_diagnostic_report.json"
    shutil.copy2(V2_REPORT, tampered_report)
    tampered_report.write_bytes(tampered_report.read_bytes() + b" ")
    output = tmp_path / "must-not-exist"
    with pytest.raises(PermissionError, match="locked SHA-256"):
        REPORT.generate_report(
            repository_root=REPOSITORY_ROOT,
            v1_report_path=V1_REPORT,
            audit_report_path=AUDIT_REPORT,
            v2_report_path=tampered_report,
            output_directory=output,
            generator_path=SCRIPT_PATH,
        )
    assert not output.exists()

    raw_relative, raw_hash = REPORT.LOCKED_V2_RAW_FILES["candidate"]["episodes_csv"]
    tampered_raw = tmp_path / "episodes.csv"
    shutil.copy2(REPOSITORY_ROOT / raw_relative, tampered_raw)
    tampered_raw.write_bytes(tampered_raw.read_bytes() + b"\n")
    with pytest.raises(PermissionError, match="locked SHA-256"):
        REPORT._require_file_sha(tampered_raw, raw_hash, "tampered v2 raw episodes")


def test_scope_validators_reject_v1_promotion_pooling_and_test_access() -> None:
    v1 = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    v1["status"] = "passed"
    with pytest.raises(PermissionError, match="remain invalid_manipulation"):
        REPORT._validate_v1(v1)

    audit = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    audit["checker_conclusion"][
        "primary_efficacy_result_is_not_promoted_by_this_post_hoc_audit"
    ] = False
    with pytest.raises(PermissionError, match="diagnostic-only boundary"):
        REPORT._validate_audit(audit)

    v2 = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    v2["prior_evidence"]["restrictions"]["v1_v2_pooled_fallback"] = "allowed"
    with pytest.raises(PermissionError, match="pooling"):
        REPORT._validate_v2(v2)

    v2 = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    v2["test_split_was_run"] = True
    with pytest.raises(PermissionError, match="diagnostic-only"):
        REPORT._validate_v2(v2)


def test_exact_mcnemar_and_seed_boundaries_are_fixed() -> None:
    assert REPORT.EXPECTED_V1_SEEDS == tuple(range(25728, 25848))
    assert REPORT.EXPECTED_V2_SEEDS == tuple(range(54775, 54895))
    assert set(REPORT.EXPECTED_V1_SEEDS).isdisjoint(REPORT.EXPECTED_V2_SEEDS)
    assert REPORT._exact_one_sided_mcnemar_p(35, 6) == pytest.approx(2.436799150018487e-06)
    assert REPORT._linear_quantile([1.0, 2.0, 4.0], 0.9) == pytest.approx(3.6)
