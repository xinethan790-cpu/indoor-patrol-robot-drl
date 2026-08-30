from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/audit_front_brake_manipulation_boundary.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_front_brake_manipulation_boundary", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

PAIRED_REPORT = (
    REPOSITORY_ROOT
    / "artifacts/results/webots_front_brake_paired_diagnostic_v1/paired_diagnostic_report.json"
)


def _run(tmp_path: Path) -> dict[str, object]:
    return AUDIT.audit_front_brake_manipulation(
        paired_report_path=PAIRED_REPORT,
        output_directory=tmp_path / "audit",
        repository_root=REPOSITORY_ROOT,
    )


def test_locked_actual_artifacts_recover_all_3197_operational_steps(tmp_path: Path) -> None:
    input_paths = [
        PAIRED_REPORT.parent / value[0] for value in AUDIT.ORIGINAL_REPORT_ARTIFACTS.values()
    ]
    input_paths.append(REPOSITORY_ROOT / AUDIT.MODEL_PATH)
    for artifacts in AUDIT.ARM_ARTIFACTS.values():
        input_paths.extend(REPOSITORY_ROOT / value[0] for value in artifacts.values())
    before = {path: AUDIT._sha256_file(path) for path in input_paths}

    report = _run(tmp_path)

    assert report["status"] == "passed"
    assert report["original_report_facts"]["status"] == "invalid_manipulation"
    assert report["original_report_facts"]["primary_efficacy_gate_passed"] is True
    assert report["original_report_facts"]["reported_counts"] == {
        "active_step_count": 3197,
        "braking_step_count": 3197,
        "speed_cap_binding_step_count": 3196,
    }
    counts = report["recomputed_counts"]["candidate"]
    assert counts["active_step_count"] == 3197
    assert counts["active_steps_satisfying_operational_semantics"] == 3197
    assert counts["legacy_double_cap_delta_count"] == 3196
    assert counts["recorded_intervention_matches_float32_norm_step_count"] == 12461
    reference = report["recomputed_counts"]["reference"]
    assert reference["active_step_count"] == 0
    assert reference["recorded_intervention_matches_float32_norm_step_count"] == 9400
    assert report["checker_conclusion"]["corrected_manipulation_integrity_passed"] is True
    assert report["checker_conclusion"]["next_allowed_stage"] == (
        "independent_diagnostic_confirmation_only"
    )
    assert report["scope"]["webots_was_run"] is False
    assert report["scope"]["validation_split_was_accessed"] is False
    assert report["scope"]["test_split_was_accessed"] is False
    assert report["inputs"]["hash_snapshots"]["unchanged"] is True
    assert report["inputs"]["hash_snapshots"]["entry"] == report["inputs"]["hash_snapshots"]["exit"]
    assert report["inputs"]["audit_implementation"] == {
        "path": "scripts/audit_front_brake_manipulation_boundary.py",
        "sha256": AUDIT._sha256_file(SCRIPT_PATH),
    }
    assert set(report["inputs"]["paired_diagnostic_report_artifacts"]) == {
        "json",
        "yaml",
        "sha256",
    }
    assert all(
        binding["verified_from_git_blob"] is True
        for binding in report["inputs"]["source_snapshot"]["bindings"].values()
    )
    assert {path: AUDIT._sha256_file(path) for path in input_paths} == before

    output = tmp_path / "audit"
    json_path = output / "manipulation_boundary_reaudit.json"
    yaml_path = output / "manipulation_boundary_reaudit.yaml"
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "passed"
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["status"] == "passed"
    sidecar = (output / "manipulation_boundary_reaudit.sha256").read_text(encoding="utf-8")
    assert sidecar == (
        f"{AUDIT._sha256_file(json_path)}  {json_path.name}\n"
        f"{AUDIT._sha256_file(yaml_path)}  {yaml_path.name}\n"
    )


def test_seed_25737_step_21_is_float_double_boundary(tmp_path: Path) -> None:
    boundary = _run(tmp_path)["numerical_boundary"]
    assert boundary["scenario_seed"] == 25737
    assert boundary["step"] == 21
    assert boundary["epsilon"] == pytest.approx(1e-6)
    assert boundary["action_braking_delta"] == pytest.approx(1.0132789611816406e-6)
    assert boundary["policy_to_double_cap_delta"] == pytest.approx(9.98510554706189e-7)
    assert boundary["action_to_double_cap_error"] == pytest.approx(1.4768406497722708e-8)
    assert boundary["operational_float32_action_test_passed"] is True
    assert boundary["legacy_double_cap_test_passed"] is False
    assert boundary["boundary_condition_verified"] is True


def test_report_sha_is_fail_closed(tmp_path: Path) -> None:
    changed = tmp_path / "paired_diagnostic_report.json"
    shutil.copy2(PAIRED_REPORT, changed)
    changed.write_text(changed.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(PermissionError, match="paired_report_json SHA-256 mismatch"):
        AUDIT.audit_front_brake_manipulation(
            paired_report_path=changed,
            output_directory=tmp_path / "out",
            repository_root=REPOSITORY_ROOT,
        )


def test_active_semantics_use_admitted_action_not_double_cap_delta() -> None:
    step = {
        "shield_active": True,
        "policy_action": [1.0, 0.11189452558755875],
        "shielded_action": [0.9999989867210388, 0.11189452558755875],
        "shield_normalized_speed_limit": 0.9999990014894453,
        "shield_intervention": 1.0132789611816406e-6,
    }
    result = AUDIT._active_step_semantics(step, 1e-6, enabled=True)
    assert result["angular_action_float32_bits_unchanged"] is True
    assert result["shielded_linear_matches_cap_float32_bits"] is True
    assert result["linear_float32_delta_exceeds_epsilon"] is True
    assert result["recorded_intervention_matches_float32_norm"] is True
    assert result["action_braking_delta"] > result["epsilon"]
    assert result["policy_to_double_cap_delta"] <= result["epsilon"]
    assert result["legacy_double_cap_test_passed"] is False
    assert result["operational_float32_action_test_passed"] is True


def test_recorded_intervention_tamper_fails_operational_semantics() -> None:
    step = {
        "shield_active": True,
        "policy_action": [1.0, 0.11189452558755875],
        "shielded_action": [0.9999989867210388, 0.11189452558755875],
        "shield_normalized_speed_limit": 0.9999990014894453,
        "shield_intervention": 1.0132789611816406e-6 + 1e-12,
    }

    result = AUDIT._active_step_semantics(step, 1e-6, enabled=True)

    assert result["recomputed_active_from_float32_action"] is True
    assert result["recorded_intervention_matches_float32_norm"] is False
    assert result["operational_float32_action_test_passed"] is False


def test_one_float32_ulp_cap_tamper_fails_bit_exact_semantics() -> None:
    original_cap = 0.9999990014894453
    tampered_cap = float(
        np.nextafter(np.float32(original_cap), np.float32(np.inf), dtype=np.float32)
    )
    step = {
        "shield_active": True,
        "policy_action": [1.0, 0.11189452558755875],
        "shielded_action": [0.9999989867210388, 0.11189452558755875],
        "shield_normalized_speed_limit": tampered_cap,
        "shield_intervention": 1.0132789611816406e-6,
    }

    result = AUDIT._active_step_semantics(step, 1e-6, enabled=True)

    assert result["action_to_double_cap_error"] < 1e-6
    assert result["shielded_linear_matches_cap_float32_bits"] is False
    assert result["operational_float32_action_test_passed"] is False


def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        AUDIT.audit_front_brake_manipulation(
            paired_report_path=PAIRED_REPORT,
            output_directory=output,
            repository_root=REPOSITORY_ROOT,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
