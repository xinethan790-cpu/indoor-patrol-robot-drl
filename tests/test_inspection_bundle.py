from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from indoor_patrol_rl.inspection import (
    EnvironmentalInspection,
    load_inspection_config,
    verify_inspection_bundle,
    write_inspection_bundle,
)

_BUNDLE_FILES = {
    "inspection_samples.csv",
    "alarm_events.jsonl",
    "inspection_visits.csv",
    "inspection_summary.json",
    "inspection_config.yaml",
    "inspection_manifest.json",
    "inspection_manifest.sha256",
}

_HASHED_PAYLOAD_FILES = {
    "inspection_samples.csv",
    "alarm_events.jsonl",
    "inspection_visits.csv",
    "inspection_summary.json",
    "inspection_config.yaml",
}


def _config_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scenario_id": "bundle-fixture-v1",
        "samples_per_visit": 3,
        "sample_interval_s": 0.25,
        "sensors": [
            {
                "id": "carbon-monoxide",
                "quantity": "carbon_monoxide",
                "display_name": "Carbon monoxide",
                "unit": "ppm",
                "ambient": 4.0,
                "minimum": 0.0,
                "maximum": 1000.0,
                "resolution": 0.1,
                "noise_half_range": 0.0,
            }
        ],
        "sources": [
            {
                "id": "co-leak",
                "sensor_id": "carbon-monoxide",
                "position": [1.0, 1.0],
                "peak_delta": 80.0,
                "decay_length_m": 0.5,
            }
        ],
        "alarm_rules": [
            {
                "id": "co-warning",
                "sensor_id": "carbon-monoxide",
                "severity": "critical",
                "direction": "above",
                "raise_threshold": 35.0,
                "clear_threshold": 20.0,
                "raise_after_samples": 2,
                "clear_after_samples": 2,
            }
        ],
    }


def _write_yaml(path: Path, value: Any) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rehash_bundle_file(output_directory: Path, filename: str) -> None:
    manifest_path = output_directory / "inspection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_path = output_directory / filename
    manifest["files"][filename]["sha256"] = _sha256(payload_path)
    manifest["files"][filename]["size_bytes"] = payload_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_directory / "inspection_manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  inspection_manifest.json\n",
        encoding="ascii",
    )


def _collect_returned_path_names(value: object) -> set[str]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Path):
        return {value.name}
    if isinstance(value, str):
        return {Path(value).name}
    if isinstance(value, dict):
        names: set[str] = set()
        for nested in value.values():
            names.update(_collect_returned_path_names(nested))
        return names
    if isinstance(value, (list, tuple, set)):
        names = set()
        for nested in value:
            names.update(_collect_returned_path_names(nested))
        return names
    return set()


def _populated_inspection(config_path: Path) -> EnvironmentalInspection:
    inspection = EnvironmentalInspection(
        load_inspection_config(config_path),
        run_id="webots-patrol-seed7000",
        scenario_seed=7000,
    )
    inspection.sample_visit("P1", 1, 100, 12.8, (1.0, 1.0))
    inspection.sample_visit("P2", 2, 220, 28.16, (4.0, 4.0))
    return inspection


def _write_complete_bundle(
    tmp_path: Path,
    *,
    bind_patrol_summary: bool = True,
) -> tuple[Path, Path, object]:
    config_path = _write_yaml(tmp_path / "source-inspection.yaml", _config_payload())
    inspection = _populated_inspection(config_path)
    patrol_summary = tmp_path / "patrol_summary.json"
    patrol_summary.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "webots-patrol-seed7000",
                "scenario_seed": 7000,
                "outcome": "completed",
                "checkpoints_total": 2,
                "checkpoint_records": [
                    {"checkpoint_id": "P1", "reached": True},
                    {"checkpoint_id": "P2", "reached": True},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_directory = tmp_path / "inspection-bundle"
    result = write_inspection_bundle(
        output_directory,
        inspection,
        config_path,
        expected_checkpoint_count=2,
        patrol_summary_path=patrol_summary if bind_patrol_summary else None,
    )
    return output_directory, patrol_summary, result


def test_write_bundle_creates_complete_machine_readable_evidence(tmp_path: Path) -> None:
    output_directory, patrol_summary, result = _write_complete_bundle(tmp_path)

    assert {path.name for path in output_directory.iterdir()} == _BUNDLE_FILES
    assert _BUNDLE_FILES <= _collect_returned_path_names(result)

    with (output_directory / "inspection_samples.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        samples = list(csv.DictReader(stream))
    with (output_directory / "inspection_visits.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        visits = list(csv.DictReader(stream))
    events = [
        json.loads(line)
        for line in (output_directory / "alarm_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    summary = json.loads((output_directory / "inspection_summary.json").read_text(encoding="utf-8"))

    assert len(samples) == 6
    assert {row["checkpoint_id"] for row in samples} == {"P1", "P2"}
    assert len(visits) == 2
    assert [row["checkpoint_id"] for row in visits] == ["P1", "P2"]
    assert [event["event_type"] for event in events] == ["raised", "cleared"]
    assert summary["run_id"] == "webots-patrol-seed7000"
    assert summary["scenario_id"] == "bundle-fixture-v1"
    assert summary["scenario_seed"] == 7000
    assert summary["expected_checkpoint_count"] == 2
    assert summary["visited_checkpoint_count"] == 2
    assert summary["coverage_ratio"] == pytest.approx(1.0)
    assert summary["sample_count"] == 6
    assert summary["event_count"] == 2

    manifest_text = (output_directory / "inspection_manifest.json").read_text(encoding="utf-8")
    for filename in _HASHED_PAYLOAD_FILES:
        assert filename in manifest_text
        assert _sha256(output_directory / filename) in manifest_text
    assert patrol_summary.name in manifest_text
    assert _sha256(patrol_summary) in manifest_text

    digest, filename = (
        (output_directory / "inspection_manifest.sha256")
        .read_text(encoding="utf-8")
        .strip()
        .split(maxsplit=1)
    )
    assert filename.lstrip("*") == "inspection_manifest.json"
    assert digest == _sha256(output_directory / "inspection_manifest.json")
    verify_inspection_bundle(output_directory)


def test_bundle_snapshots_exact_configuration_semantics(tmp_path: Path) -> None:
    output_directory, _, _ = _write_complete_bundle(tmp_path, bind_patrol_summary=False)
    source = yaml.safe_load((tmp_path / "source-inspection.yaml").read_text(encoding="utf-8"))
    snapshot = yaml.safe_load(
        (output_directory / "inspection_config.yaml").read_text(encoding="utf-8")
    )

    assert snapshot == source
    verify_inspection_bundle(output_directory)


def test_writer_refuses_to_overwrite_existing_bundle_by_default(tmp_path: Path) -> None:
    output_directory, _, _ = _write_complete_bundle(tmp_path)
    config_path = tmp_path / "source-inspection.yaml"
    inspection = _populated_inspection(config_path)

    with pytest.raises(FileExistsError, match="(?i)(exist|overwrite)"):
        write_inspection_bundle(
            output_directory,
            inspection,
            config_path,
            expected_checkpoint_count=2,
        )


def test_writer_rejects_config_changed_after_engine_creation(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "source-inspection.yaml", _config_payload())
    inspection = _populated_inspection(config_path)
    changed = _config_payload()
    changed["sensors"][0]["ambient"] = 5.0
    _write_yaml(config_path, changed)

    with pytest.raises(ValueError, match="configuration used by this run"):
        write_inspection_bundle(
            tmp_path / "mismatched-config",
            inspection,
            config_path,
            expected_checkpoint_count=2,
        )


@pytest.mark.parametrize(("field", "value"), [("run_id", "another-run"), ("scenario_seed", 3)])
def test_writer_rejects_unrelated_patrol_summary(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config_path = _write_yaml(tmp_path / "source-inspection.yaml", _config_payload())
    inspection = _populated_inspection(config_path)
    payload = {
        "schema_version": "1.0",
        "run_id": inspection.run_id,
        "scenario_seed": inspection.scenario_seed,
        "checkpoints_total": 2,
        "checkpoint_records": [
            {"checkpoint_id": "P1", "reached": True},
            {"checkpoint_id": "P2", "reached": True},
        ],
    }
    payload[field] = value
    patrol_summary = tmp_path / "wrong-patrol.json"
    patrol_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        write_inspection_bundle(
            tmp_path / f"wrong-{field}",
            inspection,
            config_path,
            expected_checkpoint_count=2,
            patrol_summary_path=patrol_summary,
        )


def test_writer_requires_full_patrol_checkpoint_evidence(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "source-inspection.yaml", _config_payload())
    inspection = _populated_inspection(config_path)
    patrol_summary = tmp_path / "incomplete-patrol.json"
    patrol_summary.write_text(
        json.dumps(
            {
                "run_id": inspection.run_id,
                "scenario_seed": inspection.scenario_seed,
                "checkpoints_total": 2,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint_records"):
        write_inspection_bundle(
            tmp_path / "incomplete-patrol-bundle",
            inspection,
            config_path,
            expected_checkpoint_count=2,
            patrol_summary_path=patrol_summary,
        )


@pytest.mark.parametrize(
    "filename",
    [
        "inspection_samples.csv",
        "alarm_events.jsonl",
        "inspection_visits.csv",
        "inspection_summary.json",
        "inspection_config.yaml",
    ],
)
def test_verifier_detects_tampered_payload_files(tmp_path: Path, filename: str) -> None:
    output_directory, _, _ = _write_complete_bundle(tmp_path)
    path = output_directory / filename
    path.write_bytes(path.read_bytes() + b"\nTAMPERED\n")

    with pytest.raises(ValueError, match="(?i)(sha|hash|digest|tamper|mismatch)"):
        verify_inspection_bundle(output_directory)


def test_verifier_detects_manifest_tampering_via_sidecar(tmp_path: Path) -> None:
    output_directory, _, _ = _write_complete_bundle(tmp_path)
    manifest = output_directory / "inspection_manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(ValueError, match="(?i)(manifest|sha|hash|digest|mismatch)"):
        verify_inspection_bundle(output_directory)


def test_verifier_rejects_semantically_invalid_payload_even_when_rehashed(
    tmp_path: Path,
) -> None:
    output_directory, _, _ = _write_complete_bundle(tmp_path)
    summary_path = output_directory / "inspection_summary.json"
    summary_path.write_text("not-json\n", encoding="utf-8")
    _rehash_bundle_file(output_directory, "inspection_summary.json")

    with pytest.raises(ValueError, match="(?i)(summary|json|readable)"):
        verify_inspection_bundle(output_directory)


def test_verifier_rejects_rehashed_alarm_that_disagrees_with_rule(tmp_path: Path) -> None:
    output_directory, _, _ = _write_complete_bundle(tmp_path)
    events_path = output_directory / "alarm_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[0]["severity"] = "info"
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    summary_path = output_directory / "inspection_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["alarm_events"][0]["severity"] = "info"
    summary["highest_event_severity"] = "info"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _rehash_bundle_file(output_directory, "alarm_events.jsonl")
    _rehash_bundle_file(output_directory, "inspection_summary.json")

    with pytest.raises(ValueError, match="(?i)(severity|rule)"):
        verify_inspection_bundle(output_directory)


def test_verifier_rejects_missing_required_file(tmp_path: Path) -> None:
    output_directory, _, _ = _write_complete_bundle(tmp_path)
    (output_directory / "alarm_events.jsonl").unlink()

    with pytest.raises((FileNotFoundError, ValueError), match="(?i)(missing|exist|alarm_events)"):
        verify_inspection_bundle(output_directory)


def test_manifest_omits_patrol_binding_when_not_requested(tmp_path: Path) -> None:
    output_directory, patrol_summary, _ = _write_complete_bundle(
        tmp_path,
        bind_patrol_summary=False,
    )
    manifest_text = (output_directory / "inspection_manifest.json").read_text(encoding="utf-8")

    assert _sha256(patrol_summary) not in manifest_text
    verify_inspection_bundle(output_directory)


def test_duplicate_visits_do_not_inflate_checkpoint_coverage(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "source-inspection.yaml", _config_payload())
    inspection = EnvironmentalInspection(
        load_inspection_config(config_path),
        run_id="repeat-run",
        scenario_seed=9,
    )
    inspection.sample_visit("P1", 1, 10, 1.0, (1.0, 1.0))
    inspection.sample_visit("P1", 2, 20, 2.0, (1.0, 1.0))
    output = tmp_path / "repeat-bundle"

    write_inspection_bundle(output, inspection, config_path, expected_checkpoint_count=2)

    summary = json.loads((output / "inspection_summary.json").read_text(encoding="utf-8"))
    assert summary["visit_count"] == 2
    assert summary["visited_checkpoint_count"] == 1
    assert summary["coverage_ratio"] == pytest.approx(0.5)
    verify_inspection_bundle(output)


@pytest.mark.parametrize("expected_checkpoint_count", [0, -1, True, 1.5])
def test_writer_rejects_invalid_expected_checkpoint_count(
    tmp_path: Path,
    expected_checkpoint_count: object,
) -> None:
    config_path = _write_yaml(tmp_path / "source-inspection.yaml", _config_payload())
    inspection = _populated_inspection(config_path)

    with pytest.raises((TypeError, ValueError), match="(?i)(checkpoint|positive|integer)"):
        write_inspection_bundle(
            tmp_path / f"invalid-{expected_checkpoint_count}",
            inspection,
            config_path,
            expected_checkpoint_count=expected_checkpoint_count,  # type: ignore[arg-type]
        )
