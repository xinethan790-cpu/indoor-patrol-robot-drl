"""Auditable on-disk bundle for environmental patrol inspection results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TypeVar

from indoor_patrol_rl.inspection.environmental import (
    AlarmEvent,
    EnvironmentalInspection,
    EnvironmentalSample,
    InspectionVisitRecord,
    load_inspection_config,
)

_CONFIG_FILE = "inspection_config.yaml"
_SAMPLES_FILE = "inspection_samples.csv"
_EVENTS_FILE = "alarm_events.jsonl"
_VISITS_FILE = "inspection_visits.csv"
_SUMMARY_FILE = "inspection_summary.json"
_MANIFEST_FILE = "inspection_manifest.json"
_MANIFEST_DIGEST_FILE = "inspection_manifest.sha256"


@dataclass(frozen=True)
class InspectionSummary:
    """Compact aggregate derived from one inspection engine's raw records."""

    schema_version: str
    run_id: str
    scenario_id: str
    scenario_seed: int
    expected_checkpoint_count: int
    visit_count: int
    visited_checkpoint_count: int
    coverage_ratio: float
    sensor_count: int
    sample_count: int
    event_count: int
    alarms_raised: int
    alarms_cleared: int
    active_alarm_count: int
    active_alarm_rule_ids: tuple[str, ...]
    highest_event_severity: str | None
    highest_active_severity: str | None
    config_file: str
    config_sha256: str
    samples_file: str
    alarm_events_file: str
    visits_file: str


def write_inspection_bundle(
    output_directory: str | Path,
    inspection: EnvironmentalInspection,
    config_path: str | Path,
    *,
    expected_checkpoint_count: int,
    patrol_summary_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write a self-contained, hash-bound bundle without overwriting evidence."""

    if (
        isinstance(expected_checkpoint_count, bool)
        or not isinstance(expected_checkpoint_count, int)
        or expected_checkpoint_count <= 0
    ):
        raise ValueError("expected_checkpoint_count must be a positive integer.")
    if len(inspection.visits) > expected_checkpoint_count:
        raise ValueError("Inspection visits exceed expected_checkpoint_count.")
    source_config = Path(config_path).expanduser().resolve()
    if not source_config.is_file():
        raise FileNotFoundError(f"Inspection config does not exist: {source_config}")
    if load_inspection_config(source_config) != inspection.config:
        raise ValueError(
            "Inspection config file no longer matches the configuration used by this run."
        )
    patrol_path = Path(patrol_summary_path).expanduser().resolve() if patrol_summary_path else None
    if patrol_path is not None and not patrol_path.is_file():
        raise FileNotFoundError(f"Patrol summary does not exist: {patrol_path}")
    if patrol_path is not None:
        patrol_payload = _load_json_mapping(patrol_path, "patrol summary")
        _validate_patrol_identity(
            patrol_payload,
            run_id=inspection.run_id,
            scenario_seed=inspection.scenario_seed,
            inspection_checkpoint_ids={visit.checkpoint_id for visit in inspection.visits},
            expected_checkpoint_count=expected_checkpoint_count,
        )

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = (
        _CONFIG_FILE,
        _SAMPLES_FILE,
        _EVENTS_FILE,
        _VISITS_FILE,
        _SUMMARY_FILE,
        _MANIFEST_FILE,
        _MANIFEST_DIGEST_FILE,
    )
    existing = [output / name for name in names if (output / name).exists()]
    if existing:
        raise FileExistsError(
            "Inspection bundle refuses to overwrite existing evidence: "
            + ", ".join(path.name for path in existing)
        )

    config_sha256 = _sha256_file(source_config)
    raised = sum(event.event_type == "raised" for event in inspection.events)
    cleared = sum(event.event_type == "cleared" for event in inspection.events)
    severity_order = {"info": 1, "warning": 2, "critical": 3}
    highest_event = (
        max((event.severity for event in inspection.events), key=severity_order.__getitem__)
        if inspection.events
        else None
    )
    active_rules = [
        rule
        for rule in inspection.config.alarm_rules
        if rule.id in inspection.active_alarm_rule_ids
    ]
    highest_active = (
        max((rule.severity for rule in active_rules), key=severity_order.__getitem__)
        if active_rules
        else None
    )
    visited_checkpoint_ids = {visit.checkpoint_id for visit in inspection.visits}
    if len(visited_checkpoint_ids) > expected_checkpoint_count:
        raise ValueError("Unique inspection checkpoints exceed expected_checkpoint_count.")
    summary = InspectionSummary(
        schema_version="1.0",
        run_id=inspection.run_id,
        scenario_id=inspection.config.scenario_id,
        scenario_seed=inspection.scenario_seed,
        expected_checkpoint_count=expected_checkpoint_count,
        visit_count=len(inspection.visits),
        visited_checkpoint_count=len(visited_checkpoint_ids),
        coverage_ratio=len(visited_checkpoint_ids) / expected_checkpoint_count,
        sensor_count=len(inspection.config.sensors),
        sample_count=len(inspection.samples),
        event_count=len(inspection.events),
        alarms_raised=raised,
        alarms_cleared=cleared,
        active_alarm_count=len(inspection.active_alarm_rule_ids),
        active_alarm_rule_ids=inspection.active_alarm_rule_ids,
        highest_event_severity=highest_event,
        highest_active_severity=highest_active,
        config_file=_CONFIG_FILE,
        config_sha256=config_sha256,
        samples_file=_SAMPLES_FILE,
        alarm_events_file=_EVENTS_FILE,
        visits_file=_VISITS_FILE,
    )

    with tempfile.TemporaryDirectory(prefix=".inspection-staging-", dir=output) as staging_value:
        staging = Path(staging_value)
        shutil.copyfile(source_config, staging / _CONFIG_FILE)
        _write_dataclass_csv(staging / _SAMPLES_FILE, inspection.samples, EnvironmentalSample)
        _write_json_lines(staging / _EVENTS_FILE, inspection.events)
        _write_dataclass_csv(staging / _VISITS_FILE, inspection.visits, InspectionVisitRecord)
        summary_payload = asdict(summary)
        summary_payload["latest_readings"] = _latest_readings(inspection)
        summary_payload["visit_records"] = [asdict(visit) for visit in inspection.visits]
        summary_payload["alarm_events"] = [asdict(event) for event in inspection.events]
        _write_json(staging / _SUMMARY_FILE, summary_payload)

        record_counts = {
            _CONFIG_FILE: 1,
            _SAMPLES_FILE: len(inspection.samples),
            _EVENTS_FILE: len(inspection.events),
            _VISITS_FILE: len(inspection.visits),
            _SUMMARY_FILE: 1,
        }
        files = {
            name: {
                "sha256": _sha256_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
                "record_count": record_counts[name],
            }
            for name in record_counts
        }
        patrol_reference: dict[str, object] | None = None
        if patrol_path is not None:
            try:
                reference = patrol_path.relative_to(output).as_posix()
            except ValueError:
                reference = str(patrol_path)
            patrol_reference = {
                "path": reference,
                "sha256": _sha256_file(patrol_path),
                "size_bytes": patrol_path.stat().st_size,
            }
        manifest = {
            "schema_version": 1,
            "bundle_type": "environmental_patrol_inspection",
            "run_id": inspection.run_id,
            "scenario_id": inspection.config.scenario_id,
            "scenario_seed": inspection.scenario_seed,
            "files": files,
            "patrol_summary": patrol_reference,
        }
        _write_json(staging / _MANIFEST_FILE, manifest)
        manifest_sha256 = _sha256_file(staging / _MANIFEST_FILE)
        (staging / _MANIFEST_DIGEST_FILE).write_text(
            f"{manifest_sha256}  {_MANIFEST_FILE}\n", encoding="ascii"
        )

        for name in names:
            os.replace(staging / name, output / name)

    return {
        "config": output / _CONFIG_FILE,
        "samples": output / _SAMPLES_FILE,
        "events": output / _EVENTS_FILE,
        "visits": output / _VISITS_FILE,
        "summary": output / _SUMMARY_FILE,
        "manifest": output / _MANIFEST_FILE,
        "manifest_sha256": output / _MANIFEST_DIGEST_FILE,
    }


def verify_inspection_bundle(output_directory: str | Path) -> dict[str, object]:
    """Verify the sidecar, every bundled file, counts, and patrol-summary binding."""

    output = Path(output_directory).expanduser().resolve()
    manifest_path = output / _MANIFEST_FILE
    sidecar_path = output / _MANIFEST_DIGEST_FILE
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("Inspection manifest and SHA-256 sidecar are required.")
    sidecar_parts = sidecar_path.read_text(encoding="ascii").strip().split()
    if len(sidecar_parts) != 2 or sidecar_parts[1] != _MANIFEST_FILE:
        raise ValueError("Inspection manifest SHA-256 sidecar has an invalid format.")
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if sidecar_parts[0].casefold() != actual_manifest_sha256:
        raise ValueError("Inspection manifest SHA-256 verification failed.")
    manifest = _load_json_mapping(manifest_path, "inspection manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "bundle_type",
            "run_id",
            "scenario_id",
            "scenario_seed",
            "files",
            "patrol_summary",
        },
        "inspection manifest",
    )
    if manifest["schema_version"] != 1:
        raise ValueError("Inspection manifest schema_version must be 1.")
    if manifest["bundle_type"] != "environmental_patrol_inspection":
        raise ValueError("Inspection manifest bundle_type is invalid.")
    files = _mapping(manifest["files"], "inspection manifest files")
    expected_files = {
        _CONFIG_FILE,
        _SAMPLES_FILE,
        _EVENTS_FILE,
        _VISITS_FILE,
        _SUMMARY_FILE,
    }
    if set(files) != expected_files:
        raise ValueError("Inspection manifest file set is incomplete or unexpected.")
    for name, value in files.items():
        metadata = _mapping(value, f"inspection manifest files.{name}")
        _exact_keys(metadata, {"sha256", "size_bytes", "record_count"}, f"files.{name}")
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(f"Bundled inspection file is missing: {path}")
        if _sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"SHA-256 verification failed for {name}.")
        if path.stat().st_size != metadata["size_bytes"]:
            raise ValueError(f"Size verification failed for {name}.")
        actual_records = _record_count(path)
        if actual_records != metadata["record_count"]:
            raise ValueError(f"Record-count verification failed for {name}.")

    checkpoint_ids, expected_count = _validate_bundle_semantics(output, manifest)

    patrol_value = manifest["patrol_summary"]
    if patrol_value is not None:
        patrol = _mapping(patrol_value, "patrol_summary")
        _exact_keys(patrol, {"path", "sha256", "size_bytes"}, "patrol_summary")
        raw_path = patrol["path"]
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("patrol_summary.path must be a non-empty string.")
        patrol_path = Path(raw_path)
        if not patrol_path.is_absolute():
            patrol_path = output / patrol_path
        patrol_path = patrol_path.resolve()
        if not patrol_path.is_file():
            raise FileNotFoundError(f"Bound patrol summary is missing: {patrol_path}")
        if _sha256_file(patrol_path) != patrol["sha256"]:
            raise ValueError("Bound patrol summary SHA-256 verification failed.")
        if patrol_path.stat().st_size != patrol["size_bytes"]:
            raise ValueError("Bound patrol summary size verification failed.")
        patrol_payload = _load_json_mapping(patrol_path, "bound patrol summary")
        _validate_patrol_identity(
            patrol_payload,
            run_id=manifest["run_id"],
            scenario_seed=manifest["scenario_seed"],
            inspection_checkpoint_ids=checkpoint_ids,
            expected_checkpoint_count=expected_count,
        )

    return dict(manifest)


def _latest_readings(inspection: EnvironmentalInspection) -> list[dict[str, object]]:
    latest: dict[str, EnvironmentalSample] = {}
    for sample in inspection.samples:
        latest[sample.sensor_id] = sample
    return [
        asdict(latest[sensor.id])
        for sensor in inspection.config.sensors
        if sensor.id in latest
    ]


RecordT = TypeVar("RecordT")


def _write_dataclass_csv(path: Path, records: list[RecordT], record_type: type[RecordT]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(record_type)])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_json_lines(path: Path, records: list[AlarmEvent]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _record_count(path: Path) -> int:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    return 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not a readable JSON object: {error}") from error
    return _mapping(raw, label)


def _validate_bundle_semantics(
    output: Path,
    manifest: Mapping[str, object],
) -> tuple[set[str], int]:
    config = load_inspection_config(output / _CONFIG_FILE)
    summary = _load_json_mapping(output / _SUMMARY_FILE, "inspection summary")
    expected_summary_keys = {field.name for field in fields(InspectionSummary)} | {
        "latest_readings",
        "visit_records",
        "alarm_events",
    }
    _exact_keys(summary, expected_summary_keys, "inspection summary")
    if summary["schema_version"] != "1.0":
        raise ValueError("Inspection summary schema_version must be '1.0'.")
    for key in ("run_id", "scenario_id", "scenario_seed"):
        if summary[key] != manifest[key]:
            raise ValueError(f"Inspection summary {key} does not match the manifest.")
    if config.scenario_id != manifest["scenario_id"]:
        raise ValueError("Inspection config scenario_id does not match the manifest.")
    if summary["config_file"] != _CONFIG_FILE:
        raise ValueError("Inspection summary config_file is invalid.")
    if summary["config_sha256"] != _sha256_file(output / _CONFIG_FILE):
        raise ValueError("Inspection summary config_sha256 is invalid.")
    expected_references = {
        "samples_file": _SAMPLES_FILE,
        "alarm_events_file": _EVENTS_FILE,
        "visits_file": _VISITS_FILE,
    }
    for key, expected in expected_references.items():
        if summary[key] != expected:
            raise ValueError(f"Inspection summary {key} is invalid.")

    sample_rows = _read_csv_records(output / _SAMPLES_FILE, EnvironmentalSample)
    visit_rows = _read_csv_records(output / _VISITS_FILE, InspectionVisitRecord)
    events = _read_json_lines(output / _EVENTS_FILE)
    expected_count = _required_integer(
        summary, "expected_checkpoint_count", "inspection summary", minimum=1
    )
    visit_count = _required_integer(summary, "visit_count", "inspection summary")
    sample_count = _required_integer(summary, "sample_count", "inspection summary")
    event_count = _required_integer(summary, "event_count", "inspection summary")
    if visit_count != len(visit_rows):
        raise ValueError("Inspection summary visit_count does not match visits CSV.")
    if sample_count != len(sample_rows):
        raise ValueError("Inspection summary sample_count does not match samples CSV.")
    if event_count != len(events):
        raise ValueError("Inspection summary event_count does not match alarm events JSONL.")
    if summary["sensor_count"] != len(config.sensors):
        raise ValueError("Inspection summary sensor_count does not match the config.")

    sensor_by_id = {sensor.id: sensor for sensor in config.sensors}
    sample_by_index: dict[int, Mapping[str, str]] = {}
    sample_counts_by_visit: dict[int, int] = {}
    checkpoint_by_visit: dict[int, str] = {}
    for row_number, row in enumerate(sample_rows, 1):
        label = f"inspection samples row {row_number}"
        if row["schema_version"] != "1.0" or row["run_id"] != manifest["run_id"]:
            raise ValueError(f"{label} identity does not match the manifest.")
        if row["scenario_id"] != manifest["scenario_id"]:
            raise ValueError(f"{label} scenario_id does not match the manifest.")
        if _csv_integer(row, "scenario_seed", label) != manifest["scenario_seed"]:
            raise ValueError(f"{label} scenario_seed does not match the manifest.")
        sample_index = _csv_integer(row, "sample_index", label, minimum=1)
        if sample_index in sample_by_index:
            raise ValueError(f"Duplicate inspection sample_index: {sample_index}.")
        sample_by_index[sample_index] = row
        visit_index = _csv_integer(row, "visit_index", label, minimum=1)
        checkpoint_id = row["checkpoint_id"]
        if not checkpoint_id:
            raise ValueError(f"{label} checkpoint_id is empty.")
        previous_checkpoint = checkpoint_by_visit.setdefault(visit_index, checkpoint_id)
        if previous_checkpoint != checkpoint_id:
            raise ValueError(f"{label} changes checkpoint_id within one visit.")
        sample_counts_by_visit[visit_index] = sample_counts_by_visit.get(visit_index, 0) + 1
        sensor = sensor_by_id.get(row["sensor_id"])
        if sensor is None:
            raise ValueError(f"{label} references an unknown sensor.")
        if (
            row["quantity"] != sensor.quantity
            or row["display_name"] != sensor.display_name
            or row["unit"] != sensor.unit
        ):
            raise ValueError(f"{label} sensor metadata does not match the config.")
        observed = _csv_number(row, "observed_value", label)
        if not sensor.minimum <= observed <= sensor.maximum:
            raise ValueError(f"{label} observed_value is outside the sensor range.")
    if sorted(sample_by_index) != list(range(1, len(sample_rows) + 1)):
        raise ValueError("Inspection sample_index values must be contiguous from one.")

    visit_indices: set[int] = set()
    checkpoint_ids: set[str] = set()
    visit_payloads = summary["visit_records"]
    if not isinstance(visit_payloads, list) or len(visit_payloads) != len(visit_rows):
        raise ValueError("Inspection summary visit_records do not match visits CSV.")
    expected_samples_per_visit = config.samples_per_visit * len(config.sensors)
    for row_number, (row, raw_visit) in enumerate(zip(visit_rows, visit_payloads, strict=True), 1):
        label = f"inspection visits row {row_number}"
        if row["schema_version"] != "1.0" or row["run_id"] != manifest["run_id"]:
            raise ValueError(f"{label} identity does not match the manifest.")
        visit_index = _csv_integer(row, "visit_index", label, minimum=1)
        if visit_index in visit_indices:
            raise ValueError(f"Duplicate inspection visit_index: {visit_index}.")
        visit_indices.add(visit_index)
        checkpoint_id = row["checkpoint_id"]
        checkpoint_ids.add(checkpoint_id)
        if checkpoint_by_visit.get(visit_index) != checkpoint_id:
            raise ValueError(f"{label} has no matching samples.")
        if _csv_integer(row, "sample_count", label) != expected_samples_per_visit:
            raise ValueError(f"{label} sample_count does not match the config.")
        if sample_counts_by_visit.get(visit_index) != expected_samples_per_visit:
            raise ValueError(f"{label} sample_count does not match samples CSV.")
        embedded = _mapping(raw_visit, f"inspection summary visit_records[{row_number}]")
        _exact_keys(
            embedded,
            {field.name for field in fields(InspectionVisitRecord)},
            f"inspection summary visit_records[{row_number}]",
        )
        if (
            embedded.get("visit_index") != visit_index
            or embedded.get("checkpoint_id") != checkpoint_id
        ):
            raise ValueError(f"{label} does not match the embedded visit record.")

    if len(checkpoint_ids) > expected_count:
        raise ValueError("Unique inspection checkpoints exceed the expected count.")
    if summary["visited_checkpoint_count"] != len(checkpoint_ids):
        raise ValueError("Inspection summary visited_checkpoint_count is inconsistent.")
    coverage = _required_number(summary, "coverage_ratio", "inspection summary")
    if not math.isclose(coverage, len(checkpoint_ids) / expected_count, abs_tol=1e-12):
        raise ValueError("Inspection summary coverage_ratio is inconsistent.")

    rule_by_id = {rule.id: rule for rule in config.alarm_rules}
    active_rules: set[str] = set()
    raised = 0
    cleared = 0
    event_severities: list[str] = []
    for index, event in enumerate(events, 1):
        label = f"alarm event {index}"
        _exact_keys(event, {field.name for field in fields(AlarmEvent)}, label)
        if (
            event.get("schema_version") != "1.0"
            or event.get("event_index") != index
            or event.get("run_id") != manifest["run_id"]
        ):
            raise ValueError(f"{label} index or run_id is invalid.")
        rule_id = event.get("rule_id")
        rule = rule_by_id.get(rule_id) if isinstance(rule_id, str) else None
        if rule is None or event.get("sensor_id") != rule.sensor_id:
            raise ValueError(f"{label} rule or sensor identity is invalid.")
        sensor = sensor_by_id[rule.sensor_id]
        if event.get("severity") != rule.severity or event.get("unit") != sensor.unit:
            raise ValueError(f"{label} severity or unit does not match its rule.")
        event_severities.append(rule.severity)
        event_type = event.get("event_type")
        if event_type == "raised":
            if rule.id in active_rules:
                raise ValueError(f"{label} raises an already-active alarm.")
            active_rules.add(rule.id)
            raised += 1
            expected_threshold = rule.raise_threshold
        elif event_type == "cleared":
            if rule.id not in active_rules:
                raise ValueError(f"{label} clears an inactive alarm.")
            active_rules.remove(rule.id)
            cleared += 1
            expected_threshold = rule.clear_threshold
        else:
            raise ValueError(f"{label} event_type is invalid.")
        if not math.isclose(
            _required_number(event, "threshold", label),
            expected_threshold,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} threshold does not match its rule.")
        sample_index = _required_integer(event, "sample_index", label, minimum=1)
        sample = sample_by_index.get(sample_index)
        if sample is None:
            raise ValueError(f"{label} references an unknown sample_index.")
        event_visit = _required_integer(event, "visit_index", label, minimum=1)
        event_step = _required_integer(event, "control_step", label)
        if (
            event.get("checkpoint_id") != sample["checkpoint_id"]
            or event_visit != int(sample["visit_index"])
            or event_step != int(sample["control_step"])
            or event.get("sensor_id") != sample["sensor_id"]
        ):
            raise ValueError(f"{label} context does not match its referenced sample.")
        if not math.isclose(
            _required_number(event, "simulation_time_s", label),
            float(sample["simulation_time_s"]),
            abs_tol=1e-12,
        ) or not math.isclose(
            _required_number(event, "observed_value", label),
            float(sample["observed_value"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} value or time does not match its referenced sample.")
    embedded_events = summary["alarm_events"]
    if embedded_events != events:
        raise ValueError("Inspection summary alarm_events do not match alarm_events.jsonl.")
    if summary["alarms_raised"] != raised or summary["alarms_cleared"] != cleared:
        raise ValueError("Inspection summary alarm transition counts are inconsistent.")
    ordered_active = [rule.id for rule in config.alarm_rules if rule.id in active_rules]
    if summary["active_alarm_rule_ids"] != ordered_active:
        raise ValueError("Inspection summary active alarm rule ids are inconsistent.")
    if summary["active_alarm_count"] != len(active_rules):
        raise ValueError("Inspection summary active_alarm_count is inconsistent.")
    severity_order = {"info": 1, "warning": 2, "critical": 3}
    highest_event = (
        max(event_severities, key=severity_order.__getitem__) if event_severities else None
    )
    if summary["highest_event_severity"] != highest_event:
        raise ValueError("Inspection summary highest_event_severity is inconsistent.")
    active_severities = [rule_by_id[rule_id].severity for rule_id in active_rules]
    highest_active = (
        max(active_severities, key=severity_order.__getitem__) if active_severities else None
    )
    if summary["highest_active_severity"] != highest_active:
        raise ValueError("Inspection summary highest_active_severity is inconsistent.")

    latest = summary["latest_readings"]
    if not isinstance(latest, list):
        raise ValueError("Inspection summary latest_readings must be a list.")
    latest_by_sensor: dict[str, Mapping[str, object]] = {}
    for index, raw_latest in enumerate(latest, 1):
        item = _mapping(raw_latest, f"inspection summary latest_readings[{index}]")
        _exact_keys(
            item,
            {field.name for field in fields(EnvironmentalSample)},
            f"inspection summary latest_readings[{index}]",
        )
        sensor_id = item.get("sensor_id")
        if not isinstance(sensor_id, str) or sensor_id in latest_by_sensor:
            raise ValueError("Inspection summary latest_readings sensor ids are invalid.")
        latest_by_sensor[sensor_id] = item
    expected_latest: dict[str, Mapping[str, str]] = {}
    for row in sample_rows:
        expected_latest[row["sensor_id"]] = row
    if set(latest_by_sensor) != set(expected_latest):
        raise ValueError("Inspection summary latest_readings sensor set is inconsistent.")
    for sensor_id, row in expected_latest.items():
        item = latest_by_sensor[sensor_id]
        if item.get("sample_index") != int(row["sample_index"]):
            raise ValueError("Inspection summary latest_readings sample index is inconsistent.")
        if not math.isclose(
            _required_number(item, "observed_value", "latest reading"),
            float(row["observed_value"]),
            abs_tol=1e-12,
        ):
            raise ValueError("Inspection summary latest_readings value is inconsistent.")

    return checkpoint_ids, expected_count


def _validate_patrol_identity(
    patrol: Mapping[str, object],
    *,
    run_id: object,
    scenario_seed: object,
    inspection_checkpoint_ids: set[str],
    expected_checkpoint_count: int,
) -> None:
    if patrol.get("run_id") != run_id:
        raise ValueError("Patrol summary run_id does not match the inspection run_id.")
    if patrol.get("scenario_seed") != scenario_seed:
        raise ValueError(
            "Patrol summary scenario_seed does not match the inspection scenario_seed."
        )
    checkpoints_total = patrol.get("checkpoints_total")
    if (
        isinstance(checkpoints_total, bool)
        or not isinstance(checkpoints_total, int)
        or checkpoints_total != expected_checkpoint_count
    ):
        raise ValueError(
            "Patrol checkpoints_total does not match inspection expected_checkpoint_count."
        )
    raw_records = patrol.get("checkpoint_records")
    if not isinstance(raw_records, list):
        raise ValueError("Patrol summary must contain checkpoint_records as a list.")
    if len(raw_records) != expected_checkpoint_count:
        raise ValueError(
            "Patrol checkpoint count does not match inspection expected_checkpoint_count."
        )
    reached_ids: set[str] = set()
    all_ids: set[str] = set()
    for index, raw_record in enumerate(raw_records, 1):
        record = _mapping(raw_record, f"patrol checkpoint_records[{index}]")
        checkpoint_id = record.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("Patrol checkpoint record has an invalid checkpoint_id.")
        if checkpoint_id in all_ids:
            raise ValueError(f"Patrol checkpoint id is duplicated: {checkpoint_id!r}.")
        all_ids.add(checkpoint_id)
        if record.get("reached") is True:
            reached_ids.add(checkpoint_id)
    unexpected = sorted(inspection_checkpoint_ids - reached_ids)
    if unexpected:
        raise ValueError(
            "Inspection contains checkpoints not marked reached by patrol: "
            + ", ".join(unexpected)
        )


def _read_csv_records(path: Path, record_type: type[object]) -> list[dict[str, str]]:
    expected_fields = [field.name for field in fields(record_type)]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_fields:
            raise ValueError(f"{path.name} header does not match the supported schema.")
        return [dict(row) for row in reader]


def _read_json_lines(path: Path) -> list[Mapping[str, object]]:
    records: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"{path.name} line {line_number} is not valid JSON: {error}"
                ) from error
            records.append(_mapping(raw, f"{path.name} line {line_number}"))
    return records


def _required_integer(
    raw: Mapping[str, object],
    key: str,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label}.{key} must be an integer >= {minimum}.")
    return value


def _required_number(raw: Mapping[str, object], key: str, label: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}.{key} must be a finite number.")
    return number


def _csv_integer(
    row: Mapping[str, str],
    key: str,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    try:
        value = int(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{label}.{key} must be an integer.") from error
    if minimum is not None and value < minimum:
        raise ValueError(f"{label}.{key} must be >= {minimum}.")
    return value


def _csv_number(row: Mapping[str, str], key: str, label: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{label}.{key} must be a finite number.") from error
    if not math.isfinite(value):
        raise ValueError(f"{label}.{key} must be a finite number.")
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings.")
    return value


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise ValueError(f"{label} keys do not match the supported schema.")
