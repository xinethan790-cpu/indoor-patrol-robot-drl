from __future__ import annotations

import copy
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from indoor_patrol_rl.inspection import EnvironmentalInspection, load_inspection_config


def _config_payload(*, samples_per_visit: int = 3) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scenario_id": "environmental-patrol-fixture-v1",
        "samples_per_visit": samples_per_visit,
        "sample_interval_s": 0.5,
        "sensors": [
            {
                "id": "temperature",
                "quantity": "temperature",
                "display_name": "Temperature",
                "unit": "Cel",
                "ambient": 20.0,
                "minimum": -20.0,
                "maximum": 100.0,
                "resolution": 0.001,
                "noise_half_range": 0.25,
            }
        ],
        "sources": [
            {
                "id": "hot-cabinet",
                "sensor_id": "temperature",
                "position": [0.0, 0.0],
                "peak_delta": 30.0,
                "decay_length_m": 1.0,
            }
        ],
        "alarm_rules": [
            {
                "id": "temperature-high",
                "sensor_id": "temperature",
                "severity": "warning",
                "direction": "above",
                "raise_threshold": 40.0,
                "clear_threshold": 35.0,
                "raise_after_samples": 2,
                "clear_after_samples": 2,
            }
        ],
    }


def _write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _record(value: object) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(vars(value))
    except TypeError as error:  # pragma: no cover - produces a clearer contract failure
        raise AssertionError(f"Inspection record is not serializable: {value!r}") from error


def _observed_values(inspection: EnvironmentalInspection) -> list[float]:
    return [float(_record(sample)["observed_value"]) for sample in inspection.samples]


def test_same_seed_replays_identical_samples_events_and_visits(tmp_path: Path) -> None:
    config = load_inspection_config(_write_config(tmp_path / "inspection.yaml", _config_payload()))

    first = EnvironmentalInspection(config, run_id="fixture-run", scenario_seed=7012)
    second = EnvironmentalInspection(config, run_id="fixture-run", scenario_seed=7012)
    for inspection in (first, second):
        inspection.sample_visit(
            checkpoint_id="P1",
            visit_index=1,
            control_step=120,
            simulation_time_s=15.36,
            position=(0.0, 0.0),
        )
        inspection.sample_visit(
            checkpoint_id="P2",
            visit_index=2,
            control_step=240,
            simulation_time_s=30.72,
            position=(4.0, 0.0),
        )

    assert [_record(row) for row in first.samples] == [_record(row) for row in second.samples]
    assert [_record(row) for row in first.events] == [_record(row) for row in second.events]
    assert [_record(row) for row in first.visits] == [_record(row) for row in second.visits]


def test_different_seed_changes_noise_without_changing_visit_shape(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["sensors"][0]["noise_half_range"] = 2.0
    payload["sensors"][0]["resolution"] = 0.000001
    config = load_inspection_config(_write_config(tmp_path / "inspection.yaml", payload))

    first = EnvironmentalInspection(config, run_id="fixture-run", scenario_seed=17)
    second = EnvironmentalInspection(config, run_id="fixture-run", scenario_seed=18)
    for inspection in (first, second):
        result = inspection.sample_visit(
            checkpoint_id="P1",
            visit_index=1,
            control_step=10,
            simulation_time_s=1.28,
            position=(2.0, 0.0),
        )
        assert len(result.samples) == payload["samples_per_visit"]

    assert _observed_values(first) != _observed_values(second)
    assert len(first.samples) == len(second.samples)
    assert len(first.visits) == len(second.visits) == 1


def test_radial_source_decays_exponentially_with_distance(tmp_path: Path) -> None:
    payload = _config_payload(samples_per_visit=1)
    payload["sensors"][0]["noise_half_range"] = 0.0
    payload["sensors"][0]["resolution"] = 0.000001
    config = load_inspection_config(_write_config(tmp_path / "inspection.yaml", payload))
    inspection = EnvironmentalInspection(config, run_id="radial-run", scenario_seed=1)

    near = inspection.sample_visit("P1", 1, 10, 1.0, (0.0, 0.0))
    far = inspection.sample_visit("P2", 2, 20, 2.0, (2.0, 0.0))
    near_value = float(_record(near.samples[0])["observed_value"])
    far_value = float(_record(far.samples[0])["observed_value"])

    assert near_value == pytest.approx(50.0, abs=1e-6)
    assert far_value == pytest.approx(20.0 + 30.0 * math.exp(-2.0), abs=2e-6)
    assert near_value > far_value > 20.0


def test_alarm_requires_consecutive_samples_and_clears_with_hysteresis(
    tmp_path: Path,
) -> None:
    payload = _config_payload(samples_per_visit=1)
    payload["sensors"][0]["noise_half_range"] = 0.0
    config = load_inspection_config(_write_config(tmp_path / "inspection.yaml", payload))
    inspection = EnvironmentalInspection(config, run_id="alarm-run", scenario_seed=3)

    first_hot = inspection.sample_visit("P1", 1, 10, 1.0, (0.0, 0.0))
    assert first_hot.events == () or first_hot.events == []
    assert inspection.active_alarm_rule_ids == () or inspection.active_alarm_rule_ids == set()

    second_hot = inspection.sample_visit("P1", 2, 20, 2.0, (0.0, 0.0))
    assert [_record(event)["event_type"] for event in second_hot.events] == ["raised"]
    assert set(inspection.active_alarm_rule_ids) == {"temperature-high"}

    # 37 Cel lies between the 40 Cel raise and 35 Cel clear thresholds.  The
    # alarm must remain active instead of chattering at the raise threshold.
    middle_radius = math.log(30.0 / 17.0)
    middle = inspection.sample_visit("P1", 3, 30, 3.0, (middle_radius, 0.0))
    assert middle.events == () or middle.events == []
    assert set(inspection.active_alarm_rule_ids) == {"temperature-high"}

    first_clear = inspection.sample_visit("P2", 4, 40, 4.0, (20.0, 0.0))
    assert first_clear.events == () or first_clear.events == []
    assert set(inspection.active_alarm_rule_ids) == {"temperature-high"}

    second_clear = inspection.sample_visit("P2", 5, 50, 5.0, (20.0, 0.0))
    assert [_record(event)["event_type"] for event in second_clear.events] == ["cleared"]
    assert not inspection.active_alarm_rule_ids


def test_non_consecutive_excursion_resets_pending_raise_counter(tmp_path: Path) -> None:
    payload = _config_payload(samples_per_visit=1)
    payload["sensors"][0]["noise_half_range"] = 0.0
    config = load_inspection_config(_write_config(tmp_path / "inspection.yaml", payload))
    inspection = EnvironmentalInspection(config, run_id="counter-run", scenario_seed=4)

    inspection.sample_visit("P1", 1, 10, 1.0, (0.0, 0.0))
    inspection.sample_visit("P2", 2, 20, 2.0, (20.0, 0.0))
    third = inspection.sample_visit("P1", 3, 30, 3.0, (0.0, 0.0))

    assert third.events == () or third.events == []
    assert not inspection.active_alarm_rule_ids
    fourth = inspection.sample_visit("P1", 4, 40, 4.0, (0.0, 0.0))
    assert [_record(event)["event_type"] for event in fourth.events] == ["raised"]


@pytest.mark.parametrize("section", ["root", "sensor", "source", "alarm"])
def test_config_rejects_unknown_keys_at_every_schema_level(
    tmp_path: Path,
    section: str,
) -> None:
    payload = _config_payload()
    target: dict[str, Any]
    if section == "root":
        target = payload
    elif section == "sensor":
        target = payload["sensors"][0]
    elif section == "source":
        target = payload["sources"][0]
    else:
        target = payload["alarm_rules"][0]
    target["unexpected_typo"] = True

    with pytest.raises(ValueError, match="(?i)(unexpected|unknown|exact|keys)"):
        load_inspection_config(_write_config(tmp_path / f"{section}.yaml", payload))


@pytest.mark.parametrize("collection", ["sensors", "sources", "alarm_rules"])
def test_config_rejects_duplicate_ids(tmp_path: Path, collection: str) -> None:
    payload = _config_payload()
    payload[collection].append(copy.deepcopy(payload[collection][0]))

    with pytest.raises(ValueError, match="(?i)(duplicate|unique)"):
        load_inspection_config(_write_config(tmp_path / f"{collection}.yaml", payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ambient", float("nan")),
        ("minimum", float("-inf")),
        ("maximum", float("inf")),
        ("resolution", float("nan")),
        ("noise_half_range", float("nan")),
    ],
)
def test_config_rejects_non_finite_sensor_numbers(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    payload = _config_payload()
    payload["sensors"][0][field] = value

    with pytest.raises(ValueError, match="(?i)(finite|nan|inf)"):
        load_inspection_config(_write_config(tmp_path / f"{field}.yaml", payload))


@pytest.mark.parametrize(
    ("direction", "raise_threshold", "clear_threshold"),
    [
        ("above", 40.0, 40.0),
        ("above", 40.0, 45.0),
        ("below", 10.0, 10.0),
        ("below", 10.0, 5.0),
    ],
)
def test_config_rejects_alarm_thresholds_without_a_hysteresis_band(
    tmp_path: Path,
    direction: str,
    raise_threshold: float,
    clear_threshold: float,
) -> None:
    payload = _config_payload()
    rule = payload["alarm_rules"][0]
    rule["direction"] = direction
    rule["raise_threshold"] = raise_threshold
    rule["clear_threshold"] = clear_threshold

    with pytest.raises(ValueError, match="(?i)(threshold|hysteresis|clear)"):
        load_inspection_config(_write_config(tmp_path / "threshold.yaml", payload))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("samples_per_visit",), 0),
        (("sample_interval_s",), 0.0),
        (("sensors", 0, "resolution"), 0.0),
        (("sensors", 0, "noise_half_range"), -0.1),
        (("sources", 0, "decay_length_m"), 0.0),
        (("alarm_rules", 0, "direction"), "high"),
        (("alarm_rules", 0, "raise_after_samples"), 0),
        (("alarm_rules", 0, "clear_after_samples"), 0),
    ],
)
def test_config_rejects_invalid_schema_and_domains(
    tmp_path: Path,
    path: tuple[object, ...],
    value: object,
) -> None:
    payload = _config_payload()
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        load_inspection_config(_write_config(tmp_path / "invalid.yaml", payload))


def test_config_rejects_unknown_sensor_references(tmp_path: Path) -> None:
    for collection in ("sources", "alarm_rules"):
        payload = _config_payload()
        payload[collection][0]["sensor_id"] = "missing-sensor"

        with pytest.raises(ValueError, match="(?i)(sensor|reference|missing)"):
            load_inspection_config(_write_config(tmp_path / f"unknown-{collection}.yaml", payload))


def test_sample_visit_records_checkpoint_context_and_monotonic_indices(
    tmp_path: Path,
) -> None:
    config = load_inspection_config(_write_config(tmp_path / "inspection.yaml", _config_payload()))
    inspection = EnvironmentalInspection(config, run_id="context-run", scenario_seed=77)

    result = inspection.sample_visit("P4", 4, 321, 41.088, (1.25, -0.5))

    assert len(result.samples) == 3
    assert _record(result.visit)["sample_count"] == 3
    rows = [_record(sample) for sample in result.samples]
    assert {row["checkpoint_id"] for row in rows} == {"P4"}
    assert {row["visit_index"] for row in rows} == {4}
    assert {row["control_step"] for row in rows} == {321}
    assert [row["sample_index"] for row in rows] == [1, 2, 3]
    assert [row["simulation_time_s"] for row in rows] == pytest.approx([41.088, 41.588, 42.088])


def test_quantization_cannot_push_observation_outside_sensor_range(tmp_path: Path) -> None:
    payload = _config_payload(samples_per_visit=1)
    sensor = payload["sensors"][0]
    sensor["ambient"] = 99.0
    sensor["resolution"] = 60.0
    sensor["noise_half_range"] = 0.0
    payload["sources"] = []
    config = load_inspection_config(_write_config(tmp_path / "inspection.yaml", payload))
    inspection = EnvironmentalInspection(config, run_id="clamp-run", scenario_seed=1)

    result = inspection.sample_visit("P1", 1, 1, 0.0, (0.0, 0.0))

    assert _record(result.samples[0])["observed_value"] == 100.0
