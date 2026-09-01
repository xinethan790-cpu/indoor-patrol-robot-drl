"""Deterministic simulated environmental sensing for indoor patrol visits.

The navigation policy never consumes these values.  They model the application
layer that runs after a robot reaches a checkpoint: sample the local environment,
apply stateful alarm rules, and retain replayable evidence for the dashboard.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from indoor_patrol_rl.identifiers import validate_patrol_run_id

AlarmDirection = Literal["above", "below"]
AlarmSeverity = Literal["info", "warning", "critical"]
AlarmEventType = Literal["raised", "cleared"]

_SEVERITY_ORDER: dict[str, int] = {"info": 1, "warning": 2, "critical": 3}


@dataclass(frozen=True)
class EnvironmentalSensor:
    """One simulated environmental quantity and its measurement limits."""

    id: str
    quantity: str
    display_name: str
    unit: str
    ambient: float
    minimum: float
    maximum: float
    resolution: float
    noise_half_range: float


@dataclass(frozen=True)
class EnvironmentalSource:
    """A radial environmental anomaly source in world coordinates."""

    id: str
    sensor_id: str
    position: tuple[float, float]
    peak_delta: float
    decay_length_m: float


@dataclass(frozen=True)
class AlarmRule:
    """A hysteretic alarm that requires consecutive samples to change state."""

    id: str
    sensor_id: str
    severity: AlarmSeverity
    direction: AlarmDirection
    raise_threshold: float
    clear_threshold: float
    raise_after_samples: int
    clear_after_samples: int


@dataclass(frozen=True)
class InspectionConfig:
    """Validated configuration for one environmental inspection scenario."""

    schema_version: int
    scenario_id: str
    samples_per_visit: int
    sample_interval_s: float
    sensors: tuple[EnvironmentalSensor, ...]
    sources: tuple[EnvironmentalSource, ...]
    alarm_rules: tuple[AlarmRule, ...]


@dataclass(frozen=True)
class EnvironmentalSample:
    """One replayable simulated sensor observation."""

    schema_version: str
    run_id: str
    scenario_id: str
    scenario_seed: int
    sample_index: int
    visit_index: int
    checkpoint_id: str
    control_step: int
    simulation_time_s: float
    position_x_m: float
    position_y_m: float
    sensor_id: str
    quantity: str
    display_name: str
    unit: str
    truth_value: float
    noise_value: float
    observed_value: float
    quality: str


@dataclass(frozen=True)
class AlarmEvent:
    """A transition into or out of an alarm state."""

    schema_version: str
    run_id: str
    event_index: int
    event_type: AlarmEventType
    rule_id: str
    sensor_id: str
    severity: AlarmSeverity
    checkpoint_id: str
    visit_index: int
    sample_index: int
    control_step: int
    simulation_time_s: float
    observed_value: float
    threshold: float
    unit: str
    message: str


@dataclass(frozen=True)
class InspectionVisitRecord:
    """Aggregate of the sampling window at one reached checkpoint."""

    schema_version: str
    run_id: str
    visit_index: int
    checkpoint_id: str
    control_step: int
    simulation_time_s: float
    position_x_m: float
    position_y_m: float
    sample_count: int
    alarm_event_count: int
    active_alarm_count: int
    active_alarm_rule_ids: str
    highest_active_severity: str | None


@dataclass(frozen=True)
class InspectionVisitResult:
    """Samples, state transitions, and visit aggregate produced together."""

    samples: tuple[EnvironmentalSample, ...]
    events: tuple[AlarmEvent, ...]
    visit: InspectionVisitRecord


@dataclass
class _AlarmState:
    active: bool = False
    raise_count: int = 0
    clear_count: int = 0


def load_inspection_config(path: str | Path) -> InspectionConfig:
    """Load a strict, typo-resistant environmental scenario YAML file."""

    config_path = Path(path).expanduser()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read inspection config {config_path}: {error}") from error
    root = _mapping(raw, "inspection config")
    _exact_keys(
        root,
        {
            "schema_version",
            "scenario_id",
            "samples_per_visit",
            "sample_interval_s",
            "sensors",
            "sources",
            "alarm_rules",
        },
        "inspection config",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValueError("inspection config schema_version must be 1.")
    scenario_id = _non_empty_string(root["scenario_id"], "scenario_id")
    samples_per_visit = _positive_integer(root["samples_per_visit"], "samples_per_visit")
    sample_interval_s = _finite_number(root["sample_interval_s"], "sample_interval_s")
    if sample_interval_s <= 0.0:
        raise ValueError("sample_interval_s must be positive.")

    sensor_items = _sequence(root["sensors"], "sensors")
    if not sensor_items:
        raise ValueError("sensors must contain at least one sensor.")
    sensors = tuple(_load_sensor(item, index) for index, item in enumerate(sensor_items, 1))
    _require_unique((sensor.id for sensor in sensors), "sensor id")
    sensor_by_id = {sensor.id: sensor for sensor in sensors}

    source_items = _sequence(root["sources"], "sources")
    sources = tuple(_load_source(item, index) for index, item in enumerate(source_items, 1))
    _require_unique((source.id for source in sources), "source id")
    for source in sources:
        if source.sensor_id not in sensor_by_id:
            raise ValueError(
                f"source {source.id!r} references unknown sensor {source.sensor_id!r}."
            )

    rule_items = _sequence(root["alarm_rules"], "alarm_rules")
    if not rule_items:
        raise ValueError("alarm_rules must contain at least one rule.")
    rules = tuple(_load_alarm_rule(item, index) for index, item in enumerate(rule_items, 1))
    _require_unique((rule.id for rule in rules), "alarm rule id")
    for rule in rules:
        sensor = sensor_by_id.get(rule.sensor_id)
        if sensor is None:
            raise ValueError(
                f"alarm rule {rule.id!r} references unknown sensor {rule.sensor_id!r}."
            )
        if not sensor.minimum <= rule.raise_threshold <= sensor.maximum:
            raise ValueError(f"alarm rule {rule.id!r} raise_threshold is outside sensor range.")
        if not sensor.minimum <= rule.clear_threshold <= sensor.maximum:
            raise ValueError(f"alarm rule {rule.id!r} clear_threshold is outside sensor range.")
        if rule.direction == "above" and rule.clear_threshold >= rule.raise_threshold:
            raise ValueError(
                f"alarm rule {rule.id!r} requires clear_threshold < raise_threshold."
            )
        if rule.direction == "below" and rule.clear_threshold <= rule.raise_threshold:
            raise ValueError(
                f"alarm rule {rule.id!r} requires clear_threshold > raise_threshold."
            )

    return InspectionConfig(
        schema_version=1,
        scenario_id=scenario_id,
        samples_per_visit=samples_per_visit,
        sample_interval_s=sample_interval_s,
        sensors=sensors,
        sources=sources,
        alarm_rules=rules,
    )


class EnvironmentalInspection:
    """Stateful deterministic sampler and alarm evaluator for one patrol run."""

    def __init__(self, config: InspectionConfig, *, run_id: str, scenario_seed: int) -> None:
        self.config = config
        self.run_id = validate_patrol_run_id(run_id)
        if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int):
            raise ValueError("scenario_seed must be an integer.")
        self.scenario_seed = scenario_seed
        self.samples: list[EnvironmentalSample] = []
        self.events: list[AlarmEvent] = []
        self.visits: list[InspectionVisitRecord] = []
        self._states = {rule.id: _AlarmState() for rule in config.alarm_rules}
        self._sources_by_sensor = {
            sensor.id: tuple(source for source in config.sources if source.sensor_id == sensor.id)
            for sensor in config.sensors
        }
        self._rules_by_sensor = {
            sensor.id: tuple(rule for rule in config.alarm_rules if rule.sensor_id == sensor.id)
            for sensor in config.sensors
        }
        self._visited_indices: set[int] = set()

    @property
    def active_alarm_rule_ids(self) -> tuple[str, ...]:
        """Return active rule ids in stable configuration order."""

        return tuple(rule.id for rule in self.config.alarm_rules if self._states[rule.id].active)

    @property
    def sampling_duration_s(self) -> float:
        """Duration a live simulator must dwell to realize one sampling window."""

        return (self.config.samples_per_visit - 1) * self.config.sample_interval_s

    def sample_visit(
        self,
        checkpoint_id: str,
        visit_index: int,
        control_step: int,
        simulation_time_s: float,
        position: tuple[float, float],
    ) -> InspectionVisitResult:
        """Simulate a sampling window beginning at ``simulation_time_s``.

        A live adapter must keep the robot stationary and advance its simulator through
        :attr:`sampling_duration_s` before marking this window complete.  The shared
        ``control_step`` is the navigation step at which that dwell began.
        """

        checkpoint = _non_empty_string(checkpoint_id, "checkpoint_id")
        visit = _positive_integer(visit_index, "visit_index")
        if visit in self._visited_indices:
            raise ValueError(f"visit_index {visit} was already sampled.")
        if isinstance(control_step, bool) or not isinstance(control_step, int) or control_step < 0:
            raise ValueError("control_step must be a non-negative integer.")
        time_s = _finite_number(simulation_time_s, "simulation_time_s")
        if time_s < 0.0:
            raise ValueError("simulation_time_s must be non-negative.")
        point = _point(position, "position")

        visit_samples: list[EnvironmentalSample] = []
        visit_events: list[AlarmEvent] = []
        for sample_round in range(self.config.samples_per_visit):
            sample_time = time_s + sample_round * self.config.sample_interval_s
            for sensor in self.config.sensors:
                sample_index = len(self.samples) + 1
                sample = self._sample_sensor(
                    sensor=sensor,
                    sample_index=sample_index,
                    visit_index=visit,
                    checkpoint_id=checkpoint,
                    control_step=control_step,
                    simulation_time_s=sample_time,
                    position=point,
                )
                self.samples.append(sample)
                visit_samples.append(sample)
                for rule in self._rules_by_sensor[sensor.id]:
                    event = self._apply_alarm_rule(rule, sensor, sample)
                    if event is not None:
                        self.events.append(event)
                        visit_events.append(event)

        active_ids = self.active_alarm_rule_ids
        active_severities = [
            rule.severity for rule in self.config.alarm_rules if rule.id in active_ids
        ]
        visit_record = InspectionVisitRecord(
            schema_version="1.0",
            run_id=self.run_id,
            visit_index=visit,
            checkpoint_id=checkpoint,
            control_step=control_step,
            simulation_time_s=time_s,
            position_x_m=point[0],
            position_y_m=point[1],
            sample_count=len(visit_samples),
            alarm_event_count=len(visit_events),
            active_alarm_count=len(active_ids),
            active_alarm_rule_ids=";".join(active_ids),
            highest_active_severity=_highest_severity(active_severities),
        )
        self.visits.append(visit_record)
        self._visited_indices.add(visit)
        return InspectionVisitResult(
            samples=tuple(visit_samples),
            events=tuple(visit_events),
            visit=visit_record,
        )

    def _sample_sensor(
        self,
        *,
        sensor: EnvironmentalSensor,
        sample_index: int,
        visit_index: int,
        checkpoint_id: str,
        control_step: int,
        simulation_time_s: float,
        position: tuple[float, float],
    ) -> EnvironmentalSample:
        truth = sensor.ambient
        for source in self._sources_by_sensor[sensor.id]:
            distance = math.hypot(
                position[0] - source.position[0],
                position[1] - source.position[1],
            )
            truth += source.peak_delta * math.exp(-distance / source.decay_length_m)
        noise = sensor.noise_half_range * _deterministic_unit_noise(
            self.config.scenario_id,
            self.scenario_seed,
            sensor.id,
            sample_index,
        )
        observed = min(sensor.maximum, max(sensor.minimum, truth + noise))
        observed = round(round(observed / sensor.resolution) * sensor.resolution, 12)
        observed = min(sensor.maximum, max(sensor.minimum, observed))
        return EnvironmentalSample(
            schema_version="1.0",
            run_id=self.run_id,
            scenario_id=self.config.scenario_id,
            scenario_seed=self.scenario_seed,
            sample_index=sample_index,
            visit_index=visit_index,
            checkpoint_id=checkpoint_id,
            control_step=control_step,
            simulation_time_s=round(simulation_time_s, 12),
            position_x_m=position[0],
            position_y_m=position[1],
            sensor_id=sensor.id,
            quantity=sensor.quantity,
            display_name=sensor.display_name,
            unit=sensor.unit,
            truth_value=round(truth, 12),
            noise_value=round(noise, 12),
            observed_value=observed,
            quality="ok",
        )

    def _apply_alarm_rule(
        self,
        rule: AlarmRule,
        sensor: EnvironmentalSensor,
        sample: EnvironmentalSample,
    ) -> AlarmEvent | None:
        state = self._states[rule.id]
        event_type: AlarmEventType | None = None
        threshold = rule.raise_threshold
        if not state.active:
            breached = (
                sample.observed_value >= rule.raise_threshold
                if rule.direction == "above"
                else sample.observed_value <= rule.raise_threshold
            )
            state.raise_count = state.raise_count + 1 if breached else 0
            state.clear_count = 0
            if state.raise_count >= rule.raise_after_samples:
                state.active = True
                state.raise_count = 0
                event_type = "raised"
        else:
            cleared = (
                sample.observed_value <= rule.clear_threshold
                if rule.direction == "above"
                else sample.observed_value >= rule.clear_threshold
            )
            state.clear_count = state.clear_count + 1 if cleared else 0
            state.raise_count = 0
            if state.clear_count >= rule.clear_after_samples:
                state.active = False
                state.clear_count = 0
                event_type = "cleared"
                threshold = rule.clear_threshold
        if event_type is None:
            return None
        action = "raised" if event_type == "raised" else "cleared"
        return AlarmEvent(
            schema_version="1.0",
            run_id=self.run_id,
            event_index=len(self.events) + 1,
            event_type=event_type,
            rule_id=rule.id,
            sensor_id=rule.sensor_id,
            severity=rule.severity,
            checkpoint_id=sample.checkpoint_id,
            visit_index=sample.visit_index,
            sample_index=sample.sample_index,
            control_step=sample.control_step,
            simulation_time_s=sample.simulation_time_s,
            observed_value=sample.observed_value,
            threshold=threshold,
            unit=sensor.unit,
            message=(
                f"{sensor.display_name} alarm {action}: "
                f"{sample.observed_value:g} {sensor.unit} (threshold {threshold:g})"
            ),
        )


def _load_sensor(value: object, index: int) -> EnvironmentalSensor:
    label = f"sensors[{index}]"
    raw = _mapping(value, label)
    _exact_keys(
        raw,
        {
            "id",
            "quantity",
            "display_name",
            "unit",
            "ambient",
            "minimum",
            "maximum",
            "resolution",
            "noise_half_range",
        },
        label,
    )
    minimum = _finite_number(raw["minimum"], f"{label}.minimum")
    maximum = _finite_number(raw["maximum"], f"{label}.maximum")
    ambient = _finite_number(raw["ambient"], f"{label}.ambient")
    resolution = _finite_number(raw["resolution"], f"{label}.resolution")
    noise = _finite_number(raw["noise_half_range"], f"{label}.noise_half_range")
    if minimum >= maximum:
        raise ValueError(f"{label}.minimum must be less than maximum.")
    if not minimum <= ambient <= maximum:
        raise ValueError(f"{label}.ambient must be inside the sensor range.")
    if resolution <= 0.0:
        raise ValueError(f"{label}.resolution must be positive.")
    if noise < 0.0:
        raise ValueError(f"{label}.noise_half_range must be non-negative.")
    return EnvironmentalSensor(
        id=_non_empty_string(raw["id"], f"{label}.id"),
        quantity=_non_empty_string(raw["quantity"], f"{label}.quantity"),
        display_name=_non_empty_string(raw["display_name"], f"{label}.display_name"),
        unit=_non_empty_string(raw["unit"], f"{label}.unit"),
        ambient=ambient,
        minimum=minimum,
        maximum=maximum,
        resolution=resolution,
        noise_half_range=noise,
    )


def _load_source(value: object, index: int) -> EnvironmentalSource:
    label = f"sources[{index}]"
    raw = _mapping(value, label)
    _exact_keys(
        raw,
        {"id", "sensor_id", "position", "peak_delta", "decay_length_m"},
        label,
    )
    decay = _finite_number(raw["decay_length_m"], f"{label}.decay_length_m")
    if decay <= 0.0:
        raise ValueError(f"{label}.decay_length_m must be positive.")
    return EnvironmentalSource(
        id=_non_empty_string(raw["id"], f"{label}.id"),
        sensor_id=_non_empty_string(raw["sensor_id"], f"{label}.sensor_id"),
        position=_point(raw["position"], f"{label}.position"),
        peak_delta=_finite_number(raw["peak_delta"], f"{label}.peak_delta"),
        decay_length_m=decay,
    )


def _load_alarm_rule(value: object, index: int) -> AlarmRule:
    label = f"alarm_rules[{index}]"
    raw = _mapping(value, label)
    _exact_keys(
        raw,
        {
            "id",
            "sensor_id",
            "severity",
            "direction",
            "raise_threshold",
            "clear_threshold",
            "raise_after_samples",
            "clear_after_samples",
        },
        label,
    )
    severity = _non_empty_string(raw["severity"], f"{label}.severity")
    if severity not in _SEVERITY_ORDER:
        raise ValueError(f"{label}.severity must be info, warning, or critical.")
    direction = _non_empty_string(raw["direction"], f"{label}.direction")
    if direction not in {"above", "below"}:
        raise ValueError(f"{label}.direction must be above or below.")
    return AlarmRule(
        id=_non_empty_string(raw["id"], f"{label}.id"),
        sensor_id=_non_empty_string(raw["sensor_id"], f"{label}.sensor_id"),
        severity=severity,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
        raise_threshold=_finite_number(raw["raise_threshold"], f"{label}.raise_threshold"),
        clear_threshold=_finite_number(raw["clear_threshold"], f"{label}.clear_threshold"),
        raise_after_samples=_positive_integer(
            raw["raise_after_samples"], f"{label}.raise_after_samples"
        ),
        clear_after_samples=_positive_integer(
            raw["clear_after_samples"], f"{label}.clear_after_samples"
        ),
    )


def _deterministic_unit_noise(
    scenario_id: str,
    scenario_seed: int,
    sensor_id: str,
    sample_index: int,
) -> float:
    payload = f"{scenario_id}|{scenario_seed}|{sensor_id}|{sample_index}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")
    return 2.0 * integer / (2**64 - 1) - 1.0


def _highest_severity(severities: Sequence[str]) -> str | None:
    return max(severities, key=_SEVERITY_ORDER.__getitem__) if severities else None


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings.")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence.")
    return value


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys mismatch; missing={missing}, extra={extra}.")


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _point(value: object, label: str) -> tuple[float, float]:
    sequence = _sequence(value, label)
    if len(sequence) != 2:
        raise ValueError(f"{label} must contain exactly two coordinates.")
    return (
        _finite_number(sequence[0], f"{label}[0]"),
        _finite_number(sequence[1], f"{label}[1]"),
    )


def _require_unique(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"Duplicate {label}: {value!r}.")
        seen.add(value)
