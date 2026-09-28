"""Strict, source-bound configuration for the four-point patrol scenario.

The R2a deployment profile qualifies a model-plus-shield component in its
recorded evaluation world.  Application scenarios are separate identities:
neither the operational nor the stress patrol inherits whole-system R2a
qualification merely because it uses that component.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from yaml.resolver import BaseResolver

PATROL_SCENARIO_SCHEMA_VERSION = 1
OPERATIONAL_OBSTACLE_SPEED_MIN_MPS = 0.10
OPERATIONAL_OBSTACLE_SPEED_MAX_MPS = 0.20
UNQUALIFIED_PATROL_EVIDENCE_SCOPE = "validated_component_outside_patrol_scenario"

DEFAULT_OPERATIONAL_PATROL_SCENARIO_PATH = Path(
    "configs/patrol/scenario_operational_v1.yaml"
)
DEFAULT_STRESS_PATROL_SCENARIO_PATH = Path("configs/patrol/scenario_stress_v1.yaml")

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SCENARIO_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
_DEF_NAME_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,127}")
_ROOT_FIELDS = {
    "schema_version",
    "scenario_id",
    "mode",
    "inherits_r2a_system_qualification",
    "dynamic_obstacles",
}
_OBSTACLE_FIELDS = {"def_name", "axis", "bounds_m", "speed_mps"}
_OBSTACLE_OPTIONAL_FIELDS = {
    "start_position_m",
    "start_direction",
    "dwell_points_m",
    "dwell_seconds",
}


class PatrolScenarioConfigError(ValueError):
    """A patrol scenario file is malformed or contradicts its declared mode."""


class PatrolScenarioIntegrityError(PatrolScenarioConfigError):
    """A loaded patrol scenario source no longer matches its SHA-256."""


class PatrolScenarioMode(StrEnum):
    """Supported application scenario modes."""

    OPERATIONAL = "operational"
    STRESS = "stress"


@dataclass(frozen=True, slots=True)
class PatrolDynamicObstacle:
    """One deterministic dynamic obstacle configured in the Webots world.

    The obstacle sweeps along ``axis`` between the configured bounds.  When the
    optional hotel-style fields are provided the motion is fully deterministic:
    ``start_position_m``/``start_direction`` replace the random initial state,
    and the obstacle parks at each ``dwell_points_m`` coordinate for
    ``dwell_seconds`` on every pass, modelling an agent that steps onto the
    patrol route and pauses there.
    """

    def_name: str
    axis: int
    lower_bound_m: float
    upper_bound_m: float
    speed_mps: float
    start_position_m: float | None = None
    start_direction: float | None = None
    dwell_points_m: tuple[float, ...] | None = None
    dwell_seconds: float | tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.def_name, str)
            or _DEF_NAME_PATTERN.fullmatch(self.def_name) is None
        ):
            raise PatrolScenarioConfigError(
                "dynamic obstacle def_name must be an uppercase Webots DEF identifier."
            )
        if type(self.axis) is not int or self.axis not in {0, 1}:
            raise PatrolScenarioConfigError("dynamic obstacle axis must be integer 0 or 1.")
        lower = _finite_number(self.lower_bound_m, "dynamic obstacle lower_bound_m")
        upper = _finite_number(self.upper_bound_m, "dynamic obstacle upper_bound_m")
        speed = _finite_number(self.speed_mps, "dynamic obstacle speed_mps")
        if lower >= upper:
            raise PatrolScenarioConfigError(
                "dynamic obstacle bounds must satisfy lower_bound_m < upper_bound_m."
            )
        if speed <= 0.0:
            raise PatrolScenarioConfigError("dynamic obstacle speed_mps must be positive.")
        start = self.start_position_m
        if start is not None:
            start = _finite_number(start, "dynamic obstacle start_position_m")
            if not lower <= start <= upper:
                raise PatrolScenarioConfigError(
                    "dynamic obstacle start_position_m must lie within bounds_m."
                )
        direction = self.start_direction
        if direction is not None:
            direction = _finite_number(direction, "dynamic obstacle start_direction")
            if direction not in (-1.0, 1.0):
                raise PatrolScenarioConfigError(
                    "dynamic obstacle start_direction must be exactly -1 or 1."
                )
        dwell_points = self.dwell_points_m
        if dwell_points is not None:
            if not isinstance(dwell_points, tuple) or not dwell_points:
                raise PatrolScenarioConfigError(
                    "dynamic obstacle dwell_points_m must be a non-empty sequence."
                )
            dwell_points = tuple(
                _finite_number(point, "dynamic obstacle dwell_points_m entry")
                for point in dwell_points
            )
            for point in dwell_points:
                if not lower <= point <= upper:
                    raise PatrolScenarioConfigError(
                        "dynamic obstacle dwell_points_m entries must lie within bounds_m."
                    )
        dwell_seconds = self.dwell_seconds
        if dwell_seconds is not None:
            if isinstance(dwell_seconds, (list, tuple)):
                if dwell_points is None or len(dwell_seconds) != len(dwell_points):
                    raise PatrolScenarioConfigError(
                        "dynamic obstacle dwell_seconds list must match dwell_points_m length."
                    )
                normalized: float | tuple[float, ...] = tuple(
                    _finite_number(item, "dynamic obstacle dwell_seconds entry")
                    for item in dwell_seconds
                )
            else:
                normalized = _finite_number(dwell_seconds, "dynamic obstacle dwell_seconds")
            entries = (
                normalized if isinstance(normalized, tuple) else (normalized,)
            )
            if any(entry <= 0.0 for entry in entries):
                raise PatrolScenarioConfigError(
                    "dynamic obstacle dwell_seconds must be positive."
                )
            if dwell_points is None:
                raise PatrolScenarioConfigError(
                    "dynamic obstacle dwell_seconds requires dwell_points_m."
                )
        elif dwell_points is not None:
            raise PatrolScenarioConfigError(
                "dynamic obstacle dwell_points_m requires dwell_seconds."
            )
        object.__setattr__(self, "lower_bound_m", lower)
        object.__setattr__(self, "upper_bound_m", upper)
        object.__setattr__(self, "speed_mps", speed)
        object.__setattr__(self, "start_position_m", start)
        object.__setattr__(self, "start_direction", direction)
        object.__setattr__(self, "dwell_points_m", dwell_points)
        object.__setattr__(self, "dwell_seconds", normalized if dwell_seconds is not None else None)


@dataclass(frozen=True, slots=True)
class PatrolScenarioConfig:
    """A validated patrol scenario plus the exact YAML source identity."""

    source_path: Path
    source_sha256: str
    scenario_id: str
    mode: PatrolScenarioMode
    inherits_r2a_system_qualification: bool
    dynamic_obstacles: tuple[PatrolDynamicObstacle, PatrolDynamicObstacle]

    def __post_init__(self) -> None:
        source_path = Path(self.source_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Patrol scenario source does not exist: {source_path}")
        if (
            not isinstance(self.source_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.source_sha256) is None
        ):
            raise PatrolScenarioConfigError(
                "Patrol scenario source_sha256 must be a lowercase SHA-256 digest."
            )
        if _file_sha256(source_path) != self.source_sha256:
            raise PatrolScenarioIntegrityError(
                "Patrol scenario source SHA-256 changed while it was being loaded."
            )
        if (
            not isinstance(self.scenario_id, str)
            or _SCENARIO_ID_PATTERN.fullmatch(self.scenario_id) is None
        ):
            raise PatrolScenarioConfigError(
                "scenario_id must be a lowercase path-safe identifier."
            )
        if not isinstance(self.mode, PatrolScenarioMode):
            raise PatrolScenarioConfigError("mode must be operational or stress.")
        if self.inherits_r2a_system_qualification is not False:
            raise PatrolScenarioConfigError(
                "A patrol scenario must explicitly decline whole-system R2a qualification."
            )
        if type(self.dynamic_obstacles) is not tuple or len(self.dynamic_obstacles) != 2:
            raise PatrolScenarioConfigError(
                "dynamic_obstacles must contain exactly two obstacle definitions."
            )
        if not all(
            isinstance(obstacle, PatrolDynamicObstacle)
            for obstacle in self.dynamic_obstacles
        ):
            raise PatrolScenarioConfigError(
                "dynamic_obstacles must contain only PatrolDynamicObstacle values."
            )
        def_names = {obstacle.def_name for obstacle in self.dynamic_obstacles}
        if len(def_names) != 2:
            raise PatrolScenarioConfigError("dynamic obstacle DEF names must be unique.")
        if {obstacle.axis for obstacle in self.dynamic_obstacles} != {0, 1}:
            raise PatrolScenarioConfigError(
                "The two patrol obstacles must cover cross axes 0 and 1 exactly once."
            )

        speeds = tuple(obstacle.speed_mps for obstacle in self.dynamic_obstacles)
        if self.mode is PatrolScenarioMode.OPERATIONAL:
            if not all(
                OPERATIONAL_OBSTACLE_SPEED_MIN_MPS
                <= speed
                <= OPERATIONAL_OBSTACLE_SPEED_MAX_MPS
                for speed in speeds
            ):
                raise PatrolScenarioConfigError(
                    "Operational obstacle speeds must remain within 0.10-0.20 m/s."
                )
        elif not any(speed > OPERATIONAL_OBSTACLE_SPEED_MAX_MPS for speed in speeds):
            raise PatrolScenarioConfigError(
                "Stress mode must contain at least one obstacle above 0.20 m/s."
            )

        object.__setattr__(self, "source_path", source_path)

    @property
    def patrol_scenario_qualified(self) -> bool:
        """Return the fixed application-level qualification state."""

        return False

    @property
    def deployment_evidence_scope(self) -> str:
        """Describe how R2a component evidence applies to this scenario."""

        return UNQUALIFIED_PATROL_EVIDENCE_SCOPE

    def obstacle(self, def_name: str) -> PatrolDynamicObstacle:
        """Look up one configured obstacle by its exact Webots DEF name."""

        for obstacle in self.dynamic_obstacles:
            if obstacle.def_name == def_name:
                return obstacle
        raise KeyError(f"Patrol scenario has no dynamic obstacle {def_name!r}.")

    def verify_source_integrity(self) -> None:
        """Fail if the YAML source was removed or changed after loading."""

        if not self.source_path.is_file():
            raise FileNotFoundError(
                f"Patrol scenario source does not exist: {self.source_path}"
            )
        actual = _file_sha256(self.source_path)
        if actual != self.source_sha256:
            raise PatrolScenarioIntegrityError(
                "Patrol scenario source SHA-256 changed after it was loaded."
            )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise PatrolScenarioConfigError(
                "Patrol scenario mapping keys must be scalar."
            ) from error
        if duplicate:
            raise PatrolScenarioConfigError(
                f"Duplicate patrol scenario YAML field is forbidden: {key!r}."
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PatrolScenarioConfigError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise PatrolScenarioConfigError(f"{label} must be a finite number.")
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PatrolScenarioConfigError(f"{label} must be a mapping.")
    return value


def _exact_mapping(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    actual = set(mapping)
    if actual != expected:
        raise PatrolScenarioConfigError(
            f"{label} fields differ from the schema; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(str(name) for name in actual - expected)}."
        )
    return mapping


def _resolve_source_path(
    path: str | Path,
    repository_root: str | Path | None,
) -> Path:
    source = Path(path).expanduser()
    if repository_root is None:
        resolved = source.resolve()
    else:
        root = Path(repository_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Repository root does not exist: {root}")
        resolved = (source if source.is_absolute() else root / source).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise PatrolScenarioConfigError(
                "Patrol scenario source must stay inside the repository."
            ) from error
    if not resolved.is_file():
        raise FileNotFoundError(f"Patrol scenario source does not exist: {resolved}")
    return resolved


def load_patrol_scenario_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> PatrolScenarioConfig:
    """Load one strict YAML scenario and bind the exact source bytes."""

    source_path = _resolve_source_path(path, repository_root)
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise PatrolScenarioConfigError(
            f"Cannot read patrol scenario source {source_path}: {error}"
        ) from error
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PatrolScenarioConfigError(
            f"Patrol scenario source must be UTF-8: {source_path}"
        ) from error
    try:
        loaded = yaml.load(source_text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise PatrolScenarioConfigError(
            f"Patrol scenario source is not valid YAML: {source_path}"
        ) from error

    raw = _exact_mapping(loaded, _ROOT_FIELDS, "patrol scenario")
    if type(raw["schema_version"]) is not int or (
        raw["schema_version"] != PATROL_SCENARIO_SCHEMA_VERSION
    ):
        raise PatrolScenarioConfigError("Patrol scenario schema_version must be 1.")
    scenario_id = raw["scenario_id"]
    if (
        not isinstance(scenario_id, str)
        or _SCENARIO_ID_PATTERN.fullmatch(scenario_id) is None
    ):
        raise PatrolScenarioConfigError(
            "scenario_id must be a lowercase path-safe identifier."
        )
    try:
        mode = PatrolScenarioMode(raw["mode"])
    except (TypeError, ValueError) as error:
        raise PatrolScenarioConfigError("mode must be operational or stress.") from error
    inherits_qualification = raw["inherits_r2a_system_qualification"]
    if inherits_qualification is not False:
        raise PatrolScenarioConfigError(
            "inherits_r2a_system_qualification must be explicitly false."
        )

    raw_obstacles = raw["dynamic_obstacles"]
    if isinstance(raw_obstacles, (str, bytes)) or not isinstance(
        raw_obstacles, Sequence
    ):
        raise PatrolScenarioConfigError("dynamic_obstacles must be a sequence.")
    obstacles: list[PatrolDynamicObstacle] = []
    for index, value in enumerate(raw_obstacles):
        mapping = _mapping(value, f"dynamic_obstacles[{index}]")
        actual = set(mapping)
        allowed = _OBSTACLE_FIELDS | _OBSTACLE_OPTIONAL_FIELDS
        if not _OBSTACLE_FIELDS <= actual or not actual <= allowed:
            raise PatrolScenarioConfigError(
                f"dynamic_obstacles[{index}] fields differ from the schema; "
                f"missing={sorted(_OBSTACLE_FIELDS - actual)}, "
                f"extra={sorted(str(name) for name in actual - allowed)}."
            )
        bounds = mapping["bounds_m"]
        if (
            isinstance(bounds, (str, bytes))
            or not isinstance(bounds, Sequence)
            or len(bounds) != 2
        ):
            raise PatrolScenarioConfigError(
                f"dynamic_obstacles[{index}].bounds_m must contain two numbers."
            )
        dwell_points_raw = mapping.get("dwell_points_m")
        if dwell_points_raw is not None:
            if isinstance(dwell_points_raw, (str, bytes)) or not isinstance(
                dwell_points_raw, Sequence
            ):
                raise PatrolScenarioConfigError(
                    f"dynamic_obstacles[{index}].dwell_points_m must be a sequence."
                )
            dwell_points = tuple(float(point) for point in dwell_points_raw)
        else:
            dwell_points = None
        obstacles.append(
            PatrolDynamicObstacle(
                def_name=mapping["def_name"],
                axis=mapping["axis"],
                lower_bound_m=bounds[0],
                upper_bound_m=bounds[1],
                speed_mps=mapping["speed_mps"],
                start_position_m=mapping.get("start_position_m"),
                start_direction=mapping.get("start_direction"),
                dwell_points_m=dwell_points,
                dwell_seconds=mapping.get("dwell_seconds"),
            )
        )
    if len(obstacles) != 2:
        raise PatrolScenarioConfigError(
            "dynamic_obstacles must contain exactly two obstacle definitions."
        )

    return PatrolScenarioConfig(
        source_path=source_path,
        source_sha256=source_sha256,
        scenario_id=scenario_id,
        mode=mode,
        inherits_r2a_system_qualification=inherits_qualification,
        dynamic_obstacles=(obstacles[0], obstacles[1]),
    )
