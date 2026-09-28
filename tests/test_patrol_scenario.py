from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from indoor_patrol_rl.application.scenario import (
    DEFAULT_OPERATIONAL_PATROL_SCENARIO_PATH,
    DEFAULT_STRESS_PATROL_SCENARIO_PATH,
    OPERATIONAL_OBSTACLE_SPEED_MAX_MPS,
    OPERATIONAL_OBSTACLE_SPEED_MIN_MPS,
    UNQUALIFIED_PATROL_EVIDENCE_SCOPE,
    PatrolScenarioConfigError,
    PatrolScenarioIntegrityError,
    PatrolScenarioMode,
    load_patrol_scenario_config,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario(*, mode: str = "operational") -> dict[str, Any]:
    speeds = (0.18, 0.16) if mode == "operational" else (0.31, 0.27)
    return {
        "schema_version": 1,
        "scenario_id": f"patrol_{mode}_test",
        "mode": mode,
        "inherits_r2a_system_qualification": False,
        "dynamic_obstacles": [
            {
                "def_name": "DYNAMIC_OBSTACLE_0",
                "axis": 1,
                "bounds_m": [-1.2, 1.2],
                "speed_mps": speeds[0],
            },
            {
                "def_name": "DYNAMIC_OBSTACLE_1",
                "axis": 0,
                "bounds_m": [-1.55, 1.55],
                "speed_mps": speeds[1],
            },
        ],
    }


def _write_scenario(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_loads_operational_scenario_with_source_identity() -> None:
    config = load_patrol_scenario_config(
        DEFAULT_OPERATIONAL_PATROL_SCENARIO_PATH,
        repository_root=_REPOSITORY_ROOT,
    )

    assert config.scenario_id == "patrol_operational_v1"
    assert config.mode is PatrolScenarioMode.OPERATIONAL
    assert config.inherits_r2a_system_qualification is False
    assert config.patrol_scenario_qualified is False
    assert config.deployment_evidence_scope == UNQUALIFIED_PATROL_EVIDENCE_SCOPE
    assert config.source_path == (
        _REPOSITORY_ROOT / DEFAULT_OPERATIONAL_PATROL_SCENARIO_PATH
    ).resolve()
    assert config.source_sha256 == hashlib.sha256(config.source_path.read_bytes()).hexdigest()
    assert len(config.dynamic_obstacles) == 2
    assert {obstacle.axis for obstacle in config.dynamic_obstacles} == {0, 1}
    assert all(
        OPERATIONAL_OBSTACLE_SPEED_MIN_MPS
        <= obstacle.speed_mps
        <= OPERATIONAL_OBSTACLE_SPEED_MAX_MPS
        for obstacle in config.dynamic_obstacles
    )
    config.verify_source_integrity()


def test_loads_stress_scenario_with_original_fast_obstacles() -> None:
    config = load_patrol_scenario_config(
        DEFAULT_STRESS_PATROL_SCENARIO_PATH,
        repository_root=_REPOSITORY_ROOT,
    )

    assert config.scenario_id == "patrol_stress_v1"
    assert config.mode is PatrolScenarioMode.STRESS
    assert config.inherits_r2a_system_qualification is False
    assert config.patrol_scenario_qualified is False
    assert config.obstacle("DYNAMIC_OBSTACLE_0").speed_mps == pytest.approx(0.31)
    assert config.obstacle("DYNAMIC_OBSTACLE_1").speed_mps == pytest.approx(0.27)
    assert config.obstacle("DYNAMIC_OBSTACLE_0").axis == 1
    assert config.obstacle("DYNAMIC_OBSTACLE_1").axis == 0
    with pytest.raises(KeyError, match="UNKNOWN"):
        config.obstacle("UNKNOWN")


def test_rejects_source_changed_after_loading(tmp_path: Path) -> None:
    source = _write_scenario(tmp_path, _scenario())
    config = load_patrol_scenario_config(source)
    source.write_text(source.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    with pytest.raises(PatrolScenarioIntegrityError, match="changed after"):
        config.verify_source_integrity()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(schema_version=2), "schema_version"),
        (lambda raw: raw.update(scenario_id="INVALID ID"), "scenario_id"),
        (lambda raw: raw.update(mode="demo"), "mode"),
        (
            lambda raw: raw.update(inherits_r2a_system_qualification=True),
            "explicitly false",
        ),
        (lambda raw: raw.update(extra=True), "fields differ"),
        (lambda raw: raw["dynamic_obstacles"].pop(), "exactly two"),
        (
            lambda raw: raw["dynamic_obstacles"][1].update(
                def_name="DYNAMIC_OBSTACLE_0"
            ),
            "DEF names must be unique",
        ),
        (
            lambda raw: raw["dynamic_obstacles"][1].update(axis=1),
            "cross axes",
        ),
        (
            lambda raw: raw["dynamic_obstacles"][0].update(bounds_m=[1.0, 1.0]),
            "lower_bound_m < upper_bound_m",
        ),
        (
            lambda raw: raw["dynamic_obstacles"][0].update(speed_mps=0.31),
            "0.10-0.20",
        ),
    ],
)
def test_rejects_malformed_operational_scenario(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    raw = _scenario()
    mutation(raw)
    source = _write_scenario(tmp_path, raw)

    with pytest.raises(PatrolScenarioConfigError, match=message):
        load_patrol_scenario_config(source)


def test_rejects_stress_label_without_an_above_envelope_obstacle(tmp_path: Path) -> None:
    raw = _scenario(mode="stress")
    raw["dynamic_obstacles"][0]["speed_mps"] = 0.18
    raw["dynamic_obstacles"][1]["speed_mps"] = 0.16
    source = _write_scenario(tmp_path, raw)

    with pytest.raises(PatrolScenarioConfigError, match="above 0.20"):
        load_patrol_scenario_config(source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("axis", True),
        ("axis", 2),
        ("speed_mps", 0.0),
        ("speed_mps", float("nan")),
        ("bounds_m", [float("-inf"), 1.0]),
    ],
)
def test_rejects_invalid_obstacle_values(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    raw = _scenario()
    raw["dynamic_obstacles"][0][field] = value
    source = _write_scenario(tmp_path, raw)

    with pytest.raises(PatrolScenarioConfigError):
        load_patrol_scenario_config(source)


def test_rejects_duplicate_yaml_fields(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.yaml"
    source.write_text(
        """schema_version: 1
scenario_id: patrol_duplicate_test
scenario_id: patrol_shadowed_test
mode: operational
inherits_r2a_system_qualification: false
dynamic_obstacles: []
""",
        encoding="utf-8",
    )

    with pytest.raises(PatrolScenarioConfigError, match="Duplicate"):
        load_patrol_scenario_config(source)


def _hotel_obstacle(**overrides: Any) -> dict[str, Any]:
    obstacle = {
        "def_name": "DYNAMIC_OBSTACLE_0",
        "axis": 1,
        "bounds_m": [-1.35, -0.62],
        "speed_mps": 0.16,
        "start_position_m": -0.62,
        "start_direction": -1,
        "dwell_points_m": [-0.80],
        "dwell_seconds": 3.0,
    }
    obstacle.update(overrides)
    return obstacle


def _hotel_scenario() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scenario_id": "hotel_corridor_test",
        "mode": "operational",
        "inherits_r2a_system_qualification": False,
        "dynamic_obstacles": [
            _hotel_obstacle(),
            {
                "def_name": "DYNAMIC_OBSTACLE_1",
                "axis": 0,
                "bounds_m": [-1.75, 1.75],
                "speed_mps": 0.16,
            },
        ],
    }


def test_loads_hotel_dwell_scenario_fields(tmp_path: Path) -> None:
    source = _write_scenario(tmp_path, _hotel_scenario())

    config = load_patrol_scenario_config(source)

    obstacle = config.obstacle("DYNAMIC_OBSTACLE_0")
    assert obstacle.start_position_m == -0.62
    assert obstacle.start_direction == -1.0
    assert obstacle.dwell_points_m == (-0.80,)
    assert obstacle.dwell_seconds == 3.0
    plain = config.obstacle("DYNAMIC_OBSTACLE_1")
    assert plain.start_position_m is None
    assert plain.dwell_points_m is None


def test_rejects_start_position_outside_bounds(tmp_path: Path) -> None:
    raw = _hotel_scenario()
    raw["dynamic_obstacles"][0]["start_position_m"] = -2.0

    with pytest.raises(PatrolScenarioConfigError, match="start_position_m"):
        load_patrol_scenario_config(_write_scenario(tmp_path, raw))


def test_rejects_dwell_point_outside_bounds(tmp_path: Path) -> None:
    raw = _hotel_scenario()
    raw["dynamic_obstacles"][0]["dwell_points_m"] = [0.0]

    with pytest.raises(PatrolScenarioConfigError, match="dwell_points_m"):
        load_patrol_scenario_config(_write_scenario(tmp_path, raw))


def test_rejects_dwell_seconds_without_points(tmp_path: Path) -> None:
    raw = _hotel_scenario()
    raw["dynamic_obstacles"][0]["dwell_seconds"] = 3.0
    del raw["dynamic_obstacles"][0]["dwell_points_m"]

    with pytest.raises(PatrolScenarioConfigError, match="dwell_seconds requires"):
        load_patrol_scenario_config(_write_scenario(tmp_path, raw))


def test_rejects_nonbinary_start_direction(tmp_path: Path) -> None:
    raw = _hotel_scenario()
    raw["dynamic_obstacles"][0]["start_direction"] = 0.5

    with pytest.raises(PatrolScenarioConfigError, match="start_direction"):
        load_patrol_scenario_config(_write_scenario(tmp_path, raw))


def test_rejects_unknown_obstacle_field(tmp_path: Path) -> None:
    raw = _hotel_scenario()
    raw["dynamic_obstacles"][0]["path_m"] = [[0.0, 0.0]]

    with pytest.raises(PatrolScenarioConfigError, match="fields differ"):
        load_patrol_scenario_config(_write_scenario(tmp_path, raw))


def test_loads_repository_hotel_scenario_with_source_identity() -> None:
    config = load_patrol_scenario_config(
        Path("configs/patrol/scenario_hotel_corridor_v1.yaml"),
        repository_root=_REPOSITORY_ROOT,
    )

    assert config.scenario_id == "hotel_corridor_v1"
    assert config.mode is PatrolScenarioMode.OPERATIONAL
    assert config.source_sha256 == _sha256_of(
        _REPOSITORY_ROOT / "configs/patrol/scenario_hotel_corridor_v1.yaml"
    )
    crossing = config.obstacle("DYNAMIC_OBSTACLE_0")
    assert crossing.dwell_points_m is None
    assert crossing.start_position_m == -0.70
    parked = config.obstacle("DYNAMIC_OBSTACLE_1")
    assert parked.dwell_points_m == (1.55, 2.00)
    assert parked.dwell_seconds == (4.0, 30.0)
    assert parked.lower_bound_m == 1.30
    assert parked.upper_bound_m == 2.00


def test_repository_root_rejects_source_escape(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = _write_scenario(tmp_path, _scenario())

    with pytest.raises(PatrolScenarioConfigError, match="inside the repository"):
        load_patrol_scenario_config(outside, repository_root=repository)
