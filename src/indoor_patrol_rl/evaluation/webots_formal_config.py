"""Validated configuration for the held-out G0-G3 Webots experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml

from indoor_patrol_rl.planning import OccupancyGrid, plan_world_path

FormalPlanner = Literal["reactive", "dwa", "ppo"]

FORMAL_SPLIT = "test"
FORMAL_SCENARIO_SEEDS = tuple(range(10000, 10020))
FORMAL_PPO_TRAINING_SEEDS = (11, 22, 33)
FORMAL_SERIES = ("reactive", "dwa", "ppo_seed11", "ppo_seed22", "ppo_seed33")
FORMAL_CONDITION_IDS = ("G0", "G1", "G2", "G3")

DYNAMIC_OBSTACLE_DEFS = tuple(f"TRAIN_OBSTACLE_{index}" for index in range(5))
DYNAMIC_OBSTACLE_RADII = (0.20, 0.18, 0.18, 0.17, 0.19)
FORMAL_G2_CROSS_LANES = (0.00, -0.39, 0.39, 0.75, -0.77)
FORMAL_OBSTACLE_LANE_MARGIN = 0.01
FORMAL_ROBOT_BOUNDING_RADIUS = 0.122
FORMAL_MIN_ENDPOINT_CLEARANCE = 0.015
STATIC_LAYOUT_PLACEMENTS: dict[str, tuple[tuple[str, tuple[float, float, float]], ...]] = {
    # Training used the empty RectangleArena plus its perimeter walls.  Keeping
    # this layout empty makes G0 genuinely match training instead of silently
    # introducing evaluation-only geometry under a misleading name.
    "train_a": (),
    "unseen_b": (
        ("STATIC_UNSEEN_B_0", (-1.15, 0.90, 0.25)),
        ("STATIC_UNSEEN_B_1", (1.15, -0.95, 0.25)),
        ("STATIC_UNSEEN_B_2", (1.25, 0.95, 0.25)),
    ),
}
# Rotation is axis-angle, size is the fixed Box geometry in the world.  G1 is a
# deliberately asymmetric chicane rather than a reflected/rotated G0 layout.
STATIC_NODE_GEOMETRY: dict[
    str, tuple[tuple[float, float, float, float], tuple[float, float, float]]
] = {
    "STATIC_UNSEEN_B_0": ((0.0, 0.0, 1.0, 0.20), (0.50, 0.14, 0.50)),
    "STATIC_UNSEEN_B_1": ((0.0, 0.0, 1.0, -0.40), (0.18, 0.45, 0.50)),
    "STATIC_UNSEEN_B_2": ((0.0, 0.0, 1.0, 1.00), (0.42, 0.14, 0.50)),
}
STATIC_OBSTACLE_DEFS = tuple(
    def_name
    for placements in STATIC_LAYOUT_PLACEMENTS.values()
    for def_name, _translation in placements
)
HIDDEN_NODE_TRANSLATION = (0.0, 0.0, -5.0)
FORMAL_MAP_ORIGIN = (-2.25, -1.75)
FORMAL_MAP_SIZE = (4.50, 3.50)
FORMAL_MAP_RESOLUTION = 0.05
FORMAL_STATIC_INFLATION_RADIUS = 0.20

_ROOT_KEYS = {
    "schema_version",
    "experiment_id",
    "webots_version",
    "world",
    "split",
    "ppo_training_seeds",
    "scenario_seeds",
    "planners",
    "conditions",
    "control",
    "required_runs",
    "required_episode_count",
}
_CONDITION_KEYS = {
    "id",
    "description",
    "static_layout",
    "obstacle_count",
    "obstacle_cross_lanes",
    "obstacle_speed_range",
    "horizontal_route_half_length",
    "vertical_route_half_length",
}
_CONTROL_KEYS = {
    "action_repeat",
    "max_episode_steps",
    "max_linear_speed",
    "max_angular_speed",
    "max_linear_acceleration",
    "max_angular_acceleration",
}


@dataclass(frozen=True)
class WebotsFormalControl:
    action_repeat: int
    max_episode_steps: int
    max_linear_speed: float
    max_angular_speed: float
    max_linear_acceleration: float
    max_angular_acceleration: float


@dataclass(frozen=True)
class WebotsFormalCondition:
    condition_id: str
    description: str
    static_layout: str
    obstacle_count: int
    obstacle_cross_lanes: tuple[float, ...] | None
    obstacle_speed_range: tuple[float, float]
    horizontal_route_half_length: float
    vertical_route_half_length: float

    @property
    def semantic_sha256(self) -> str:
        payload = {
            "condition_id": self.condition_id,
            "static_layout": self.static_layout,
            "obstacle_count": self.obstacle_count,
            "obstacle_cross_lanes": (
                list(self.obstacle_cross_lanes)
                if self.obstacle_cross_lanes is not None
                else None
            ),
            "obstacle_spawn_protocol": (
                "fixed_cross_lanes_v1"
                if self.obstacle_cross_lanes is not None
                else "training_random_cross_axis"
            ),
            "obstacle_lane_seed_transform": (
                "parity_mirror_v1" if self.obstacle_cross_lanes is not None else None
            ),
            "obstacle_radii": list(DYNAMIC_OBSTACLE_RADII[: self.obstacle_count]),
            "robot_bounding_radius": FORMAL_ROBOT_BOUNDING_RADIUS,
            "minimum_endpoint_clearance": FORMAL_MIN_ENDPOINT_CLEARANCE,
            "obstacle_speed_range": list(self.obstacle_speed_range),
            "horizontal_route_half_length": self.horizontal_route_half_length,
            "vertical_route_half_length": self.vertical_route_half_length,
            "static_placements": [
                {
                    "def": def_name,
                    "translation": list(translation),
                    "rotation": list(STATIC_NODE_GEOMETRY[def_name][0]),
                    "box_size": list(STATIC_NODE_GEOMETRY[def_name][1]),
                }
                for def_name, translation in STATIC_LAYOUT_PLACEMENTS[self.static_layout]
            ],
            "reference_path": {
                "algorithm": "astar_8_connected_no_corner_cutting",
                "map_origin": list(FORMAL_MAP_ORIGIN),
                "map_size": list(FORMAL_MAP_SIZE),
                "resolution": FORMAL_MAP_RESOLUTION,
                "static_inflation_radius": FORMAL_STATIC_INFLATION_RADIUS,
            },
        }
        return _semantic_sha256(payload)


@dataclass(frozen=True)
class WebotsFormalConfig:
    source_path: Path
    source_sha256: str
    experiment_id: str
    webots_version: str
    world: str
    split: str
    scenario_seed_start: int
    scenario_seed_count: int
    ppo_training_seeds: tuple[int, ...]
    planners: tuple[str, ...]
    conditions: tuple[WebotsFormalCondition, ...]
    control: WebotsFormalControl
    required_runs: int
    required_episode_count: int

    @property
    def scenario_seeds(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.scenario_seed_start,
                self.scenario_seed_start + self.scenario_seed_count,
            )
        )

    def condition(self, condition_id: str) -> WebotsFormalCondition:
        matching = [item for item in self.conditions if item.condition_id == condition_id]
        if len(matching) != 1:
            raise ValueError(f"Unknown formal Webots condition: {condition_id!r}.")
        return matching[0]


@dataclass(frozen=True)
class FormalTestRequest:
    condition: WebotsFormalCondition
    planner: FormalPlanner
    planner_label: str
    training_seed: int | None
    scenario_seeds: tuple[int, ...]


_EXPECTED_CONDITIONS = {
    "G0": ("train_a", 3, (0.20, 0.42)),
    "G1": ("unseen_b", 3, (0.20, 0.42)),
    "G2": ("train_a", 5, (0.20, 0.42)),
    "G3": ("train_a", 3, (0.45, 0.85)),
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{label} fields differ from the formal schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return result


def _float_pair(value: Any, label: str) -> tuple[float, float]:
    values = _sequence(value, label)
    if len(values) != 2:
        raise ValueError(f"{label} must contain exactly two values.")
    lower = _positive_float(values[0], f"{label}[0]")
    upper = _positive_float(values[1], f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} must be ordered.")
    return lower, upper


def _optional_float_tuple(
    value: Any,
    label: str,
    *,
    expected_length: int,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    values = _sequence(value, label)
    if len(values) != expected_length:
        raise ValueError(f"{label} must contain exactly {expected_length} values.")
    result: list[float] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label}[{index}] must be numeric.")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] must be finite.")
        result.append(number)
    return tuple(result)


def _validate_formal_lanes(
    lanes: tuple[float, ...] | None,
    *,
    obstacle_count: int,
) -> None:
    if lanes is None:
        return
    radii = DYNAMIC_OBSTACLE_RADII[:obstacle_count]
    for first in range(obstacle_count):
        for second in range(first + 1, obstacle_count):
            required = radii[first] + radii[second] + FORMAL_OBSTACLE_LANE_MARGIN
            if abs(lanes[first] - lanes[second]) + 1e-12 < required:
                raise ValueError(
                    "Formal obstacle cross lanes do not keep dynamic cylinders disjoint."
                )


def _semantic_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_webots_formal_config(path: Path) -> WebotsFormalConfig:
    """Load the one supported formal matrix and reject protocol drift."""

    source_path = path.resolve()
    with source_path.open(encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), "Webots formal config")
    _exact_keys(raw, _ROOT_KEYS, "Webots formal config")
    if raw.get("schema_version") != 1:
        raise ValueError("Webots formal config schema_version must be 1.")
    seed_spec = _mapping(raw.get("scenario_seeds"), "scenario_seeds")
    _exact_keys(seed_spec, {"start", "count"}, "scenario_seeds")
    seed_start = _integer(seed_spec.get("start"), "scenario_seeds.start")
    seed_count = _integer(seed_spec.get("count"), "scenario_seeds.count", minimum=1)
    training_seeds = tuple(
        _integer(value, "ppo_training_seeds item")
        for value in _sequence(raw.get("ppo_training_seeds"), "ppo_training_seeds")
    )
    planners = tuple(str(value) for value in _sequence(raw.get("planners"), "planners"))

    conditions: list[WebotsFormalCondition] = []
    for index, value in enumerate(_sequence(raw.get("conditions"), "conditions")):
        item = _mapping(value, f"conditions[{index}]")
        _exact_keys(item, _CONDITION_KEYS, f"conditions[{index}]")
        obstacle_count = _integer(item.get("obstacle_count"), "condition.obstacle_count")
        if obstacle_count > len(DYNAMIC_OBSTACLE_DEFS):
            raise ValueError("Formal obstacle_count exceeds the Webots world capacity.")
        lanes = _optional_float_tuple(
            item.get("obstacle_cross_lanes"),
            "condition.obstacle_cross_lanes",
            expected_length=obstacle_count,
        )
        _validate_formal_lanes(lanes, obstacle_count=obstacle_count)
        condition = WebotsFormalCondition(
            condition_id=str(item.get("id", "")),
            description=str(item.get("description", "")),
            static_layout=str(item.get("static_layout", "")),
            obstacle_count=obstacle_count,
            obstacle_cross_lanes=lanes,
            obstacle_speed_range=_float_pair(
                item.get("obstacle_speed_range"), "condition.obstacle_speed_range"
            ),
            horizontal_route_half_length=_positive_float(
                item.get("horizontal_route_half_length"),
                "condition.horizontal_route_half_length",
            ),
            vertical_route_half_length=_positive_float(
                item.get("vertical_route_half_length"),
                "condition.vertical_route_half_length",
            ),
        )
        if condition.static_layout not in STATIC_LAYOUT_PLACEMENTS:
            raise ValueError(f"Unknown static_layout: {condition.static_layout!r}.")
        conditions.append(condition)

    control_raw = _mapping(raw.get("control"), "control")
    _exact_keys(control_raw, _CONTROL_KEYS, "control")
    control = WebotsFormalControl(
        action_repeat=_integer(
            control_raw.get("action_repeat"), "control.action_repeat", minimum=1
        ),
        max_episode_steps=_integer(
            control_raw.get("max_episode_steps"), "control.max_episode_steps", minimum=1
        ),
        max_linear_speed=_positive_float(
            control_raw.get("max_linear_speed"), "control.max_linear_speed"
        ),
        max_angular_speed=_positive_float(
            control_raw.get("max_angular_speed"), "control.max_angular_speed"
        ),
        max_linear_acceleration=_positive_float(
            control_raw.get("max_linear_acceleration"),
            "control.max_linear_acceleration",
        ),
        max_angular_acceleration=_positive_float(
            control_raw.get("max_angular_acceleration"),
            "control.max_angular_acceleration",
        ),
    )
    config = WebotsFormalConfig(
        source_path=source_path,
        source_sha256=_file_sha256(source_path),
        experiment_id=str(raw.get("experiment_id", "")),
        webots_version=str(raw.get("webots_version", "")),
        world=str(raw.get("world", "")),
        split=str(raw.get("split", "")),
        scenario_seed_start=seed_start,
        scenario_seed_count=seed_count,
        ppo_training_seeds=training_seeds,
        planners=planners,
        conditions=tuple(conditions),
        control=control,
        required_runs=_integer(raw.get("required_runs"), "required_runs", minimum=1),
        required_episode_count=_integer(
            raw.get("required_episode_count"), "required_episode_count", minimum=1
        ),
    )
    _validate_fixed_protocol(config)
    return config


def _validate_fixed_protocol(config: WebotsFormalConfig) -> None:
    if not config.experiment_id:
        raise ValueError("Formal experiment_id must not be empty.")
    if config.webots_version != "R2025a":
        raise ValueError("Formal Webots version must be R2025a.")
    if config.world.replace("\\", "/") != "worlds/webots_ppo_evaluate.wbt":
        raise ValueError("Formal world must be worlds/webots_ppo_evaluate.wbt.")
    if config.split != FORMAL_SPLIT or config.scenario_seeds != FORMAL_SCENARIO_SEEDS:
        raise ValueError("Formal split must be test with seeds 10000..10019 exactly.")
    if config.ppo_training_seeds != FORMAL_PPO_TRAINING_SEEDS:
        raise ValueError("Formal PPO training seeds must be exactly 11, 22, and 33.")
    if config.planners != FORMAL_SERIES:
        raise ValueError("Formal planners must be reactive, dwa, and PPO seeds 11/22/33.")
    if tuple(item.condition_id for item in config.conditions) != FORMAL_CONDITION_IDS:
        raise ValueError("Formal conditions must be exactly G0, G1, G2, and G3.")
    for condition in config.conditions:
        layout, count, speeds = _EXPECTED_CONDITIONS[condition.condition_id]
        expected_lanes = FORMAL_G2_CROSS_LANES if condition.condition_id == "G2" else None
        if (
            condition.static_layout != layout
            or condition.obstacle_count != count
            or condition.obstacle_cross_lanes != expected_lanes
            or condition.obstacle_speed_range != speeds
            or condition.horizontal_route_half_length != 1.55
            or condition.vertical_route_half_length != 1.10
        ):
            raise ValueError(f"Formal condition {condition.condition_id} has protocol drift.")
        if condition.obstacle_cross_lanes is not None:
            for lane, radius in zip(
                condition.obstacle_cross_lanes,
                DYNAMIC_OBSTACLE_RADII[: condition.obstacle_count],
                strict=True,
            ):
                endpoint_clearance = min(
                    condition.horizontal_route_half_length,
                    condition.vertical_route_half_length,
                ) - abs(lane) - radius - FORMAL_ROBOT_BOUNDING_RADIUS
                if endpoint_clearance + 1e-12 < FORMAL_MIN_ENDPOINT_CLEARANCE:
                    raise ValueError(
                        "Formal obstacle lane can overlap a robot route endpoint."
                    )
    expected_runs = len(FORMAL_CONDITION_IDS) * len(FORMAL_SERIES)
    expected_episodes = expected_runs * len(FORMAL_SCENARIO_SEEDS)
    if config.required_runs != expected_runs or config.required_episode_count != expected_episodes:
        raise ValueError("Formal required counts must describe 20 runs and 400 episodes.")
    expected_control = WebotsFormalControl(4, 300, 0.22, 1.50, 0.50, 3.00)
    if asdict(config.control) != asdict(expected_control):
        raise ValueError("Formal Webots control parameters have protocol drift.")


def resolve_formal_test_request(
    config: WebotsFormalConfig,
    *,
    condition_id: str,
    planner: str,
    training_seed: int | None,
    seed_start: int,
    episode_count: int,
    confirmed: bool,
) -> FormalTestRequest:
    """Resolve one of the twenty locked formal runs without consuming test data."""

    if not confirmed:
        raise PermissionError("Formal Webots test evaluation requires explicit confirmation.")
    if seed_start != config.scenario_seed_start or episode_count != config.scenario_seed_count:
        raise ValueError("Formal test must use all and only scenario seeds 10000..10019.")
    condition = config.condition(condition_id)
    if planner not in {"reactive", "dwa", "ppo"}:
        raise ValueError("WEBOTS_EVAL_PLANNER must be reactive, dwa, or ppo.")
    if planner == "ppo":
        if training_seed not in config.ppo_training_seeds:
            raise ValueError("Formal PPO training seed must be 11, 22, or 33.")
        planner_label = f"ppo_seed{training_seed}"
    else:
        if training_seed is not None:
            raise ValueError("Formal baseline evaluation must not claim a PPO training seed.")
        planner_label = planner
    if planner_label not in config.planners:
        raise ValueError("Resolved planner label is not part of the formal matrix.")
    return FormalTestRequest(
        condition=condition,
        planner=cast(FormalPlanner, planner),
        planner_label=planner_label,
        training_seed=training_seed,
        scenario_seeds=config.scenario_seeds,
    )


def static_node_translations(
    static_layout: str,
) -> dict[str, tuple[float, float, float]]:
    """Return an exhaustive placement map so every unused static node is hidden."""

    if static_layout not in STATIC_LAYOUT_PLACEMENTS:
        raise ValueError(f"Unknown static_layout: {static_layout!r}.")
    translations = {
        def_name: HIDDEN_NODE_TRANSLATION for def_name in STATIC_OBSTACLE_DEFS
    }
    translations.update(dict(STATIC_LAYOUT_PLACEMENTS[static_layout]))
    return translations


def inactive_dynamic_obstacle_defs(obstacle_count: int) -> tuple[str, ...]:
    """Return the dynamic nodes that must be hidden for one formal condition."""

    if not 0 <= obstacle_count <= len(DYNAMIC_OBSTACLE_DEFS):
        raise ValueError(
            f"obstacle_count must be between 0 and {len(DYNAMIC_OBSTACLE_DEFS)}."
        )
    return DYNAMIC_OBSTACLE_DEFS[obstacle_count:]


def obstacle_cross_lanes_for_seed(
    condition: WebotsFormalCondition,
    scenario_seed: int,
) -> tuple[float, ...] | None:
    """Mirror formal lanes by seed parity to remove a fixed left/right bias."""

    if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int):
        raise ValueError("scenario_seed must be an integer.")
    if condition.obstacle_cross_lanes is None:
        return None
    mirror = 1.0 if scenario_seed % 2 == 0 else -1.0
    return tuple(mirror * lane for lane in condition.obstacle_cross_lanes)


def formal_reference_path_length(
    static_layout: str,
    start: tuple[float, float],
    goal: tuple[float, float],
) -> float:
    """Return the inflated-grid A* reference length for one formal static layout."""

    if static_layout not in STATIC_LAYOUT_PLACEMENTS:
        raise ValueError(f"Unknown static_layout: {static_layout!r}.")
    coordinates = (*start, *goal)
    if len(start) != 2 or len(goal) != 2 or not all(math.isfinite(v) for v in coordinates):
        raise ValueError("Formal reference endpoints must be finite two-dimensional points.")
    width = int(math.ceil(FORMAL_MAP_SIZE[0] / FORMAL_MAP_RESOLUTION))
    height = int(math.ceil(FORMAL_MAP_SIZE[1] / FORMAL_MAP_RESOLUTION))
    x_centers = FORMAL_MAP_ORIGIN[0] + (
        np.arange(width, dtype=np.float64) + 0.5
    ) * FORMAL_MAP_RESOLUTION
    y_centers = FORMAL_MAP_ORIGIN[1] + (
        np.arange(height, dtype=np.float64) + 0.5
    ) * FORMAL_MAP_RESOLUTION
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    occupied = np.zeros((height, width), dtype=bool)
    # The RectangleArena perimeter is static geometry and part of both layouts.
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    for def_name, translation in STATIC_LAYOUT_PLACEMENTS[static_layout]:
        rotation, size = STATIC_NODE_GEOMETRY[def_name]
        angle = rotation[3]
        delta_x = x_grid - translation[0]
        delta_y = y_grid - translation[1]
        local_x = math.cos(angle) * delta_x + math.sin(angle) * delta_y
        local_y = -math.sin(angle) * delta_x + math.cos(angle) * delta_y
        occupied |= (np.abs(local_x) <= size[0] / 2.0) & (
            np.abs(local_y) <= size[1] / 2.0
        )
    grid = OccupancyGrid(
        occupied=occupied,
        resolution=FORMAL_MAP_RESOLUTION,
        origin=FORMAL_MAP_ORIGIN,
    ).inflate(FORMAL_STATIC_INFLATION_RADIUS)
    path = plan_world_path(grid, start, goal)
    if not math.isfinite(path.length) or path.length <= 0.0:
        raise RuntimeError("Formal A* reference path has an invalid length.")
    return path.length


__all__ = [
    "DYNAMIC_OBSTACLE_DEFS",
    "DYNAMIC_OBSTACLE_RADII",
    "FORMAL_CONDITION_IDS",
    "FORMAL_G2_CROSS_LANES",
    "FORMAL_MAP_ORIGIN",
    "FORMAL_MAP_RESOLUTION",
    "FORMAL_MAP_SIZE",
    "FORMAL_MIN_ENDPOINT_CLEARANCE",
    "FORMAL_PPO_TRAINING_SEEDS",
    "FORMAL_OBSTACLE_LANE_MARGIN",
    "FORMAL_ROBOT_BOUNDING_RADIUS",
    "FORMAL_SCENARIO_SEEDS",
    "FORMAL_SERIES",
    "FORMAL_SPLIT",
    "FORMAL_STATIC_INFLATION_RADIUS",
    "FormalTestRequest",
    "HIDDEN_NODE_TRANSLATION",
    "STATIC_LAYOUT_PLACEMENTS",
    "STATIC_NODE_GEOMETRY",
    "STATIC_OBSTACLE_DEFS",
    "WebotsFormalCondition",
    "WebotsFormalConfig",
    "WebotsFormalControl",
    "inactive_dynamic_obstacle_defs",
    "formal_reference_path_length",
    "load_webots_formal_config",
    "obstacle_cross_lanes_for_seed",
    "resolve_formal_test_request",
    "static_node_translations",
]
