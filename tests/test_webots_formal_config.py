from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from indoor_patrol_rl.evaluation.webots_formal_config import (
    DYNAMIC_OBSTACLE_DEFS,
    DYNAMIC_OBSTACLE_RADII,
    FORMAL_G2_CROSS_LANES,
    FORMAL_SCENARIO_SEEDS,
    HIDDEN_NODE_TRANSLATION,
    STATIC_NODE_GEOMETRY,
    STATIC_OBSTACLE_DEFS,
    formal_reference_path_length,
    inactive_dynamic_obstacle_defs,
    load_webots_formal_config,
    obstacle_cross_lanes_for_seed,
    resolve_formal_test_request,
    static_node_translations,
)

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
FORMAL_CONFIG = REPOSITORY_ROOT / "configs/eval/webots_formal.yaml"


def test_repository_formal_config_loads_the_locked_matrix() -> None:
    config = load_webots_formal_config(FORMAL_CONFIG)

    assert config.scenario_seeds == FORMAL_SCENARIO_SEEDS
    assert config.ppo_training_seeds == (11, 22, 33)
    assert config.planners == (
        "reactive",
        "dwa",
        "ppo_seed11",
        "ppo_seed22",
        "ppo_seed33",
    )
    identity = [
        (item.condition_id, item.static_layout, item.obstacle_count)
        for item in config.conditions
    ]
    assert identity == [
        ("G0", "train_a", 3),
        ("G1", "unseen_b", 3),
        ("G2", "train_a", 5),
        ("G3", "train_a", 3),
    ]
    assert config.condition("G2").obstacle_cross_lanes == FORMAL_G2_CROSS_LANES
    assert all(
        config.condition(condition_id).obstacle_cross_lanes is None
        for condition_id in ("G0", "G1", "G3")
    )
    assert len(config.source_sha256) == 64
    assert len({item.semantic_sha256 for item in config.conditions}) == 4


@pytest.mark.parametrize(
    ("planner", "training_seed", "expected_label"),
    [
        ("reactive", None, "reactive"),
        ("dwa", None, "dwa"),
        ("ppo", 11, "ppo_seed11"),
        ("ppo", 22, "ppo_seed22"),
        ("ppo", 33, "ppo_seed33"),
    ],
)
def test_formal_request_resolves_only_canonical_labels(
    planner: str,
    training_seed: int | None,
    expected_label: str,
) -> None:
    config = load_webots_formal_config(FORMAL_CONFIG)

    request = resolve_formal_test_request(
        config,
        condition_id="G2",
        planner=planner,
        training_seed=training_seed,
        seed_start=10_000,
        episode_count=20,
        confirmed=True,
    )

    assert request.planner_label == expected_label
    assert request.scenario_seeds == FORMAL_SCENARIO_SEEDS
    assert request.condition.obstacle_cross_lanes == FORMAL_G2_CROSS_LANES


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confirmed": False},
        {"seed_start": 9999},
        {"episode_count": 19},
        {"planner": "ppo", "training_seed": 44},
        {"planner": "reactive", "training_seed": 11},
        {"planner": "unknown"},
        {"condition_id": "G4"},
    ],
)
def test_formal_request_fails_closed(kwargs: dict[str, object]) -> None:
    config = load_webots_formal_config(FORMAL_CONFIG)
    values: dict[str, object] = {
        "condition_id": "G0",
        "planner": "reactive",
        "training_seed": None,
        "seed_start": 10_000,
        "episode_count": 20,
        "confirmed": True,
    }
    values.update(kwargs)

    with pytest.raises((PermissionError, ValueError)):
        resolve_formal_test_request(config, **values)  # type: ignore[arg-type]


def test_g2_lanes_keep_the_fixed_world_cylinders_disjoint() -> None:
    for first in range(5):
        for second in range(first + 1, 5):
            clearance = abs(
                FORMAL_G2_CROSS_LANES[first] - FORMAL_G2_CROSS_LANES[second]
            ) - DYNAMIC_OBSTACLE_RADII[first] - DYNAMIC_OBSTACLE_RADII[second]
            assert clearance >= 0.01 - 1e-12


def test_g2_lanes_are_mirrored_deterministically_by_seed_parity() -> None:
    condition = load_webots_formal_config(FORMAL_CONFIG).condition("G2")

    even = obstacle_cross_lanes_for_seed(condition, 10_000)
    odd = obstacle_cross_lanes_for_seed(condition, 10_001)

    assert even == FORMAL_G2_CROSS_LANES
    assert odd == tuple(-value for value in FORMAL_G2_CROSS_LANES)


def test_world_node_maps_hide_every_unused_node() -> None:
    train = static_node_translations("train_a")
    unseen = static_node_translations("unseen_b")

    assert set(train) == set(STATIC_OBSTACLE_DEFS)
    assert set(train.values()) == {HIDDEN_NODE_TRANSLATION}
    assert all(unseen[name] != HIDDEN_NODE_TRANSLATION for name in STATIC_OBSTACLE_DEFS)
    assert inactive_dynamic_obstacle_defs(3) == DYNAMIC_OBSTACLE_DEFS[3:]
    assert inactive_dynamic_obstacle_defs(5) == ()


def test_unseen_layout_uses_an_astar_reference_not_euclidean_distance() -> None:
    start = (-1.65, 0.90)
    goal = (-0.60, 0.90)
    direct = math.dist(start, goal)

    training_length = formal_reference_path_length("train_a", start, goal)
    unseen_length = formal_reference_path_length("unseen_b", start, goal)

    assert training_length == pytest.approx(direct, abs=0.08)
    assert unseen_length > training_length + 0.05


def _route_endpoints(seed: int) -> tuple[tuple[float, float], tuple[float, float]]:
    random_generator = np.random.default_rng(seed)
    route_axis = str(random_generator.choice(["horizontal", "vertical"]))
    if route_axis == "horizontal":
        start = (-1.55, float(random_generator.uniform(-0.55, 0.55)))
        goal = (1.55, float(random_generator.uniform(-0.55, 0.55)))
    else:
        start = (float(random_generator.uniform(-0.70, 0.70)), -1.10)
        goal = (float(random_generator.uniform(-0.70, 0.70)), 1.10)
    if bool(random_generator.integers(0, 2)):
        start, goal = goal, start
    return start, goal


def test_every_formal_seed_has_a_static_map_reference_path() -> None:
    for seed in FORMAL_SCENARIO_SEEDS:
        start, goal = _route_endpoints(seed)
        for static_layout in ("train_a", "unseen_b"):
            assert formal_reference_path_length(static_layout, start, goal) > 0.0


def test_config_rejects_lane_protocol_drift(tmp_path: Path) -> None:
    raw = yaml.safe_load(FORMAL_CONFIG.read_text(encoding="utf-8"))
    raw["conditions"][2]["obstacle_cross_lanes"][4] = -0.60
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="cross lanes|protocol drift"):
        load_webots_formal_config(changed)


def test_world_contains_the_locked_dynamic_and_static_geometry() -> None:
    world = (REPOSITORY_ROOT / "worlds/webots_ppo_evaluate.wbt").read_text(
        encoding="utf-8"
    )
    assert world.startswith("#VRML_SIM R2025a utf8")
    for def_name in (*DYNAMIC_OBSTACLE_DEFS, *STATIC_OBSTACLE_DEFS):
        assert f"DEF {def_name} Solid" in world
    for radius in DYNAMIC_OBSTACLE_RADII:
        assert world.count(f"radius {radius:.2f}") >= 2
    for def_name, (rotation, size) in STATIC_NODE_GEOMETRY.items():
        assert f"DEF {def_name} Solid" in world
        assert f"rotation 0 0 1 {rotation[3]:.2f}" in world
        size_text = " ".join(f"{value:.2f}" for value in size)
        assert world.count(f"Box {{ size {size_text} }}") >= 2
