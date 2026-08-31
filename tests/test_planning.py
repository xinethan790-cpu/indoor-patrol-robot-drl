from __future__ import annotations

import numpy as np
import pytest

from indoor_patrol_rl.planning import (
    AxisAlignedRectangle,
    OccupancyGrid,
    astar,
    grid_from_rectangles,
    grid_from_rectangles_with_clearances,
    simplify_path,
)


def test_astar_finds_path_through_gap() -> None:
    occupied = np.zeros((7, 7), dtype=bool)
    occupied[3, :] = True
    occupied[3, 4] = False
    grid = OccupancyGrid(occupied, resolution=0.1)

    path = astar(grid, (1, 1), (5, 5))

    assert path is not None
    assert path[0] == (1, 1)
    assert path[-1] == (5, 5)
    assert (3, 4) in path
    assert all(grid.is_free(cell) for cell in path)


def test_astar_does_not_cut_blocked_corner() -> None:
    occupied = np.array([[False, True], [True, False]], dtype=bool)
    grid = OccupancyGrid(occupied, resolution=1.0)
    assert astar(grid, (0, 0), (1, 1)) is None


def test_grid_coordinate_round_trip_uses_cell_center() -> None:
    grid = OccupancyGrid(np.zeros((4, 5), dtype=bool), resolution=0.5, origin=(-1.0, -2.0))
    cell = grid.world_to_grid((-0.24, -0.74))
    assert cell == (2, 1)
    assert grid.world_to_grid(grid.grid_to_world(cell)) == cell


def test_inflation_marks_disk_around_obstacle() -> None:
    occupied = np.zeros((7, 7), dtype=bool)
    occupied[3, 3] = True
    grid = OccupancyGrid(occupied, resolution=0.1).inflate(0.15)
    assert grid.occupied[3, 3]
    assert grid.occupied[3, 4]
    assert grid.occupied[4, 3]
    assert not grid.occupied[1, 1]


def test_simplify_path_keeps_turns() -> None:
    path = [(1, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
    assert simplify_path(path) == [(1, 1), (1, 3), (3, 5), (4, 5)]


def test_static_rectangles_and_map_border_are_rasterized() -> None:
    grid = grid_from_rectangles(
        origin=(-2.0, -1.5),
        size=(4.0, 3.0),
        resolution=0.5,
        rectangles=[AxisAlignedRectangle(center=(0.0, 0.0), size=(1.0, 1.0))],
    )

    assert grid.occupied[0, :].all()
    assert grid.occupied[-1, :].all()
    assert grid.occupied[:, 0].all()
    assert grid.occupied[:, -1].all()
    assert not grid.is_free(grid.world_to_grid((0.0, 0.0)))
    assert grid.is_free(grid.world_to_grid((-1.25, -0.75)))


def test_static_obstacles_and_border_use_independent_clearances() -> None:
    grid = grid_from_rectangles_with_clearances(
        origin=(-2.0, -2.0),
        size=(4.0, 4.0),
        resolution=0.1,
        rectangles=[AxisAlignedRectangle(center=(0.0, 0.0), size=(1.2, 0.8))],
        obstacle_inflation_radius=0.38,
        border_inflation_radius=0.15,
    )

    # The cabinet reaches column 25; 0.38 m adds three 0.1 m cells, not four.
    assert grid.occupied[20, 28]
    assert not grid.occupied[20, 29]
    # The left border occupies column 0; 0.15 m adds one cell, not three.
    assert grid.occupied[30, 1]
    assert not grid.occupied[30, 2]


def test_zero_separate_clearances_match_original_grid_api() -> None:
    arguments = {
        "origin": (-2.0, -1.5),
        "size": (4.0, 3.0),
        "resolution": 0.5,
        "rectangles": [AxisAlignedRectangle(center=(0.0, 0.0), size=(1.0, 1.0))],
    }

    original = grid_from_rectangles(**arguments)
    separated = grid_from_rectangles_with_clearances(
        **arguments,
        obstacle_inflation_radius=0.0,
        border_inflation_radius=0.0,
    )

    assert separated.origin == original.origin
    assert separated.resolution == original.resolution
    np.testing.assert_array_equal(separated.occupied, original.occupied)


@pytest.mark.parametrize("invalid_radius", [True, "0.15", None])
@pytest.mark.parametrize(
    "radius_name",
    ["obstacle_inflation_radius", "border_inflation_radius"],
)
def test_separate_clearances_reject_non_numeric_radii(
    invalid_radius: object,
    radius_name: str,
) -> None:
    radii: dict[str, object] = {
        "obstacle_inflation_radius": 0.38,
        "border_inflation_radius": 0.15,
    }
    radii[radius_name] = invalid_radius

    with pytest.raises(TypeError, match=radius_name):
        grid_from_rectangles_with_clearances(
            origin=(-1.0, -1.0),
            size=(2.0, 2.0),
            resolution=0.1,
            rectangles=(),
            **radii,
        )


@pytest.mark.parametrize("invalid_radius", [-0.01, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "radius_name",
    ["obstacle_inflation_radius", "border_inflation_radius"],
)
def test_separate_clearances_reject_invalid_numeric_radii(
    invalid_radius: float,
    radius_name: str,
) -> None:
    radii = {
        "obstacle_inflation_radius": 0.38,
        "border_inflation_radius": 0.15,
    }
    radii[radius_name] = invalid_radius

    with pytest.raises(ValueError, match=radius_name):
        grid_from_rectangles_with_clearances(
            origin=(-1.0, -1.0),
            size=(2.0, 2.0),
            resolution=0.1,
            rectangles=(),
            **radii,
        )
