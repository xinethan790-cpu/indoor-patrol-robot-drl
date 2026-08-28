from __future__ import annotations

import numpy as np

from indoor_patrol_rl.planning import (
    AxisAlignedRectangle,
    OccupancyGrid,
    astar,
    grid_from_rectangles,
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
