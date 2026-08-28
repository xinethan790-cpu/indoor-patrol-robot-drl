from __future__ import annotations

import numpy as np
import pytest

from indoor_patrol_rl.planning import (
    NoPathError,
    OccupancyGrid,
    PlannedPath,
    WaypointManager,
    plan_world_path,
)


def test_world_path_preserves_exact_endpoints_and_avoids_wall() -> None:
    occupied = np.zeros((9, 9), dtype=bool)
    occupied[4, 1:8] = True
    occupied[4, 6] = False
    grid = OccupancyGrid(occupied, resolution=0.5, origin=(-2.25, -2.25))
    start = (-1.75, -1.75)
    goal = (1.75, 1.75)

    path = plan_world_path(grid, start, goal)

    assert path.waypoints[0] == start
    assert path.waypoints[-1] == goal
    assert all(grid.is_free(cell) for cell in path.cells)
    assert (4, 6) in path.cells
    assert path.length > float(np.linalg.norm(np.subtract(goal, start)))


def test_world_path_rejects_occupied_goal() -> None:
    occupied = np.zeros((3, 3), dtype=bool)
    occupied[2, 2] = True
    grid = OccupancyGrid(occupied, resolution=1.0)

    with pytest.raises(NoPathError, match="occupied"):
        plan_world_path(grid, (0.5, 0.5), (2.5, 2.5))


def test_waypoint_manager_advances_in_order_and_only_completes_at_end() -> None:
    path = PlannedPath(
        start=(0.0, 0.0),
        goal=(1.0, 1.0),
        cells=((0, 0), (0, 1), (1, 1)),
        waypoints=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        length=2.0,
    )
    manager = WaypointManager(waypoint_radius=0.1, final_radius=0.15)

    initial = manager.reset(path)
    near_future_but_not_current = manager.update((0.95, 0.95))
    first_advance = manager.update((1.0, 0.0))
    completed = manager.update((1.0, 1.0))

    assert initial.target_index == 1
    assert not near_future_but_not_current.advanced
    assert near_future_but_not_current.target_index == 1
    assert first_advance.advanced and first_advance.target_index == 2
    assert not first_advance.completed
    assert completed.advanced and completed.completed


def test_waypoint_remaining_distance_is_continuous_at_corner() -> None:
    path = PlannedPath(
        start=(0.0, 0.0),
        goal=(1.0, 1.0),
        cells=((0, 0), (0, 1), (1, 1)),
        waypoints=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        length=2.0,
    )
    manager = WaypointManager(waypoint_radius=0.01, final_radius=0.1)
    manager.reset(path)

    before = manager.update((0.98, 0.0))
    after = manager.update((1.0, 0.0))

    assert before.remaining_distance == pytest.approx(1.02)
    assert after.remaining_distance == pytest.approx(1.0)
