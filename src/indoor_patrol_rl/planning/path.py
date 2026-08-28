"""World-coordinate paths built on top of the occupancy-grid A* search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from indoor_patrol_rl.planning.astar import astar, simplify_path
from indoor_patrol_rl.planning.grid import GridCell, OccupancyGrid, WorldPoint


class NoPathError(RuntimeError):
    """Raised when a requested world-coordinate path cannot be planned."""


@dataclass(frozen=True)
class PlannedPath:
    """Dense A* cells and simplified world-coordinate tracking waypoints."""

    start: WorldPoint
    goal: WorldPoint
    cells: tuple[GridCell, ...]
    waypoints: tuple[WorldPoint, ...]
    length: float


def plan_world_path(
    grid: OccupancyGrid,
    start: WorldPoint,
    goal: WorldPoint,
) -> PlannedPath:
    """Plan from exact world start to goal without silently moving invalid points."""

    start_cell = grid.world_to_grid(start)
    goal_cell = grid.world_to_grid(goal)
    if not grid.in_bounds(start_cell):
        raise NoPathError(f"Start point {start} is outside the map.")
    if not grid.in_bounds(goal_cell):
        raise NoPathError(f"Goal point {goal} is outside the map.")
    if not grid.is_free(start_cell):
        raise NoPathError(f"Start point {start} lies in an occupied cell.")
    if not grid.is_free(goal_cell):
        raise NoPathError(f"Goal point {goal} lies in an occupied cell.")

    cells = astar(grid, start_cell, goal_cell)
    if cells is None:
        raise NoPathError(f"No path exists from {start} to {goal}.")

    simplified_cells = simplify_path(cells)
    if len(simplified_cells) == 1:
        waypoint_list = [start]
        if not np.allclose(start, goal):
            waypoint_list.append(goal)
    else:
        waypoint_list = [grid.grid_to_world(cell) for cell in simplified_cells]
        waypoint_list[0] = (float(start[0]), float(start[1]))
        waypoint_list[-1] = (float(goal[0]), float(goal[1]))

    length = float(
        sum(
            np.linalg.norm(np.subtract(second, first))
            for first, second in zip(waypoint_list, waypoint_list[1:], strict=False)
        )
    )
    return PlannedPath(
        start=(float(start[0]), float(start[1])),
        goal=(float(goal[0]), float(goal[1])),
        cells=tuple(cells),
        waypoints=tuple(waypoint_list),
        length=length,
    )
