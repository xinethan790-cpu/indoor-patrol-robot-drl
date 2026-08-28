"""A* search for known static indoor maps."""

from __future__ import annotations

import heapq
import itertools
import math

from indoor_patrol_rl.planning.grid import GridCell, OccupancyGrid


def _octile_distance(first: GridCell, second: GridCell) -> float:
    row_distance = abs(first[0] - second[0])
    column_distance = abs(first[1] - second[1])
    diagonal = min(row_distance, column_distance)
    straight = max(row_distance, column_distance) - diagonal
    return diagonal * math.sqrt(2.0) + straight


def _neighbors(grid: OccupancyGrid, cell: GridCell) -> list[tuple[GridCell, float]]:
    row, column = cell
    result: list[tuple[GridCell, float]] = []
    for row_offset, column_offset in itertools.product((-1, 0, 1), repeat=2):
        if row_offset == 0 and column_offset == 0:
            continue
        neighbor = row + row_offset, column + column_offset
        if not grid.is_free(neighbor):
            continue
        diagonal = row_offset != 0 and column_offset != 0
        if diagonal:
            horizontal = row, column + column_offset
            vertical = row + row_offset, column
            if not grid.is_free(horizontal) or not grid.is_free(vertical):
                continue
        result.append((neighbor, math.sqrt(2.0) if diagonal else 1.0))
    return result


def astar(grid: OccupancyGrid, start: GridCell, goal: GridCell) -> list[GridCell] | None:
    """Return an inclusive shortest path or ``None`` when no path exists."""
    if not grid.is_free(start) or not grid.is_free(goal):
        return None
    if start == goal:
        return [start]

    counter = itertools.count()
    frontier: list[tuple[float, int, GridCell]] = [(0.0, next(counter), start)]
    came_from: dict[GridCell, GridCell] = {}
    cost_so_far: dict[GridCell, float] = {start: 0.0}

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while current != start:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor, movement_cost in _neighbors(grid, current):
            new_cost = cost_so_far[current] + movement_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + _octile_distance(neighbor, goal)
                heapq.heappush(frontier, (priority, next(counter), neighbor))
                came_from[neighbor] = current
    return None


def simplify_path(path: list[GridCell]) -> list[GridCell]:
    """Remove intermediate collinear cells without changing the route geometry."""
    if len(path) <= 2:
        return list(path)

    simplified = [path[0]]
    previous_direction = _direction(path[0], path[1])
    for index in range(1, len(path) - 1):
        next_direction = _direction(path[index], path[index + 1])
        if next_direction != previous_direction:
            simplified.append(path[index])
        previous_direction = next_direction
    simplified.append(path[-1])
    return simplified


def _direction(first: GridCell, second: GridCell) -> GridCell:
    row_delta = second[0] - first[0]
    column_delta = second[1] - first[1]
    return int(math.copysign(1, row_delta)) if row_delta else 0, int(
        math.copysign(1, column_delta)
    ) if column_delta else 0
