"""Small dependency-free occupancy-grid representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

GridCell: TypeAlias = tuple[int, int]
WorldPoint: TypeAlias = tuple[float, float]


@dataclass(frozen=True)
class AxisAlignedRectangle:
    center: WorldPoint
    size: WorldPoint

    def __post_init__(self) -> None:
        if min(self.size) <= 0.0:
            raise ValueError("Rectangle dimensions must be positive.")


@dataclass(frozen=True)
class OccupancyGrid:
    """Boolean grid indexed as ``(row_y, column_x)``."""

    occupied: np.ndarray
    resolution: float
    origin: WorldPoint = (0.0, 0.0)

    def __post_init__(self) -> None:
        data = np.asarray(self.occupied, dtype=bool)
        if data.ndim != 2:
            raise ValueError("occupied must be a two-dimensional array.")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive.")
        object.__setattr__(self, "occupied", data.copy())

    @property
    def height(self) -> int:
        return int(self.occupied.shape[0])

    @property
    def width(self) -> int:
        return int(self.occupied.shape[1])

    def in_bounds(self, cell: GridCell) -> bool:
        row, column = cell
        return 0 <= row < self.height and 0 <= column < self.width

    def is_free(self, cell: GridCell) -> bool:
        return self.in_bounds(cell) and not bool(self.occupied[cell])

    def world_to_grid(self, point: WorldPoint) -> GridCell:
        x, y = point
        column = int(np.floor((x - self.origin[0]) / self.resolution))
        row = int(np.floor((y - self.origin[1]) / self.resolution))
        return row, column

    def grid_to_world(self, cell: GridCell) -> WorldPoint:
        if not self.in_bounds(cell):
            raise ValueError(f"Cell {cell} is outside the grid.")
        row, column = cell
        x = self.origin[0] + (column + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return float(x), float(y)

    def inflate(self, radius: float) -> OccupancyGrid:
        """Inflate obstacles by an Euclidean disk with the given world radius."""
        if radius < 0.0:
            raise ValueError("Inflation radius cannot be negative.")
        if radius == 0.0 or not self.occupied.any():
            return OccupancyGrid(self.occupied, self.resolution, self.origin)

        cells = int(np.ceil(radius / self.resolution))
        offsets = [
            (row_offset, column_offset)
            for row_offset in range(-cells, cells + 1)
            for column_offset in range(-cells, cells + 1)
            if np.hypot(row_offset, column_offset) * self.resolution <= radius + 1e-9
        ]
        inflated = self.occupied.copy()
        for row, column in np.argwhere(self.occupied):
            for row_offset, column_offset in offsets:
                candidate = int(row + row_offset), int(column + column_offset)
                if self.in_bounds(candidate):
                    inflated[candidate] = True
        return OccupancyGrid(inflated, self.resolution, self.origin)


def grid_from_rectangles(
    *,
    origin: WorldPoint,
    size: WorldPoint,
    resolution: float,
    rectangles: list[AxisAlignedRectangle] | tuple[AxisAlignedRectangle, ...],
    occupy_border: bool = True,
) -> OccupancyGrid:
    """Rasterize known axis-aligned static geometry into a Boolean grid."""

    if min(size) <= 0.0 or resolution <= 0.0:
        raise ValueError("Map size and resolution must be positive.")
    width = int(np.ceil(size[0] / resolution))
    height = int(np.ceil(size[1] / resolution))
    x_centers = origin[0] + (np.arange(width) + 0.5) * resolution
    y_centers = origin[1] + (np.arange(height) + 0.5) * resolution
    occupied = np.zeros((height, width), dtype=bool)
    for rectangle in rectangles:
        x_mask = np.abs(x_centers - rectangle.center[0]) <= rectangle.size[0] / 2.0
        y_mask = np.abs(y_centers - rectangle.center[1]) <= rectangle.size[1] / 2.0
        occupied[np.ix_(y_mask, x_mask)] = True
    if occupy_border:
        occupied[[0, -1], :] = True
        occupied[:, [0, -1]] = True
    return OccupancyGrid(occupied, resolution, origin)
