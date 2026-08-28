"""Simulator-independent state machine for following an A* waypoint path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from indoor_patrol_rl.planning.grid import WorldPoint
from indoor_patrol_rl.planning.path import PlannedPath


@dataclass(frozen=True)
class WaypointUpdate:
    target: WorldPoint | None
    target_index: int
    advanced: bool
    completed: bool
    remaining_distance: float


class WaypointManager:
    """Advance monotonically through a planned path without skipping corners."""

    def __init__(self, waypoint_radius: float = 0.15, final_radius: float = 0.22) -> None:
        if min(waypoint_radius, final_radius) <= 0.0:
            raise ValueError("Waypoint acceptance radii must be positive.")
        self.waypoint_radius = waypoint_radius
        self.final_radius = final_radius
        self._path: PlannedPath | None = None
        self._target_index = -1
        self._completed = False

    def reset(self, path: PlannedPath) -> WaypointUpdate:
        if not path.waypoints:
            raise ValueError("A planned path must contain at least one waypoint.")
        self._path = path
        if len(path.waypoints) == 1:
            self._target_index = -1
            self._completed = True
        else:
            # The first waypoint is the exact planning start and is already reached.
            self._target_index = 1
            self._completed = False
        return self._snapshot(path.start, advanced=False)

    def update(self, position: WorldPoint) -> WaypointUpdate:
        if self._path is None:
            raise RuntimeError("WaypointManager.reset() must be called before update().")
        advanced = False
        if not self._completed:
            target = self._path.waypoints[self._target_index]
            radius = (
                self.final_radius
                if self._target_index == len(self._path.waypoints) - 1
                else self.waypoint_radius
            )
            if float(np.linalg.norm(np.subtract(position, target))) <= radius:
                advanced = True
                if self._target_index == len(self._path.waypoints) - 1:
                    self._target_index = -1
                    self._completed = True
                else:
                    self._target_index += 1
        return self._snapshot(position, advanced=advanced)

    def _snapshot(self, position: WorldPoint, *, advanced: bool) -> WaypointUpdate:
        if self._path is None:
            raise RuntimeError("WaypointManager has no active path.")
        if self._completed:
            return WaypointUpdate(None, -1, advanced, True, 0.0)

        target = self._path.waypoints[self._target_index]
        remaining = float(np.linalg.norm(np.subtract(position, target)))
        remaining += sum(
            float(np.linalg.norm(np.subtract(second, first)))
            for first, second in zip(
                self._path.waypoints[self._target_index :],
                self._path.waypoints[self._target_index + 1 :],
                strict=False,
            )
        )
        return WaypointUpdate(
            target=target,
            target_index=self._target_index,
            advanced=advanced,
            completed=False,
            remaining_distance=remaining,
        )
