"""Move a simulated pedestrian repeatedly across the patrol route."""

from __future__ import annotations

from controller import Supervisor


def main() -> None:
    obstacle = Supervisor()
    time_step = int(obstacle.getBasicTimeStep())
    translation = obstacle.getSelf().getField("translation")
    speed = 0.34
    direction = 1.0
    lower = -1.05
    upper = 1.05

    while obstacle.step(time_step) != -1:
        position = translation.getSFVec3f()
        position[1] += direction * speed * time_step / 1000.0
        if position[1] >= upper:
            position[1] = upper
            direction = -1.0
        elif position[1] <= lower:
            position[1] = lower
            direction = 1.0
        translation.setSFVec3f(position)


if __name__ == "__main__":
    main()
