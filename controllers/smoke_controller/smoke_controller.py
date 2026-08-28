"""Minimal Webots Python/Supervisor compatibility test."""

from controller import Supervisor


def main() -> None:
    robot = Supervisor()
    time_step = int(robot.getBasicTimeStep())
    translation = robot.getSelf().getField("translation")

    for index in range(20):
        if robot.step(time_step) == -1:
            raise RuntimeError("Webots ended before the smoke test completed.")
        position = translation.getSFVec3f()
        position[0] = 0.02 * index
        translation.setSFVec3f(position)

    print("WEBOTS_SMOKE_OK steps=20 supervisor=true python=true", flush=True)
    robot.step(time_step)
    robot.simulationQuit(0)


if __name__ == "__main__":
    main()
