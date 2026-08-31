"""A* global routing, local waypoints, and PPO dynamic avoidance in Webots.

The controller executes one continuous four-checkpoint patrol.  It is separate
from the episodic training controller because changing a local waypoint is not
an environment reset.  Every sample written by this file comes from the live
Webots simulation loop.
"""

from __future__ import annotations

import math
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from controller import Supervisor
from stable_baselines3 import PPO

from indoor_patrol_rl.application import (
    PatrolLiveAlert,
    PatrolLiveOutcome,
    PatrolLiveReadings,
    PatrolLiveSnapshot,
    PatrolLiveStatusWriter,
    PatrolRuntimeIdentity,
    PatrolTaskCheckpoint,
    PatrolTaskMachine,
    PatrolTaskState,
    validate_checkpoint_layout,
)
from indoor_patrol_rl.core import DifferentialDriveMapper, ObservationBuilder, VelocityRateLimiter
from indoor_patrol_rl.envs.webots_patrol import _webots_scan_to_policy_order
from indoor_patrol_rl.evaluation import (
    PatrolCheckpointRecord,
    PatrolCollisionRecord,
    PatrolRunSummary,
    PatrolTrajectorySample,
    portable_model_path,
    select_patrol_model,
    write_patrol_outputs,
)
from indoor_patrol_rl.inspection import (
    EnvironmentalInspection,
    load_inspection_config,
    write_inspection_bundle,
)
from indoor_patrol_rl.planning import (
    AxisAlignedRectangle,
    PlannedPath,
    WaypointManager,
    grid_from_rectangles,
    plan_world_path,
)

_emergency_live_failure: Callable[[], None] | None = None


@dataclass
class MovingObstacleControl:
    translation: Any
    axis: int
    lower: float
    upper: float
    speed: float
    direction: float


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def environment_flag(name: str, *, default: bool = False) -> bool:
    """Parse a strict environment boolean for deployment-safety switches."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off.")


def main() -> None:
    global _emergency_live_failure
    _emergency_live_failure = None
    repository_root = Path(__file__).absolute().parents[2]
    supervisor = Supervisor()
    try:
        run_patrol(supervisor, repository_root)
    except Exception:
        traceback.print_exc()
        if _emergency_live_failure is not None:
            try:
                _emergency_live_failure()
            except Exception:
                traceback.print_exc()
        _stop_robot(supervisor)
        supervisor.step(int(supervisor.getBasicTimeStep()))
        supervisor.simulationQuit(1)


def run_patrol(supervisor: Supervisor, repository_root: Path) -> None:
    seed = int(os.environ.get("WEBOTS_PATROL_SEED", "7000"))
    max_control_steps = int(os.environ.get("WEBOTS_PATROL_MAX_STEPS", "3600"))
    action_repeat = int(os.environ.get("WEBOTS_PATROL_ACTION_REPEAT", "4"))
    if min(max_control_steps, action_repeat) <= 0:
        raise ValueError("Patrol step limits must be positive.")
    random = np.random.default_rng(seed)

    route_path = repository_root / "configs/maps/patrol_route.yaml"
    with route_path.open(encoding="utf-8") as stream:
        route_config = yaml.safe_load(stream)
    map_config = route_config["map"]
    rectangles = [
        AxisAlignedRectangle(tuple(item["center"]), tuple(item["size"]))
        for item in map_config["static_rectangles"]
    ]
    grid = grid_from_rectangles(
        origin=tuple(map_config["origin"]),
        size=tuple(map_config["size"]),
        resolution=float(map_config["resolution"]),
        rectangles=rectangles,
    ).inflate(float(map_config["inflation_radius"]))
    checkpoints = [
        (str(item["id"]), (float(item["position"][0]), float(item["position"][1])))
        for item in route_config["checkpoints"]
    ]
    configured_start = tuple(float(value) for value in route_config["robot_start"])
    nominal_route_length = _nominal_route_length(grid, configured_start, checkpoints)

    basic_time_step = int(supervisor.getBasicTimeStep())
    control_dt = basic_time_step * action_repeat / 1000.0
    lidar = supervisor.getDevice("LDS-01")
    gps = supervisor.getDevice("gps")
    imu = supervisor.getDevice("imu")
    for sensor in (lidar, gps, imu):
        sensor.enable(basic_time_step)
    left_motor = supervisor.getDevice("left wheel motor")
    right_motor = supervisor.getDevice("right wheel motor")
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    robot_node = supervisor.getSelf()
    local_target_node = supervisor.getFromDef("LOCAL_TARGET")
    if local_target_node is None:
        raise RuntimeError("The Webots world is missing DEF LOCAL_TARGET.")
    local_target_field = local_target_node.getField("translation")
    _validate_route_checkpoint_nodes(supervisor, checkpoints)
    moving_obstacles = _initialize_obstacles(supervisor, random)

    for _ in range(2):
        if supervisor.step(basic_time_step) == -1:
            raise RuntimeError("Webots ended while initializing the patrol controller.")

    explicit_model = os.environ.get("WEBOTS_PATROL_MODEL")
    selection_manifest = os.environ.get(
        "WEBOTS_PATROL_SELECTION_MANIFEST",
        "artifacts/results/webots_validation_w1_selection_seed11/selection.json",
    )
    model_selection = select_patrol_model(
        repository_root,
        explicit_model,
        selection_manifest=selection_manifest or None,
        allow_unvalidated_fallback=environment_flag(
            "WEBOTS_PATROL_ALLOW_UNVALIDATED_MODEL"
        ),
    )
    model = PPO.load(model_selection.path, device="cpu")
    mapper = DifferentialDriveMapper(0.22, 1.5, 0.033, 0.16, 6.0)
    limiter = VelocityRateLimiter(0.50, 3.0, control_dt)
    observation_builder = ObservationBuilder(36, float(lidar.getMaxRange()), 4.0)
    if model.observation_space.shape != observation_builder.space.shape:
        raise ValueError(
            "The selected PPO observation shape is incompatible with the Webots patrol: "
            f"{model.observation_space.shape} != {observation_builder.space.shape}."
        )
    if model.action_space.shape != (2,):
        raise ValueError(
            "The selected PPO action shape must be (2,), "
            f"got {model.action_space.shape}."
        )

    run_id = os.environ.get(
        "WEBOTS_PATROL_RUN_ID",
        f"webots-patrol-seed{seed}-{time.time_ns()}",
    ).strip()
    if not run_id:
        raise ValueError("WEBOTS_PATROL_RUN_ID must be non-empty.")
    if any(character in run_id for character in "/\\"):
        raise ValueError("WEBOTS_PATROL_RUN_ID cannot contain path separators.")
    inspection_config_value = os.environ.get(
        "WEBOTS_PATROL_INSPECTION_CONFIG",
        "configs/inspection/environmental_patrol_v1.yaml",
    )
    inspection_config_path = Path(inspection_config_value).expanduser()
    if not inspection_config_path.is_absolute():
        inspection_config_path = repository_root / inspection_config_path
    inspection_config_path = inspection_config_path.resolve()
    inspection_config = load_inspection_config(inspection_config_path)
    required_live_sensor_ids = {
        "temperature",
        "smoke_obscuration",
        "carbon_monoxide",
    }
    configured_sensor_ids = {sensor.id for sensor in inspection_config.sensors}
    missing_live_sensor_ids = required_live_sensor_ids - configured_sensor_ids
    if missing_live_sensor_ids:
        raise ValueError(
            "The patrol dashboard inspection config is missing required sensors: "
            f"{sorted(missing_live_sensor_ids)}."
        )
    inspection = EnvironmentalInspection(
        inspection_config,
        run_id=run_id,
        scenario_seed=seed,
    )
    task_timeout_s = float(
        os.environ.get(
            "WEBOTS_PATROL_TIMEOUT_S",
            str(
                max_control_steps * control_dt
                + len(checkpoints) * (inspection.sampling_duration_s + control_dt)
            ),
        )
    )
    task_machine = PatrolTaskMachine(
        [
            PatrolTaskCheckpoint(checkpoint_id, checkpoint_position)
            for checkpoint_id, checkpoint_position in checkpoints
        ],
        arrival_radius_m=0.22,
        dwell_time_s=inspection.sampling_duration_s,
        timeout_s=task_timeout_s,
    )

    output_value = os.environ.get(
        "WEBOTS_PATROL_OUTPUT_DIR",
        f"artifacts/results/{run_id}",
    )
    output_directory = Path(output_value).expanduser()
    if not output_directory.is_absolute():
        output_directory = repository_root / output_directory
    output_directory = output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            "Patrol output directory already contains files; choose a new run: "
            f"{output_directory}"
        )

    print(
        f"PATROL_MODEL path={portable_model_path(model_selection.path, repository_root)} "
        f"sha256={model_selection.model_sha256} "
        f"training_domain={model_selection.training_domain} "
        f"reason={model_selection.selection_reason}",
        flush=True,
    )
    if (
        model_selection.training_domain != "webots"
        or model_selection.selection_reason.startswith("UNVALIDATED")
    ):
        print(
            "PATROL_MODEL_WARNING selected model is not a formal validation selection; "
            "the generated files report only this Webots run's measured behavior.",
            flush=True,
        )

    def position() -> tuple[float, float]:
        values = gps.getValues()
        return float(values[0]), float(values[1])

    checkpoint_index = 0
    active_path = plan_world_path(grid, position(), checkpoints[checkpoint_index][1])
    waypoint_manager = WaypointManager(waypoint_radius=0.16, final_radius=0.22)
    waypoint_state = waypoint_manager.reset(active_path)
    planned_segment_lengths = [active_path.length]
    print(
        f"PPO_ROUTE_PLANNED checkpoint={checkpoints[0][0]} cells={len(active_path.cells)} "
        f"waypoints={len(active_path.waypoints)} length={active_path.length:.3f}",
        flush=True,
    )

    initial_scan = _range_image(lidar)
    observation_builder.reset(initial_scan)
    current_linear = 0.0
    current_angular = 0.0
    previous_action = np.array([-1.0, 0.0], dtype=np.float32)
    previous_position = np.asarray(position(), dtype=np.float64)
    total_path_length = 0.0
    segment_path_length = 0.0
    minimum_lidar_range = _minimum(initial_scan)
    minimum_clearance = max(0.0, minimum_lidar_range - 0.10)
    segment_minimum_clearance = minimum_clearance
    physics_step_count = 0
    segment_start_step = 0
    trajectory: list[PatrolTrajectorySample] = []
    checkpoint_records: list[PatrolCheckpointRecord] = []
    collision_records: list[PatrolCollisionRecord] = []
    inference_latencies_ms: list[float] = []
    live_writer = PatrolLiveStatusWriter(output_directory, run_id=run_id)
    runtime_identity = PatrolRuntimeIdentity(
        model_id=portable_model_path(model_selection.path, repository_root),
        model_sha256=model_selection.model_sha256,
        safety_shield_enabled=False,
    )
    live_sequence = -1
    latest_live_readings = PatrolLiveReadings()
    active_live_alerts: dict[str, PatrolLiveAlert] = {}

    def publish_live(
        *,
        state: PatrolTaskState,
        outcome: PatrolLiveOutcome,
        control_step: int,
        simulation_time_s: float,
        current_position: tuple[float, float],
        local_goal: tuple[float, float] | None = None,
        remaining_distance_m: float | None = None,
    ) -> None:
        nonlocal live_sequence
        completed_count = len(task_machine.completed_checkpoint_ids)
        current_checkpoint = task_machine.current_checkpoint
        live_sequence += 1
        live_writer.publish(
            PatrolLiveSnapshot(
                run_id=run_id,
                sequence=live_sequence,
                state=state,
                outcome=outcome,
                seed=seed,
                control_step=control_step,
                simulation_time_s=simulation_time_s,
                position=(float(current_position[0]), float(current_position[1])),
                current_checkpoint_id=(
                    None if current_checkpoint is None else current_checkpoint.checkpoint_id
                ),
                current_checkpoint_ordinal=(
                    None if current_checkpoint is None else completed_count + 1
                ),
                completed_checkpoint_count=completed_count,
                checkpoint_count=len(checkpoints),
                progress_ratio=completed_count / len(checkpoints),
                local_goal=local_goal,
                remaining_distance_m=remaining_distance_m,
                latest_readings=latest_live_readings,
                active_alerts=tuple(
                    active_live_alerts[rule.id]
                    for rule in inspection.config.alarm_rules
                    if rule.id in active_live_alerts
                ),
                runtime_identity=runtime_identity,
            )
        )

    def update_live_inspection(inspection_result: Any) -> None:
        nonlocal latest_live_readings
        latest_by_sensor = {
            sample.sensor_id: sample for sample in inspection_result.samples
        }
        if not required_live_sensor_ids.issubset(latest_by_sensor):
            raise RuntimeError(
                "The live dashboard requires temperature, smoke, and carbon-monoxide "
                "readings from every inspection visit."
            )
        latest_live_readings = PatrolLiveReadings(
            temperature_c=latest_by_sensor["temperature"].observed_value,
            smoke_obscuration_pct_per_m=(
                latest_by_sensor["smoke_obscuration"].observed_value
            ),
            carbon_monoxide_ppm=(
                latest_by_sensor["carbon_monoxide"].observed_value
            ),
            sampled_at_s=max(
                sample.simulation_time_s for sample in inspection_result.samples
            ),
            checkpoint_id=inspection_result.visit.checkpoint_id,
        )
        for event in inspection_result.events:
            if event.event_type == "raised":
                active_live_alerts[event.rule_id] = PatrolLiveAlert(
                    rule_id=event.rule_id,
                    sensor_id=event.sensor_id,
                    severity=event.severity,
                    observed_value=event.observed_value,
                    threshold=event.threshold,
                    unit=event.unit,
                    message=event.message,
                    raised_at_s=event.simulation_time_s,
                )
            else:
                active_live_alerts.pop(event.rule_id, None)

    def publish_unhandled_failure() -> None:
        last_snapshot = live_writer.last_snapshot
        if last_snapshot is None or last_snapshot.state in {
            PatrolTaskState.COMPLETED,
            PatrolTaskState.FAILED,
        }:
            return
        publish_live(
            state=PatrolTaskState.FAILED,
            outcome=PatrolLiveOutcome.ERROR,
            control_step=last_snapshot.control_step,
            simulation_time_s=last_snapshot.simulation_time_s,
            current_position=last_snapshot.position,
        )

    global _emergency_live_failure
    _emergency_live_failure = publish_unhandled_failure

    def finish_run(
        *,
        outcome: str,
        control_step_count: int,
        exit_code: int,
        failure_reason: str | None = None,
    ) -> None:
        _stop_robot(supervisor)
        terminal_state = (
            PatrolTaskState.COMPLETED
            if outcome == "completed"
            else PatrolTaskState.FAILED
        )
        terminal_outcome = {
            "completed": PatrolLiveOutcome.SUCCESS,
            "collision": PatrolLiveOutcome.COLLISION,
            "timeout": PatrolLiveOutcome.TIMEOUT,
            "error": (
                PatrolLiveOutcome.INSPECTION_FAILED
                if failure_reason is not None
                and failure_reason.startswith("inspection sampling failed")
                else PatrolLiveOutcome.ERROR
            ),
        }[outcome]
        output_checkpoints = list(checkpoint_records)
        recorded_ids = {record.checkpoint_id for record in output_checkpoints}
        for index, (checkpoint_id, target) in enumerate(checkpoints):
            if checkpoint_id in recorded_ids:
                continue
            is_active = index == checkpoint_index and checkpoint_index < len(checkpoints)
            planned_length = active_path.length if is_active else None
            actual_length = segment_path_length if is_active else 0.0
            segment_steps = control_step_count - segment_start_step if is_active else 0
            output_checkpoints.append(
                PatrolCheckpointRecord(
                    checkpoint_id=checkpoint_id,
                    visit_order=index + 1,
                    target_x_m=target[0],
                    target_y_m=target[1],
                    reached=False,
                    reached_control_step=None,
                    arrival_simulation_time_s=None,
                    segment_control_steps=max(0, segment_steps),
                    actual_segment_path_m=actual_length,
                    planned_segment_path_m=planned_length,
                    segment_path_ratio=None,
                    segment_minimum_clearance_m=(
                        segment_minimum_clearance
                        if is_active and math.isfinite(segment_minimum_clearance)
                        else None
                    ),
                )
            )
        output_checkpoints.sort(key=lambda record: record.visit_order)
        visited = sum(record.reached for record in output_checkpoints)
        completed_ratio = None
        attempted_reference = float(sum(planned_segment_lengths))
        if outcome == "completed" and attempted_reference > 0.0:
            completed_ratio = total_path_length / attempted_reference
        latencies = np.asarray(inference_latencies_ms, dtype=np.float64)
        latency_p50 = float(np.percentile(latencies, 50)) if latencies.size else 0.0
        latency_p95 = float(np.percentile(latencies, 95)) if latencies.size else 0.0
        summary = PatrolRunSummary(
            schema_version="1.0",
            run_id=run_id,
            backend="Webots R2025a",
            planner="astar_local_waypoint_ppo",
            scenario_seed=seed,
            model_path=portable_model_path(model_selection.path, repository_root),
            model_sha256=model_selection.model_sha256,
            model_training_domain=model_selection.training_domain,
            model_selection_reason=model_selection.selection_reason,
            outcome=outcome,  # type: ignore[arg-type]
            failure_reason=failure_reason,
            checkpoints_total=len(checkpoints),
            checkpoints_visited=visited,
            completion_ratio=visited / len(checkpoints),
            collision_count=len(collision_records),
            control_step_count=control_step_count,
            physics_step_count=physics_step_count,
            elapsed_simulation_time_s=physics_step_count * basic_time_step / 1000.0,
            actual_path_length_m=total_path_length,
            planned_attempted_path_length_m=attempted_reference,
            nominal_route_length_m=nominal_route_length,
            completed_path_ratio=completed_ratio,
            minimum_lidar_range_m=minimum_lidar_range,
            minimum_clearance_m=minimum_clearance,
            inference_latency_p50_ms=latency_p50,
            inference_latency_p95_ms=latency_p95,
            trajectory_file="trajectory.csv",
            checkpoints_file="checkpoints.csv",
            collisions_file="collisions.csv",
        )
        try:
            paths = write_patrol_outputs(
                output_directory,
                trajectory=trajectory,
                checkpoints=output_checkpoints,
                collisions=collision_records,
                summary=summary,
            )
            inspection_paths = write_inspection_bundle(
                output_directory,
                inspection,
                inspection_config_path,
                expected_checkpoint_count=len(checkpoints),
                patrol_summary_path=paths["summary"],
            )
        except Exception:
            try:
                publish_live(
                    state=PatrolTaskState.FAILED,
                    outcome=PatrolLiveOutcome.ERROR,
                    control_step=control_step_count,
                    simulation_time_s=physics_step_count * basic_time_step / 1000.0,
                    current_position=position(),
                )
            except Exception:
                pass
            raise
        publish_live(
            state=terminal_state,
            outcome=terminal_outcome,
            control_step=control_step_count,
            simulation_time_s=physics_step_count * basic_time_step / 1000.0,
            current_position=position(),
        )
        print(
            f"WEBOTS_PATROL_FINISHED outcome={outcome} visited={visited}/{len(checkpoints)} "
            f"collisions={len(collision_records)} path={total_path_length:.3f} "
            f"samples={len(inspection.samples)} alarms={len(inspection.events)} "
            f"results={inspection_paths['summary'].parent}",
            flush=True,
        )
        supervisor.step(basic_time_step)
        _finish_simulation(supervisor, exit_code)

    initial_goal = waypoint_state.target
    publish_live(
        state=PatrolTaskState.NAVIGATING,
        outcome=PatrolLiveOutcome.RUNNING,
        control_step=0,
        simulation_time_s=0.0,
        current_position=(float(previous_position[0]), float(previous_position[1])),
        local_goal=initial_goal,
        remaining_distance_m=(
            None
            if initial_goal is None
            else _remaining_path_distance(
                active_path,
                waypoint_state.target_index,
                (float(previous_position[0]), float(previous_position[1])),
            )
        ),
    )

    for control_step in range(1, max_control_steps + 1):
        current_position = position()
        waypoint_state = waypoint_manager.update(current_position)
        simulation_time_s = physics_step_count * basic_time_step / 1000.0
        task_update = task_machine.update(
            position=current_position,
            collision=False,
            simulation_time_s=simulation_time_s,
            control_step=control_step - 1,
        )
        if task_update.state is PatrolTaskState.FAILED:
            finish_run(
                outcome="timeout",
                control_step_count=control_step - 1,
                exit_code=3,
                failure_reason=task_update.failure_reason,
            )
            return
        if task_update.state is PatrolTaskState.DWELLING:
            if not waypoint_state.completed:
                raise RuntimeError(
                    "Patrol task entered DWELLING before the active waypoint path completed."
                )
            checkpoint_id, target = checkpoints[checkpoint_index]
            completed_actions = control_step - 1
            arrival_time_s = simulation_time_s
            arrival_position = current_position
            _stop_robot(supervisor)
            current_linear = 0.0
            current_angular = 0.0
            previous_action = np.array([-1.0, 0.0], dtype=np.float32)
            publish_live(
                state=PatrolTaskState.DWELLING,
                outcome=PatrolLiveOutcome.RUNNING,
                control_step=completed_actions,
                simulation_time_s=simulation_time_s,
                current_position=arrival_position,
            )
            print(
                f"PPO_POINT_ARRIVED id={checkpoint_id} step={completed_actions} "
                f"dwell={inspection.sampling_duration_s:.3f}",
                flush=True,
            )

            dwell_aborted = False
            while task_update.state is PatrolTaskState.DWELLING:
                _move_obstacles(moving_obstacles, basic_time_step)
                if supervisor.step(basic_time_step) == -1:
                    raise RuntimeError("Webots ended during checkpoint inspection dwell.")
                physics_step_count += 1
                dwell_position = np.asarray(position(), dtype=np.float64)
                traveled = float(np.linalg.norm(dwell_position - previous_position))
                total_path_length += traveled
                segment_path_length += traveled
                previous_position = dwell_position
                dwell_scan = _range_image(lidar)
                dwell_lidar_minimum = _minimum(dwell_scan)
                contact_collision = _has_contact_collision(robot_node)
                lidar_collision = (
                    dwell_lidar_minimum <= float(lidar.getMinRange()) + 0.01
                )
                collision = bool(contact_collision or lidar_collision)
                clearance = 0.0 if collision else max(0.0, dwell_lidar_minimum - 0.10)
                minimum_lidar_range = min(minimum_lidar_range, dwell_lidar_minimum)
                minimum_clearance = min(minimum_clearance, clearance)
                segment_minimum_clearance = min(segment_minimum_clearance, clearance)
                simulation_time_s = physics_step_count * basic_time_step / 1000.0
                task_update = task_machine.update(
                    position=(float(dwell_position[0]), float(dwell_position[1])),
                    collision=collision,
                    simulation_time_s=simulation_time_s,
                    control_step=completed_actions,
                )
                if collision:
                    detection_source = "contact" if contact_collision else "lidar_threshold"
                    collision_records.append(
                        PatrolCollisionRecord(
                            collision_index=len(collision_records) + 1,
                            control_step=completed_actions,
                            simulation_time_s=simulation_time_s,
                            position_x_m=float(dwell_position[0]),
                            position_y_m=float(dwell_position[1]),
                            detection_source=detection_source,
                            minimum_lidar_range_m=dwell_lidar_minimum,
                        )
                    )
                    finish_run(
                        outcome="collision",
                        control_step_count=completed_actions,
                        exit_code=2,
                        failure_reason=f"collision during inspection dwell by {detection_source}",
                    )
                    return
                if task_update.state is PatrolTaskState.FAILED:
                    finish_run(
                        outcome="timeout",
                        control_step_count=completed_actions,
                        exit_code=3,
                        failure_reason=task_update.failure_reason,
                    )
                    return
                if task_update.state is PatrolTaskState.NAVIGATING:
                    current_position = (float(dwell_position[0]), float(dwell_position[1]))
                    active_path = plan_world_path(grid, current_position, target)
                    waypoint_state = waypoint_manager.reset(active_path)
                    observation_builder.reset(dwell_scan)
                    drift_goal = waypoint_state.target
                    publish_live(
                        state=PatrolTaskState.NAVIGATING,
                        outcome=PatrolLiveOutcome.RUNNING,
                        control_step=completed_actions,
                        simulation_time_s=simulation_time_s,
                        current_position=current_position,
                        local_goal=drift_goal,
                        remaining_distance_m=(
                            None
                            if drift_goal is None
                            else _remaining_path_distance(
                                active_path,
                                waypoint_state.target_index,
                                current_position,
                            )
                        ),
                    )
                    dwell_aborted = True
                    print(
                        f"PPO_DWELL_RESET id={checkpoint_id} reason=left_arrival_radius",
                        flush=True,
                    )
                    break
                if task_update.state is PatrolTaskState.DWELLING:
                    publish_live(
                        state=PatrolTaskState.DWELLING,
                        outcome=PatrolLiveOutcome.RUNNING,
                        control_step=completed_actions,
                        simulation_time_s=simulation_time_s,
                        current_position=(
                            float(dwell_position[0]),
                            float(dwell_position[1]),
                        ),
                    )

            if dwell_aborted:
                continue
            if not task_update.sampling_requested:
                raise RuntimeError("Checkpoint dwell ended without requesting sampling.")
            publish_live(
                state=PatrolTaskState.SAMPLING,
                outcome=PatrolLiveOutcome.RUNNING,
                control_step=completed_actions,
                simulation_time_s=simulation_time_s,
                current_position=position(),
            )
            try:
                inspection_result = inspection.sample_visit(
                    checkpoint_id,
                    checkpoint_index + 1,
                    completed_actions,
                    arrival_time_s,
                    arrival_position,
                )
                update_live_inspection(inspection_result)
            except Exception as error:
                task_machine.record_sampling_result(
                    success=False,
                    failure_reason=f"inspection sampling failed: {error}",
                    simulation_time_s=simulation_time_s,
                    control_step=completed_actions,
                )
                finish_run(
                    outcome="error",
                    control_step_count=completed_actions,
                    exit_code=1,
                    failure_reason=f"inspection sampling failed: {error}",
                )
                return
            task_update = task_machine.record_sampling_result(
                success=True,
                simulation_time_s=simulation_time_s,
                control_step=completed_actions,
            )
            checkpoint_records.append(
                PatrolCheckpointRecord(
                    checkpoint_id=checkpoint_id,
                    visit_order=checkpoint_index + 1,
                    target_x_m=target[0],
                    target_y_m=target[1],
                    reached=True,
                    reached_control_step=completed_actions,
                    arrival_simulation_time_s=arrival_time_s,
                    segment_control_steps=completed_actions - segment_start_step,
                    actual_segment_path_m=segment_path_length,
                    planned_segment_path_m=active_path.length,
                    segment_path_ratio=(
                        segment_path_length / active_path.length
                        if active_path.length > 0.0
                        else None
                    ),
                    segment_minimum_clearance_m=segment_minimum_clearance,
                )
            )
            latest_by_sensor = {
                sample.sensor_id: sample for sample in inspection_result.samples
            }
            print(
                "INSPECTION_COMPLETED "
                f"id={checkpoint_id} "
                + " ".join(
                    f"{sensor_id}={sample.observed_value:g}{sample.unit}"
                    for sensor_id, sample in latest_by_sensor.items()
                ),
                flush=True,
            )
            for event in inspection_result.events:
                print(
                    f"INSPECTION_ALARM type={event.event_type} rule={event.rule_id} "
                    f"severity={event.severity} value={event.observed_value:g}{event.unit}",
                    flush=True,
                )
            print(
                f"PPO_POINT_VISITED id={checkpoint_id} step={completed_actions} "
                f"segment_path={segment_path_length:.3f}",
                flush=True,
            )
            checkpoint_index += 1
            if checkpoint_index == len(checkpoints):
                if task_update.state is not PatrolTaskState.COMPLETED:
                    raise RuntimeError("Final patrol sampling did not complete the task.")
                finish_run(outcome="completed", control_step_count=completed_actions, exit_code=0)
                return
            current_position = position()
            active_path = plan_world_path(
                grid,
                current_position,
                checkpoints[checkpoint_index][1],
            )
            planned_segment_lengths.append(active_path.length)
            waypoint_state = waypoint_manager.reset(active_path)
            segment_path_length = 0.0
            segment_minimum_clearance = math.inf
            segment_start_step = completed_actions
            current_scan = _range_image(lidar)
            observation_builder.reset(current_scan)
            previous_action = np.array([-1.0, 0.0], dtype=np.float32)
            previous_position = np.asarray(current_position, dtype=np.float64)
            next_goal = waypoint_state.target
            publish_live(
                state=PatrolTaskState.NAVIGATING,
                outcome=PatrolLiveOutcome.RUNNING,
                control_step=completed_actions,
                simulation_time_s=simulation_time_s,
                current_position=current_position,
                local_goal=next_goal,
                remaining_distance_m=(
                    None
                    if next_goal is None
                    else _remaining_path_distance(
                        active_path,
                        waypoint_state.target_index,
                        current_position,
                    )
                ),
            )
            print(
                f"PPO_ROUTE_PLANNED checkpoint={checkpoints[checkpoint_index][0]} "
                f"cells={len(active_path.cells)} waypoints={len(active_path.waypoints)} "
                f"length={active_path.length:.3f}",
                flush=True,
            )

        if waypoint_state.target is None:
            raise RuntimeError("The active A* path has no local target.")
        target = waypoint_state.target
        local_target_field.setSFVec3f([float(target[0]), float(target[1]), 0.055])

        scan = _range_image(lidar)
        dx = target[0] - current_position[0]
        dy = target[1] - current_position[1]
        yaw = float(imu.getRollPitchYaw()[2])
        bearing = wrap_angle(math.atan2(dy, dx) - yaw)
        observation = observation_builder.build(
            raw_lidar=scan,
            goal_distance=math.hypot(dx, dy),
            goal_bearing=bearing,
            linear_speed=current_linear,
            max_linear_speed=mapper.max_linear_speed,
            angular_speed=current_angular,
            max_angular_speed=mapper.max_angular_speed,
            previous_action=previous_action,
        )
        inference_started = time.perf_counter()
        action, _ = model.predict(observation, deterministic=True)
        inference_latency_ms = 1000.0 * (time.perf_counter() - inference_started)
        inference_latencies_ms.append(inference_latency_ms)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        requested_linear, requested_angular = mapper.normalized_to_velocity(action)
        limited_linear, limited_angular = limiter.limit(
            requested_linear,
            requested_angular,
            current_linear,
            current_angular,
        )
        left_speed, right_speed = mapper.velocity_to_wheels(limited_linear, limited_angular)
        current_linear = 0.5 * mapper.wheel_radius * (left_speed + right_speed)
        current_angular = (
            mapper.wheel_radius * (right_speed - left_speed) / mapper.axle_length
        )
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        contact_collision = False
        for _ in range(action_repeat):
            _move_obstacles(moving_obstacles, basic_time_step)
            if supervisor.step(basic_time_step) == -1:
                raise RuntimeError("Webots ended before the patrol run was finalized.")
            physics_step_count += 1
            contact_collision = contact_collision or _has_contact_collision(robot_node)
            if contact_collision:
                _stop_robot(supervisor)
                break

        new_position = np.asarray(position(), dtype=np.float64)
        traveled = float(np.linalg.norm(new_position - previous_position))
        total_path_length += traveled
        segment_path_length += traveled
        previous_position = new_position
        post_scan = _range_image(lidar)
        lidar_minimum = _minimum(post_scan)
        lidar_collision = lidar_minimum <= float(lidar.getMinRange()) + 0.01
        collision = bool(contact_collision or lidar_collision)
        clearance = 0.0 if collision else max(0.0, lidar_minimum - 0.10)
        minimum_lidar_range = min(minimum_lidar_range, lidar_minimum)
        minimum_clearance = min(minimum_clearance, clearance)
        segment_minimum_clearance = min(segment_minimum_clearance, clearance)
        remaining_distance = _remaining_path_distance(
            active_path,
            waypoint_state.target_index,
            (float(new_position[0]), float(new_position[1])),
        )
        previous_action = action.copy()
        yaw = float(imu.getRollPitchYaw()[2])
        trajectory.append(
            PatrolTrajectorySample(
                control_step=control_step,
                simulation_time_s=physics_step_count * basic_time_step / 1000.0,
                position_x_m=float(new_position[0]),
                position_y_m=float(new_position[1]),
                yaw_rad=yaw,
                checkpoint_id=checkpoints[checkpoint_index][0],
                waypoint_index=waypoint_state.target_index,
                local_target_x_m=float(target[0]),
                local_target_y_m=float(target[1]),
                remaining_path_m=remaining_distance,
                requested_linear_mps=requested_linear,
                requested_angular_radps=requested_angular,
                executed_linear_mps=current_linear,
                executed_angular_radps=current_angular,
                minimum_lidar_range_m=lidar_minimum,
                clearance_m=clearance,
                collision=collision,
                inference_latency_ms=inference_latency_ms,
            )
        )
        if collision:
            task_machine.update(
                position=(float(new_position[0]), float(new_position[1])),
                collision=True,
                simulation_time_s=physics_step_count * basic_time_step / 1000.0,
                control_step=control_step,
            )
            detection_source = "contact" if contact_collision else "lidar_threshold"
            collision_records.append(
                PatrolCollisionRecord(
                    collision_index=len(collision_records) + 1,
                    control_step=control_step,
                    simulation_time_s=physics_step_count * basic_time_step / 1000.0,
                    position_x_m=float(new_position[0]),
                    position_y_m=float(new_position[1]),
                    detection_source=detection_source,
                    minimum_lidar_range_m=lidar_minimum,
                )
            )
            finish_run(
                outcome="collision",
                control_step_count=control_step,
                exit_code=2,
                failure_reason=f"collision detected by {detection_source}",
            )
            return
        publish_live(
            state=PatrolTaskState.NAVIGATING,
            outcome=PatrolLiveOutcome.RUNNING,
            control_step=control_step,
            simulation_time_s=physics_step_count * basic_time_step / 1000.0,
            current_position=(float(new_position[0]), float(new_position[1])),
            local_goal=(float(target[0]), float(target[1])),
            remaining_distance_m=remaining_distance,
        )

    finish_run(
        outcome="timeout",
        control_step_count=max_control_steps,
        exit_code=3,
        failure_reason="maximum control-step budget reached",
    )


def _initialize_obstacles(
    supervisor: Supervisor,
    random: np.random.Generator,
) -> list[MovingObstacleControl]:
    configurations = (
        ("DYNAMIC_OBSTACLE_0", 1, -1.20, 1.20, 0.31),
        ("DYNAMIC_OBSTACLE_1", 0, -1.55, 1.55, 0.27),
    )
    result = []
    for def_name, axis, lower, upper, speed in configurations:
        node = supervisor.getFromDef(def_name)
        if node is None:
            raise RuntimeError(f"The Webots world is missing DEF {def_name}.")
        translation = node.getField("translation")
        values = translation.getSFVec3f()
        values[axis] = float(random.uniform(lower, upper))
        translation.setSFVec3f(values)
        node.resetPhysics()
        result.append(
            MovingObstacleControl(
                translation=translation,
                axis=axis,
                lower=lower,
                upper=upper,
                speed=speed,
                direction=float(random.choice([-1.0, 1.0])),
            )
        )
    return result


def _validate_route_checkpoint_nodes(
    supervisor: Supervisor,
    checkpoints: list[tuple[str, tuple[float, float]]],
) -> None:
    world_positions: dict[str, tuple[float, float]] = {}
    task_checkpoints = [
        PatrolTaskCheckpoint(checkpoint_id, position)
        for checkpoint_id, position in checkpoints
    ]
    for checkpoint in task_checkpoints:
        def_name = f"ROUTE_CHECKPOINT_{checkpoint.checkpoint_id}"
        node = supervisor.getFromDef(def_name)
        if node is None:
            raise RuntimeError(f"The Webots world is missing DEF {def_name}.")
        values = node.getField("translation").getSFVec3f()
        world_positions[checkpoint.checkpoint_id] = (float(values[0]), float(values[1]))
    validate_checkpoint_layout(task_checkpoints, world_positions)


def _move_obstacles(obstacles: list[MovingObstacleControl], basic_time_step: int) -> None:
    seconds = basic_time_step / 1000.0
    for obstacle in obstacles:
        values = obstacle.translation.getSFVec3f()
        values[obstacle.axis] += obstacle.direction * obstacle.speed * seconds
        if values[obstacle.axis] >= obstacle.upper:
            values[obstacle.axis] = obstacle.upper
            obstacle.direction = -1.0
        elif values[obstacle.axis] <= obstacle.lower:
            values[obstacle.axis] = obstacle.lower
            obstacle.direction = 1.0
        obstacle.translation.setSFVec3f(values)


def _nominal_route_length(grid: Any, start: tuple[float, float], checkpoints: list[Any]) -> float:
    total = 0.0
    cursor = start
    for _, target in checkpoints:
        path = plan_world_path(grid, cursor, target)
        total += path.length
        cursor = target
    return total


def _remaining_path_distance(
    path: PlannedPath,
    target_index: int,
    position: tuple[float, float],
) -> float:
    if target_index < 0:
        return 0.0
    waypoints = path.waypoints
    remaining = float(np.linalg.norm(np.subtract(waypoints[target_index], position)))
    remaining += sum(
        float(np.linalg.norm(np.subtract(second, first)))
        for first, second in zip(
            waypoints[target_index:],
            waypoints[target_index + 1 :],
            strict=False,
        )
    )
    return remaining


def _range_image(lidar: Any) -> np.ndarray:
    raw = np.asarray(lidar.getRangeImage(), dtype=np.float64)
    minimum = float(lidar.getMinRange())
    maximum = float(lidar.getMaxRange())
    raw = np.nan_to_num(raw, nan=maximum, posinf=maximum, neginf=minimum)
    return _webots_scan_to_policy_order(np.clip(raw, minimum, maximum))


def _minimum(scan: np.ndarray) -> float:
    finite = scan[np.isfinite(scan)]
    if not finite.size:
        raise ValueError("The Webots LiDAR scan contains no finite values.")
    return float(finite.min())


def _has_contact_collision(robot_node: Any) -> bool:
    contacts = robot_node.getContactPoints(includeDescendants=True)
    return any(float(contact.getPoint()[2]) > 0.025 for contact in contacts)


def _stop_robot(supervisor: Supervisor) -> None:
    for device_name in ("left wheel motor", "right wheel motor"):
        motor = supervisor.getDevice(device_name)
        if motor is not None:
            motor.setVelocity(0.0)


def _finish_simulation(supervisor: Supervisor, exit_code: int) -> None:
    if os.environ.get("WEBOTS_AUTO_QUIT") == "1":
        supervisor.simulationQuit(exit_code)
    else:
        supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    main()
