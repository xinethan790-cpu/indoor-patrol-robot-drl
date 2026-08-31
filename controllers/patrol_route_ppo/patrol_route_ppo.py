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
from typing import Any, Literal

import numpy as np
import yaml
from controller import Supervisor
from stable_baselines3 import PPO

from indoor_patrol_rl.application import (
    DEFAULT_OPERATIONAL_PATROL_SCENARIO_PATH,
    PATROL_TTC_ESCAPE_GUARD_ID,
    UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE,
    CriticalProximityConfig,
    CriticalProximityMonitor,
    PatrolActionPipeline,
    PatrolControlMode,
    PatrolDynamicObstacle,
    PatrolLiveAlert,
    PatrolLiveOutcome,
    PatrolLiveReadings,
    PatrolLiveSnapshot,
    PatrolLiveStatusWriter,
    PatrolRuntimeIdentity,
    PatrolTaskCheckpoint,
    PatrolTaskMachine,
    PatrolTaskState,
    PatrolTTCEscapeDecision,
    PatrolTTCEscapeGuard,
    load_patrol_scenario_config,
    load_r2a_system_deployment_profile,
    validate_checkpoint_layout,
)
from indoor_patrol_rl.core import (
    DifferentialDriveMapper,
    LidarSafetyShield,
    ObservationBuilder,
    SafetyShieldConfig,
    VelocityRateLimiter,
)
from indoor_patrol_rl.envs.webots_patrol import _webots_scan_to_policy_order
from indoor_patrol_rl.evaluation import (
    PatrolCheckpointRecord,
    PatrolCollisionRecord,
    PatrolCriticalProximityRecord,
    PatrolModelSelection,
    PatrolRunSummary,
    PatrolTrajectorySample,
    PatrolTTCEscapeGuardEventRecord,
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
    grid_from_rectangles_with_clearances,
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

    scenario_path = os.environ.get(
        "WEBOTS_PATROL_SCENARIO_CONFIG",
        DEFAULT_OPERATIONAL_PATROL_SCENARIO_PATH.as_posix(),
    )
    scenario_config = load_patrol_scenario_config(
        scenario_path,
        repository_root=repository_root,
    )
    scenario_config.verify_source_integrity()

    route_path = repository_root / "configs/maps/patrol_route.yaml"
    with route_path.open(encoding="utf-8") as stream:
        route_config = yaml.safe_load(stream)
    map_config = route_config["map"]
    rectangles = [
        AxisAlignedRectangle(tuple(item["center"]), tuple(item["size"]))
        for item in map_config["static_rectangles"]
    ]
    grid = grid_from_rectangles_with_clearances(
        origin=tuple(map_config["origin"]),
        size=tuple(map_config["size"]),
        resolution=float(map_config["resolution"]),
        rectangles=rectangles,
        obstacle_inflation_radius=float(map_config["obstacle_inflation_radius"]),
        border_inflation_radius=float(map_config["border_inflation_radius"]),
    )
    checkpoints = [
        (str(item["id"]), (float(item["position"][0]), float(item["position"][1])))
        for item in route_config["checkpoints"]
    ]
    configured_start = tuple(float(value) for value in route_config["robot_start"])
    nominal_route_length = _nominal_route_length(grid, configured_start, checkpoints)

    basic_time_step = int(supervisor.getBasicTimeStep())
    physics_dt = basic_time_step / 1000.0
    control_dt = physics_dt * action_repeat
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
    moving_obstacles = _initialize_obstacles(
        supervisor,
        random,
        scenario_config.dynamic_obstacles,
    )

    for _ in range(2):
        if supervisor.step(basic_time_step) == -1:
            raise RuntimeError("Webots ended while initializing the patrol controller.")

    explicit_model = os.environ.get("WEBOTS_PATROL_MODEL")
    deployment_profile = None
    if explicit_model is None:
        deployment_profile = load_r2a_system_deployment_profile(
            repository_root=repository_root
        )
        deployment_profile.verify_integrity()
        shield_lidar_range = deployment_profile.safety_shield_config.lidar_max_range
        live_lidar_range = float(lidar.getMaxRange())
        if not math.isclose(shield_lidar_range, live_lidar_range, abs_tol=1e-9):
            raise ValueError(
                "The audited R2a safety-shield LiDAR range does not match the live device: "
                f"{shield_lidar_range} != {live_lidar_range}."
            )
        model_selection = PatrolModelSelection(
            path=deployment_profile.model_path,
            training_domain="webots",
            selection_reason=(
                "pinned R2a model-plus-shield component profile; current patrol "
                "scenario is outside its recorded evaluation domain"
            ),
            model_sha256=deployment_profile.model_sha256,
        )
        safety_shield = LidarSafetyShield(deployment_profile.safety_shield_config)
        safety_shield_enabled = True
    else:
        model_selection = select_patrol_model(
            repository_root,
            explicit_model,
            selection_manifest=None,
            allow_unvalidated_fallback=environment_flag(
                "WEBOTS_PATROL_ALLOW_UNVALIDATED_MODEL"
            ),
        )
        safety_shield = LidarSafetyShield(
            SafetyShieldConfig(lidar_max_range=float(lidar.getMaxRange()))
        )
        safety_shield_enabled = False
    if deployment_profile is not None:
        deployment_profile.verify_integrity()
    model = PPO.load(model_selection.path, device="cpu")
    mapper = DifferentialDriveMapper(0.22, 1.5, 0.033, 0.16, 6.0)
    limiter = VelocityRateLimiter(0.50, 3.0, control_dt)
    action_pipeline = PatrolActionPipeline(
        safety_shield=safety_shield,
        mapper=mapper,
        rate_limiter=limiter,
        mode=(
            PatrolControlMode.PINNED_COMPONENT
            if safety_shield_enabled
            else PatrolControlMode.EXPERIMENTAL_SHIELD_DISABLED
        ),
    )
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
    checkpoint_arrival_radius_m = 0.10
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
        arrival_radius_m=checkpoint_arrival_radius_m,
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
    print(
        f"PATROL_SCENARIO id={scenario_config.scenario_id} "
        f"mode={scenario_config.mode.value} "
        f"sha256={scenario_config.source_sha256} "
        "obstacle_speeds="
        + ",".join(
            f"{obstacle.def_name}:{obstacle.speed_mps:.3f}"
            for obstacle in scenario_config.dynamic_obstacles
        ),
        flush=True,
    )
    print(
        f"PATROL_TTC_ESCAPE_GUARD id={PATROL_TTC_ESCAPE_GUARD_ID} "
        f"evidence_scope={UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE}",
        flush=True,
    )
    if deployment_profile is not None:
        print(
            "PATROL_COMPONENT_PROFILE "
            f"system_id={deployment_profile.system_id} "
            f"profile_sha256={deployment_profile.profile_sha256} "
            f"shield={deployment_profile.safety_shield_id} "
            f"component_validation_success={deployment_profile.validation_success_rate:.3f} "
            f"component_validation_collision={deployment_profile.validation_collision_rate:.3f}",
            flush=True,
        )
        print(
            "PATROL_SCENARIO_WARNING the current four-checkpoint application scenario "
            "is outside the recorded R2a component evaluation domain; this run is "
            "application-specific measured evidence, not a production safety certification.",
            flush=True,
        )
    elif (
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
    waypoint_manager = WaypointManager(
        waypoint_radius=0.16,
        final_radius=checkpoint_arrival_radius_m,
    )
    waypoint_state = waypoint_manager.reset(active_path)
    planned_segment_lengths = [active_path.length]
    print(
        f"PPO_ROUTE_PLANNED checkpoint={checkpoints[0][0]} cells={len(active_path.cells)} "
        f"waypoints={len(active_path.waypoints)} length={active_path.length:.3f}",
        flush=True,
    )

    initial_scan = _range_image(lidar)
    observation_builder.reset(initial_scan)
    proximity_monitor = CriticalProximityMonitor(
        CriticalProximityConfig.from_lidar_min_range(float(lidar.getMinRange()))
    )
    latest_proximity = proximity_monitor.update(initial_scan, dt_s=physics_dt)
    ttc_escape_guard = PatrolTTCEscapeGuard()
    current_linear = 0.0
    current_angular = 0.0
    previous_action = np.array([-1.0, 0.0], dtype=np.float32)
    previous_position = np.asarray(position(), dtype=np.float64)
    total_path_length = 0.0
    segment_path_length = 0.0
    minimum_lidar_range = latest_proximity.minimum_distance_m
    minimum_clearance = max(0.0, minimum_lidar_range - 0.10)
    segment_minimum_clearance = minimum_clearance
    physics_step_count = 0
    segment_start_step = 0
    trajectory: list[PatrolTrajectorySample] = []
    checkpoint_records: list[PatrolCheckpointRecord] = []
    collision_records: list[PatrolCollisionRecord] = []
    critical_proximity_records: list[PatrolCriticalProximityRecord] = []
    ttc_escape_guard_events: list[PatrolTTCEscapeGuardEventRecord] = []
    inference_latencies_ms: list[float] = []
    live_writer = PatrolLiveStatusWriter(output_directory, run_id=run_id)
    runtime_identity = PatrolRuntimeIdentity(
        model_id=portable_model_path(model_selection.path, repository_root),
        model_sha256=model_selection.model_sha256,
        safety_shield_enabled=safety_shield_enabled,
        safety_shield_id=(
            None
            if deployment_profile is None
            else deployment_profile.safety_shield_id
        ),
        safety_shield_sha256=(
            None
            if deployment_profile is None
            else deployment_profile.safety_shield_sha256
        ),
    )
    live_sequence = -1
    latest_live_readings = PatrolLiveReadings()
    active_live_alerts: dict[str, PatrolLiveAlert] = {}
    safety_shield_intervention_steps = 0
    ttc_escape_guard_intervention_steps = 0
    ttc_escape_guard_activation_count = 0
    ttc_escape_guard_recovery_count = 0
    ttc_escape_guard_unrecoverable_count = 0

    def update_ttc_escape_guard(
        *,
        front_clearance_m: float,
        dt_s: float,
        phase: Literal["navigation", "dwell"],
        control_step: int,
        simulation_time_s: float,
        count_active_update: bool,
    ) -> PatrolTTCEscapeDecision:
        """Advance, count, and record one auditable patrol guard update."""

        nonlocal ttc_escape_guard_intervention_steps
        nonlocal ttc_escape_guard_activation_count
        nonlocal ttc_escape_guard_recovery_count
        nonlocal ttc_escape_guard_unrecoverable_count
        was_active = ttc_escape_guard.active
        was_unrecoverable = ttc_escape_guard.unrecoverable
        decision = ttc_escape_guard.update(
            latest_proximity,
            front_clearance_m=front_clearance_m,
            dt_s=dt_s,
        )
        if not was_active and decision.active:
            ttc_escape_guard_activation_count += 1
        if was_active and not decision.active:
            ttc_escape_guard_recovery_count += 1
        if not was_unrecoverable and decision.unrecoverable:
            ttc_escape_guard_unrecoverable_count += 1
        counts_as_intervention_step = decision.active and count_active_update
        ttc_escape_guard_intervention_steps += int(counts_as_intervention_step)
        ttc_escape_guard_events.append(
            PatrolTTCEscapeGuardEventRecord(
                event_index=len(ttc_escape_guard_events) + 1,
                control_step=control_step,
                simulation_time_s=simulation_time_s,
                phase=phase,
                active=decision.active,
                mode=decision.mode,
                target_linear_mps=(
                    decision.target_linear_mps if decision.active else None
                ),
                target_angular_radps=(
                    decision.target_angular_radps if decision.active else None
                ),
                counts_as_intervention_step=counts_as_intervention_step,
            )
        )
        return decision

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
            "safety_stop": PatrolLiveOutcome.SAFETY_STOP,
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
        scenario_config.verify_source_integrity()
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
            system_deployment_id=(
                None if deployment_profile is None else deployment_profile.system_id
            ),
            system_deployment_profile_sha256=(
                None
                if deployment_profile is None
                else deployment_profile.profile_sha256
            ),
            patrol_scenario_qualified=False,
            deployment_evidence_scope=(
                "experimental_patrol_scenario"
                if deployment_profile is None
                else scenario_config.deployment_evidence_scope
            ),
            safety_shield_id=(
                None
                if deployment_profile is None
                else deployment_profile.safety_shield_id
            ),
            safety_shield_sha256=(
                None
                if deployment_profile is None
                else deployment_profile.safety_shield_sha256
            ),
            safety_shield_enabled=safety_shield_enabled,
            safety_shield_intervention_steps=safety_shield_intervention_steps,
            safety_shield_intervention_rate=(
                safety_shield_intervention_steps / control_step_count
                if control_step_count > 0
                else 0.0
            ),
            critical_proximity_count=len(critical_proximity_records),
            proximity_events_file="proximity_events.csv",
            patrol_scenario_id=scenario_config.scenario_id,
            patrol_scenario_mode=scenario_config.mode.value,
            patrol_scenario_config_sha256=scenario_config.source_sha256,
            dynamic_obstacle_count=len(scenario_config.dynamic_obstacles),
            dynamic_obstacle_speed_max_mps=max(
                obstacle.speed_mps for obstacle in scenario_config.dynamic_obstacles
            ),
            ttc_escape_guard_enabled=True,
            ttc_escape_guard_id=PATROL_TTC_ESCAPE_GUARD_ID,
            ttc_escape_guard_evidence_scope=(
                UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE
            ),
            ttc_escape_guard_intervention_steps=(
                ttc_escape_guard_intervention_steps
            ),
            ttc_escape_guard_activation_count=ttc_escape_guard_activation_count,
            ttc_escape_guard_recovery_count=ttc_escape_guard_recovery_count,
            ttc_escape_guard_unrecoverable_count=(
                ttc_escape_guard_unrecoverable_count
            ),
        )
        try:
            paths = write_patrol_outputs(
                output_directory,
                trajectory=trajectory,
                checkpoints=output_checkpoints,
                collisions=collision_records,
                summary=summary,
                critical_proximities=critical_proximity_records,
                ttc_escape_guard_events=ttc_escape_guard_events,
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
            f"collisions={len(collision_records)} "
            f"safety_stops={len(critical_proximity_records)} "
            f"ttc_guard_steps={ttc_escape_guard_intervention_steps} "
            f"ttc_guard_activations={ttc_escape_guard_activation_count} "
            f"ttc_guard_recoveries={ttc_escape_guard_recovery_count} "
            f"ttc_guard_unrecoverable={ttc_escape_guard_unrecoverable_count} "
            f"path={total_path_length:.3f} "
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
            dwell_escape_started = False
            dwell_guard_active_ticks = 0
            dwell_task_update_due = False
            dwell_scan = _range_image(lidar)
            while (
                task_update.state is PatrolTaskState.DWELLING
                or dwell_escape_started
            ):
                dwell_position = np.asarray(position(), dtype=np.float64)
                simulation_time_s = physics_step_count * basic_time_step / 1000.0
                dwell_shield_diagnostics = safety_shield.filter_action(
                    dwell_scan,
                    np.array([-1.0, 0.0], dtype=np.float32),
                    enabled=False,
                )
                guard_decision = update_ttc_escape_guard(
                    front_clearance_m=dwell_shield_diagnostics.front_clearance,
                    dt_s=physics_dt,
                    phase="dwell",
                    control_step=completed_actions,
                    simulation_time_s=simulation_time_s,
                    count_active_update=(
                        dwell_guard_active_ticks % action_repeat == 0
                    ),
                )
                if guard_decision.active:
                    # Dwell runs at the physics cadence.  Count one intervention
                    # step per batch of at most ``action_repeat`` active ticks so
                    # this aggregate retains the navigation control-step scale.
                    dwell_guard_active_ticks += 1
                    dwell_escape_started = True
                else:
                    dwell_guard_active_ticks = 0

                obstacle_positions = _dynamic_obstacle_positions(moving_obstacles)
                if guard_decision.unrecoverable:
                    _stop_robot(supervisor)
                    if _has_contact_collision(robot_node):
                        collision_records.append(
                            PatrolCollisionRecord(
                                collision_index=len(collision_records) + 1,
                                control_step=completed_actions,
                                simulation_time_s=simulation_time_s,
                                position_x_m=float(dwell_position[0]),
                                position_y_m=float(dwell_position[1]),
                                detection_source="contact",
                                minimum_lidar_range_m=(
                                    latest_proximity.minimum_distance_m
                                ),
                                minimum_lidar_beam_index=(
                                    latest_proximity.minimum_beam_index
                                ),
                                minimum_lidar_angle_rad=(
                                    latest_proximity.minimum_angle_rad
                                ),
                                dynamic_obstacle_1_x_m=obstacle_positions[0][0],
                                dynamic_obstacle_1_y_m=obstacle_positions[0][1],
                                dynamic_obstacle_2_x_m=obstacle_positions[1][0],
                                dynamic_obstacle_2_y_m=obstacle_positions[1][1],
                            )
                        )
                        finish_run(
                            outcome="collision",
                            control_step_count=completed_actions,
                            exit_code=2,
                            failure_reason="physical contact during inspection dwell",
                        )
                        return
                    critical_proximity_records.append(
                        PatrolCriticalProximityRecord(
                            proximity_index=len(critical_proximity_records) + 1,
                            control_step=completed_actions,
                            simulation_time_s=simulation_time_s,
                            position_x_m=float(dwell_position[0]),
                            position_y_m=float(dwell_position[1]),
                            detection_source="ttc_escape_guard_timeout",
                            minimum_lidar_range_m=(
                                latest_proximity.minimum_distance_m
                            ),
                            consecutive_frames=(
                                latest_proximity.consecutive_critical_frames
                            ),
                            closing_rate_mps=latest_proximity.closing_rate_mps,
                            estimated_ttc_s=latest_proximity.ttc_s,
                            minimum_lidar_beam_index=(
                                latest_proximity.minimum_beam_index
                            ),
                            minimum_lidar_angle_rad=(
                                latest_proximity.minimum_angle_rad
                            ),
                            dynamic_obstacle_1_x_m=obstacle_positions[0][0],
                            dynamic_obstacle_1_y_m=obstacle_positions[0][1],
                            dynamic_obstacle_2_x_m=obstacle_positions[1][0],
                            dynamic_obstacle_2_y_m=obstacle_positions[1][1],
                        )
                    )
                    finish_run(
                        outcome="safety_stop",
                        control_step_count=completed_actions,
                        exit_code=4,
                        failure_reason=(
                            "patrol TTC escape guard reached its bounded intervention "
                            "timeout during inspection dwell"
                        ),
                    )
                    return

                inside_arrival_radius = (
                    math.dist(
                        (float(dwell_position[0]), float(dwell_position[1])),
                        target,
                    )
                    <= checkpoint_arrival_radius_m
                )
                if dwell_task_update_due:
                    should_update_task = not dwell_escape_started or (
                        task_update.state is PatrolTaskState.DWELLING
                        and not inside_arrival_radius
                    )
                    if should_update_task:
                        task_update = task_machine.update(
                            position=(
                                float(dwell_position[0]),
                                float(dwell_position[1]),
                            ),
                            collision=False,
                            simulation_time_s=simulation_time_s,
                            control_step=completed_actions,
                        )
                if task_update.state is PatrolTaskState.FAILED:
                    finish_run(
                        outcome="timeout",
                        control_step_count=completed_actions,
                        exit_code=3,
                        failure_reason=task_update.failure_reason,
                    )
                    return
                if (
                    dwell_escape_started
                    and task_update.state is PatrolTaskState.NAVIGATING
                    and not guard_decision.active
                ):
                    _stop_robot(supervisor)
                    current_linear = 0.0
                    current_angular = 0.0
                    previous_action = np.array([-1.0, 0.0], dtype=np.float32)
                    current_position = (
                        float(dwell_position[0]),
                        float(dwell_position[1]),
                    )
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
                        f"PPO_DWELL_RESET id={checkpoint_id} "
                        "reason=ttc_escape_left_arrival_radius",
                        flush=True,
                    )
                    break
                if task_update.state is PatrolTaskState.SAMPLING:
                    break

                if guard_decision.active:
                    target_linear_mps = guard_decision.target_linear_mps
                    target_angular_radps = guard_decision.target_angular_radps
                elif dwell_escape_started:
                    # The hazard has released before the task machine observed an
                    # out-of-radius position.  Complete that short, now-clear reset
                    # displacement so the interrupted dwell cannot be counted.
                    target_linear_mps = ttc_escape_guard.config.forward_linear_mps
                    target_angular_radps = 0.0
                else:
                    target_linear_mps = 0.0
                    target_angular_radps = 0.0
                (
                    left_speed,
                    right_speed,
                    current_linear,
                    current_angular,
                ) = _body_command_to_wheels(
                    mapper,
                    target_linear_mps,
                    target_angular_radps,
                )
                left_motor.setVelocity(left_speed)
                right_motor.setVelocity(right_speed)

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
                latest_proximity = proximity_monitor.update(
                    dwell_scan,
                    dt_s=physics_dt,
                )
                dwell_lidar_minimum = latest_proximity.minimum_distance_m
                obstacle_positions = _dynamic_obstacle_positions(moving_obstacles)
                contact_collision = _has_contact_collision(robot_node)
                clearance = (
                    0.0
                    if contact_collision or latest_proximity.critical
                    else max(0.0, dwell_lidar_minimum - 0.10)
                )
                minimum_lidar_range = min(minimum_lidar_range, dwell_lidar_minimum)
                minimum_clearance = min(minimum_clearance, clearance)
                segment_minimum_clearance = min(segment_minimum_clearance, clearance)
                simulation_time_s = physics_step_count * basic_time_step / 1000.0
                if contact_collision:
                    _stop_robot(supervisor)
                    task_machine.update(
                        position=(float(dwell_position[0]), float(dwell_position[1])),
                        collision=True,
                        simulation_time_s=simulation_time_s,
                        control_step=completed_actions,
                    )
                    collision_records.append(
                        PatrolCollisionRecord(
                            collision_index=len(collision_records) + 1,
                            control_step=completed_actions,
                            simulation_time_s=simulation_time_s,
                            position_x_m=float(dwell_position[0]),
                            position_y_m=float(dwell_position[1]),
                            detection_source="contact",
                            minimum_lidar_range_m=dwell_lidar_minimum,
                            minimum_lidar_beam_index=(
                                latest_proximity.minimum_beam_index
                            ),
                            minimum_lidar_angle_rad=(
                                latest_proximity.minimum_angle_rad
                            ),
                            dynamic_obstacle_1_x_m=obstacle_positions[0][0],
                            dynamic_obstacle_1_y_m=obstacle_positions[0][1],
                            dynamic_obstacle_2_x_m=obstacle_positions[1][0],
                            dynamic_obstacle_2_y_m=obstacle_positions[1][1],
                        )
                    )
                    finish_run(
                        outcome="collision",
                        control_step_count=completed_actions,
                        exit_code=2,
                        failure_reason="physical contact during inspection dwell",
                    )
                    return
                dwell_task_update_due = True
                publish_live(
                    state=(
                        PatrolTaskState.NAVIGATING
                        if task_update.state is PatrolTaskState.NAVIGATING
                        else PatrolTaskState.DWELLING
                    ),
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
        policy_action, _ = model.predict(observation, deterministic=True)
        inference_latency_ms = 1000.0 * (time.perf_counter() - inference_started)
        inference_latencies_ms.append(inference_latency_ms)
        action_result = action_pipeline.execute(
            raw_lidar=scan,
            policy_action=policy_action,
            current_linear_mps=current_linear,
            current_angular_radps=current_angular,
        )
        policy_action = action_result.policy_action
        shielded_action = action_result.shielded_action
        shield_result = action_result.shield_result
        safety_shield_intervention_steps += int(shield_result.active)
        requested_linear = action_result.requested_linear_mps
        requested_angular = action_result.requested_angular_radps
        guard_decision = update_ttc_escape_guard(
            front_clearance_m=shield_result.front_clearance,
            dt_s=control_dt,
            phase="navigation",
            control_step=control_step,
            simulation_time_s=simulation_time_s,
            count_active_update=True,
        )
        if guard_decision.active:
            # Navigation contributes exactly one intervention for each active
            # policy control step, including the terminal unrecoverable step.
            (
                left_speed,
                right_speed,
                current_linear,
                current_angular,
            ) = _body_command_to_wheels(
                mapper,
                guard_decision.target_linear_mps,
                guard_decision.target_angular_radps,
            )
        else:
            left_speed = action_result.left_wheel_radps
            right_speed = action_result.right_wheel_radps
            current_linear = action_result.executable_linear_mps
            current_angular = action_result.executable_angular_radps
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        contact_collision = False
        post_scan = scan
        proximity = None
        for _ in range(action_repeat):
            _move_obstacles(moving_obstacles, basic_time_step)
            if supervisor.step(basic_time_step) == -1:
                raise RuntimeError("Webots ended before the patrol run was finalized.")
            physics_step_count += 1
            contact_collision = contact_collision or _has_contact_collision(robot_node)
            post_scan = _range_image(lidar)
            latest_proximity = proximity_monitor.update(post_scan, dt_s=physics_dt)
            proximity = latest_proximity
            if contact_collision:
                _stop_robot(supervisor)
                break
        if proximity is None:
            raise RuntimeError("The patrol action completed without a proximity assessment.")

        new_position = np.asarray(position(), dtype=np.float64)
        traveled = float(np.linalg.norm(new_position - previous_position))
        total_path_length += traveled
        segment_path_length += traveled
        previous_position = new_position
        lidar_minimum = proximity.minimum_distance_m
        minimum_lidar_beam_index = proximity.minimum_beam_index
        minimum_lidar_angle_rad = proximity.minimum_angle_rad
        obstacle_positions = _dynamic_obstacle_positions(moving_obstacles)
        critical_proximity = bool(proximity.critical and not contact_collision)
        clearance = (
            0.0
            if contact_collision or critical_proximity
            else max(0.0, lidar_minimum - 0.10)
        )
        minimum_lidar_range = min(minimum_lidar_range, lidar_minimum)
        minimum_clearance = min(minimum_clearance, clearance)
        segment_minimum_clearance = min(segment_minimum_clearance, clearance)
        remaining_distance = _remaining_path_distance(
            active_path,
            waypoint_state.target_index,
            (float(new_position[0]), float(new_position[1])),
        )
        previous_action = action_result.previous_action
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
                collision=contact_collision,
                inference_latency_ms=inference_latency_ms,
                policy_action_linear=float(policy_action[0]),
                policy_action_angular=float(policy_action[1]),
                shielded_action_linear=float(shielded_action[0]),
                shielded_action_angular=float(shielded_action[1]),
                safety_shield_active=shield_result.active,
                safety_shield_intervention=shield_result.intervention,
                shield_front_clearance_m=shield_result.front_clearance,
                shield_left_clearance_m=shield_result.left_clearance,
                shield_right_clearance_m=shield_result.right_clearance,
                minimum_lidar_beam_index=minimum_lidar_beam_index,
                minimum_lidar_angle_rad=minimum_lidar_angle_rad,
                dynamic_obstacle_1_x_m=obstacle_positions[0][0],
                dynamic_obstacle_1_y_m=obstacle_positions[0][1],
                dynamic_obstacle_2_x_m=obstacle_positions[1][0],
                dynamic_obstacle_2_y_m=obstacle_positions[1][1],
                critical_proximity=critical_proximity,
                proximity_closing_rate_mps=proximity.closing_rate_mps,
                estimated_ttc_s=proximity.ttc_s,
                ttc_escape_guard_active=guard_decision.active,
                ttc_escape_guard_mode=guard_decision.mode,
                ttc_escape_guard_target_linear_mps=(
                    guard_decision.target_linear_mps
                    if guard_decision.active
                    else None
                ),
                ttc_escape_guard_target_angular_radps=(
                    guard_decision.target_angular_radps
                    if guard_decision.active
                    else None
                ),
            )
        )
        if contact_collision:
            task_machine.update(
                position=(float(new_position[0]), float(new_position[1])),
                collision=True,
                simulation_time_s=physics_step_count * basic_time_step / 1000.0,
                control_step=control_step,
            )
            collision_records.append(
                PatrolCollisionRecord(
                    collision_index=len(collision_records) + 1,
                    control_step=control_step,
                    simulation_time_s=physics_step_count * basic_time_step / 1000.0,
                    position_x_m=float(new_position[0]),
                    position_y_m=float(new_position[1]),
                    detection_source="contact",
                    minimum_lidar_range_m=lidar_minimum,
                    minimum_lidar_beam_index=minimum_lidar_beam_index,
                    minimum_lidar_angle_rad=minimum_lidar_angle_rad,
                    dynamic_obstacle_1_x_m=obstacle_positions[0][0],
                    dynamic_obstacle_1_y_m=obstacle_positions[0][1],
                    dynamic_obstacle_2_x_m=obstacle_positions[1][0],
                    dynamic_obstacle_2_y_m=obstacle_positions[1][1],
                )
            )
            finish_run(
                outcome="collision",
                control_step_count=control_step,
                exit_code=2,
                failure_reason="physical contact detected by Webots contact points",
            )
            return
        if guard_decision.unrecoverable:
            critical_proximity_records.append(
                PatrolCriticalProximityRecord(
                    proximity_index=len(critical_proximity_records) + 1,
                    control_step=control_step,
                    simulation_time_s=physics_step_count * basic_time_step / 1000.0,
                    position_x_m=float(new_position[0]),
                    position_y_m=float(new_position[1]),
                    detection_source="ttc_escape_guard_timeout",
                    minimum_lidar_range_m=lidar_minimum,
                    consecutive_frames=proximity.consecutive_critical_frames,
                    closing_rate_mps=proximity.closing_rate_mps,
                    estimated_ttc_s=proximity.ttc_s,
                    minimum_lidar_beam_index=minimum_lidar_beam_index,
                    minimum_lidar_angle_rad=minimum_lidar_angle_rad,
                    dynamic_obstacle_1_x_m=obstacle_positions[0][0],
                    dynamic_obstacle_1_y_m=obstacle_positions[0][1],
                    dynamic_obstacle_2_x_m=obstacle_positions[1][0],
                    dynamic_obstacle_2_y_m=obstacle_positions[1][1],
                )
            )
            finish_run(
                outcome="safety_stop",
                control_step_count=control_step,
                exit_code=4,
                failure_reason=(
                    "patrol TTC escape guard reached its bounded intervention timeout"
                ),
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
    configurations: tuple[PatrolDynamicObstacle, PatrolDynamicObstacle],
) -> list[MovingObstacleControl]:
    result = []
    for configuration in configurations:
        node = supervisor.getFromDef(configuration.def_name)
        if node is None:
            raise RuntimeError(
                f"The Webots world is missing DEF {configuration.def_name}."
            )
        translation = node.getField("translation")
        values = translation.getSFVec3f()
        values[configuration.axis] = float(
            random.uniform(
                configuration.lower_bound_m,
                configuration.upper_bound_m,
            )
        )
        translation.setSFVec3f(values)
        node.resetPhysics()
        result.append(
            MovingObstacleControl(
                translation=translation,
                axis=configuration.axis,
                lower=configuration.lower_bound_m,
                upper=configuration.upper_bound_m,
                speed=configuration.speed_mps,
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


def _body_command_to_wheels(
    mapper: DifferentialDriveMapper,
    linear_mps: float,
    angular_radps: float,
) -> tuple[float, float, float, float]:
    """Map a body target to saturated wheels and recover its exact body velocity."""

    left_wheel_radps, right_wheel_radps = mapper.velocity_to_wheels(
        linear_mps,
        angular_radps,
    )
    executable_linear_mps = 0.5 * mapper.wheel_radius * (
        left_wheel_radps + right_wheel_radps
    )
    executable_angular_radps = (
        mapper.wheel_radius
        * (right_wheel_radps - left_wheel_radps)
        / mapper.axle_length
    )
    return (
        left_wheel_radps,
        right_wheel_radps,
        executable_linear_mps,
        executable_angular_radps,
    )


def _dynamic_obstacle_positions(
    obstacles: list[MovingObstacleControl],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Capture the two moving obstacles used by the patrol world."""

    if len(obstacles) != 2:
        raise ValueError("The patrol scene diagnostic requires exactly two obstacles.")
    positions = []
    for obstacle in obstacles:
        values = obstacle.translation.getSFVec3f()
        position = (float(values[0]), float(values[1]))
        if not all(math.isfinite(value) for value in position):
            raise ValueError("Dynamic-obstacle positions must be finite.")
        positions.append(position)
    return positions[0], positions[1]


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
