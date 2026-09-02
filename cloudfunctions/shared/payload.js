'use strict';

const { DASHBOARD_SCHEMA_VERSION } = require('./config');
const { badRequest } = require('./errors');

const MAX_TRAJECTORY_POINTS = 300;
const MAX_CHECKPOINTS = 64;
const MAX_STATIC_OBSTACLES = 128;
const MAX_DYNAMIC_OBSTACLES = 32;
const MAX_ALARM_EVENTS = 200;
const DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M = 1e-6;
const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const RUNTIME_ABORT_EVIDENCE_SCOPE = 'runtime_abort_minimal_evidence';
const RUNTIME_ABORT_REASON_CODE = 'unhandled_controller_exception';
const RUNTIME_ABORT_EXCEPTION_TYPE_PATTERN = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;
const ACTIVE_RUN_STATES = new Set(['navigating', 'dwelling', 'sampling']);
const TERMINAL_RESULT_OUTCOME_BY_RUN_OUTCOME = new Map([
  ['success', 'completed'],
  ['collision', 'collision'],
  ['safety_stop', 'safety_stop'],
  ['timeout', 'timeout'],
  ['inspection_failed', 'error'],
  ['error', 'error'],
]);

function hasOwn(value, key) {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function unavailableAvoidance() {
  return {
    available: false,
    measured_control_step: null,
    measured_at_s: null,
    minimum_lidar_range_m: null,
    robust_ttc_s: null,
    closing_rate_mps: null,
    critical_proximity: false,
    actions: { policy: null, filtered: null, executed: null },
    safety_shield: { active: false, intervention: 0 },
    escape_guard: { active: false, mode: 'inactive', unrecoverable: false },
  };
}

function isPlainObject(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function invalidPayload(message, code) {
  throw badRequest(code || 'INVALID_PAYLOAD', message);
}

function requireObject(value, label) {
  if (!isPlainObject(value)) invalidPayload(`${label} must be an object`);
  return value;
}

function requireFiniteNumber(value, label, minimum) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    invalidPayload(`${label} must be a finite number`);
  }
  if (minimum !== undefined && value < minimum) {
    invalidPayload(`${label} is below the supported minimum`);
  }
  return value;
}

function requireIntegerInRange(value, label, minimum, maximum) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    invalidPayload(`${label} must be a safe integer between ${minimum} and ${maximum}`);
  }
  return value;
}

function requireString(value, label, options) {
  const config = options || {};
  if (typeof value !== 'string' || (config.allowEmpty !== true && value.trim() === '')) {
    invalidPayload(`${label} must be a non-empty string`);
  }
  const maximum = config.maxLength || 256;
  if (value.length > maximum) invalidPayload(`${label} exceeds the supported length`);
  return value;
}

function requireBoolean(value, label) {
  if (typeof value !== 'boolean') invalidPayload(`${label} must be a boolean`);
  return value;
}

function requireNullableString(value, label) {
  if (value === null) return value;
  return requireString(value, label);
}

function requireNullableFiniteNumber(value, label) {
  if (value === null) return value;
  return requireFiniteNumber(value, label);
}

function validatePoint(value, label) {
  const point = requireObject(value, label);
  requireFiniteNumber(point.x_m, `${label}.x_m`);
  requireFiniteNumber(point.y_m, `${label}.y_m`);
  return point;
}

function validateSize(value, label) {
  const size = requireObject(value, label);
  if (requireFiniteNumber(size.width_m, `${label}.width_m`) <= 0) {
    invalidPayload(`${label}.width_m must be greater than zero`);
  }
  if (requireFiniteNumber(size.height_m, `${label}.height_m`) <= 0) {
    invalidPayload(`${label}.height_m must be greater than zero`);
  }
  return size;
}

function validateDynamicObstacles(value, label, map) {
  if (!Array.isArray(value) || value.length > MAX_DYNAMIC_OBSTACLES) {
    invalidPayload(`${label} must contain at most ${MAX_DYNAMIC_OBSTACLES} items`);
  }
  const ids = new Set();
  value.forEach((obstacle, index) => {
    const itemLabel = `${label}[${index}]`;
    const item = requireObject(obstacle, itemLabel);
    const id = requireString(item.id, `${itemLabel}.id`, { maxLength: 64 });
    if (ids.has(id)) invalidPayload(`dynamic obstacle id ${id} is duplicated`);
    ids.add(id);
    const position = validatePoint(item.position, `${itemLabel}.position`);
    const radius = requireFiniteNumber(item.radius_m, `${itemLabel}.radius_m`);
    if (radius <= 0) {
      invalidPayload(`${itemLabel}.radius_m must be greater than zero`);
    }
    const maximumRadius = Math.min(map.size.width_m, map.size.height_m) / 2;
    if (radius > maximumRadius) {
      invalidPayload(`${itemLabel}.radius_m exceeds the supported map-relative maximum`);
    }
    const minimumX = map.origin.x_m;
    const maximumX = minimumX + map.size.width_m;
    const minimumY = map.origin.y_m;
    const maximumY = minimumY + map.size.height_m;
    if (position.x_m - radius < minimumX - DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M
        || position.x_m + radius > maximumX + DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M
        || position.y_m - radius < minimumY - DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M
        || position.y_m + radius > maximumY + DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M) {
      invalidPayload(`${itemLabel} circular envelope is outside map bounds`);
    }
    if (item.velocity !== null) {
      const velocity = requireObject(item.velocity, `${itemLabel}.velocity`);
      requireFiniteNumber(velocity.x_mps, `${itemLabel}.velocity.x_mps`);
      requireFiniteNumber(velocity.y_mps, `${itemLabel}.velocity.y_mps`);
    }
  });
}

function validateNormalizedAction(value, label) {
  if (value === null) return null;
  const action = requireObject(value, label);
  const linear = requireFiniteNumber(action.linear_normalized, `${label}.linear_normalized`);
  const angular = requireFiniteNumber(action.angular_normalized, `${label}.angular_normalized`);
  if (linear < -1 || linear > 1 || angular < -1 || angular > 1) {
    invalidPayload(`${label} must stay within normalized bounds`);
  }
  return action;
}

function validateExecutedAction(value, label) {
  if (value === null) return null;
  const action = requireObject(value, label);
  requireFiniteNumber(action.linear_mps, `${label}.linear_mps`);
  requireFiniteNumber(action.angular_radps, `${label}.angular_radps`);
  return action;
}

function validateAvoidance(value, run) {
  const avoidance = requireObject(value, 'avoidance');
  const available = requireBoolean(avoidance.available, 'avoidance.available');
  const actions = requireObject(avoidance.actions, 'avoidance.actions');
  const policy = validateNormalizedAction(actions.policy, 'avoidance.actions.policy');
  const filtered = validateNormalizedAction(actions.filtered, 'avoidance.actions.filtered');
  validateExecutedAction(actions.executed, 'avoidance.actions.executed');
  if ((policy === null) !== (filtered === null)) {
    invalidPayload('avoidance policy and filtered actions must be provided together');
  }
  const shield = requireObject(avoidance.safety_shield, 'avoidance.safety_shield');
  const shieldActive = requireBoolean(shield.active, 'avoidance.safety_shield.active');
  const intervention = requireFiniteNumber(
    shield.intervention,
    'avoidance.safety_shield.intervention',
    0,
  );
  if (policy === null) {
    if (shieldActive || intervention !== 0) {
      invalidPayload('safety-shield evidence requires policy and filtered actions');
    }
  } else {
    const expected = Math.hypot(
      filtered.linear_normalized - policy.linear_normalized,
      filtered.angular_normalized - policy.angular_normalized,
    );
    if (Math.abs(expected - intervention) > 1e-6
        || shieldActive !== (intervention > 1e-6)) {
      invalidPayload('safety-shield state does not match the action-vector difference');
    }
  }
  const guard = requireObject(avoidance.escape_guard, 'avoidance.escape_guard');
  const guardActive = requireBoolean(guard.active, 'avoidance.escape_guard.active');
  const mode = requireString(guard.mode, 'avoidance.escape_guard.mode', { maxLength: 32 });
  const guardUnrecoverable = requireBoolean(
    guard.unrecoverable,
    'avoidance.escape_guard.unrecoverable',
  );
  if (!['inactive', 'reverse', 'forward', 'rotate', 'unrecoverable'].includes(mode)
      || guardActive !== (mode !== 'inactive')
      || guardUnrecoverable !== (mode === 'unrecoverable')) {
    invalidPayload('avoidance escape-guard state is inconsistent');
  }
  const critical = requireBoolean(
    avoidance.critical_proximity,
    'avoidance.critical_proximity',
  );
  if (!available) {
    if (avoidance.measured_control_step !== null
        || avoidance.measured_at_s !== null
        || avoidance.minimum_lidar_range_m !== null
        || avoidance.robust_ttc_s !== null
        || avoidance.closing_rate_mps !== null
        || policy !== null
        || filtered !== null
        || actions.executed !== null
        || critical
        || shieldActive
        || intervention !== 0
        || guardActive
        || mode !== 'inactive'
        || guardUnrecoverable) {
      invalidPayload('unavailable avoidance telemetry must be empty and inactive');
    }
    return avoidance;
  }
  if (run.control_step === null) {
    invalidPayload('available avoidance telemetry requires run.control_step');
  }
  const measuredStep = requireIntegerInRange(
    avoidance.measured_control_step,
    'avoidance.measured_control_step',
    0,
    run.control_step,
  );
  const measuredAt = requireFiniteNumber(
    avoidance.measured_at_s,
    'avoidance.measured_at_s',
    0,
  );
  if (measuredStep > run.control_step || measuredAt > run.simulation_time_s + 1e-9) {
    invalidPayload('avoidance measurement cannot come from the future');
  }
  if (requireFiniteNumber(
    avoidance.minimum_lidar_range_m,
    'avoidance.minimum_lidar_range_m',
  ) <= 0) {
    invalidPayload('avoidance.minimum_lidar_range_m must be greater than zero');
  }
  if (avoidance.robust_ttc_s !== null) {
    requireFiniteNumber(avoidance.robust_ttc_s, 'avoidance.robust_ttc_s', 0);
  }
  if (avoidance.closing_rate_mps !== null) {
    requireFiniteNumber(avoidance.closing_rate_mps, 'avoidance.closing_rate_mps');
  }
  return avoidance;
}

function normalizeDashboardPayload(payload) {
  const root = requireObject(payload, 'payload');
  const run = requireObject(root.run, 'run');
  const map = requireObject(root.map, 'map');
  const missingControlStep = !hasOwn(run, 'control_step');
  const missingDynamicObstacles = !hasOwn(map, 'dynamic_obstacles');
  const missingAvoidance = !hasOwn(root, 'avoidance');
  if (!missingControlStep && !missingDynamicObstacles && !missingAvoidance) return root;
  return {
    ...root,
    run: missingControlStep ? { ...run, control_step: null } : run,
    map: missingDynamicObstacles ? { ...map, dynamic_obstacles: [] } : map,
    avoidance: missingAvoidance ? unavailableAvoidance() : root.avoidance,
  };
}

function validateAlarmEvent(event, label) {
  const value = requireObject(event, label);
  requireString(value.rule_id, `${label}.rule_id`);
  requireString(value.sensor_id, `${label}.sensor_id`);
  requireString(value.severity, `${label}.severity`);
  requireFiniteNumber(value.observed_value, `${label}.observed_value`);
  requireFiniteNumber(value.threshold, `${label}.threshold`);
  requireString(value.unit, `${label}.unit`, { allowEmpty: true, maxLength: 32 });
  requireString(value.message, `${label}.message`, { maxLength: 500 });
  if (value.checkpoint_id !== undefined && value.checkpoint_id !== null) {
    requireString(value.checkpoint_id, `${label}.checkpoint_id`, { maxLength: 64 });
  }
  if (value.simulation_time_s !== undefined) {
    requireFiniteNumber(value.simulation_time_s, `${label}.simulation_time_s`, 0);
  }
  if (value.event_type !== undefined && !['raised', 'cleared'].includes(value.event_type)) {
    invalidPayload(`${label}.event_type is invalid`);
  }
  return value;
}

function validateSensorReading(reading, label) {
  const value = requireObject(reading, label);
  requireNullableFiniteNumber(value.value, `${label}.value`);
  requireString(value.unit, `${label}.unit`, { allowEmpty: true, maxLength: 32 });
  return value;
}

function rejectDangerousKeys(value, depth) {
  if (depth > 32) invalidPayload('payload nesting is too deep');
  if (Array.isArray(value)) {
    for (const item of value) rejectDangerousKeys(item, depth + 1);
    return;
  }
  if (!isPlainObject(value)) return;
  for (const [key, child] of Object.entries(value)) {
    if (key === '__proto__' || key === 'prototype' || key === 'constructor') {
      invalidPayload('payload contains a forbidden property name');
    }
    rejectDangerousKeys(child, depth + 1);
  }
}

function isNullish(value) {
  return value === null || value === undefined;
}

function validateRunResultLifecycle(source, run, result, task, runtimeAbort) {
  if (!run.terminal) {
    if (!ACTIVE_RUN_STATES.has(run.state)
        || run.outcome !== 'running'
        || source.mode !== 'live') {
      invalidPayload('a non-terminal run must be active with source=live and outcome=running');
    }
    if (result.available !== false
        || result.validated !== false
        || result.incomplete === true
        || !isNullish(result.validation)
        || !isNullish(result.outcome)
        || !isNullish(result.success)
        || !isNullish(result.failure_reason)) {
      invalidPayload('a non-terminal run cannot contain terminal result markers');
    }
    return;
  }

  if (runtimeAbort) return;

  const expectedResultOutcome = TERMINAL_RESULT_OUTCOME_BY_RUN_OUTCOME.get(run.outcome);
  const expectedState = run.outcome === 'success' ? 'completed' : 'failed';
  if (expectedResultOutcome === undefined || run.state !== expectedState) {
    invalidPayload('a terminal run state and outcome are inconsistent');
  }
  if (source.mode !== 'terminal') {
    invalidPayload('a complete terminal run must use source.mode=terminal');
  }
  if (!result.available
      || !result.validated
      || result.incomplete === true
      || result.outcome !== expectedResultOutcome
      || result.success !== (expectedResultOutcome === 'completed')) {
    invalidPayload('a terminal result does not match the run outcome');
  }
  const validation = requireString(
    result.validation,
    'result.validation',
    { maxLength: 128 },
  );
  if (source.terminal_validation !== validation) {
    invalidPayload('a terminal result does not match the source validation scope');
  }
  if (result.checkpoints_visited !== task.completed_checkpoint_count
      || result.checkpoints_total !== task.checkpoint_count) {
    invalidPayload('a terminal result does not match task checkpoint progress');
  }
}

function validateDashboardPayload(payload) {
  const root = normalizeDashboardPayload(payload);
  rejectDangerousKeys(root, 0);
  if (root.schema_version !== DASHBOARD_SCHEMA_VERSION) {
    invalidPayload('unsupported dashboard schema version', 'SCHEMA_MISMATCH');
  }
  requireString(root.generated_at_utc, 'generated_at_utc', { maxLength: 64 });
  if (Number.isNaN(Date.parse(root.generated_at_utc))) {
    invalidPayload('generated_at_utc must be a valid timestamp');
  }
  const source = requireObject(root.source, 'source');
  requireString(source.mode, 'source.mode', { maxLength: 32 });

  const run = requireObject(root.run, 'run');
  if (typeof run.run_id !== 'string' || !RUN_ID_PATTERN.test(run.run_id)) {
    invalidPayload('run.run_id has an invalid format', 'INVALID_RUN_ID');
  }
  if (!Number.isSafeInteger(run.sequence) || run.sequence < 0) {
    invalidPayload('run.sequence must be a non-negative safe integer', 'INVALID_SEQUENCE');
  }
  if (typeof run.terminal !== 'boolean') {
    invalidPayload('run.terminal must be a boolean', 'INVALID_TERMINAL');
  }
  requireString(run.state, 'run.state', { maxLength: 32 });
  requireString(run.outcome, 'run.outcome', { maxLength: 32 });
  if (run.control_step !== null) {
    requireIntegerInRange(run.control_step, 'run.control_step', 0, Number.MAX_SAFE_INTEGER);
  }
  requireFiniteNumber(run.simulation_time_s, 'run.simulation_time_s', 0);

  const map = requireObject(root.map, 'map');
  validatePoint(map.origin, 'map.origin');
  validateSize(map.size, 'map.size');
  validatePoint(map.robot, 'map.robot');
  if (map.local_goal !== null) validatePoint(map.local_goal, 'map.local_goal');

  if (!Array.isArray(map.checkpoints)
      || map.checkpoints.length < 1
      || map.checkpoints.length > MAX_CHECKPOINTS) {
    invalidPayload(`map.checkpoints must contain 1 to ${MAX_CHECKPOINTS} items`);
  }
  const checkpointIds = new Set();
  map.checkpoints.forEach((checkpoint, index) => {
    const label = `map.checkpoints[${index}]`;
    const item = requireObject(checkpoint, label);
    const id = requireString(item.id, `${label}.id`, { maxLength: 64 });
    if (checkpointIds.has(id)) invalidPayload(`checkpoint id ${id} is duplicated`);
    checkpointIds.add(id);
    validatePoint(item.position, `${label}.position`);
    if (!['pending', 'current', 'active', 'completed'].includes(item.status)) {
      invalidPayload(`${label}.status is invalid`);
    }
  });

  if (!Array.isArray(map.static_obstacles)
      || map.static_obstacles.length > MAX_STATIC_OBSTACLES) {
    invalidPayload(`map.static_obstacles must contain at most ${MAX_STATIC_OBSTACLES} items`);
  }
  map.static_obstacles.forEach((obstacle, index) => {
    const label = `map.static_obstacles[${index}]`;
    const item = requireObject(obstacle, label);
    requireString(item.id, `${label}.id`, { maxLength: 64 });
    validatePoint(item.center, `${label}.center`);
    validateSize(item.size, `${label}.size`);
  });
  validateDynamicObstacles(map.dynamic_obstacles, 'map.dynamic_obstacles', map);

  if (!Array.isArray(map.trajectory) || map.trajectory.length > MAX_TRAJECTORY_POINTS) {
    invalidPayload(
      `map.trajectory must be an array with at most ${MAX_TRAJECTORY_POINTS} points`,
      'INVALID_TRAJECTORY',
    );
  }
  let previousTrajectorySequence = -1;
  map.trajectory.forEach((point, index) => {
    const label = `map.trajectory[${index}]`;
    const item = validatePoint(point, label);
    if (!Number.isSafeInteger(item.sequence) || item.sequence < previousTrajectorySequence) {
      invalidPayload('map.trajectory sequence must increase monotonically');
    }
    previousTrajectorySequence = item.sequence;
    requireFiniteNumber(item.simulation_time_s, `${label}.simulation_time_s`, 0);
  });
  requireIntegerInRange(
    map.trajectory_point_count,
    'map.trajectory_point_count',
    0,
    MAX_TRAJECTORY_POINTS,
  );
  if (map.trajectory_point_count !== map.trajectory.length) {
    invalidPayload('map.trajectory_point_count does not match map.trajectory');
  }

  const task = requireObject(root.task, 'task');
  const checkpointCount = requireIntegerInRange(
    task.checkpoint_count,
    'task.checkpoint_count',
    1,
    MAX_CHECKPOINTS,
  );
  const completedCount = requireIntegerInRange(
    task.completed_checkpoint_count,
    'task.completed_checkpoint_count',
    0,
    checkpointCount,
  );
  if (checkpointCount !== map.checkpoints.length) {
    invalidPayload('task.checkpoint_count does not match map.checkpoints');
  }
  requireNullableString(task.current_checkpoint_id, 'task.current_checkpoint_id');
  if (task.current_checkpoint_id !== null && !checkpointIds.has(task.current_checkpoint_id)) {
    invalidPayload('task.current_checkpoint_id does not exist in map.checkpoints');
  }
  const progressRatio = requireFiniteNumber(task.progress_ratio, 'task.progress_ratio');
  const progressPercent = requireFiniteNumber(task.progress_percent, 'task.progress_percent');
  if (progressRatio < 0 || progressRatio > 1 || progressPercent < 0 || progressPercent > 100) {
    invalidPayload('task progress is outside the supported range');
  }
  const expectedProgressRatio = completedCount / checkpointCount;
  const expectedProgressPercent = Math.round(expectedProgressRatio * 1000) / 10;
  if (Math.abs(progressRatio - expectedProgressRatio) > 1e-9
      || Math.abs(progressPercent - expectedProgressPercent) > 1e-9) {
    invalidPayload('task progress does not match completed checkpoint counts');
  }
  if (run.terminal && completedCount === checkpointCount && task.current_checkpoint_id !== null) {
    invalidPayload('a completed terminal task cannot retain a current checkpoint');
  }

  const sensors = requireObject(root.sensors, 'sensors');
  requireBoolean(sensors.available, 'sensors.available');
  if (sensors.checkpoint_id !== null && sensors.checkpoint_id !== undefined) {
    requireString(sensors.checkpoint_id, 'sensors.checkpoint_id', { maxLength: 64 });
    if (!checkpointIds.has(sensors.checkpoint_id)) {
      invalidPayload('sensors.checkpoint_id does not exist in map.checkpoints');
    }
  }
  if (sensors.sampled_at_s !== null && sensors.sampled_at_s !== undefined) {
    requireFiniteNumber(sensors.sampled_at_s, 'sensors.sampled_at_s', 0);
  }
  validateSensorReading(sensors.temperature, 'sensors.temperature');
  validateSensorReading(sensors.smoke, 'sensors.smoke');
  validateSensorReading(sensors.carbon_monoxide, 'sensors.carbon_monoxide');

  const alarms = requireObject(root.alarms, 'alarms');
  if (!Array.isArray(alarms.active) || alarms.active.length > MAX_ALARM_EVENTS) {
    invalidPayload(`alarms.active must contain at most ${MAX_ALARM_EVENTS} items`);
  }
  if (!Array.isArray(alarms.history) || alarms.history.length > MAX_ALARM_EVENTS) {
    invalidPayload(
      `alarms.history must contain at most ${MAX_ALARM_EVENTS} items`,
      'INVALID_ALARM_HISTORY',
    );
  }
  alarms.active.forEach((event, index) => validateAlarmEvent(event, `alarms.active[${index}]`));
  alarms.history.forEach((event, index) => validateAlarmEvent(event, `alarms.history[${index}]`));
  requireIntegerInRange(alarms.active_count, 'alarms.active_count', 0, MAX_ALARM_EVENTS);
  requireIntegerInRange(
    alarms.history_count,
    'alarms.history_count',
    0,
    Number.MAX_SAFE_INTEGER,
  );
  requireIntegerInRange(
    alarms.history_returned_count,
    'alarms.history_returned_count',
    0,
    MAX_ALARM_EVENTS,
  );
  requireBoolean(alarms.history_truncated, 'alarms.history_truncated');
  if (alarms.active_count !== alarms.active.length
      || alarms.history_returned_count !== alarms.history.length
      || alarms.history_count < alarms.history_returned_count) {
    invalidPayload('alarm counts do not match the returned alarm arrays');
  }

  validateAvoidance(root.avoidance, run);

  const result = requireObject(root.result, 'result');
  requireBoolean(result.available, 'result.available');
  requireBoolean(result.validated, 'result.validated');
  if (result.available) {
    requireBoolean(result.success, 'result.success');
    requireIntegerInRange(
      result.checkpoints_visited,
      'result.checkpoints_visited',
      0,
      checkpointCount,
    );
    requireIntegerInRange(
      result.checkpoints_total,
      'result.checkpoints_total',
      1,
      MAX_CHECKPOINTS,
    );
    requireIntegerInRange(
      result.collision_count,
      'result.collision_count',
      0,
      Number.MAX_SAFE_INTEGER,
    );
    requireIntegerInRange(
      result.sample_count,
      'result.sample_count',
      0,
      Number.MAX_SAFE_INTEGER,
    );
  }
  const runtimeAbort = result.incomplete === true
    || result.outcome === 'aborted'
    || source.mode === 'aborted';
  validateRunResultLifecycle(source, run, result, task, runtimeAbort);
  if (runtimeAbort) {
    if (!run.terminal || run.state !== 'failed' || run.outcome !== 'error') {
      invalidPayload('runtime abort evidence requires a failed error terminal run');
    }
    if (source.mode !== 'aborted'
        || source.terminal_validation !== RUNTIME_ABORT_EVIDENCE_SCOPE) {
      invalidPayload('runtime abort source markers are inconsistent');
    }
    if (result.available !== true
        || result.validated !== false
        || result.validation !== RUNTIME_ABORT_EVIDENCE_SCOPE
        || result.incomplete !== true
        || result.outcome !== 'aborted'
        || result.success !== false
        || result.failure_code !== RUNTIME_ABORT_REASON_CODE) {
      invalidPayload('runtime abort result markers are inconsistent');
    }
    const failureType = requireString(
      result.failure_type,
      'result.failure_type',
      { maxLength: 64 },
    );
    if (!RUNTIME_ABORT_EXCEPTION_TYPE_PATTERN.test(failureType)) {
      invalidPayload('result.failure_type is invalid');
    }
    const failureReason = requireString(
      result.failure_reason,
      'result.failure_reason',
      { maxLength: 128 },
    );
    if (failureReason !== `unhandled controller exception (${failureType})`) {
      invalidPayload('runtime abort failure reason is not the redacted representation');
    }
    if (result.checkpoints_visited !== completedCount
        || result.checkpoints_total !== checkpointCount) {
      invalidPayload('runtime abort checkpoint counts do not match task progress');
    }
    const abortElapsed = requireFiniteNumber(
      result.elapsed_simulation_time_s,
      'result.elapsed_simulation_time_s',
      0,
    );
    if (Math.abs(abortElapsed - run.simulation_time_s) > 1e-9) {
      invalidPayload('runtime abort elapsed time does not match the terminal run');
    }
  }
  return root;
}

function parseDashboardBody(rawBody) {
  let parsed;
  try {
    parsed = JSON.parse(rawBody);
  } catch (error) {
    throw badRequest('INVALID_JSON', 'request body must be valid JSON');
  }
  return validateDashboardPayload(parsed);
}

module.exports = {
  MAX_ALARM_EVENTS,
  MAX_CHECKPOINTS,
  MAX_DYNAMIC_OBSTACLES,
  DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M,
  MAX_STATIC_OBSTACLES,
  MAX_TRAJECTORY_POINTS,
  RUN_ID_PATTERN,
  RUNTIME_ABORT_EVIDENCE_SCOPE,
  RUNTIME_ABORT_REASON_CODE,
  isPlainObject,
  normalizeDashboardPayload,
  parseDashboardBody,
  unavailableAvoidance,
  validateDashboardPayload,
};
