'use strict';

const { DASHBOARD_SCHEMA_VERSION } = require('./config');
const { badRequest } = require('./errors');

const MAX_TRAJECTORY_POINTS = 300;
const MAX_CHECKPOINTS = 64;
const MAX_STATIC_OBSTACLES = 128;
const MAX_ALARM_EVENTS = 200;
const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

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

function validateDashboardPayload(payload) {
  const root = requireObject(payload, 'payload');
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
  MAX_STATIC_OBSTACLES,
  MAX_TRAJECTORY_POINTS,
  RUN_ID_PATTERN,
  isPlainObject,
  parseDashboardBody,
  validateDashboardPayload,
};
