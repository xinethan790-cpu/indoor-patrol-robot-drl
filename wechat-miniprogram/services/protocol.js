'use strict';

const DASHBOARD_SCHEMA_VERSION = 'patrol-dashboard/v1';
const FIXTURE_SCHEMA_VERSION = 'wechat-patrol-fixture/v1';
const MAX_DASHBOARD_PAYLOAD_BYTES = 128 * 1024;
const MAX_TRAJECTORY_POINTS = 300;
const MAX_ALARM_EVENTS = 200;
const MAX_DYNAMIC_OBSTACLES = 32;
const DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M = 1e-6;
const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const RUNTIME_ABORT_EVIDENCE_SCOPE = 'runtime_abort_minimal_evidence';
const RUNTIME_ABORT_REASON_CODE = 'unhandled_controller_exception';
const RUNTIME_ABORT_EXCEPTION_TYPE_PATTERN = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;
const NON_TERMINAL_RUN_STATES = new Set(['navigating', 'dwelling', 'sampling']);
const TERMINAL_RESULT_OUTCOME_BY_RUN_OUTCOME = Object.freeze({
  success: 'completed',
  collision: 'collision',
  safety_stop: 'safety_stop',
  timeout: 'timeout',
  inspection_failed: 'error',
  error: 'error',
});

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

class PayloadProtocolError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'PayloadProtocolError';
    this.code = code;
  }
}

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function requireObject(value, label) {
  if (!isObject(value)) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 必须是对象`);
  }
  return value;
}

function requireFiniteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 必须是有限数字`);
  }
  return value;
}

function requireNonNegativeNumber(value, label) {
  const number = requireFiniteNumber(value, label);
  if (number < 0) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 必须是非负数字`);
  }
  return number;
}

function requireIntegerInRange(value, label, minimum, maximum) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new PayloadProtocolError(
      'INVALID_PAYLOAD',
      `${label} 必须是 ${minimum} 至 ${maximum} 的安全整数`,
    );
  }
  return value;
}

function requireString(value, label, options) {
  const config = options || {};
  if (typeof value !== 'string' || (config.allowEmpty !== true && value.trim() === '')) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 必须是非空字符串`);
  }
  const maximum = config.maxLength || 256;
  if (value.length > maximum) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 超出长度限制`);
  }
  return value;
}

function requireBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 必须是布尔值`);
  }
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
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label}.width_m 必须大于 0`);
  }
  if (requireFiniteNumber(size.height_m, `${label}.height_m`) <= 0) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label}.height_m 必须大于 0`);
  }
  return size;
}

function validateDynamicObstacles(value, label, map) {
  if (!Array.isArray(value) || value.length > MAX_DYNAMIC_OBSTACLES) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 必须是最多 32 项的数组`);
  }
  const maximumRadiusM = Math.min(map.size.width_m, map.size.height_m) / 2;
  const minX = map.origin.x_m;
  const minY = map.origin.y_m;
  const maxX = minX + map.size.width_m;
  const maxY = minY + map.size.height_m;
  const ids = new Set();
  value.forEach((obstacle, index) => {
    const itemLabel = `${label}[${index}]`;
    const item = requireObject(obstacle, itemLabel);
    const id = requireString(item.id, `${itemLabel}.id`, { maxLength: 64 });
    if (ids.has(id)) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', `动态障碍物 ${id} 重复`);
    }
    ids.add(id);
    validatePoint(item.position, `${itemLabel}.position`);
    const radiusM = requireFiniteNumber(item.radius_m, `${itemLabel}.radius_m`);
    if (radiusM <= 0 || radiusM > maximumRadiusM) {
      throw new PayloadProtocolError(
        'INVALID_PAYLOAD',
        `${itemLabel}.radius_m 必须大于 0 且不超过地图短边的一半`,
      );
    }
    const position = item.position;
    if (position.x_m - radiusM < minX - DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M
      || position.x_m + radiusM > maxX + DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M
      || position.y_m - radiusM < minY - DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M
      || position.y_m + radiusM > maxY + DYNAMIC_OBSTACLE_BOUNDARY_TOLERANCE_M) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', `${itemLabel} 的圆形包络超出地图边界`);
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
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 超出归一化范围`);
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

function validateAvoidance(value, run, label = 'avoidance') {
  const avoidance = requireObject(value, label);
  const available = requireBoolean(avoidance.available, `${label}.available`);
  const actions = requireObject(avoidance.actions, `${label}.actions`);
  const policy = validateNormalizedAction(actions.policy, `${label}.actions.policy`);
  const filtered = validateNormalizedAction(actions.filtered, `${label}.actions.filtered`);
  validateExecutedAction(actions.executed, `${label}.actions.executed`);
  if ((policy === null) !== (filtered === null)) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 的策略和过滤动作必须同时存在`);
  }
  const shield = requireObject(avoidance.safety_shield, `${label}.safety_shield`);
  const shieldActive = requireBoolean(shield.active, `${label}.safety_shield.active`);
  const intervention = requireNonNegativeNumber(
    shield.intervention,
    `${label}.safety_shield.intervention`,
  );
  if (policy === null) {
    if (shieldActive || intervention !== 0) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 的安全过滤证据缺少动作`);
    }
  } else {
    const expected = Math.hypot(
      filtered.linear_normalized - policy.linear_normalized,
      filtered.angular_normalized - policy.angular_normalized,
    );
    if (Math.abs(expected - intervention) > 1e-6
      || shieldActive !== (intervention > 1e-6)) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 的安全过滤状态与动作不一致`);
    }
  }
  const guard = requireObject(avoidance.escape_guard, `${label}.escape_guard`);
  const guardActive = requireBoolean(guard.active, `${label}.escape_guard.active`);
  const mode = requireString(guard.mode, `${label}.escape_guard.mode`, { maxLength: 32 });
  const guardUnrecoverable = requireBoolean(
    guard.unrecoverable,
    `${label}.escape_guard.unrecoverable`,
  );
  if (!['inactive', 'reverse', 'forward', 'rotate', 'unrecoverable'].includes(mode)
    || guardActive !== (mode !== 'inactive')
    || guardUnrecoverable !== (mode === 'unrecoverable')) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 的逃逸保护状态不一致`);
  }
  const critical = requireBoolean(avoidance.critical_proximity, `${label}.critical_proximity`);
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
      throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 不可用时必须为空且未介入`);
    }
    return avoidance;
  }
  if (run.control_step === null) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 可用时必须提供 control_step`);
  }
  requireIntegerInRange(
    avoidance.measured_control_step,
    `${label}.measured_control_step`,
    0,
    run.control_step,
  );
  const measuredAt = requireNonNegativeNumber(avoidance.measured_at_s, `${label}.measured_at_s`);
  if (measuredAt > run.simulation_time_s + 1e-9) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label} 不能来自未来时刻`);
  }
  if (requireFiniteNumber(
    avoidance.minimum_lidar_range_m,
    `${label}.minimum_lidar_range_m`,
  ) <= 0) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label}.minimum_lidar_range_m 必须大于 0`);
  }
  if (avoidance.robust_ttc_s !== null) {
    requireNonNegativeNumber(avoidance.robust_ttc_s, `${label}.robust_ttc_s`);
  }
  if (avoidance.closing_rate_mps !== null) {
    requireFiniteNumber(avoidance.closing_rate_mps, `${label}.closing_rate_mps`);
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
    requireNonNegativeNumber(value.simulation_time_s, `${label}.simulation_time_s`);
  }
  if (value.event_type !== undefined && !['raised', 'cleared'].includes(value.event_type)) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', `${label}.event_type 无效`);
  }
  return value;
}

function validateSensorReading(reading, label) {
  const value = requireObject(reading, label);
  requireNullableFiniteNumber(value.value, `${label}.value`);
  requireString(value.unit, `${label}.unit`, { allowEmpty: true, maxLength: 32 });
  return value;
}

function validatePayloadSize(payload) {
  let serialized;
  try {
    serialized = JSON.stringify(payload);
  } catch (error) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'payload 必须可以安全序列化');
  }
  if (!serialized || utf8ByteLength(serialized) > MAX_DASHBOARD_PAYLOAD_BYTES) {
    throw new PayloadProtocolError('PAYLOAD_TOO_LARGE', 'payload 超出小程序安全展示大小');
  }
}

function isNullish(value) {
  return value === null || value === undefined;
}

function validateRunResultConsistency(source, run, result, task, runtimeAbort) {
  if (runtimeAbort) return;

  if (!run.terminal) {
    if (!NON_TERMINAL_RUN_STATES.has(run.state)
      || run.outcome !== 'running'
      || source.mode !== 'live') {
      throw new PayloadProtocolError('INVALID_PAYLOAD', '非终态运行状态必须保持 live/running');
    }
    if (result.available !== false
      || result.validated !== false
      || result.incomplete === true
      || !isNullish(result.validation)
      || !isNullish(result.outcome)
      || !isNullish(result.success)
      || !isNullish(result.failure_reason)) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', '非终态运行不能携带终态结果');
    }
    return;
  }

  const expectedResultOutcome = TERMINAL_RESULT_OUTCOME_BY_RUN_OUTCOME[run.outcome];
  const expectedState = run.outcome === 'success' ? 'completed' : 'failed';
  if (run.state !== expectedState || expectedResultOutcome === undefined) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', '终态运行状态与运行结果不一致');
  }
  if (source.mode !== 'terminal') {
    throw new PayloadProtocolError('INVALID_PAYLOAD', '完整终态必须标记为 terminal 来源');
  }
  if (result.available !== true
    || result.validated !== true
    || result.incomplete === true
    || result.outcome !== expectedResultOutcome
    || result.success !== (expectedResultOutcome === 'completed')) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', '完整终态的验证结果与运行结果不一致');
  }
  const validation = requireString(result.validation, 'result.validation', { maxLength: 128 });
  if (source.terminal_validation !== validation) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', '完整终态的证据验证范围不一致');
  }
  if (result.checkpoints_visited !== task.completed_checkpoint_count
    || result.checkpoints_total !== task.checkpoint_count) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', '终态结果的巡检进度与任务状态不一致');
  }
}

function utf8ByteLength(value) {
  let length = 0;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= 0x7f) length += 1;
    else if (code <= 0x7ff) length += 2;
    else if (code >= 0xd800 && code <= 0xdbff
      && index + 1 < value.length
      && value.charCodeAt(index + 1) >= 0xdc00
      && value.charCodeAt(index + 1) <= 0xdfff) {
      length += 4;
      index += 1;
    } else length += 3;
  }
  return length;
}

function validateDashboardPayload(payload) {
  const sourcePayload = requireObject(payload, 'payload');
  validatePayloadSize(sourcePayload);
  const root = normalizeDashboardPayload(sourcePayload);
  if (root.schema_version !== DASHBOARD_SCHEMA_VERSION) {
    throw new PayloadProtocolError(
      'SCHEMA_MISMATCH',
      `不支持的数据协议：${String(root.schema_version)}`,
    );
  }
  requireString(root.generated_at_utc, 'generated_at_utc', { maxLength: 64 });
  if (Number.isNaN(Date.parse(root.generated_at_utc))) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'generated_at_utc 必须是有效时间');
  }
  const source = requireObject(root.source, 'source');
  requireString(source.mode, 'source.mode', { maxLength: 32 });
  const run = requireObject(root.run, 'run');
  requireString(run.run_id, 'run.run_id', { maxLength: 128 });
  if (!RUN_ID_PATTERN.test(run.run_id)) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'run.run_id 格式无效');
  }
  if (!Number.isSafeInteger(run.sequence) || run.sequence < 0) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'run.sequence 必须是非负安全整数');
  }
  requireBoolean(run.terminal, 'run.terminal');
  requireString(run.state, 'run.state', { maxLength: 32 });
  requireString(run.outcome, 'run.outcome', { maxLength: 32 });
  if (run.control_step !== null) {
    requireIntegerInRange(run.control_step, 'run.control_step', 0, Number.MAX_SAFE_INTEGER);
  }
  requireNonNegativeNumber(run.simulation_time_s, 'run.simulation_time_s');

  const map = requireObject(root.map, 'map');
  validatePoint(map.origin, 'map.origin');
  validateSize(map.size, 'map.size');
  validatePoint(map.robot, 'map.robot');
  if (map.local_goal !== null) validatePoint(map.local_goal, 'map.local_goal');
  if (!Array.isArray(map.checkpoints) || map.checkpoints.length < 1 || map.checkpoints.length > 64) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'map.checkpoints 数量必须为 1 至 64');
  }
  const checkpointIds = new Set();
  map.checkpoints.forEach((checkpoint, index) => {
    const item = requireObject(checkpoint, `map.checkpoints[${index}]`);
    const id = requireString(item.id, `map.checkpoints[${index}].id`, { maxLength: 64 });
    if (checkpointIds.has(id)) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', `巡检点 ${id} 重复`);
    }
    checkpointIds.add(id);
    validatePoint(item.position, `map.checkpoints[${index}].position`);
    if (!['pending', 'current', 'active', 'completed'].includes(item.status)) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', `map.checkpoints[${index}].status 无效`);
    }
  });
  if (!Array.isArray(map.static_obstacles) || map.static_obstacles.length > 128) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'map.static_obstacles 必须是最多 128 项的数组');
  }
  map.static_obstacles.forEach((obstacle, index) => {
    const item = requireObject(obstacle, `map.static_obstacles[${index}]`);
    requireString(item.id, `map.static_obstacles[${index}].id`, { maxLength: 64 });
    validatePoint(item.center, `map.static_obstacles[${index}].center`);
    validateSize(item.size, `map.static_obstacles[${index}].size`);
  });
  validateDynamicObstacles(map.dynamic_obstacles, 'map.dynamic_obstacles', map);
  if (!Array.isArray(map.trajectory) || map.trajectory.length > MAX_TRAJECTORY_POINTS) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'map.trajectory 必须是数组');
  }
  let previousTrajectorySequence = -1;
  map.trajectory.forEach((point, index) => {
    const item = validatePoint(point, `map.trajectory[${index}]`);
    if (!Number.isSafeInteger(item.sequence) || item.sequence < previousTrajectorySequence) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', 'map.trajectory sequence 必须单调递增');
    }
    previousTrajectorySequence = item.sequence;
    requireNonNegativeNumber(item.simulation_time_s, `map.trajectory[${index}].simulation_time_s`);
  });
  requireIntegerInRange(
    map.trajectory_point_count,
    'map.trajectory_point_count',
    0,
    MAX_TRAJECTORY_POINTS,
  );
  if (map.trajectory_point_count !== map.trajectory.length) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'map.trajectory_point_count 与轨迹数量不一致');
  }

  const task = requireObject(root.task, 'task');
  const checkpointCount = requireIntegerInRange(task.checkpoint_count, 'task.checkpoint_count', 1, 64);
  const completedCount = requireIntegerInRange(
    task.completed_checkpoint_count,
    'task.completed_checkpoint_count',
    0,
    checkpointCount,
  );
  if (checkpointCount !== map.checkpoints.length) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'task.checkpoint_count 与地图巡检点数量不一致');
  }
  requireNullableString(task.current_checkpoint_id, 'task.current_checkpoint_id');
  if (task.current_checkpoint_id !== null && !checkpointIds.has(task.current_checkpoint_id)) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'task.current_checkpoint_id 不在地图巡检点中');
  }
  const progressRatio = requireFiniteNumber(task.progress_ratio, 'task.progress_ratio');
  const progressPercent = requireFiniteNumber(task.progress_percent, 'task.progress_percent');
  if (progressRatio < 0 || progressRatio > 1 || progressPercent < 0 || progressPercent > 100) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', '任务进度必须位于有效范围');
  }
  const expectedProgressRatio = completedCount / checkpointCount;
  const expectedProgressPercent = Math.round(expectedProgressRatio * 1000) / 10;
  if (Math.abs(progressRatio - expectedProgressRatio) > 1e-9
    || Math.abs(progressPercent - expectedProgressPercent) > 1e-9) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', '任务进度与已完成巡检点数量不一致');
  }
  if (run.terminal && completedCount === checkpointCount && task.current_checkpoint_id !== null) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', '终态完成任务不应仍有当前巡检点');
  }

  const sensors = requireObject(root.sensors, 'sensors');
  requireBoolean(sensors.available, 'sensors.available');
  if (sensors.checkpoint_id !== null && sensors.checkpoint_id !== undefined) {
    requireString(sensors.checkpoint_id, 'sensors.checkpoint_id', { maxLength: 64 });
    if (!checkpointIds.has(sensors.checkpoint_id)) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', 'sensors.checkpoint_id 不在地图巡检点中');
    }
  }
  if (sensors.sampled_at_s !== null && sensors.sampled_at_s !== undefined) {
    requireNonNegativeNumber(sensors.sampled_at_s, 'sensors.sampled_at_s');
  }
  validateSensorReading(sensors.temperature, 'sensors.temperature');
  validateSensorReading(sensors.smoke, 'sensors.smoke');
  validateSensorReading(sensors.carbon_monoxide, 'sensors.carbon_monoxide');

  const alarms = requireObject(root.alarms, 'alarms');
  if (!Array.isArray(alarms.active) || alarms.active.length > MAX_ALARM_EVENTS) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'alarms.active 数量无效');
  }
  if (!Array.isArray(alarms.history) || alarms.history.length > MAX_ALARM_EVENTS) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'alarms.history 数量无效');
  }
  alarms.active.forEach((event, index) => validateAlarmEvent(event, `alarms.active[${index}]`));
  alarms.history.forEach((event, index) => validateAlarmEvent(event, `alarms.history[${index}]`));
  requireIntegerInRange(alarms.active_count, 'alarms.active_count', 0, MAX_ALARM_EVENTS);
  requireIntegerInRange(alarms.history_count, 'alarms.history_count', 0, Number.MAX_SAFE_INTEGER);
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
    throw new PayloadProtocolError('INVALID_PAYLOAD', '告警计数与告警数组不一致');
  }

  validateAvoidance(root.avoidance, run);

  const result = requireObject(root.result, 'result');
  requireBoolean(result.available, 'result.available');
  requireBoolean(result.validated, 'result.validated');
  if (result.available) {
    requireBoolean(result.success, 'result.success');
    requireIntegerInRange(result.checkpoints_visited, 'result.checkpoints_visited', 0, checkpointCount);
    requireIntegerInRange(result.checkpoints_total, 'result.checkpoints_total', 1, 64);
    requireIntegerInRange(result.collision_count, 'result.collision_count', 0, Number.MAX_SAFE_INTEGER);
    requireIntegerInRange(result.sample_count, 'result.sample_count', 0, Number.MAX_SAFE_INTEGER);
  }
  const runtimeAbort = result.incomplete === true
    || result.outcome === 'aborted'
    || source.mode === 'aborted';
  if (runtimeAbort) {
    if (!run.terminal || run.state !== 'failed' || run.outcome !== 'error') {
      throw new PayloadProtocolError('INVALID_PAYLOAD', '异常中止证据必须绑定 failed/error 终态');
    }
    if (source.mode !== 'aborted'
      || source.terminal_validation !== RUNTIME_ABORT_EVIDENCE_SCOPE) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', '异常中止来源标记不一致');
    }
    if (result.available !== true
      || result.validated !== false
      || result.validation !== RUNTIME_ABORT_EVIDENCE_SCOPE
      || result.incomplete !== true
      || result.outcome !== 'aborted'
      || result.success !== false
      || result.failure_code !== RUNTIME_ABORT_REASON_CODE) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', '异常中止结果标记不一致');
    }
    const failureType = requireString(result.failure_type, 'result.failure_type', { maxLength: 64 });
    if (!RUNTIME_ABORT_EXCEPTION_TYPE_PATTERN.test(failureType)) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', 'result.failure_type 格式无效');
    }
    const failureReason = requireString(result.failure_reason, 'result.failure_reason', { maxLength: 128 });
    if (failureReason !== `unhandled controller exception (${failureType})`) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', '异常中止原因不是脱敏后的固定表达');
    }
    if (result.checkpoints_visited !== completedCount
      || result.checkpoints_total !== checkpointCount) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', '异常中止进度与任务状态不一致');
    }
    const abortElapsed = requireNonNegativeNumber(
      result.elapsed_simulation_time_s,
      'result.elapsed_simulation_time_s',
    );
    if (Math.abs(abortElapsed - run.simulation_time_s) > 1e-9) {
      throw new PayloadProtocolError('INVALID_PAYLOAD', '异常中止时间与运行状态不一致');
    }
  }
  validateRunResultConsistency(source, run, result, task, runtimeAbort);
  return root;
}

function validateFixture(fixture) {
  const root = requireObject(fixture, 'fixture');
  if (root.schema_version !== FIXTURE_SCHEMA_VERSION) {
    throw new PayloadProtocolError(
      'FIXTURE_SCHEMA_MISMATCH',
      `不支持的离线数据协议：${String(root.schema_version)}`,
    );
  }
  if (root.dashboard_schema_version !== DASHBOARD_SCHEMA_VERSION) {
    throw new PayloadProtocolError('SCHEMA_MISMATCH', '离线数据与看板协议不一致');
  }
  const run = requireObject(root.run, 'fixture.run');
  if (typeof run.run_id !== 'string' || run.run_id.trim() === '') {
    throw new PayloadProtocolError('INVALID_FIXTURE', 'fixture.run.run_id 无效');
  }
  const source = requireObject(root.source, 'fixture.source');
  if (source.run_id !== run.run_id) {
    throw new PayloadProtocolError('RUN_ID_MISMATCH', 'fixture.source 与 fixture.run 的 run_id 不一致');
  }
  if (!Array.isArray(root.frames) || root.frames.length < 2 || root.frames.length > 120) {
    throw new PayloadProtocolError('INVALID_FIXTURE', 'fixture.frames 数量必须为 2 至 120');
  }
  let previousSequence = -1;
  for (const frame of root.frames) {
    requireObject(frame, 'fixture frame');
    if (!Number.isSafeInteger(frame.sequence) || frame.sequence <= previousSequence) {
      throw new PayloadProtocolError('INVALID_FIXTURE', 'fixture frame sequence 必须严格递增');
    }
    previousSequence = frame.sequence;
    requireFiniteNumber(frame.simulation_time_s, 'frame.simulation_time_s');
    requireIntegerInRange(frame.control_step, 'frame.control_step', 0, Number.MAX_SAFE_INTEGER);
    validateDynamicObstacles(frame.dynamic_obstacles, 'frame.dynamic_obstacles', root.map);
    validateAvoidance(frame.avoidance, frame, 'frame.avoidance');
  }
  if (root.frames[root.frames.length - 1].terminal !== true) {
    throw new PayloadProtocolError('INVALID_FIXTURE', 'fixture 最后一帧必须是终态');
  }
  return root;
}

function createPayloadGate(options) {
  const config = options || {};
  let pinnedRunId = config.expectedRunId || null;
  let lastSequence = -1;
  let lastFingerprint = null;
  let terminal = false;

  return {
    accept(payload) {
      const value = validateDashboardPayload(payload);
      const run = value.run;
      if (terminal) {
        return { accepted: false, reason: 'terminal', payload: value };
      }
      if (pinnedRunId === null) {
        pinnedRunId = run.run_id;
      }
      if (run.run_id !== pinnedRunId) {
        throw new PayloadProtocolError(
          'RUN_ID_MISMATCH',
          `数据混入被拒绝：期望 ${pinnedRunId}，收到 ${run.run_id}`,
        );
      }
      if (run.sequence < lastSequence) {
        throw new PayloadProtocolError(
          'STALE_SEQUENCE',
          `过期数据被拒绝：${run.sequence} < ${lastSequence}`,
        );
      }
      if (run.sequence === lastSequence) {
        const fingerprint = JSON.stringify({
          run: value.run,
          map: value.map,
          task: value.task,
          sensors: value.sensors,
          alarms: value.alarms,
          avoidance: value.avoidance,
          result: value.result,
        });
        if (fingerprint !== lastFingerprint) {
          throw new PayloadProtocolError(
            'SEQUENCE_CONFLICT',
            `同一序号 ${run.sequence} 出现不同内容，数据被拒绝`,
          );
        }
        return { accepted: false, reason: 'duplicate', payload: value };
      }
      lastSequence = run.sequence;
      lastFingerprint = JSON.stringify({
        run: value.run,
        map: value.map,
        task: value.task,
        sensors: value.sensors,
        alarms: value.alarms,
        avoidance: value.avoidance,
        result: value.result,
      });
      terminal = run.terminal;
      return { accepted: true, reason: terminal ? 'terminal' : 'fresh', payload: value };
    },

    reset(nextExpectedRunId) {
      pinnedRunId = nextExpectedRunId === undefined
        ? (config.expectedRunId || null)
        : (nextExpectedRunId || null);
      lastSequence = -1;
      lastFingerprint = null;
      terminal = false;
    },

    getState() {
      return { runId: pinnedRunId, lastSequence, terminal };
    },
  };
}

module.exports = {
  DASHBOARD_SCHEMA_VERSION,
  FIXTURE_SCHEMA_VERSION,
  PayloadProtocolError,
  MAX_DASHBOARD_PAYLOAD_BYTES,
  MAX_TRAJECTORY_POINTS,
  RUN_ID_PATTERN,
  createPayloadGate,
  normalizeDashboardPayload,
  unavailableAvoidance,
  utf8ByteLength,
  validateDashboardPayload,
  validateFixture,
};
