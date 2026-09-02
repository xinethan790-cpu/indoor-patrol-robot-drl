'use strict';

const fixture = require('../data/graduation-demo-seed7000-v1.js');
const { buildDashboardPayload } = require('./offlineReplay');
const {
  RUN_ID_PATTERN,
  PayloadProtocolError,
  validateDashboardPayload,
  validateInspection,
} = require('./protocol');

const REPORT_SCHEMA_VERSION = 'patrol-report/v1';
const REPORT_RESPONSE_SCHEMA_VERSION = 'patrol-report-response/v1';
const DEFAULT_DEVICE_ID = 'macmini-m4';
const REPORT_UNAVAILABLE_MESSAGE = '完整巡检报告暂时不可用，请稍后重试';
const REPORT_NOT_FOUND_MESSAGE = '没有找到这次巡检的完整报告';
const MAX_TRAJECTORY_POINTS = 300;
const MAX_ALARM_EVENTS = 200;
const REPORT_KEYS = [
  'schema_version',
  'generated_at_utc',
  'source',
  'run',
  'task',
  'result',
  'inspection',
  'alarms',
  'map',
];
const RESULT_KEYS = [
  'available',
  'validated',
  'validation',
  'outcome',
  'success',
  'failure_reason',
  'checkpoints_visited',
  'checkpoints_total',
  'collision_count',
  'elapsed_simulation_time_s',
  'actual_path_length_m',
  'minimum_clearance_m',
  'minimum_lidar_range_m',
  'critical_proximity_count',
  'safety_shield_intervention_steps',
  'safety_shield_intervention_rate',
  'ttc_escape_guard_intervention_steps',
  'ttc_escape_guard_activation_count',
  'ttc_escape_guard_recovery_count',
  'ttc_escape_guard_unrecoverable_count',
  'inference_latency_p50_ms',
  'inference_latency_p95_ms',
  'dynamic_obstacle_count',
  'dynamic_obstacle_speed_max_mps',
  'sample_count',
  'alarms_raised',
  'alarms_cleared',
  'highest_event_severity',
];
const ALARM_KEYS = [
  'event_id',
  'event_index',
  'event_type',
  'rule_id',
  'sensor_id',
  'severity',
  'checkpoint_id',
  'visit_index',
  'sample_index',
  'control_step',
  'simulation_time_s',
  'observed_value',
  'threshold',
  'unit',
  'message',
];
const ALARM_SENSOR_UNITS = Object.freeze({
  temperature: '°C',
  smoke_obscuration: '%/m',
  carbon_monoxide: 'ppm',
});
const SAFE_MODEL_ID_PATTERN = /^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)*$/;

class ReportProtocolError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'ReportProtocolError';
    this.code = code;
  }
}

function fail(message) {
  throw new ReportProtocolError('INVALID_REPORT_RESPONSE', message);
}

function requireObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`${label} 必须是对象`);
  return value;
}

function requireExactObject(value, expectedKeys, label) {
  const object = requireObject(value, label);
  const actual = Object.keys(object).sort();
  const expected = expectedKeys.slice().sort();
  if (actual.length !== expected.length
    || actual.some((key, index) => key !== expected[index])) fail(`${label} 字段不符合协议`);
  return object;
}

function requireString(value, label, maximum = 256, allowEmpty = false) {
  if (typeof value !== 'string'
    || (!allowEmpty && value.trim() === '')
    || value.length > maximum) fail(`${label} 必须是有效字符串`);
  return value;
}

function requireNullableString(value, label, maximum = 256) {
  if (value === null || value === undefined) return null;
  return requireString(value, label, maximum);
}

function requireBoolean(value, label) {
  if (typeof value !== 'boolean') fail(`${label} 必须是布尔值`);
  return value;
}

function requireFinite(value, label, minimum) {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(`${label} 必须是有限数字`);
  if (minimum !== undefined && value < minimum) fail(`${label} 超出允许范围`);
  return value;
}

function requireNullableFinite(value, label, minimum) {
  if (value === null || value === undefined) return null;
  return requireFinite(value, label, minimum);
}

function requireInteger(value, label, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    fail(`${label} 必须是有效整数`);
  }
  return value;
}

function requireTimestamp(value, label) {
  requireString(value, label, 64);
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$/.test(value)
    || Number.isNaN(Date.parse(value))) fail(`${label} 必须是规范 UTC 时间`);
  return value;
}

function projectPoint(input, label) {
  const point = requireExactObject(input, ['x_m', 'y_m'], label);
  return {
    x_m: requireFinite(point.x_m, `${label}.x_m`),
    y_m: requireFinite(point.y_m, `${label}.y_m`),
  };
}

function projectSize(input, label) {
  const size = requireExactObject(input, ['width_m', 'height_m'], label);
  return {
    width_m: requireFinite(size.width_m, `${label}.width_m`, Number.MIN_VALUE),
    height_m: requireFinite(size.height_m, `${label}.height_m`, Number.MIN_VALUE),
  };
}

function projectRuntimeIdentity(input) {
  const identity = requireExactObject(input, [
    'model_id',
    'model_sha256',
    'safety_shield_enabled',
    'safety_shield_id',
    'safety_shield_sha256',
  ], 'report.run.runtime_identity');
  const modelHash = requireString(identity.model_sha256, 'model_sha256', 64);
  const shieldEnabled = requireBoolean(
    identity.safety_shield_enabled,
    'safety_shield_enabled',
  );
  const shieldId = shieldEnabled
    ? requireString(identity.safety_shield_id, 'safety_shield_id', 128)
    : requireNullableString(identity.safety_shield_id, 'safety_shield_id', 128);
  const shieldHash = shieldEnabled
    ? requireString(identity.safety_shield_sha256, 'safety_shield_sha256', 64)
    : requireNullableString(identity.safety_shield_sha256, 'safety_shield_sha256', 64);
  if (!/^[a-f0-9]{64}$/.test(modelHash)
    || (shieldHash !== null && !/^[a-f0-9]{64}$/.test(shieldHash))) {
    fail('运行身份哈希格式无效');
  }
  if (!shieldEnabled && (shieldId !== null || shieldHash !== null)) {
    fail('安全盾关闭时身份字段必须为空');
  }
  const modelId = requireString(identity.model_id, 'model_id', 512);
  const modelSegments = modelId.split('/');
  if (!SAFE_MODEL_ID_PATTERN.test(modelId)
    || modelSegments.some((segment) => segment === '.' || segment === '..')) {
    fail('模型身份不是安全的项目内标识');
  }
  return {
    model_id: modelId,
    model_sha256: modelHash,
    safety_shield_enabled: shieldEnabled,
    safety_shield_id: shieldId,
    safety_shield_sha256: shieldHash,
  };
}

function projectResult(input) {
  const value = requireExactObject(input, RESULT_KEYS, 'report.result');
  const result = {
    available: requireBoolean(value.available, 'result.available'),
    validated: requireBoolean(value.validated, 'result.validated'),
    validation: requireString(value.validation, 'result.validation', 128),
    outcome: requireString(value.outcome, 'result.outcome', 32),
    success: requireBoolean(value.success, 'result.success'),
    failure_reason: requireNullableString(value.failure_reason, 'result.failure_reason', 256),
    checkpoints_visited: requireInteger(value.checkpoints_visited, 'result.checkpoints_visited'),
    checkpoints_total: requireInteger(value.checkpoints_total, 'result.checkpoints_total', 1, 64),
    collision_count: requireInteger(value.collision_count, 'result.collision_count'),
    elapsed_simulation_time_s: requireFinite(
      value.elapsed_simulation_time_s,
      'result.elapsed_simulation_time_s',
      0,
    ),
    actual_path_length_m: requireNullableFinite(
      value.actual_path_length_m,
      'result.actual_path_length_m',
      0,
    ),
    minimum_clearance_m: requireNullableFinite(
      value.minimum_clearance_m,
      'result.minimum_clearance_m',
      0,
    ),
    minimum_lidar_range_m: requireNullableFinite(
      value.minimum_lidar_range_m,
      'result.minimum_lidar_range_m',
      0,
    ),
    critical_proximity_count: value.critical_proximity_count === null
      ? null : requireInteger(value.critical_proximity_count, 'result.critical_proximity_count'),
    safety_shield_intervention_steps: value.safety_shield_intervention_steps === null
      ? null : requireInteger(
        value.safety_shield_intervention_steps,
        'result.safety_shield_intervention_steps',
      ),
    safety_shield_intervention_rate: requireNullableFinite(
      value.safety_shield_intervention_rate,
      'result.safety_shield_intervention_rate',
      0,
    ),
    ttc_escape_guard_intervention_steps: value.ttc_escape_guard_intervention_steps === null
      ? null : requireInteger(
        value.ttc_escape_guard_intervention_steps,
        'result.ttc_escape_guard_intervention_steps',
      ),
    ttc_escape_guard_activation_count: value.ttc_escape_guard_activation_count === null
      ? null : requireInteger(
        value.ttc_escape_guard_activation_count,
        'result.ttc_escape_guard_activation_count',
      ),
    ttc_escape_guard_recovery_count: value.ttc_escape_guard_recovery_count === null
      ? null : requireInteger(
        value.ttc_escape_guard_recovery_count,
        'result.ttc_escape_guard_recovery_count',
      ),
    ttc_escape_guard_unrecoverable_count: value.ttc_escape_guard_unrecoverable_count === null
      ? null : requireInteger(
        value.ttc_escape_guard_unrecoverable_count,
        'result.ttc_escape_guard_unrecoverable_count',
      ),
    inference_latency_p50_ms: requireNullableFinite(
      value.inference_latency_p50_ms,
      'result.inference_latency_p50_ms',
      0,
    ),
    inference_latency_p95_ms: requireNullableFinite(
      value.inference_latency_p95_ms,
      'result.inference_latency_p95_ms',
      0,
    ),
    dynamic_obstacle_count: value.dynamic_obstacle_count === null
      ? null : requireInteger(value.dynamic_obstacle_count, 'result.dynamic_obstacle_count'),
    dynamic_obstacle_speed_max_mps: requireNullableFinite(
      value.dynamic_obstacle_speed_max_mps,
      'result.dynamic_obstacle_speed_max_mps',
      0,
    ),
    sample_count: requireInteger(value.sample_count, 'result.sample_count'),
    alarms_raised: requireInteger(value.alarms_raised, 'result.alarms_raised'),
    alarms_cleared: requireInteger(value.alarms_cleared, 'result.alarms_cleared'),
    highest_event_severity: requireNullableString(
      value.highest_event_severity,
      'result.highest_event_severity',
      32,
    ),
  };
  if (result.safety_shield_intervention_rate !== null
    && result.safety_shield_intervention_rate > 1) fail('安全盾介入率超出范围');
  if (result.checkpoints_visited > result.checkpoints_total) fail('巡检点结果计数不一致');
  if (result.highest_event_severity !== null
    && !['info', 'warning', 'critical'].includes(result.highest_event_severity)) {
    fail('最高告警级别无效');
  }
  if ((result.outcome === 'completed'
    && (result.failure_reason !== null || result.collision_count !== 0))
    || (result.outcome !== 'completed' && result.failure_reason === null)
    || (result.outcome === 'collision' && result.collision_count < 1)
    || (result.outcome !== 'collision' && result.collision_count !== 0)
    || (result.outcome === 'safety_stop'
      && (result.critical_proximity_count === null
        || result.critical_proximity_count < 1))) {
    fail('报告结果的失败原因或碰撞计数与终态不一致');
  }
  if ((result.dynamic_obstacle_count === null)
    !== (result.dynamic_obstacle_speed_max_mps === null)
    || (result.dynamic_obstacle_count === 0 && result.dynamic_obstacle_speed_max_mps !== 0)) {
    fail('动态障碍物指标可用性不一致');
  }
  if (result.inference_latency_p50_ms !== null
    && result.inference_latency_p95_ms !== null
    && result.inference_latency_p95_ms < result.inference_latency_p50_ms) {
    fail('推理延迟分位数不一致');
  }
  return result;
}

function projectAlarmEvent(input, index) {
  const value = requireExactObject(
    input,
    ALARM_KEYS,
    `report.alarms.history[${index}]`,
  );
  const eventType = requireString(value.event_type, 'alarm.event_type', 16);
  const severity = requireString(value.severity, 'alarm.severity', 16);
  const sensorId = requireString(value.sensor_id, 'alarm.sensor_id', 64);
  if (!['raised', 'cleared'].includes(eventType)) fail('告警事件类型无效');
  if (!['info', 'warning', 'critical'].includes(severity)) fail('告警严重程度无效');
  if (!Object.hasOwn(ALARM_SENSOR_UNITS, sensorId)
    || value.unit !== ALARM_SENSOR_UNITS[sensorId]) fail('告警传感器或单位无效');
  return {
    event_index: requireInteger(value.event_index, 'alarm.event_index', 1),
    event_id: requireString(value.event_id, 'alarm.event_id', 128),
    event_type: eventType,
    rule_id: requireString(value.rule_id, 'alarm.rule_id', 128),
    sensor_id: sensorId,
    severity,
    checkpoint_id: requireString(value.checkpoint_id, 'alarm.checkpoint_id', 64),
    visit_index: requireInteger(value.visit_index, 'alarm.visit_index', 1, 64),
    sample_index: requireInteger(value.sample_index, 'alarm.sample_index', 1),
    control_step: requireInteger(value.control_step, 'alarm.control_step'),
    simulation_time_s: requireFinite(value.simulation_time_s, 'alarm.simulation_time_s', 0),
    observed_value: requireFinite(value.observed_value, 'alarm.observed_value'),
    threshold: requireFinite(value.threshold, 'alarm.threshold'),
    unit: requireString(value.unit, 'alarm.unit', 32),
    message: requireString(value.message, 'alarm.message', 500),
  };
}

function projectAlarms(input, checkpointIds, run) {
  const alarms = requireExactObject(input, ['history_count', 'history'], 'report.alarms');
  const historyCount = requireInteger(alarms.history_count, 'alarms.history_count', 0, MAX_ALARM_EVENTS);
  if (!Array.isArray(alarms.history) || alarms.history.length !== historyCount) {
    fail('告警时间线与计数不一致');
  }
  const history = alarms.history.map(projectAlarmEvent);
  let previousTime = -1;
  let previousControlStep = -1;
  const eventIds = new Set();
  history.forEach((event, index) => {
    if (event.event_index !== index + 1
      || event.event_id !== `alarm-${event.event_index}`
      || event.simulation_time_s < previousTime
      || event.control_step < previousControlStep
      || event.simulation_time_s > run.simulation_time_s + 1e-9
      || event.control_step > run.control_step) {
      fail('告警时间线顺序无效');
    }
    if (event.checkpoint_id !== null && !checkpointIds.has(event.checkpoint_id)) {
      fail('告警巡检点不在本次路线中');
    }
    if (event.event_id !== null) {
      if (eventIds.has(event.event_id)) fail('告警事件编号重复');
      eventIds.add(event.event_id);
    }
    previousTime = event.simulation_time_s;
    previousControlStep = event.control_step;
  });
  return { history_count: historyCount, history };
}

function validateAlarmInspectionConsistency(alarms, inspection, result) {
  const eventsByVisit = new Map(
    inspection.visits.map((visit) => [visit.visit_index, []]),
  );
  const severityRank = { info: 1, warning: 2, critical: 3 };
  const sensorBySampleRemainder = [
    'temperature',
    'smoke_obscuration',
    'carbon_monoxide',
  ];
  let raisedCount = 0;
  let clearedCount = 0;
  let highestEventSeverity = null;
  alarms.history.forEach((event) => {
    const visit = inspection.visits[event.visit_index - 1];
    const nextVisit = inspection.visits[event.visit_index] || null;
    const previousVisitSampleEnd = event.visit_index === 1
      ? 0
      : inspection.visits[event.visit_index - 2]
        .readings.carbon_monoxide.sample_index;
    const visitSampleEnd = visit && visit.readings.carbon_monoxide.sample_index;
    const expectedSensor = sensorBySampleRemainder[(event.sample_index - 1) % 3];
    if (!visit
      || visit.checkpoint_id !== event.checkpoint_id
      || visit.control_step !== event.control_step
      || event.simulation_time_s < visit.simulation_time_s - 1e-9
      || (nextVisit && event.simulation_time_s > nextVisit.simulation_time_s + 1e-9)
      || event.sample_index <= previousVisitSampleEnd
      || event.sample_index > visitSampleEnd
      || event.sensor_id !== expectedSensor) {
      fail('告警事件与对应点位采样来源不一致');
    }
    eventsByVisit.get(event.visit_index).push(event);
    if (event.event_type === 'raised') raisedCount += 1;
    else clearedCount += 1;
    if (highestEventSeverity === null
      || severityRank[event.severity] > severityRank[highestEventSeverity]) {
      highestEventSeverity = event.severity;
    }
  });
  if (raisedCount !== result.alarms_raised
    || clearedCount !== result.alarms_cleared
    || highestEventSeverity !== result.highest_event_severity) {
    fail('报告告警指标与完整时间线不一致');
  }
  const active = new Map();
  inspection.visits.forEach((visit) => {
    const visitEvents = eventsByVisit.get(visit.visit_index);
    if (visitEvents.length !== visit.alarm_event_count) {
      fail('点位告警计数与时间线不一致');
    }
    visitEvents.forEach((event) => {
      if (event.event_type === 'raised') {
        if (active.has(event.rule_id)) fail('告警状态机出现重复触发');
        active.set(event.rule_id, event);
      } else {
        const raised = active.get(event.rule_id);
        if (!raised
          || raised.sensor_id !== event.sensor_id
          || raised.severity !== event.severity) fail('告警解除事件缺少对应触发状态');
        active.delete(event.rule_id);
      }
    });
    const severities = [...active.values()].map((event) => event.severity);
    const expectedSeverity = severities.includes('critical')
      ? 'critical'
      : (severities.includes('warning')
        ? 'warning'
        : (severities.includes('info') ? 'info' : null));
    if (expectedSeverity !== visit.highest_active_severity) {
      fail('点位最高活动严重度与时间线不一致');
    }
  });
}

function projectMap(input, run) {
  const value = requireExactObject(input, [
    'origin',
    'size',
    'robot_start',
    'robot',
    'trajectory',
    'trajectory_point_count',
    'trajectory_source_point_count',
    'trajectory_truncated',
    'checkpoints',
    'static_obstacles',
  ], 'report.map');
  const origin = projectPoint(value.origin, 'map.origin');
  const size = projectSize(value.size, 'map.size');
  const insideMap = (point, label) => {
    if (point.x_m < origin.x_m - 1e-6
      || point.x_m > origin.x_m + size.width_m + 1e-6
      || point.y_m < origin.y_m - 1e-6
      || point.y_m > origin.y_m + size.height_m + 1e-6) fail(`${label} 超出报告地图边界`);
    return point;
  };
  const checkpointsInput = value.checkpoints;
  if (!Array.isArray(checkpointsInput) || checkpointsInput.length < 1 || checkpointsInput.length > 64) {
    fail('报告地图巡检点数量无效');
  }
  const checkpointIds = new Set();
  const checkpoints = checkpointsInput.map((inputCheckpoint, index) => {
    const checkpoint = requireExactObject(
      inputCheckpoint,
      ['id', 'ordinal', 'position', 'status'],
      `map.checkpoints[${index}]`,
    );
    const id = requireString(checkpoint.id, 'checkpoint.id', 64);
    if (checkpointIds.has(id)) fail('报告地图巡检点重复');
    checkpointIds.add(id);
    const status = requireString(checkpoint.status, 'checkpoint.status', 16);
    if (!['pending', 'current', 'active', 'completed'].includes(status)) fail('巡检点状态无效');
    const ordinal = requireInteger(checkpoint.ordinal, 'checkpoint.ordinal', 1, 64);
    if (ordinal !== index + 1) fail('巡检点序号不连续');
    return {
      id,
      ordinal,
      position: insideMap(projectPoint(checkpoint.position, 'checkpoint.position'), 'checkpoint.position'),
      status,
    };
  });
  const staticInput = value.static_obstacles;
  if (!Array.isArray(staticInput) || staticInput.length > 128) fail('静态障碍物数量无效');
  const obstacleIds = new Set();
  const staticObstacles = staticInput.map((inputObstacle, index) => {
    const obstacle = requireExactObject(
      inputObstacle,
      ['id', 'label', 'center', 'size'],
      `map.static_obstacles[${index}]`,
    );
    const obstacleId = requireString(obstacle.id, 'obstacle.id', 64);
    if (obstacleIds.has(obstacleId)) fail('静态障碍物编号重复');
    obstacleIds.add(obstacleId);
    const center = projectPoint(obstacle.center, 'obstacle.center');
    const obstacleSize = projectSize(obstacle.size, 'obstacle.size');
    insideMap({
      x_m: center.x_m - obstacleSize.width_m / 2,
      y_m: center.y_m - obstacleSize.height_m / 2,
    }, 'obstacle');
    insideMap({
      x_m: center.x_m + obstacleSize.width_m / 2,
      y_m: center.y_m + obstacleSize.height_m / 2,
    }, 'obstacle');
    return {
      id: obstacleId,
      label: requireString(obstacle.label, 'obstacle.label', 64),
      center,
      size: obstacleSize,
    };
  });
  if (!Array.isArray(value.trajectory) || value.trajectory.length > MAX_TRAJECTORY_POINTS) {
    fail('报告轨迹数量无效');
  }
  let previousSequence = -1;
  let previousTime = -1;
  const trajectory = value.trajectory.map((inputPoint, index) => {
    const point = requireExactObject(
      inputPoint,
      ['sequence', 'simulation_time_s', 'x_m', 'y_m'],
      `map.trajectory[${index}]`,
    );
    const sequence = requireInteger(point.sequence, 'trajectory.sequence');
    const simulationTimeS = requireFinite(
      point.simulation_time_s,
      'trajectory.simulation_time_s',
      0,
    );
    if (sequence <= previousSequence
      || simulationTimeS < previousTime
      || sequence > run.sequence
      || simulationTimeS > run.simulation_time_s + 1e-9) fail('报告轨迹顺序无效');
    previousSequence = sequence;
    previousTime = simulationTimeS;
    return insideMap({
      sequence,
      simulation_time_s: simulationTimeS,
      x_m: requireFinite(point.x_m, `map.trajectory[${index}].x_m`),
      y_m: requireFinite(point.y_m, `map.trajectory[${index}].y_m`),
    }, `map.trajectory[${index}]`);
  });
  const pointCount = requireInteger(
    value.trajectory_point_count,
    'map.trajectory_point_count',
    0,
    MAX_TRAJECTORY_POINTS,
  );
  const sourcePointCount = requireInteger(
    value.trajectory_source_point_count,
    'map.trajectory_source_point_count',
    0,
  );
  const truncated = requireBoolean(value.trajectory_truncated, 'map.trajectory_truncated');
  if (pointCount !== trajectory.length
    || sourcePointCount < pointCount
    || sourcePointCount > run.sequence + 1
    || truncated !== (sourcePointCount > pointCount)) fail('报告轨迹计数不一致');
  if (trajectory.length < 1) fail('完整报告不能缺少终态轨迹');
  const robot = insideMap(projectPoint(value.robot, 'map.robot'), 'map.robot');
  const lastPoint = trajectory[trajectory.length - 1];
  if (lastPoint.sequence !== run.sequence
    || Math.abs(lastPoint.simulation_time_s - run.simulation_time_s) > 1e-9
    || Math.abs(lastPoint.x_m - robot.x_m) > 1e-9
    || Math.abs(lastPoint.y_m - robot.y_m) > 1e-9) fail('报告轨迹末点与终态机器人状态不一致');
  return {
    map: {
      origin,
      size,
      robot_start: insideMap(projectPoint(value.robot_start, 'map.robot_start'), 'map.robot_start'),
      robot,
      trajectory,
      trajectory_point_count: pointCount,
      trajectory_source_point_count: sourcePointCount,
      trajectory_truncated: truncated,
      checkpoints,
      static_obstacles: staticObstacles,
    },
    checkpointIds,
  };
}

function validatePatrolReport(input, expectedRunId) {
  const value = requireExactObject(input, REPORT_KEYS, 'report');
  if (value.schema_version !== REPORT_SCHEMA_VERSION) fail('巡检报告协议版本不受支持');
  const sourceInput = requireExactObject(value.source, [
    'terminal_validation',
    'map_source_file',
    'map_qualification',
    'map_boundary_note',
  ], 'report.source');
  const source = {
    terminal_validation: requireString(
      sourceInput.terminal_validation,
      'source.terminal_validation',
      128,
    ),
    map_source_file: requireString(sourceInput.map_source_file, 'source.map_source_file', 256),
    map_qualification: requireString(
      sourceInput.map_qualification,
      'source.map_qualification',
      64,
    ),
    map_boundary_note: requireString(
      sourceInput.map_boundary_note,
      'source.map_boundary_note',
      500,
    ),
  };
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(source.map_source_file)
    || source.map_qualification !== 'demonstration_only') {
    fail('报告地图来源或使用范围无效');
  }
  const runInput = requireExactObject(value.run, [
    'run_id',
    'seed',
    'state',
    'outcome',
    'sequence',
    'control_step',
    'simulation_time_s',
    'runtime_identity',
  ], 'report.run');
  const runId = requireString(runInput.run_id, 'run.run_id', 128);
  if (!RUN_ID_PATTERN.test(runId)
    || (expectedRunId !== undefined && runId !== expectedRunId)) fail('报告运行编号格式无效');
  const run = {
    run_id: runId,
    seed: requireInteger(runInput.seed, 'run.seed'),
    state: requireString(runInput.state, 'run.state', 32),
    outcome: requireString(runInput.outcome, 'run.outcome', 32),
    sequence: requireInteger(runInput.sequence, 'run.sequence'),
    control_step: requireInteger(runInput.control_step, 'run.control_step'),
    simulation_time_s: requireFinite(runInput.simulation_time_s, 'run.simulation_time_s', 0),
    runtime_identity: projectRuntimeIdentity(runInput.runtime_identity),
  };
  if (!['completed', 'failed'].includes(run.state) || run.outcome === 'running') {
    fail('报告必须绑定终态运行');
  }
  const taskInput = requireExactObject(value.task, [
    'completed_checkpoint_count',
    'checkpoint_count',
    'progress_ratio',
    'progress_percent',
  ], 'report.task');
  const task = {
    completed_checkpoint_count: requireInteger(
      taskInput.completed_checkpoint_count,
      'task.completed_checkpoint_count',
      0,
      64,
    ),
    checkpoint_count: requireInteger(taskInput.checkpoint_count, 'task.checkpoint_count', 1, 64),
    progress_ratio: requireFinite(taskInput.progress_ratio, 'task.progress_ratio', 0),
    progress_percent: requireFinite(taskInput.progress_percent, 'task.progress_percent', 0),
  };
  if (task.completed_checkpoint_count > task.checkpoint_count
    || task.progress_ratio > 1
    || task.progress_percent > 100
    || Math.abs(task.progress_ratio - task.completed_checkpoint_count / task.checkpoint_count) > 1e-9
    || Math.abs(task.progress_percent - Math.round(task.progress_ratio * 1000) / 10) > 1e-9) {
    fail('报告任务进度不一致');
  }
  const { map, checkpointIds } = projectMap(value.map, run);
  if (map.checkpoints.length !== task.checkpoint_count) fail('报告地图与任务点位数量不一致');
  map.checkpoints.forEach((checkpoint, index) => {
    const shouldBeCompleted = index < task.completed_checkpoint_count;
    if ((checkpoint.status === 'completed') !== shouldBeCompleted) {
      fail('报告地图巡检点状态与任务进度不一致');
    }
    if (index >= task.completed_checkpoint_count
      && (checkpoint.status === 'current' || checkpoint.status === 'active')
      && index !== task.completed_checkpoint_count) {
      fail('报告当前巡检点必须是路线中的下一个未完成点');
    }
  });
  if (map.checkpoints.filter((checkpoint) => (
    checkpoint.status === 'current' || checkpoint.status === 'active'
  )).length > 1) fail('报告地图不能同时标记多个当前巡检点');
  let inspection;
  try {
    inspection = validateInspection(value.inspection, {
      label: 'report.inspection',
      run,
      checkpointIds,
    });
  } catch (error) {
    if (error instanceof PayloadProtocolError) fail(error.message);
    throw error;
  }
  if (inspection.available !== true) fail('完整巡检报告缺少点位采样档案');
  const result = projectResult(value.result);
  const alarms = projectAlarms(value.alarms, checkpointIds, run);
  validateAlarmInspectionConsistency(alarms, inspection, result);
  const expectedResultOutcome = {
    success: 'completed',
    collision: 'collision',
    safety_stop: 'safety_stop',
    timeout: 'timeout',
    inspection_failed: 'error',
    error: 'error',
  }[run.outcome];
  if (!result.available
    || !result.validated
    || expectedResultOutcome === undefined
    || result.outcome !== expectedResultOutcome
    || result.success !== (expectedResultOutcome === 'completed')
    || run.state !== (run.outcome === 'success' ? 'completed' : 'failed')
    || source.terminal_validation !== result.validation
    || result.elapsed_simulation_time_s !== run.simulation_time_s
    || result.checkpoints_visited !== task.completed_checkpoint_count
    || result.checkpoints_total !== task.checkpoint_count
    || result.sample_count !== inspection.sample_count
    || result.alarms_raised + result.alarms_cleared !== alarms.history_count
    || inspection.visits.reduce(
      (total, visit) => total + visit.alarm_event_count,
      0,
    ) !== alarms.history_count) {
    fail('报告终态证据之间不一致');
  }
  return {
    schema_version: REPORT_SCHEMA_VERSION,
    generated_at_utc: requireTimestamp(value.generated_at_utc, 'report.generated_at_utc'),
    source,
    run,
    task,
    result,
    inspection,
    alarms,
    map,
  };
}

function validateReportResponse(input, expectedRunId, expectedDeviceId = DEFAULT_DEVICE_ID) {
  const responseObject = requireObject(input, '报告响应');
  if (responseObject.ok !== true) {
    const allowedErrorKeys = ['ok', 'error', 'message'];
    requireExactObject(responseObject, allowedErrorKeys, '报告错误响应');
  }
  const response = responseObject.ok === true
    ? requireExactObject(
      responseObject,
      ['ok', 'schema_version', 'device_id', 'report'],
      '报告响应',
    )
    : responseObject;
  if (response.ok !== true) {
    const code = typeof response.error === 'string' ? response.error : 'REPORT_UNAVAILABLE';
    const messages = {
      AUTH_REQUIRED: '请登录后读取完整巡检报告',
      REPORT_NOT_FOUND: REPORT_NOT_FOUND_MESSAGE,
      REPORT_NOT_READY: '这次巡检尚未形成完整报告',
    };
    throw new ReportProtocolError(code, messages[code] || REPORT_UNAVAILABLE_MESSAGE);
  }
  if (response.schema_version !== REPORT_RESPONSE_SCHEMA_VERSION) fail('报告响应协议版本不受支持');
  const deviceId = requireString(response.device_id, 'device_id', 64);
  if (deviceId !== expectedDeviceId) fail('device_id 与固定巡检设备不一致');
  const report = validatePatrolReport(response.report, expectedRunId);
  return report;
}

function projectSource(source) {
  return {
    terminal_validation: source.terminal_validation,
    map_source_file: source.map_source_file,
    map_qualification: source.map_qualification,
    map_boundary_note: source.map_boundary_note,
  };
}

function projectRun(run) {
  return {
    run_id: run.run_id,
    seed: run.seed,
    state: run.state,
    outcome: run.outcome,
    sequence: run.sequence,
    control_step: run.control_step,
    simulation_time_s: run.simulation_time_s,
    runtime_identity: run.runtime_identity,
  };
}

function projectTask(task) {
  return {
    completed_checkpoint_count: task.completed_checkpoint_count,
    checkpoint_count: task.checkpoint_count,
    progress_ratio: task.progress_ratio,
    progress_percent: task.progress_percent,
  };
}

function projectReportMap(map) {
  return {
    origin: map.origin,
    size: map.size,
    robot_start: map.robot_start,
    robot: map.robot,
    trajectory: map.trajectory,
    trajectory_point_count: map.trajectory_point_count,
    trajectory_source_point_count: map.trajectory_source_point_count,
    trajectory_truncated: map.trajectory_truncated,
    checkpoints: map.checkpoints,
    static_obstacles: map.static_obstacles,
  };
}

function buildPatrolReportFromDashboard(input) {
  let dashboard;
  try {
    dashboard = validateDashboardPayload(input);
  } catch (error) {
    throw new ReportProtocolError('REPORT_NOT_READY', '当前终态数据未通过完整性校验');
  }
  if (!dashboard.run.terminal
    || dashboard.result.available !== true
    || dashboard.result.validated !== true
    || dashboard.inspection.available !== true
    || dashboard.alarms.history_truncated
    || dashboard.alarms.history_count !== dashboard.alarms.history.length) {
    throw new ReportProtocolError('REPORT_NOT_READY', '当前巡检尚未形成完整报告');
  }
  return validatePatrolReport({
    schema_version: REPORT_SCHEMA_VERSION,
    generated_at_utc: dashboard.generated_at_utc,
    source: projectSource(dashboard.source),
    run: projectRun(dashboard.run),
    task: projectTask(dashboard.task),
    result: dashboard.result,
    inspection: dashboard.inspection,
    alarms: {
      history_count: dashboard.alarms.history_count,
      history: dashboard.alarms.history.slice().sort(
        (left, right) => (
          left.simulation_time_s - right.simulation_time_s
          || (left.event_index || 0) - (right.event_index || 0)
        ),
      ),
    },
    map: projectReportMap(dashboard.map),
  });
}

function buildDemoPatrolReport() {
  const dashboard = buildDashboardPayload(fixture, fixture.frames.length - 1);
  return buildPatrolReportFromDashboard(dashboard);
}

function validateRunId(value) {
  if (typeof value !== 'string' || !RUN_ID_PATTERN.test(value)) {
    throw new ReportProtocolError('INVALID_REPORT_REQUEST', '巡检运行编号无效');
  }
  return value;
}

function unavailable(error) {
  if (error instanceof ReportProtocolError) return error;
  return new ReportProtocolError('REPORT_UNAVAILABLE', REPORT_UNAVAILABLE_MESSAGE);
}

function createPatrolReportService(options) {
  const config = options || {};
  const mode = config.mode;
  if (!['cloud', 'demo', 'local'].includes(mode)) throw new TypeError('报告来源模式无效');
  const wxApi = config.wxApi || (typeof wx === 'undefined' ? null : wx);
  const getDashboard = config.getDashboard || (() => null);
  let callFunction = config.callFunction;
  if (!callFunction && wxApi && wxApi.cloud && typeof wxApi.cloud.callFunction === 'function') {
    callFunction = wxApi.cloud.callFunction.bind(wxApi.cloud);
  }
  return Object.freeze({
    mode,
    async fetchReport(inputRunId) {
      const runId = validateRunId(inputRunId);
      if (mode === 'demo') {
        const report = buildDemoPatrolReport();
        if (report.run.run_id !== runId) throw new ReportProtocolError('REPORT_NOT_FOUND', REPORT_NOT_FOUND_MESSAGE);
        return report;
      }
      if (mode === 'local') {
        const dashboard = getDashboard();
        if (!dashboard || !dashboard.run || dashboard.run.run_id !== runId) {
          throw new ReportProtocolError('REPORT_NOT_FOUND', REPORT_NOT_FOUND_MESSAGE);
        }
        return buildPatrolReportFromDashboard(dashboard);
      }
      if (typeof callFunction !== 'function') throw unavailable();
      let response;
      try {
        response = await callFunction({ name: 'getPatrolReport', data: { run_id: runId } });
      } catch (error) {
        throw unavailable(error);
      }
      try {
        return validateReportResponse(response && response.result, runId);
      } catch (error) {
        throw unavailable(error);
      }
    },
  });
}

module.exports = {
  DEFAULT_DEVICE_ID,
  REPORT_NOT_FOUND_MESSAGE,
  REPORT_RESPONSE_SCHEMA_VERSION,
  REPORT_SCHEMA_VERSION,
  REPORT_UNAVAILABLE_MESSAGE,
  ReportProtocolError,
  buildDemoPatrolReport,
  buildPatrolReportFromDashboard,
  createPatrolReportService,
  validatePatrolReport,
  validateReportResponse,
};
