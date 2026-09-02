'use strict';

const {
  MAX_ALARM_EVENTS,
  MAX_CHECKPOINTS,
  MAX_STATIC_OBSTACLES,
  INSPECTION_SENSOR_SPEC,
  RUN_ID_PATTERN,
  isPlainObject,
  validateDashboardPayload,
  validateInspection,
} = require('./payload');
const { REPORT_SCHEMA_VERSION } = require('./config');
const { badRequest } = require('./errors');

const MAP_BOUNDARY_TOLERANCE_M = 1e-6;
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const SAFE_MAP_SOURCE_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const SAFE_MODEL_ID_PATTERN = /^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)*$/;
const UTC_TIMESTAMP_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$/;
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
const SEVERITIES = new Set(['info', 'warning', 'critical']);
const TERMINAL_RESULT_OUTCOME_BY_RUN_OUTCOME = Object.freeze({
  success: 'completed',
  collision: 'collision',
  safety_stop: 'safety_stop',
  timeout: 'timeout',
  inspection_failed: 'error',
  error: 'error',
});
const SEVERITY_RANK = Object.freeze({ info: 1, warning: 2, critical: 3 });

function exactObject(value, expectedKeys, label) {
  if (!isPlainObject(value)) throw new TypeError(`${label} must be an object`);
  const actual = Object.keys(value).sort();
  const expected = [...expectedKeys].sort();
  if (actual.length !== expected.length
      || actual.some((key, index) => key !== expected[index])) {
    throw new TypeError(`${label} contains unsupported or missing fields`);
  }
  return value;
}

function text(value, label, maximum = 256) {
  if (typeof value !== 'string' || value.trim() === '' || value.length > maximum) {
    throw new TypeError(`${label} must be bounded non-empty text`);
  }
  return value;
}

function nullableText(value, label, maximum = 256) {
  if (value === null) return value;
  return text(value, label, maximum);
}

function integer(value, label, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new TypeError(`${label} must be a bounded safe integer`);
  }
  return value;
}

function number(value, label, minimum) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number`);
  }
  if (minimum !== undefined && value < minimum) {
    throw new TypeError(`${label} is below the supported minimum`);
  }
  return value;
}

function nullableNumber(value, label, minimum) {
  if (value === null) return null;
  return number(value, label, minimum);
}

function nullableInteger(value, label, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
  if (value === null) return null;
  return integer(value, label, minimum, maximum);
}

function boolean(value, label) {
  if (typeof value !== 'boolean') throw new TypeError(`${label} must be a boolean`);
  return value;
}

function point(value, label) {
  const item = exactObject(value, ['x_m', 'y_m'], label);
  number(item.x_m, `${label}.x_m`);
  number(item.y_m, `${label}.y_m`);
  return item;
}

function size(value, label) {
  const item = exactObject(value, ['width_m', 'height_m'], label);
  if (number(item.width_m, `${label}.width_m`) <= 0
      || number(item.height_m, `${label}.height_m`) <= 0) {
    throw new TypeError(`${label} dimensions must be greater than zero`);
  }
  return item;
}

function insideMap(position, map, label) {
  const minimumX = map.origin.x_m;
  const maximumX = minimumX + map.size.width_m;
  const minimumY = map.origin.y_m;
  const maximumY = minimumY + map.size.height_m;
  if (position.x_m < minimumX - MAP_BOUNDARY_TOLERANCE_M
      || position.x_m > maximumX + MAP_BOUNDARY_TOLERANCE_M
      || position.y_m < minimumY - MAP_BOUNDARY_TOLERANCE_M
      || position.y_m > maximumY + MAP_BOUNDARY_TOLERANCE_M) {
    throw new TypeError(`${label} is outside the demonstration map boundary`);
  }
}

function inspectionProjection(value) {
  if (value.available !== true) {
    return {
      available: false,
      sample_count: null,
      visit_count: 0,
      visits: [],
    };
  }
  return {
    available: true,
    sample_count: value.sample_count,
    visit_count: value.visit_count,
    visits: value.visits.map((visit) => ({
      visit_index: visit.visit_index,
      checkpoint_id: visit.checkpoint_id,
      control_step: visit.control_step,
      simulation_time_s: visit.simulation_time_s,
      alarm_event_count: visit.alarm_event_count,
      highest_active_severity: visit.highest_active_severity,
      readings: Object.fromEntries(
        ['temperature', 'smoke_obscuration', 'carbon_monoxide'].map((sensorId) => {
          const reading = visit.readings[sensorId];
          return [sensorId, {
            display_name: reading.display_name,
            value: reading.value,
            unit: reading.unit,
            quality: reading.quality,
            sample_index: reading.sample_index,
          }];
        }),
      ),
    })),
  };
}

function resultProjection(value) {
  return Object.fromEntries(RESULT_KEYS.map((key) => [key, value[key]]));
}

function alarmProjection(value) {
  return Object.fromEntries(ALARM_KEYS.map((key) => [key, value[key]]));
}

function buildCandidate(payload) {
  return {
    schema_version: REPORT_SCHEMA_VERSION,
    generated_at_utc: payload.generated_at_utc,
    source: {
      terminal_validation: payload.source.terminal_validation,
      map_source_file: payload.source.map_source_file,
      map_qualification: payload.source.map_qualification,
      map_boundary_note: payload.source.map_boundary_note,
    },
    run: {
      run_id: payload.run.run_id,
      seed: payload.run.seed,
      state: payload.run.state,
      outcome: payload.run.outcome,
      sequence: payload.run.sequence,
      control_step: payload.run.control_step,
      simulation_time_s: payload.run.simulation_time_s,
      runtime_identity: {
        model_id: payload.run.runtime_identity && payload.run.runtime_identity.model_id,
        model_sha256: payload.run.runtime_identity && payload.run.runtime_identity.model_sha256,
        safety_shield_enabled: payload.run.runtime_identity
          && payload.run.runtime_identity.safety_shield_enabled,
        safety_shield_id: payload.run.runtime_identity
          && payload.run.runtime_identity.safety_shield_id,
        safety_shield_sha256: payload.run.runtime_identity
          && payload.run.runtime_identity.safety_shield_sha256,
      },
    },
    task: {
      completed_checkpoint_count: payload.task.completed_checkpoint_count,
      checkpoint_count: payload.task.checkpoint_count,
      progress_ratio: payload.task.progress_ratio,
      progress_percent: payload.task.progress_percent,
    },
    result: resultProjection(payload.result),
    inspection: inspectionProjection(payload.inspection),
    alarms: {
      history_count: payload.alarms.history.length,
      history: [...payload.alarms.history].reverse().map(alarmProjection),
    },
    map: {
      origin: { x_m: payload.map.origin.x_m, y_m: payload.map.origin.y_m },
      size: {
        width_m: payload.map.size.width_m,
        height_m: payload.map.size.height_m,
      },
      robot_start: {
        x_m: payload.map.robot_start && payload.map.robot_start.x_m,
        y_m: payload.map.robot_start && payload.map.robot_start.y_m,
      },
      robot: { x_m: payload.map.robot.x_m, y_m: payload.map.robot.y_m },
      trajectory: payload.map.trajectory.map((entry) => ({
        sequence: entry.sequence,
        simulation_time_s: entry.simulation_time_s,
        x_m: entry.x_m,
        y_m: entry.y_m,
      })),
      trajectory_point_count: payload.map.trajectory_point_count,
      trajectory_source_point_count: payload.map.trajectory_source_point_count,
      trajectory_truncated: payload.map.trajectory_truncated,
      checkpoints: payload.map.checkpoints.map((checkpoint) => ({
        id: checkpoint.id,
        ordinal: checkpoint.ordinal,
        position: { x_m: checkpoint.position.x_m, y_m: checkpoint.position.y_m },
        status: checkpoint.status,
      })),
      static_obstacles: payload.map.static_obstacles.map((obstacle) => ({
        id: obstacle.id,
        label: obstacle.label,
        center: { x_m: obstacle.center.x_m, y_m: obstacle.center.y_m },
        size: { width_m: obstacle.size.width_m, height_m: obstacle.size.height_m },
      })),
    },
  };
}

function validateRuntimeIdentity(value) {
  const identity = exactObject(value, [
    'model_id',
    'model_sha256',
    'safety_shield_enabled',
    'safety_shield_id',
    'safety_shield_sha256',
  ], 'report.run.runtime_identity');
  const modelId = text(identity.model_id, 'report.run.runtime_identity.model_id', 512);
  const modelSegments = modelId.split('/');
  if (!SAFE_MODEL_ID_PATTERN.test(modelId)
      || modelSegments.some((segment) => segment === '.' || segment === '..')) {
    throw new TypeError('report model identity is not a safe project-relative identifier');
  }
  if (!SHA256_PATTERN.test(identity.model_sha256)) {
    throw new TypeError('report model SHA-256 is invalid');
  }
  const shieldEnabled = boolean(
    identity.safety_shield_enabled,
    'report.run.runtime_identity.safety_shield_enabled',
  );
  if (shieldEnabled) {
    text(identity.safety_shield_id, 'report.run.runtime_identity.safety_shield_id', 128);
    if (!SHA256_PATTERN.test(identity.safety_shield_sha256)) {
      throw new TypeError('report safety-shield SHA-256 is invalid');
    }
  } else if (identity.safety_shield_id !== null || identity.safety_shield_sha256 !== null) {
    throw new TypeError('disabled report safety shield cannot carry identity fields');
  }
}

function validateResult(value, task) {
  const result = exactObject(value, RESULT_KEYS, 'report.result');
  if (result.available !== true || result.validated !== true || result.success !== (
    result.outcome === 'completed'
  )) {
    throw new TypeError('report result is not an ordinary validated terminal result');
  }
  text(result.validation, 'report.result.validation', 128);
  if (!['completed', 'collision', 'safety_stop', 'timeout', 'error'].includes(result.outcome)) {
    throw new TypeError('report.result.outcome is invalid');
  }
  nullableText(result.failure_reason, 'report.result.failure_reason', 128);
  const visited = integer(result.checkpoints_visited, 'report.result.checkpoints_visited');
  const total = integer(result.checkpoints_total, 'report.result.checkpoints_total', 1, MAX_CHECKPOINTS);
  if (visited !== task.completed_checkpoint_count || total !== task.checkpoint_count) {
    throw new TypeError('report result checkpoint counts disagree with task');
  }
  integer(result.collision_count, 'report.result.collision_count');
  for (const key of [
    'critical_proximity_count',
    'safety_shield_intervention_steps',
    'ttc_escape_guard_intervention_steps',
    'ttc_escape_guard_activation_count',
    'ttc_escape_guard_recovery_count',
    'ttc_escape_guard_unrecoverable_count',
    'dynamic_obstacle_count',
  ]) nullableInteger(result[key], `report.result.${key}`);
  for (const key of ['sample_count', 'alarms_raised', 'alarms_cleared']) {
    integer(result[key], `report.result.${key}`);
  }
  number(
    result.elapsed_simulation_time_s,
    'report.result.elapsed_simulation_time_s',
    0,
  );
  for (const key of [
    'actual_path_length_m',
    'minimum_clearance_m',
    'minimum_lidar_range_m',
    'inference_latency_p50_ms',
    'inference_latency_p95_ms',
    'dynamic_obstacle_speed_max_mps',
  ]) nullableNumber(result[key], `report.result.${key}`, 0);
  const rate = nullableNumber(
    result.safety_shield_intervention_rate,
    'report.result.safety_shield_intervention_rate',
    0,
  );
  if (rate !== null && rate > 1) {
    throw new TypeError('report safety-shield intervention rate exceeds one');
  }
  if ((result.dynamic_obstacle_count === null)
      !== (result.dynamic_obstacle_speed_max_mps === null)) {
    throw new TypeError('report dynamic-obstacle metrics must be available together');
  }
  if (result.dynamic_obstacle_count === 0 && result.dynamic_obstacle_speed_max_mps !== 0) {
    throw new TypeError('zero dynamic obstacles require zero maximum speed');
  }
  if (result.inference_latency_p50_ms !== null
      && result.inference_latency_p95_ms !== null
      && result.inference_latency_p95_ms < result.inference_latency_p50_ms) {
    throw new TypeError('report inference latency percentiles are inconsistent');
  }
  if (result.highest_event_severity !== null
      && !SEVERITIES.has(result.highest_event_severity)) {
    throw new TypeError('report.result.highest_event_severity is invalid');
  }
  if ((result.outcome === 'completed'
      && (result.failure_reason !== null || result.collision_count !== 0))
      || (result.outcome !== 'completed' && result.failure_reason === null)
      || (result.outcome === 'collision' && result.collision_count < 1)
      || (result.outcome !== 'collision' && result.collision_count !== 0)
      || (result.outcome === 'safety_stop'
        && (result.critical_proximity_count === null
          || result.critical_proximity_count < 1))) {
    throw new TypeError('report terminal evidence counters disagree with result outcome');
  }
  return result;
}

function validateAlarms(value, run, checkpointIds) {
  const alarms = exactObject(value, ['history_count', 'history'], 'report.alarms');
  const historyCount = integer(alarms.history_count, 'report.alarms.history_count');
  if (!Array.isArray(alarms.history)
      || alarms.history.length > MAX_ALARM_EVENTS
      || alarms.history.length !== historyCount) {
    throw new TypeError('report alarm history cardinality is invalid');
  }
  const eventIds = new Set();
  const eventIndexes = new Set();
  let previousControlStep = -1;
  let previousSimulationTime = -1;
  alarms.history.forEach((raw, index) => {
    const label = `report.alarms.history[${index}]`;
    const event = exactObject(raw, ALARM_KEYS, label);
    const eventIndex = integer(event.event_index, `${label}.event_index`, 1);
    if (eventIndex !== index + 1 || eventIndexes.has(eventIndex)) {
      throw new TypeError('report alarm event indexes must be contiguous from one');
    }
    eventIndexes.add(eventIndex);
    const eventId = text(event.event_id, `${label}.event_id`, 128);
    if (eventId !== `alarm-${eventIndex}` || eventIds.has(eventId)) {
      throw new TypeError('report alarm event identity is invalid');
    }
    eventIds.add(eventId);
    if (!['raised', 'cleared'].includes(event.event_type)) {
      throw new TypeError(`${label}.event_type is invalid`);
    }
    text(event.rule_id, `${label}.rule_id`, 128);
    const sensorId = text(event.sensor_id, `${label}.sensor_id`, 64);
    if (!Object.hasOwn(INSPECTION_SENSOR_SPEC, sensorId)) {
      throw new TypeError(`${label}.sensor_id is not in the inspection whitelist`);
    }
    if (!SEVERITIES.has(event.severity)) throw new TypeError(`${label}.severity is invalid`);
    const checkpointId = text(event.checkpoint_id, `${label}.checkpoint_id`, 64);
    if (!checkpointIds.has(checkpointId)) throw new TypeError(`${label} checkpoint is unknown`);
    integer(event.visit_index, `${label}.visit_index`, 1, MAX_CHECKPOINTS);
    integer(event.sample_index, `${label}.sample_index`, 1);
    const controlStep = integer(
      event.control_step,
      `${label}.control_step`,
      0,
      run.control_step,
    );
    const eventTime = number(event.simulation_time_s, `${label}.simulation_time_s`, 0);
    if (controlStep < previousControlStep
        || eventTime < previousSimulationTime
        || eventTime > run.simulation_time_s + 1e-9) {
      throw new TypeError(`${label} is not ordered within the terminal runtime`);
    }
    previousControlStep = controlStep;
    previousSimulationTime = eventTime;
    number(event.observed_value, `${label}.observed_value`);
    number(event.threshold, `${label}.threshold`);
    const unit = text(event.unit, `${label}.unit`, 32);
    if (unit !== INSPECTION_SENSOR_SPEC[sensorId].unit) {
      throw new TypeError(`${label}.unit does not match its inspection sensor`);
    }
    text(event.message, `${label}.message`, 500);
  });
  return alarms;
}

function validateAlarmInspectionConsistency(alarms, inspection, result) {
  const eventsByVisit = new Map(
    inspection.visits.map((visit) => [visit.visit_index, []]),
  );
  let raisedCount = 0;
  let clearedCount = 0;
  let highestEventSeverity = null;
  const sensorBySampleRemainder = [
    'temperature',
    'smoke_obscuration',
    'carbon_monoxide',
  ];
  for (const event of alarms.history) {
    const visit = inspection.visits[event.visit_index - 1];
    const nextVisit = inspection.visits[event.visit_index];
    const previousVisitSampleEnd = event.visit_index === 1
      ? 0
      : inspection.visits[event.visit_index - 2]
        .readings.carbon_monoxide.sample_index;
    const visitSampleEnd = visit
      && visit.readings.carbon_monoxide.sample_index;
    const expectedSensor = sensorBySampleRemainder[(event.sample_index - 1) % 3];
    if (!visit
        || visit.visit_index !== event.visit_index
        || visit.checkpoint_id !== event.checkpoint_id
        || visit.control_step !== event.control_step
        || event.simulation_time_s < visit.simulation_time_s - 1e-9
        || (nextVisit
          && event.simulation_time_s > nextVisit.simulation_time_s + 1e-9)
        || event.sample_index <= previousVisitSampleEnd
        || event.sample_index > visitSampleEnd
        || event.sensor_id !== expectedSensor) {
      throw new TypeError('report alarm provenance does not match its inspection visit');
    }
    eventsByVisit.get(event.visit_index).push(event);
    if (event.event_type === 'raised') raisedCount += 1;
    else clearedCount += 1;
    if (highestEventSeverity === null
        || SEVERITY_RANK[event.severity] > SEVERITY_RANK[highestEventSeverity]) {
      highestEventSeverity = event.severity;
    }
  }
  if (raisedCount !== result.alarms_raised
      || clearedCount !== result.alarms_cleared
      || highestEventSeverity !== result.highest_event_severity) {
    throw new TypeError('report result alarm metrics disagree with the complete history');
  }

  const activeRules = new Map();
  for (const visit of inspection.visits) {
    const visitEvents = eventsByVisit.get(visit.visit_index);
    if (visitEvents.length !== visit.alarm_event_count) {
      throw new TypeError('report inspection visit alarm count disagrees with history');
    }
    for (const event of visitEvents) {
      const current = activeRules.get(event.rule_id);
      if (event.event_type === 'raised') {
        if (current) throw new TypeError('report alarm history raises an active rule');
        activeRules.set(event.rule_id, {
          sensorId: event.sensor_id,
          severity: event.severity,
        });
      } else {
        if (!current
            || current.sensorId !== event.sensor_id
            || current.severity !== event.severity) {
          throw new TypeError('report alarm history clears an inactive or different rule');
        }
        activeRules.delete(event.rule_id);
      }
    }
    let highestActiveSeverity = null;
    for (const active of activeRules.values()) {
      if (highestActiveSeverity === null
          || SEVERITY_RANK[active.severity] > SEVERITY_RANK[highestActiveSeverity]) {
        highestActiveSeverity = active.severity;
      }
    }
    if (visit.highest_active_severity !== highestActiveSeverity) {
      throw new TypeError('report inspection active severity disagrees with alarm history');
    }
  }
}

function validateMap(value, run) {
  const map = exactObject(value, [
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
  point(map.origin, 'report.map.origin');
  size(map.size, 'report.map.size');
  insideMap(point(map.robot_start, 'report.map.robot_start'), map, 'report.map.robot_start');
  insideMap(point(map.robot, 'report.map.robot'), map, 'report.map.robot');

  if (!Array.isArray(map.trajectory)
      || map.trajectory.length < 1
      || map.trajectory.length > 300) {
    throw new TypeError('report.map.trajectory is invalid');
  }
  let previousSequence = -1;
  let previousTime = -1;
  map.trajectory.forEach((raw, index) => {
    const label = `report.map.trajectory[${index}]`;
    const entry = exactObject(raw, ['sequence', 'simulation_time_s', 'x_m', 'y_m'], label);
    const sequence = integer(entry.sequence, `${label}.sequence`);
    const simulationTime = number(entry.simulation_time_s, `${label}.simulation_time_s`, 0);
    if (sequence <= previousSequence || simulationTime < previousTime
        || sequence > run.sequence || simulationTime > run.simulation_time_s + 1e-9) {
      throw new TypeError('report trajectory is not monotonic within the terminal runtime');
    }
    previousSequence = sequence;
    previousTime = simulationTime;
    insideMap(point({ x_m: entry.x_m, y_m: entry.y_m }, label), map, label);
  });
  const returnedCount = integer(
    map.trajectory_point_count,
    'report.map.trajectory_point_count',
    0,
    300,
  );
  const sourceCount = integer(
    map.trajectory_source_point_count,
    'report.map.trajectory_source_point_count',
  );
  const truncated = boolean(map.trajectory_truncated, 'report.map.trajectory_truncated');
  if (returnedCount !== map.trajectory.length
      || sourceCount < returnedCount
      || sourceCount > run.sequence + 1
      || truncated !== (sourceCount > returnedCount)) {
    throw new TypeError('report trajectory counters are inconsistent');
  }
  const finalTrajectory = map.trajectory[map.trajectory.length - 1];
  if (finalTrajectory.sequence !== run.sequence
      || Math.abs(finalTrajectory.simulation_time_s - run.simulation_time_s) > 1e-9
      || Math.abs(finalTrajectory.x_m - map.robot.x_m) > 1e-9
      || Math.abs(finalTrajectory.y_m - map.robot.y_m) > 1e-9) {
    throw new TypeError('report final trajectory point does not match the terminal run pose');
  }

  if (!Array.isArray(map.checkpoints)
      || map.checkpoints.length < 1
      || map.checkpoints.length > MAX_CHECKPOINTS) {
    throw new TypeError('report.map.checkpoints cardinality is invalid');
  }
  const checkpointIds = new Set();
  map.checkpoints.forEach((raw, index) => {
    const label = `report.map.checkpoints[${index}]`;
    const checkpoint = exactObject(raw, ['id', 'ordinal', 'position', 'status'], label);
    const id = text(checkpoint.id, `${label}.id`, 64);
    if (checkpointIds.has(id) || checkpoint.ordinal !== index + 1) {
      throw new TypeError('report checkpoint identity or ordinal is invalid');
    }
    checkpointIds.add(id);
    if (!['pending', 'current', 'active', 'completed'].includes(checkpoint.status)) {
      throw new TypeError(`${label}.status is invalid`);
    }
    insideMap(point(checkpoint.position, `${label}.position`), map, label);
  });

  if (!Array.isArray(map.static_obstacles)
      || map.static_obstacles.length > MAX_STATIC_OBSTACLES) {
    throw new TypeError('report.map.static_obstacles cardinality is invalid');
  }
  const obstacleIds = new Set();
  map.static_obstacles.forEach((raw, index) => {
    const label = `report.map.static_obstacles[${index}]`;
    const obstacle = exactObject(raw, ['id', 'label', 'center', 'size'], label);
    const id = text(obstacle.id, `${label}.id`, 64);
    if (obstacleIds.has(id)) throw new TypeError('report obstacle ids must be unique');
    obstacleIds.add(id);
    text(obstacle.label, `${label}.label`, 128);
    const center = point(obstacle.center, `${label}.center`);
    const dimensions = size(obstacle.size, `${label}.size`);
    const halfWidth = dimensions.width_m / 2;
    const halfHeight = dimensions.height_m / 2;
    insideMap({ x_m: center.x_m - halfWidth, y_m: center.y_m - halfHeight }, map, label);
    insideMap({ x_m: center.x_m + halfWidth, y_m: center.y_m + halfHeight }, map, label);
  });
  return checkpointIds;
}

function validatePatrolReport(value, expectedRunId) {
  const report = exactObject(value, REPORT_KEYS, 'report');
  if (report.schema_version !== REPORT_SCHEMA_VERSION) {
    throw new TypeError('report schema is unsupported');
  }
  if (typeof report.generated_at_utc !== 'string'
      || report.generated_at_utc.length > 64
      || !UTC_TIMESTAMP_PATTERN.test(report.generated_at_utc)
      || Number.isNaN(Date.parse(report.generated_at_utc))) {
    throw new TypeError('report generated timestamp is invalid');
  }
  const source = exactObject(report.source, [
    'terminal_validation',
    'map_source_file',
    'map_qualification',
    'map_boundary_note',
  ], 'report.source');
  text(source.terminal_validation, 'report.source.terminal_validation', 128);
  if (typeof source.map_source_file !== 'string'
      || !SAFE_MAP_SOURCE_PATTERN.test(source.map_source_file)) {
    throw new TypeError('report map source is not a safe file identifier');
  }
  if (source.map_qualification !== 'demonstration_only') {
    throw new TypeError('report map qualification is invalid');
  }
  text(source.map_boundary_note, 'report.source.map_boundary_note', 500);

  const run = exactObject(report.run, [
    'run_id',
    'seed',
    'state',
    'outcome',
    'sequence',
    'control_step',
    'simulation_time_s',
    'runtime_identity',
  ], 'report.run');
  if (typeof run.run_id !== 'string'
      || !RUN_ID_PATTERN.test(run.run_id)
      || (expectedRunId !== undefined && run.run_id !== expectedRunId)) {
    throw new TypeError('report run identity is invalid');
  }
  integer(run.seed, 'report.run.seed');
  if (!['completed', 'failed'].includes(run.state) || run.outcome === 'running') {
    throw new TypeError('report run is not terminal');
  }
  text(run.outcome, 'report.run.outcome', 32);
  integer(run.sequence, 'report.run.sequence');
  integer(run.control_step, 'report.run.control_step');
  number(run.simulation_time_s, 'report.run.simulation_time_s', 0);
  validateRuntimeIdentity(run.runtime_identity);

  const task = exactObject(report.task, [
    'completed_checkpoint_count',
    'checkpoint_count',
    'progress_ratio',
    'progress_percent',
  ], 'report.task');
  const checkpointCount = integer(
    task.checkpoint_count,
    'report.task.checkpoint_count',
    1,
    MAX_CHECKPOINTS,
  );
  const completed = integer(
    task.completed_checkpoint_count,
    'report.task.completed_checkpoint_count',
    0,
    checkpointCount,
  );
  const progressRatio = number(task.progress_ratio, 'report.task.progress_ratio', 0);
  const progressPercent = number(task.progress_percent, 'report.task.progress_percent', 0);
  const expectedRatio = completed / checkpointCount;
  if (progressRatio > 1
      || progressPercent > 100
      || Math.abs(progressRatio - expectedRatio) > 1e-9
      || Math.abs(progressPercent - Math.round(expectedRatio * 1000) / 10) > 1e-9) {
    throw new TypeError('report task progress is inconsistent');
  }
  const result = validateResult(report.result, task);
  const expectedResultOutcome = TERMINAL_RESULT_OUTCOME_BY_RUN_OUTCOME[run.outcome];
  const expectedRunState = run.outcome === 'success' ? 'completed' : 'failed';
  if (expectedResultOutcome === undefined
      || run.state !== expectedRunState
      || result.outcome !== expectedResultOutcome
      || result.success !== (expectedResultOutcome === 'completed')) {
    throw new TypeError('report terminal run/result lifecycle is inconsistent');
  }
  if (run.outcome === 'success') {
    if (completed !== checkpointCount
        || result.failure_reason !== null
        || result.collision_count !== 0) {
      throw new TypeError('successful report terminal metrics are inconsistent');
    }
  } else if (result.failure_reason === null) {
    throw new TypeError('failed report must include a bounded public failure reason');
  }
  if (run.outcome === 'collision' && result.collision_count < 1) {
    throw new TypeError('collision report must include a collision count');
  }
  if (source.terminal_validation !== result.validation
      || result.elapsed_simulation_time_s !== run.simulation_time_s) {
    throw new TypeError('report terminal validation or elapsed time disagrees with run');
  }
  const checkpointIds = validateMap(report.map, run);
  if (report.map.checkpoints.length !== checkpointCount) {
    throw new TypeError('report checkpoint map and task counts disagree');
  }
  let activeRemainingCheckpointCount = 0;
  report.map.checkpoints.forEach((checkpoint, index) => {
    const shouldBeCompleted = index < completed;
    if ((checkpoint.status === 'completed') !== shouldBeCompleted) {
      throw new TypeError('report terminal checkpoint status disagrees with task progress');
    }
    if (index >= completed && ['current', 'active'].includes(checkpoint.status)) {
      if (index !== completed) {
        throw new TypeError('report active checkpoint is not the next unfinished route point');
      }
      activeRemainingCheckpointCount += 1;
    }
  });
  if (activeRemainingCheckpointCount > 1) {
    throw new TypeError('report cannot contain multiple active checkpoints');
  }
  const alarms = validateAlarms(report.alarms, run, checkpointIds);
  const totalAlarmEvents = result.alarms_raised + result.alarms_cleared;
  if (totalAlarmEvents !== alarms.history_count) {
    throw new TypeError('report result alarm counters disagree with complete history');
  }
  const inspection = validateInspection(report.inspection, {
    alarms: { history_count: totalAlarmEvents },
    checkpointIds,
    checkpointSequence: report.map.checkpoints.map((checkpoint) => checkpoint.id),
    ordinaryTerminal: true,
    result,
    run,
    task,
  });
  validateAlarmInspectionConsistency(alarms, inspection, result);
  return report;
}

function isOrdinaryValidatedTerminal(payload) {
  return payload.run.terminal === true
    && payload.source.mode === 'terminal'
    && payload.result.available === true
    && payload.result.validated === true
    && payload.result.incomplete !== true
    && payload.inspection.available === true
    && payload.alarms.history_truncated === false
    && payload.alarms.history_count === payload.alarms.history.length
    && payload.alarms.history_returned_count === payload.alarms.history.length;
}

function buildPatrolReport(payload) {
  const normalized = validateDashboardPayload(payload);
  if (!isOrdinaryValidatedTerminal(normalized)) return null;
  try {
    return validatePatrolReport(buildCandidate(normalized), normalized.run.run_id);
  } catch (error) {
    if (error && error.code === 'INVALID_PAYLOAD') throw error;
    throw badRequest('INVALID_PAYLOAD', error && error.message
      ? error.message
      : 'terminal report evidence is invalid');
  }
}

module.exports = {
  ALARM_KEYS,
  REPORT_KEYS,
  RESULT_KEYS,
  buildPatrolReport,
  validatePatrolReport,
};
