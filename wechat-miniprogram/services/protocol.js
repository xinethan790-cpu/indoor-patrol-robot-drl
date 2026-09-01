'use strict';

const DASHBOARD_SCHEMA_VERSION = 'patrol-dashboard/v1';
const FIXTURE_SCHEMA_VERSION = 'wechat-patrol-fixture/v1';
const MAX_DASHBOARD_PAYLOAD_BYTES = 128 * 1024;
const MAX_TRAJECTORY_POINTS = 300;
const MAX_ALARM_EVENTS = 200;
const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

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
  const root = requireObject(payload, 'payload');
  validatePayloadSize(root);
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
  utf8ByteLength,
  validateDashboardPayload,
  validateFixture,
};
