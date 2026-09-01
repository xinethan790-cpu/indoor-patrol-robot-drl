'use strict';

const DASHBOARD_SCHEMA_VERSION = 'patrol-dashboard/v1';
const FIXTURE_SCHEMA_VERSION = 'wechat-patrol-fixture/v1';

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

function validateDashboardPayload(payload) {
  const root = requireObject(payload, 'payload');
  if (root.schema_version !== DASHBOARD_SCHEMA_VERSION) {
    throw new PayloadProtocolError(
      'SCHEMA_MISMATCH',
      `不支持的数据协议：${String(root.schema_version)}`,
    );
  }
  const run = requireObject(root.run, 'run');
  if (typeof run.run_id !== 'string' || run.run_id.trim() === '') {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'run.run_id 必须是非空字符串');
  }
  if (!Number.isSafeInteger(run.sequence) || run.sequence < 0) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'run.sequence 必须是非负安全整数');
  }
  if (typeof run.terminal !== 'boolean') {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'run.terminal 必须是布尔值');
  }
  requireFiniteNumber(run.simulation_time_s, 'run.simulation_time_s');
  requireObject(root.map, 'map');
  requireObject(root.task, 'task');
  requireObject(root.sensors, 'sensors');
  requireObject(root.alarms, 'alarms');
  if (!Array.isArray(root.map.trajectory)) {
    throw new PayloadProtocolError('INVALID_PAYLOAD', 'map.trajectory 必须是数组');
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
  createPayloadGate,
  validateDashboardPayload,
  validateFixture,
};
