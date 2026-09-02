'use strict';

const HISTORY_SCHEMA_VERSION = 'patrol-history/v1';
const DEFAULT_PAGE_SIZE = 10;
const DEFAULT_DEVICE_ID = 'macmini-m4';
const MAX_PAGE_SIZE = 20;
const STALE_AFTER_MS = 8000;
const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const CURSOR_PATTERN = /^[A-Za-z0-9_-]{1,512}$/;
const HISTORY_UNAVAILABLE_MESSAGE = '巡检历史暂时不可用，请稍后重试';
const HISTORY_STATUSES = new Set([
  'active',
  'completed',
  'aborted',
  'interrupted',
  'unknown_terminal',
]);

class HistoryProtocolError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'HistoryProtocolError';
    this.code = code;
  }
}

function fail(message) {
  throw new HistoryProtocolError('INVALID_HISTORY_RESPONSE', message);
}

function unavailable() {
  return new HistoryProtocolError('HISTORY_UNAVAILABLE', HISTORY_UNAVAILABLE_MESSAGE);
}

function requireObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`${label} 必须是对象`);
  return value;
}

function requireString(value, label, maximum) {
  if (typeof value !== 'string' || value.trim() === '' || value.length > maximum) {
    fail(`${label} 必须是有效字符串`);
  }
  return value;
}

function requireInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) fail(`${label} 必须是非负安全整数`);
  return value;
}

function nullableInteger(value, label) {
  if (value === null) return null;
  return requireInteger(value, label);
}

function nullableString(value, label) {
  if (value === null) return null;
  return requireString(value, label, 128);
}

function requireTimestamp(value, label) {
  requireString(value, label, 64);
  if (Number.isNaN(Date.parse(value))) fail(`${label} 必须是有效时间`);
  return value;
}

function projectHistoryItem(input) {
  const item = requireObject(input, '历史记录');
  const runId = requireString(item.run_id, 'run_id', 128);
  if (!RUN_ID_PATTERN.test(runId)) fail('run_id 格式无效');
  if (!HISTORY_STATUSES.has(item.status)) fail('历史状态无效');
  if (typeof item.terminal !== 'boolean') fail('terminal 必须是布尔值');
  const nonTerminalStatus = item.status === 'active' || item.status === 'interrupted';
  if (nonTerminalStatus !== !item.terminal) {
    fail('历史状态与 terminal 不一致');
  }
  const progress = item.progress_percent;
  if (
    progress !== null
    && (typeof progress !== 'number' || !Number.isFinite(progress) || progress < 0 || progress > 100)
  ) {
    fail('progress_percent 超出范围');
  }
  const completed = nullableInteger(
    item.completed_checkpoint_count,
    'completed_checkpoint_count',
  );
  const checkpointCount = nullableInteger(item.checkpoint_count, 'checkpoint_count');
  if (completed !== null && checkpointCount !== null && completed > checkpointCount) {
    fail('巡检点数量不一致');
  }
  if (typeof item.result_validated !== 'boolean') fail('result_validated 必须是布尔值');
  if (item.status !== 'completed' && item.result_validated) {
    fail('未完成记录不能声明已验证终态');
  }
  if (item.status === 'aborted' && item.result_outcome !== 'aborted') {
    fail('异常中止记录缺少 aborted 结果标记');
  }
  const firstReceivedAtUtc = requireTimestamp(
    item.first_received_at_utc,
    'first_received_at_utc',
  );
  const latestReceivedAtUtc = requireTimestamp(
    item.latest_received_at_utc,
    'latest_received_at_utc',
  );
  if (Date.parse(latestReceivedAtUtc) < Date.parse(firstReceivedAtUtc)) {
    fail('历史记录时间顺序无效');
  }
  return {
    run_id: runId,
    status: item.status,
    terminal: item.terminal,
    first_received_at_utc: firstReceivedAtUtc,
    latest_received_at_utc: latestReceivedAtUtc,
    latest_age_ms: requireInteger(item.latest_age_ms, 'latest_age_ms'),
    latest_sequence: requireInteger(item.latest_sequence, 'latest_sequence'),
    state: nullableString(item.state, 'state'),
    outcome: nullableString(item.outcome, 'outcome'),
    progress_percent: progress,
    completed_checkpoint_count: completed,
    checkpoint_count: checkpointCount,
    active_alarm_count: nullableInteger(item.active_alarm_count, 'active_alarm_count'),
    alarm_history_count: nullableInteger(item.alarm_history_count, 'alarm_history_count'),
    result_outcome: nullableString(item.result_outcome, 'result_outcome'),
    result_validated: item.result_validated,
  };
}

function validateHistoryResponse(input, requestedPageSize, expectedDeviceId) {
  const response = requireObject(input, '历史响应');
  if (response.ok !== true) {
    const code = typeof response.error === 'string' ? response.error : 'HISTORY_UNAVAILABLE';
    const message = code === 'AUTH_REQUIRED'
      ? '请登录后读取巡检历史'
      : '巡检历史暂时不可用，请稍后重试';
    throw new HistoryProtocolError(code, message);
  }
  if (response.schema_version !== HISTORY_SCHEMA_VERSION) fail('历史协议版本不受支持');
  const deviceId = requireString(response.device_id, 'device_id', 64);
  if (deviceId !== (expectedDeviceId || DEFAULT_DEVICE_ID)) fail('device_id 与固定巡检设备不一致');
  if (response.page_size !== requestedPageSize) fail('page_size 与请求不一致');
  const serverTimeEpochMs = requireInteger(
    response.server_time_epoch_ms,
    'server_time_epoch_ms',
  );
  if (response.stale_after_ms !== STALE_AFTER_MS) fail('stale_after_ms 与客户端协议不一致');
  if (!Array.isArray(response.items) || response.items.length > requestedPageSize) {
    fail('items 数量超出分页上限');
  }
  if (typeof response.has_more !== 'boolean') fail('has_more 必须是布尔值');
  const nextCursor = response.next_cursor;
  if (
    (response.has_more && (typeof nextCursor !== 'string' || !CURSOR_PATTERN.test(nextCursor)))
    || (!response.has_more && nextCursor !== null)
  ) {
    fail('next_cursor 与 has_more 不一致');
  }
  const items = response.items.map(projectHistoryItem);
  for (const item of items) {
    const expectedAgeMs = Math.max(
      0,
      serverTimeEpochMs - Date.parse(item.latest_received_at_utc),
    );
    if (item.latest_age_ms !== expectedAgeMs) fail('历史记录数据年龄与服务器时间不一致');
    if (!item.terminal) {
      const expectedStatus = expectedAgeMs > STALE_AFTER_MS ? 'interrupted' : 'active';
      if (item.status !== expectedStatus) fail('非终态记录的新鲜度状态不一致');
    }
  }
  if (new Set(items.map((item) => item.run_id)).size !== items.length) {
    fail('同一页出现重复 run_id');
  }
  return {
    deviceId,
    pageSize: requestedPageSize,
    serverTimeEpochMs,
    items,
    hasMore: response.has_more,
    nextCursor,
  };
}

function createPatrolHistoryService(options) {
  const config = options || {};
  const wxApi = config.wxApi || (typeof wx === 'undefined' ? null : wx);
  const pageSize = config.pageSize ?? DEFAULT_PAGE_SIZE;
  const deviceId = config.deviceId || DEFAULT_DEVICE_ID;
  if (deviceId !== DEFAULT_DEVICE_ID) {
    throw new RangeError('历史服务只允许读取固定巡检设备');
  }
  if (!Number.isSafeInteger(pageSize) || pageSize < 1 || pageSize > MAX_PAGE_SIZE) {
    throw new RangeError(`pageSize 必须是 1 至 ${MAX_PAGE_SIZE} 的整数`);
  }
  const callFunction = config.callFunction
    || (wxApi && wxApi.cloud && typeof wxApi.cloud.callFunction === 'function'
      ? wxApi.cloud.callFunction.bind(wxApi.cloud)
      : null);
  if (typeof callFunction !== 'function') throw new TypeError('当前微信基础库不支持云函数');

  return Object.freeze({
    pageSize,
    async fetchPage(cursor) {
      const safeCursor = cursor === undefined ? null : cursor;
      if (safeCursor !== null && (
        typeof safeCursor !== 'string' || !CURSOR_PATTERN.test(safeCursor)
      )) {
        throw new HistoryProtocolError('INVALID_CURSOR', '分页游标无效');
      }
      let response;
      try {
        response = await callFunction({
          name: 'getPatrolHistory',
          data: { page_size: pageSize, cursor: safeCursor },
        });
      } catch (error) {
        throw unavailable();
      }
      try {
        return validateHistoryResponse(response && response.result, pageSize, deviceId);
      } catch (error) {
        if (error instanceof HistoryProtocolError) throw error;
        throw unavailable();
      }
    },
  });
}

module.exports = {
  DEFAULT_PAGE_SIZE,
  DEFAULT_DEVICE_ID,
  HISTORY_SCHEMA_VERSION,
  STALE_AFTER_MS,
  HistoryProtocolError,
  createPatrolHistoryService,
  projectHistoryItem,
  validateHistoryResponse,
};
