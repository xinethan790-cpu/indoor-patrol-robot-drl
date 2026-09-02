'use strict';

const crypto = require('node:crypto');

const {
  createDefaultRepository,
  opaqueId,
} = require('./cloud-repository');
const {
  DEFAULT_HISTORY_PAGE_SIZE,
  HISTORY_SCHEMA_VERSION,
  MAX_HISTORY_PAGE_SIZE,
  RUN_SCHEMA_VERSION,
  STALE_AFTER_MS,
  requireUploadSecret,
  runtimeConfig,
} = require('./config');

const CURSOR_VERSION = 1;
const CURSOR_HMAC_DOMAIN = 'patrol-history-cursor/v1';
const MAX_CURSOR_LENGTH = 512;
const MAX_CURSOR_OFFSET = 1_000_000;
const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const CURSOR_PATTERN = /^[A-Za-z0-9_-]+$/;
const CURSOR_MAC_PATTERN = /^[A-Za-z0-9_-]{43}$/;

class HistoryRequestError extends Error {}

function invalidRequest(message) {
  return new HistoryRequestError(message);
}

function exactKeys(value, expected, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw invalidRequest(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw invalidRequest(`${label} contains unsupported or missing fields`);
  }
  return value;
}

function cursorDeviceBinding(deviceId) {
  return opaqueId('patrol-history-cursor', deviceId);
}

function cursorPayload(offset, pageSize, deviceId) {
  return {
    v: CURSOR_VERSION,
    o: offset,
    s: pageSize,
    d: cursorDeviceBinding(deviceId),
  };
}

function cursorMac(payload, secret) {
  const canonical = JSON.stringify(payload);
  return crypto.createHmac('sha256', secret)
    .update(`${CURSOR_HMAC_DOMAIN}\n${canonical}`, 'utf8')
    .digest('base64url');
}

function encodeCursor(offset, pageSize, deviceId, secret) {
  const cursorSecret = requireUploadSecret(secret);
  const payload = cursorPayload(offset, pageSize, deviceId);
  const raw = JSON.stringify({ ...payload, m: cursorMac(payload, cursorSecret) });
  return Buffer.from(raw, 'utf8').toString('base64url');
}

function decodeCursor(cursor, pageSize, deviceId, secret) {
  if (cursor === null || cursor === undefined || cursor === '') return 0;
  if (
    typeof cursor !== 'string'
    || cursor.length > MAX_CURSOR_LENGTH
    || !CURSOR_PATTERN.test(cursor)
  ) {
    throw invalidRequest('cursor is invalid');
  }
  let decoded;
  try {
    const bytes = Buffer.from(cursor, 'base64url');
    if (bytes.toString('base64url') !== cursor) throw new Error('non-canonical cursor');
    decoded = JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw invalidRequest('cursor is invalid');
  }
  exactKeys(decoded, ['v', 'o', 's', 'd', 'm'], 'cursor');
  const payload = cursorPayload(decoded.o, decoded.s, deviceId);
  const suppliedMac = decoded.m;
  const expectedMac = cursorMac(payload, requireUploadSecret(secret));
  const macMatches = typeof suppliedMac === 'string'
    && CURSOR_MAC_PATTERN.test(suppliedMac)
    && crypto.timingSafeEqual(Buffer.from(suppliedMac), Buffer.from(expectedMac));
  if (
    decoded.v !== CURSOR_VERSION
    || !Number.isSafeInteger(decoded.o)
    || decoded.o < 0
    || decoded.o > MAX_CURSOR_OFFSET
    || decoded.s !== pageSize
    || decoded.d !== cursorDeviceBinding(deviceId)
    || !macMatches
  ) {
    throw invalidRequest('cursor does not match this history query');
  }
  return decoded.o;
}

function parseHistoryRequest(event, deviceId, cursorSecret) {
  const input = event === undefined || event === null ? {} : event;
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw invalidRequest('history request must be an object');
  }
  const allowed = new Set(['page_size', 'cursor']);
  if (Object.keys(input).some((key) => !allowed.has(key))) {
    throw invalidRequest('history request contains unsupported fields');
  }
  const pageSize = input.page_size === undefined
    ? DEFAULT_HISTORY_PAGE_SIZE
    : input.page_size;
  if (
    !Number.isSafeInteger(pageSize)
    || pageSize < 1
    || pageSize > MAX_HISTORY_PAGE_SIZE
  ) {
    throw invalidRequest(`page_size must be an integer between 1 and ${MAX_HISTORY_PAGE_SIZE}`);
  }
  return {
    pageSize,
    offset: decodeCursor(input.cursor, pageSize, deviceId, cursorSecret),
  };
}

function requiredInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`${label} is invalid`);
  return value;
}

function nullableInteger(value, label) {
  if (value === undefined || value === null) return null;
  return requiredInteger(value, label);
}

function nullableText(value, label) {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'string' || value.trim() === '' || value.length > 128) {
    throw new TypeError(`${label} is invalid`);
  }
  return value;
}

function requiredTimestamp(utc, epochMs, label) {
  if (typeof utc !== 'string' || utc.length > 64 || Number.isNaN(Date.parse(utc))) {
    throw new TypeError(`${label} UTC timestamp is invalid`);
  }
  requiredInteger(epochMs, `${label} epoch timestamp`);
  if (Date.parse(utc) !== epochMs) {
    throw new TypeError(`${label} timestamps disagree`);
  }
  return utc;
}

function historyStatus(document, latestAgeMs) {
  if (typeof document.terminal !== 'boolean') throw new TypeError('run terminal is invalid');
  const incomplete = document.incomplete === undefined
    ? false
    : document.incomplete;
  if (typeof incomplete !== 'boolean') throw new TypeError('run incomplete marker is invalid');
  if (!document.terminal) {
    if (incomplete || document.result_outcome === 'aborted') {
      throw new TypeError('non-terminal run has terminal abort markers');
    }
    return latestAgeMs > STALE_AFTER_MS ? 'interrupted' : 'active';
  }
  const terminalSummaryFields = ['incomplete', 'result_outcome', 'result_validated'];
  if (!terminalSummaryFields.every((field) => Object.hasOwn(document, field))) {
    return 'unknown_terminal';
  }
  return incomplete || document.result_outcome === 'aborted' ? 'aborted' : 'completed';
}

function publicHistoryItem(document, expectedDeviceId, serverTimeEpochMs) {
  if (!document || typeof document !== 'object' || Array.isArray(document)) {
    throw new TypeError('run history document is invalid');
  }
  if (document.schema_version !== RUN_SCHEMA_VERSION) {
    throw new TypeError('run history schema is unsupported');
  }
  if (document.device_id !== expectedDeviceId) {
    throw new TypeError('run history device does not match configured device');
  }
  if (typeof document.run_id !== 'string' || !RUN_ID_PATTERN.test(document.run_id)) {
    throw new TypeError('run history run_id is invalid');
  }
  const firstUtc = requiredTimestamp(
    document.first_received_at_utc,
    document.first_received_at_epoch_ms,
    'first received',
  );
  const latestUtc = requiredTimestamp(
    document.latest_received_at_utc,
    document.latest_received_at_epoch_ms,
    'latest received',
  );
  if (document.latest_received_at_epoch_ms < document.first_received_at_epoch_ms) {
    throw new TypeError('run history timestamps are not monotonic');
  }
  requiredInteger(serverTimeEpochMs, 'server time');
  const latestAgeMs = Math.max(
    0,
    serverTimeEpochMs - document.latest_received_at_epoch_ms,
  );
  const progress = document.progress_percent === undefined
    ? null
    : document.progress_percent;
  if (
    progress !== null
    && (typeof progress !== 'number' || !Number.isFinite(progress) || progress < 0 || progress > 100)
  ) {
    throw new TypeError('run progress is invalid');
  }
  const completed = nullableInteger(
    document.completed_checkpoint_count,
    'completed checkpoint count',
  );
  const checkpointCount = nullableInteger(document.checkpoint_count, 'checkpoint count');
  if (completed !== null && checkpointCount !== null && completed > checkpointCount) {
    throw new TypeError('run checkpoint counts disagree');
  }
  const resultValidated = document.result_validated === undefined
    ? false
    : document.result_validated;
  if (typeof resultValidated !== 'boolean') {
    throw new TypeError('run result_validated is invalid');
  }
  const status = historyStatus(document, latestAgeMs);
  if (
    status === 'aborted'
    && (resultValidated || document.result_outcome !== 'aborted')
  ) {
    throw new TypeError('aborted run has inconsistent terminal result markers');
  }
  if (status !== 'completed' && resultValidated) {
    throw new TypeError('unclassified or non-terminal run cannot claim a validated terminal result');
  }
  return {
    run_id: document.run_id,
    status,
    terminal: document.terminal,
    first_received_at_utc: firstUtc,
    latest_received_at_utc: latestUtc,
    latest_age_ms: latestAgeMs,
    latest_sequence: requiredInteger(document.latest_sequence, 'latest sequence'),
    state: nullableText(document.state, 'run state'),
    outcome: nullableText(document.runtime_outcome, 'run outcome'),
    progress_percent: progress,
    completed_checkpoint_count: completed,
    checkpoint_count: checkpointCount,
    active_alarm_count: nullableInteger(document.active_alarm_count, 'active alarm count'),
    alarm_history_count: nullableInteger(document.alarm_history_count, 'alarm history count'),
    result_outcome: nullableText(document.result_outcome, 'result outcome'),
    result_validated: resultValidated,
  };
}

function createHistoryHandler(options) {
  const dependencies = options || {};
  const env = dependencies.env || process.env;
  const now = dependencies.now || Date.now;

  return async function getPatrolHistory(event, context) {
    let config;
    let identity;
    try {
      config = runtimeConfig(env);
      const getWxContext = dependencies.getWxContext || (() => {
        // eslint-disable-next-line global-require, import/no-extraneous-dependencies
        const cloud = require('wx-server-sdk');
        cloud.init({ env: config.cloudEnvId || cloud.DYNAMIC_CURRENT_ENV });
        return cloud.getWXContext();
      });
      identity = getWxContext(event, context) || {};
    } catch (error) {
      return {
        ok: false,
        error: 'HISTORY_UNAVAILABLE',
        message: 'Patrol history is temporarily unavailable',
      };
    }
    if (!identity.OPENID && !identity.openid) {
      return { ok: false, error: 'AUTH_REQUIRED', message: 'WeChat identity is required' };
    }

    try {
      const cursorSecret = requireUploadSecret(config.uploadSecret);
      const request = parseHistoryRequest(event, config.deviceId, cursorSecret);
      const repository = dependencies.repository || createDefaultRepository(config);
      const serverTimeEpochMs = now();
      requiredInteger(serverTimeEpochMs, 'server time');
      const page = await repository.getPatrolHistoryPage({
        deviceId: config.deviceId,
        pageSize: request.pageSize,
        offset: request.offset,
      });
      const items = page.documents.map((document) => publicHistoryItem(
        document,
        config.deviceId,
        serverTimeEpochMs,
      ));
      const nextCursor = page.hasMore
        ? encodeCursor(page.nextOffset, request.pageSize, config.deviceId, cursorSecret)
        : null;
      return {
        ok: true,
        schema_version: HISTORY_SCHEMA_VERSION,
        device_id: config.deviceId,
        page_size: request.pageSize,
        server_time_epoch_ms: serverTimeEpochMs,
        stale_after_ms: STALE_AFTER_MS,
        items,
        has_more: page.hasMore,
        next_cursor: nextCursor,
      };
    } catch (error) {
      if (error instanceof HistoryRequestError) {
        return { ok: false, error: 'INVALID_HISTORY_REQUEST', message: error.message };
      }
      return {
        ok: false,
        error: 'HISTORY_UNAVAILABLE',
        message: 'Patrol history is temporarily unavailable',
      };
    }
  };
}

const main = createHistoryHandler();

module.exports = {
  CURSOR_VERSION,
  HistoryRequestError,
  createHistoryHandler,
  decodeCursor,
  encodeCursor,
  main,
  parseHistoryRequest,
  publicHistoryItem,
};
