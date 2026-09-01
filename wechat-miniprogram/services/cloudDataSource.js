'use strict';

const {
  PayloadProtocolError,
  createPayloadGate,
  validateDashboardPayload,
} = require('./protocol');

const CLOUD_SNAPSHOT_SCHEMA_VERSION = 'patrol-cloud-snapshot/v1';
const DEFAULT_COLLECTION = 'patrol_live';
const DEFAULT_DOCUMENT_ID = 'main';
const DEFAULT_STALE_AFTER_MS = 8000;
const DEFAULT_POLL_INTERVAL_MS = 2000;
const DEFAULT_TERMINAL_POLL_INTERVAL_MS = 15000;
const DEFAULT_MAX_POLL_BACKOFF_MS = 30000;
const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

function defaultScheduler() {
  return { setTimeout, clearTimeout };
}

function requireString(value, label, maximum) {
  if (typeof value !== 'string' || value.trim() === '' || value.length > (maximum || 256)) {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', `${label} 必须是有效字符串`);
  }
  return value;
}

function validateCloudSnapshot(input) {
  if (input === null || typeof input !== 'object' || Array.isArray(input)) {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', '云端快照必须是对象');
  }
  if (input._id !== DEFAULT_DOCUMENT_ID) {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', '云端快照文档必须为 patrol_live/main');
  }
  if (input.schema_version !== CLOUD_SNAPSHOT_SCHEMA_VERSION) {
    throw new PayloadProtocolError(
      'CLOUD_SCHEMA_MISMATCH',
      `不支持的云端快照协议：${String(input.schema_version)}`,
    );
  }
  requireString(input.device_id, 'device_id', 64);
  requireString(input.run_id, 'run_id', 128);
  if (!RUN_ID_PATTERN.test(input.run_id)) {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', 'run_id 格式无效');
  }
  if (!Number.isSafeInteger(input.sequence) || input.sequence < 0) {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', 'sequence 必须是非负安全整数');
  }
  if (typeof input.terminal !== 'boolean') {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', 'terminal 必须是布尔值');
  }
  if (typeof input.body_sha256 !== 'string' || !/^[a-f0-9]{64}$/i.test(input.body_sha256)) {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', 'body_sha256 必须是 SHA-256 摘要');
  }
  requireString(input.received_at_utc, 'received_at_utc', 64);
  const receivedAtUtcMs = Date.parse(input.received_at_utc);
  if (Number.isNaN(receivedAtUtcMs)) {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', 'received_at_utc 必须是有效时间');
  }
  if (!Number.isSafeInteger(input.received_at_epoch_ms) || input.received_at_epoch_ms < 0) {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', 'received_at_epoch_ms 必须是非负安全整数');
  }
  if (Math.abs(receivedAtUtcMs - input.received_at_epoch_ms) > 1000) {
    throw new PayloadProtocolError(
      'INVALID_CLOUD_SNAPSHOT',
      '云端快照的 UTC 时间与毫秒时间戳不一致',
    );
  }
  const payload = validateDashboardPayload(input.payload);
  if (
    input.run_id !== payload.run.run_id
    || input.sequence !== payload.run.sequence
    || input.terminal !== payload.run.terminal
  ) {
    throw new PayloadProtocolError(
      'CLOUD_METADATA_MISMATCH',
      '云端索引字段与巡检 payload 不一致',
    );
  }
  return { snapshot: input, payload };
}

function pickSnapshot(change) {
  if (change && Array.isArray(change.docs)) {
    return change.docs.find((document) => document && document._id === DEFAULT_DOCUMENT_ID) || null;
  }
  if (change && change.data && typeof change.data === 'object') return change.data;
  if (change && change._id) return change;
  return null;
}

function validatePollingFreshness(result, snapshot) {
  if (!Number.isSafeInteger(result.server_time_epoch_ms) || result.server_time_epoch_ms < 0) {
    throw new PayloadProtocolError(
      'INVALID_CLOUD_SNAPSHOT',
      'server_time_epoch_ms 必须是非负安全整数',
    );
  }
  if (!Number.isSafeInteger(result.age_ms) || result.age_ms < 0) {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', 'age_ms 必须是非负安全整数');
  }
  if (typeof result.stale !== 'boolean') {
    throw new PayloadProtocolError('INVALID_CLOUD_SNAPSHOT', 'stale 必须是布尔值');
  }
  const expectedAgeMs = Math.max(
    0,
    result.server_time_epoch_ms - snapshot.received_at_epoch_ms,
  );
  if (result.age_ms !== expectedAgeMs) {
    throw new PayloadProtocolError(
      'INVALID_CLOUD_SNAPSHOT',
      '云端快照的数据年龄与服务器时间不一致',
    );
  }
  if (result.stale !== (result.age_ms > DEFAULT_STALE_AFTER_MS)) {
    throw new PayloadProtocolError(
      'INVALID_CLOUD_SNAPSHOT',
      '云端快照的陈旧标记与数据年龄不一致',
    );
  }
  return {
    serverTimeEpochMs: result.server_time_epoch_ms,
    ageMs: result.age_ms,
    stale: result.stale,
  };
}

function createCloudDataSource(options) {
  const config = options || {};
  const wxApi = config.wxApi || (typeof wx === 'undefined' ? null : wx);
  if (!wxApi || !wxApi.cloud || typeof wxApi.cloud.database !== 'function') {
    throw new TypeError('需要提供已启用云开发的微信 API');
  }
  const env = requireString(config.env, 'env', 128);
  const collectionName = config.collection || DEFAULT_COLLECTION;
  const documentId = config.documentId || DEFAULT_DOCUMENT_ID;
  if (collectionName !== DEFAULT_COLLECTION || documentId !== DEFAULT_DOCUMENT_ID) {
    throw new RangeError('当前客户端只允许读取 patrol_live/main');
  }
  const staleAfterMs = config.staleAfterMs ?? DEFAULT_STALE_AFTER_MS;
  const pollIntervalMs = config.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const terminalPollIntervalMs = config.terminalPollIntervalMs
    ?? DEFAULT_TERMINAL_POLL_INTERVAL_MS;
  const maxPollBackoffMs = config.maxPollBackoffMs ?? DEFAULT_MAX_POLL_BACKOFF_MS;
  if (!Number.isFinite(staleAfterMs) || staleAfterMs < 5000 || staleAfterMs > 10000) {
    throw new RangeError('staleAfterMs 必须位于 5000 至 10000 ms');
  }
  if (!Number.isFinite(pollIntervalMs) || pollIntervalMs < 500) {
    throw new RangeError('pollIntervalMs 必须至少为 500 ms');
  }
  if (!Number.isFinite(terminalPollIntervalMs) || terminalPollIntervalMs < pollIntervalMs) {
    throw new RangeError('terminalPollIntervalMs 不能短于普通轮询间隔');
  }
  if (!Number.isFinite(maxPollBackoffMs) || maxPollBackoffMs < pollIntervalMs) {
    throw new RangeError('maxPollBackoffMs 不能短于普通轮询间隔');
  }

  const scheduler = config.scheduler || defaultScheduler();
  const clock = config.clock || Date.now;
  const gate = createPayloadGate({ expectedRunId: config.expectedRunId || null });
  const database = config.database || wxApi.cloud.database({ env });
  const document = database.collection(collectionName).doc(documentId);
  const callFunction = config.callFunction
    || (typeof wxApi.cloud.callFunction === 'function'
      ? wxApi.cloud.callFunction.bind(wxApi.cloud)
      : null);
  if (typeof callFunction !== 'function') {
    throw new TypeError('当前微信基础库不支持云函数轮询兜底');
  }
  let callbacks = {};
  let status = 'idle';
  let transport = null;
  let observer = null;
  let pollTimer = null;
  let staleTimer = null;
  let generation = 0;
  let lastFreshSequence = -1;
  let lastFreshAt = null;
  let pollingFailures = 0;

  function announce(nextStatus, details) {
    status = nextStatus;
    if (typeof callbacks.onStatus === 'function') {
      callbacks.onStatus({ status: nextStatus, transport, ...(details || {}) });
    }
  }

  function reportError(error, details) {
    if (typeof callbacks.onError === 'function') callbacks.onError(error, details || {});
  }

  function clearTimer(name) {
    if (name === 'poll' && pollTimer !== null) {
      scheduler.clearTimeout(pollTimer);
      pollTimer = null;
    }
    if (name === 'stale' && staleTimer !== null) {
      scheduler.clearTimeout(staleTimer);
      staleTimer = null;
    }
  }

  function closeObserver() {
    const activeObserver = observer;
    observer = null;
    if (activeObserver && typeof activeObserver.close === 'function') {
      try {
        const closing = activeObserver.close();
        if (closing && typeof closing.catch === 'function') closing.catch(() => {});
      } catch (error) {
        // Closing an already-disconnected SDK observer is safe to ignore.
      }
    }
  }

  function armStaleTimer(sequence, delayMs) {
    clearTimer('stale');
    if (status === 'terminal' || status === 'stopped' || status === 'error') return;
    staleTimer = scheduler.setTimeout(() => {
      staleTimer = null;
      if (status === 'terminal' || status === 'stopped' || status === 'error') return;
      if (lastFreshSequence !== sequence) return;
      announce('stale', {
        sequence,
        staleAfterMs,
        lastFreshAt,
      });
    }, delayMs === undefined ? staleAfterMs : delayMs);
  }

  function closeTransport() {
    clearTimer('poll');
    clearTimer('stale');
    closeObserver();
    transport = null;
  }

  function failClosed(error) {
    closeTransport();
    reportError(error, { retrying: false, protocol: true });
    announce('error', { code: error.code || 'PROTOCOL_ERROR' });
  }

  function acceptDocument(input, activeTransport, pollingFreshness) {
    const { snapshot, payload } = validateCloudSnapshot(input);
    pollingFailures = 0;
    const gateState = gate.getState();
    if (gateState.terminal && gateState.runId !== payload.run.run_id) {
      gate.reset(payload.run.run_id);
      lastFreshSequence = -1;
      lastFreshAt = null;
      announce('new_run', { previousRunId: gateState.runId, runId: payload.run.run_id });
    }
    const decision = gate.accept(payload);
    if (!decision.accepted) return decision;
    lastFreshSequence = payload.run.sequence;
    lastFreshAt = snapshot.received_at_epoch_ms;
    if (typeof callbacks.onData === 'function') {
      callbacks.onData(payload, {
        transport: activeTransport,
        deviceId: snapshot.device_id,
        sequence: snapshot.sequence,
        receivedAtUtc: snapshot.received_at_utc,
        receivedAtEpochMs: snapshot.received_at_epoch_ms,
        bodySha256: snapshot.body_sha256,
        ...(pollingFreshness || {}),
      });
    }
    if (payload.run.terminal) {
      announce('terminal', { runId: payload.run.run_id, sequence: payload.run.sequence });
      return decision;
    }
    const ageMs = pollingFreshness
      ? pollingFreshness.ageMs
      : Math.max(0, clock() - snapshot.received_at_epoch_ms);
    const isStale = pollingFreshness
      ? pollingFreshness.stale
      : ageMs >= staleAfterMs;
    if (isStale) {
      announce('stale', {
        sequence: payload.run.sequence,
        staleAfterMs,
        lastFreshAt,
        ageMs,
        ...(pollingFreshness
          ? { serverTimeEpochMs: pollingFreshness.serverTimeEpochMs }
          : {}),
      });
    } else {
      announce('running', {
        sequence: payload.run.sequence,
        receivedAtUtc: snapshot.received_at_utc,
      });
      armStaleTimer(payload.run.sequence, staleAfterMs - ageMs);
    }
    return decision;
  }

  function schedulePoll(token, delayMs) {
    if (token !== generation || status === 'stopped' || status === 'error') {
      return;
    }
    clearTimer('poll');
    pollTimer = scheduler.setTimeout(() => {
      pollTimer = null;
      poll(token);
    }, delayMs);
  }

  function poll(token) {
    if (token !== generation || status === 'stopped' || status === 'error') {
      return;
    }
    transport = 'polling';
    let request;
    try {
      request = callFunction({ name: 'getPatrolSnapshot' });
    } catch (error) {
      pollingFailures += 1;
      const retryDelayMs = Math.min(
        maxPollBackoffMs,
        pollIntervalMs * (2 ** Math.min(pollingFailures - 1, 16)),
      );
      announce('polling', { failures: pollingFailures, retryDelayMs });
      reportError(error, { retrying: true, transport: 'polling', failures: pollingFailures });
      schedulePoll(token, retryDelayMs);
      return;
    }
    Promise.resolve(request).then((response) => {
      if (token !== generation || status === 'stopped') return;
      const result = response && response.result;
      const snapshot = result && result.found === true ? result.snapshot : null;
      if (!snapshot) throw new Error('云端尚未收到巡检快照');
      const pollingFreshness = validatePollingFreshness(result, snapshot);
      acceptDocument(snapshot, 'polling', pollingFreshness);
      if (status !== 'error') {
        schedulePoll(
          token,
          status === 'terminal' ? terminalPollIntervalMs : pollIntervalMs,
        );
      }
    }).catch((error) => {
      if (token !== generation || status === 'stopped') return;
      if (error instanceof PayloadProtocolError) {
        failClosed(error);
        return;
      }
      pollingFailures += 1;
      const retryDelayMs = Math.min(
        maxPollBackoffMs,
        pollIntervalMs * (2 ** Math.min(pollingFailures - 1, 16)),
      );
      announce('polling', { failures: pollingFailures, retryDelayMs });
      reportError(error, { retrying: true, transport: 'polling', failures: pollingFailures });
      schedulePoll(token, retryDelayMs);
    });
  }

  function enterPolling(error, token) {
    if (token !== generation || status === 'stopped' || status === 'error') {
      return;
    }
    closeObserver();
    transport = 'polling';
    announce('polling', { fallback: true });
    if (error) reportError(error, { retrying: true, transport: 'polling', fallback: true });
    poll(token);
  }

  function startWatch(token) {
    transport = 'watch';
    announce('connecting', { env, collection: collectionName, documentId });
    try {
      announce('watching', { env, collection: collectionName, documentId });
      const nextObserver = document.watch({
        onChange(change) {
          if (token !== generation || status === 'stopped') return;
          const snapshot = pickSnapshot(change);
          if (!snapshot) {
            enterPolling(new Error('云端尚未收到巡检快照'), token);
            return;
          }
          try {
            acceptDocument(snapshot, 'watch');
          } catch (error) {
            if (error instanceof PayloadProtocolError) failClosed(error);
            else enterPolling(error, token);
          }
        },
        onError(error) {
          // The SDK has already closed a watcher before invoking onError. Calling
          // close() again can trigger its CLOSED -> rebuildWatchFail state error.
          observer = null;
          enterPolling(
            new Error(error && error.errMsg ? error.errMsg : '云端实时订阅中断'),
            token,
          );
        },
      });
      if (!nextObserver || typeof nextObserver.close !== 'function') {
        throw new Error('当前云数据库不支持实时订阅');
      }
      observer = nextObserver;
    } catch (error) {
      enterPolling(error, token);
    }
  }

  return {
    start(nextCallbacks) {
      if (['connecting', 'watching', 'running', 'polling', 'stale', 'terminal'].includes(status)
          && transport !== null) return;
      if (status === 'stopped') gate.reset(config.expectedRunId || null);
      callbacks = nextCallbacks || {};
      generation += 1;
      const token = generation;
      startWatch(token);
    },

    stop() {
      generation += 1;
      closeTransport();
      announce('stopped');
    },

    close() {
      this.stop();
    },

    reset(nextExpectedRunId) {
      this.stop();
      callbacks = {};
      gate.reset(nextExpectedRunId === undefined ? config.expectedRunId : nextExpectedRunId);
      lastFreshSequence = -1;
      lastFreshAt = null;
      pollingFailures = 0;
      status = 'idle';
    },

    getState() {
      return {
        status,
        transport,
        env,
        collection: collectionName,
        documentId,
        staleAfterMs,
        terminalPollIntervalMs,
        maxPollBackoffMs,
        lastFreshSequence,
        lastFreshAt,
        pollingFailures,
        ...gate.getState(),
      };
    },
  };
}

module.exports = {
  CLOUD_SNAPSHOT_SCHEMA_VERSION,
  DEFAULT_COLLECTION,
  DEFAULT_DOCUMENT_ID,
  DEFAULT_STALE_AFTER_MS,
  DEFAULT_TERMINAL_POLL_INTERVAL_MS,
  DEFAULT_MAX_POLL_BACKOFF_MS,
  createCloudDataSource,
  pickSnapshot,
  validatePollingFreshness,
  validateCloudSnapshot,
};
