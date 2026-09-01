'use strict';

const { PayloadProtocolError, createPayloadGate } = require('./protocol');

function defaultScheduler() {
  return { setTimeout, clearTimeout };
}

function normalizeBackoff(input) {
  const source = input || {};
  const result = {
    initialMs: source.initialMs ?? 1000,
    maxMs: source.maxMs ?? 15000,
    multiplier: source.multiplier ?? 2,
    jitterRatio: source.jitterRatio ?? 0.15,
  };
  if (
    !Number.isFinite(result.initialMs) || result.initialMs < 0
    || !Number.isFinite(result.maxMs) || result.maxMs < result.initialMs
    || !Number.isFinite(result.multiplier) || result.multiplier < 1
    || !Number.isFinite(result.jitterRatio) || result.jitterRatio < 0 || result.jitterRatio > 1
  ) {
    throw new RangeError('backoff 参数无效');
  }
  return result;
}

function createPollingClient(options) {
  const config = options || {};
  const wxApi = config.wxApi || (typeof wx === 'undefined' ? null : wx);
  if (!wxApi || typeof wxApi.request !== 'function') {
    throw new TypeError('需要提供支持 request 的微信 API');
  }
  const endpoint = config.endpoint || 'http://127.0.0.1:8765/api/dashboard';
  if (typeof endpoint !== 'string' || endpoint.trim() === '') {
    throw new TypeError('endpoint 必须是非空字符串');
  }
  const intervalMs = config.intervalMs ?? 1000;
  const timeoutMs = config.timeoutMs ?? 4000;
  if (!Number.isFinite(intervalMs) || intervalMs < 0) throw new RangeError('intervalMs 无效');
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new RangeError('timeoutMs 无效');
  const backoff = normalizeBackoff(config.backoff);
  const scheduler = config.scheduler || defaultScheduler();
  const random = config.random || Math.random;
  const gate = createPayloadGate({ expectedRunId: config.expectedRunId || null });
  let status = 'idle';
  let timer = null;
  let watchdog = null;
  let activeRequest = null;
  let callbacks = {};
  let failures = 0;

  function announce(nextStatus, details) {
    status = nextStatus;
    if (typeof callbacks.onStatus === 'function') {
      callbacks.onStatus({ status: nextStatus, ...(details || {}) });
    }
  }

  function clearTimer(refName) {
    if (refName === 'poll' && timer !== null) {
      scheduler.clearTimeout(timer);
      timer = null;
    }
    if (refName === 'watchdog' && watchdog !== null) {
      scheduler.clearTimeout(watchdog);
      watchdog = null;
    }
  }

  function schedule(delayMs) {
    if (status !== 'running' && status !== 'backoff') return;
    clearTimer('poll');
    timer = scheduler.setTimeout(() => {
      timer = null;
      poll();
    }, delayMs);
  }

  function retryDelay() {
    const base = Math.min(
      backoff.maxMs,
      backoff.initialMs * (backoff.multiplier ** Math.max(0, failures - 1)),
    );
    const jitter = base * backoff.jitterRatio * ((random() * 2) - 1);
    return Math.max(0, Math.round(base + jitter));
  }

  function reportError(error, retrying) {
    if (typeof callbacks.onError === 'function') {
      callbacks.onError(error, { retrying, failures });
    }
  }

  function handleTransient(error) {
    if (status === 'stopped' || status === 'terminal') return;
    failures += 1;
    const delayMs = retryDelay();
    announce('backoff', { failures, delayMs });
    reportError(error, true);
    schedule(delayMs);
  }

  function failClosed(error) {
    clearTimer('poll');
    clearTimer('watchdog');
    reportError(error, false);
    announce('error', { code: error.code || 'PROTOCOL_ERROR' });
  }

  function poll() {
    if (status !== 'running' && status !== 'backoff') return;
    announce('requesting', { failures });
    let settled = false;

    function settle() {
      if (settled) return false;
      settled = true;
      clearTimer('watchdog');
      activeRequest = null;
      return true;
    }

    watchdog = scheduler.setTimeout(() => {
      const requestToAbort = activeRequest;
      if (!settle() || status === 'stopped') return;
      if (requestToAbort && typeof requestToAbort.abort === 'function') requestToAbort.abort();
      handleTransient(new Error(`巡检请求超过 ${timeoutMs} ms`));
    }, timeoutMs);

    const requestTask = wxApi.request({
      url: endpoint,
      method: 'GET',
      timeout: timeoutMs,
      header: { Accept: 'application/json' },
      success(response) {
        if (!settle() || status === 'stopped') return;
        if (!response || response.statusCode < 200 || response.statusCode >= 300) {
          const code = response && response.statusCode;
          handleTransient(new Error(`巡检接口返回 HTTP ${String(code)}`));
          return;
        }
        try {
          const decision = gate.accept(response.data);
          failures = 0;
          if (decision.accepted && typeof callbacks.onData === 'function') {
            callbacks.onData(decision.payload);
          }
          if (decision.payload.run.terminal || decision.reason === 'terminal') {
            announce('terminal', {
              runId: decision.payload.run.run_id,
              sequence: decision.payload.run.sequence,
            });
            return;
          }
          announce('running', { duplicate: decision.reason === 'duplicate' });
          schedule(intervalMs);
        } catch (error) {
          if (error instanceof PayloadProtocolError) failClosed(error);
          else handleTransient(error);
        }
      },
      fail(error) {
        if (!settle() || status === 'stopped') return;
        handleTransient(new Error(error && error.errMsg ? error.errMsg : '巡检请求失败'));
      },
    });
    activeRequest = settled ? null : requestTask;
  }

  return {
    start(nextCallbacks) {
      if (status === 'running' || status === 'requesting' || status === 'backoff') return;
      if (status === 'terminal') return;
      callbacks = nextCallbacks || {};
      announce('running', { mode: 'local_polling' });
      poll();
    },

    stop() {
      clearTimer('poll');
      clearTimer('watchdog');
      const wasTerminal = status === 'terminal';
      if (!wasTerminal) announce('stopped');
      if (activeRequest && typeof activeRequest.abort === 'function') activeRequest.abort();
      activeRequest = null;
    },

    reset(nextExpectedRunId) {
      this.stop();
      failures = 0;
      callbacks = {};
      gate.reset(nextExpectedRunId === undefined ? config.expectedRunId : nextExpectedRunId);
      status = 'idle';
    },

    getState() {
      return { status, failures, endpoint, ...gate.getState() };
    },
  };
}

module.exports = { createPollingClient, normalizeBackoff };
