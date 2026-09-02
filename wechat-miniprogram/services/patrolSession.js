'use strict';

function requireMode(value) {
  if (!['cloud', 'demo', 'local'].includes(value)) {
    throw new RangeError(`未知数据源模式：${String(value)}`);
  }
  return value;
}

function createPatrolSession(options) {
  const config = options || {};
  if (typeof config.createSource !== 'function') {
    throw new TypeError('createSource 必须是函数');
  }

  let mode = requireMode(config.initialMode || 'cloud');
  let source = null;
  let started = false;
  let closed = false;
  let generation = 0;
  let lastPayload = null;
  let lastMeta = null;
  let lastStatus = { status: 'idle', mode, transport: null };
  let lastError = null;
  let lastErrorDetails = null;
  const subscribers = new Set();

  function invoke(subscriber, method, args) {
    if (typeof subscriber[method] !== 'function') return;
    try {
      subscriber[method](...args);
    } catch (error) {
      // A rendering failure in one page must never tear down the shared transport
      // or prevent the remaining pages from receiving the same verified update.
    }
  }

  function notify(method, ...args) {
    for (const subscriber of Array.from(subscribers)) {
      invoke(subscriber, method, args);
    }
  }

  function publishStatus(event) {
    lastStatus = { ...(event || {}), status: (event && event.status) || 'idle', mode };
    notify('onStatus', lastStatus);
  }

  function publishError(error, details) {
    lastError = error;
    lastErrorDetails = { ...(details || {}), mode };
    notify('onError', error, lastErrorDetails);
  }

  function callbacksFor(token) {
    return {
      onData(payload, meta) {
        if (token !== generation || !started || closed) return;
        lastPayload = payload;
        lastMeta = { ...(meta || {}), mode };
        lastError = null;
        lastErrorDetails = null;
        notify('onData', payload, lastMeta);
      },
      onStatus(event) {
        if (token !== generation || !started || closed) return;
        publishStatus(event);
      },
      onError(error, details) {
        if (token !== generation || !started || closed) return;
        publishError(error, details);
      },
    };
  }

  function ensureSource() {
    if (source) return source;
    source = config.createSource(mode);
    if (!source || typeof source.start !== 'function' || typeof source.stop !== 'function') {
      source = null;
      throw new TypeError('createSource 必须返回可启动和停止的数据源');
    }
    return source;
  }

  function stopSource() {
    generation += 1;
    const activeSource = source;
    started = false;
    if (activeSource) activeSource.stop();
  }

  function discardSource() {
    const activeSource = source;
    source = null;
    if (activeSource && typeof activeSource.close === 'function') activeSource.close();
  }

  function clearSnapshot(reason) {
    lastPayload = null;
    lastMeta = null;
    lastError = null;
    lastErrorDetails = null;
    lastStatus = { status: 'idle', mode, transport: null };
    notify('onClear', { mode, reason: reason || 'clear' });
  }

  function callSource(method, ...args) {
    const activeSource = ensureSource();
    if (typeof activeSource[method] !== 'function') return null;
    return activeSource[method](...args);
  }

  const session = {
    subscribe(callbacks) {
      if (closed) throw new Error('巡检会话已关闭');
      const subscriber = callbacks || {};
      subscribers.add(subscriber);
      if (lastPayload !== null) invoke(subscriber, 'onData', [lastPayload, lastMeta]);
      invoke(subscriber, 'onStatus', [lastStatus]);
      if (lastError) invoke(subscriber, 'onError', [lastError, lastErrorDetails]);
      let subscribed = true;
      return () => {
        if (!subscribed) return;
        subscribed = false;
        subscribers.delete(subscriber);
      };
    },

    start() {
      if (closed) throw new Error('巡检会话已关闭');
      if (started) return;
      started = true;
      generation += 1;
      const token = generation;
      try {
        ensureSource().start(callbacksFor(token));
      } catch (error) {
        started = false;
        publishError(error, { retrying: false, initialization: true });
        publishStatus({ status: 'error', code: 'SOURCE_INITIALIZATION_FAILED' });
      }
    },

    stop() {
      if (closed || !started) return;
      const terminalEvent = lastStatus.status === 'terminal' ? lastStatus : null;
      stopSource();
      if (terminalEvent) {
        lastStatus = terminalEvent;
        notify('onStatus', terminalEvent);
      } else {
        publishStatus({ status: 'stopped' });
      }
    },

    setMode(nextMode) {
      if (closed) throw new Error('巡检会话已关闭');
      const safeMode = requireMode(nextMode);
      if (safeMode === mode) return false;
      const shouldRestart = started;
      stopSource();
      discardSource();
      mode = safeMode;
      clearSnapshot('mode_changed');
      publishStatus({ status: 'idle', reason: 'mode_changed' });
      if (shouldRestart) session.start();
      return true;
    },

    reset(optionsInput) {
      if (closed) throw new Error('巡检会话已关闭');
      const resetOptions = optionsInput || {};
      const shouldRestart = resetOptions.restart === undefined ? started : resetOptions.restart === true;
      stopSource();
      if (resetOptions.recreate === true) discardSource();
      else if (source && typeof source.reset === 'function') source.reset();
      else discardSource();
      clearSnapshot('reset');
      publishStatus({ status: 'idle', reason: 'reset' });
      if (shouldRestart) session.start();
    },

    jumpToNextKeyEvent() {
      if (closed || mode !== 'demo') return null;
      return callSource('jumpToNextKeyEvent');
    },

    setPlaybackRate(rate) {
      if (closed || mode !== 'demo') return null;
      return callSource('setPlaybackRate', rate);
    },

    close() {
      if (closed) return;
      stopSource();
      discardSource();
      closed = true;
      subscribers.clear();
    },

    getState() {
      return {
        mode,
        started,
        closed,
        subscriberCount: subscribers.size,
        payload: lastPayload,
        meta: lastMeta,
        status: lastStatus.status,
        statusEvent: lastStatus,
        error: lastError,
        errorDetails: lastErrorDetails,
        sourceState: source && typeof source.getState === 'function' ? source.getState() : null,
      };
    },
  };

  return session;
}

module.exports = { createPatrolSession };
