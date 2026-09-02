'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const fixture = require('../data/graduation-demo-seed7000-v1.js');
const { buildDashboardPayload, identifyReplayKeyEvents } = require('../services/offlineReplay');

function instantiatePage(definition) {
  return {
    ...definition,
    data: structuredClone(definition.data),
    setData(update) {
      this._setDataCount = (this._setDataCount || 0) + 1;
      this.data = { ...this.data, ...update };
    },
  };
}

function loadPage(modulePath, app, wxApi) {
  let definition;
  global.getApp = () => app;
  global.wx = wxApi || {};
  global.Page = (value) => { definition = value; };
  delete require.cache[require.resolve(modulePath)];
  require(modulePath);
  return instantiatePage(definition);
}

function createAppHarness(mode) {
  const subscribers = [];
  let unsubscribeCount = 0;
  let jumpCount = 0;
  const playbackRates = [];
  let sessionState = {
    mode: mode || 'cloud',
    status: 'running',
    statusEvent: { mode: mode || 'cloud', status: 'running' },
    started: true,
    payload: null,
    meta: null,
    error: null,
  };
  const app = {
    globalData: {
      mode: mode || 'cloud',
      cloudEnvId: 'cloud1-5gpkygv2be4de05a',
      dashboard: null,
      dashboardMeta: { mode: mode || 'cloud', status: 'running' },
    },
    subscribePatrol(callbacks) {
      subscribers.push(callbacks);
      return () => { unsubscribeCount += 1; };
    },
    getPatrolSessionState() { return sessionState; },
    setMode(nextMode) { this.globalData.mode = nextMode; },
    setDashboard(payload, meta) {
      this.globalData.dashboard = payload;
      this.globalData.dashboardMeta = meta;
    },
    resetPatrolSession() {},
    pausePatrolSession() {},
    resumePatrolSession() {},
    jumpToNextDemoKeyEvent() { jumpCount += 1; return { run: { sequence: 98 } }; },
    setDemoPlaybackRate(rate) { playbackRates.push(rate); return rate; },
  };
  return {
    app,
    subscribers,
    setSessionState(update) { sessionState = { ...sessionState, ...update }; },
    get unsubscribeCount() { return unsubscribeCount; },
    get jumpCount() { return jumpCount; },
    get playbackRates() { return playbackRates; },
  };
}

test('dashboard subscribes once and page unload does not stop the shared session', () => {
  const harness = createAppHarness('cloud');
  const page = loadPage('../pages/dashboard/index.js', harness.app, {
    getDeviceInfo: () => ({ platform: 'ios' }),
  });
  page.onLoad();
  page.onShow();
  assert.equal(harness.subscribers.length, 1);
  assert.equal(typeof page.onHide, 'function');

  const payload = buildDashboardPayload(fixture, 1);
  harness.subscribers[0].onData(payload, {
    mode: 'cloud',
    receivedAtUtc: '2026-09-01T06:00:00.000Z',
  });
  assert.equal(page.data.sequenceText, `SEQ ${payload.run.sequence}`);
  assert.equal(page.data.mode, 'cloud');
  assert.notEqual(page.data.avoidanceStatus, '等待避障遥测');
  assert.match(page.data.executedActionText, /m\/s/);

  page.onHide();
  const hiddenSequence = page.data.sequenceText;
  const hiddenSetDataCount = page._setDataCount;
  const newer = buildDashboardPayload(fixture, 2);
  harness.subscribers[0].onData(newer, { mode: 'cloud' });
  assert.equal(page.data.sequenceText, hiddenSequence);
  assert.equal(page._setDataCount, hiddenSetDataCount);
  harness.app.globalData.dashboard = newer;
  harness.app.globalData.dashboardMeta = { mode: 'cloud', status: 'running' };
  harness.setSessionState({
    payload: newer,
    meta: { mode: 'cloud' },
    status: 'running',
    statusEvent: { mode: 'cloud', status: 'running' },
  });
  page.onShow();
  assert.equal(page.data.sequenceText, `SEQ ${newer.run.sequence}`);

  page.onUnload();
  assert.equal(harness.unsubscribeCount, 1);
});

test('dashboard exposes accessible demo key-event and playback controls without affecting cloud', () => {
  const keyEvent = identifyReplayKeyEvents(fixture)[0];
  const demoHarness = createAppHarness('demo');
  const toasts = [];
  const demoPage = loadPage('../pages/dashboard/index.js', demoHarness.app, {
    getDeviceInfo: () => ({ platform: 'devtools' }),
    showToast: (options) => toasts.push(options.title),
  });
  demoPage.onLoad();
  demoPage.onShow();
  demoHarness.subscribers[0].onData(buildDashboardPayload(fixture, keyEvent.frameIndex), {
    mode: 'demo',
    replay: {
      playbackRate: 1,
      keyEvent,
      keyEventCount: 7,
      keyEventHoldMs: 1800,
    },
  });
  assert.equal(demoPage.data.demoKeyEventActive, true);
  assert.equal(demoPage.data.demoKeyEventLabel, '脱困保护介入');
  assert.match(demoPage.data.demoKeyEventPositionText, /1 \/ 7/);

  demoPage.setDemoPlaybackRate({ currentTarget: { dataset: { rate: '0.5' } } });
  assert.deepEqual(demoHarness.playbackRates, [0.5]);
  assert.equal(demoPage.data.demoHalfSpeedActive, true);
  demoPage.jumpToNextDemoEvent();
  assert.equal(demoHarness.jumpCount, 1);
  assert.deepEqual(toasts, []);

  const cloudHarness = createAppHarness('cloud');
  const cloudPage = loadPage('../pages/dashboard/index.js', cloudHarness.app, {
    getDeviceInfo: () => ({ platform: 'ios' }),
  });
  cloudPage.onLoad();
  cloudPage.onShow();
  cloudPage.setDemoPlaybackRate({ currentTarget: { dataset: { rate: '0.5' } } });
  cloudPage.jumpToNextDemoEvent();
  assert.deepEqual(cloudHarness.playbackRates, []);
  assert.equal(cloudHarness.jumpCount, 0);
});

test('visible pages atomically clear old demo values when cloud has no snapshot', () => {
  const payload = buildDashboardPayload(fixture, fixture.frames.length - 1);

  const dashboardHarness = createAppHarness('demo');
  const dashboard = loadPage('../pages/dashboard/index.js', dashboardHarness.app, {
    getDeviceInfo: () => ({ platform: 'ios' }),
  });
  dashboard.onLoad();
  dashboard.onShow();
  dashboardHarness.subscribers[0].onData(payload, { mode: 'demo' });
  assert.notEqual(dashboard.data.dashboardMap, null);
  dashboardHarness.subscribers[0].onClear({ mode: 'cloud', reason: 'mode_changed' });
  assert.equal(dashboard.data.dashboardMap, null);
  assert.equal(dashboard.data.runId, '');
  assert.equal(dashboard.data.activeAlarmCount, 0);
  assert.equal(dashboard.data.resultAvailable, false);

  const alarmHarness = createAppHarness('demo');
  const alarms = loadPage('../pages/alarms/index.js', alarmHarness.app);
  alarms.onLoad();
  alarms.onShow();
  alarmHarness.subscribers[0].onData(payload, { mode: 'demo' });
  assert.ok(alarms.data.historyCount > 0);
  alarmHarness.subscribers[0].onClear({ mode: 'cloud', reason: 'mode_changed' });
  assert.equal(alarms.data.historyCount, 0);
  assert.equal(alarms.data.sourceLabel, '正在准备云端异常记录');

  const resultHarness = createAppHarness('demo');
  const result = loadPage('../pages/result/index.js', resultHarness.app);
  result.onLoad();
  result.onShow();
  resultHarness.subscribers[0].onData(payload, { mode: 'demo' });
  assert.equal(result.data.resultTone, 'success');
  resultHarness.subscribers[0].onClear({ mode: 'cloud', reason: 'mode_changed' });
  assert.equal(result.data.resultTone, 'waiting');
  assert.equal(result.data.dashboardMap, null);
});

test('cloud pages distinguish healthy waiting, degraded polling, and fail-closed errors', () => {
  const dashboardHarness = createAppHarness('cloud');
  const dashboard = loadPage('../pages/dashboard/index.js', dashboardHarness.app, {
    getDeviceInfo: () => ({ platform: 'ios' }),
  });
  dashboard.onLoad();
  dashboard.onShow();

  dashboardHarness.subscribers[0].onStatus({ status: 'waiting', mode: 'cloud', transport: 'watch' });
  assert.equal(dashboard.data.connectionLabel, '云端已连接，等待 Mac 开始巡检');
  assert.equal(dashboard.data.cloudTransportLabel, '等待任务');
  assert.equal(dashboard.data.errorMessage, '');

  dashboardHarness.subscribers[0].onStatus({ status: 'waiting', mode: 'cloud', transport: 'polling' });
  assert.equal(dashboard.data.connectionLabel, '云端已连接，正在轮询等待巡检任务');
  assert.equal(dashboard.data.cloudTransportLabel, '轮询等待');

  dashboardHarness.subscribers[0].onStatus({ status: 'polling', mode: 'cloud', transport: 'polling' });
  assert.match(dashboard.data.connectionLabel, /实时订阅中断/);
  assert.equal(dashboard.data.cloudTransportLabel, '轮询保障');

  dashboardHarness.subscribers[0].onStatus({
    status: 'error',
    mode: 'cloud',
    code: 'CLOUD_SNAPSHOT_REMOVED',
  });
  assert.match(dashboard.data.connectionLabel, /快照已被移除/);
  assert.equal(dashboard.data.connectionTone, 'offline');

  const alarmHarness = createAppHarness('cloud');
  const alarms = loadPage('../pages/alarms/index.js', alarmHarness.app);
  alarms.onLoad();
  alarms.onShow();
  alarmHarness.subscribers[0].onStatus({ status: 'waiting', mode: 'cloud' });
  assert.equal(alarms.data.sourceLabel, '云端已连接 · 等待巡检任务');
  assert.equal(alarms.data.modeTone, 'waiting');
  alarmHarness.subscribers[0].onStatus({ status: 'error', mode: 'cloud' });
  assert.equal(alarms.data.sourceLabel, '云端数据异常 · 保留最后异常记录');
  assert.equal(alarms.data.modeTone, 'offline');
});

test('alarms labels cloud correctly and updates while subscribed', () => {
  const harness = createAppHarness('cloud');
  const page = loadPage('../pages/alarms/index.js', harness.app);
  page.onLoad();
  page.onShow();
  assert.equal(page.data.sourceLabel, '云端实时异常记录');
  assert.equal(page.data.modeTone, 'online');

  const frameIndex = fixture.frames.findIndex((frame) => frame.active_alerts.length > 0);
  const payload = buildDashboardPayload(fixture, frameIndex);
  harness.subscribers[0].onData(payload, { mode: 'cloud' });
  assert.equal(page.data.historyCount, payload.alarms.history.length);
  assert.equal(page.data.sourceLabel, '云端实时异常记录');

  page.onHide();
  const hiddenCount = page._setDataCount;
  const terminal = buildDashboardPayload(fixture, fixture.frames.length - 1);
  harness.subscribers[0].onData(terminal, { mode: 'cloud' });
  assert.equal(page._setDataCount, hiddenCount);
  harness.app.globalData.dashboard = terminal;
  harness.app.globalData.dashboardMeta = { mode: 'cloud', status: 'terminal' };
  page.onShow();
  assert.equal(page.data.historyCount, terminal.alarms.history.length);

  page.onUnload();
  assert.equal(harness.unsubscribeCount, 1);
});

test('result page receives newer terminal data and only unsubscribes itself', () => {
  const harness = createAppHarness('cloud');
  const page = loadPage('../pages/result/index.js', harness.app);
  page.onLoad();
  page.onShow();
  assert.equal(page.data.resultTone, 'waiting');

  const payload = buildDashboardPayload(fixture, fixture.frames.length - 1);
  harness.subscribers[0].onData(payload, { mode: 'cloud' });
  assert.equal(page.data.resultTone, 'success');
  assert.equal(page.data.metrics[0].value, '4 / 4');

  page.onHide();
  const hiddenCount = page._setDataCount;
  harness.subscribers[0].onClear({ mode: 'cloud', reason: 'reset' });
  assert.equal(page._setDataCount, hiddenCount);
  harness.app.globalData.dashboard = null;
  page.onShow();
  assert.equal(page.data.resultTone, 'waiting');

  page.onUnload();
  assert.equal(harness.unsubscribeCount, 1);
});
