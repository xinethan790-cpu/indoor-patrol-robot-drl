'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const fixture = require('../data/graduation-demo-seed7000-v1.js');
const { buildDashboardPayload } = require('../services/offlineReplay');

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
  };
  return {
    app,
    subscribers,
    setSessionState(update) { sessionState = { ...sessionState, ...update }; },
    get unsubscribeCount() { return unsubscribeCount; },
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
  assert.equal(alarms.data.sourceLabel, '云端实时异常记录');

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
