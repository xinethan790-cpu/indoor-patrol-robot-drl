'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const {
  HistoryProtocolError,
  createPatrolHistoryService,
  projectHistoryItem,
  validateHistoryResponse,
} = require('../services/patrolHistory');

function historyItem(overrides) {
  return {
    run_id: 'history-run-001',
    status: 'completed',
    terminal: true,
    first_received_at_utc: '2026-09-01T06:00:00.000Z',
    latest_received_at_utc: '2026-09-01T06:01:10.000Z',
    latest_age_ms: 10_000,
    latest_sequence: 88,
    state: 'completed',
    outcome: 'success',
    progress_percent: 100,
    completed_checkpoint_count: 4,
    checkpoint_count: 4,
    active_alarm_count: 0,
    alarm_history_count: 6,
    result_outcome: 'completed',
    result_validated: true,
    report_available: false,
    ...(overrides || {}),
  };
}

function historyResponse(items, options) {
  const settings = options || {};
  return {
    ok: true,
    schema_version: 'patrol-history/v1',
    device_id: 'macmini-m4',
    page_size: settings.pageSize || 2,
    server_time_epoch_ms: settings.serverTimeEpochMs
      || Date.parse('2026-09-01T06:01:20.000Z'),
    stale_after_ms: 8000,
    items,
    has_more: settings.hasMore || false,
    next_cursor: settings.hasMore ? (settings.cursor || 'cursor_2') : null,
  };
}

function instantiatePage(definition) {
  return {
    ...definition,
    data: structuredClone(definition.data),
    setData(update) { this.data = { ...this.data, ...update }; },
  };
}

function createTimerHarness() {
  let nextId = 1;
  const tasks = new Map();
  const cleared = [];
  return {
    cleared,
    setTimer(callback, delayMs) {
      const id = nextId;
      nextId += 1;
      tasks.set(id, { callback, delayMs });
      return id;
    },
    clearTimer(id) {
      if (tasks.delete(id)) cleared.push(id);
    },
    pending() {
      return [...tasks.entries()].map(([id, task]) => ({ id, delayMs: task.delayMs }));
    },
    async run(id) {
      const task = tasks.get(id);
      assert.ok(task, `timer ${id} should be pending`);
      tasks.delete(id);
      await task.callback();
    },
  };
}

function loadHistoryPage(responses) {
  let definition;
  const calls = [];
  global.wx = {
    cloud: {
      callFunction(options) {
        calls.push(structuredClone(options));
        const response = responses.shift();
        return response instanceof Error
          ? Promise.reject(response)
          : Promise.resolve({ result: response });
      },
    },
    stopPullDownRefresh() {},
  };
  global.getApp = () => ({ globalData: { mode: 'cloud', dashboard: null } });
  global.Page = (value) => { definition = value; };
  const modulePath = '../pages/history/index.js';
  delete require.cache[require.resolve(modulePath)];
  const historyModule = require(modulePath);
  return { page: instantiatePage(definition), calls, historyModule };
}

function loadModeHistoryPage(mode, dashboard) {
  let definition;
  const calls = [];
  const navigations = [];
  global.getApp = () => ({ globalData: { mode, dashboard: dashboard || null } });
  global.wx = {
    cloud: {
      callFunction(options) {
        calls.push(structuredClone(options));
        return Promise.reject(new Error('cloud must not be called'));
      },
    },
    navigateTo(options) { navigations.push(structuredClone(options)); },
    stopPullDownRefresh() {},
  };
  global.Page = (value) => { definition = value; };
  const modulePath = '../pages/history/index.js';
  delete require.cache[require.resolve(modulePath)];
  const historyModule = require(modulePath);
  return { page: instantiatePage(definition), calls, navigations, historyModule };
}

test('history page is registered as a tab without replacing existing pages', () => {
  const appConfig = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../app.json'), 'utf8'));
  assert.equal(appConfig.pages.includes('pages/history/index'), true);
  assert.equal(
    appConfig.tabBar.list.some((item) => (
      item.pagePath === 'pages/history/index' && item.text === '历史'
    )),
    true,
  );
  assert.equal(appConfig.tabBar.list.length, 4);
});

test('history duration formatting carries rounded seconds into minutes', () => {
  const { historyModule } = loadHistoryPage([]);
  const origin = '2026-09-01T06:00:00.000Z';

  assert.equal(historyModule.formatDuration(origin, '2026-09-01T06:00:59.500Z'), '1 分');
  assert.equal(historyModule.formatDuration(origin, '2026-09-01T06:01:59.500Z'), '2 分');
  assert.equal(historyModule.formatDuration(origin, '2026-09-01T06:01:59.499Z'), '1 分 59 秒');
});

test('history card keeps a selectable run id, accessible report action, and animated loading', () => {
  const { historyModule } = loadHistoryPage([]);
  const runId = 'graduation-demo-seed7000-v1';
  const card = historyModule.projectHistoryCard(historyItem({ run_id: runId }));
  const template = fs.readFileSync(
    path.resolve(__dirname, '../pages/history/index.wxml'),
    'utf8',
  );
  const styles = fs.readFileSync(
    path.resolve(__dirname, '../pages/history/index.wxss'),
    'utf8',
  );

  assert.equal(card.runId, runId);
  assert.match(template, /history-card__run-id[^>]*user-select="true"[^>]*>\{\{item\.runId\}\}/);
  assert.doesNotMatch(template, /\sselectable(?:\s|=|>)/);
  assert.match(template, /role="button"/);
  assert.match(template, /aria-label="查看运行 \{\{item\.runId\}\} 的完整巡检报告"/);
  assert.match(styles, /\.history-state__spinner[\s\S]*animation:\s*history-spinner-rotate/);
  assert.match(styles, /@keyframes\s+history-spinner-rotate/);
});

test('history service requests strict cursor pages and safely projects summaries', async () => {
  const calls = [];
  const source = historyItem({
    secret: 'must-not-survive',
    nonce: 'must-not-survive',
    body_sha256: 'a'.repeat(64),
    terminal_payload: { private: true },
  });
  const service = createPatrolHistoryService({
    pageSize: 2,
    callFunction(options) {
      calls.push(options);
      return Promise.resolve({
        result: historyResponse([source], { pageSize: 2, hasMore: true }),
      });
    },
  });

  const page = await service.fetchPage(null);

  assert.deepEqual(calls, [{
    name: 'getPatrolHistory',
    data: { page_size: 2, cursor: null },
  }]);
  assert.equal(page.hasMore, true);
  assert.equal(page.nextCursor, 'cursor_2');
  assert.equal(page.items[0].secret, undefined);
  assert.equal(page.items[0].nonce, undefined);
  assert.equal(page.items[0].body_sha256, undefined);
  assert.equal(page.items[0].terminal_payload, undefined);
  assert.deepEqual(Object.keys(page.items[0]).sort(), [
    'active_alarm_count',
    'alarm_history_count',
    'checkpoint_count',
    'completed_checkpoint_count',
    'first_received_at_utc',
    'latest_age_ms',
    'latest_received_at_utc',
    'latest_sequence',
    'outcome',
    'progress_percent',
    'report_available',
    'result_outcome',
    'result_validated',
    'run_id',
    'state',
    'status',
    'terminal',
  ]);
});

test('history response validation fails closed on forged pagination and status metadata', () => {
  const badCursor = historyResponse([historyItem()], { pageSize: 2, hasMore: true });
  badCursor.next_cursor = null;
  assert.throws(
    () => validateHistoryResponse(badCursor, 2),
    (error) => error instanceof HistoryProtocolError,
  );

  const badStatus = historyItem({ status: 'interrupted', terminal: true });
  assert.throws(
    () => projectHistoryItem(badStatus),
    (error) => error instanceof HistoryProtocolError,
  );

  const reversedTime = historyItem({
    first_received_at_utc: '2026-09-01T06:02:00.000Z',
  });
  assert.throws(
    () => projectHistoryItem(reversedTime),
    (error) => error instanceof HistoryProtocolError,
  );

  const forgedFreshness = historyResponse([historyItem({
    status: 'active',
    terminal: false,
    result_outcome: null,
    result_validated: false,
  })]);
  assert.throws(
    () => validateHistoryResponse(forgedFreshness, 2),
    (error) => error instanceof HistoryProtocolError,
  );

  assert.throws(
    () => validateHistoryResponse({
      ok: false,
      error: 'HISTORY_UNAVAILABLE',
      message: 'database password and collection internals',
    }, 2),
    (error) => (
      error instanceof HistoryProtocolError
      && error.message === '巡检历史暂时不可用，请稍后重试'
    ),
  );
});

test('history service hides SDK, trace, and cloud-function rejection details', async () => {
  const service = createPatrolHistoryService({
    pageSize: 2,
    callFunction() {
      return Promise.reject(new Error(
        'cloud.callFunction:fail FUNCTION_NOT_FOUND callId=private trace=private URL=private',
      ));
    },
  });

  await assert.rejects(
    () => service.fetchPage(null),
    (error) => (
      error instanceof HistoryProtocolError
      && error.code === 'HISTORY_UNAVAILABLE'
      && error.message === '巡检历史暂时不可用，请稍后重试'
      && !/FUNCTION_NOT_FOUND|callId|trace|URL/.test(error.message)
    ),
  );
});

test('history page paginates and deduplicates records by run id', async () => {
  const first = historyResponse([
    historyItem({ run_id: 'run-3' }),
    historyItem({
      run_id: 'run-unvalidated-success',
      result_validated: false,
    }),
    historyItem({
      run_id: 'run-collision',
      outcome: 'collision',
    }),
    historyItem({
      run_id: 'run-legacy-terminal',
      status: 'unknown_terminal',
      state: null,
      outcome: null,
      result_outcome: null,
      result_validated: false,
    }),
    historyItem({
      run_id: 'run-2',
      status: 'aborted',
      state: 'failed',
      outcome: 'error',
      result_outcome: 'aborted',
      result_validated: false,
    }),
  ], { pageSize: 10, hasMore: true, cursor: 'cursor_next' });
  const second = historyResponse([
    historyItem({
      run_id: 'run-2',
      status: 'aborted',
      state: 'failed',
      outcome: 'error',
      result_outcome: 'aborted',
      result_validated: false,
    }),
    historyItem({
      run_id: 'run-1',
      status: 'interrupted',
      terminal: false,
      result_outcome: null,
      result_validated: false,
      progress_percent: 25,
      completed_checkpoint_count: 1,
    }),
    historyItem({
      run_id: 'run-active',
      status: 'active',
      terminal: false,
      latest_received_at_utc: '2026-09-01T06:01:15.000Z',
      latest_age_ms: 5000,
      state: 'navigating',
      outcome: 'running',
      result_outcome: null,
      result_validated: false,
      progress_percent: 50,
      completed_checkpoint_count: 2,
    }),
  ], { pageSize: 10 });
  const { page, calls } = loadHistoryPage([first, second]);

  page.onLoad();
  await page.onShow();
  assert.deepEqual(page.data.items.map((item) => item.runId), [
    'run-3',
    'run-unvalidated-success',
    'run-collision',
    'run-legacy-terminal',
    'run-2',
  ]);
  assert.equal(page.data.items[0].statusLabel, '正常完成');
  assert.equal(page.data.items[1].statusLabel, '任务已结束');
  assert.equal(page.data.items[1].statusDescription, '终态未验证');
  assert.equal(page.data.items[2].statusLabel, '任务已结束');
  assert.match(page.data.items[2].statusDescription, /碰撞/);
  assert.equal(page.data.items[3].statusLabel, '旧版终态');
  assert.equal(page.data.items[3].statusDescription, '终态类型未回填');
  assert.equal(page.data.items[4].statusLabel, '异常中止');
  assert.equal(page.data.hasMore, true);

  await page.onReachBottom();
  assert.deepEqual(page.data.items.map((item) => item.runId), [
    'run-3',
    'run-unvalidated-success',
    'run-collision',
    'run-legacy-terminal',
    'run-2',
    'run-1',
    'run-active',
  ]);
  assert.equal(page.data.items[5].statusLabel, '上报失联');
  assert.match(page.data.items[5].statusDescription, /超过 8 秒/);
  assert.equal(page.data.items[6].statusLabel, '正在巡检');
  assert.equal(page.data.hasMore, false);
  assert.deepEqual(calls.map((call) => call.data.cursor), [null, 'cursor_next']);
});

test('history page refreshes stale status on a later show without request storms', async () => {
  const latestReceivedAtUtc = '2026-09-01T06:01:15.000Z';
  const active = historyResponse([historyItem({
    run_id: 'run-changing-state',
    status: 'active',
    terminal: false,
    latest_received_at_utc: latestReceivedAtUtc,
    latest_age_ms: 5000,
    state: 'navigating',
    outcome: 'running',
    result_outcome: null,
    result_validated: false,
  })], {
    pageSize: 10,
    serverTimeEpochMs: Date.parse(latestReceivedAtUtc) + 5000,
  });
  const interrupted = historyResponse([historyItem({
    run_id: 'run-changing-state',
    status: 'interrupted',
    terminal: false,
    latest_received_at_utc: latestReceivedAtUtc,
    latest_age_ms: 9000,
    state: 'navigating',
    outcome: 'running',
    result_outcome: null,
    result_validated: false,
  })], {
    pageSize: 10,
    serverTimeEpochMs: Date.parse(latestReceivedAtUtc) + 9000,
  });
  const {
    page,
    historyModule,
    calls,
  } = loadHistoryPage([active, interrupted]);
  let now = 1000;

  page.onLoad();
  page._now = () => now;
  await page.onShow();
  assert.equal(page.data.items[0].statusLabel, '正在巡检');
  assert.equal(calls.length, 1);

  await page.onShow();
  now += historyModule.AUTO_REFRESH_MIN_INTERVAL_MS - 1;
  await page.onShow();
  assert.equal(calls.length, 1);

  now += 1;
  await page.onShow();
  assert.equal(calls.length, 2);
  assert.equal(page.data.items[0].statusLabel, '上报失联');
});

test('visible active history schedules bounded refreshes and cancels hidden or stale timers', async () => {
  const latestReceivedAtUtc = '2026-09-01T06:01:15.000Z';
  const activeResponse = (latestAgeMs) => historyResponse([historyItem({
    run_id: 'run-live-history',
    status: 'active',
    terminal: false,
    latest_received_at_utc: latestReceivedAtUtc,
    latest_age_ms: latestAgeMs,
    state: 'navigating',
    outcome: 'running',
    result_outcome: null,
    result_validated: false,
  })], {
    pageSize: 10,
    serverTimeEpochMs: Date.parse(latestReceivedAtUtc) + latestAgeMs,
  });
  const terminalResponse = historyResponse([historyItem({
    run_id: 'run-live-history',
  })], { pageSize: 10 });
  const { page, historyModule, calls } = loadHistoryPage([
    activeResponse(5000),
    activeResponse(100),
    activeResponse(7900),
    terminalResponse,
  ]);
  const timers = createTimerHarness();
  let now = 1000;

  page.onLoad();
  page._now = () => now;
  page._setTimer = timers.setTimer;
  page._clearTimer = timers.clearTimer;
  await page.onShow();

  const firstTimer = timers.pending()[0];
  assert.equal(calls.length, 1);
  assert.equal(firstTimer.delayMs, 3001);

  now += 100;
  await page.onPullDownRefresh();
  assert.equal(calls.length, 2);
  assert.equal(timers.cleared.includes(firstTimer.id), true);
  const manualRefreshTimer = timers.pending()[0];
  assert.equal(manualRefreshTimer.delayMs, 7901);

  page.onHide();
  assert.equal(timers.cleared.includes(manualRefreshTimer.id), true);
  assert.deepEqual(timers.pending(), []);
  await page.onShow();
  assert.equal(calls.length, 2);
  const resumedTimer = timers.pending()[0];
  assert.equal(resumedTimer.delayMs, 7901);

  now += resumedTimer.delayMs;
  await timers.run(resumedTimer.id);
  assert.equal(calls.length, 3);
  assert.equal(page.data.items[0].statusLabel, '正在巡检');
  const boundedTimer = timers.pending()[0];
  assert.equal(boundedTimer.delayMs, historyModule.ACTIVE_REFRESH_MIN_DELAY_MS);

  now += boundedTimer.delayMs;
  await timers.run(boundedTimer.id);
  assert.equal(calls.length, 4);
  assert.equal(page.data.items[0].statusLabel, '正常完成');
  assert.deepEqual(timers.pending(), []);

  page.onUnload();
  assert.deepEqual(timers.pending(), []);
});

test('history page exposes honest empty and recoverable error states', async () => {
  const emptyHarness = loadHistoryPage([historyResponse([], { pageSize: 10 })]);
  emptyHarness.page.onLoad();
  await emptyHarness.page.onShow();
  assert.equal(emptyHarness.page.data.empty, true);
  assert.equal(emptyHarness.page.data.errorMessage, '');

  const errorHarness = loadHistoryPage([
    new Error('cloud.callFunction:fail FUNCTION_NOT_FOUND callId=private trace=private'),
    historyResponse([historyItem()], { pageSize: 10 }),
  ]);
  errorHarness.page.onLoad();
  await errorHarness.page.onShow();
  assert.equal(errorHarness.page.data.empty, false);
  assert.equal(errorHarness.page.data.errorMessage, '巡检历史暂时不可用，请稍后重试');
  await errorHarness.page.retry();
  assert.equal(errorHarness.page.data.errorMessage, '');
  assert.equal(errorHarness.page.data.items.length, 1);
});

test('demo history is exactly one verified local archive and never reads cloud', async () => {
  const { page, calls, navigations } = loadModeHistoryPage('demo');

  page.onLoad();
  await page.onShow();

  assert.equal(calls.length, 0);
  assert.equal(page.data.items.length, 1);
  assert.equal(page.data.items[0].runId, 'graduation-demo-seed7000-v1');
  assert.equal(page.data.items[0].reportAvailable, true);
  assert.equal(page.data.items[0].reportSource, 'demo');
  assert.equal(page.data.items[0].timeText, '内置演示档案');
  page.openReport({ currentTarget: { dataset: {
    runId: page.data.items[0].runId,
    source: 'demo',
  } } });
  assert.equal(navigations.length, 1);
  assert.match(navigations[0].url, /^\/pages\/report\/index\?/);
});

test('local history honestly stays empty until the current session has a complete terminal report', async () => {
  const { page, calls } = loadModeHistoryPage('local', null);

  page.onLoad();
  await page.onShow();

  assert.equal(calls.length, 0);
  assert.equal(page.data.items.length, 0);
  assert.equal(page.data.empty, true);
  assert.match(page.data.emptyMessage, /当前会话/);
});
