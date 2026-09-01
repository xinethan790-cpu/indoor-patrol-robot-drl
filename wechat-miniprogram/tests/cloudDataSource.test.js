'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const fixture = require('../data/graduation-demo-seed7000-v1.js');
const {
  CLOUD_SNAPSHOT_SCHEMA_VERSION,
  createCloudDataSource,
  validatePollingFreshness,
  validateCloudSnapshot,
} = require('../services/cloudDataSource');
const { buildDashboardPayload } = require('../services/offlineReplay');
const { PayloadProtocolError } = require('../services/protocol');
const { createManualScheduler } = require('./helpers');

const ENV_ID = 'cloud1-5gpkygv2be4de05a';

function createSnapshot(frameIndex, runId) {
  const payload = buildDashboardPayload(fixture, frameIndex);
  if (runId) payload.run.run_id = runId;
  return {
    _id: 'main',
    schema_version: CLOUD_SNAPSHOT_SCHEMA_VERSION,
    device_id: 'macmini-m4-webots',
    run_id: payload.run.run_id,
    sequence: payload.run.sequence,
    terminal: payload.run.terminal,
    body_sha256: 'a'.repeat(64),
    received_at_utc: '2026-09-01T06:00:00.000Z',
    received_at_epoch_ms: 1788242400000,
    payload,
  };
}

function createPollingResult(frameIndex, options) {
  const config = options || {};
  const snapshot = createSnapshot(frameIndex, config.runId);
  const ageMs = config.ageMs === undefined ? 0 : config.ageMs;
  const serverTimeEpochMs = snapshot.received_at_epoch_ms + ageMs;
  return {
    found: true,
    snapshot,
    server_time_epoch_ms: serverTimeEpochMs,
    age_ms: ageMs,
    stale: ageMs > 8000,
  };
}

function createCloudHarness(getFactory) {
  const harness = {
    watchOptions: null,
    observerClosed: false,
    callFunctionCalls: [],
    databaseOptions: null,
  };
  const document = {
    watch(options) {
      harness.watchOptions = options;
      return {
        close() { harness.observerClosed = true; },
      };
    },
  };
  harness.wxApi = {
    cloud: {
      callFunction(options) {
        harness.callFunctionCalls.push(options);
        const callIndex = harness.callFunctionCalls.length;
        if (getFactory) return getFactory(callIndex);
        return Promise.resolve({ result: createPollingResult(0) });
      },
      database(options) {
        harness.databaseOptions = options;
        return {
          collection(name) {
            assert.equal(name, 'patrol_live');
            return {
              doc(id) {
                assert.equal(id, 'main');
                return document;
              },
            };
          },
        };
      },
    },
  };
  return harness;
}

test('cloud snapshot validates its envelope and unchanged dashboard payload', () => {
  const snapshot = createSnapshot(2);
  const result = validateCloudSnapshot(snapshot);
  assert.equal(result.snapshot, snapshot);
  assert.equal(result.payload, snapshot.payload);

  const mismatch = structuredClone(snapshot);
  mismatch.sequence += 1;
  assert.throws(
    () => validateCloudSnapshot(mismatch),
    (error) => error instanceof PayloadProtocolError && error.code === 'CLOUD_METADATA_MISMATCH',
  );
});

test('new mini accepts and normalizes a legacy cloud snapshot during rolling deployment', () => {
  const legacy = createSnapshot(2);
  delete legacy.payload.run.control_step;
  delete legacy.payload.map.dynamic_obstacles;
  delete legacy.payload.avoidance;

  const result = validateCloudSnapshot(legacy);

  assert.notEqual(result.snapshot, legacy);
  assert.notEqual(result.payload, legacy.payload);
  assert.equal(result.snapshot.payload, result.payload);
  assert.equal(result.payload.run.control_step, null);
  assert.deepEqual(result.payload.map.dynamic_obstacles, []);
  assert.equal(result.payload.avoidance.available, false);
  assert.equal(Object.hasOwn(legacy.payload, 'avoidance'), false);
});

test('polling freshness metadata is internally consistent and fails closed when forged', () => {
  const result = createPollingResult(0, { ageMs: 5000 });
  assert.deepEqual(validatePollingFreshness(result, result.snapshot), {
    serverTimeEpochMs: result.server_time_epoch_ms,
    ageMs: 5000,
    stale: false,
  });

  for (const mutate of [
    (value) => { delete value.server_time_epoch_ms; },
    (value) => { value.age_ms += 1; },
    (value) => { value.stale = true; },
  ]) {
    const invalid = structuredClone(result);
    mutate(invalid);
    assert.throws(
      () => validatePollingFreshness(invalid, invalid.snapshot),
      (error) => error instanceof PayloadProtocolError
        && error.code === 'INVALID_CLOUD_SNAPSHOT',
    );
  }
});

test('watch accepts only new sequences and marks unchanged data stale after eight seconds', () => {
  const scheduler = createManualScheduler();
  const harness = createCloudHarness();
  const data = [];
  const statuses = [];
  const source = createCloudDataSource({
    env: ENV_ID,
    wxApi: harness.wxApi,
    scheduler,
    clock: () => 1788242400000,
    staleAfterMs: 8000,
  });
  source.start({
    onData: (payload) => data.push(payload.run.sequence),
    onStatus: (event) => statuses.push(event.status),
  });
  assert.deepEqual(harness.databaseOptions, { env: ENV_ID });
  harness.watchOptions.onChange({ docs: [createSnapshot(0)] });
  harness.watchOptions.onChange({ docs: [createSnapshot(0)] });
  assert.deepEqual(data, [0]);
  assert.equal(scheduler.pending().length, 1);
  assert.equal(scheduler.pending()[0].delay, 8000);
  scheduler.runNext();
  assert.equal(source.getState().status, 'stale');
  assert.equal(statuses.at(-1), 'stale');

  harness.watchOptions.onChange({ docs: [createSnapshot(1)] });
  assert.deepEqual(data, [0, fixture.frames[1].sequence]);
  assert.equal(source.getState().status, 'running');
  assert.equal(scheduler.pending()[0].delay, 8000);
});

test('watch failure avoids re-closing the SDK observer and falls back to cloud-function polling', async () => {
  const scheduler = createManualScheduler();
  const harness = createCloudHarness(() => Promise.resolve({
    result: createPollingResult(1),
  }));
  const data = [];
  const errors = [];
  const source = createCloudDataSource({
    env: ENV_ID,
    wxApi: harness.wxApi,
    scheduler,
    clock: () => 1788242400000,
    staleAfterMs: 8000,
    pollIntervalMs: 2000,
  });
  source.start({
    onData: (payload, meta) => data.push({ sequence: payload.run.sequence, meta }),
    onError: (error, meta) => errors.push({ error, meta }),
  });
  harness.watchOptions.onError({ errMsg: 'watch disconnected' });
  await Promise.resolve();
  await Promise.resolve();

  assert.equal(harness.observerClosed, false);
  assert.deepEqual(harness.callFunctionCalls, [{ name: 'getPatrolSnapshot' }]);
  assert.equal(data.length, 1);
  assert.equal(data[0].meta.transport, 'polling');
  assert.equal(errors[0].meta.fallback, true);
  assert.equal(source.getState().transport, 'polling');
  assert.deepEqual(scheduler.pending().map((task) => task.delay).sort((a, b) => a - b), [2000, 8000]);

  source.stop();
  assert.equal(source.getState().status, 'stopped');
  assert.equal(scheduler.pending().length, 0);
});

test('a delayed error from an old watch cannot detach the restarted watch', () => {
  const watchCallbacks = [];
  const closeCounts = [];
  const document = {
    watch(callbacks) {
      const index = watchCallbacks.length;
      watchCallbacks.push(callbacks);
      closeCounts.push(0);
      return {
        close() { closeCounts[index] += 1; },
      };
    },
  };
  const wxApi = {
    cloud: {
      callFunction() {
        throw new Error('polling must not start for an obsolete watch');
      },
      database() {
        return {
          collection() {
            return { doc() { return document; } };
          },
        };
      },
    },
  };
  const source = createCloudDataSource({ env: ENV_ID, wxApi });

  source.start({});
  const oldWatch = watchCallbacks[0];
  source.stop();
  source.start({});
  oldWatch.onError({ errMsg: 'late error from closed watch' });
  source.stop();

  assert.equal(watchCallbacks.length, 2);
  assert.deepEqual(closeCounts, [1, 1]);
  assert.equal(source.getState().status, 'stopped');
});

test('cloud-function polling backs off on failure and slows down after terminal', async () => {
  const failureScheduler = createManualScheduler();
  const failureHarness = createCloudHarness(() => Promise.reject(new Error('cloud unavailable')));
  const failureSource = createCloudDataSource({
    env: ENV_ID,
    wxApi: failureHarness.wxApi,
    scheduler: failureScheduler,
    pollIntervalMs: 2000,
    maxPollBackoffMs: 30000,
  });
  failureSource.start({});
  failureHarness.watchOptions.onError({ errMsg: 'watch disconnected' });
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(failureScheduler.pending()[0].delay, 2000);
  failureScheduler.runNext();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(failureScheduler.pending()[0].delay, 4000);

  const terminalScheduler = createManualScheduler();
  const terminalHarness = createCloudHarness(() => Promise.resolve({
    result: {
      found: true,
      ...createPollingResult(fixture.frames.length - 1),
    },
  }));
  const terminalSource = createCloudDataSource({
    env: ENV_ID,
    wxApi: terminalHarness.wxApi,
    scheduler: terminalScheduler,
    clock: () => 1788242400000,
    pollIntervalMs: 2000,
    terminalPollIntervalMs: 15000,
  });
  terminalSource.start({});
  terminalHarness.watchOptions.onError({ errMsg: 'watch disconnected' });
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(terminalSource.getState().status, 'terminal');
  assert.equal(terminalScheduler.pending()[0].delay, 15000);
});

test('polling uses server freshness when the device clock is far ahead', async () => {
  const scheduler = createManualScheduler();
  const harness = createCloudHarness(() => Promise.resolve({
    result: createPollingResult(1, { ageMs: 1000 }),
  }));
  const data = [];
  const source = createCloudDataSource({
    env: ENV_ID,
    wxApi: harness.wxApi,
    scheduler,
    clock: () => 4102444800000,
    staleAfterMs: 8000,
    pollIntervalMs: 2000,
  });
  source.start({ onData: (payload, meta) => data.push({ payload, meta }) });
  harness.watchOptions.onError({ errMsg: 'watch disconnected' });
  await Promise.resolve();
  await Promise.resolve();

  assert.equal(source.getState().status, 'running');
  assert.equal(data[0].meta.ageMs, 1000);
  assert.equal(data[0].meta.stale, false);
  assert.equal(data[0].meta.serverTimeEpochMs, 1788242401000);
  assert.deepEqual(scheduler.pending().map((task) => task.delay).sort((a, b) => a - b), [2000, 7000]);
});

test('polling trusts a stale server result when the device clock is far behind', async () => {
  const scheduler = createManualScheduler();
  const harness = createCloudHarness(() => Promise.resolve({
    result: createPollingResult(1, { ageMs: 9000 }),
  }));
  const source = createCloudDataSource({
    env: ENV_ID,
    wxApi: harness.wxApi,
    scheduler,
    clock: () => 0,
    staleAfterMs: 8000,
    pollIntervalMs: 2000,
  });
  source.start({});
  harness.watchOptions.onError({ errMsg: 'watch disconnected' });
  await Promise.resolve();
  await Promise.resolve();

  assert.equal(source.getState().status, 'stale');
  assert.deepEqual(scheduler.pending().map((task) => task.delay), [2000]);
});

test('polling freshness metadata violation fails closed without retrying', async () => {
  const scheduler = createManualScheduler();
  const invalidResult = createPollingResult(1, { ageMs: 1000 });
  invalidResult.server_time_epoch_ms += 1;
  const harness = createCloudHarness(() => Promise.resolve({ result: invalidResult }));
  const errors = [];
  const source = createCloudDataSource({
    env: ENV_ID,
    wxApi: harness.wxApi,
    scheduler,
    pollIntervalMs: 2000,
  });
  source.start({ onError: (error, meta) => errors.push({ error, meta }) });
  harness.watchOptions.onError({ errMsg: 'watch disconnected' });
  await Promise.resolve();
  await Promise.resolve();

  assert.equal(source.getState().status, 'error');
  assert.equal(errors.at(-1).error.code, 'INVALID_CLOUD_SNAPSHOT');
  assert.deepEqual(errors.at(-1).meta, { retrying: false, protocol: true });
  assert.equal(scheduler.pending().length, 0);
});

test('an already old initial cloud snapshot is marked stale immediately', () => {
  const scheduler = createManualScheduler();
  const harness = createCloudHarness();
  const source = createCloudDataSource({
    env: ENV_ID,
    wxApi: harness.wxApi,
    scheduler,
    clock: () => 1788242409000,
    staleAfterMs: 8000,
  });
  source.start({});
  harness.watchOptions.onChange({ docs: [createSnapshot(0)] });
  assert.equal(source.getState().status, 'stale');
  assert.equal(scheduler.pending().length, 0);
});

test('cloud protocol violation fails closed without polling untrusted data', () => {
  const scheduler = createManualScheduler();
  const harness = createCloudHarness();
  const errors = [];
  const source = createCloudDataSource({ env: ENV_ID, wxApi: harness.wxApi, scheduler });
  source.start({ onError: (error, meta) => errors.push({ error, meta }) });
  const invalid = createSnapshot(0);
  invalid.payload.map.robot.x_m = 'not-a-number';
  harness.watchOptions.onChange({ docs: [invalid] });
  assert.equal(source.getState().status, 'error');
  assert.equal(harness.observerClosed, true);
  assert.equal(errors[0].meta.retrying, false);
  assert.equal(scheduler.pending().length, 0);
});

test('terminal cloud payload keeps watch ready for a subsequent patrol run', () => {
  const scheduler = createManualScheduler();
  const harness = createCloudHarness();
  const source = createCloudDataSource({
    env: ENV_ID,
    wxApi: harness.wxApi,
    scheduler,
    clock: () => 1788242400000,
  });
  source.start({});
  harness.watchOptions.onChange({ docs: [createSnapshot(fixture.frames.length - 1)] });
  assert.equal(source.getState().status, 'terminal');
  assert.equal(harness.observerClosed, false);
  assert.equal(scheduler.pending().length, 0);

  harness.watchOptions.onChange({ docs: [createSnapshot(0, 'next-patrol-run')] });
  assert.equal(source.getState().status, 'running');
  assert.equal(source.getState().runId, 'next-patrol-run');
  assert.equal(source.getState().lastSequence, 0);
  assert.equal(scheduler.pending().length, 1);
});
