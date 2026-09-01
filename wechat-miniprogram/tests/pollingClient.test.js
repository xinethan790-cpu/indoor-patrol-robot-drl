'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const fixture = require('../data/graduation-demo-seed7000-v1.js');
const { buildDashboardPayload } = require('../services/offlineReplay');
const { createPollingClient } = require('../services/pollingClient');
const { createManualScheduler } = require('./helpers');

function createManualWx() {
  const requests = [];
  return {
    requests,
    request(options) {
      const task = {
        aborted: false,
        abort() { this.aborted = true; },
      };
      requests.push({ options, task });
      return task;
    },
  };
}

test('polling accepts one terminal payload and schedules no more requests', () => {
  const scheduler = createManualScheduler();
  const wxApi = createManualWx();
  const received = [];
  const source = createPollingClient({
    wxApi,
    scheduler,
    expectedRunId: fixture.run.run_id,
    intervalMs: 10,
    timeoutMs: 50,
  });
  source.start({ onData: (payload) => received.push(payload) });
  assert.equal(wxApi.requests.length, 1);
  wxApi.requests[0].options.success({
    statusCode: 200,
    data: buildDashboardPayload(fixture, fixture.frames.length - 1),
  });
  assert.equal(received.length, 1);
  assert.equal(source.getState().status, 'terminal');
  assert.equal(scheduler.pending().length, 0);
});

test('request timeout aborts and retries with deterministic exponential backoff', () => {
  const scheduler = createManualScheduler();
  const wxApi = createManualWx();
  const errors = [];
  const source = createPollingClient({
    wxApi,
    scheduler,
    timeoutMs: 50,
    backoff: { initialMs: 100, maxMs: 500, multiplier: 2, jitterRatio: 0 },
  });
  source.start({ onError: (error, meta) => errors.push({ error, meta }) });
  const timeoutTask = scheduler.runNext();
  assert.equal(timeoutTask.delay, 50);
  assert.equal(wxApi.requests[0].task.aborted, true);
  assert.equal(errors.length, 1);
  assert.equal(errors[0].meta.retrying, true);
  assert.equal(source.getState().status, 'backoff');
  const retryTask = scheduler.runNext();
  assert.equal(retryTask.delay, 100);
  assert.equal(wxApi.requests.length, 2);
  source.stop();
});

test('network failure backs off while protocol mixing fails closed', () => {
  const scheduler = createManualScheduler();
  const wxApi = createManualWx();
  const errors = [];
  const source = createPollingClient({
    wxApi,
    scheduler,
    expectedRunId: fixture.run.run_id,
    timeoutMs: 50,
    backoff: { initialMs: 25, maxMs: 100, multiplier: 2, jitterRatio: 0 },
  });
  source.start({ onError: (error, meta) => errors.push({ error, meta }) });
  wxApi.requests[0].options.fail({ errMsg: 'connection refused' });
  assert.equal(source.getState().status, 'backoff');
  assert.equal(scheduler.pending()[0].delay, 25);
  scheduler.runNext();
  const foreign = buildDashboardPayload(fixture, 0);
  foreign.run.run_id = 'foreign-run';
  wxApi.requests[1].options.success({ statusCode: 200, data: foreign });
  assert.equal(source.getState().status, 'error');
  assert.equal(errors.at(-1).meta.retrying, false);
  assert.equal(errors.at(-1).error.code, 'RUN_ID_MISMATCH');
  assert.equal(scheduler.pending().length, 0);
});
