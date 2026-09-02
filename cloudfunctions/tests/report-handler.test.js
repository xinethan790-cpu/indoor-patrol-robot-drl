'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  createReportHandler,
  parseReportRequest,
} = require('../shared/report-handler');
const { buildPatrolReport } = require('../shared/report');
const { TEST_DEVICE_ID, testEnv, validReportPayload } = require('./helpers');

function handlerWithRepository(repository, identity) {
  return createReportHandler({
    env: testEnv(),
    getWxContext: () => identity === undefined ? { OPENID: 'wx-openid' } : identity,
    repository,
  });
}

test('report requires a logged-in WeChat identity before storage access', async () => {
  let reads = 0;
  const handler = handlerWithRepository({
    async getPatrolReport() { reads += 1; },
  }, {});

  assert.deepEqual(await handler({ run_id: 'test-run-001' }), {
    ok: false,
    error: 'AUTH_REQUIRED',
    message: 'WeChat identity is required',
  });
  assert.equal(reads, 0);
});

test('report request accepts exactly one bounded run_id and cannot select a device', async () => {
  assert.deepEqual(parseReportRequest({ run_id: 'run-001' }), { runId: 'run-001' });
  const calls = [];
  const handler = handlerWithRepository({
    async getPatrolReport(options) {
      calls.push(options);
      return null;
    },
  });
  for (const request of [
    null,
    {},
    { run_id: '../escape' },
    { run_id: 'run-001', device_id: 'other-device' },
    { run_id: 'x'.repeat(129) },
  ]) {
    const response = await handler(request);
    assert.equal(response.error, 'INVALID_REPORT_REQUEST');
  }
  assert.equal(calls.length, 0);
});

test('report returns the stored safe report under the configured fixed device', async () => {
  const report = buildPatrolReport(validReportPayload());
  const calls = [];
  const handler = handlerWithRepository({
    async getPatrolReport(options) {
      calls.push(options);
      return report;
    },
  });

  const response = await handler({ run_id: report.run.run_id });

  assert.equal(response.ok, true);
  assert.equal(response.schema_version, 'patrol-report-response/v1');
  assert.equal(response.device_id, TEST_DEVICE_ID);
  assert.equal(response.report, report);
  assert.deepEqual(calls, [{ deviceId: TEST_DEVICE_ID, runId: report.run.run_id }]);
  const serialized = JSON.stringify(response);
  for (const forbidden of [
    'terminal_payload',
    'latest_body_sha256',
    'body_sha256',
    'relay_nonce',
    'upload_secret',
  ]) assert.equal(serialized.includes(forbidden), false, forbidden);
});

test('missing, legacy, active, malformed, and storage-failed reports share one safe error', async () => {
  const safeError = {
    ok: false,
    error: 'REPORT_UNAVAILABLE',
    message: 'Patrol report is temporarily unavailable',
  };
  for (const repository of [
    { async getPatrolReport() { return null; } },
    { async getPatrolReport() { return { schema_version: 'legacy-report/v0' }; } },
    { async getPatrolReport() { return { terminal_payload: { secret: 'private' } }; } },
    { async getPatrolReport() { throw new Error('database password and collection name'); } },
  ]) {
    const handler = handlerWithRepository(repository);
    assert.deepEqual(await handler({ run_id: 'test-run-001' }), safeError);
  }
});
