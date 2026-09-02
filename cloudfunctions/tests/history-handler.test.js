'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  createHistoryHandler,
  encodeCursor,
  publicHistoryItem,
} = require('../shared/history-handler');
const { TEST_SECRET, validRuntimeAbortPayload } = require('./helpers');

const DEVICE_ID = 'macmini-m4';
const BASE_MS = 1_700_000_000_000;

function runDocument(overrides) {
  const changes = overrides || {};
  const terminal = changes.terminal ?? true;
  return {
    schema_version: 'patrol-cloud-run/v1',
    device_id: changes.deviceId || DEVICE_ID,
    run_id: changes.runId || 'run-completed',
    first_received_at_utc: new Date(BASE_MS).toISOString(),
    first_received_at_epoch_ms: BASE_MS,
    latest_sequence: changes.sequence ?? 9,
    terminal,
    latest_received_at_utc: new Date(BASE_MS + 9000).toISOString(),
    latest_received_at_epoch_ms: BASE_MS + 9000,
    state: terminal ? 'completed' : 'navigating',
    runtime_outcome: terminal ? 'success' : 'running',
    progress_percent: terminal ? 100 : 50,
    completed_checkpoint_count: terminal ? 4 : 2,
    checkpoint_count: 4,
    active_alarm_count: 0,
    alarm_history_count: 2,
    incomplete: false,
    result_outcome: terminal ? 'completed' : null,
    result_validated: terminal,
    latest_body_sha256: 'a'.repeat(64),
    terminal_payload: { nonce: 'private', secret: 'private' },
    relay_nonce: 'private',
    ...changes,
  };
}

function handlerWithPages(pages, calls) {
  return createHistoryHandler({
    env: {
      PATROL_CLOUDBASE_ENV_ID: 'cloud1-5gpkygv2be4de05a',
      PATROL_RELAY_DEVICE_ID: DEVICE_ID,
      PATROL_RELAY_UPLOAD_SECRET: TEST_SECRET,
    },
    getWxContext: () => ({ OPENID: 'wx-openid' }),
    now: () => BASE_MS + 20_000,
    repository: {
      async getPatrolHistoryPage(options) {
        calls.push(options);
        return pages.shift();
      },
    },
  });
}

test('history requires a logged-in WeChat identity before reading the repository', async () => {
  let reads = 0;
  const handler = createHistoryHandler({
    env: { PATROL_RELAY_DEVICE_ID: DEVICE_ID },
    getWxContext: () => ({}),
    repository: {
      async getPatrolHistoryPage() { reads += 1; },
    },
  });

  assert.deepEqual(await handler({ page_size: 10, cursor: null }), {
    ok: false,
    error: 'AUTH_REQUIRED',
    message: 'WeChat identity is required',
  });
  assert.equal(reads, 0);
});

test('history fails safely before storage access when its cursor-signing secret is unavailable', async () => {
  let reads = 0;
  const handler = createHistoryHandler({
    env: { PATROL_RELAY_DEVICE_ID: DEVICE_ID },
    getWxContext: () => ({ OPENID: 'wx-openid' }),
    repository: {
      async getPatrolHistoryPage() { reads += 1; },
    },
  });

  assert.deepEqual(await handler({ page_size: 10, cursor: null }), {
    ok: false,
    error: 'HISTORY_UNAVAILABLE',
    message: 'Patrol history is temporarily unavailable',
  });
  assert.equal(reads, 0);
});

test('history pagination binds an opaque cursor to page size and configured device', async () => {
  const calls = [];
  const handler = handlerWithPages([
    {
      documents: [runDocument({ runId: 'run-b' }), runDocument({ runId: 'run-a' })],
      hasMore: true,
      nextOffset: 2,
    },
    {
      documents: [runDocument({ runId: 'run-older' })],
      hasMore: false,
      nextOffset: 3,
    },
  ], calls);

  const first = await handler({ page_size: 2, cursor: null });
  assert.equal(first.ok, true);
  assert.equal(first.server_time_epoch_ms, BASE_MS + 20_000);
  assert.equal(first.stale_after_ms, 8000);
  assert.equal(first.has_more, true);
  assert.equal(typeof first.next_cursor, 'string');
  assert.deepEqual(first.items.map((item) => item.run_id), ['run-b', 'run-a']);
  assert.deepEqual(calls[0], { deviceId: DEVICE_ID, pageSize: 2, offset: 0 });

  const second = await handler({ page_size: 2, cursor: first.next_cursor });
  assert.equal(second.has_more, false);
  assert.equal(second.next_cursor, null);
  assert.deepEqual(calls[1], { deviceId: DEVICE_ID, pageSize: 2, offset: 2 });

  const wrongPageSize = await handler({ page_size: 3, cursor: first.next_cursor });
  assert.equal(wrongPageSize.error, 'INVALID_HISTORY_REQUEST');
  const foreignCursor = encodeCursor(2, 2, 'other-device', TEST_SECRET);
  const wrongDevice = await handler({ page_size: 2, cursor: foreignCursor });
  assert.equal(wrongDevice.error, 'INVALID_HISTORY_REQUEST');

  const tamperedEnvelope = JSON.parse(
    Buffer.from(first.next_cursor, 'base64url').toString('utf8'),
  );
  tamperedEnvelope.o = 1_000_000;
  const forgedDeepOffset = Buffer.from(
    JSON.stringify(tamperedEnvelope),
    'utf8',
  ).toString('base64url');
  const forged = await handler({ page_size: 2, cursor: forgedDeepOffset });
  assert.equal(forged.error, 'INVALID_HISTORY_REQUEST');
  assert.equal(calls.length, 2);
});

test('history request strictly bounds page size, cursor, and fields', async () => {
  const calls = [];
  const handler = handlerWithPages([], calls);
  const invalid = [
    { page_size: 0, cursor: null },
    { page_size: 21, cursor: null },
    { page_size: 1.5, cursor: null },
    { page_size: '10', cursor: null },
    { page_size: 10, cursor: '***' },
    { page_size: 10, cursor: null, device_id: 'other-device' },
  ];
  for (const request of invalid) {
    const response = await handler(request);
    assert.equal(response.error, 'INVALID_HISTORY_REQUEST');
  }
  assert.equal(calls.length, 0);
});

test('public history projection distinguishes active, completed, aborted, and interrupted safely', () => {
  const completed = publicHistoryItem(runDocument(), DEVICE_ID, BASE_MS + 20_000);
  const aborted = publicHistoryItem(runDocument({
    runId: 'run-aborted',
    incomplete: true,
    result_outcome: 'aborted',
    result_validated: false,
    state: 'failed',
    runtime_outcome: 'error',
  }), DEVICE_ID, BASE_MS + 20_000);
  const active = publicHistoryItem(runDocument({
    runId: 'run-active',
    terminal: false,
    result_outcome: null,
    result_validated: false,
  }), DEVICE_ID, BASE_MS + 10_000);
  const interrupted = publicHistoryItem(runDocument({
    runId: 'run-interrupted',
    terminal: false,
    result_outcome: null,
    result_validated: false,
  }), DEVICE_ID, BASE_MS + 20_000);
  const unvalidatedSuccess = publicHistoryItem(runDocument({
    runId: 'run-unvalidated-success',
    result_validated: false,
  }), DEVICE_ID, BASE_MS + 20_000);

  assert.equal(completed.status, 'completed');
  assert.equal(aborted.status, 'aborted');
  assert.equal(active.status, 'active');
  assert.equal(interrupted.status, 'interrupted');
  assert.equal(unvalidatedSuccess.status, 'completed');
  assert.equal(unvalidatedSuccess.result_validated, false);
  for (const item of [completed, aborted, active, interrupted, unvalidatedSuccess]) {
    assert.equal(item.device_id, undefined);
    assert.equal(item.latest_body_sha256, undefined);
    assert.equal(item.terminal_payload, undefined);
    assert.equal(item.nonce, undefined);
    assert.equal(item.secret, undefined);
    assert.deepEqual(Object.keys(item).sort(), [
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
      'result_outcome',
      'result_validated',
      'run_id',
      'state',
      'status',
      'terminal',
    ]);
  }
});

test('a real legacy runtime-abort archive stays visible as an unknown terminal without reading its payload', () => {
  const legacy = runDocument({
    runId: 'run-legacy-abort',
    terminal_payload: validRuntimeAbortPayload({
      runId: 'run-legacy-abort',
      sequence: 9,
      failureType: 'LegacyControllerError',
    }),
  });
  for (const field of [
    'state',
    'runtime_outcome',
    'progress_percent',
    'completed_checkpoint_count',
    'checkpoint_count',
    'active_alarm_count',
    'alarm_history_count',
    'incomplete',
    'result_outcome',
    'result_validated',
  ]) delete legacy[field];

  const projected = publicHistoryItem(legacy, DEVICE_ID, BASE_MS + 20_000);
  assert.equal(projected.status, 'unknown_terminal');
  assert.equal(projected.progress_percent, null);
  assert.equal(projected.checkpoint_count, null);
  assert.equal(projected.state, null);
  assert.equal(projected.result_validated, false);
  assert.equal(projected.terminal_payload, undefined);
});

test('history projection requires UTC and epoch timestamps to agree to the millisecond', () => {
  const document = runDocument();
  document.latest_received_at_utc = new Date(
    document.latest_received_at_epoch_ms + 1,
  ).toISOString();

  assert.throws(
    () => publicHistoryItem(document, DEVICE_ID, BASE_MS + 20_000),
    /latest received timestamps disagree/,
  );
});

test('history storage failures return one fixed safe error without internal details', async () => {
  const handler = createHistoryHandler({
    env: {
      PATROL_RELAY_DEVICE_ID: DEVICE_ID,
      PATROL_RELAY_UPLOAD_SECRET: TEST_SECRET,
    },
    getWxContext: () => ({ OPENID: 'wx-openid' }),
    repository: {
      async getPatrolHistoryPage() {
        throw new Error('database password and collection internals');
      },
    },
  });

  assert.deepEqual(await handler({ page_size: 10, cursor: null }), {
    ok: false,
    error: 'HISTORY_UNAVAILABLE',
    message: 'Patrol history is temporarily unavailable',
  });
});
