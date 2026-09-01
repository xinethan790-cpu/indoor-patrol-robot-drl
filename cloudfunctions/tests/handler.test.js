'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { conflict } = require('../shared/errors');
const { createIngestHandler } = require('../shared/ingest-handler');
const { createSnapshotHandler } = require('../shared/snapshot-handler');
const {
  TEST_NOW_MS,
  signedEvent,
  testEnv,
  validPayload,
} = require('./helpers');

test('ingest success returns the sidecar acknowledgement contract', async () => {
  let committed;
  const handler = createIngestHandler({
    env: testEnv(),
    now: () => TEST_NOW_MS,
    repository: {
      async commitUpload(upload) {
        committed = upload;
        return { disposition: 'new_run', received_at_utc: upload.received_at_utc };
      },
    },
  });
  const response = await handler(signedEvent(validPayload({ sequence: 7, terminal: true })));
  assert.equal(response.statusCode, 200);
  const body = JSON.parse(response.body);
  assert.deepEqual(
    {
      accepted: body.accepted,
      device_id: body.device_id,
      run_id: body.run_id,
      sequence: body.sequence,
      terminal: body.terminal,
    },
    {
      accepted: true,
      device_id: 'macmini-m4',
      run_id: 'test-run-001',
      sequence: 7,
      terminal: true,
    },
  );
  assert.equal(body.disposition, 'new_run');
  assert.equal(committed.request_timestamp_epoch_s, 1_700_000_000);
  assert.equal(committed.received_at_utc, '2023-11-14T22:13:20.000Z');
});

test('new cloud accepts a signed legacy outbox body and commits compatibility defaults', async () => {
  const legacy = validPayload({ sequence: 3 });
  delete legacy.run.control_step;
  delete legacy.map.dynamic_obstacles;
  delete legacy.avoidance;
  let committed;
  const handler = createIngestHandler({
    env: testEnv(),
    now: () => TEST_NOW_MS,
    repository: {
      async commitUpload(upload) {
        committed = upload;
        return { disposition: 'new_run', received_at_utc: upload.received_at_utc };
      },
    },
  });

  const response = await handler(signedEvent(legacy));

  assert.equal(response.statusCode, 200);
  assert.equal(committed.payload.run.control_step, null);
  assert.deepEqual(committed.payload.map.dynamic_obstacles, []);
  assert.equal(committed.payload.avoidance.available, false);
  assert.equal(Object.hasOwn(legacy.run, 'control_step'), false);
});

test('ingest maps protocol and repository conflicts to safe HTTP errors', async () => {
  const protocolHandler = createIngestHandler({
    env: testEnv(),
    now: () => TEST_NOW_MS,
    repository: { commitUpload: async () => ({}) },
  });
  const invalid = validPayload();
  invalid.run.sequence = -1;
  const invalidResponse = await protocolHandler(signedEvent(invalid));
  assert.equal(invalidResponse.statusCode, 400);
  assert.equal(JSON.parse(invalidResponse.body).error, 'INVALID_SEQUENCE');

  const conflictHandler = createIngestHandler({
    env: testEnv(),
    now: () => TEST_NOW_MS,
    repository: {
      async commitUpload() {
        throw conflict('NONCE_REPLAY', 'nonce was already used');
      },
    },
  });
  const conflictResponse = await conflictHandler(signedEvent(validPayload()));
  assert.equal(conflictResponse.statusCode, 409);
  assert.equal(JSON.parse(conflictResponse.body).error, 'NONCE_REPLAY');
});

test('ingest does not disclose unexpected internal error details', async () => {
  const logged = [];
  const handler = createIngestHandler({
    env: testEnv(),
    now: () => TEST_NOW_MS,
    logger: { error: (...args) => logged.push(args) },
    repository: {
      async commitUpload() {
        throw new Error('database credential: should-not-leak');
      },
    },
  });
  const response = await handler(signedEvent(validPayload()));
  assert.equal(response.statusCode, 500);
  assert.equal(response.body.includes('should-not-leak'), false);
  assert.equal(logged.length, 1);
  assert.equal(JSON.stringify(logged).includes('should-not-leak'), false);
});

test('ingest fails closed without logging an invalid configured secret', async () => {
  const unsafeSecret = 'your-example-default-secret-value-123456';
  const logged = [];
  const handler = createIngestHandler({
    env: {
      ...testEnv(),
      PATROL_RELAY_UPLOAD_SECRET: unsafeSecret,
    },
    logger: { error: (...args) => logged.push(args) },
    now: () => TEST_NOW_MS,
    repository: { commitUpload: async () => ({}) },
  });
  const response = await handler(signedEvent(validPayload(), { secret: unsafeSecret }));
  assert.equal(response.statusCode, 500);
  assert.equal(response.body.includes(unsafeSecret), false);
  assert.equal(JSON.stringify(logged).includes(unsafeSecret), false);
});

test('snapshot requires WeChat identity and reports freshness', async () => {
  const document = {
    schema_version: 'patrol-cloud-snapshot/v1',
    device_id: 'macmini-m4',
    run_id: 'test-run-001',
    sequence: 9,
    terminal: false,
    body_sha256: 'a'.repeat(64),
    received_at_utc: '2023-11-14T22:13:15.000Z',
    received_at_epoch_ms: TEST_NOW_MS - 5_000,
    payload: validPayload({ sequence: 9 }),
    private_field: 'not-returned',
  };
  const repository = { getLiveSnapshot: async () => document };
  const unauthorized = createSnapshotHandler({
    getWxContext: () => ({}),
    now: () => TEST_NOW_MS,
    repository,
  });
  assert.equal((await unauthorized()).error, 'AUTH_REQUIRED');

  const handler = createSnapshotHandler({
    getWxContext: () => ({ OPENID: 'test-openid' }),
    now: () => TEST_NOW_MS,
    repository,
  });
  const result = await handler();
  assert.equal(result.found, true);
  assert.equal(result.stale, false);
  assert.equal(result.age_ms, 5_000);
  assert.equal(result.snapshot.private_field, undefined);
  assert.equal(result.snapshot._id, 'main');
  assert.equal(result.snapshot.payload.run.sequence, 9);
});

test('snapshot reports empty and stale states without fabricating data', async () => {
  const empty = createSnapshotHandler({
    getWxContext: () => ({ OPENID: 'test-openid' }),
    now: () => TEST_NOW_MS,
    repository: { getLiveSnapshot: async () => null },
  });
  assert.deepEqual(await empty(), { found: false, server_time_epoch_ms: TEST_NOW_MS });

  const stale = createSnapshotHandler({
    getWxContext: () => ({ OPENID: 'test-openid' }),
    now: () => TEST_NOW_MS,
    repository: {
      getLiveSnapshot: async () => ({
        device_id: 'macmini-m4',
        run_id: 'run',
        sequence: 1,
        terminal: false,
        body_sha256: 'a'.repeat(64),
        received_at_utc: '2023-11-14T22:13:00.000Z',
        received_at_epoch_ms: TEST_NOW_MS - 10_001,
        payload: validPayload(),
      }),
    },
  });
  assert.equal((await stale()).stale, true);
});

test('snapshot fallback normalizes a legacy stored cloud payload', async () => {
  const payload = validPayload({ sequence: 4 });
  delete payload.run.control_step;
  delete payload.map.dynamic_obstacles;
  delete payload.avoidance;
  const handler = createSnapshotHandler({
    getWxContext: () => ({ OPENID: 'test-openid' }),
    now: () => TEST_NOW_MS,
    repository: {
      getLiveSnapshot: async () => ({
        device_id: 'macmini-m4',
        run_id: payload.run.run_id,
        sequence: payload.run.sequence,
        terminal: payload.run.terminal,
        body_sha256: 'a'.repeat(64),
        received_at_utc: new Date(TEST_NOW_MS).toISOString(),
        received_at_epoch_ms: TEST_NOW_MS,
        payload,
      }),
    },
  });

  const result = await handler();

  assert.equal(result.snapshot.payload.run.control_step, null);
  assert.deepEqual(result.snapshot.payload.map.dynamic_obstacles, []);
  assert.equal(result.snapshot.payload.avoidance.available, false);
});
