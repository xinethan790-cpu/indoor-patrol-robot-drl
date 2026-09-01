'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const test = require('node:test');

const fixture = require('../../wechat-miniprogram/data/graduation-demo-seed7000-v1.js');
const {
  validateCloudSnapshot,
} = require('../../wechat-miniprogram/services/cloudDataSource');
const {
  buildDashboardPayload,
} = require('../../wechat-miniprogram/services/offlineReplay');
const { createIngestHandler } = require('../shared/ingest-handler');
const { liveDocument } = require('../shared/cloud-repository');
const { publicSnapshot } = require('../shared/snapshot-handler');

const REPOSITORY_ROOT = path.resolve(__dirname, '../..');
const REAL_RUN_DIRECTORY = path.join(
  REPOSITORY_ROOT,
  'artifacts/results/graduation-demo-seed7000-v1',
);
const CONTRACT_SCRIPT = path.join(REPOSITORY_ROOT, 'scripts/build_patrol_contract_request.py');
const TEST_NOW_MS = 1_788_192_000_000;
const TEST_TIMESTAMP = Math.floor(TEST_NOW_MS / 1000);
const TEST_NONCE = 'offline-contract-vector-0001';
const TEST_SECRET = 'offline-contract-test-secret-at-least-32-bytes';
const EXPECTED_RUN_ID = 'graduation-demo-seed7000-v1';
const EXPECTED_SEQUENCE = 822;
const EXPECTED_EVENTS_SHA256 = 'b7977c8a3e0d196d98a5cb31cd223cc4057ac1fe3830c1424f177bd3ce70b1e1';
const EXPECTED_MANIFEST_SHA256 = '7aa30d06d1434e5cb04818dbfb8e25fe765cca411619ea98a312230b5e48db45';

function resolvePython() {
  if (process.env.PATROL_TEST_PYTHON) return process.env.PATROL_TEST_PYTHON;
  const candidates = process.platform === 'win32'
    ? [path.join(REPOSITORY_ROOT, '.venv/Scripts/python.exe')]
    : [path.join(REPOSITORY_ROOT, '.venv/bin/python')];
  const local = candidates.find((candidate) => fs.existsSync(candidate));
  if (local) return local;
  return process.platform === 'win32' ? 'python' : 'python';
}

function trackedGraduationPayload() {
  assert.equal(fixture.source.run_id, EXPECTED_RUN_ID);
  assert.equal(fixture.source.terminal_validation, 'inspection_bundle_and_runtime_consistency');
  assert.equal(fixture.source.source_event_count, 823);
  assert.equal(fixture.source.events_sha256, EXPECTED_EVENTS_SHA256);
  assert.equal(fixture.source.inspection_manifest_sha256, EXPECTED_MANIFEST_SHA256);
  assert.equal(fixture.source.qualification, 'graduation_demonstration_only');

  const payload = buildDashboardPayload(fixture, fixture.frames.length - 1);
  payload.generated_at_utc = new Date(TEST_NOW_MS).toISOString();
  assert.equal(payload.run.run_id, EXPECTED_RUN_ID);
  assert.equal(payload.run.sequence, EXPECTED_SEQUENCE);
  assert.equal(payload.run.terminal, true);
  assert.equal(payload.task.completed_checkpoint_count, 4);
  assert.equal(payload.result.collision_count, 0);
  assert.equal(payload.result.sample_count, 36);
  assert.equal(payload.alarms.history_count, 6);
  return payload;
}

function buildPythonRequest() {
  const argumentsList = [
    CONTRACT_SCRIPT,
    '--timestamp', String(TEST_TIMESTAMP),
    '--nonce', TEST_NONCE,
    '--expected-run-id', EXPECTED_RUN_ID,
    '--expected-sequence', String(EXPECTED_SEQUENCE),
  ];
  let input = '';
  let expectedSource;
  const useRepository = process.env.PATROL_CONTRACT_FORCE_TRACKED_FIXTURE !== '1'
    && fs.existsSync(path.join(REAL_RUN_DIRECTORY, 'status.json'));
  if (useRepository) {
    argumentsList.push('--run-directory', REAL_RUN_DIRECTORY);
    expectedSource = 'verified_repository_projection';
  } else {
    input = JSON.stringify(trackedGraduationPayload());
    expectedSource = 'tracked_verified_fixture_projection';
  }

  const execution = spawnSync(resolvePython(), argumentsList, {
    cwd: REPOSITORY_ROOT,
    encoding: 'utf8',
    env: {
      ...process.env,
      PATROL_RELAY_UPLOAD_SECRET: TEST_SECRET,
    },
    input,
    maxBuffer: 2 * 1024 * 1024,
    timeout: 30_000,
  });
  assert.equal(
    execution.status,
    0,
    `Python contract vector failed:\n${execution.stderr || execution.stdout}`,
  );
  assert.equal(execution.signal, null);
  const vector = JSON.parse(execution.stdout);
  assert.equal(vector.schema_version, 'patrol-relay-contract-request/v1');
  assert.equal(vector.payload_source, expectedSource);
  assert.ok(vector.body_bytes > 1_000);
  assert.ok(vector.body_bytes <= 128 * 1024);
  assert.equal(
    crypto.createHash('sha256').update(vector.event.body, 'utf8').digest('hex'),
    vector.body_sha256,
  );
  return vector;
}

test('real graduation payload crosses Python HMAC, cloud ingest, and mini protocol offline', async () => {
  const vector = buildPythonRequest();
  const committed = [];
  const repository = {
    async commitUpload(upload) {
      committed.push(upload);
      return {
        disposition: 'new_run',
        received_at_epoch_ms: upload.received_at_epoch_ms,
        received_at_utc: upload.received_at_utc,
      };
    },
  };
  const handler = createIngestHandler({
    now: () => TEST_NOW_MS,
    env: {
      PATROL_CLOUDBASE_ENV_ID: 'cloud1-5gpkygv2be4de05a',
      PATROL_RELAY_DEVICE_ID: 'macmini-m4',
      PATROL_RELAY_UPLOAD_SECRET: TEST_SECRET,
    },
    repository,
    logger: { error() {} },
  });

  const response = await handler(vector.event);
  assert.equal(response.statusCode, 200);
  const acknowledgement = JSON.parse(response.body);
  assert.deepEqual(
    {
      accepted: acknowledgement.accepted,
      device_id: acknowledgement.device_id,
      run_id: acknowledgement.run_id,
      sequence: acknowledgement.sequence,
      terminal: acknowledgement.terminal,
      disposition: acknowledgement.disposition,
    },
    {
      accepted: true,
      device_id: 'macmini-m4',
      run_id: EXPECTED_RUN_ID,
      sequence: EXPECTED_SEQUENCE,
      terminal: true,
      disposition: 'new_run',
    },
  );
  assert.equal(committed.length, 1);
  assert.equal(committed[0].body_sha256, vector.body_sha256);
  assert.equal(committed[0].payload.run.run_id, EXPECTED_RUN_ID);
  assert.equal(committed[0].payload.run.sequence, EXPECTED_SEQUENCE);
  assert.equal(committed[0].payload.run.terminal, true);

  const snapshot = publicSnapshot(liveDocument(committed[0]));
  const miniAccepted = validateCloudSnapshot(snapshot);
  assert.equal(snapshot.schema_version, 'patrol-cloud-snapshot/v1');
  assert.equal(snapshot._id, 'main');
  assert.equal(miniAccepted.snapshot, snapshot);
  assert.equal(miniAccepted.payload, committed[0].payload);
  assert.equal(miniAccepted.payload.result.success, true);
  assert.equal(miniAccepted.payload.result.collision_count, 0);
  assert.equal(miniAccepted.payload.result.sample_count, 36);

  const tampered = structuredClone(vector.event);
  const tamperedBody = JSON.parse(tampered.body);
  tamperedBody.generated_at_utc = '2099-01-01T00:00:00.000Z';
  tampered.body = JSON.stringify(tamperedBody);
  const rejected = await handler(tampered);
  assert.equal(rejected.statusCode, 401);
  assert.equal(JSON.parse(rejected.body).error, 'BODY_DIGEST_MISMATCH');
  assert.equal(committed.length, 1);
});
