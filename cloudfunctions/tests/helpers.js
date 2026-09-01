'use strict';

const crypto = require('node:crypto');

const { SIGNATURE_VERSION } = require('../shared/config');

const TEST_NOW_MS = 1_700_000_000_000;
const TEST_SECRET = 'graduation-project-test-secret-32-bytes-minimum';
const TEST_DEVICE_ID = 'macmini-m4';

function validPayload(overrides) {
  const changes = overrides || {};
  const terminal = changes.terminal ?? false;
  const completedCheckpointCount = terminal ? 1 : 0;
  return {
    schema_version: 'patrol-dashboard/v1',
    generated_at_utc: '2023-11-14T22:13:20.000Z',
    source: { mode: 'live' },
    run: {
      run_id: changes.runId || 'test-run-001',
      sequence: changes.sequence ?? 0,
      terminal: changes.terminal ?? false,
      simulation_time_s: changes.simulationTime ?? 0,
    },
    map: {
      origin: { x_m: -2.25, y_m: -1.75 },
      size: { width_m: 4.5, height_m: 3.5 },
      robot: { x_m: -1.6, y_m: 0 },
      local_goal: terminal ? null : { x_m: 1.6, y_m: 0 },
      checkpoints: [{
        id: 'P1',
        position: { x_m: 1.6, y_m: 0 },
        status: terminal ? 'completed' : 'current',
      }],
      static_obstacles: [],
      trajectory: [],
      trajectory_point_count: 0,
    },
    task: {
      current_checkpoint_id: terminal ? null : 'P1',
      completed_checkpoint_count: completedCheckpointCount,
      checkpoint_count: 1,
      progress_ratio: completedCheckpointCount,
      progress_percent: completedCheckpointCount * 100,
    },
    sensors: {
      available: false,
      checkpoint_id: null,
      sampled_at_s: null,
      temperature: { value: null, unit: '°C' },
      smoke: { value: null, unit: '%/m' },
      carbon_monoxide: { value: null, unit: 'ppm' },
    },
    alarms: {
      active_count: 0,
      active: [],
      history_count: 0,
      history_returned_count: 0,
      history_truncated: false,
      history: [],
    },
    result: terminal
      ? {
        available: true,
        validated: true,
        success: true,
        checkpoints_visited: 1,
        checkpoints_total: 1,
        collision_count: 0,
        sample_count: 3,
      }
      : { available: false, validated: false },
  };
}

function signedEvent(payload, options) {
  const config = options || {};
  const rawBody = typeof payload === 'string' ? payload : JSON.stringify(payload);
  const deviceId = config.deviceId || TEST_DEVICE_ID;
  const timestamp = config.timestamp || String(Math.floor(TEST_NOW_MS / 1000));
  const nonce = config.nonce || 'nonce-0000000000000001';
  const secret = config.secret || TEST_SECRET;
  const bodySha256 = crypto.createHash('sha256').update(rawBody, 'utf8').digest('hex');
  const canonical = `${deviceId}\n${timestamp}\n${nonce}\n${bodySha256}`;
  const signature = crypto.createHmac('sha256', secret).update(canonical, 'utf8').digest('hex');
  return {
    httpMethod: config.method || 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Patrol-Signature-Version': config.signatureVersion || SIGNATURE_VERSION,
      'X-Patrol-Device-Id': deviceId,
      'X-Patrol-Timestamp': timestamp,
      'X-Patrol-Nonce': nonce,
      'X-Patrol-Body-SHA256': config.bodySha256 || bodySha256,
      'X-Patrol-Signature': config.signature || signature,
    },
    body: rawBody,
  };
}

function testEnv() {
  return {
    PATROL_CLOUDBASE_ENV_ID: 'cloud1-5gpkygv2be4de05a',
    PATROL_RELAY_DEVICE_ID: TEST_DEVICE_ID,
    PATROL_RELAY_UPLOAD_SECRET: TEST_SECRET,
  };
}

module.exports = {
  TEST_DEVICE_ID,
  TEST_NOW_MS,
  TEST_SECRET,
  signedEvent,
  testEnv,
  validPayload,
};
