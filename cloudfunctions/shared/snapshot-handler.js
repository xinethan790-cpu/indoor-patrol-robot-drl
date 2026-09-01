'use strict';

const { createDefaultRepository } = require('./cloud-repository');
const {
  LIVE_DOCUMENT_ID,
  SNAPSHOT_SCHEMA_VERSION,
  STALE_AFTER_MS,
  runtimeConfig,
} = require('./config');
const { validateDashboardPayload } = require('./payload');

function publicSnapshot(document) {
  if (!document) return null;
  return {
    _id: LIVE_DOCUMENT_ID,
    schema_version: SNAPSHOT_SCHEMA_VERSION,
    device_id: document.device_id,
    run_id: document.run_id,
    sequence: document.sequence,
    terminal: document.terminal,
    body_sha256: document.body_sha256,
    received_at_utc: document.received_at_utc,
    received_at_epoch_ms: document.received_at_epoch_ms,
    payload: validateDashboardPayload(document.payload),
  };
}

function createSnapshotHandler(options) {
  const dependencies = options || {};
  const now = dependencies.now || Date.now;
  const env = dependencies.env || process.env;

  return async function getPatrolSnapshot(event, context) {
    const config = runtimeConfig(env);
    const getWxContext = dependencies.getWxContext || (() => {
      // eslint-disable-next-line global-require, import/no-extraneous-dependencies
      const cloud = require('wx-server-sdk');
      cloud.init({ env: config.cloudEnvId || cloud.DYNAMIC_CURRENT_ENV });
      return cloud.getWXContext();
    });
    const identity = getWxContext(event, context) || {};
    if (!identity.OPENID && !identity.openid) {
      return { found: false, error: 'AUTH_REQUIRED', message: 'WeChat identity is required' };
    }

    const repository = dependencies.repository || createDefaultRepository(config);
    const document = await repository.getLiveSnapshot();
    const serverTimeEpochMs = now();
    if (!document) {
      return { found: false, server_time_epoch_ms: serverTimeEpochMs };
    }
    const snapshot = publicSnapshot(document);
    const ageMs = Math.max(0, serverTimeEpochMs - Number(snapshot.received_at_epoch_ms || 0));
    return {
      found: true,
      stale: ageMs > STALE_AFTER_MS,
      age_ms: ageMs,
      server_time_epoch_ms: serverTimeEpochMs,
      snapshot,
    };
  };
}

const main = createSnapshotHandler();

module.exports = { createSnapshotHandler, main, publicSnapshot };
