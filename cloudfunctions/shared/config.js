'use strict';

const CLOUD_ENV_ID = 'cloud1-5gpkygv2be4de05a';
const DASHBOARD_SCHEMA_VERSION = 'patrol-dashboard/v1';
const SNAPSHOT_SCHEMA_VERSION = 'patrol-cloud-snapshot/v1';
const RELAY_STATE_SCHEMA_VERSION = 'patrol-relay-state/v1';
const RUN_SCHEMA_VERSION = 'patrol-cloud-run/v1';
const NONCE_SCHEMA_VERSION = 'patrol-relay-nonce/v1';
const SIGNATURE_VERSION = 'patrol-hmac-sha256-v1';

const DEFAULT_DEVICE_ID = 'macmini-m4';
const MAX_BODY_BYTES = 128 * 1024;
const MAX_CLOCK_SKEW_SECONDS = 300;
const NONCE_RETENTION_MS = 24 * 60 * 60 * 1000;
const STALE_AFTER_MS = 8 * 1000;

const COLLECTIONS = Object.freeze({
  live: 'patrol_live',
  relayState: 'patrol_relay_state',
  nonces: 'patrol_relay_nonces',
  runs: 'patrol_runs',
});

const LIVE_DOCUMENT_ID = 'main';

function runtimeConfig(env) {
  const source = env || process.env;
  return Object.freeze({
    cloudEnvId: source.PATROL_CLOUDBASE_ENV_ID || CLOUD_ENV_ID,
    uploadSecret: source.PATROL_RELAY_UPLOAD_SECRET || '',
    deviceId: source.PATROL_RELAY_DEVICE_ID || DEFAULT_DEVICE_ID,
  });
}

module.exports = {
  CLOUD_ENV_ID,
  COLLECTIONS,
  DASHBOARD_SCHEMA_VERSION,
  DEFAULT_DEVICE_ID,
  LIVE_DOCUMENT_ID,
  MAX_BODY_BYTES,
  MAX_CLOCK_SKEW_SECONDS,
  NONCE_RETENTION_MS,
  NONCE_SCHEMA_VERSION,
  RELAY_STATE_SCHEMA_VERSION,
  RUN_SCHEMA_VERSION,
  SIGNATURE_VERSION,
  SNAPSHOT_SCHEMA_VERSION,
  STALE_AFTER_MS,
  runtimeConfig,
};
