'use strict';

const CLOUD_ENV_ID = 'cloud1-5gpkygv2be4de05a';
const DASHBOARD_SCHEMA_VERSION = 'patrol-dashboard/v1';
const SNAPSHOT_SCHEMA_VERSION = 'patrol-cloud-snapshot/v1';
const RELAY_STATE_SCHEMA_VERSION = 'patrol-relay-state/v1';
const RUN_SCHEMA_VERSION = 'patrol-cloud-run/v1';
const HISTORY_SCHEMA_VERSION = 'patrol-history/v1';
const NONCE_SCHEMA_VERSION = 'patrol-relay-nonce/v1';
const SIGNATURE_VERSION = 'patrol-hmac-sha256-v1';

const DEFAULT_DEVICE_ID = 'macmini-m4';
const MAX_BODY_BYTES = 128 * 1024;
const MAX_CLOCK_SKEW_SECONDS = 300;
const NONCE_RETENTION_MS = 24 * 60 * 60 * 1000;
const STALE_AFTER_MS = 8 * 1000;
const DEFAULT_HISTORY_PAGE_SIZE = 10;
const MAX_HISTORY_PAGE_SIZE = 20;
const MIN_UPLOAD_SECRET_BYTES = 32;

const SECRET_TEMPLATE_PATTERN = /(?:\{\{[^{}\r\n]+\}\}|\$\{[^{}\r\n]+\})/;
const EXAMPLE_SECRET_PATTERN = /(?:change.?me|replace.?me|example|sample|default|placeholder|your.?secret|your.?key|test.?secret|demo.?secret|password|not.?for.?production)/i;

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

function inspectUploadSecret(secret) {
  if (typeof secret !== 'string' || secret.length === 0) {
    return Object.freeze({
      ready: false,
      code: 'MISSING_SECRET',
      reason: 'upload secret is not set',
    });
  }
  const normalized = secret.trim();
  if (secret !== normalized) {
    return Object.freeze({
      ready: false,
      code: 'WHITESPACE_SECRET',
      reason: 'upload secret must not contain leading or trailing whitespace',
    });
  }
  if (SECRET_TEMPLATE_PATTERN.test(normalized)) {
    return Object.freeze({
      ready: false,
      code: 'TEMPLATE_SECRET',
      reason: 'upload secret still contains a template expression',
    });
  }
  if (Buffer.byteLength(normalized, 'utf8') < MIN_UPLOAD_SECRET_BYTES) {
    return Object.freeze({
      ready: false,
      code: 'SHORT_SECRET',
      reason: `upload secret must contain at least ${MIN_UPLOAD_SECRET_BYTES} UTF-8 bytes`,
    });
  }
  const uniqueCharacters = new Set([...normalized]).size;
  if (
    normalized.length === 0
    || EXAMPLE_SECRET_PATTERN.test(normalized)
    || uniqueCharacters < 8
  ) {
    return Object.freeze({
      ready: false,
      code: 'LOW_QUALITY_SECRET',
      reason: 'upload secret matches a sample, default, or low-diversity value',
    });
  }
  return Object.freeze({
    ready: true,
    code: 'READY',
    reason: 'upload secret passed local readiness checks',
  });
}

function requireUploadSecret(secret) {
  const inspection = inspectUploadSecret(secret);
  if (!inspection.ready) {
    throw new Error(`PATROL_RELAY_UPLOAD_SECRET is not ready: ${inspection.reason}`);
  }
  return secret;
}

module.exports = {
  CLOUD_ENV_ID,
  COLLECTIONS,
  DASHBOARD_SCHEMA_VERSION,
  DEFAULT_HISTORY_PAGE_SIZE,
  DEFAULT_DEVICE_ID,
  LIVE_DOCUMENT_ID,
  HISTORY_SCHEMA_VERSION,
  MAX_BODY_BYTES,
  MAX_CLOCK_SKEW_SECONDS,
  MAX_HISTORY_PAGE_SIZE,
  MIN_UPLOAD_SECRET_BYTES,
  NONCE_RETENTION_MS,
  NONCE_SCHEMA_VERSION,
  RELAY_STATE_SCHEMA_VERSION,
  RUN_SCHEMA_VERSION,
  SIGNATURE_VERSION,
  SNAPSHOT_SCHEMA_VERSION,
  STALE_AFTER_MS,
  inspectUploadSecret,
  requireUploadSecret,
  runtimeConfig,
};
