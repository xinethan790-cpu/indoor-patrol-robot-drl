'use strict';

const crypto = require('node:crypto');

const {
  COLLECTIONS,
  LIVE_DOCUMENT_ID,
  NONCE_RETENTION_MS,
  NONCE_SCHEMA_VERSION,
  RELAY_STATE_SCHEMA_VERSION,
  RUN_SCHEMA_VERSION,
  SNAPSHOT_SCHEMA_VERSION,
} = require('./config');
const { conflict } = require('./errors');
const { decideUpload } = require('./state-machine');

function opaqueId(...parts) {
  return crypto.createHash('sha256').update(parts.join('\n'), 'utf8').digest('hex');
}

function documentMissing(error) {
  if (!error) return false;
  const token = `${error.errCode || ''} ${error.code || ''} ${error.message || ''}`.toLowerCase();
  return token.includes('document_not_exist')
    || token.includes('document not exist')
    || token.includes('does not exist')
    || token.includes('not found');
}

async function readOptional(documentReference) {
  try {
    const result = await documentReference.get();
    return result && result.data ? result.data : null;
  } catch (error) {
    if (documentMissing(error)) return null;
    throw error;
  }
}

function liveDocument(upload) {
  return {
    schema_version: SNAPSHOT_SCHEMA_VERSION,
    device_id: upload.device_id,
    run_id: upload.run_id,
    sequence: upload.sequence,
    terminal: upload.terminal,
    body_sha256: upload.body_sha256,
    received_at_utc: upload.received_at_utc,
    received_at_epoch_ms: upload.received_at_epoch_ms,
    payload: upload.payload,
  };
}

function stateDocument(upload, runStartedAt) {
  return {
    schema_version: RELAY_STATE_SCHEMA_VERSION,
    device_id: upload.device_id,
    run_id: upload.run_id,
    sequence: upload.sequence,
    terminal: upload.terminal,
    body_sha256: upload.body_sha256,
    run_started_at_utc: runStartedAt.utc,
    run_started_at_epoch_ms: runStartedAt.epochMs,
    received_at_utc: upload.received_at_utc,
    received_at_epoch_ms: upload.received_at_epoch_ms,
  };
}

function runDocument(upload, runStartedAt) {
  const result = {
    schema_version: RUN_SCHEMA_VERSION,
    device_id: upload.device_id,
    run_id: upload.run_id,
    first_received_at_utc: runStartedAt.utc,
    first_received_at_epoch_ms: runStartedAt.epochMs,
    latest_sequence: upload.sequence,
    terminal: upload.terminal,
    latest_body_sha256: upload.body_sha256,
    latest_received_at_utc: upload.received_at_utc,
    latest_received_at_epoch_ms: upload.received_at_epoch_ms,
  };
  if (upload.terminal) {
    result.terminal_payload = upload.payload;
    result.terminal_received_at_utc = upload.received_at_utc;
    result.terminal_received_at_epoch_ms = upload.received_at_epoch_ms;
  }
  return result;
}

function nonceDocument(upload) {
  return {
    schema_version: NONCE_SCHEMA_VERSION,
    device_id: upload.device_id,
    nonce_sha256: opaqueId(upload.device_id, upload.nonce),
    request_timestamp_epoch_s: upload.request_timestamp_epoch_s,
    received_at_utc: upload.received_at_utc,
    received_at_epoch_ms: upload.received_at_epoch_ms,
    expires_at_epoch_ms: upload.received_at_epoch_ms + NONCE_RETENTION_MS,
    expires_at: new Date(upload.received_at_epoch_ms + NONCE_RETENTION_MS),
  };
}

class CloudPatrolRepository {
  constructor(database) {
    if (!database || typeof database.runTransaction !== 'function') {
      throw new TypeError('CloudPatrolRepository requires a transaction-capable database');
    }
    this.database = database;
  }

  async commitUpload(upload) {
    const db = this.database;
    const stateId = opaqueId(upload.device_id);
    const nonceId = opaqueId(upload.device_id, upload.nonce);
    const runId = opaqueId(upload.device_id, upload.run_id);

    return db.runTransaction(async (transaction) => {
      const nonceRef = transaction.collection(COLLECTIONS.nonces).doc(nonceId);
      if (await readOptional(nonceRef)) {
        throw conflict('NONCE_REPLAY', 'nonce was already used');
      }

      const stateRef = transaction.collection(COLLECTIONS.relayState).doc(stateId);
      const current = await readOptional(stateRef);
      let registeredRun = null;
      if (!current || current.run_id !== upload.run_id) {
        registeredRun = await readOptional(transaction.collection(COLLECTIONS.runs).doc(runId));
      }
      const decision = decideUpload(current, upload, Boolean(registeredRun));

      await nonceRef.set({ data: nonceDocument(upload) });
      if (!decision.writeSnapshot) {
        return {
          disposition: decision.disposition,
          received_at_epoch_ms: current.received_at_epoch_ms,
          received_at_utc: current.received_at_utc,
        };
      }

      const runStartedAt = decision.disposition === 'new_run'
        ? { utc: upload.received_at_utc, epochMs: upload.received_at_epoch_ms }
        : {
          utc: current.run_started_at_utc,
          epochMs: current.run_started_at_epoch_ms,
        };
      await stateRef.set({ data: stateDocument(upload, runStartedAt) });
      await transaction.collection(COLLECTIONS.live).doc(LIVE_DOCUMENT_ID).set({
        data: liveDocument(upload),
      });
      if (decision.disposition === 'new_run' || upload.terminal) {
        await transaction.collection(COLLECTIONS.runs).doc(runId).set({
          data: runDocument(upload, runStartedAt),
        });
      }
      return {
        disposition: decision.disposition,
        received_at_epoch_ms: upload.received_at_epoch_ms,
        received_at_utc: upload.received_at_utc,
      };
    });
  }

  async getLiveSnapshot() {
    return readOptional(this.database.collection(COLLECTIONS.live).doc(LIVE_DOCUMENT_ID));
  }
}

let defaultRepository;

function createDefaultRepository(config) {
  if (defaultRepository) return defaultRepository;
  // Loaded lazily so the pure validation and state-machine tests need no cloud SDK.
  // eslint-disable-next-line global-require, import/no-extraneous-dependencies
  const cloud = require('wx-server-sdk');
  cloud.init({ env: config.cloudEnvId || cloud.DYNAMIC_CURRENT_ENV });
  defaultRepository = new CloudPatrolRepository(cloud.database());
  return defaultRepository;
}

module.exports = {
  CloudPatrolRepository,
  createDefaultRepository,
  liveDocument,
  opaqueId,
  readOptional,
};
