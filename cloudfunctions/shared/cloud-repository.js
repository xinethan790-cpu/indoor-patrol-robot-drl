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
const { RUN_ID_PATTERN } = require('./payload');
const { buildPatrolReport } = require('./report');
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
  const payload = upload.payload;
  const task = payload.task;
  const alarms = payload.alarms;
  const terminalResult = upload.terminal && payload.result.available === true
    ? payload.result
    : null;
  const report = buildPatrolReport(payload);
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
    state: typeof payload.run.state === 'string' ? payload.run.state : null,
    runtime_outcome: typeof payload.run.outcome === 'string' ? payload.run.outcome : null,
    progress_percent: task.progress_percent,
    completed_checkpoint_count: task.completed_checkpoint_count,
    checkpoint_count: task.checkpoint_count,
    active_alarm_count: alarms.active_count,
    alarm_history_count: alarms.history_count,
    incomplete: Boolean(terminalResult && terminalResult.incomplete === true),
    result_outcome: terminalResult && typeof terminalResult.outcome === 'string'
      ? terminalResult.outcome
      : null,
    result_validated: terminalResult ? terminalResult.validated === true : false,
    report_available: report !== null,
  };
  if (report !== null) result.report = report;
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
      await transaction.collection(COLLECTIONS.runs).doc(runId).set({
        data: runDocument(upload, runStartedAt),
      });
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

  async getPatrolHistoryPage(options) {
    const settings = options || {};
    const { deviceId, pageSize, offset } = settings;
    if (typeof deviceId !== 'string' || deviceId.length === 0 || deviceId.length > 64) {
      throw new TypeError('history deviceId must be a non-empty string');
    }
    if (!Number.isSafeInteger(pageSize) || pageSize < 1 || pageSize > 20) {
      throw new TypeError('history pageSize must be an integer between 1 and 20');
    }
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > 1_000_000) {
      throw new TypeError('history offset must be a bounded non-negative integer');
    }

    const result = await this.database.collection(COLLECTIONS.runs)
      .where({ device_id: deviceId })
      .orderBy('first_received_at_epoch_ms', 'desc')
      .orderBy('run_id', 'desc')
      .skip(offset)
      .limit(pageSize + 1)
      .field({
        schema_version: true,
        device_id: true,
        run_id: true,
        first_received_at_utc: true,
        first_received_at_epoch_ms: true,
        latest_sequence: true,
        latest_received_at_utc: true,
        latest_received_at_epoch_ms: true,
        terminal: true,
        state: true,
        runtime_outcome: true,
        progress_percent: true,
        completed_checkpoint_count: true,
        checkpoint_count: true,
        active_alarm_count: true,
        alarm_history_count: true,
        incomplete: true,
        result_outcome: true,
        result_validated: true,
        report_available: true,
      })
      .get();
    if (!result || !Array.isArray(result.data)) {
      throw new TypeError('history query returned an invalid data envelope');
    }
    const rows = result.data;
    return {
      documents: rows.slice(0, pageSize),
      hasMore: rows.length > pageSize,
      nextOffset: offset + Math.min(rows.length, pageSize),
    };
  }

  async getPatrolReport(options) {
    const settings = options || {};
    const { deviceId, runId } = settings;
    if (typeof deviceId !== 'string' || deviceId.length === 0 || deviceId.length > 64) {
      throw new TypeError('report deviceId must be a non-empty string');
    }
    if (typeof runId !== 'string' || !RUN_ID_PATTERN.test(runId)) {
      throw new TypeError('report runId is invalid');
    }
    const document = await readOptional(
      this.database.collection(COLLECTIONS.runs).doc(opaqueId(deviceId, runId)),
    );
    if (!document
        || document.device_id !== deviceId
        || document.run_id !== runId
        || document.terminal !== true
        || document.report_available !== true
        || !document.report) {
      return null;
    }
    return document.report;
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
  runDocument,
};
