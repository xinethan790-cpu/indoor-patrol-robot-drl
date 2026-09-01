'use strict';

const { authenticateUpload } = require('./auth');
const { createDefaultRepository } = require('./cloud-repository');
const { runtimeConfig } = require('./config');
const { PatrolCloudError } = require('./errors');
const { jsonResponse } = require('./http-response');
const { parseDashboardBody } = require('./payload');

function createIngestHandler(options) {
  const dependencies = options || {};
  const now = dependencies.now || Date.now;
  const logger = dependencies.logger || console;
  const env = dependencies.env || process.env;

  return async function ingestPatrol(event) {
    try {
      const receivedAtEpochMs = now();
      const config = runtimeConfig(env);
      const authenticated = authenticateUpload(event, config, receivedAtEpochMs);
      const payload = parseDashboardBody(authenticated.rawBody);
      const repository = dependencies.repository || createDefaultRepository(config);
      const upload = {
        body_sha256: authenticated.bodySha256,
        device_id: authenticated.deviceId,
        nonce: authenticated.nonce,
        payload,
        received_at_epoch_ms: receivedAtEpochMs,
        received_at_utc: new Date(receivedAtEpochMs).toISOString(),
        request_timestamp_epoch_s: authenticated.requestTimestampEpochS,
        run_id: payload.run.run_id,
        sequence: payload.run.sequence,
        terminal: payload.run.terminal,
      };
      const committed = await repository.commitUpload(upload);
      return jsonResponse(200, {
        accepted: true,
        device_id: upload.device_id,
        run_id: upload.run_id,
        sequence: upload.sequence,
        terminal: upload.terminal,
        disposition: committed.disposition,
        received_at_utc: committed.received_at_utc,
      });
    } catch (error) {
      if (error instanceof PatrolCloudError) {
        return jsonResponse(error.statusCode, {
          accepted: false,
          error: error.code,
          message: error.message,
        });
      }
      if (logger && typeof logger.error === 'function') {
        logger.error('ingestPatrol internal failure', {
          name: error && error.name,
        });
      }
      return jsonResponse(500, {
        accepted: false,
        error: 'INTERNAL_ERROR',
        message: 'upload could not be committed',
      });
    }
  };
}

const main = createIngestHandler();

module.exports = { createIngestHandler, main };
