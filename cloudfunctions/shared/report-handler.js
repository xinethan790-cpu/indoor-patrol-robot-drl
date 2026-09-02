'use strict';

const { createDefaultRepository } = require('./cloud-repository');
const {
  REPORT_RESPONSE_SCHEMA_VERSION,
  runtimeConfig,
} = require('./config');
const { RUN_ID_PATTERN } = require('./payload');
const { validatePatrolReport } = require('./report');

class ReportRequestError extends Error {}

function parseReportRequest(event) {
  if (!event || typeof event !== 'object' || Array.isArray(event)) {
    throw new ReportRequestError('report request must be an object');
  }
  const keys = Object.keys(event);
  if (keys.length !== 1 || keys[0] !== 'run_id') {
    throw new ReportRequestError('report request contains unsupported or missing fields');
  }
  if (typeof event.run_id !== 'string' || !RUN_ID_PATTERN.test(event.run_id)) {
    throw new ReportRequestError('run_id is invalid');
  }
  return { runId: event.run_id };
}

function unavailableResponse() {
  return {
    ok: false,
    error: 'REPORT_UNAVAILABLE',
    message: 'Patrol report is temporarily unavailable',
  };
}

function createReportHandler(options) {
  const dependencies = options || {};
  const env = dependencies.env || process.env;

  return async function getPatrolReport(event, context) {
    let config;
    let identity;
    try {
      config = runtimeConfig(env);
      const getWxContext = dependencies.getWxContext || (() => {
        // eslint-disable-next-line global-require, import/no-extraneous-dependencies
        const cloud = require('wx-server-sdk');
        cloud.init({ env: config.cloudEnvId || cloud.DYNAMIC_CURRENT_ENV });
        return cloud.getWXContext();
      });
      identity = getWxContext(event, context) || {};
    } catch (error) {
      return unavailableResponse();
    }
    if (!identity.OPENID && !identity.openid) {
      return { ok: false, error: 'AUTH_REQUIRED', message: 'WeChat identity is required' };
    }

    let request;
    try {
      request = parseReportRequest(event);
    } catch (error) {
      if (error instanceof ReportRequestError) {
        return {
          ok: false,
          error: 'INVALID_REPORT_REQUEST',
          message: error.message,
        };
      }
      return unavailableResponse();
    }

    try {
      const repository = dependencies.repository || createDefaultRepository(config);
      const stored = await repository.getPatrolReport({
        deviceId: config.deviceId,
        runId: request.runId,
      });
      if (!stored) return unavailableResponse();
      const report = validatePatrolReport(stored, request.runId);
      return {
        ok: true,
        schema_version: REPORT_RESPONSE_SCHEMA_VERSION,
        device_id: config.deviceId,
        report,
      };
    } catch (error) {
      return unavailableResponse();
    }
  };
}

const main = createReportHandler();

module.exports = {
  ReportRequestError,
  createReportHandler,
  main,
  parseReportRequest,
};
