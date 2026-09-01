'use strict';

const defaultFixture = require('../data/graduation-demo-seed7000-v1.js');
const { createCloudDataSource } = require('./cloudDataSource');
const { createOfflineReplay } = require('./offlineReplay');
const { createPollingClient } = require('./pollingClient');

function createOfflineDataSource(options) {
  return createOfflineReplay({ fixture: defaultFixture, ...(options || {}) });
}

function createPollingDataSource(options) {
  const config = { ...(options || {}) };
  if (!config.endpoint && config.apiBaseUrl) {
    config.endpoint = `${String(config.apiBaseUrl).replace(/\/+$/, '')}/api/dashboard`;
  }
  return createPollingClient(config);
}

function createCloudPatrolDataSource(options) {
  return createCloudDataSource(options || {});
}

function createDataSource(options) {
  const config = options || {};
  if (['offline', 'demo'].includes(config.mode || 'offline')) return createOfflineDataSource(config);
  if (config.mode === 'local') return createPollingDataSource(config);
  if (config.mode === 'cloud') return createCloudPatrolDataSource(config);
  throw new RangeError(`未知数据源模式：${String(config.mode)}`);
}

module.exports = {
  createDataSource,
  createCloudPatrolDataSource,
  createOfflineDataSource,
  createPollingDataSource,
};
