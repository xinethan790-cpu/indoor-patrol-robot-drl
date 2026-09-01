'use strict';

const { COLLECTIONS } = require('./config');

// CloudBase queries can return more rows than expected in mocks or after SDK changes.
// Keep this application-side cap in addition to query.limit() so one timer invocation
// can never issue an unbounded number of deletes.
const DELETE_BATCH_LIMIT = 100;
const DELETE_CONCURRENCY = 10;

function assertDatabase(database) {
  if (!database || typeof database.collection !== 'function') {
    throw new TypeError('cleanupPatrolNonces requires a CloudBase database');
  }
  if (!database.command || typeof database.command.lt !== 'function') {
    throw new TypeError('cleanupPatrolNonces requires database.command.lt');
  }
}

function normaliseCandidates(result) {
  if (!result || !Array.isArray(result.data)) return [];
  return result.data.slice(0, DELETE_BATCH_LIMIT);
}

async function cleanupExpiredNonces(database, nowMs) {
  assertDatabase(database);
  if (!Number.isSafeInteger(nowMs) || nowMs < 0) {
    throw new TypeError('cleanupPatrolNonces requires a non-negative epoch millisecond');
  }

  const collection = database.collection(COLLECTIONS.nonces);
  const result = await collection
    .where({ expires_at_epoch_ms: database.command.lt(nowMs) })
    .field({ _id: true })
    .limit(DELETE_BATCH_LIMIT)
    .get();
  const candidates = normaliseCandidates(result);

  let deleted = 0;
  for (let offset = 0; offset < candidates.length; offset += DELETE_CONCURRENCY) {
    const batch = candidates.slice(offset, offset + DELETE_CONCURRENCY);
    const results = await Promise.all(batch.map(async (candidate) => {
      if (!candidate || typeof candidate._id !== 'string' || candidate._id.length === 0) {
        return 0;
      }
      const response = await collection.doc(candidate._id).remove();
      const removed = response && response.stats && Number(response.stats.removed);
      return Number.isFinite(removed) ? Math.max(0, Math.min(1, removed)) : 0;
    }));
    deleted += results.reduce((total, count) => total + count, 0);
  }

  return {
    ok: true,
    cutoff_epoch_ms: nowMs,
    batch_limit: DELETE_BATCH_LIMIT,
    matched: candidates.length,
    deleted,
    // A full page may or may not have another row. Reporting true is conservative;
    // the next scheduled (or manual admin) invocation is safe and idempotent.
    has_more: candidates.length === DELETE_BATCH_LIMIT,
  };
}

function createCleanupHandler(options) {
  const settings = options || {};
  const now = settings.now || Date.now;
  let database = settings.database || null;

  return async function cleanupPatrolNonces() {
    if (!database) {
      // Loaded lazily so unit tests and bundle verification do not need cloud credentials.
      // eslint-disable-next-line global-require, import/no-extraneous-dependencies
      const cloud = require('wx-server-sdk');
      cloud.init({
        env: process.env.PATROL_CLOUDBASE_ENV_ID || cloud.DYNAMIC_CURRENT_ENV,
      });
      database = cloud.database();
    }
    return cleanupExpiredNonces(database, now());
  };
}

const main = createCleanupHandler();

module.exports = {
  DELETE_BATCH_LIMIT,
  DELETE_CONCURRENCY,
  cleanupExpiredNonces,
  createCleanupHandler,
  main,
};
