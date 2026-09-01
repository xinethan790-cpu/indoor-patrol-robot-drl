'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  DELETE_BATCH_LIMIT,
  cleanupExpiredNonces,
  createCleanupHandler,
} = require('../shared/cleanup-handler');

class MemoryNonceCollection {
  constructor(database) {
    this.database = database;
    this.cutoff = null;
    this.queryLimit = null;
  }

  where(condition) {
    assert.equal(condition.expires_at_epoch_ms.operator, 'lt');
    this.cutoff = condition.expires_at_epoch_ms.value;
    return this;
  }

  field(projection) {
    assert.deepEqual(projection, { _id: true });
    return this;
  }

  limit(value) {
    this.queryLimit = value;
    this.database.queryLimits.push(value);
    return this;
  }

  async get() {
    const matches = [...this.database.documents.values()]
      .filter((document) => document.expires_at_epoch_ms < this.cutoff)
      .sort((left, right) => left._id.localeCompare(right._id))
      .map((document) => ({ _id: document._id }));
    return {
      data: this.database.overReturn ? matches : matches.slice(0, this.queryLimit),
    };
  }

  doc(id) {
    return {
      remove: async () => {
        const removed = this.database.documents.delete(id) ? 1 : 0;
        this.database.deletedIds.push(id);
        return { stats: { removed } };
      },
    };
  }
}

class MemoryDatabase {
  constructor(documents, options) {
    this.documents = new Map(documents.map((document) => [document._id, document]));
    this.overReturn = Boolean(options && options.overReturn);
    this.deletedIds = [];
    this.queryLimits = [];
    this.command = {
      lt: (value) => ({ operator: 'lt', value }),
    };
  }

  collection(name) {
    assert.equal(name, 'patrol_relay_nonces');
    return new MemoryNonceCollection(this);
  }
}

function nonceDocuments(count, expiresAt, prefix) {
  return Array.from({ length: count }, (_, index) => ({
    _id: `${prefix}-${String(index).padStart(3, '0')}`,
    expires_at_epoch_ms: expiresAt,
  }));
}

test('minute timer cleanup deletes only expired nonces in bounded repeatable batches', async () => {
  const nowMs = 1_700_000_000_000;
  const database = new MemoryDatabase([
    ...nonceDocuments(103, nowMs - 1, 'expired'),
    { _id: 'at-cutoff', expires_at_epoch_ms: nowMs },
    { _id: 'future', expires_at_epoch_ms: nowMs + 1 },
  ]);
  const handler = createCleanupHandler({ database, now: () => nowMs });

  const first = await handler({ limit: 10_000 });
  assert.deepEqual(first, {
    ok: true,
    cutoff_epoch_ms: nowMs,
    batch_limit: DELETE_BATCH_LIMIT,
    matched: 100,
    deleted: 100,
    has_more: true,
  });
  assert.equal(database.deletedIds.length, 100);
  assert.deepEqual(database.queryLimits, [DELETE_BATCH_LIMIT]);
  assert.equal(database.documents.has('at-cutoff'), true);
  assert.equal(database.documents.has('future'), true);

  const second = await handler();
  assert.equal(second.matched, 3);
  assert.equal(second.deleted, 3);
  assert.equal(second.has_more, false);

  const third = await handler();
  assert.equal(third.matched, 0);
  assert.equal(third.deleted, 0);
  assert.equal(third.has_more, false);
  assert.deepEqual([...database.documents.keys()].sort(), ['at-cutoff', 'future']);
});

test('application-side cap remains strict even if an SDK returns too many rows', async () => {
  const nowMs = 1_700_000_000_000;
  const database = new MemoryDatabase(
    nonceDocuments(150, nowMs - 1, 'expired'),
    { overReturn: true },
  );

  const result = await cleanupExpiredNonces(database, nowMs);
  assert.equal(result.matched, DELETE_BATCH_LIMIT);
  assert.equal(result.deleted, DELETE_BATCH_LIMIT);
  assert.equal(database.deletedIds.length, DELETE_BATCH_LIMIT);
  assert.equal(database.documents.size, 50);
});

test('cleanup rejects an invalid clock value before issuing a query', async () => {
  const database = new MemoryDatabase([]);
  await assert.rejects(
    cleanupExpiredNonces(database, Number.NaN),
    /non-negative epoch millisecond/,
  );
  assert.deepEqual(database.queryLimits, []);
});
