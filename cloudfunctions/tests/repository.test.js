'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { CloudPatrolRepository } = require('../shared/cloud-repository');
const { validPayload } = require('./helpers');

class MemoryDocument {
  constructor(store, collection, id) {
    this.store = store;
    this.collectionName = collection;
    this.id = id;
  }

  async get() {
    const collection = this.store.get(this.collectionName);
    if (!collection || !collection.has(this.id)) throw new Error('document not found');
    return { data: structuredClone(collection.get(this.id)) };
  }

  async set({ data }) {
    if (!this.store.has(this.collectionName)) this.store.set(this.collectionName, new Map());
    this.store.get(this.collectionName).set(this.id, structuredClone(data));
    return { updated: 1 };
  }
}

class MemoryCollection {
  constructor(store, name) {
    this.store = store;
    this.name = name;
  }

  doc(id) {
    return new MemoryDocument(this.store, this.name, id);
  }
}

class MemoryDatabase {
  constructor() {
    this.store = new Map();
  }

  collection(name) {
    return new MemoryCollection(this.store, name);
  }

  async runTransaction(callback) {
    return callback(this);
  }
}

function upload(overrides) {
  const changes = overrides || {};
  const sequence = changes.sequence ?? 0;
  const terminal = changes.terminal ?? false;
  const runId = changes.runId || 'run-a';
  return {
    device_id: 'macmini-m4',
    nonce: changes.nonce || `nonce-${String(sequence).padStart(20, '0')}`,
    request_timestamp_epoch_s: 1_700_000_000 + sequence,
    body_sha256: changes.hash || String(sequence).padStart(64, 'a').slice(-64),
    run_id: runId,
    sequence,
    terminal,
    received_at_epoch_ms: 1_700_000_000_000 + (sequence * 1000),
    received_at_utc: new Date(1_700_000_000_000 + (sequence * 1000)).toISOString(),
    payload: validPayload({ runId, sequence, terminal }),
  };
}

test('repository enforces nonce replay, idempotency, conflict, and terminal locking', async () => {
  const repository = new CloudPatrolRepository(new MemoryDatabase());
  const first = upload();
  assert.equal((await repository.commitUpload(first)).disposition, 'new_run');

  await assert.rejects(
    repository.commitUpload(first),
    (error) => error.code === 'NONCE_REPLAY',
  );

  const advanced = upload({ sequence: 1, hash: 'b'.repeat(64) });
  assert.equal((await repository.commitUpload(advanced)).disposition, 'advanced');
  const duplicate = upload({
    sequence: 1,
    hash: 'b'.repeat(64),
    nonce: 'nonce-fresh-idempotent-0001',
  });
  assert.equal((await repository.commitUpload(duplicate)).disposition, 'idempotent');

  await assert.rejects(
    repository.commitUpload(upload({
      sequence: 1,
      hash: 'c'.repeat(64),
      nonce: 'nonce-fresh-conflict-00001',
    })),
    (error) => error.code === 'SEQUENCE_CONFLICT',
  );

  const terminal = upload({ sequence: 2, terminal: true, hash: 'd'.repeat(64) });
  assert.equal((await repository.commitUpload(terminal)).disposition, 'terminal');
  await assert.rejects(
    repository.commitUpload(upload({ sequence: 3, nonce: 'nonce-after-terminal-00001' })),
    (error) => error.code === 'TERMINAL_LOCKED',
  );
});

test('repository allows a new run only after terminal and rejects old run-id reuse', async () => {
  const repository = new CloudPatrolRepository(new MemoryDatabase());
  await repository.commitUpload(upload());
  await assert.rejects(
    repository.commitUpload(upload({
      runId: 'run-b',
      nonce: 'nonce-active-other-run-001',
    })),
    (error) => error.code === 'ACTIVE_RUN_CONFLICT',
  );
  await repository.commitUpload(upload({
    sequence: 1,
    terminal: true,
    hash: 'b'.repeat(64),
    nonce: 'nonce-terminal-run-a-0001',
  }));

  const newRun = upload({
    runId: 'run-b',
    sequence: 42,
    hash: 'c'.repeat(64),
    nonce: 'nonce-new-run-b-00000001',
  });
  assert.equal((await repository.commitUpload(newRun)).disposition, 'new_run');
  await repository.commitUpload(upload({
    runId: 'run-b',
    sequence: 43,
    terminal: true,
    hash: 'd'.repeat(64),
    nonce: 'nonce-terminal-run-b-0001',
  }));

  await assert.rejects(
    repository.commitUpload(upload({
      runId: 'run-a',
      sequence: 99,
      hash: 'e'.repeat(64),
      nonce: 'nonce-reuse-run-a-0000001',
    })),
    (error) => error.code === 'RUN_ID_REUSE',
  );
});

test('live snapshot has the documented public shape and does not refresh on retry', async () => {
  const repository = new CloudPatrolRepository(new MemoryDatabase());
  await repository.commitUpload(upload());
  const before = await repository.getLiveSnapshot();
  assert.deepEqual(Object.keys(before).sort(), [
    'body_sha256',
    'device_id',
    'payload',
    'received_at_epoch_ms',
    'received_at_utc',
    'run_id',
    'schema_version',
    'sequence',
    'terminal',
  ]);
  await repository.commitUpload(upload({ nonce: 'nonce-idempotent-later-001' }));
  const after = await repository.getLiveSnapshot();
  assert.equal(after.received_at_epoch_ms, before.received_at_epoch_ms);
});
