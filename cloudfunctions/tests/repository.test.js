'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { CloudPatrolRepository, runDocument } = require('../shared/cloud-repository');
const { validPayload, validReportPayload, validRuntimeAbortPayload } = require('./helpers');

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

  where(filter) {
    return new MemoryQuery(this.store, this.name, filter);
  }
}

class MemoryQuery {
  constructor(store, name, filter) {
    this.store = store;
    this.name = name;
    this.filter = filter;
    this.ordering = [];
    this.offset = 0;
    this.maximum = Number.MAX_SAFE_INTEGER;
    this.projection = null;
  }

  orderBy(field, direction) {
    this.ordering.push([field, direction]);
    return this;
  }

  skip(offset) {
    this.offset = offset;
    return this;
  }

  limit(maximum) {
    this.maximum = maximum;
    return this;
  }

  field(projection) {
    this.projection = projection;
    return this;
  }

  async get() {
    const collection = this.store.get(this.name) || new Map();
    const rows = [...collection.entries()]
      .map(([id, value]) => ({ _id: id, ...structuredClone(value) }))
      .filter((row) => Object.entries(this.filter).every(([key, value]) => row[key] === value))
      .sort((left, right) => {
        for (const [field, direction] of this.ordering) {
          if (left[field] === right[field]) continue;
          const result = left[field] < right[field] ? -1 : 1;
          return direction === 'desc' ? -result : result;
        }
        return 0;
      })
      .slice(this.offset, this.offset + this.maximum)
      .map((row) => {
        if (!this.projection) return row;
        return Object.fromEntries(
          Object.entries(row).filter(([key]) => this.projection[key] === true),
        );
      });
    return { data: rows };
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
    received_at_epoch_ms: changes.receivedAtEpochMs
      ?? (1_700_000_000_000 + (sequence * 1000)),
    received_at_utc: new Date(
      changes.receivedAtEpochMs ?? (1_700_000_000_000 + (sequence * 1000)),
    ).toISOString(),
    payload: changes.payload || validPayload({ runId, sequence, terminal }),
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

test('an incomplete runtime-abort terminal releases the device for a new run', async () => {
  const repository = new CloudPatrolRepository(new MemoryDatabase());
  await repository.commitUpload(upload());
  const abortedPayload = validRuntimeAbortPayload({ runId: 'run-a', sequence: 1 });
  const aborted = upload({
    sequence: 1,
    terminal: true,
    hash: 'b'.repeat(64),
    nonce: 'nonce-runtime-abort-00001',
    payload: abortedPayload,
  });

  assert.equal((await repository.commitUpload(aborted)).disposition, 'terminal');
  const terminalSnapshot = await repository.getLiveSnapshot();
  assert.equal(terminalSnapshot.terminal, true);
  assert.equal(terminalSnapshot.payload.result.validated, false);
  assert.equal(terminalSnapshot.payload.result.incomplete, true);
  const abortedRun = [...repository.database.store.get('patrol_runs').values()][0];
  assert.equal(abortedRun.incomplete, true);
  assert.equal(abortedRun.result_outcome, 'aborted');
  assert.equal(abortedRun.result_validated, false);
  assert.equal(abortedRun.report_available, false);
  assert.equal(abortedRun.report, undefined);

  const next = upload({
    runId: 'run-b',
    sequence: 0,
    hash: 'c'.repeat(64),
    nonce: 'nonce-after-runtime-abort-1',
  });
  assert.equal((await repository.commitUpload(next)).disposition, 'new_run');
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

test('run audit advances monotonically while an idempotent retry does not refresh it', async () => {
  const database = new MemoryDatabase();
  const repository = new CloudPatrolRepository(database);
  await repository.commitUpload(upload());
  const firstRun = [...database.store.get('patrol_runs').values()][0];
  assert.equal(firstRun.latest_sequence, 0);
  assert.equal(firstRun.terminal, false);
  assert.equal(firstRun.progress_percent, 0);
  assert.equal(firstRun.report_available, false);

  const advancedPayload = validPayload({ runId: 'run-a', sequence: 5 });
  advancedPayload.run.state = 'navigating';
  advancedPayload.run.outcome = 'running';
  advancedPayload.task.completed_checkpoint_count = 1;
  advancedPayload.task.progress_ratio = 1;
  advancedPayload.task.progress_percent = 100;
  const advanced = upload({
    sequence: 5,
    hash: 'f'.repeat(64),
    payload: advancedPayload,
  });
  await repository.commitUpload(advanced);
  const afterAdvance = structuredClone([...database.store.get('patrol_runs').values()][0]);
  assert.equal(afterAdvance.latest_sequence, 5);
  assert.equal(afterAdvance.progress_percent, 100);
  assert.equal(afterAdvance.completed_checkpoint_count, 1);
  assert.equal(afterAdvance.active_alarm_count, 0);
  assert.equal(afterAdvance.alarm_history_count, 0);
  assert.equal(afterAdvance.state, 'navigating');
  assert.equal(afterAdvance.runtime_outcome, 'running');
  assert.equal(afterAdvance.latest_received_at_epoch_ms, advanced.received_at_epoch_ms);

  await repository.commitUpload(upload({
    sequence: 5,
    hash: 'f'.repeat(64),
    nonce: 'nonce-history-idempotent-01',
    receivedAtEpochMs: advanced.received_at_epoch_ms + 99_000,
    payload: advancedPayload,
  }));
  const afterRetry = [...database.store.get('patrol_runs').values()][0];
  assert.deepEqual(afterRetry, afterAdvance);
});

test('repository stores and reads only a safe report for an ordinary terminal run', async () => {
  const database = new MemoryDatabase();
  const repository = new CloudPatrolRepository(database);
  await repository.commitUpload(upload());

  const terminalPayload = validReportPayload({
    runId: 'run-a',
    sequence: 1,
    terminal: true,
  });
  await repository.commitUpload(upload({
    sequence: 1,
    terminal: true,
    hash: 'b'.repeat(64),
    nonce: 'nonce-report-terminal-0001',
    payload: terminalPayload,
  }));

  const stored = [...database.store.get('patrol_runs').values()][0];
  assert.equal(stored.report_available, true);
  assert.equal(stored.report.schema_version, 'patrol-report/v1');
  assert.ok(stored.terminal_payload);

  const report = await repository.getPatrolReport({
    deviceId: 'macmini-m4',
    runId: 'run-a',
  });
  assert.deepEqual(report, stored.report);
  assert.equal(JSON.stringify(report).includes('terminal_payload'), false);
  assert.equal(await repository.getPatrolReport({
    deviceId: 'macmini-m4',
    runId: 'unknown-run',
  }), null);
  assert.equal(await repository.getPatrolReport({
    deviceId: 'other-device',
    runId: 'run-a',
  }), null);
});

test('history repository rejects a malformed SDK data envelope instead of fabricating emptiness', async () => {
  const query = {
    where() { return this; },
    orderBy() { return this; },
    skip() { return this; },
    limit() { return this; },
    field() { return this; },
    async get() { return { data: { unexpected: 'object' } }; },
  };
  const repository = new CloudPatrolRepository({
    async runTransaction() {},
    collection() { return query; },
  });

  await assert.rejects(
    repository.getPatrolHistoryPage({
      deviceId: 'macmini-m4',
      pageSize: 10,
      offset: 0,
    }),
    /invalid data envelope/,
  );
});

test('history pagination keeps immutable ordering and does not lose old runs after updates and insertions', async () => {
  const database = new MemoryDatabase();
  const repository = new CloudPatrolRepository(database);
  for (const [index, runId] of ['run-a', 'run-b', 'run-c'].entries()) {
    const terminal = runId !== 'run-c';
    await repository.commitUpload(upload({
      runId,
      sequence: 0,
      terminal,
      nonce: `nonce-history-${runId}-000001`,
      hash: String(index + 1).repeat(64),
      receivedAtEpochMs: 1_700_000_000_000 + (index * 1000),
      payload: validPayload({ runId, terminal }),
    }));
  }
  database.store.get('patrol_runs').set('other-device-run', {
    ...runDocument(upload({ runId: 'foreign', terminal: true }), {
      utc: '2023-11-14T22:13:20.000Z',
      epochMs: 1_700_000_000_000,
    }),
    device_id: 'other-device',
    terminal_payload: { secret: 'must-not-be-read' },
  });

  const first = await repository.getPatrolHistoryPage({
    deviceId: 'macmini-m4',
    pageSize: 2,
    offset: 0,
  });
  assert.deepEqual(first.documents.map((item) => item.run_id), ['run-c', 'run-b']);
  assert.equal(first.hasMore, true);
  assert.equal(first.nextOffset, 2);
  assert.equal(first.documents.some((item) => 'terminal_payload' in item), false);
  assert.equal(first.documents.some((item) => 'report' in item), false);
  assert.equal(first.documents.some((item) => 'latest_body_sha256' in item), false);
  assert.equal(first.documents.every((item) => typeof item.report_available === 'boolean'), true);

  const oldRun = [...database.store.get('patrol_runs').values()]
    .find((document) => document.run_id === 'run-a');
  oldRun.latest_sequence = 99;
  oldRun.progress_percent = 75;
  oldRun.latest_received_at_epoch_ms = 1_700_000_099_000;
  oldRun.latest_received_at_utc = new Date(oldRun.latest_received_at_epoch_ms).toISOString();

  await repository.commitUpload(upload({
    runId: 'run-c',
    sequence: 1,
    nonce: 'nonce-history-run-c-update1',
    hash: '4'.repeat(64),
    receivedAtEpochMs: 1_700_000_003_000,
    payload: validPayload({ runId: 'run-c', sequence: 1 }),
  }));
  const afterProgressUpdate = await repository.getPatrolHistoryPage({
    deviceId: 'macmini-m4',
    pageSize: 2,
    offset: 0,
  });
  assert.deepEqual(
    afterProgressUpdate.documents.map((item) => item.run_id),
    ['run-c', 'run-b'],
  );

  await repository.commitUpload(upload({
    runId: 'run-c',
    sequence: 2,
    terminal: true,
    nonce: 'nonce-history-run-c-terminal',
    hash: '5'.repeat(64),
    receivedAtEpochMs: 1_700_000_004_000,
    payload: validPayload({ runId: 'run-c', sequence: 2, terminal: true }),
  }));
  await repository.commitUpload(upload({
    runId: 'run-d',
    sequence: 0,
    terminal: true,
    nonce: 'nonce-history-run-d-000001',
    hash: '6'.repeat(64),
    receivedAtEpochMs: 1_700_000_005_000,
    payload: validPayload({ runId: 'run-d', terminal: true }),
  }));
  await repository.commitUpload(upload({
    runId: 'run-e',
    sequence: 0,
    terminal: true,
    nonce: 'nonce-history-run-e-000001',
    hash: '7'.repeat(64),
    receivedAtEpochMs: 1_700_000_006_000,
    payload: validPayload({ runId: 'run-e', terminal: true }),
  }));
  const second = await repository.getPatrolHistoryPage({
    deviceId: 'macmini-m4',
    pageSize: 2,
    offset: first.nextOffset,
  });
  assert.deepEqual(second.documents.map((item) => item.run_id), ['run-c', 'run-b']);
  assert.equal(second.hasMore, true);
  const third = await repository.getPatrolHistoryPage({
    deviceId: 'macmini-m4',
    pageSize: 2,
    offset: second.nextOffset,
  });
  assert.deepEqual(third.documents.map((item) => item.run_id), ['run-a']);
  assert.equal(third.hasMore, false);
  assert.deepEqual(
    new Set(
      [...first.documents, ...second.documents, ...third.documents]
        .map((item) => item.run_id),
    ),
    new Set(['run-a', 'run-b', 'run-c']),
  );
});
