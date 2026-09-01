'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { decideUpload } = require('../shared/state-machine');

function state(overrides) {
  return {
    run_id: 'run-a',
    sequence: 4,
    terminal: false,
    body_sha256: 'a'.repeat(64),
    ...(overrides || {}),
  };
}

test('accepts a first upload and monotonic advances', () => {
  assert.deepEqual(decideUpload(null, state(), false), {
    disposition: 'new_run',
    writeSnapshot: true,
  });
  assert.deepEqual(decideUpload(state(), state({ sequence: 5, body_sha256: 'b'.repeat(64) }), false), {
    disposition: 'advanced',
    writeSnapshot: true,
  });
  assert.deepEqual(
    decideUpload(state(), state({ sequence: 5, terminal: true, body_sha256: 'c'.repeat(64) }), false),
    { disposition: 'terminal', writeSnapshot: true },
  );
});

test('treats exact retry as idempotent but rejects conflict and rollback', () => {
  assert.deepEqual(decideUpload(state(), state(), false), {
    disposition: 'idempotent',
    writeSnapshot: false,
  });
  assert.throws(
    () => decideUpload(state(), state({ body_sha256: 'b'.repeat(64) }), false),
    (error) => error.code === 'SEQUENCE_CONFLICT',
  );
  assert.throws(
    () => decideUpload(state(), state({ sequence: 3 }), false),
    (error) => error.code === 'SEQUENCE_ROLLBACK',
  );
});

test('locks terminal runs and permits only an unregistered new run', () => {
  const terminal = state({ terminal: true });
  assert.throws(
    () => decideUpload(terminal, state({ sequence: 5, body_sha256: 'b'.repeat(64) }), false),
    (error) => error.code === 'TERMINAL_LOCKED',
  );
  assert.deepEqual(
    decideUpload(terminal, state({ run_id: 'run-b', sequence: 18 }), false),
    { disposition: 'new_run', writeSnapshot: true },
  );
  assert.throws(
    () => decideUpload(terminal, state({ run_id: 'run-old' }), true),
    (error) => error.code === 'RUN_ID_REUSE',
  );
});

test('does not replace an active run with another run id', () => {
  assert.throws(
    () => decideUpload(state(), state({ run_id: 'run-b' }), false),
    (error) => error.code === 'ACTIVE_RUN_CONFLICT',
  );
});
