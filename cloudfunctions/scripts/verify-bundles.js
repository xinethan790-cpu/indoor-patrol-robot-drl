'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..', '.dist');
const sourceRoot = path.resolve(__dirname, '..');
const ingest = require(path.join(root, 'ingestPatrol'));
const snapshot = require(path.join(root, 'getPatrolSnapshot'));
const history = require(path.join(root, 'getPatrolHistory'));
const report = require(path.join(root, 'getPatrolReport'));
const cleanup = require(path.join(root, 'cleanupPatrolNonces'));

assert.equal(typeof ingest.main, 'function');
assert.equal(typeof ingest.createIngestHandler, 'function');
assert.equal(typeof snapshot.main, 'function');
assert.equal(typeof snapshot.createSnapshotHandler, 'function');
assert.equal(typeof history.main, 'function');
assert.equal(typeof history.createHistoryHandler, 'function');
assert.equal(typeof report.main, 'function');
assert.equal(typeof report.createReportHandler, 'function');
assert.equal(typeof cleanup.main, 'function');
assert.equal(typeof cleanup.createCleanupHandler, 'function');
assert.equal(cleanup.DELETE_BATCH_LIMIT, 100);

for (const functionName of [
  'ingestPatrol',
  'getPatrolSnapshot',
  'getPatrolHistory',
  'getPatrolReport',
  'cleanupPatrolNonces',
]) {
  const directory = path.join(root, functionName);
  const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'package.json'), 'utf8'));
  const lock = JSON.parse(fs.readFileSync(path.join(directory, 'package-lock.json'), 'utf8'));
  assert.equal(manifest.dependencies['wx-server-sdk'], '3.0.1');
  assert.equal(lock.packages[''].dependencies['wx-server-sdk'], '3.0.1');
  assert.equal(lock.packages['node_modules/wx-server-sdk'].version, '3.0.1');
}

const cloudbaseConfig = JSON.parse(
  fs.readFileSync(path.join(sourceRoot, 'cloudbaserc.json'), 'utf8'),
);
const cleanupConfig = cloudbaseConfig.functions.find(
  (definition) => definition.name === 'cleanupPatrolNonces',
);
const historyConfig = cloudbaseConfig.functions.find(
  (definition) => definition.name === 'getPatrolHistory',
);
const reportConfig = cloudbaseConfig.functions.find(
  (definition) => definition.name === 'getPatrolReport',
);
const ingestConfig = cloudbaseConfig.functions.find(
  (definition) => definition.name === 'ingestPatrol',
);
assert.ok(ingestConfig, 'ingestPatrol must be present in cloudbaserc.json');
assert.equal(
  Object.hasOwn(ingestConfig.envVariables || {}, 'PATROL_RELAY_UPLOAD_SECRET'),
  false,
  'cloudbaserc.json must not contain a deployable upload-secret placeholder',
);
assert.ok(cleanupConfig, 'cleanupPatrolNonces must be present in cloudbaserc.json');
assert.ok(historyConfig, 'getPatrolHistory must be present in cloudbaserc.json');
assert.ok(reportConfig, 'getPatrolReport must be present in cloudbaserc.json');
assert.deepEqual(historyConfig.triggers, []);
assert.deepEqual(reportConfig.triggers, []);
assert.equal(historyConfig.envVariables.PATROL_RELAY_DEVICE_ID, 'macmini-m4');
assert.equal(reportConfig.envVariables.PATROL_RELAY_DEVICE_ID, 'macmini-m4');
assert.equal(
  Object.hasOwn(historyConfig.envVariables || {}, 'PATROL_RELAY_UPLOAD_SECRET'),
  false,
);
assert.equal(
  Object.hasOwn(reportConfig.envVariables || {}, 'PATROL_RELAY_UPLOAD_SECRET'),
  false,
);
assert.deepEqual(cleanupConfig.triggers, [{
  name: 'cleanup-patrol-nonces-minutely',
  type: 'timer',
  config: '0 * * * * * *',
}]);

const functionRules = JSON.parse(
  fs.readFileSync(path.join(sourceRoot, 'function.rules.json'), 'utf8'),
);
assert.equal(functionRules.cleanupPatrolNonces.invoke, false);
assert.equal(
  functionRules.getPatrolHistory.invoke,
  "auth.loginType != 'ANONYMOUS' && auth != null",
);
assert.equal(
  functionRules.getPatrolReport.invoke,
  "auth.loginType != 'ANONYMOUS' && auth != null",
);

const collections = JSON.parse(
  fs.readFileSync(path.join(sourceRoot, 'database/collections.json'), 'utf8'),
);
const runs = collections.collections.find((definition) => definition.name === 'patrol_runs');
assert.ok(runs, 'patrol_runs collection must be declared');
assert.equal(runs.rules, 'rules/admin_only.json');
assert.equal(
  collections.collections.reduce(
    (count, definition) => count + (definition.indexes || []).length,
    0,
  ),
  3,
  'the collection manifest must retain exactly three declared indexes',
);
assert.deepEqual(
  runs.indexes.find((index) => index.name === 'device_latest_1').fields,
  [
    { field: 'device_id', direction: 'asc' },
    { field: 'latest_received_at_epoch_ms', direction: 'desc' },
  ],
);
assert.deepEqual(
  runs.indexes.find((index) => index.name === 'device_first_run_1').fields,
  [
    { field: 'device_id', direction: 'asc' },
    { field: 'first_received_at_epoch_ms', direction: 'desc' },
    { field: 'run_id', direction: 'desc' },
  ],
);

process.stdout.write('CloudBase deployment bundles are self-contained and loadable\n');
