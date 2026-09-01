'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..', '.dist');
const sourceRoot = path.resolve(__dirname, '..');
const ingest = require(path.join(root, 'ingestPatrol'));
const snapshot = require(path.join(root, 'getPatrolSnapshot'));
const cleanup = require(path.join(root, 'cleanupPatrolNonces'));

assert.equal(typeof ingest.main, 'function');
assert.equal(typeof ingest.createIngestHandler, 'function');
assert.equal(typeof snapshot.main, 'function');
assert.equal(typeof snapshot.createSnapshotHandler, 'function');
assert.equal(typeof cleanup.main, 'function');
assert.equal(typeof cleanup.createCleanupHandler, 'function');
assert.equal(cleanup.DELETE_BATCH_LIMIT, 100);

for (const functionName of ['ingestPatrol', 'getPatrolSnapshot', 'cleanupPatrolNonces']) {
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
assert.ok(cleanupConfig, 'cleanupPatrolNonces must be present in cloudbaserc.json');
assert.deepEqual(cleanupConfig.triggers, [{
  name: 'cleanup-patrol-nonces-minutely',
  type: 'timer',
  config: '0 * * * * * *',
}]);

const functionRules = JSON.parse(
  fs.readFileSync(path.join(sourceRoot, 'function.rules.json'), 'utf8'),
);
assert.equal(functionRules.cleanupPatrolNonces.invoke, false);

process.stdout.write('CloudBase deployment bundles are self-contained and loadable\n');
