'use strict';

const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, '.dist');
const DEVTOOLS_DIST = path.resolve(ROOT, '..', 'wechat-miniprogram', 'cloudfunctions');
const SHARED = path.join(ROOT, 'shared');

const functions = [
  { name: 'ingestPatrol', handler: 'ingest-handler' },
  { name: 'getPatrolSnapshot', handler: 'snapshot-handler' },
  { name: 'getPatrolHistory', handler: 'history-handler' },
  { name: 'cleanupPatrolNonces', handler: 'cleanup-handler' },
];

function copyDirectory(source, destination) {
  fs.mkdirSync(destination, { recursive: true });
  for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
    const sourcePath = path.join(source, entry.name);
    const destinationPath = path.join(destination, entry.name);
    if (entry.isDirectory()) copyDirectory(sourcePath, destinationPath);
    else if (entry.isFile()) fs.copyFileSync(sourcePath, destinationPath);
  }
}

for (const outputRoot of [DIST, DEVTOOLS_DIST]) {
  fs.rmSync(outputRoot, { recursive: true, force: true });
  fs.mkdirSync(outputRoot, { recursive: true });
  for (const definition of functions) {
    const destination = path.join(outputRoot, definition.name);
    fs.mkdirSync(destination, { recursive: true });
    copyDirectory(SHARED, path.join(destination, 'shared'));
    fs.copyFileSync(
      path.join(ROOT, definition.name, 'package.json'),
      path.join(destination, 'package.json'),
    );
    fs.copyFileSync(
      path.join(ROOT, definition.name, 'package-lock.json'),
      path.join(destination, 'package-lock.json'),
    );
    fs.writeFileSync(
      path.join(destination, 'index.js'),
      `'use strict';\n\nmodule.exports = require('./shared/${definition.handler}');\n`,
      'utf8',
    );
  }
}

process.stdout.write(
  `Built ${functions.length} CloudBase deployment bundles for CLI and WeChat DevTools\n`,
);
