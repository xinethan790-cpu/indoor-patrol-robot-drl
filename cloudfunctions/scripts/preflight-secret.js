'use strict';

const { inspectUploadSecret } = require('../shared/config');

function runSecretPreflight(env, output) {
  const source = env || process.env;
  const destination = output || process.stdout;
  const inspection = inspectUploadSecret(source.PATROL_RELAY_UPLOAD_SECRET);
  const status = inspection.ready ? 'ready' : 'not ready';
  destination.write(
    `PATROL_RELAY_UPLOAD_SECRET: ${status} (${inspection.code}: ${inspection.reason})\n`,
  );
  return inspection.ready ? 0 : 1;
}

if (require.main === module) {
  process.exitCode = runSecretPreflight();
}

module.exports = { runSecretPreflight };
