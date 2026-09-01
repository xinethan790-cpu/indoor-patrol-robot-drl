'use strict';

const { conflict } = require('./errors');

function decideUpload(current, incoming, runAlreadyRegistered) {
  if (!current) {
    if (runAlreadyRegistered) throw conflict('RUN_ID_REUSE', 'run id was already registered');
    return { disposition: 'new_run', writeSnapshot: true };
  }

  if (current.run_id !== incoming.run_id) {
    if (!current.terminal) {
      throw conflict('ACTIVE_RUN_CONFLICT', 'a different run is still active');
    }
    if (runAlreadyRegistered) throw conflict('RUN_ID_REUSE', 'run id was already registered');
    return { disposition: 'new_run', writeSnapshot: true };
  }

  if (incoming.sequence < current.sequence) {
    throw conflict('SEQUENCE_ROLLBACK', 'incoming sequence is older than the stored sequence');
  }
  if (incoming.sequence === current.sequence) {
    if (incoming.body_sha256 !== current.body_sha256) {
      throw conflict('SEQUENCE_CONFLICT', 'the same sequence has different content');
    }
    return { disposition: 'idempotent', writeSnapshot: false };
  }
  if (current.terminal) {
    throw conflict('TERMINAL_LOCKED', 'a terminal run cannot accept later sequences');
  }
  return { disposition: incoming.terminal ? 'terminal' : 'advanced', writeSnapshot: true };
}

module.exports = { decideUpload };
