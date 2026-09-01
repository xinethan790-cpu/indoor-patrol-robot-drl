'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const test = require('node:test');

const { inspectUploadSecret, requireUploadSecret } = require('../shared/config');
const { runSecretPreflight } = require('../scripts/preflight-secret');

test('secret readiness rejects missing, short, template, and common example values', () => {
  const invalidSecrets = [
    { value: undefined, code: 'MISSING_SECRET' },
    { value: '', code: 'MISSING_SECRET' },
    { value: 'short-secret', code: 'SHORT_SECRET' },
    { value: `abcdefgh${' '.repeat(24)}`, code: 'WHITESPACE_SECRET' },
    { value: ` ${crypto.randomBytes(32).toString('hex')}`, code: 'WHITESPACE_SECRET' },
    { value: `${crypto.randomBytes(32).toString('hex')}\n`, code: 'WHITESPACE_SECRET' },
    { value: '{{env.PATROL_RELAY_UPLOAD_SECRET}}', code: 'TEMPLATE_SECRET' },
    { value: '${PATROL_RELAY_UPLOAD_SECRET}', code: 'TEMPLATE_SECRET' },
    { value: 'change-me-before-production-change-me', code: 'LOW_QUALITY_SECRET' },
    { value: 'your-example-default-secret-value-123456', code: 'LOW_QUALITY_SECRET' },
    { value: 'x'.repeat(32), code: 'LOW_QUALITY_SECRET' },
  ];

  for (const fixture of invalidSecrets) {
    const inspection = inspectUploadSecret(fixture.value);
    assert.equal(inspection.ready, false);
    assert.equal(inspection.code, fixture.code);
    assert.throws(() => requireUploadSecret(fixture.value), /is not ready/);
  }
});

test('a random secret of at least 32 bytes passes readiness checks', () => {
  const secret = crypto.randomBytes(32).toString('base64url');
  const inspection = inspectUploadSecret(secret);
  assert.deepEqual(inspection, {
    ready: true,
    code: 'READY',
    reason: 'upload secret passed local readiness checks',
  });
  assert.equal(requireUploadSecret(secret), secret);
});

test('offline preflight reports only readiness and a non-sensitive reason', () => {
  const validSecret = crypto.randomBytes(32).toString('base64url');
  let validOutput = '';
  const validExit = runSecretPreflight(
    { PATROL_RELAY_UPLOAD_SECRET: validSecret },
    { write: (text) => { validOutput += text; } },
  );
  assert.equal(validExit, 0);
  assert.match(validOutput, /: ready \(READY:/);
  assert.equal(validOutput.includes(validSecret), false);

  const invalidSecret = '{{env.PATROL_RELAY_UPLOAD_SECRET}}';
  let invalidOutput = '';
  const invalidExit = runSecretPreflight(
    { PATROL_RELAY_UPLOAD_SECRET: invalidSecret },
    { write: (text) => { invalidOutput += text; } },
  );
  assert.equal(invalidExit, 1);
  assert.match(invalidOutput, /: not ready \(TEMPLATE_SECRET:/);
  assert.equal(invalidOutput.includes(invalidSecret), false);

  const paddedSecret = `abcdefgh${' '.repeat(24)}`;
  let paddedOutput = '';
  const paddedExit = runSecretPreflight(
    { PATROL_RELAY_UPLOAD_SECRET: paddedSecret },
    { write: (text) => { paddedOutput += text; } },
  );
  assert.equal(paddedExit, 1);
  assert.match(paddedOutput, /: not ready \(WHITESPACE_SECRET:/);
  assert.equal(paddedOutput.includes(paddedSecret), false);
});
