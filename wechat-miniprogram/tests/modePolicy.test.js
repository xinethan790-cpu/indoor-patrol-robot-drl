'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  getPlatform,
  normalizeMode,
  resolveModeForPlatform,
} = require('../services/modePolicy');

test('cloud is the safe default while all declared modes stay available', () => {
  assert.equal(normalizeMode(undefined), 'cloud');
  assert.equal(normalizeMode('cloud'), 'cloud');
  assert.equal(normalizeMode('demo'), 'demo');
  assert.equal(normalizeMode('local'), 'local');
});

test('real devices automatically downgrade local loopback mode to cloud', () => {
  assert.deepEqual(resolveModeForPlatform('local', 'devtools'), {
    mode: 'local',
    requestedMode: 'local',
    downgraded: false,
    reason: null,
  });
  assert.deepEqual(resolveModeForPlatform('local', 'ios'), {
    mode: 'cloud',
    requestedMode: 'local',
    downgraded: true,
    reason: 'local_mode_requires_devtools',
  });
  assert.equal(resolveModeForPlatform('demo', 'android').mode, 'demo');
});

test('platform detection supports both current and legacy WeChat APIs', () => {
  assert.equal(getPlatform({ getDeviceInfo: () => ({ platform: 'devtools' }) }), 'devtools');
  assert.equal(getPlatform({ getSystemInfoSync: () => ({ platform: 'ios' }) }), 'ios');
  assert.equal(getPlatform({ getDeviceInfo: () => { throw new Error('unavailable'); } }), 'unknown');
});
