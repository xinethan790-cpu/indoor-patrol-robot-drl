'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

test('app initializes the declared CloudBase environment and safely restores mode', () => {
  const stored = {
    patrol_data_mode: 'local',
    patrol_api_base_url: 'http://127.0.0.1:8765',
  };
  const initCalls = [];
  let definition;
  global.wx = {
    cloud: { init: (options) => initCalls.push(options) },
    getDeviceInfo: () => ({ platform: 'ios' }),
    getStorageSync: (key) => stored[key],
    setStorageSync: (key, value) => { stored[key] = value; },
  };
  global.App = (value) => { definition = value; };
  delete require.cache[require.resolve('../app.js')];
  require('../app.js');

  definition.onLaunch.call(definition);
  assert.deepEqual(initCalls, [{ env: 'cloud1-5gpkygv2be4de05a' }]);
  assert.equal(definition.globalData.cloudReady, true);
  assert.equal(definition.globalData.mode, 'cloud');
  assert.equal(stored.patrol_data_mode, 'cloud');
  assert.equal('secret' in initCalls[0], false);
});
