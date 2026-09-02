'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

test('tracked project pins the review base library and demo controls expose accessible names', () => {
  const root = path.resolve(__dirname, '..');
  const project = JSON.parse(fs.readFileSync(path.join(root, 'project.config.json'), 'utf8'));
  const dashboard = fs.readFileSync(path.join(root, 'pages/dashboard/index.wxml'), 'utf8');
  assert.equal(project.libVersion, '3.17.2');
  assert.match(dashboard, /aria-live="polite"/);
  assert.match(dashboard, /aria-label="跳转到下一个关键事件"/);
  assert.match(dashboard, /aria-label="演示播放速度"/);
  assert.match(dashboard, /aria-pressed="\{\{demoHalfSpeedActive\}\}"/);
});

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

test('app owns one patrol session and keeps it alive when page subscribers change', () => {
  const stored = { patrol_data_mode: 'demo' };
  let definition;
  global.wx = {
    cloud: { init() {} },
    getDeviceInfo: () => ({ platform: 'ios' }),
    getStorageSync: (key) => stored[key],
    setStorageSync: (key, value) => { stored[key] = value; },
  };
  global.App = (value) => { definition = value; };
  delete require.cache[require.resolve('../app.js')];
  require('../app.js');
  definition.onLaunch.call(definition);

  const first = [];
  const second = [];
  const unsubscribeFirst = definition.subscribePatrol.call(definition, {
    onData: (payload) => first.push(payload.run.sequence),
  });
  definition.subscribePatrol.call(definition, {
    onData: (payload) => second.push(payload.run.sequence),
  });
  definition.onShow.call(definition);
  unsubscribeFirst();

  assert.deepEqual(first, [0]);
  assert.deepEqual(second, [0]);
  assert.equal(definition.getPatrolSessionState.call(definition).started, true);
  assert.equal(definition.getPatrolSessionState.call(definition).subscriberCount, 2);
  assert.equal(definition.setDemoPlaybackRate.call(definition, 0.5), 0.5);
  const jumped = definition.jumpToNextDemoKeyEvent.call(definition);
  assert.equal(jumped.run.sequence, 98);
  assert.deepEqual(second, [0, 98]);
  definition.onHide.call(definition);
  assert.equal(definition.getPatrolSessionState.call(definition).started, false);
});
