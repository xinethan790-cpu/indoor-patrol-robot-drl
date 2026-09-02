'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

test('tab bar uses distinct, safe, lightweight RGBA icons for every state', () => {
  const root = path.resolve(__dirname, '..');
  const appConfig = JSON.parse(fs.readFileSync(path.join(root, 'app.json'), 'utf8'));
  const expectedTabs = [
    {
      pagePath: 'pages/dashboard/index',
      iconPath: 'assets/tabbar/patrol.png',
      selectedIconPath: 'assets/tabbar/patrol-selected.png',
    },
    {
      pagePath: 'pages/alarms/index',
      iconPath: 'assets/tabbar/alarms.png',
      selectedIconPath: 'assets/tabbar/alarms-selected.png',
    },
    {
      pagePath: 'pages/history/index',
      iconPath: 'assets/tabbar/history.png',
      selectedIconPath: 'assets/tabbar/history-selected.png',
    },
    {
      pagePath: 'pages/about/index',
      iconPath: 'assets/tabbar/project.png',
      selectedIconPath: 'assets/tabbar/project-selected.png',
    },
  ];

  assert.equal(appConfig.tabBar.color, '#748497');
  assert.equal(appConfig.tabBar.selectedColor, '#0878f9');
  assert.equal(appConfig.tabBar.borderStyle, 'black');
  assert.equal(appConfig.tabBar.list.length, expectedTabs.length);

  const configuredPaths = [];
  appConfig.tabBar.list.forEach((tab, index) => {
    const expected = expectedTabs[index];
    assert.equal(tab.pagePath, expected.pagePath);
    assert.equal(tab.iconPath, expected.iconPath);
    assert.equal(tab.selectedIconPath, expected.selectedIconPath);

    const stateBuffers = [tab.iconPath, tab.selectedIconPath].map((relativePath) => {
      assert.equal(path.posix.isAbsolute(relativePath), false);
      assert.equal(path.win32.isAbsolute(relativePath), false);
      assert.equal(relativePath.includes('\\'), false);
      assert.equal(relativePath.split('/').includes('..'), false);
      assert.equal(path.posix.extname(relativePath), '.png');

      const resolvedPath = path.resolve(root, relativePath);
      assert.equal(resolvedPath.startsWith(`${root}${path.sep}`), true);
      assert.equal(fs.statSync(resolvedPath).isFile(), true);
      const realPath = fs.realpathSync(resolvedPath);
      assert.equal(realPath.startsWith(`${fs.realpathSync(root)}${path.sep}`), true);

      const png = fs.readFileSync(realPath);
      assert.equal(png.length < 40 * 1024, true);
      assert.deepEqual(
        Array.from(png.subarray(0, 8)),
        [137, 80, 78, 71, 13, 10, 26, 10],
      );
      assert.equal(png.subarray(12, 16).toString('ascii'), 'IHDR');
      assert.equal(png.readUInt32BE(16), 81);
      assert.equal(png.readUInt32BE(20), 81);
      assert.equal(png[24], 8);
      assert.equal(png[25], 6);
      configuredPaths.push(relativePath);
      return png;
    });

    assert.equal(stateBuffers[0].equals(stateBuffers[1]), false);
  });

  assert.equal(new Set(configuredPaths).size, 8);
});

test('interactive patrol views expose selection state and preserve page scrolling', () => {
  const root = path.resolve(__dirname, '..');
  const dashboard = fs.readFileSync(path.join(root, 'pages/dashboard/index.wxml'), 'utf8');
  const alarms = fs.readFileSync(path.join(root, 'pages/alarms/index.wxml'), 'utf8');
  const patrolMap = fs.readFileSync(path.join(root, 'components/patrol-map/index.wxml'), 'utf8');
  const modeSwitch = dashboard.match(/<view[^>]*class="mode-switch"[^>]*>/);

  assert.ok(modeSwitch);
  assert.match(modeSwitch[0], /aria-role="button"/);
  assert.match(modeSwitch[0], /aria-label="切换巡检数据来源"/);
  assert.match(alarms, /aria-pressed="\{\{activeFilter === item\.value\}\}"/);
  assert.doesNotMatch(patrolMap, /\bdisable-scroll=/);
});

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
