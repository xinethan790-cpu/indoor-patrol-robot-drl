'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  formatAlarmEvent,
  formatDistance,
  formatDuration,
  formatNumber,
  formatPercent,
  formatState,
  formatTemperature,
} = require('../utils/format');
const {
  getMapBounds,
  rectToPercent,
  trajectoryToPercent,
  worldToPercent,
} = require('../utils/mapTransform');

const map = {
  origin: { x_m: -2, y_m: -1 },
  size: { width_m: 4, height_m: 2 },
};

test('world coordinates map to CSS percentages with inverted Y axis', () => {
  assert.deepEqual(getMapBounds(map), {
    minX: -2, minY: -1, maxX: 2, maxY: 1, widthM: 4, heightM: 2,
  });
  assert.deepEqual(worldToPercent({ x_m: -2, y_m: -1 }, map), {
    leftPercent: 0, topPercent: 100,
  });
  assert.deepEqual(worldToPercent({ x_m: 2, y_m: 1 }, map), {
    leftPercent: 100, topPercent: 0,
  });
  assert.deepEqual(worldToPercent({ x_m: 0, y_m: 0 }, map, { paddingPercent: 10 }), {
    leftPercent: 50, topPercent: 50,
  });
});

test('map transforms clamp points and project rectangles and trajectories', () => {
  assert.deepEqual(worldToPercent({ x: 10, y: -10 }, map), {
    leftPercent: 100, topPercent: 100,
  });
  assert.deepEqual(rectToPercent({ center: { x: 0, y: 0 }, size: { width: 2, height: 1 } }, map), {
    leftPercent: 25, topPercent: 25, widthPercent: 50, heightPercent: 50,
  });
  const projected = trajectoryToPercent([{ sequence: 1, x_m: 0, y_m: 0 }], map);
  assert.deepEqual(projected, [{ sequence: 1, x_m: 0, y_m: 0, leftPercent: 50, topPercent: 50 }]);
});

test('formatters render dashboard values and preserve missing values', () => {
  assert.equal(formatNumber(null, 1), '--');
  assert.equal(formatTemperature(23.94), '23.9 °C');
  assert.equal(formatDistance(11.7429), '11.74 m');
  assert.equal(formatDuration(88.352), '1分28秒');
  assert.equal(formatPercent(0.75), '75%');
  assert.equal(formatState('sampling'), '采集环境数据');
});

test('alarm formatter localizes sensor and event action', () => {
  assert.equal(
    formatAlarmEvent({
      checkpoint_id: 'P3',
      sensor_id: 'carbon_monoxide',
      event_type: 'raised',
      observed_value: 77,
      unit: 'ppm',
    }),
    'P3 · 一氧化碳浓度触发告警：77.00 ppm',
  );
});
