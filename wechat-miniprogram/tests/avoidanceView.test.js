'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { buildAvoidanceView } = require('../utils/avoidanceView');

function availableAvoidance(overrides) {
  return {
    available: true,
    minimum_lidar_range_m: 0.2861,
    robust_ttc_s: 0.43,
    critical_proximity: false,
    actions: {
      policy: { linear_normalized: 0.52, angular_normalized: -0.2 },
      filtered: { linear_normalized: -0.48, angular_normalized: -0.2 },
      executed: { linear_mps: -0.12, angular_radps: -0.3 },
    },
    safety_shield: { active: true, intervention: 1.0 },
    escape_guard: { active: false, mode: 'inactive', unrecoverable: false },
    ...(overrides || {}),
  };
}

test('avoidance card makes policy filtering and executed command visible', () => {
  const view = buildAvoidanceView(availableAvoidance(), {
    dynamic_obstacles: [{ id: 'moving-person' }, { id: 'cart' }],
  });

  assert.equal(view.avoidanceStatus, '安全盾介入');
  assert.equal(view.avoidanceTone, 'warning');
  assert.equal(view.minimumRangeText, '0.29 m');
  assert.equal(view.ttcText, '0.43 s');
  assert.equal(view.dynamicObstacleText, '2 个');
  assert.equal(view.policyActionText, 'v +0.52 · ω -0.20');
  assert.equal(view.filteredActionText, 'v -0.48 · ω -0.20');
  assert.equal(view.executedActionText, '-0.12 m/s · -0.30 rad/s');
});

test('escape guard receives display priority over the ordinary shield state', () => {
  const avoidance = availableAvoidance({
    escape_guard: { active: true, mode: 'rotate', unrecoverable: false },
  });
  const view = buildAvoidanceView(avoidance, { dynamic_obstacles: [] });

  assert.equal(view.avoidanceStatus, '旋转脱困');
  assert.equal(view.avoidanceTone, 'guard');
  assert.match(view.avoidanceSummary, /脱困/);
});

test('unavailable telemetry has an explicit honest empty state', () => {
  const view = buildAvoidanceView({ available: false }, {
    dynamic_obstacles: [{ id: 'moving-person' }],
  });

  assert.equal(view.avoidanceAvailable, false);
  assert.equal(view.avoidanceStatus, '等待避障遥测');
  assert.equal(view.dynamicObstacleText, '1 个');
  assert.equal(view.policyActionText, '等待数据');
});

test('available dwell telemetry does not pretend a PPO action was produced', () => {
  const view = buildAvoidanceView(availableAvoidance({
    actions: {
      policy: null,
      filtered: null,
      executed: { linear_mps: 0, angular_radps: 0 },
    },
    safety_shield: { active: false, intervention: 0 },
    robust_ttc_s: null,
  }), { dynamic_obstacles: [] });

  assert.equal(view.avoidanceStatus, '驻留 / 停车');
  assert.match(view.avoidanceSummary, /没有新的 PPO/);
  assert.equal(view.policyActionText, '等待数据');
  assert.equal(view.executedActionText, '0.00 m/s · 0.00 rad/s');
});
