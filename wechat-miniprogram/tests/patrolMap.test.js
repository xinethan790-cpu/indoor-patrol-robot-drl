'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

let componentDefinition = null;
global.Component = (definition) => {
  componentDefinition = definition;
};
require('../components/patrol-map/index.js');
delete global.Component;

function recordingContext() {
  const calls = [];
  const methods = [
    'save', 'restore', 'beginPath', 'arc', 'stroke', 'fill', 'setLineDash',
    'moveTo', 'lineTo', 'closePath', 'fillText',
  ];
  const context = {};
  methods.forEach((name) => {
    context[name] = (...args) => calls.push([name, ...args]);
  });
  return { context, calls };
}

test('patrol map draws a current dynamic obstacle and its velocity arrow', () => {
  assert.ok(componentDefinition);
  const { context, calls } = recordingContext();
  const instance = {
    properties: { compact: false },
    _mapPoint: () => ({ x: 80, y: 60 }),
  };
  const map = {
    dynamic_obstacles: [{
      id: 'DYNAMIC_OBSTACLE_0',
      position: { x_m: -0.92, y_m: -0.24 },
      radius_m: 0.18,
      velocity: { x_mps: 0, y_mps: 0.18 },
    }],
  };

  componentDefinition.methods._drawDynamicObstacles.call(
    instance,
    context,
    map,
    {
      scale: 100,
      width: 450,
      height: 350,
      bounds: {
        minX: -2.25, maxX: 2.25, minY: -1.75, maxY: 1.75, widthM: 4.5, heightM: 3.5,
      },
    },
  );

  assert.ok(calls.some(([name, , , radius]) => name === 'arc' && radius === 18));
  assert.ok(calls.some(([name]) => name === 'lineTo'));
  assert.ok(calls.some(([name, text]) => (
    name === 'fillText' && text === 'DYNAMIC_OB'
  )));
});

test('patrol map isolates malformed dynamic-obstacle entries', () => {
  const { context, calls } = recordingContext();
  const instance = {
    properties: { compact: true },
    _mapPoint: () => {
      throw new Error('must not be called');
    },
  };

  componentDefinition.methods._drawDynamicObstacles.call(
    instance,
    context,
    { dynamic_obstacles: [{ position: { x_m: 0, y_m: 0 }, radius_m: -1 }] },
    {
      scale: 100,
      width: 450,
      height: 350,
      bounds: {
        minX: -2.25, maxX: 2.25, minY: -1.75, maxY: 1.75, widthM: 4.5, heightM: 3.5,
      },
    },
  );

  assert.deepEqual(calls, []);
});

test('patrol map skips off-map dynamic obstacles and caps enormous pixel radii', () => {
  const { context, calls } = recordingContext();
  const instance = {
    properties: { compact: false },
    _mapPoint: () => ({ x: 80, y: 60 }),
  };
  const frame = {
    scale: 100,
    width: 450,
    height: 350,
    bounds: {
      minX: -2.25, maxX: 2.25, minY: -1.75, maxY: 1.75, widthM: 4.5, heightM: 3.5,
    },
  };
  componentDefinition.methods._drawDynamicObstacles.call(instance, context, {
    dynamic_obstacles: [
      { id: 'outside', position: { x_m: 50, y_m: 50 }, radius_m: 0.2, velocity: null },
      { id: 'huge', position: { x_m: 0, y_m: 0 }, radius_m: 1000, velocity: null },
    ],
  }, frame);

  const obstacleArcs = calls.filter(([name]) => name === 'arc');
  assert.equal(obstacleArcs.length, 2);
  assert.ok(obstacleArcs.every(([, , , radius]) => radius <= 92));
  assert.equal(calls.filter(([name]) => name === 'fillText').length, 1);
});
