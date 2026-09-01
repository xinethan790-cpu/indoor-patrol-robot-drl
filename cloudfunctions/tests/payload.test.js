'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { parseDashboardBody, validateDashboardPayload } = require('../shared/payload');
const { validPayload } = require('./helpers');

function checkpoint(index) {
  return {
    id: `P${index + 1}`,
    position: { x_m: index / 10, y_m: index / 20 },
    status: index === 0 ? 'current' : 'pending',
  };
}

function obstacle(index) {
  return {
    id: `obstacle-${index}`,
    center: { x_m: index / 100, y_m: index / 200 },
    size: { width_m: 0.2, height_m: 0.3 },
  };
}

function alarmEvent(index) {
  return {
    rule_id: `rule-${index}`,
    sensor_id: 'temperature',
    severity: 'warning',
    observed_value: 39,
    threshold: 38,
    unit: '°C',
    message: `alarm ${index}`,
    checkpoint_id: 'P1',
    simulation_time_s: index,
    event_type: 'raised',
  };
}

test('accepts the patrol-dashboard/v1 envelope', () => {
  const payload = validPayload({ sequence: 12, simulationTime: 4.8 });
  assert.equal(parseDashboardBody(JSON.stringify(payload)).run.sequence, 12);
});

test('rejects unsupported schemas and malformed run identity', () => {
  const schema = validPayload();
  schema.schema_version = 'patrol-dashboard/v2';
  assert.throws(() => validateDashboardPayload(schema), (error) => error.code === 'SCHEMA_MISMATCH');

  const runId = validPayload();
  runId.run.run_id = '../escape';
  assert.throws(() => validateDashboardPayload(runId), (error) => error.code === 'INVALID_RUN_ID');
});

test('rejects sequence, terminal, trajectory, and alarm-history violations', () => {
  const badSequence = validPayload();
  badSequence.run.sequence = -1;
  assert.throws(() => validateDashboardPayload(badSequence), (error) => error.code === 'INVALID_SEQUENCE');

  const badTerminal = validPayload();
  badTerminal.run.terminal = 'false';
  assert.throws(() => validateDashboardPayload(badTerminal), (error) => error.code === 'INVALID_TERMINAL');

  const tooManyPoints = validPayload();
  tooManyPoints.map.trajectory = Array.from({ length: 301 }, () => ({}));
  assert.throws(() => validateDashboardPayload(tooManyPoints), (error) => error.code === 'INVALID_TRAJECTORY');

  const tooManyAlarms = validPayload();
  tooManyAlarms.alarms.history = Array.from({ length: 201 }, () => ({}));
  assert.throws(
    () => validateDashboardPayload(tooManyAlarms),
    (error) => error.code === 'INVALID_ALARM_HISTORY',
  );
});

test('accepts the exact supported collection limits and consistent counters', () => {
  const payload = validPayload();
  payload.map.checkpoints = Array.from({ length: 64 }, (_, index) => checkpoint(index));
  payload.map.static_obstacles = Array.from({ length: 128 }, (_, index) => obstacle(index));
  payload.map.trajectory = Array.from({ length: 300 }, (_, index) => ({
    sequence: index,
    simulation_time_s: index / 10,
    x_m: index / 100,
    y_m: index / 200,
  }));
  payload.map.trajectory_point_count = 300;
  payload.task.checkpoint_count = 64;
  payload.alarms.active = Array.from({ length: 200 }, (_, index) => alarmEvent(index));
  payload.alarms.active_count = 200;
  payload.alarms.history = Array.from({ length: 200 }, (_, index) => alarmEvent(index));
  payload.alarms.history_count = 201;
  payload.alarms.history_returned_count = 200;
  payload.alarms.history_truncated = true;

  assert.equal(validateDashboardPayload(payload), payload);
});

test('rejects checkpoint cardinality, duplicate ids, and task-count mismatches', () => {
  const empty = validPayload();
  empty.map.checkpoints = [];
  empty.task.checkpoint_count = 0;
  assert.throws(() => validateDashboardPayload(empty), (error) => error.code === 'INVALID_PAYLOAD');

  const excessive = validPayload();
  excessive.map.checkpoints = Array.from({ length: 65 }, (_, index) => checkpoint(index));
  assert.throws(
    () => validateDashboardPayload(excessive),
    (error) => error.code === 'INVALID_PAYLOAD',
  );

  const duplicate = validPayload();
  duplicate.map.checkpoints.push(structuredClone(duplicate.map.checkpoints[0]));
  duplicate.task.checkpoint_count = 2;
  assert.throws(
    () => validateDashboardPayload(duplicate),
    (error) => error.code === 'INVALID_PAYLOAD',
  );

  const countMismatch = validPayload();
  countMismatch.task.checkpoint_count = 2;
  assert.throws(
    () => validateDashboardPayload(countMismatch),
    (error) => error.code === 'INVALID_PAYLOAD',
  );
});

test('rejects excessive static obstacles and inconsistent trajectory metadata', () => {
  const excessiveObstacles = validPayload();
  excessiveObstacles.map.static_obstacles = Array.from(
    { length: 129 },
    (_, index) => obstacle(index),
  );
  assert.throws(
    () => validateDashboardPayload(excessiveObstacles),
    (error) => error.code === 'INVALID_PAYLOAD',
  );

  const countMismatch = validPayload();
  countMismatch.map.trajectory = [{
    sequence: 0,
    simulation_time_s: 0,
    x_m: -1.6,
    y_m: 0,
  }];
  assert.throws(
    () => validateDashboardPayload(countMismatch),
    (error) => error.code === 'INVALID_PAYLOAD',
  );

  const sequenceRollback = validPayload();
  sequenceRollback.map.trajectory = [
    { sequence: 2, simulation_time_s: 0, x_m: 0, y_m: 0 },
    { sequence: 1, simulation_time_s: 1, x_m: 1, y_m: 1 },
  ];
  sequenceRollback.map.trajectory_point_count = 2;
  assert.throws(
    () => validateDashboardPayload(sequenceRollback),
    (error) => error.code === 'INVALID_PAYLOAD',
  );
});

test('rejects alarm-array overflow and every inconsistent alarm count', () => {
  const excessiveActive = validPayload();
  excessiveActive.alarms.active = Array.from({ length: 201 }, (_, index) => alarmEvent(index));
  excessiveActive.alarms.active_count = 201;
  assert.throws(
    () => validateDashboardPayload(excessiveActive),
    (error) => error.code === 'INVALID_PAYLOAD',
  );

  for (const mutate of [
    (payload) => { payload.alarms.active_count = 1; },
    (payload) => { payload.alarms.history_returned_count = 1; },
    (payload) => { payload.alarms.history_count = 0; payload.alarms.history_returned_count = 1; },
  ]) {
    const payload = validPayload();
    mutate(payload);
    assert.throws(
      () => validateDashboardPayload(payload),
      (error) => error.code === 'INVALID_PAYLOAD',
    );
  }
});

test('rejects missing or mistyped task, sensor, and result fields', () => {
  const cases = [
    (payload) => { payload.task.completed_checkpoint_count = '0'; },
    (payload) => { payload.task.current_checkpoint_id = 'unknown'; },
    (payload) => { payload.sensors.available = 'false'; },
    (payload) => { payload.sensors.temperature.value = '23'; },
    (payload) => { delete payload.sensors.carbon_monoxide; },
    (payload) => { delete payload.result; },
    (payload) => { payload.result.validated = 'false'; },
    (payload) => {
      payload.result = { available: true, validated: true, success: true };
    },
  ];

  for (const mutate of cases) {
    const payload = validPayload();
    mutate(payload);
    assert.throws(
      () => validateDashboardPayload(payload),
      (error) => error.code === 'INVALID_PAYLOAD',
    );
  }
});

test('rejects invalid JSON and excessive object depth', () => {
  assert.throws(() => parseDashboardBody('{'), (error) => error.code === 'INVALID_JSON');
  const payload = validPayload();
  let cursor = payload.source;
  for (let index = 0; index < 40; index += 1) {
    cursor.next = {};
    cursor = cursor.next;
  }
  assert.throws(() => validateDashboardPayload(payload), (error) => error.code === 'INVALID_PAYLOAD');
});
