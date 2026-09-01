'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const fixture = require('../data/graduation-demo-seed7000-v1.js');
const auditFixture = require('../data/graduation-demo-seed7000-v1.json');
const { buildDashboardPayload } = require('../services/offlineReplay');
const {
  PayloadProtocolError,
  createPayloadGate,
  validateDashboardPayload,
  validateFixture,
} = require('../services/protocol');

test('verified fixture keeps the bounded evidence timeline', () => {
  assert.equal(validateFixture(fixture), fixture);
  assert.equal(fixture.source.run_id, 'graduation-demo-seed7000-v1');
  assert.equal(fixture.source.terminal_validation, 'inspection_bundle_and_runtime_consistency');
  assert.equal(fixture.source.source_event_count, 823);
  assert.equal(fixture.frames.length, 90);
  assert.equal(fixture.frames[0].sequence, 0);
  assert.equal(fixture.frames.at(-1).sequence, 822);
  assert.equal(fixture.frames.at(-1).terminal, true);
  assert.equal(fixture.sensor_timeline.length, 4);
  assert.equal(fixture.alarm_timeline.length, 6);
});

test('runtime module exactly matches the auditable JSON fixture', () => {
  assert.deepEqual(fixture, auditFixture);
});

test('dashboard validator rejects an unknown schema', () => {
  const payload = buildDashboardPayload(fixture, 0);
  payload.schema_version = 'patrol-dashboard/v999';
  assert.throws(
    () => validateDashboardPayload(payload),
    (error) => error instanceof PayloadProtocolError && error.code === 'SCHEMA_MISMATCH',
  );
});

test('dashboard validator deeply rejects malformed remote map, sensor and alarm content', () => {
  const invalidMap = buildDashboardPayload(fixture, 0);
  invalidMap.map.robot.x_m = '1.0';
  assert.throws(
    () => validateDashboardPayload(invalidMap),
    (error) => error instanceof PayloadProtocolError && error.code === 'INVALID_PAYLOAD',
  );

  const invalidSensor = buildDashboardPayload(fixture, fixture.frames.length - 1);
  invalidSensor.sensors.temperature.value = Number.POSITIVE_INFINITY;
  assert.throws(() => validateDashboardPayload(invalidSensor), PayloadProtocolError);

  const invalidAlarmCount = buildDashboardPayload(fixture, fixture.frames.length - 1);
  invalidAlarmCount.alarms.history_returned_count += 1;
  assert.throws(() => validateDashboardPayload(invalidAlarmCount), PayloadProtocolError);
});

test('payload gate pins run_id and rejects foreign evidence', () => {
  const gate = createPayloadGate({ expectedRunId: fixture.run.run_id });
  gate.accept(buildDashboardPayload(fixture, 0));
  const foreign = buildDashboardPayload(fixture, 1);
  foreign.run.run_id = 'foreign-run';
  assert.throws(
    () => gate.accept(foreign),
    (error) => error instanceof PayloadProtocolError && error.code === 'RUN_ID_MISMATCH',
  );
});

test('payload gate ignores duplicates and rejects sequence rollback', () => {
  const gate = createPayloadGate();
  const first = buildDashboardPayload(fixture, 2);
  assert.equal(gate.accept(first).accepted, true);
  assert.deepEqual(gate.accept(first), { accepted: false, reason: 'duplicate', payload: first });
  assert.throws(
    () => gate.accept(buildDashboardPayload(fixture, 1)),
    (error) => error instanceof PayloadProtocolError && error.code === 'STALE_SEQUENCE',
  );
});

test('payload gate rejects conflicting content at one sequence', () => {
  const gate = createPayloadGate();
  const first = buildDashboardPayload(fixture, 2);
  gate.accept(first);
  const conflict = buildDashboardPayload(fixture, 2);
  conflict.task.progress_percent = 99;
  assert.throws(
    () => gate.accept(conflict),
    (error) => error instanceof PayloadProtocolError && error.code === 'SEQUENCE_CONFLICT',
  );
});

test('terminal gate cannot be reopened by later payloads', () => {
  const gate = createPayloadGate({ expectedRunId: fixture.run.run_id });
  const terminal = buildDashboardPayload(fixture, fixture.frames.length - 1);
  assert.equal(gate.accept(terminal).reason, 'terminal');
  const result = gate.accept(terminal);
  assert.equal(result.accepted, false);
  assert.equal(result.reason, 'terminal');
});
