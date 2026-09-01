'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const fixture = require('../data/graduation-demo-seed7000-v1.js');
const { buildDashboardPayload, createOfflineReplay } = require('../services/offlineReplay');
const { createManualScheduler } = require('./helpers');

test('offline projection grows trajectory and reveals alarms by simulation time', () => {
  const beforeAlarmIndex = fixture.frames.findIndex(
    (frame) => frame.simulation_time_s >= fixture.alarm_timeline[0].simulation_time_s,
  );
  assert.ok(beforeAlarmIndex > 0);
  const before = buildDashboardPayload(fixture, beforeAlarmIndex - 1);
  const after = buildDashboardPayload(fixture, beforeAlarmIndex);
  assert.equal(before.alarms.history_count, 0);
  assert.equal(after.alarms.history_count, 1);
  assert.equal(after.map.trajectory.length, beforeAlarmIndex + 1);
  assert.equal(after.run.run_id, fixture.run.run_id);
  assert.ok(after.map.checkpoints.every((checkpoint) => (
    ['completed', 'current', 'pending'].includes(checkpoint.status)
  )));
});

test('offline replay emits every frame once and stops at terminal', () => {
  const scheduler = createManualScheduler();
  const received = [];
  const statuses = [];
  const source = createOfflineReplay({ fixture, intervalMs: 1, scheduler });
  source.start({
    onData: (payload) => received.push(payload),
    onStatus: (value) => statuses.push(value.status),
  });
  scheduler.runAll();
  assert.equal(received.length, fixture.frames.length);
  assert.equal(received[0].run.sequence, 0);
  assert.equal(received.at(-1).run.sequence, 822);
  assert.equal(received.at(-1).run.terminal, true);
  assert.equal(source.getState().status, 'terminal');
  assert.equal(scheduler.pending().length, 0);
  assert.equal(statuses.at(-1), 'terminal');
  source.start({ onData: () => received.push('unexpected') });
  assert.equal(received.length, fixture.frames.length);
});

test('offline replay reset returns to the first frame', () => {
  const scheduler = createManualScheduler();
  const source = createOfflineReplay({ fixture, intervalMs: 1, scheduler });
  const sequences = [];
  source.start({ onData: (payload) => sequences.push(payload.run.sequence) });
  source.stop();
  source.reset();
  source.start({ onData: (payload) => sequences.push(payload.run.sequence) });
  source.stop();
  assert.deepEqual(sequences, [0, 0]);
});
