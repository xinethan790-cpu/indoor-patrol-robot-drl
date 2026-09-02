'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const fixture = require('../data/graduation-demo-seed7000-v1.js');
const {
  DEFAULT_KEY_EVENT_HOLD_MS,
  buildDashboardPayload,
  createOfflineReplay,
  identifyReplayKeyEvents,
} = require('../services/offlineReplay');
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

test('offline replay reveals the verified inspection archive only at terminal', () => {
  const running = buildDashboardPayload(fixture, fixture.frames.length - 2);
  const terminal = buildDashboardPayload(fixture, fixture.frames.length - 1);

  assert.deepEqual(running.inspection, {
    available: false,
    sample_count: null,
    visit_count: 0,
    visits: [],
  });
  assert.equal(terminal.inspection.available, true);
  assert.deepEqual(
    terminal.inspection.visits.map((visit) => visit.checkpoint_id),
    ['P1', 'P2', 'P3', 'P4'],
  );
  assert.equal(
    terminal.inspection.visits.reduce((total, visit) => total + visit.alarm_event_count, 0),
    fixture.alarm_timeline.length,
  );
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

test('offline replay identifies a bounded set of guard, alarm, and terminal key events', () => {
  const events = identifyReplayKeyEvents(fixture);
  assert.equal(events.length, 7);
  assert.deepEqual(events.map((event) => event.type), [
    'guard',
    'alarm',
    'alarm-critical',
    'guard',
    'guard',
    'alarm-cleared',
    'terminal',
  ]);
  assert.equal(events[0].sequence, 98);
  assert.equal(events.at(-1).sequence, 822);
  assert.ok(events.every((event, index) => (
    event.ordinal === index + 1 && event.total === events.length
  )));
});

test('key-event jump holds evidence and playback rate changes only ordinary frame timing', () => {
  const scheduler = createManualScheduler();
  const received = [];
  const source = createOfflineReplay({ fixture, intervalMs: 100, scheduler });
  source.start({
    onData: (payload, meta) => received.push({ sequence: payload.run.sequence, replay: meta.replay }),
  });

  assert.equal(scheduler.pending()[0].delay, 100);
  source.jumpToNextKeyEvent();
  assert.equal(received.at(-1).sequence, 98);
  assert.equal(received.at(-1).replay.keyEvent.type, 'guard');
  assert.equal(scheduler.pending()[0].delay, DEFAULT_KEY_EVENT_HOLD_MS);

  assert.equal(source.setPlaybackRate(0.5), 0.5);
  assert.equal(scheduler.pending()[0].delay, DEFAULT_KEY_EVENT_HOLD_MS);
  scheduler.runNext();
  assert.equal(scheduler.pending()[0].delay, 200);
  assert.equal(source.getState().playbackRate, 0.5);
  assert.throws(() => source.setPlaybackRate(2), /0.5x.*1x/);
});

test('offline replay validates the fixture once instead of rescanning every emitted frame', () => {
  let iteratorCount = 0;
  const frames = new Proxy(fixture.frames, {
    get(target, property, receiver) {
      if (property === Symbol.iterator) {
        iteratorCount += 1;
        if (iteratorCount > 1) throw new Error('fixture frames were revalidated');
      }
      return Reflect.get(target, property, receiver);
    },
  });
  const scheduler = createManualScheduler();
  const source = createOfflineReplay({ fixture: { ...fixture, frames }, intervalMs: 1, scheduler });
  source.start({ onData() {} });
  scheduler.runNext();
  scheduler.runNext();
  source.stop();
  assert.equal(iteratorCount, 1);
});
