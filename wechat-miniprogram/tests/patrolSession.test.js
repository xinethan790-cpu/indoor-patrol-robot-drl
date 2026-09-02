'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { createPatrolSession } = require('../services/patrolSession');

function createSourceFactory() {
  const created = [];
  function createSource(mode) {
    const source = {
      mode,
      callbacks: null,
      starts: 0,
      stops: 0,
      resets: 0,
      closes: 0,
      jumps: 0,
      playbackRates: [],
      start(callbacks) {
        this.callbacks = callbacks;
        this.starts += 1;
        callbacks.onStatus({ status: 'running', transport: `${mode}-transport` });
      },
      stop() { this.stops += 1; },
      reset() { this.resets += 1; },
      close() { this.closes += 1; },
      jumpToNextKeyEvent() { this.jumps += 1; return { jumped: true }; },
      setPlaybackRate(rate) { this.playbackRates.push(rate); return rate; },
      getState() { return { status: 'running', mode }; },
      emit(payload, meta) { this.callbacks.onData(payload, meta); },
    };
    created.push(source);
    return source;
  }
  return { createSource, created };
}

test('one shared source fans out updates and page unsubscribe never stops it', () => {
  const factory = createSourceFactory();
  const session = createPatrolSession({ initialMode: 'cloud', createSource: factory.createSource });
  const first = [];
  const second = [];
  const unsubscribeFirst = session.subscribe({ onData: (payload) => first.push(payload.sequence) });
  session.subscribe({ onData: (payload) => second.push(payload.sequence) });

  session.start();
  const source = factory.created[0];
  source.emit({ sequence: 1 }, { transport: 'watch' });
  unsubscribeFirst();
  source.emit({ sequence: 2 }, { transport: 'watch' });

  assert.deepEqual(first, [1]);
  assert.deepEqual(second, [1, 2]);
  assert.equal(source.starts, 1);
  assert.equal(source.stops, 0);
  assert.equal(session.getState().subscriberCount, 1);
});

test('one broken page subscriber cannot interrupt updates for the other pages', () => {
  const factory = createSourceFactory();
  const session = createPatrolSession({ initialMode: 'cloud', createSource: factory.createSource });
  const received = [];
  session.subscribe({ onData() { throw new Error('page render failed'); } });
  session.subscribe({ onData: (payload) => received.push(payload.sequence) });
  session.start();

  assert.doesNotThrow(() => factory.created[0].emit({ sequence: 3 }));
  assert.deepEqual(received, [3]);
  assert.equal(session.getState().started, true);
});

test('late pages immediately receive the latest payload and connection status', () => {
  const factory = createSourceFactory();
  const session = createPatrolSession({ initialMode: 'local', createSource: factory.createSource });
  session.start();
  factory.created[0].emit({ sequence: 7 }, { receivedAtUtc: '2026-09-01T00:00:00Z' });
  const replay = [];
  session.subscribe({
    onData: (payload, meta) => replay.push(['data', payload.sequence, meta.mode]),
    onStatus: (event) => replay.push(['status', event.status, event.mode]),
  });

  assert.deepEqual(replay, [
    ['data', 7, 'local'],
    ['status', 'running', 'local'],
  ]);
});

test('mode replacement is atomic, restarts one source, and ignores the old source', () => {
  const factory = createSourceFactory();
  const session = createPatrolSession({ initialMode: 'cloud', createSource: factory.createSource });
  const received = [];
  const clears = [];
  session.subscribe({
    onClear: (event) => clears.push([event.mode, event.reason]),
    onData: (payload, meta) => received.push([payload.sequence, meta.mode]),
  });
  session.start();
  const cloud = factory.created[0];
  cloud.emit({ sequence: 1 });

  assert.equal(session.setMode('demo'), true);
  const demo = factory.created[1];
  cloud.emit({ sequence: 99 });
  demo.emit({ sequence: 0 });

  assert.equal(cloud.stops, 1);
  assert.equal(cloud.closes, 1);
  assert.equal(demo.starts, 1);
  assert.deepEqual(received, [[1, 'cloud'], [0, 'demo']]);
  assert.deepEqual(clears, [['demo', 'mode_changed']]);
  assert.equal(session.getState().mode, 'demo');
});

test('demo pause, resume, and reset preserve explicit playback semantics', () => {
  const factory = createSourceFactory();
  const session = createPatrolSession({ initialMode: 'demo', createSource: factory.createSource });
  session.start();
  const demo = factory.created[0];

  session.stop();
  assert.equal(session.getState().started, false);
  assert.equal(demo.stops, 1);
  session.start();
  assert.equal(demo.starts, 2);
  session.reset({ restart: true });

  assert.equal(demo.resets, 1);
  assert.equal(demo.starts, 3);
  assert.equal(session.getState().started, true);
  assert.equal(session.getState().payload, null);
});

test('demo presentation controls are delegated only to the offline source', () => {
  const demoFactory = createSourceFactory();
  const demo = createPatrolSession({ initialMode: 'demo', createSource: demoFactory.createSource });
  demo.start();
  assert.equal(demo.setPlaybackRate(0.5), 0.5);
  assert.deepEqual(demo.jumpToNextKeyEvent(), { jumped: true });
  assert.deepEqual(demoFactory.created[0].playbackRates, [0.5]);
  assert.equal(demoFactory.created[0].jumps, 1);

  const cloudFactory = createSourceFactory();
  const cloud = createPatrolSession({ initialMode: 'cloud', createSource: cloudFactory.createSource });
  cloud.start();
  assert.equal(cloud.setPlaybackRate(0.5), null);
  assert.equal(cloud.jumpToNextKeyEvent(), null);
  assert.deepEqual(cloudFactory.created[0].playbackRates, []);
  assert.equal(cloudFactory.created[0].jumps, 0);
});

for (const mode of ['demo', 'local']) {
  test(`${mode} terminal status survives app hide and show`, () => {
    let terminal = false;
    let callbacks = null;
    const source = {
      start(nextCallbacks) {
        callbacks = nextCallbacks;
        if (!terminal) callbacks.onStatus({ status: 'running' });
      },
      stop() {},
      getState() { return { status: terminal ? 'terminal' : 'running' }; },
      finish() {
        terminal = true;
        callbacks.onStatus({ status: 'terminal', runId: 'finished-run', sequence: 9 });
      },
    };
    const session = createPatrolSession({ initialMode: mode, createSource: () => source });
    session.start();
    source.finish();
    session.stop();
    assert.equal(session.getState().status, 'terminal');
    assert.equal(session.getState().started, false);

    session.start();
    assert.equal(session.getState().status, 'terminal');
    assert.equal(session.getState().started, true);
  });
}

test('source initialization errors are replayed and the session remains fail-closed', () => {
  const failure = new Error('cloud unavailable');
  const session = createPatrolSession({
    initialMode: 'cloud',
    createSource() { throw failure; },
  });
  session.start();
  const events = [];
  session.subscribe({
    onStatus: (event) => events.push(['status', event.status]),
    onError: (error, details) => events.push(['error', error.message, details.retrying]),
  });

  assert.equal(session.getState().started, false);
  assert.deepEqual(events, [
    ['status', 'error'],
    ['error', 'cloud unavailable', false],
  ]);
});
