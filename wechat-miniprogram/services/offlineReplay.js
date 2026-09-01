'use strict';

const {
  DASHBOARD_SCHEMA_VERSION,
  createPayloadGate,
  validateFixture,
} = require('./protocol');

function cloneObject(value) {
  return JSON.parse(JSON.stringify(value));
}

function sensorProjection(readings) {
  const source = readings || {};
  const available = source.checkpoint_id !== null && source.checkpoint_id !== undefined;
  return {
    provenance_label: '上次巡检点采样',
    available,
    checkpoint_id: source.checkpoint_id === undefined ? null : source.checkpoint_id,
    sampled_at_s: source.sampled_at_s === undefined ? null : source.sampled_at_s,
    temperature: { display_name: '温度', value: source.temperature_c ?? null, unit: '°C' },
    smoke: {
      display_name: '烟雾遮蔽率',
      value: source.smoke_obscuration_pct_per_m ?? null,
      unit: '%/m',
    },
    carbon_monoxide: {
      display_name: '一氧化碳浓度',
      value: source.carbon_monoxide_ppm ?? null,
      unit: 'ppm',
    },
  };
}

function checkpointProjection(checkpoints, task) {
  return checkpoints.map((checkpoint, index) => {
    let status = 'pending';
    if (index < task.completed_checkpoint_count) {
      status = 'completed';
    } else if (checkpoint.id === task.current_checkpoint_id) {
      status = 'current';
    }
    return { ...checkpoint, status };
  });
}

function alarmProjection(fixture, frame) {
  const visible = fixture.alarm_timeline.filter(
    (event) => event.simulation_time_s <= frame.simulation_time_s + Number.EPSILON,
  );
  const active = frame.active_alerts.map((alert) => {
    const origin = visible.slice().reverse().find(
      (event) => event.rule_id === alert.rule_id && event.event_type === 'raised',
    );
    return { ...alert, checkpoint_id: origin ? origin.checkpoint_id : null };
  });
  const history = visible.slice().reverse();
  return {
    active_count: active.length,
    active,
    history_count: visible.length,
    history_returned_count: history.length,
    history_truncated: false,
    history,
  };
}

function buildDashboardPayload(fixtureInput, frameIndex) {
  const fixture = validateFixture(fixtureInput);
  if (!Number.isSafeInteger(frameIndex) || frameIndex < 0 || frameIndex >= fixture.frames.length) {
    throw new RangeError('frameIndex 超出离线回放范围');
  }
  const frame = fixture.frames[frameIndex];
  const task = {
    ...frame.task,
    progress_percent: Math.round(frame.task.progress_ratio * 1000) / 10,
  };
  const trajectory = fixture.frames.slice(0, frameIndex + 1).map((item) => ({
    sequence: item.sequence,
    simulation_time_s: item.simulation_time_s,
    x_m: item.position.x_m,
    y_m: item.position.y_m,
  }));
  const map = cloneObject(fixture.map);
  map.robot = cloneObject(frame.position);
  map.local_goal = frame.local_goal === null ? null : cloneObject(frame.local_goal);
  map.dynamic_obstacles = cloneObject(frame.dynamic_obstacles);
  map.trajectory = trajectory;
  map.trajectory_point_count = trajectory.length;
  map.trajectory_source_point_count = fixture.source.source_event_count;
  map.trajectory_truncated = fixture.source.source_event_count > trajectory.length;
  map.checkpoints = checkpointProjection(map.checkpoints, task);

  return {
    schema_version: DASHBOARD_SCHEMA_VERSION,
    generated_at_utc: new Date().toISOString(),
    source: {
      run_directory_name: fixture.source.run_directory_name,
      mode: frame.terminal ? 'terminal' : 'live',
      client_mode: 'offline_replay',
      alarm_history_source: 'verified_fixture_timeline',
      terminal_validation: fixture.source.terminal_validation,
      map_source_file: fixture.source.map_source_file,
      map_qualification: fixture.source.map_qualification,
      map_boundary_note: fixture.source.map_boundary_note,
    },
    run: {
      run_id: fixture.run.run_id,
      seed: fixture.run.seed,
      state: frame.state,
      outcome: frame.outcome,
      terminal: frame.terminal,
      sequence: frame.sequence,
      control_step: frame.control_step,
      simulation_time_s: frame.simulation_time_s,
      runtime_identity: cloneObject(fixture.run.runtime_identity),
    },
    map,
    task,
    sensors: sensorProjection(frame.readings),
    alarms: alarmProjection(fixture, frame),
    avoidance: cloneObject(frame.avoidance),
    result: frame.terminal
      ? cloneObject(fixture.result)
      : {
        available: false,
        validated: false,
        validation: null,
        outcome: null,
        success: null,
        failure_reason: null,
      },
  };
}

function defaultScheduler() {
  return { setTimeout, clearTimeout };
}

function createOfflineReplay(options) {
  const config = options || {};
  const fixture = validateFixture(config.fixture);
  const intervalMs = config.intervalMs ?? fixture.replay.default_interval_ms;
  if (!Number.isFinite(intervalMs) || intervalMs < 0) {
    throw new RangeError('intervalMs 必须是非负有限数字');
  }
  const scheduler = config.scheduler || defaultScheduler();
  const gate = createPayloadGate({ expectedRunId: fixture.run.run_id });
  let index = 0;
  let timer = null;
  let status = 'idle';
  let callbacks = {};

  function announce(nextStatus, details) {
    status = nextStatus;
    if (typeof callbacks.onStatus === 'function') {
      callbacks.onStatus({ status: nextStatus, ...(details || {}) });
    }
  }

  function clearTimer() {
    if (timer !== null) {
      scheduler.clearTimeout(timer);
      timer = null;
    }
  }

  function scheduleNext() {
    if (status !== 'running') return;
    timer = scheduler.setTimeout(() => {
      timer = null;
      emitNext();
    }, intervalMs);
  }

  function emitNext() {
    if (status !== 'running' || index >= fixture.frames.length) return null;
    const payload = buildDashboardPayload(fixture, index);
    index += 1;
    const decision = gate.accept(payload);
    if (decision.accepted && typeof callbacks.onData === 'function') {
      callbacks.onData(decision.payload);
    }
    if (payload.run.terminal) {
      clearTimer();
      announce('terminal', { runId: payload.run.run_id, sequence: payload.run.sequence });
    } else {
      scheduleNext();
    }
    return payload;
  }

  return {
    start(nextCallbacks) {
      if (status === 'running') return;
      callbacks = nextCallbacks || {};
      if (status === 'terminal') return;
      announce('running', { mode: 'offline_replay' });
      emitNext();
    },

    stop() {
      clearTimer();
      if (status !== 'terminal') announce('stopped');
    },

    next: emitNext,

    reset() {
      clearTimer();
      index = 0;
      callbacks = {};
      gate.reset(fixture.run.run_id);
      status = 'idle';
    },

    getState() {
      return { status, frameIndex: index, frameCount: fixture.frames.length, runId: fixture.run.run_id };
    },
  };
}

module.exports = { buildDashboardPayload, createOfflineReplay };
