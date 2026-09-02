'use strict';

const {
  DASHBOARD_SCHEMA_VERSION,
  createPayloadGate,
  unavailableInspection,
  validateFixture,
} = require('./protocol');

function cloneObject(value) {
  return JSON.parse(JSON.stringify(value));
}

const DEFAULT_KEY_EVENT_HOLD_MS = 1800;
const PLAYBACK_RATES = new Set([0.5, 1]);

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

function inspectionProjection(fixture, terminal) {
  if (!terminal) return unavailableInspection();
  const activeByRule = new Map();
  const visits = fixture.sensor_timeline.map((visit) => {
    const visitEvents = fixture.alarm_timeline.filter(
      (event) => event.visit_index === visit.visit_index,
    );
    visitEvents.forEach((event) => {
      if (event.event_type === 'raised') activeByRule.set(event.rule_id, event);
      else activeByRule.delete(event.rule_id);
    });
    const activeSeverities = [...activeByRule.values()].map((event) => event.severity);
    const highestActiveSeverity = activeSeverities.includes('critical')
      ? 'critical'
      : (activeSeverities.includes('warning') ? 'warning' : null);
    const startFrame = fixture.frames.find((frame) => (
      frame.control_step === visit.control_step
      && frame.state === 'dwelling'
      && frame.task.current_checkpoint_id === visit.checkpoint_id
    ));
    if (!startFrame) throw new Error(`离线演示缺少 ${visit.checkpoint_id} 的采样窗口起点`);
    return {
      visit_index: visit.visit_index,
      checkpoint_id: visit.checkpoint_id,
      control_step: visit.control_step,
      simulation_time_s: startFrame.simulation_time_s,
      alarm_event_count: visitEvents.length,
      highest_active_severity: highestActiveSeverity,
      readings: cloneObject(visit.readings),
    };
  });
  return {
    available: true,
    sample_count: fixture.result.sample_count,
    visit_count: visits.length,
    visits,
  };
}

function buildDashboardPayloadFromValidatedFixture(fixture, frameIndex) {
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
    inspection: inspectionProjection(fixture, frame.terminal),
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

function buildDashboardPayload(fixtureInput, frameIndex) {
  return buildDashboardPayloadFromValidatedFixture(validateFixture(fixtureInput), frameIndex);
}

function alarmEventsVisibleAt(fixture, simulationTimeS) {
  return fixture.alarm_timeline.filter(
    (event) => event.simulation_time_s <= simulationTimeS + Number.EPSILON,
  );
}

function describeAlarmChange(events) {
  const raised = events.filter((event) => event.event_type === 'raised');
  const cleared = events.filter((event) => event.event_type === 'cleared');
  const critical = raised.filter((event) => event.severity === 'critical');
  if (critical.length > 0) {
    return {
      type: 'alarm-critical',
      tone: 'critical',
      label: '严重环境告警触发',
      description: `${critical.length} 项严重指标超过阈值，异常记录已同步`,
    };
  }
  if (raised.length > 0) {
    return {
      type: 'alarm',
      tone: 'warning',
      label: '环境告警触发',
      description: `${raised.length} 项环境指标超过阈值`,
    };
  }
  return {
    type: 'alarm-cleared',
    tone: 'success',
    label: '环境告警解除',
    description: `${cleared.length} 项环境指标恢复正常`,
  };
}

function identifyReplayKeyEvents(validatedFixture) {
  const fixture = validatedFixture;
  const events = [];
  let previousGuardActive = false;
  let previousAlarmEventCount = 0;

  fixture.frames.forEach((frame, frameIndex) => {
    const guard = frame.avoidance && frame.avoidance.escape_guard;
    const guardActive = Boolean(guard && guard.active === true);
    const visibleAlarmEvents = alarmEventsVisibleAt(fixture, frame.simulation_time_s);
    const alarmChanges = visibleAlarmEvents.slice(previousAlarmEventCount);
    let presentation = null;

    if (frame.terminal === true) {
      presentation = {
        type: 'terminal',
        tone: 'success',
        label: '巡检任务完成',
        description: `${frame.task.completed_checkpoint_count}/${frame.task.checkpoint_count} 个巡检点已完成，终态证据已校验`,
      };
    } else if (alarmChanges.length > 0) {
      presentation = describeAlarmChange(alarmChanges);
    } else if (guardActive && !previousGuardActive) {
      presentation = {
        type: 'guard',
        tone: 'guard',
        label: '脱困保护介入',
        description: 'TTC 有界脱困覆盖常规策略动作，避免机器人受困',
      };
    }

    if (presentation) {
      events.push({
        ...presentation,
        frameIndex,
        sequence: frame.sequence,
        simulationTimeS: frame.simulation_time_s,
      });
    }
    previousGuardActive = guardActive;
    previousAlarmEventCount = visibleAlarmEvents.length;
  });

  return events.map((event, index) => ({
    ...event,
    ordinal: index + 1,
    total: events.length,
  }));
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
  const keyEventHoldMs = config.keyEventHoldMs ?? DEFAULT_KEY_EVENT_HOLD_MS;
  if (!Number.isFinite(keyEventHoldMs) || keyEventHoldMs < 0) {
    throw new RangeError('keyEventHoldMs 必须是非负有限数字');
  }
  const scheduler = config.scheduler || defaultScheduler();
  const gate = createPayloadGate({ expectedRunId: fixture.run.run_id });
  const keyEvents = identifyReplayKeyEvents(fixture);
  const keyEventByFrame = new Map(keyEvents.map((event) => [event.frameIndex, event]));
  let index = 0;
  let timer = null;
  let status = 'idle';
  let callbacks = {};
  let playbackRate = 1;
  let lastKeyEvent = null;

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

  function scheduleNext(keyEvent) {
    if (status !== 'running') return;
    const delay = keyEvent ? keyEventHoldMs : intervalMs / playbackRate;
    timer = scheduler.setTimeout(() => {
      timer = null;
      emitNext();
    }, delay);
  }

  function emitNext() {
    if (status !== 'running' || index >= fixture.frames.length) return null;
    const frameIndex = index;
    const payload = buildDashboardPayloadFromValidatedFixture(fixture, frameIndex);
    const keyEvent = keyEventByFrame.get(frameIndex) || null;
    index += 1;
    const decision = gate.accept(payload);
    if (decision.accepted && typeof callbacks.onData === 'function') {
      callbacks.onData(decision.payload, {
        replay: {
          frameIndex,
          frameCount: fixture.frames.length,
          playbackRate,
          keyEvent: keyEvent ? cloneObject(keyEvent) : null,
          keyEventCount: keyEvents.length,
          keyEventHoldMs,
        },
      });
    }
    lastKeyEvent = keyEvent;
    if (payload.run.terminal) {
      clearTimer();
      announce('terminal', { runId: payload.run.run_id, sequence: payload.run.sequence });
    } else {
      scheduleNext(keyEvent);
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

    jumpToNextKeyEvent() {
      if (status !== 'running') return null;
      const nextEvent = keyEvents.find((event) => event.frameIndex >= index);
      if (!nextEvent) return null;
      clearTimer();
      index = nextEvent.frameIndex;
      return emitNext();
    },

    setPlaybackRate(nextRate) {
      const numericRate = Number(nextRate);
      if (!PLAYBACK_RATES.has(numericRate)) {
        throw new RangeError('回放速度仅支持 0.5x 或 1x');
      }
      if (numericRate === playbackRate) return playbackRate;
      playbackRate = numericRate;
      if (status === 'running' && timer !== null) {
        clearTimer();
        scheduleNext(lastKeyEvent);
      }
      return playbackRate;
    },

    reset() {
      clearTimer();
      index = 0;
      callbacks = {};
      gate.reset(fixture.run.run_id);
      status = 'idle';
      playbackRate = 1;
      lastKeyEvent = null;
    },

    getState() {
      const nextKeyEvent = keyEvents.find((event) => event.frameIndex >= index) || null;
      return {
        status,
        frameIndex: index,
        frameCount: fixture.frames.length,
        runId: fixture.run.run_id,
        playbackRate,
        keyEventCount: keyEvents.length,
        nextKeyEvent: nextKeyEvent ? cloneObject(nextKeyEvent) : null,
      };
    },
  };
}

module.exports = {
  DEFAULT_KEY_EVENT_HOLD_MS,
  buildDashboardPayload,
  createOfflineReplay,
  identifyReplayKeyEvents,
  inspectionProjection,
};
