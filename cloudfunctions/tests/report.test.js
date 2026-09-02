'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  buildPatrolReport,
  validatePatrolReport,
} = require('../shared/report');
const {
  validPayload,
  validReportPayload,
  validRuntimeAbortPayload,
} = require('./helpers');

function alarmEvent(index, eventType) {
  return {
    event_id: `alarm-${index}`,
    event_index: index,
    event_type: eventType,
    rule_id: 'high_temperature',
    sensor_id: 'temperature',
    severity: 'warning',
    checkpoint_id: 'P1',
    visit_index: 1,
    sample_index: index,
    control_step: 0,
    simulation_time_s: 0,
    observed_value: eventType === 'raised' ? 39 : 35,
    threshold: eventType === 'raised' ? 38 : 36,
    unit: '°C',
    message: `temperature alarm ${eventType}`,
  };
}

function alarmReportPayload() {
  const payload = validReportPayload({ simulationTime: 1, controlStep: 3 });
  payload.inspection.visits[0].simulation_time_s = 0.2;
  payload.inspection.sample_count = 6;
  payload.result.sample_count = 6;
  payload.inspection.visits[0].readings.temperature.sample_index = 4;
  payload.inspection.visits[0].readings.smoke_obscuration.sample_index = 5;
  payload.inspection.visits[0].readings.carbon_monoxide.sample_index = 6;
  const raised = alarmEvent(1, 'raised');
  const cleared = alarmEvent(2, 'cleared');
  cleared.sample_index = 4;
  raised.control_step = 3;
  raised.simulation_time_s = 0.5;
  cleared.control_step = 3;
  cleared.simulation_time_s = 0.7;
  // Dashboard history is newest first. The report must become canonical oldest first.
  payload.alarms.history = [cleared, raised];
  payload.alarms.history_count = 2;
  payload.alarms.history_returned_count = 2;
  payload.inspection.visits[0].alarm_event_count = 2;
  payload.result.alarms_raised = 1;
  payload.result.alarms_cleared = 1;
  payload.result.highest_event_severity = 'warning';
  return payload;
}

function zeroVisitFailurePayload() {
  const payload = validPayload({ terminal: true });
  payload.run.state = 'failed';
  payload.run.outcome = 'collision';
  payload.map.checkpoints[0].status = 'current';
  payload.task.completed_checkpoint_count = 0;
  payload.task.progress_ratio = 0;
  payload.task.progress_percent = 0;
  payload.result.outcome = 'collision';
  payload.result.success = false;
  payload.result.failure_reason = 'collision detected before P1';
  payload.result.checkpoints_visited = 0;
  payload.result.collision_count = 1;
  payload.result.sample_count = 0;
  payload.inspection = {
    available: true,
    sample_count: 0,
    visit_count: 0,
    visits: [],
  };
  return payload;
}

test('builds a standalone patrol-report/v1 whitelist without relay-private fields', () => {
  const payload = alarmReportPayload();
  payload.source.alarm_history_source = 'verified_inspection_bundle';
  payload.source.nonce = 'must-not-leak';
  payload.run.body_sha256 = 'c'.repeat(64);
  payload.run.secret = 'must-not-leak';
  payload.result.private_metric = 'must-not-leak';
  payload.map.dynamic_obstacles = [{
    id: 'dynamic-private',
    position: { x_m: 0, y_m: 0 },
    radius_m: 0.2,
    velocity: null,
  }];

  const report = buildPatrolReport(payload);

  assert.equal(report.schema_version, 'patrol-report/v1');
  assert.equal(report.run.run_id, 'test-run-001');
  assert.deepEqual(report.alarms.history.map((event) => event.event_index), [1, 2]);
  assert.equal(report.alarms.history_count, 2);
  assert.equal(report.inspection.available, true);
  assert.equal(report.inspection.visit_count, 1);
  assert.deepEqual(Object.keys(report).sort(), [
    'alarms',
    'generated_at_utc',
    'inspection',
    'map',
    'result',
    'run',
    'schema_version',
    'source',
    'task',
  ]);
  assert.deepEqual(Object.keys(report.source).sort(), [
    'map_boundary_note',
    'map_qualification',
    'map_source_file',
    'terminal_validation',
  ]);
  assert.equal(report.map.dynamic_obstacles, undefined);
  const serialized = JSON.stringify(report);
  for (const forbidden of [
    'must-not-leak',
    'terminal_payload',
    'body_sha256',
    'latest_body_sha256',
    'nonce',
    'secret',
    'private_metric',
  ]) assert.equal(serialized.includes(forbidden), false, forbidden);
});

test('keeps a legacy terminal inspection unavailable and does not advertise a full report', () => {
  const legacy = validPayload({ terminal: true });
  delete legacy.inspection;

  assert.equal(buildPatrolReport(legacy), null);
});

test('creates reports only for ordinary validated terminal payloads', () => {
  assert.equal(buildPatrolReport(validPayload()), null);
  assert.equal(buildPatrolReport(validRuntimeAbortPayload()), null);
  assert.equal(buildPatrolReport(validPayload({ terminal: true })), null);
  assert.ok(buildPatrolReport(validReportPayload()));
});

test('keeps a complete zero-visit ordinary failure report distinct from unavailable evidence', () => {
  const payload = zeroVisitFailurePayload();

  const report = buildPatrolReport(payload);

  assert.equal(report.run.state, 'failed');
  assert.equal(report.run.outcome, 'collision');
  assert.equal(report.result.outcome, 'collision');
  assert.deepEqual(report.inspection, {
    available: true,
    sample_count: 0,
    visit_count: 0,
    visits: [],
  });
});

test('accepts nullable public metrics but does not build a report from truncated alarms', () => {
  const nullable = validReportPayload();
  for (const key of [
    'actual_path_length_m',
    'minimum_clearance_m',
    'minimum_lidar_range_m',
    'critical_proximity_count',
    'safety_shield_intervention_steps',
    'safety_shield_intervention_rate',
    'ttc_escape_guard_intervention_steps',
    'ttc_escape_guard_activation_count',
    'ttc_escape_guard_recovery_count',
    'ttc_escape_guard_unrecoverable_count',
    'inference_latency_p50_ms',
    'inference_latency_p95_ms',
    'dynamic_obstacle_count',
    'dynamic_obstacle_speed_max_mps',
  ]) nullable.result[key] = null;
  assert.ok(buildPatrolReport(nullable));

  const truncated = validReportPayload();
  truncated.alarms.history_count = 1;
  truncated.alarms.history_returned_count = 0;
  truncated.alarms.history_truncated = true;
  truncated.inspection.visits[0].alarm_event_count = 1;
  truncated.inspection.visits[0].highest_active_severity = 'warning';
  truncated.result.alarms_raised = 1;
  truncated.result.highest_event_severity = 'warning';
  assert.equal(buildPatrolReport(truncated), null);
});

test('fails closed when report source, identity, map boundary, or stored shape is forged', () => {
  const mutations = [
    (payload) => { payload.source.map_source_file = '../secret.yaml'; },
    (payload) => { payload.source.map_qualification = 'surveyed'; },
    (payload) => { payload.run.runtime_identity.model_id = '/Users/private/model.zip'; },
    (payload) => {
      payload.run.runtime_identity.model_id = 'C:\\Users\\private\\model.zip';
    },
    (payload) => { payload.run.runtime_identity.model_sha256 = 'not-a-digest'; },
    (payload) => { payload.map.robot_start.x_m = -99; },
    (payload) => { payload.map.checkpoints[0].position.y_m = 99; },
  ];
  for (const mutate of mutations) {
    const payload = validReportPayload();
    mutate(payload);
    assert.throws(
      () => buildPatrolReport(payload),
      (error) => error.code === 'INVALID_PAYLOAD',
    );
  }

  const extra = buildPatrolReport(validReportPayload());
  extra.terminal_payload = { secret: 'private' };
  assert.throws(
    () => validatePatrolReport(extra, 'test-run-001'),
    /unsupported or missing fields/,
  );

  const wrongRun = buildPatrolReport(validReportPayload());
  assert.throws(
    () => validatePatrolReport(wrongRun, 'another-run'),
    /run identity is invalid/,
  );
});

test('rejects non-canonical stored alarm ordering and hidden alarm-history rows', () => {
  const reversed = buildPatrolReport(alarmReportPayload());
  reversed.alarms.history.reverse();
  assert.throws(
    () => validatePatrolReport(reversed, reversed.run.run_id),
    /event indexes must be contiguous from one/,
  );

  const mismatched = buildPatrolReport(alarmReportPayload());
  mismatched.alarms.history_count = 3;
  assert.throws(
    () => validatePatrolReport(mismatched, mismatched.run.run_id),
    /cardinality is invalid/,
  );
});

test('rejects terminal lifecycle and alarm-to-visit provenance forgery', () => {
  const lifecycle = buildPatrolReport(validReportPayload());
  lifecycle.run.outcome = 'collision';
  assert.throws(
    () => validatePatrolReport(lifecycle, lifecycle.run.run_id),
    /lifecycle is inconsistent/,
  );

  for (const mutate of [
    (report) => { report.alarms.history[0].checkpoint_id = 'unknown'; },
    (report) => { report.alarms.history[0].visit_index = 2; },
    (report) => { report.alarms.history[0].control_step = 1; },
    (report) => { report.alarms.history[0].sample_index = 2; },
    (report) => { report.alarms.history[0].simulation_time_s = 0.1; },
    (report) => { report.inspection.visits[0].alarm_event_count = 1; },
    (report) => { report.inspection.visits[0].highest_active_severity = 'warning'; },
  ]) {
    const report = buildPatrolReport(alarmReportPayload());
    mutate(report);
    assert.throws(
      () => validatePatrolReport(report, report.run.run_id),
    );
  }
});

test('enforces the formal terminal collision and critical-proximity outcome semantics', () => {
  const missingCollision = buildPatrolReport(zeroVisitFailurePayload());
  missingCollision.result.collision_count = 0;
  assert.throws(
    () => validatePatrolReport(missingCollision, missingCollision.run.run_id),
    /evidence counters disagree with result outcome/,
  );

  for (const [runOutcome, resultOutcome] of [
    ['timeout', 'timeout'],
    ['error', 'error'],
    ['inspection_failed', 'error'],
  ]) {
    const forgedContact = buildPatrolReport(zeroVisitFailurePayload());
    forgedContact.run.outcome = runOutcome;
    forgedContact.result.outcome = resultOutcome;
    forgedContact.result.failure_reason = `${runOutcome} terminal`;
    assert.throws(
      () => validatePatrolReport(forgedContact, forgedContact.run.run_id),
      /evidence counters disagree with result outcome/,
    );

    forgedContact.result.collision_count = 0;
    assert.doesNotThrow(
      () => validatePatrolReport(forgedContact, forgedContact.run.run_id),
    );
  }

  for (const criticalProximityCount of [null, 0]) {
    const missingProximity = buildPatrolReport(zeroVisitFailurePayload());
    missingProximity.run.outcome = 'safety_stop';
    missingProximity.result.outcome = 'safety_stop';
    missingProximity.result.collision_count = 0;
    missingProximity.result.critical_proximity_count = criticalProximityCount;
    assert.throws(
      () => validatePatrolReport(missingProximity, missingProximity.run.run_id),
      /evidence counters disagree with result outcome/,
    );
  }

  const safetyStopWithContact = buildPatrolReport(zeroVisitFailurePayload());
  safetyStopWithContact.run.outcome = 'safety_stop';
  safetyStopWithContact.result.outcome = 'safety_stop';
  safetyStopWithContact.result.critical_proximity_count = 1;
  assert.throws(
    () => validatePatrolReport(safetyStopWithContact, safetyStopWithContact.run.run_id),
    /evidence counters disagree with result outcome/,
  );

  safetyStopWithContact.result.collision_count = 0;
  assert.doesNotThrow(
    () => validatePatrolReport(safetyStopWithContact, safetyStopWithContact.run.run_id),
  );

  const completedWithContact = buildPatrolReport(validReportPayload());
  completedWithContact.result.collision_count = 1;
  assert.throws(
    () => validatePatrolReport(completedWithContact, completedWithContact.run.run_id),
    /evidence counters disagree with result outcome/,
  );

  const recoveredAfterProximity = buildPatrolReport(validReportPayload());
  recoveredAfterProximity.result.critical_proximity_count = 1;
  assert.doesNotThrow(
    () => validatePatrolReport(recoveredAfterProximity, recoveredAfterProximity.run.run_id),
  );
});

test('rejects non-UTC timestamps, incomplete final trajectories, and forged terminal map status', () => {
  for (const mutate of [
    (report) => { report.generated_at_utc = 'September 1 2026'; },
    (report) => { report.generated_at_utc = '2026-09-01T12:00:00+08:00'; },
    (report) => {
      report.map.trajectory = [];
      report.map.trajectory_point_count = 0;
    },
    (report) => { report.map.trajectory[0].x_m += 0.1; },
    (report) => { report.map.trajectory_source_point_count = report.run.sequence + 2; },
    (report) => { report.map.checkpoints[0].status = 'pending'; },
  ]) {
    const report = buildPatrolReport(validReportPayload());
    mutate(report);
    assert.throws(
      () => validatePatrolReport(report, report.run.run_id),
    );
  }
});
