'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const fixture = require('../data/graduation-demo-seed7000-v1.js');
const { buildDashboardPayload } = require('../services/offlineReplay');
const {
  ReportProtocolError,
  buildDemoPatrolReport,
  buildPatrolReportFromDashboard,
  createPatrolReportService,
  validatePatrolReport,
  validateReportResponse,
} = require('../services/patrolReport');

function response(report) {
  return {
    ok: true,
    schema_version: 'patrol-report-response/v1',
    device_id: 'macmini-m4',
    report,
  };
}

function zeroVisitCollisionReport() {
  const report = structuredClone(buildDemoPatrolReport());
  const start = report.map.robot_start;
  report.run.state = 'failed';
  report.run.outcome = 'collision';
  report.run.sequence = 0;
  report.run.control_step = 0;
  report.run.simulation_time_s = 0;
  report.task.completed_checkpoint_count = 0;
  report.task.progress_ratio = 0;
  report.task.progress_percent = 0;
  report.result.outcome = 'collision';
  report.result.success = false;
  report.result.failure_reason = 'physical contact detected by Webots contact points';
  report.result.checkpoints_visited = 0;
  report.result.collision_count = 1;
  report.result.elapsed_simulation_time_s = 0;
  report.result.actual_path_length_m = 0;
  report.result.minimum_clearance_m = null;
  report.result.minimum_lidar_range_m = null;
  report.result.critical_proximity_count = null;
  report.result.safety_shield_intervention_steps = null;
  report.result.safety_shield_intervention_rate = null;
  report.result.ttc_escape_guard_intervention_steps = null;
  report.result.ttc_escape_guard_activation_count = null;
  report.result.ttc_escape_guard_recovery_count = null;
  report.result.ttc_escape_guard_unrecoverable_count = null;
  report.result.inference_latency_p50_ms = null;
  report.result.inference_latency_p95_ms = null;
  report.result.dynamic_obstacle_count = null;
  report.result.dynamic_obstacle_speed_max_mps = null;
  report.result.sample_count = 0;
  report.result.alarms_raised = 0;
  report.result.alarms_cleared = 0;
  report.result.highest_event_severity = null;
  report.inspection = { available: true, sample_count: 0, visit_count: 0, visits: [] };
  report.alarms = { history_count: 0, history: [] };
  report.map.robot = structuredClone(start);
  report.map.trajectory = [{ sequence: 0, simulation_time_s: 0, ...start }];
  report.map.trajectory_point_count = 1;
  report.map.trajectory_source_point_count = 1;
  report.map.trajectory_truncated = false;
  report.map.checkpoints.forEach((checkpoint) => { checkpoint.status = 'pending'; });
  return report;
}

function partialTimeoutReport() {
  const report = structuredClone(buildDemoPatrolReport());
  report.run.state = 'failed';
  report.run.outcome = 'timeout';
  report.task.completed_checkpoint_count = 2;
  report.task.progress_ratio = 0.5;
  report.task.progress_percent = 50;
  report.result.outcome = 'timeout';
  report.result.success = false;
  report.result.failure_reason = 'maximum control-step budget reached';
  report.result.checkpoints_visited = 2;
  report.result.sample_count = 18;
  report.result.alarms_raised = 1;
  report.result.alarms_cleared = 0;
  report.result.highest_event_severity = 'warning';
  report.inspection.sample_count = 18;
  report.inspection.visit_count = 2;
  report.inspection.visits = report.inspection.visits.slice(0, 2);
  report.alarms.history_count = 1;
  report.alarms.history = report.alarms.history.slice(0, 1);
  report.map.checkpoints.forEach((checkpoint, index) => {
    checkpoint.status = index < 2 ? 'completed' : (index === 2 ? 'current' : 'pending');
  });
  return report;
}

test('demo report is one validated standalone patrol-report/v1 with complete point evidence', () => {
  const report = buildDemoPatrolReport();

  assert.equal(report.schema_version, 'patrol-report/v1');
  assert.equal(report.run.run_id, 'graduation-demo-seed7000-v1');
  assert.equal(report.inspection.available, true);
  assert.equal(report.inspection.sample_count, 36);
  assert.equal(report.inspection.visit_count, 4);
  assert.deepEqual(
    report.inspection.visits.map((visit) => visit.checkpoint_id),
    ['P1', 'P2', 'P3', 'P4'],
  );
  assert.deepEqual(report.alarms.history.map((event) => event.event_index), [1, 2, 3, 4, 5, 6]);
  assert.equal(report.map.dynamic_obstacles, undefined);
  assert.equal(report.map.local_goal, undefined);
  assert.equal(report.source.run_directory_name, undefined);
});

test('report response rejects unknown relay and storage fields at every public boundary', () => {
  const valid = buildDemoPatrolReport();
  const mutations = [
    (report) => { report.secret = 'root-private'; },
    (report) => { report.source.secret = 'source-private'; },
    (report) => { report.run.secret = 'run-private'; },
    (report) => { report.result.secret = 'result-private'; },
    (report) => { report.alarms.secret = 'alarms-private'; },
    (report) => { report.alarms.history[0].secret = 'event-private'; },
    (report) => { report.map.secret = 'map-private'; },
    (report) => { report.map.checkpoints[0].secret = 'checkpoint-private'; },
  ];
  mutations.forEach((mutate) => {
    const report = structuredClone(valid);
    mutate(report);
    assert.throws(
      () => validateReportResponse(response(report), valid.run.run_id),
      ReportProtocolError,
    );
  });

  assert.throws(
    () => validateReportResponse({ ...response(valid), trace: 'private' }, valid.run.run_id),
    ReportProtocolError,
  );
});

test('report validation fails closed on identity, timeline, inspection and map contradictions', () => {
  const valid = buildDemoPatrolReport();
  const mutations = [
    (report) => { report.schema_version = 'patrol-report/v999'; },
    (report) => { report.run.run_id = 'foreign-run'; },
    (report) => { report.run.runtime_identity.model_id = '/private/model.zip'; },
    (report) => { report.run.runtime_identity.model_id = 'C:\\private\\model.zip'; },
    (report) => { report.run.runtime_identity.model_id = 'models\\policy.zip'; },
    (report) => { report.run.runtime_identity.model_id = 'models:policy/policy.zip'; },
    (report) => { report.inspection.visits[1].simulation_time_s = 1; },
    (report) => { report.inspection.visits[0].readings.temperature.display_name = '室温'; },
    (report) => { report.inspection.visits[0].alarm_event_count = 1; },
    (report) => { report.alarms.history.reverse(); },
    (report) => { report.map.robot.x_m = 999; },
    (report) => { report.result.sample_count += 1; },
  ];
  mutations.forEach((mutate) => {
    const report = structuredClone(valid);
    mutate(report);
    assert.throws(
      () => validatePatrolReport(report, valid.run.run_id),
      ReportProtocolError,
    );
  });
  assert.throws(
    () => validateReportResponse({ ...response(valid), device_id: 'foreign-device' }, valid.run.run_id),
    ReportProtocolError,
  );
});

test('an active checkpoint can only be the immediate next unfinished route point', () => {
  const report = partialTimeoutReport();
  assert.equal(validatePatrolReport(report, report.run.run_id).map.checkpoints[2].status, 'current');

  const misplaced = structuredClone(report);
  misplaced.map.checkpoints[2].status = 'pending';
  misplaced.map.checkpoints[3].status = 'active';
  assert.throws(
    () => validatePatrolReport(misplaced, report.run.run_id),
    ReportProtocolError,
  );
});

test('disabled safety shield uses explicit null identity fields', () => {
  const report = structuredClone(buildDemoPatrolReport());
  report.run.runtime_identity.safety_shield_enabled = false;
  report.run.runtime_identity.safety_shield_id = null;
  report.run.runtime_identity.safety_shield_sha256 = null;
  assert.equal(validatePatrolReport(report).run.runtime_identity.safety_shield_id, null);

  const forged = structuredClone(report);
  forged.run.runtime_identity.safety_shield_id = 'old-shield';
  assert.throws(() => validatePatrolReport(forged), ReportProtocolError);
});

test('a validated ordinary failure before P1 remains a readable zero-visit report', () => {
  const report = zeroVisitCollisionReport();
  const validated = validatePatrolReport(report, report.run.run_id);

  assert.equal(validated.inspection.available, true);
  assert.equal(validated.inspection.sample_count, 0);
  assert.equal(validated.inspection.visit_count, 0);
  assert.deepEqual(validated.inspection.visits, []);
});

test('cloud report service calls only getPatrolReport and hides SDK rejection details', async () => {
  const report = buildDemoPatrolReport();
  const calls = [];
  const service = createPatrolReportService({
    mode: 'cloud',
    callFunction(options) {
      calls.push(structuredClone(options));
      return Promise.resolve({ result: response(report) });
    },
  });

  const received = await service.fetchReport(report.run.run_id);
  assert.equal(received.run.run_id, report.run.run_id);
  assert.deepEqual(calls, [{
    name: 'getPatrolReport',
    data: { run_id: report.run.run_id },
  }]);

  const rejected = createPatrolReportService({
    mode: 'cloud',
    callFunction() {
      return Promise.reject(new Error('FUNCTION_NOT_FOUND trace=private path=/private')); 
    },
  });
  await assert.rejects(
    () => rejected.fetchReport(report.run.run_id),
    (error) => (
      error instanceof ReportProtocolError
      && error.message === '完整巡检报告暂时不可用，请稍后重试'
      && !/FUNCTION_NOT_FOUND|trace|private/.test(error.message)
    ),
  );
});

test('demo and local report services never call cloud and accept only the requested current run', async () => {
  const terminal = buildDashboardPayload(fixture, fixture.frames.length - 1);
  let cloudCalls = 0;
  const demo = createPatrolReportService({
    mode: 'demo',
    callFunction() { cloudCalls += 1; },
  });
  const local = createPatrolReportService({
    mode: 'local',
    callFunction() { cloudCalls += 1; },
    getDashboard: () => terminal,
  });

  assert.equal((await demo.fetchReport(fixture.run.run_id)).run.run_id, fixture.run.run_id);
  assert.equal((await local.fetchReport(fixture.run.run_id)).run.run_id, fixture.run.run_id);
  assert.equal(cloudCalls, 0);
  await assert.rejects(() => local.fetchReport('another-run'), ReportProtocolError);
});

test('report builder refuses running, truncated-alarm and legacy inspection snapshots', () => {
  assert.throws(
    () => buildPatrolReportFromDashboard(buildDashboardPayload(fixture, 0)),
    (error) => error instanceof ReportProtocolError && error.code === 'REPORT_NOT_READY',
  );

  const truncated = buildDashboardPayload(fixture, fixture.frames.length - 1);
  truncated.alarms.history_truncated = true;
  assert.throws(() => buildPatrolReportFromDashboard(truncated), ReportProtocolError);

  const legacy = buildDashboardPayload(fixture, fixture.frames.length - 1);
  delete legacy.inspection;
  assert.throws(() => buildPatrolReportFromDashboard(legacy), ReportProtocolError);
});

test('app registers the non-tab report page and its view states the simulation boundary', () => {
  const config = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../app.json'), 'utf8'));
  const template = fs.readFileSync(path.resolve(__dirname, '../pages/report/index.wxml'), 'utf8');

  assert.equal(config.pages.includes('pages/report/index'), true);
  assert.equal(config.tabBar.list.some((item) => item.pagePath === 'pages/report/index'), false);
  assert.match(template, /点位采样/);
  assert.match(template, /告警时间线/);
  assert.match(template, /不构成真实场地、消防、气体监测或机器人导航安全认证/);
  assert.doesNotMatch(template, /\sselectable(?:\s|=|>)/);
  assert.match(template, /user-select="true"/);
});

test('report page projects P2-P4 alarms and a friendly zero-visit failure state', () => {
  global.Page = () => {};
  global.getApp = () => ({ globalData: { dashboard: null } });
  global.wx = {};
  const modulePath = '../pages/report/index.js';
  delete require.cache[require.resolve(modulePath)];
  const { projectReportView } = require(modulePath);
  const demoView = projectReportView(buildDemoPatrolReport(), 'demo');

  assert.equal(demoView.hasVisits, true);
  assert.deepEqual(demoView.visits.map((visit) => visit.checkpointId), ['P1', 'P2', 'P3', 'P4']);
  assert.deepEqual(
    [...new Set(demoView.alarms.map((event) => event.checkpointText))],
    ['P2', 'P3', 'P4'],
  );
  assert.equal(demoView.alarms.length, 6);

  const emptyView = projectReportView(zeroVisitCollisionReport(), 'cloud');
  assert.equal(emptyView.hasVisits, false);
  assert.equal(emptyView.sampleSummary, '未到达巡检点 · 无点位采样');
});
