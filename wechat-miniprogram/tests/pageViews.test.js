"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const fixture = require("../data/graduation-demo-seed7000-v1.js");
const { buildDashboardPayload } = require("../services/offlineReplay");

global.getApp = () => ({
  globalData: {},
  setDashboard() {},
  setMode() {},
});
global.Page = () => {};
global.wx = {};

const { buildView, formatReceivedTime } = require("../pages/dashboard/index.js");
const { filterEvents, projectEvent } = require("../pages/alarms/index.js");
const { projectResult, shortHash } = require("../pages/result/index.js");

test("dashboard view projects the verified terminal result", () => {
  const payload = buildDashboardPayload(fixture, fixture.frames.length - 1);
  const view = buildView(payload, "demo");
  assert.equal(view.progressText, "100%");
  assert.equal(view.progressStyle, "width: 100%;");
  assert.equal(view.sequenceText, "SEQ 822");
  assert.equal(view.currentCheckpointText, "完成");
  assert.equal(view.resultAvailable, true);
  assert.equal(view.resultSuccess, true);
  assert.equal(view.sensorCards[0].value, "23.9");
  assert.equal(view.checkpoints.every((item) => item.status === "completed"), true);
});

test("dashboard view identifies CloudBase as a live read-only source", () => {
  const payload = buildDashboardPayload(fixture, 0);
  const view = buildView(payload, "cloud");
  assert.equal(view.mapModeLabel, "LIVE");
  assert.match(view.modeDescription, /微信云开发只读同步/);
  assert.match(formatReceivedTime("2026-09-01T06:00:00.000Z"), /^最近入云 \d{2}:\d{2}:\d{2}$/);
  assert.equal(formatReceivedTime("invalid"), "等待首个快照");
});

test("dashboard view localizes active alarms and current checkpoint", () => {
  const frameIndex = fixture.frames.findIndex((frame) => frame.active_alerts.length > 0);
  const payload = buildDashboardPayload(fixture, frameIndex);
  const view = buildView(payload, "demo");
  assert.equal(view.resultAvailable, false);
  assert.equal(view.activeAlarmCount > 0, true);
  assert.match(view.activeAlarms[0].message, /告警触发/);
  assert.equal(view.checkpoints.some((item) => item.status === "current"), true);
});

test("alarm page filters projected event types", () => {
  const projected = fixture.alarm_timeline.map(projectEvent);
  assert.equal(filterEvents(projected, "raised").length, 3);
  assert.equal(filterEvents(projected, "cleared").length, 3);
  assert.equal(filterEvents(projected, "critical").length, 4);
  assert.match(projected[0].message, /告警触发/);
});

test("result page exposes auditable success metrics", () => {
  const payload = buildDashboardPayload(fixture, fixture.frames.length - 1);
  const view = projectResult(payload);
  assert.equal(view.resultTone, "success");
  assert.equal(view.metrics[0].value, "4 / 4");
  assert.equal(view.metrics[1].value, "0");
  assert.equal(view.avoidanceEvidenceAvailable, true);
  assert.equal(view.avoidanceMetrics[0].value, "0.286 m");
  assert.equal(view.avoidanceMetrics[2].value, "243 步");
  assert.equal(view.avoidanceMetrics[3].value, "4 / 4");
  assert.equal(view.avoidanceMetrics[4].value, "0.52 ms");
  assert.equal(view.failureReason, "无");
  assert.match(shortHash(fixture.run.runtime_identity.model_sha256), /…/);
});
