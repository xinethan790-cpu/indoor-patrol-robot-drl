"use strict";

const { formatNumber } = require("../../utils/format");

const SENSOR_LABELS = {
  temperature: "温度",
  smoke_obscuration: "烟雾遮蔽率",
  carbon_monoxide: "一氧化碳浓度",
};

const SEVERITY_LABELS = {
  info: "提示",
  warning: "警告",
  critical: "严重",
};

function localizeMessage(message) {
  return String(message || "")
    .replace(" alarm raised: ", "告警触发：")
    .replace(" alarm cleared: ", "告警解除：")
    .replace(" (threshold ", "（阈值 ")
    .replace(/\)$/, "）");
}

function projectEvent(event) {
  const raised = event.event_type === "raised";
  const severity = event.severity || "info";
  return {
    ...event,
    tone: raised ? `raised-${severity}` : "cleared",
    eventLabel: raised ? "触发" : "解除",
    severityLabel: SEVERITY_LABELS[severity] || severity,
    sensorLabel: SENSOR_LABELS[event.sensor_id] || event.sensor_id || "环境指标",
    checkpointLabel: event.checkpoint_id ? `${event.checkpoint_id} 巡检点` : "未知点位",
    timeText: `${formatNumber(event.simulation_time_s, 1, "—")} s`,
    observedText: `${formatNumber(event.observed_value, 2, "—")} ${event.unit || ""}`,
    thresholdText: `${formatNumber(event.threshold, 2, "—")} ${event.unit || ""}`,
    message: localizeMessage(event.message),
  };
}

function filterEvents(events, filter) {
  if (filter === "raised") return events.filter((item) => item.event_type === "raised");
  if (filter === "cleared") return events.filter((item) => item.event_type === "cleared");
  if (filter === "critical") return events.filter((item) => item.severity === "critical");
  return events;
}

function sourcePresentation(mode, status) {
  if (mode === "cloud") {
    if (status === "stale") return { label: "云端数据停滞 · 显示最后异常记录", tone: "offline" };
    if (status === "stopped") return { label: "云端同步已暂停 · 显示最后异常记录", tone: "demo" };
    return { label: "云端实时异常记录", tone: "online" };
  }
  if (mode === "local") return { label: "本机实时异常记录", tone: "online" };
  return { label: "已校验仿真事件记录", tone: "demo" };
}

Page({
  data: {
    filters: [
      { value: "all", label: "全部" },
      { value: "raised", label: "触发" },
      { value: "cleared", label: "已解除" },
      { value: "critical", label: "严重" },
    ],
    activeFilter: "all",
    allEvents: [],
    filteredEvents: [],
    historyCount: 0,
    sourceLabel: "等待巡检数据",
    modeTone: "demo",
  },

  onLoad() {
    const app = getApp();
    this._visible = false;
    this._unsubscribePatrol = app.subscribePatrol({
      onClear: (event) => {
        if (this._visible) this._renderPayload(null, { mode: event.mode, status: "idle" });
      },
      onData: (payload, meta) => {
        if (this._visible) this._renderPayload(payload, meta);
      },
      onStatus: (event) => {
        if (!this._visible) return;
        const presentation = sourcePresentation(event.mode || app.globalData.mode, event.status);
        this.setData({ sourceLabel: presentation.label, modeTone: presentation.tone });
      },
    });
  },

  onShow() {
    const app = getApp();
    this._visible = true;
    this._renderPayload(app.globalData.dashboard, app.globalData.dashboardMeta);
  },

  onHide() {
    this._visible = false;
  },

  onUnload() {
    if (this._unsubscribePatrol) this._unsubscribePatrol();
    this._unsubscribePatrol = null;
  },

  _renderPayload(payload, meta) {
    const app = getApp();
    const mode = (meta && meta.mode) || app.globalData.mode;
    const sessionState = app.getPatrolSessionState ? app.getPatrolSessionState() : null;
    const status = (meta && meta.status)
      || (sessionState && sessionState.status)
      || "idle";
    const rawEvents = payload && payload.alarms && Array.isArray(payload.alarms.history)
      ? payload.alarms.history
      : [];
    const allEvents = rawEvents.map(projectEvent);
    const presentation = sourcePresentation(mode, status);
    this.setData({
      allEvents,
      filteredEvents: filterEvents(allEvents, this.data.activeFilter),
      historyCount: rawEvents.length,
      sourceLabel: presentation.label,
      modeTone: presentation.tone,
    });
  },

  selectFilter(event) {
    const filter = event.currentTarget.dataset.value || "all";
    this.setData({
      activeFilter: filter,
      filteredEvents: filterEvents(this.data.allEvents, filter),
    });
  },
});

module.exports = { filterEvents, projectEvent, sourcePresentation };
