"use strict";

const { getPlatform, resolveModeForPlatform } = require("../../services/modePolicy");
const { formatNumber } = require("../../utils/format");
const { buildAvoidanceView, emptyAvoidanceView } = require("../../utils/avoidanceView");

const app = getApp();

const CHECKPOINT_STATUS_LABELS = {
  completed: "已完成",
  current: "当前点",
  active: "当前点",
  pending: "待巡检",
};

const SENSOR_LABELS = {
  temperature: "温度",
  smoke_obscuration: "烟雾遮蔽率",
  carbon_monoxide: "一氧化碳浓度",
};

const FAILURE_REASON_LABELS = {
  "physical contact detected by Webots contact points": "Webots 接触点检测到物理碰撞",
  "physical contact during inspection dwell": "巡检驻留期间发生物理碰撞",
  "maximum control-step budget reached": "达到最大控制步数",
};

function localizeAlarmMessage(message) {
  return String(message || "异常指标超过阈值")
    .replace(" alarm raised: ", "告警触发：")
    .replace(" alarm cleared: ", "告警解除：")
    .replace(" (threshold ", "（阈值 ")
    .replace(/\)$/, "）");
}

function formatReceivedTime(value) {
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return "等待首个快照";
  const date = new Date(timestamp);
  return `最近入云 ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}:${String(date.getSeconds()).padStart(2, "0")}`;
}

function sensorState(key, value) {
  if (typeof value !== "number") return "等待数据";
  if (key === "temperature") return value >= 38 ? "温度偏高" : "正常范围";
  if (key === "smoke_obscuration") return value >= 5 ? "烟雾告警" : "正常范围";
  return value >= 50 ? "浓度告警" : "正常范围";
}

function buildSensorCards(sensors) {
  const source = sensors || {};
  const items = [
    {
      key: "temperature",
      label: "温度",
      icon: "℃",
      tone: "warm",
      reading: source.temperature || {},
      digits: 1,
    },
    {
      key: "smoke_obscuration",
      label: "烟雾",
      icon: "≈",
      tone: "cool",
      reading: source.smoke || {},
      digits: 2,
    },
    {
      key: "carbon_monoxide",
      label: "一氧化碳",
      icon: "CO",
      tone: "mint",
      reading: source.carbon_monoxide || {},
      digits: 0,
    },
  ];
  return items.map((item) => ({
    key: item.key,
    label: item.label,
    icon: item.icon,
    tone: item.tone,
    value: formatNumber(item.reading.value, item.digits, "—"),
    unit: item.reading.unit || "",
    state: sensorState(item.key, item.reading.value),
  }));
}

function buildActiveAlarms(payload) {
  const alarms = payload.alarms && Array.isArray(payload.alarms.active)
    ? payload.alarms.active
    : [];
  return alarms.map((alarm) => ({
    ...alarm,
    title: SENSOR_LABELS[alarm.sensor_id] || alarm.sensor_id || "环境告警",
    checkpoint: alarm.checkpoint_id
      || (payload.sensors && payload.sensors.checkpoint_id)
      || "关联点位",
    message: localizeAlarmMessage(alarm.message),
    valueText: `${formatNumber(alarm.observed_value, 2, "—")} ${alarm.unit || ""} / 阈值 ${formatNumber(alarm.threshold, 2, "—")} ${alarm.unit || ""}`,
  }));
}

function buildDemoPresentation(sourceMeta, terminal) {
  const replay = sourceMeta && sourceMeta.replay ? sourceMeta.replay : {};
  const keyEvent = replay.keyEvent || null;
  const playbackRate = replay.playbackRate === 0.5 ? 0.5 : 1;
  return {
    demoPlaybackRate: playbackRate,
    demoHalfSpeedActive: playbackRate === 0.5,
    demoNormalSpeedActive: playbackRate === 1,
    demoKeyEventActive: Boolean(keyEvent),
    demoKeyEventTone: keyEvent ? keyEvent.tone : "normal",
    demoKeyEventLabel: keyEvent ? keyEvent.label : "答辩演示回放",
    demoKeyEventDescription: keyEvent
      ? keyEvent.description
      : "关键避障与环境告警事件将自动停留，便于观察和讲解",
    demoKeyEventPositionText: keyEvent
      ? `关键事件 ${keyEvent.ordinal} / ${keyEvent.total} · ${formatNumber(keyEvent.simulationTimeS, 1, "0.0")} s`
      : `${Number(replay.keyEventCount) || 0} 个关键事件`,
    demoNextEventDisabled: terminal === true,
  };
}

function buildView(payload, mode) {
  const run = payload.run || {};
  const task = payload.task || {};
  const map = payload.map || null;
  const sensors = payload.sensors || {};
  const alarms = payload.alarms || {};
  const result = payload.result || {};
  const completed = Number(task.completed_checkpoint_count) || 0;
  const total = Number(task.checkpoint_count) || 0;
  const progress = Math.max(0, Math.min(100, Number(task.progress_percent) || 0));
  const checkpoints = map && Array.isArray(map.checkpoints)
    ? map.checkpoints.map((checkpoint, index) => {
      const status = checkpoint.status === "active" ? "current" : checkpoint.status;
      return {
        ...checkpoint,
        ordinal: checkpoint.ordinal || index + 1,
        status,
        statusLabel: CHECKPOINT_STATUS_LABELS[status] || status,
      };
    })
    : [];
  const activeAlarms = buildActiveAlarms(payload);
  const success = result.available === true && result.success === true;
  const boundary = payload.source && payload.source.map_boundary_note;

  return {
    ...buildAvoidanceView(payload.avoidance, map),
    dashboardMap: map,
    runId: run.run_id || "",
    sequenceText: Number.isSafeInteger(run.sequence) ? `SEQ ${run.sequence}` : "",
    terminal: run.terminal === true,
    progressText: `${formatNumber(progress, 0, "0")}%`,
    progressWidth: progress,
    progressStyle: `width: ${progress}%;`,
    simulationTimeText: `${formatNumber(run.simulation_time_s, 1, "0.0")} s`,
    currentCheckpointText: task.current_checkpoint_id || (run.terminal ? "完成" : "—"),
    completedCheckpointText: `${completed} / ${total} 巡检点`,
    checkpoints,
    sensorAvailable: sensors.available === true,
    sensorSourceText: sensors.available
      ? `${sensors.provenance_label || "上次采样"} · ${sensors.checkpoint_id || "未知点位"}`
      : "等待机器人到达巡检点",
    sensorCards: buildSensorCards(sensors),
    activeAlarmCount: Number(alarms.active_count) || 0,
    activeAlarms,
    resultAvailable: result.available === true,
    resultSuccess: success,
    resultTitle: success ? "巡检任务顺利完成" : "巡检任务异常结束",
    resultSummary: result.available
      ? `${result.checkpoints_visited || 0}/${result.checkpoints_total || total} 点位 · ${result.collision_count || 0} 次碰撞 · ${result.sample_count || 0} 条采样`
      : "",
    mapModeLabel: run.terminal ? "FINAL" : (mode === "demo" ? "REPLAY" : "LIVE"),
    boundaryNote: boundary || "本二维地图来自仿真配置，仅用于毕设演示可视化。",
    modeDescription: mode === "demo"
      ? "内置已校验巡检证据，支持离线动态回放"
      : (mode === "cloud"
        ? "微信云开发只读同步 Mac 仿真数据"
        : "微信开发者工具连接 Mac 本机只读接口"),
  };
}

Page({
  data: {
    mode: "cloud",
    modeLabel: "云端实时",
    modeDescription: "微信云开发只读同步 Mac 仿真数据",
    connectionLabel: "正在准备巡检数据",
    connectionTone: "warning",
    errorMessage: "",
    demoControlLabel: "暂停演示",
    runId: "",
    sequenceText: "",
    cloudTransportLabel: "实时订阅",
    cloudReceivedText: "等待首个快照",
    terminal: false,
    progressText: "0%",
    progressWidth: 0,
    progressStyle: "width: 0%;",
    simulationTimeText: "0.0 s",
    currentCheckpointText: "—",
    completedCheckpointText: "0 / 4 巡检点",
    dashboardMap: null,
    mapModeLabel: "REPLAY",
    checkpoints: [],
    sensorAvailable: false,
    sensorSourceText: "等待机器人到达巡检点",
    sensorCards: buildSensorCards(null),
    activeAlarmCount: 0,
    activeAlarms: [],
    resultAvailable: false,
    resultSuccess: false,
    resultTitle: "",
    resultSummary: "",
    boundaryNote: "本二维地图来自仿真配置，仅用于毕设演示可视化。",
    ...buildDemoPresentation(null, false),
    ...emptyAvoidanceView(0),
  },

  onLoad() {
    this._visible = false;
    this._sourceMode = null;
    this._replaceSource(app.globalData.mode);
    this._unsubscribePatrol = app.subscribePatrol(this._callbacks());
  },

  onShow() {
    this._visible = true;
    const resolution = resolveModeForPlatform(app.globalData.mode, getPlatform(wx));
    const mode = resolution.mode;
    if (resolution.downgraded) app.setMode(mode);
    if (this._sourceMode !== mode) this._replaceSource(mode);
    this._syncSessionState();
  },

  onHide() {
    this._visible = false;
  },

  onUnload() {
    if (this._unsubscribePatrol) this._unsubscribePatrol();
    this._unsubscribePatrol = null;
  },

  onPullDownRefresh() {
    if (this._sourceMode === "demo") {
      this.resetDemo();
    } else {
      app.resetPatrolSession();
    }
    wx.stopPullDownRefresh();
  },

  _replaceSource(mode) {
    this._sourceMode = mode;
    const modeLabel = mode === "demo" ? "仿真演示" : (mode === "cloud" ? "云端实时" : "本机实时");
    this.setData({
      mode,
      modeLabel,
      modeDescription: mode === "demo"
        ? "内置已校验巡检证据，支持离线动态回放"
        : (mode === "cloud"
          ? "微信云开发只读同步 Mac 仿真数据"
          : "微信开发者工具连接 Mac 本机只读接口"),
      mapModeLabel: mode === "demo" ? "REPLAY" : "LIVE",
      errorMessage: "",
      demoControlLabel: "暂停演示",
      cloudReceivedText: "等待首个快照",
      ...buildDemoPresentation(null, false),
    });
  },

  _clearView(mode) {
    this._replaceSource(mode || app.globalData.mode);
    this.setData({
      runId: "",
      sequenceText: "",
      terminal: false,
      progressText: "0%",
      progressWidth: 0,
      progressStyle: "width: 0%;",
      simulationTimeText: "0.0 s",
      currentCheckpointText: "—",
      completedCheckpointText: "0 / 4 巡检点",
      dashboardMap: null,
      checkpoints: [],
      sensorAvailable: false,
      sensorSourceText: "等待机器人到达巡检点",
      sensorCards: buildSensorCards(null),
      activeAlarmCount: 0,
      activeAlarms: [],
      resultAvailable: false,
      resultSuccess: false,
      resultTitle: "",
      resultSummary: "",
      boundaryNote: "本二维地图来自仿真配置，仅用于毕设演示可视化。",
      ...buildDemoPresentation(null, false),
      ...emptyAvoidanceView(0),
      cloudReceivedText: "等待首个快照",
      errorMessage: "",
    });
  },

  _applyPayload(payload, sourceMeta) {
    const sourceMode = (sourceMeta && sourceMeta.mode) || app.globalData.mode;
    this._sourceMode = sourceMode;
    const view = buildView(payload, sourceMode);
    this.setData({
      ...view,
      ...(sourceMode === "demo"
        ? buildDemoPresentation(sourceMeta, payload.run && payload.run.terminal === true)
        : buildDemoPresentation(null, false)),
      errorMessage: "",
      cloudReceivedText: sourceMeta && sourceMeta.receivedAtUtc
        ? formatReceivedTime(sourceMeta.receivedAtUtc)
        : this.data.cloudReceivedText,
    });
  },

  _applyError(error, details) {
    if (this._sourceMode === "cloud" && details && details.retrying) {
      this.setData({
        errorMessage: "",
        connectionLabel: details.transport === "polling"
          ? "实时订阅中断，已自动切换轮询"
          : "云端连接中断，正在恢复",
        connectionTone: "warning",
        cloudTransportLabel: "轮询保障",
      });
      return;
    }
    this.setData({
      errorMessage: error && error.message ? error.message : "巡检数据读取失败",
      connectionLabel: details && details.retrying ? "连接中断，正在重试" : "数据协议校验失败",
      connectionTone: "offline",
    });
  },

  _syncSessionState() {
    const state = app.getPatrolSessionState();
    if (!state) return;
    if (state.payload) this._applyPayload(state.payload, state.meta);
    else this._clearView(state.mode);
    this._handleStatus(state.statusEvent || { status: state.status, mode: state.mode });
    if (state.mode === "demo" && state.sourceState) {
      const playbackRate = state.sourceState.playbackRate === 0.5 ? 0.5 : 1;
      this.setData({
        demoPlaybackRate: playbackRate,
        demoHalfSpeedActive: playbackRate === 0.5,
        demoNormalSpeedActive: playbackRate === 1,
      });
    }
    if (state.error) this._applyError(state.error, state.errorDetails);
  },

  _callbacks() {
    return {
      onClear: (event) => {
        if (this._visible) this._clearView(event.mode);
      },
      onData: (payload, sourceMeta) => {
        if (this._visible) this._applyPayload(payload, sourceMeta);
      },
      onStatus: (event) => {
        if (this._visible) this._handleStatus(event);
      },
      onError: (error, details) => {
        if (this._visible) this._applyError(error, details);
      },
    };
  },

  _handleStatus(event) {
    const status = event && event.status;
    if (status === "terminal") {
      this.setData({
        connectionLabel: "终态结果已校验",
        connectionTone: "online",
        demoControlLabel: "重新演示",
      });
      return;
    }
    if (status === "new_run") {
      this.setData({
        connectionLabel: "检测到新的巡检任务，正在切换",
        connectionTone: "warning",
      });
      return;
    }
    if (status === "backoff") {
      this.setData({ connectionLabel: "连接中断，正在重试", connectionTone: "offline" });
      return;
    }
    if (status === "connecting" || status === "watching") {
      this.setData({
        connectionLabel: status === "connecting" ? "正在连接云端巡检数据" : "云端实时订阅已建立",
        connectionTone: "warning",
        cloudTransportLabel: "实时订阅",
      });
      return;
    }
    if (status === "polling") {
      this.setData({
        connectionLabel: "实时订阅中断，已自动切换轮询",
        connectionTone: "warning",
        cloudTransportLabel: "轮询保障",
      });
      return;
    }
    if (status === "stale") {
      this.setData({
        connectionLabel: "云端数据停滞：8 秒未收到新序号",
        connectionTone: "offline",
        cloudTransportLabel: "数据停滞",
      });
      return;
    }
    if (status === "requesting") {
      this.setData({ connectionLabel: "正在读取本机巡检数据", connectionTone: "warning" });
      return;
    }
    if (status === "error") {
      this.setData({ connectionLabel: "数据协议校验失败", connectionTone: "offline" });
      return;
    }
    if (status === "stopped") {
      this.setData({
        connectionLabel: this._sourceMode === "demo" ? "演示已暂停" : "实时连接已暂停",
        connectionTone: "warning",
      });
      return;
    }
    if (status === "idle") {
      this.setData({
        connectionLabel: "正在准备巡检数据",
        connectionTone: "warning",
      });
      return;
    }
    this.setData({
      connectionLabel: this._sourceMode === "demo"
        ? "仿真证据回放中"
        : (this._sourceMode === "cloud" ? "云端实时数据已同步" : "本机实时数据已连接"),
      connectionTone: "online",
      cloudTransportLabel: event && event.transport === "polling" ? "轮询保障" : "实时订阅",
    });
  },

  toggleDemo() {
    if (this._sourceMode !== "demo") return;
    const state = app.getPatrolSessionState();
    if (!state) return;
    if (state.status === "terminal") {
      this.resetDemo();
      return;
    }
    if (state.started) {
      app.pausePatrolSession();
      this.setData({ demoControlLabel: "继续演示" });
    } else {
      app.resumePatrolSession();
      this.setData({ demoControlLabel: "暂停演示" });
    }
  },

  resetDemo() {
    if (this._sourceMode !== "demo") return;
    app.setDashboard(null, { mode: "demo" });
    this.setData({
      progressText: "0%",
      progressWidth: 0,
      progressStyle: "width: 0%;",
      simulationTimeText: "0.0 s",
      currentCheckpointText: "P1",
      resultAvailable: false,
      activeAlarmCount: 0,
      activeAlarms: [],
      errorMessage: "",
    });
    app.resetPatrolSession();
  },

  jumpToNextDemoEvent() {
    if (this._sourceMode !== "demo" || this.data.terminal) return;
    const state = app.getPatrolSessionState();
    if (!state || !state.started) {
      wx.showToast({ title: "请先继续演示", icon: "none" });
      return;
    }
    const payload = app.jumpToNextDemoKeyEvent();
    if (!payload) wx.showToast({ title: "已到最后一个关键事件", icon: "none" });
  },

  setDemoPlaybackRate(event) {
    if (this._sourceMode !== "demo") return;
    const rate = Number(event.currentTarget.dataset.rate);
    const appliedRate = app.setDemoPlaybackRate(rate);
    if (appliedRate === null) return;
    this.setData({
      demoPlaybackRate: appliedRate,
      demoHalfSpeedActive: appliedRate === 0.5,
      demoNormalSpeedActive: appliedRate === 1,
    });
  },

  chooseMode() {
    wx.showActionSheet({
      itemList: ["云端实时同步（推荐）", "仿真演示回放", "电脑本地实时接口"],
      success: ({ tapIndex }) => {
        const requestedMode = tapIndex === 0 ? "cloud" : (tapIndex === 1 ? "demo" : "local");
        const resolution = resolveModeForPlatform(requestedMode, getPlatform(wx));
        const nextMode = resolution.mode;
        if (resolution.downgraded) {
          app.setMode("cloud");
          wx.showModal({
            title: "已切换到云端实时",
            content: "真机无法通过 127.0.0.1 访问 Mac，本机模式已自动降级为安全的云端只读模式。",
            showCancel: false,
          });
        }
        if (nextMode === this._sourceMode) return;
        app.setMode(nextMode);
        this._replaceSource(nextMode);
      },
    });
  },

  openAlarms() {
    wx.switchTab({ url: "/pages/alarms/index" });
  },

  openResult() {
    if (!this.data.resultAvailable) return;
    wx.navigateTo({ url: "/pages/result/index" });
  },
});

module.exports = {
  buildDemoPresentation,
  buildSensorCards,
  buildView,
  formatReceivedTime,
  localizeAlarmMessage,
  localizeFailureReason(reason) {
    return FAILURE_REASON_LABELS[reason] || reason || "无";
  },
};
