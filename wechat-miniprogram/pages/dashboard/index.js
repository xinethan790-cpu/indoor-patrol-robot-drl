"use strict";

const {
  createOfflineDataSource,
  createPollingDataSource,
} = require("../../services/createDataSource");
const { formatNumber } = require("../../utils/format");

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

function isDevTools() {
  try {
    const info = typeof wx.getDeviceInfo === "function"
      ? wx.getDeviceInfo()
      : wx.getSystemInfoSync();
    return info.platform === "devtools";
  } catch (error) {
    return false;
  }
}

function localizeAlarmMessage(message) {
  return String(message || "异常指标超过阈值")
    .replace(" alarm raised: ", "告警触发：")
    .replace(" alarm cleared: ", "告警解除：")
    .replace(" (threshold ", "（阈值 ")
    .replace(/\)$/, "）");
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
    dashboardMap: map,
    runId: run.run_id || "",
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
      : "微信开发者工具连接 Mac 本机只读接口",
  };
}

Page({
  data: {
    mode: "demo",
    modeLabel: "仿真演示",
    modeDescription: "内置已校验巡检证据，支持离线动态回放",
    connectionLabel: "正在准备巡检数据",
    connectionTone: "warning",
    errorMessage: "",
    demoControlLabel: "暂停演示",
    runId: "",
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
  },

  onLoad() {
    this._source = null;
    this._sourceMode = null;
    this._sourceStarted = false;
  },

  onShow() {
    const mode = app.globalData.mode === "local" ? "local" : "demo";
    if (!this._source || this._sourceMode !== mode) {
      this._replaceSource(mode);
    }
    if (!this._sourceStarted) this._startSource();
  },

  onHide() {
    this._stopSource();
  },

  onUnload() {
    this._stopSource();
    this._source = null;
  },

  onPullDownRefresh() {
    if (this._sourceMode === "demo") {
      this.resetDemo();
    } else {
      this._replaceSource("local");
      this._startSource();
    }
    wx.stopPullDownRefresh();
  },

  _replaceSource(mode) {
    this._stopSource();
    this._sourceMode = mode;
    const modeLabel = mode === "demo" ? "仿真演示" : "本机实时";
    this.setData({
      mode,
      modeLabel,
      modeDescription: mode === "demo"
        ? "内置已校验巡检证据，支持离线动态回放"
        : "微信开发者工具连接 Mac 本机只读接口",
      mapModeLabel: mode === "demo" ? "REPLAY" : "LIVE",
      errorMessage: "",
      demoControlLabel: "暂停演示",
    });
    if (mode === "demo") {
      this._source = createOfflineDataSource();
    } else {
      const base = app.globalData.apiBaseUrl || "http://127.0.0.1:8765";
      this._source = createPollingDataSource({
        endpoint: `${base}/api/dashboard`,
        intervalMs: 900,
        timeoutMs: 5000,
      });
    }
  },

  _callbacks() {
    return {
      onData: (payload) => {
        const view = buildView(payload, this._sourceMode);
        this.setData({ ...view, errorMessage: "" });
        app.setDashboard(payload, { mode: this._sourceMode, receivedAt: Date.now() });
      },
      onStatus: (event) => this._handleStatus(event),
      onError: (error, details) => {
        this.setData({
          errorMessage: error && error.message ? error.message : "巡检数据读取失败",
          connectionLabel: details && details.retrying ? "连接中断，正在重试" : "数据协议校验失败",
          connectionTone: "offline",
        });
      },
    };
  },

  _startSource() {
    if (!this._source) return;
    this._sourceStarted = true;
    this._source.start(this._callbacks());
    if (this._sourceMode === "demo") this.setData({ demoControlLabel: "暂停演示" });
  },

  _stopSource() {
    if (this._source && this._sourceStarted) this._source.stop();
    this._sourceStarted = false;
  },

  _handleStatus(event) {
    const status = event && event.status;
    if (status === "terminal") {
      this._sourceStarted = false;
      this.setData({
        connectionLabel: "终态结果已校验",
        connectionTone: "online",
        demoControlLabel: "重新演示",
      });
      return;
    }
    if (status === "backoff") {
      this.setData({ connectionLabel: "连接中断，正在重试", connectionTone: "offline" });
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
    this.setData({
      connectionLabel: this._sourceMode === "demo" ? "仿真证据回放中" : "本机实时数据已连接",
      connectionTone: "online",
    });
  },

  toggleDemo() {
    if (!this._source || this._sourceMode !== "demo") return;
    const state = this._source.getState();
    if (state.status === "terminal") {
      this.resetDemo();
      return;
    }
    if (this._sourceStarted) {
      this._stopSource();
      this.setData({ demoControlLabel: "继续演示" });
    } else {
      this._startSource();
    }
  },

  resetDemo() {
    if (!this._source || this._sourceMode !== "demo") return;
    this._stopSource();
    this._source.reset();
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
    this._startSource();
  },

  chooseMode() {
    wx.showActionSheet({
      itemList: ["仿真演示回放（推荐）", "电脑本地实时接口"],
      success: ({ tapIndex }) => {
        const nextMode = tapIndex === 1 ? "local" : "demo";
        if (nextMode === "local" && !isDevTools()) {
          wx.showModal({
            title: "仅供开发者工具使用",
            content: "真机无法通过 127.0.0.1 访问 Mac。请先使用仿真演示模式，配置公网 HTTPS 服务后再启用真机实时数据。",
            showCancel: false,
          });
          return;
        }
        if (nextMode === this._sourceMode) return;
        app.setMode(nextMode);
        app.setDashboard(null, { mode: nextMode });
        this._replaceSource(nextMode);
        this._startSource();
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
  buildSensorCards,
  buildView,
  localizeAlarmMessage,
  localizeFailureReason(reason) {
    return FAILURE_REASON_LABELS[reason] || reason || "无";
  },
};
