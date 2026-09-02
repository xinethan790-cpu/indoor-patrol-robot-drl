"use strict";

const {
  HistoryProtocolError,
  STALE_AFTER_MS,
  createPatrolHistoryService,
} = require("../../services/patrolHistory");
const {
  ReportProtocolError,
  buildDemoPatrolReport,
  buildPatrolReportFromDashboard,
} = require("../../services/patrolReport");
const { normalizeMode } = require("../../services/modePolicy");

const AUTO_REFRESH_MIN_INTERVAL_MS = 3000;
const ACTIVE_REFRESH_MIN_DELAY_MS = 1000;
const HISTORY_UNAVAILABLE_MESSAGE = "巡检历史暂时不可用，请稍后重试";

const STATUS_PRESENTATION = Object.freeze({
  active: { label: "正在巡检", tone: "active", description: "仿真端仍在持续上报" },
  aborted: { label: "异常中止", tone: "critical", description: "控制器已上报最小中止证据" },
  interrupted: {
    label: "上报失联",
    tone: "warning",
    description: "超过 8 秒未收到新快照，无法判断任务是否仍在运行",
  },
  unknown_terminal: { label: "旧版终态", tone: "legacy", description: "终态类型未回填" },
});

const TERMINAL_OUTCOME_LABELS = Object.freeze({
  collision: "完整终态 · 检测到碰撞",
  safety_stop: "完整终态 · 安全保护停止",
  timeout: "完整终态 · 任务超时",
  inspection_failed: "完整终态 · 巡检采集失败",
  error: "完整终态 · 任务失败",
});

function historyPresentation(item) {
  if (item.status !== "completed") {
    return STATUS_PRESENTATION[item.status] || STATUS_PRESENTATION.interrupted;
  }
  if (item.outcome === "success" && item.result_validated === true) {
    return { label: "正常完成", tone: "success", description: "完整终态 · 巡检任务成功" };
  }
  if (item.result_validated !== true) {
    return { label: "任务已结束", tone: "terminal", description: "终态未验证" };
  }
  const outcome = item.outcome || item.result_outcome;
  return {
    label: "任务已结束",
    tone: "terminal",
    description: TERMINAL_OUTCOME_LABELS[outcome] || "已形成完整终态记录",
  };
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  const pad = (number) => String(number).padStart(2, "0");
  return `${date.getMonth() + 1}月${pad(date.getDate())}日 ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function safeErrorMessage(error) {
  return error instanceof HistoryProtocolError
    ? error.message
    : HISTORY_UNAVAILABLE_MESSAGE;
}

function formatDuration(first, latest) {
  const durationMs = Date.parse(latest) - Date.parse(first);
  if (!Number.isFinite(durationMs) || durationMs < 0) return "—";
  const totalSeconds = Math.round(durationMs / 1000);
  if (totalSeconds < 60) return `${totalSeconds} 秒`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return seconds ? `${minutes} 分 ${seconds} 秒` : `${minutes} 分`;
}

function projectHistoryCard(item, options) {
  const settings = options || {};
  const presentation = historyPresentation(item);
  const checkpointText = item.completed_checkpoint_count === null
    || item.checkpoint_count === null
    ? "—"
    : `${item.completed_checkpoint_count} / ${item.checkpoint_count}`;
  return {
    runId: item.run_id,
    status: item.status,
    statusLabel: presentation.label,
    statusTone: presentation.tone,
    statusDescription: presentation.description,
    timeText: settings.timeText || formatDateTime(item.latest_received_at_utc),
    durationText: settings.durationText
      || formatDuration(item.first_received_at_utc, item.latest_received_at_utc),
    sequenceText: `SEQ ${item.latest_sequence}`,
    progressText: item.progress_percent === null
      ? "—"
      : `${Math.round(item.progress_percent)}%`,
    progressStyle: item.progress_percent === null
      ? "width: 0%;"
      : `width: ${Math.max(0, Math.min(100, item.progress_percent))}%;`,
    checkpointText,
    alarmText: item.alarm_history_count === null ? "—" : String(item.alarm_history_count),
    activeAlarmText: item.active_alarm_count === null
      ? "—"
      : String(item.active_alarm_count),
    validated: item.result_validated,
    reportAvailable: settings.reportAvailable === undefined
      ? item.report_available === true
      : settings.reportAvailable === true,
    reportSource: settings.reportSource || "cloud",
  };
}

function formatSimulationDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "—";
  const rounded = Math.round(seconds);
  if (rounded < 60) return `${rounded} 秒`;
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  return remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分`;
}

function projectReportArchiveCard(report, source) {
  const result = report.result;
  const item = {
    run_id: report.run.run_id,
    status: "completed",
    outcome: report.run.outcome,
    result_outcome: result.outcome,
    result_validated: result.validated,
    report_available: true,
    first_received_at_utc: report.generated_at_utc,
    latest_received_at_utc: report.generated_at_utc,
    latest_sequence: report.run.sequence,
    progress_percent: report.task.progress_percent,
    completed_checkpoint_count: report.task.completed_checkpoint_count,
    checkpoint_count: report.task.checkpoint_count,
    alarm_history_count: report.alarms.history_count,
    active_alarm_count: 0,
  };
  return projectHistoryCard(item, {
    timeText: source === "demo" ? "内置演示档案" : "当前本地终态",
    durationText: formatSimulationDuration(report.run.simulation_time_s),
    reportAvailable: true,
    reportSource: source,
  });
}

Page({
  data: {
    items: [],
    loading: false,
    initialLoading: false,
    empty: false,
    errorMessage: "",
    hasMore: false,
    nextCursor: null,
    deviceLabel: "Mac mini M4 仿真端",
    sourceLabel: "云端只读档案",
    sourceDescription: "已完成任务可查看完整报告",
    emptyMessage: "仿真端首次上报后，任务摘要会显示在这里",
  },

  onLoad() {
    this._generation = 0;
    this._loaded = false;
    this._visible = false;
    this._lastRefreshStartedAt = null;
    this._activeRefreshDueAt = null;
    this._activeRefreshTimer = null;
    this._now = () => Date.now();
    this._setTimer = (callback, delayMs) => setTimeout(callback, delayMs);
    this._clearTimer = (timer) => clearTimeout(timer);
    this._mode = null;
    this._historyService = null;
    this._configureMode(this._currentMode());
  },

  onShow() {
    this._visible = true;
    const currentMode = this._currentMode();
    if (currentMode !== this._mode) this._configureMode(currentMode);
    if (this._mode !== "cloud") {
      this._loadLocalArchive();
      return Promise.resolve();
    }
    if (!this._historyService || this.data.loading) return Promise.resolve();
    const now = this._now();
    const elapsed = this._lastRefreshStartedAt === null
      ? Number.POSITIVE_INFINITY
      : now - this._lastRefreshStartedAt;
    const throttleElapsed = elapsed < 0 || elapsed >= AUTO_REFRESH_MIN_INTERVAL_MS;
    if (!this._loaded || throttleElapsed) {
      return this.refresh({ preserveItems: this._loaded });
    }
    this._resumeActiveRefresh();
    return Promise.resolve();
  },

  onHide() {
    this._visible = false;
    this._clearActiveRefreshTimer(false);
  },

  onPullDownRefresh() {
    return this.refresh({ preserveItems: this.data.items.length > 0 }).finally(() => {
      if (wx && typeof wx.stopPullDownRefresh === "function") wx.stopPullDownRefresh();
    });
  },

  onReachBottom() {
    return this.loadMore();
  },

  openReport(event) {
    const dataset = event && event.currentTarget ? event.currentTarget.dataset : {};
    const runId = dataset.runId;
    const source = dataset.source;
    const item = this.data.items.find((candidate) => (
      candidate.runId === runId
      && candidate.reportSource === source
      && candidate.reportAvailable
    ));
    if (!item || !wx || typeof wx.navigateTo !== "function") return;
    wx.navigateTo({
      url: `/pages/report/index?run_id=${encodeURIComponent(runId)}&source=${encodeURIComponent(source)}`,
    });
  },

  onUnload() {
    this._visible = false;
    this._clearActiveRefreshTimer(true);
    this._generation += 1;
    this._historyService = null;
    this._now = null;
    this._setTimer = null;
    this._clearTimer = null;
  },

  refresh(options) {
    if (this._mode !== "cloud") {
      this._loadLocalArchive();
      return Promise.resolve();
    }
    if (!this._historyService) return Promise.resolve();
    this._clearActiveRefreshTimer(true);
    const settings = options || {};
    const preserveItems = settings.preserveItems === true && this.data.items.length > 0;
    this._generation += 1;
    const generation = this._generation;
    this._lastRefreshStartedAt = this._now();
    this._loaded = false;
    this.setData({
      items: preserveItems ? this.data.items : [],
      loading: true,
      initialLoading: !preserveItems,
      empty: false,
      errorMessage: "",
      hasMore: false,
      nextCursor: null,
    });
    return this._fetchPage(null, generation, true);
  },

  loadMore() {
    if (
      !this._historyService
      || this.data.loading
      || !this.data.hasMore
      || !this.data.nextCursor
    ) return Promise.resolve();
    this._clearActiveRefreshTimer(false);
    this.setData({ loading: true, errorMessage: "" });
    return this._fetchPage(this.data.nextCursor, this._generation, false);
  },

  retry() {
    if (this.data.items.length > 0 && this.data.nextCursor) return this.loadMore();
    return this.refresh({ preserveItems: this.data.items.length > 0 });
  },

  _currentMode() {
    try {
      const app = typeof getApp === "function" ? getApp() : null;
      return normalizeMode(app && app.globalData ? app.globalData.mode : null);
    } catch (error) {
      return "cloud";
    }
  },

  _configureMode(mode) {
    this._clearActiveRefreshTimer(true);
    this._generation += 1;
    this._mode = mode;
    this._historyService = null;
    this._loaded = false;
    this._lastRefreshStartedAt = null;
    const presentations = {
      cloud: {
        deviceLabel: "Mac mini M4 仿真端",
        sourceLabel: "云端只读档案",
        sourceDescription: "已完成任务可查看完整报告",
        emptyMessage: "仿真端首次上报后，任务摘要会显示在这里",
      },
      demo: {
        deviceLabel: "Seed 7000 验证演示",
        sourceLabel: "内置可审计档案",
        sourceDescription: "来自项目随附的单次真实仿真证据",
        emptyMessage: "演示档案未通过本地完整性校验",
      },
      local: {
        deviceLabel: "开发者工具本地仿真",
        sourceLabel: "当前会话终态",
        sourceDescription: "本地模式不保存历史，只展示当前完整终态",
        emptyMessage: "当前会话还没有形成可验证的完整巡检报告",
      },
    };
    this.setData({
      ...presentations[mode],
      items: [],
      loading: false,
      initialLoading: false,
      empty: false,
      errorMessage: "",
      hasMore: false,
      nextCursor: null,
    });
    if (mode !== "cloud") {
      this._loadLocalArchive();
      return;
    }
    try {
      this._historyService = createPatrolHistoryService({ wxApi: wx, pageSize: 10 });
    } catch (error) {
      this.setData({ empty: false, errorMessage: safeErrorMessage(error) });
    }
  },

  _loadLocalArchive() {
    if (this._mode === "cloud") return;
    try {
      let report;
      if (this._mode === "demo") {
        report = buildDemoPatrolReport();
      } else {
        const app = typeof getApp === "function" ? getApp() : null;
        const dashboard = app && app.globalData ? app.globalData.dashboard : null;
        if (!dashboard) {
          this.setData({ items: [], empty: true, errorMessage: "" });
          return;
        }
        report = buildPatrolReportFromDashboard(dashboard);
      }
      this._loaded = true;
      this.setData({
        items: [projectReportArchiveCard(report, this._mode)],
        loading: false,
        initialLoading: false,
        empty: false,
        errorMessage: "",
        hasMore: false,
        nextCursor: null,
      });
    } catch (error) {
      const unavailableCurrent = this._mode === "local"
        && error instanceof ReportProtocolError
        && ["REPORT_NOT_READY", "REPORT_NOT_FOUND"].includes(error.code);
      this.setData({
        items: [],
        loading: false,
        initialLoading: false,
        empty: unavailableCurrent,
        errorMessage: unavailableCurrent ? "" : "演示档案未通过完整性校验",
      });
    }
  },

  _clearActiveRefreshTimer(clearDueAt) {
    if (this._activeRefreshTimer !== null && this._clearTimer) {
      this._clearTimer(this._activeRefreshTimer);
    }
    this._activeRefreshTimer = null;
    if (clearDueAt) this._activeRefreshDueAt = null;
  },

  _armActiveRefresh(delayMs) {
    if (!this._visible || !this._historyService || !this._setTimer) return;
    this._clearActiveRefreshTimer(false);
    const requestedDelay = Number.isFinite(delayMs) ? delayMs : ACTIVE_REFRESH_MIN_DELAY_MS;
    const boundedDelay = Math.min(
      STALE_AFTER_MS + 1,
      Math.max(ACTIVE_REFRESH_MIN_DELAY_MS, requestedDelay),
    );
    let timer;
    timer = this._setTimer(() => {
      if (this._activeRefreshTimer !== timer) return Promise.resolve();
      this._activeRefreshTimer = null;
      this._activeRefreshDueAt = null;
      if (!this._visible || !this._historyService) return Promise.resolve();
      if (this.data.loading) {
        this._activeRefreshDueAt = this._now() + ACTIVE_REFRESH_MIN_DELAY_MS;
        this._armActiveRefresh(ACTIVE_REFRESH_MIN_DELAY_MS);
        return Promise.resolve();
      }
      return this.refresh({ preserveItems: this.data.items.length > 0 });
    }, boundedDelay);
    this._activeRefreshTimer = timer;
  },

  _scheduleActiveRefresh(items) {
    this._clearActiveRefreshTimer(true);
    if (!Array.isArray(items)) return;
    const activeAges = items
      .filter((item) => item.status === "active")
      .map((item) => item.latest_age_ms);
    if (activeAges.length === 0) return;
    const oldestAgeMs = Math.max(...activeAges);
    const delayMs = Math.max(
      ACTIVE_REFRESH_MIN_DELAY_MS,
      STALE_AFTER_MS - oldestAgeMs + 1,
    );
    this._activeRefreshDueAt = this._now() + delayMs;
    if (this._visible) this._armActiveRefresh(delayMs);
  },

  _resumeActiveRefresh() {
    if (!this._visible || this._activeRefreshDueAt === null) return;
    this._armActiveRefresh(this._activeRefreshDueAt - this._now());
  },

  _fetchPage(cursor, generation, reset) {
    return this._historyService.fetchPage(cursor).then((page) => {
      if (generation !== this._generation) return;
      const existing = reset ? [] : this.data.items;
      const seen = new Set(existing.map((item) => item.runId));
      const appended = page.items
        .filter((item) => !seen.has(item.run_id))
        .map(projectHistoryCard);
      const items = [...existing, ...appended];
      this._loaded = true;
      this.setData({
        items,
        loading: false,
        initialLoading: false,
        empty: items.length === 0,
        errorMessage: "",
        hasMore: page.hasMore,
        nextCursor: page.nextCursor,
        deviceLabel: page.deviceId === "macmini-m4"
          ? "Mac mini M4 仿真端"
          : "已配置巡检设备",
      });
      if (reset) this._scheduleActiveRefresh(page.items);
      else this._resumeActiveRefresh();
    }).catch((error) => {
      if (generation !== this._generation) return;
      this._loaded = true;
      const invalidCursor = error instanceof HistoryProtocolError
        && (error.code === "INVALID_CURSOR" || error.code === "INVALID_HISTORY_REQUEST");
      this.setData({
        loading: false,
        initialLoading: false,
        empty: false,
        errorMessage: safeErrorMessage(error),
        hasMore: invalidCursor ? false : this.data.hasMore,
        nextCursor: invalidCursor ? null : this.data.nextCursor,
      });
      if (!reset) this._resumeActiveRefresh();
    });
  },
});

module.exports = {
  ACTIVE_REFRESH_MIN_DELAY_MS,
  AUTO_REFRESH_MIN_INTERVAL_MS,
  formatDateTime,
  formatDuration,
  formatSimulationDuration,
  projectHistoryCard,
  projectReportArchiveCard,
  safeErrorMessage,
};
