"use strict";

const {
  DEFAULT_MODE,
  getPlatform,
  normalizeMode,
  resolveModeForPlatform,
} = require("./services/modePolicy");
const { createDataSource } = require("./services/createDataSource");
const { createPatrolSession } = require("./services/patrolSession");

const MODE_STORAGE_KEY = "patrol_data_mode";
const API_STORAGE_KEY = "patrol_api_base_url";
const DEFAULT_API_BASE_URL = "http://127.0.0.1:8765";
const CLOUD_ENV_ID = "cloud1-5gpkygv2be4de05a";

App({
  globalData: {
    mode: DEFAULT_MODE,
    apiBaseUrl: DEFAULT_API_BASE_URL,
    cloudEnvId: CLOUD_ENV_ID,
    cloudReady: false,
    cloudInitError: "",
    dashboard: null,
    dashboardMeta: null,
  },

  onLaunch() {
    const savedMode = wx.getStorageSync(MODE_STORAGE_KEY);
    const savedApiBaseUrl = wx.getStorageSync(API_STORAGE_KEY);
    const resolved = resolveModeForPlatform(normalizeMode(savedMode), getPlatform(wx));
    this.globalData.mode = resolved.mode;
    if (resolved.downgraded) wx.setStorageSync(MODE_STORAGE_KEY, resolved.mode);
    this.globalData.apiBaseUrl = this.normalizeApiBaseUrl(savedApiBaseUrl);
    try {
      if (!wx.cloud || typeof wx.cloud.init !== "function") {
        throw new Error("当前微信基础库不支持云开发");
      }
      wx.cloud.init({ env: CLOUD_ENV_ID });
      this.globalData.cloudReady = true;
      this.globalData.cloudInitError = "";
    } catch (error) {
      this.globalData.cloudReady = false;
      this.globalData.cloudInitError = error && error.message ? error.message : "云开发初始化失败";
    }
    this.patrolSession = createPatrolSession({
      initialMode: this.globalData.mode,
      createSource: (mode) => this.createPatrolSource(mode),
    });
    this._patrolStoreUnsubscribe = this.patrolSession.subscribe({
      onClear: (event) => this.setDashboard(null, {
        mode: event.mode,
        status: "idle",
        reason: event.reason,
      }),
      onData: (payload, meta) => this.setDashboard(payload, {
        receivedAt: Date.now(),
        ...(meta || {}),
      }),
      onStatus: (event) => {
        this.globalData.dashboardMeta = {
          ...(this.globalData.dashboardMeta || {}),
          mode: event.mode,
          status: event.status,
          transport: event.transport || null,
        };
      },
    });
  },

  onShow() {
    if (this.patrolSession) this.patrolSession.start();
  },

  onHide() {
    if (this.patrolSession) this.patrolSession.stop();
  },

  createPatrolSource(mode) {
    if (mode === "demo") return createDataSource({ mode: "demo" });
    if (mode === "cloud") {
      return createDataSource({
        mode: "cloud",
        env: this.globalData.cloudEnvId,
        wxApi: wx,
        staleAfterMs: 8000,
        pollIntervalMs: 2000,
      });
    }
    return createDataSource({
      mode: "local",
      endpoint: `${this.globalData.apiBaseUrl}/api/dashboard`,
      wxApi: wx,
      intervalMs: 900,
      timeoutMs: 5000,
    });
  },

  setMode(mode) {
    const safeMode = resolveModeForPlatform(mode, getPlatform(wx)).mode;
    const changed = this.globalData.mode !== safeMode;
    this.globalData.mode = safeMode;
    wx.setStorageSync(MODE_STORAGE_KEY, safeMode);
    if (changed) {
      this.setDashboard(null, { mode: safeMode, status: "idle" });
      if (this.patrolSession) {
        this.patrolSession.setMode(safeMode);
        this.patrolSession.start();
      }
    }
    return safeMode;
  },

  setApiBaseUrl(value) {
    const safeValue = this.normalizeApiBaseUrl(value);
    this.globalData.apiBaseUrl = safeValue;
    wx.setStorageSync(API_STORAGE_KEY, safeValue);
    if (this.patrolSession && this.globalData.mode === "local") {
      this.patrolSession.reset({
        recreate: true,
        restart: this.patrolSession.getState().started,
      });
    }
    return safeValue;
  },

  normalizeApiBaseUrl(value) {
    const text = typeof value === "string" ? value.trim() : "";
    if (!/^https?:\/\/[a-z0-9.:[\]-]+(?::\d+)?$/i.test(text)) {
      return DEFAULT_API_BASE_URL;
    }
    return text.replace(/\/+$/, "");
  },

  setDashboard(payload, meta) {
    this.globalData.dashboard = payload || null;
    this.globalData.dashboardMeta = meta || null;
  },

  subscribePatrol(callbacks) {
    if (!this.patrolSession) throw new Error("巡检会话尚未初始化");
    return this.patrolSession.subscribe(callbacks);
  },

  getPatrolSessionState() {
    return this.patrolSession ? this.patrolSession.getState() : null;
  },

  pausePatrolSession() {
    if (this.patrolSession) this.patrolSession.stop();
  },

  resumePatrolSession() {
    if (this.patrolSession) this.patrolSession.start();
  },

  resetPatrolSession() {
    if (this.patrolSession) this.patrolSession.reset({ restart: true });
  },

  jumpToNextDemoKeyEvent() {
    if (!this.patrolSession || this.globalData.mode !== "demo") return null;
    return this.patrolSession.jumpToNextKeyEvent();
  },

  setDemoPlaybackRate(rate) {
    if (!this.patrolSession || this.globalData.mode !== "demo") return null;
    return this.patrolSession.setPlaybackRate(rate);
  },
});
