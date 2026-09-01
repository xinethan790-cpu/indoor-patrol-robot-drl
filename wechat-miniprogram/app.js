"use strict";

const {
  DEFAULT_MODE,
  getPlatform,
  normalizeMode,
  resolveModeForPlatform,
} = require("./services/modePolicy");

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
  },

  setMode(mode) {
    const safeMode = resolveModeForPlatform(mode, getPlatform(wx)).mode;
    this.globalData.mode = safeMode;
    wx.setStorageSync(MODE_STORAGE_KEY, safeMode);
    return safeMode;
  },

  setApiBaseUrl(value) {
    const safeValue = this.normalizeApiBaseUrl(value);
    this.globalData.apiBaseUrl = safeValue;
    wx.setStorageSync(API_STORAGE_KEY, safeValue);
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
});
