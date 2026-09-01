"use strict";

const MODE_STORAGE_KEY = "patrol_data_mode";
const API_STORAGE_KEY = "patrol_api_base_url";
const DEFAULT_API_BASE_URL = "http://127.0.0.1:8765";

App({
  globalData: {
    mode: "demo",
    apiBaseUrl: DEFAULT_API_BASE_URL,
    dashboard: null,
    dashboardMeta: null,
  },

  onLaunch() {
    const savedMode = wx.getStorageSync(MODE_STORAGE_KEY);
    const savedApiBaseUrl = wx.getStorageSync(API_STORAGE_KEY);
    this.globalData.mode = savedMode === "local" ? "local" : "demo";
    this.globalData.apiBaseUrl = this.normalizeApiBaseUrl(savedApiBaseUrl);
  },

  setMode(mode) {
    const safeMode = mode === "local" ? "local" : "demo";
    this.globalData.mode = safeMode;
    wx.setStorageSync(MODE_STORAGE_KEY, safeMode);
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

