"use strict";

const app = getApp();

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

Page({
  data: {
    mode: "demo",
    modeLabel: "仿真演示",
    apiBaseUrl: "http://127.0.0.1:8765",
    capabilities: [
      {
        icon: "↗",
        tone: "blue",
        title: "智能导航与动态避障",
        text: "深度强化学习策略结合安全保护组件，完成多点巡检。",
      },
      {
        icon: "◎",
        tone: "mint",
        title: "巡检状态可视化",
        text: "展示机器人位置、运行轨迹、P1–P4 点位和任务进度。",
      },
      {
        icon: "∿",
        tone: "amber",
        title: "环境数据采集",
        text: "到点模拟采集温度、烟雾遮蔽率和一氧化碳浓度。",
      },
      {
        icon: "!",
        tone: "violet",
        title: "异常检测与追溯",
        text: "记录告警触发、解除、阈值、时间和关联巡检点。",
      },
    ],
  },

  onShow() {
    const mode = app.globalData.mode === "local" ? "local" : "demo";
    this.setData({
      mode,
      modeLabel: mode === "demo" ? "仿真演示" : "本机实时",
      apiBaseUrl: app.globalData.apiBaseUrl || "http://127.0.0.1:8765",
    });
  },

  changeMode(event) {
    const mode = event.detail.value === "local" ? "local" : "demo";
    if (mode === "local" && !isDevTools()) {
      this.setData({ mode: "demo", modeLabel: "仿真演示" });
      wx.showModal({
        title: "真机暂不支持本机模式",
        content: "手机无法通过 127.0.0.1 访问 Mac。请使用仿真演示，或在后续配置公网 HTTPS 数据服务。",
        showCancel: false,
      });
      return;
    }
    app.setMode(mode);
    app.setDashboard(null, { mode });
    this.setData({ mode, modeLabel: mode === "demo" ? "仿真演示" : "本机实时" });
    wx.showToast({ title: mode === "demo" ? "已切换到演示模式" : "已切换到本机实时", icon: "none" });
  },

  editApiBaseUrl(event) {
    this.setData({ apiBaseUrl: event.detail.value });
  },

  saveApiBaseUrl() {
    const safeValue = app.setApiBaseUrl(this.data.apiBaseUrl);
    this.setData({ apiBaseUrl: safeValue });
  },
});
