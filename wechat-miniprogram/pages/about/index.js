"use strict";

const { getPlatform, resolveModeForPlatform } = require("../../services/modePolicy");

const app = getApp();

function modeLabel(mode) {
  if (mode === "cloud") return "云端实时";
  if (mode === "local") return "本机实时";
  return "仿真演示";
}

Page({
  data: {
    mode: "cloud",
    modeLabel: "云端实时",
    apiBaseUrl: "http://127.0.0.1:8765",
    cloudEnvId: "cloud1-5gpkygv2be4de05a",
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
      {
        icon: "☁",
        tone: "blue",
        title: "真机云端实时串联",
        text: "通过 CloudBase 只读同步 Mac 仿真快照，订阅中断时自动轮询。",
      },
    ],
  },

  onShow() {
    const resolution = resolveModeForPlatform(app.globalData.mode, getPlatform(wx));
    const mode = resolution.mode;
    if (resolution.downgraded) app.setMode(mode);
    this.setData({
      mode,
      modeLabel: modeLabel(mode),
      apiBaseUrl: app.globalData.apiBaseUrl || "http://127.0.0.1:8765",
      cloudEnvId: app.globalData.cloudEnvId || "cloud1-5gpkygv2be4de05a",
    });
  },

  changeMode(event) {
    const resolution = resolveModeForPlatform(event.detail.value, getPlatform(wx));
    const mode = resolution.mode;
    if (resolution.downgraded) {
      app.setMode("cloud");
      app.setDashboard(null, { mode: "cloud" });
      this.setData({ mode: "cloud", modeLabel: "云端实时" });
      wx.showModal({
        title: "已自动切换到云端实时",
        content: "真机无法通过 127.0.0.1 访问 Mac。系统已改用 CloudBase 安全只读链路。",
        showCancel: false,
      });
      return;
    }
    app.setMode(mode);
    app.setDashboard(null, { mode });
    this.setData({ mode, modeLabel: modeLabel(mode) });
    wx.showToast({ title: `已切换到${modeLabel(mode)}`, icon: "none" });
  },

  editApiBaseUrl(event) {
    this.setData({ apiBaseUrl: event.detail.value });
  },

  saveApiBaseUrl() {
    const safeValue = app.setApiBaseUrl(this.data.apiBaseUrl);
    this.setData({ apiBaseUrl: safeValue });
  },
});
