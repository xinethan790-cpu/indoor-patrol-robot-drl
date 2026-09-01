"use strict";

const { formatNumber } = require("../../utils/format");

const FAILURE_REASON_LABELS = {
  "physical contact detected by Webots contact points": "Webots 接触点检测到物理碰撞",
  "physical contact during inspection dwell": "巡检驻留期间发生物理碰撞",
  "patrol TTC escape guard reached its bounded intervention timeout": "TTC 脱困保护达到有界介入时限",
  "maximum control-step budget reached": "达到最大控制步数",
};

function shortHash(value) {
  if (typeof value !== "string" || !value) return "—";
  return value.length > 20 ? `${value.slice(0, 12)}…${value.slice(-8)}` : value;
}

function metric(label, value, hint, tone) {
  return { label, value, hint, tone: tone || "neutral" };
}

function projectResult(payload) {
  const run = payload && payload.run ? payload.run : {};
  const result = payload && payload.result ? payload.result : {};
  const identity = run.runtime_identity || {};
  const available = result.available === true;
  const success = available && result.success === true;
  const total = result.checkpoints_total || (payload && payload.task && payload.task.checkpoint_count) || 0;
  const visited = result.checkpoints_visited || 0;
  return {
    resultTone: !available ? "waiting" : (success ? "success" : "failure"),
    resultIcon: !available ? "…" : (success ? "✓" : "!"),
    resultTitle: !available ? "巡检仍在进行" : (success ? "巡检任务顺利完成" : "巡检任务异常结束"),
    resultSummary: !available
      ? "终态后将展示已校验结果"
      : `${visited}/${total} 个点位 · ${result.collision_count || 0} 次碰撞 · ${result.sample_count || 0} 条采样`,
    validationLabel: result.validated ? "巡检证据与运行身份一致性已校验" : "等待终态完整性校验",
    metrics: [
      metric("巡检覆盖", `${visited} / ${total}`, "已完成点位", success ? "success" : "neutral"),
      metric("物理碰撞", String(result.collision_count ?? "—"), "Webots 接触检测", result.collision_count === 0 ? "success" : "warning"),
      metric("环境采样", String(result.sample_count ?? "—"), "三类传感器数据", "neutral"),
      metric("告警触发", String(result.alarms_raised ?? "—"), "异常事件", result.alarms_raised ? "warning" : "neutral"),
      metric("运行时间", `${formatNumber(result.elapsed_simulation_time_s, 1, "—")} s`, "仿真时间", "neutral"),
      metric("路径长度", `${formatNumber(result.actual_path_length_m, 2, "—")} m`, "机器人实际轨迹", "neutral"),
      metric("最小净空", `${formatNumber(result.minimum_clearance_m, 3, "—")} m`, "与障碍物间距", "neutral"),
      metric("最高告警", result.highest_event_severity === "critical" ? "严重" : (result.highest_event_severity || "无"), "本次巡检事件", result.highest_event_severity ? "warning" : "success"),
    ],
    dashboardMap: payload && payload.map ? payload.map : null,
    runId: run.run_id || "—",
    seed: run.seed ?? "—",
    modelHash: shortHash(identity.model_sha256),
    shieldId: identity.safety_shield_enabled ? (identity.safety_shield_id || "已启用") : "未启用",
    failureReason: FAILURE_REASON_LABELS[result.failure_reason] || result.failure_reason || "无",
  };
}

Page({
  data: projectResult(null),

  onShow() {
    const payload = getApp().globalData.dashboard;
    this.setData(projectResult(payload));
  },
});

module.exports = { projectResult, shortHash };
