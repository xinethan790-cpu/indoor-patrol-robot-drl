"use strict";

const { formatNumber } = require("../../utils/format");
const {
  REPORT_UNAVAILABLE_MESSAGE,
  ReportProtocolError,
  createPatrolReportService,
} = require("../../services/patrolReport");

const SENSOR_LABELS = Object.freeze({
  temperature: "温度",
  smoke_obscuration: "烟雾遮蔽率",
  carbon_monoxide: "一氧化碳浓度",
});

const SOURCE_PRESENTATION = Object.freeze({
  cloud: { label: "云端归档报告", note: "从固定 Mac mini M4 仿真端只读同步" },
  demo: { label: "内置验证报告", note: "项目随附的 Seed 7000 单次可审计仿真档案" },
  local: { label: "当前本地终态", note: "仅在开发者工具当前会话中可用" },
});

function shortHash(value) {
  if (typeof value !== "string" || !value) return "—";
  return value.length > 20 ? `${value.slice(0, 12)}…${value.slice(-8)}` : value;
}

function metric(label, value, hint, tone) {
  return { label, value, hint, tone: tone || "neutral" };
}

function isPresent(value) {
  return value !== null && value !== undefined;
}

function sensorValue(key, reading) {
  const decimals = key === "temperature" ? 1 : (key === "smoke_obscuration" ? 2 : 0);
  return formatNumber(reading.value, decimals, "—");
}

function severityPresentation(value) {
  if (value === "critical") return { label: "严重", tone: "critical" };
  if (value === "warning") return { label: "警告", tone: "warning" };
  if (value === "info") return { label: "提示", tone: "info" };
  return { label: "正常", tone: "normal" };
}

function projectVisit(visit) {
  const severity = severityPresentation(visit.highest_active_severity);
  const readingKeys = ["temperature", "smoke_obscuration", "carbon_monoxide"];
  return {
    visitIndex: visit.visit_index,
    checkpointId: visit.checkpoint_id,
    timeText: `${formatNumber(visit.simulation_time_s, 1, "—")} s`,
    stepText: `控制步 ${visit.control_step}`,
    alarmText: visit.alarm_event_count > 0 ? `${visit.alarm_event_count} 条事件` : "无新事件",
    severityLabel: severity.label,
    severityTone: severity.tone,
    readings: readingKeys.map((key) => {
      const reading = visit.readings[key];
      return {
        key,
        label: SENSOR_LABELS[key],
        value: sensorValue(key, reading),
        unit: reading.unit,
        sampleText: `样本 #${reading.sample_index}`,
      };
    }),
  };
}

function projectAlarm(event) {
  const severity = severityPresentation(event.severity);
  const raised = event.event_type === "raised";
  return {
    eventIndex: event.event_index,
    eventType: event.event_type,
    actionLabel: raised ? "告警触发" : "告警解除",
    tone: raised ? severity.tone : "cleared",
    severityLabel: severity.label,
    sensorLabel: SENSOR_LABELS[event.sensor_id] || event.sensor_id,
    checkpointText: event.checkpoint_id || "未知点位",
    timeText: `${formatNumber(event.simulation_time_s, 1, "—")} s`,
    valueText: `${formatNumber(event.observed_value, 2, "—")} ${event.unit}`,
    thresholdText: `${raised ? "触发" : "恢复"}阈值 ${formatNumber(event.threshold, 2, "—")} ${event.unit}`,
  };
}

function projectReportView(report, source) {
  const result = report.result;
  const identity = report.run.runtime_identity;
  const shieldRate = isPresent(result.safety_shield_intervention_rate)
    ? `${formatNumber(result.safety_shield_intervention_rate * 100, 1, "—")}%`
    : "—";
  const guardActivations = result.ttc_escape_guard_activation_count;
  const guardRecoveries = result.ttc_escape_guard_recovery_count;
  const presentation = SOURCE_PRESENTATION[source] || SOURCE_PRESENTATION.cloud;
  return {
    ready: true,
    loading: false,
    errorMessage: "",
    resultTone: result.success ? "success" : "failure",
    resultIcon: result.success ? "✓" : "!",
    resultTitle: result.success ? "巡检任务顺利完成" : "巡检任务异常结束",
    resultSummary: `${result.checkpoints_visited}/${result.checkpoints_total} 个点位 · ${result.collision_count} 次碰撞 · ${result.sample_count} 条采样`,
    validationLabel: "终态、点位采样与运行身份已通过一致性校验",
    sourceLabel: presentation.label,
    sourceNote: presentation.note,
    coreMetrics: [
      metric("巡检覆盖", `${result.checkpoints_visited} / ${result.checkpoints_total}`, "已完成点位", result.success ? "success" : "neutral"),
      metric("物理碰撞", String(result.collision_count), "Webots 接触检测", result.collision_count === 0 ? "success" : "warning"),
      metric("环境采样", String(result.sample_count), "三类模拟传感器", "neutral"),
      metric("告警触发", String(result.alarms_raised), "异常事件", result.alarms_raised ? "warning" : "neutral"),
      metric("运行时间", `${formatNumber(result.elapsed_simulation_time_s, 1, "—")} s`, "仿真时间", "neutral"),
      metric("路径长度", `${formatNumber(result.actual_path_length_m, 2, "—")} m`, "机器人实际轨迹", "neutral"),
      metric("最小净空", `${formatNumber(result.minimum_clearance_m, 3, "—")} m`, "与障碍物间距", "neutral"),
      metric("最高告警", severityPresentation(result.highest_event_severity).label, "本次巡检事件", result.highest_event_severity ? "warning" : "success"),
    ],
    avoidanceMetrics: [
      metric("最小 LiDAR", `${formatNumber(result.minimum_lidar_range_m, 3, "—")} m`, "激光雷达最近量测"),
      metric("近距风险", String(result.critical_proximity_count ?? "—"), "严重接近次数", result.critical_proximity_count ? "warning" : "success"),
      metric("安全盾介入", isPresent(result.safety_shield_intervention_steps) ? `${result.safety_shield_intervention_steps} 步` : "—", `占控制步 ${shieldRate}`),
      metric("脱困保护", isPresent(guardActivations) ? `${guardRecoveries ?? 0} / ${guardActivations}` : "—", "恢复次数 / 触发次数", result.ttc_escape_guard_unrecoverable_count ? "warning" : "success"),
      metric("推理延迟 P95", `${formatNumber(result.inference_latency_p95_ms, 2, "—")} ms`, "策略单步推理"),
      metric("动态障碍", isPresent(result.dynamic_obstacle_count) ? `${result.dynamic_obstacle_count} 个` : "—", isPresent(result.dynamic_obstacle_speed_max_mps) ? `最高 ${formatNumber(result.dynamic_obstacle_speed_max_mps, 2, "—")} m/s` : "仿真场景"),
    ],
    dashboardMap: report.map,
    trajectoryNote: report.map.trajectory_truncated
      ? `展示 ${report.map.trajectory_point_count} 个关键轨迹点，原始运行共 ${report.map.trajectory_source_point_count} 个事件点`
      : `完整展示 ${report.map.trajectory_point_count} 个轨迹点`,
    visits: report.inspection.visits.map(projectVisit),
    hasVisits: report.inspection.visit_count > 0,
    sampleSummary: report.inspection.visit_count > 0
      ? `${report.inspection.visit_count} 个巡检点 · ${report.inspection.sample_count} 条原始采样`
      : "未到达巡检点 · 无点位采样",
    alarms: report.alarms.history.map(projectAlarm),
    alarmSummary: `${result.alarms_raised} 次触发 · ${result.alarms_cleared} 次解除`,
    runId: report.run.run_id,
    seed: report.run.seed,
    sequenceText: `SEQ ${report.run.sequence}`,
    modelHash: shortHash(identity.model_sha256),
    modelId: identity.model_id,
    shieldId: identity.safety_shield_enabled ? identity.safety_shield_id : "未启用",
    validationScope: report.source.terminal_validation,
    mapSource: report.source.map_source_file,
    mapBoundaryNote: report.source.map_boundary_note,
  };
}

function safeReportError(error) {
  return error instanceof ReportProtocolError ? error.message : REPORT_UNAVAILABLE_MESSAGE;
}

Page({
  data: {
    ready: false,
    loading: true,
    errorMessage: "",
  },

  onLoad(options) {
    this._generation = 0;
    this._runId = "";
    this._source = "cloud";
    let source = options && options.source;
    let runId = options && options.run_id;
    try {
      source = decodeURIComponent(source || "");
      runId = decodeURIComponent(runId || "");
    } catch (error) {
      this.setData({ loading: false, ready: false, errorMessage: "报告链接格式无效" });
      return;
    }
    if (!["cloud", "demo", "local"].includes(source)) {
      this.setData({ loading: false, ready: false, errorMessage: "报告来源无效" });
      return;
    }
    this._source = source;
    this._runId = runId;
    this._loadReport();
  },

  onPullDownRefresh() {
    return this._loadReport().finally(() => {
      if (typeof wx !== "undefined" && typeof wx.stopPullDownRefresh === "function") {
        wx.stopPullDownRefresh();
      }
    });
  },

  retry() {
    return this._loadReport();
  },

  onUnload() {
    this._generation += 1;
    this._service = null;
  },

  _loadReport() {
    this._generation += 1;
    const generation = this._generation;
    this.setData({ loading: true, ready: false, errorMessage: "" });
    try {
      const app = typeof getApp === "function" ? getApp() : null;
      this._service = createPatrolReportService({
        mode: this._source,
        wxApi: typeof wx === "undefined" ? null : wx,
        getDashboard: () => (app && app.globalData ? app.globalData.dashboard : null),
      });
    } catch (error) {
      this.setData({ loading: false, ready: false, errorMessage: safeReportError(error) });
      return Promise.resolve();
    }
    return this._service.fetchReport(this._runId).then((report) => {
      if (generation !== this._generation) return;
      this.setData(projectReportView(report, this._source));
    }).catch((error) => {
      if (generation !== this._generation) return;
      this.setData({ loading: false, ready: false, errorMessage: safeReportError(error) });
    });
  },
});

module.exports = {
  projectAlarm,
  projectReportView,
  projectVisit,
  safeReportError,
  severityPresentation,
  shortHash,
};
