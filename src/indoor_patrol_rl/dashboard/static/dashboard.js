"use strict";

const svgNS = "http://www.w3.org/2000/svg";
const stateNames = { navigating: "导航中", dwelling: "驻留检测", sampling: "环境采样", completed: "已完成", failed: "已终止" };
const outcomeNames = { running: "进行中", success: "巡检成功", collision: "发生碰撞", safety_stop: "安全停止", timeout: "任务超时", inspection_failed: "采样失败", error: "运行异常", completed: "巡检成功" };
const severityNames = { info: "提示", warning: "警告", critical: "严重" };
const eventNames = { raised: "触发", cleared: "解除" };
const sensorNames = { temperature: "温度", smoke_obscuration: "烟雾遮蔽率", carbon_monoxide: "一氧化碳浓度" };
const checkpointStatusNames = { completed: "已完成", current: "当前点位", pending: "待巡检" };
const failureReasonNames = {
  "physical contact detected by Webots contact points": "Webots 接触点检测到物理碰撞",
  "physical contact during inspection dwell": "巡检驻留期间发生物理碰撞",
  "patrol TTC escape guard reached its bounded intervention timeout": "TTC 脱困保护达到有界介入时限",
  "patrol TTC escape guard reached its bounded intervention timeout during inspection dwell": "巡检驻留期间 TTC 脱困保护达到有界介入时限",
  "maximum control-step budget reached": "达到最大控制步数",
};
const pollIntervalMs = 750;
const maxGridLinesPerAxis = 32;
const maxHistoryRows = 200;
let failureCount = 0;
let lastRunId = null;
let lastSequence = -1;
let pollingStopped = false;

const byId = (id) => document.getElementById(id);
const show = (value, digits = 1) => value === null || value === undefined ? "—" : Number(value).toFixed(digits);

function localizeAlarmMessage(message) {
  return String(message)
    .replace(" alarm raised: ", "告警触发：")
    .replace(" alarm cleared: ", "告警解除：")
    .replace(" (threshold ", "（阈值 ")
    .replace(/\)$/, "）");
}

function alarmDescription(sensorId, message) {
  const sensorLabel = sensorNames[sensorId] || sensorId;
  const localized = localizeAlarmMessage(message);
  return localized.startsWith(sensorLabel) ? localized : `${sensorLabel} · ${localized}`;
}

function localizeFailureReason(reason) {
  if (!reason) return "无";
  if (failureReasonNames[reason]) return failureReasonNames[reason];
  const samplingPrefix = "inspection sampling failed: ";
  return reason.startsWith(samplingPrefix) ? `环境采样失败：${reason.slice(samplingPrefix.length)}` : reason;
}

function svgElement(name, attrs = {}) {
  const node = document.createElementNS(svgNS, name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function gridOffsets(extent) {
  const divisions = Math.min(maxGridLinesPerAxis, Math.max(1, Math.ceil(extent / 0.5)));
  return Array.from({ length: divisions + 1 }, (_, index) => extent * index / divisions);
}

function renderMap(data) {
  const svg = byId("patrol-map");
  svg.replaceChildren();
  const map = data.map;
  const pad = 62;
  const width = 1000 - pad * 2;
  const height = 680 - pad * 2;
  const sx = width / map.size.width_m;
  const sy = height / map.size.height_m;
  const scale = Math.min(sx, sy);
  const drawW = map.size.width_m * scale;
  const drawH = map.size.height_m * scale;
  const ox = (1000 - drawW) / 2;
  const oy = (680 - drawH) / 2;
  const point = (value) => ({ x: ox + (value.x_m - map.origin.x_m) * scale, y: oy + drawH - (value.y_m - map.origin.y_m) * scale });

  svg.append(svgElement("rect", { x: ox, y: oy, width: drawW, height: drawH, rx: 14, class: "map-boundary" }));
  gridOffsets(map.size.width_m).forEach((x) => {
    const px = ox + x * scale;
    svg.append(svgElement("line", { x1: px, y1: oy, x2: px, y2: oy + drawH, class: "map-grid-line" }));
  });
  gridOffsets(map.size.height_m).forEach((y) => {
    const py = oy + y * scale;
    svg.append(svgElement("line", { x1: ox, y1: py, x2: ox + drawW, y2: py, class: "map-grid-line" }));
  });
  map.static_obstacles.forEach((obstacle) => {
    const center = point(obstacle.center);
    const w = obstacle.size.width_m * scale;
    const h = obstacle.size.height_m * scale;
    svg.append(svgElement("rect", { x: center.x - w / 2, y: center.y - h / 2, width: w, height: h, rx: 8, class: "map-obstacle" }));
    const label = svgElement("text", { x: center.x, y: center.y + 7, class: "obstacle-label" });
    label.textContent = obstacle.label;
    svg.append(label);
  });
  if (map.trajectory.length > 1) {
    const points = map.trajectory.map((item) => { const p = point({ x_m: item.x_m, y_m: item.y_m }); return `${p.x},${p.y}`; }).join(" ");
    svg.append(svgElement("polyline", { points, class: "route-shadow" }));
    svg.append(svgElement("polyline", { points, class: "route-line" }));
  }
  map.checkpoints.forEach((checkpoint) => {
    const p = point(checkpoint.position);
    svg.append(svgElement("circle", { cx: p.x, cy: p.y, r: 15, class: `checkpoint ${checkpoint.status}` }));
    const label = svgElement("text", { x: p.x, y: p.y - 24, class: "checkpoint-label" });
    label.textContent = checkpoint.id;
    svg.append(label);
  });
  if (map.local_goal) {
    const goal = point(map.local_goal);
    svg.append(svgElement("circle", { cx: goal.x, cy: goal.y, r: 10, class: "local-goal" }));
  }
  const robot = point(map.robot);
  svg.append(svgElement("circle", { cx: robot.x, cy: robot.y, r: 19, class: "robot-ring" }));
  svg.append(svgElement("circle", { cx: robot.x, cy: robot.y, r: 8, class: "robot-core" }));
}

function renderTask(data) {
  const task = data.task;
  byId("progress-percent").textContent = `${task.progress_percent}%`;
  byId("progress-fraction").textContent = `${task.completed_checkpoint_count} / ${task.checkpoint_count} 巡检点`;
  byId("progress-fill").value = task.progress_percent;
  byId("simulation-time").textContent = `${show(data.run.simulation_time_s, 1)} s`;
  byId("control-step").textContent = String(data.run.control_step);
  byId("state-pill").textContent = stateNames[data.run.state] || data.run.state;
  byId("state-pill").className = `pill ${data.run.terminal ? (data.run.outcome === "success" ? "success" : "failure") : ""}`;
  const list = byId("checkpoint-list");
  list.replaceChildren(...data.map.checkpoints.map((checkpoint) => {
    const item = document.createElement("li");
    item.className = checkpoint.status;
    const name = document.createElement("span");
    name.className = "checkpoint-name";
    name.textContent = checkpoint.id;
    const status = document.createElement("span");
    status.className = "checkpoint-status";
    status.textContent = checkpointStatusNames[checkpoint.status] || checkpoint.status;
    item.setAttribute("aria-label", `${checkpoint.id}：${status.textContent}`);
    item.append(name, status);
    return item;
  }));
}

function renderSensors(data) {
  const sensors = data.sensors;
  byId("temperature-value").textContent = show(sensors.temperature.value, 1);
  byId("smoke-value").textContent = show(sensors.smoke.value, 2);
  byId("co-value").textContent = show(sensors.carbon_monoxide.value, 0);
  byId("sensor-source").textContent = sensors.available ? sensors.checkpoint_id : "尚未采样";
  byId("sensor-provenance").textContent = sensors.available ? `${sensors.provenance_label} · ${sensors.checkpoint_id} · ${show(sensors.sampled_at_s, 1)} s` : `${sensors.provenance_label} · 等待到达巡检点`;
}

function renderAlarms(data) {
  byId("active-count").textContent = String(data.alarms.active_count);
  const active = byId("active-alarms");
  if (!data.alarms.active.length) {
    active.className = "active-alarms empty-state";
    active.textContent = "当前无活动告警";
  } else {
    active.className = "active-alarms";
    active.replaceChildren(...data.alarms.active.map((alarm) => {
      const item = document.createElement("div");
      item.className = `alert-item ${alarm.severity}`;
      const title = document.createElement("strong");
      title.textContent = `${severityNames[alarm.severity] || alarm.severity} · ${alarm.checkpoint_id || "未知点位"} · ${sensorNames[alarm.sensor_id] || alarm.sensor_id}`;
      const message = document.createElement("span");
      message.textContent = localizeAlarmMessage(alarm.message);
      const detail = document.createElement("span");
      detail.textContent = `${alarm.observed_value} ${alarm.unit} / 阈值 ${alarm.threshold} ${alarm.unit}`;
      item.append(title, message, detail);
      return item;
    }));
  }
  byId("history-source").textContent = data.source.alarm_history_source === "verified_inspection_bundle" ? "已校验事件记录" : "运行快照差分";
  const history = byId("alarm-history");
  if (!data.alarms.history.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5; cell.className = "empty-cell"; cell.textContent = "暂无异常记录";
    row.append(cell); history.replaceChildren(row); return;
  }
  history.replaceChildren(...data.alarms.history.slice(0, maxHistoryRows).map((event) => {
    const row = document.createElement("tr");
    const values = [`${show(event.simulation_time_s, 1)} s`, event.checkpoint_id || "—", severityNames[event.severity] || event.severity, eventNames[event.event_type] || event.event_type, alarmDescription(event.sensor_id, event.message)];
    values.forEach((value, index) => { const cell = document.createElement("td"); cell.textContent = value; if (index === 2) cell.className = `severity-${event.severity}`; if (index === 3) cell.className = `event-${event.event_type}`; if (index === 4) cell.title = value; row.append(cell); });
    return row;
  }));
}

function renderResult(data) {
  const result = data.result;
  const badge = byId("result-badge");
  const content = byId("result-content");
  if (!result.available) {
    badge.textContent = "进行中"; badge.className = "pill neutral";
    content.className = "result-waiting";
    content.replaceChildren(Object.assign(document.createElement("span"), { className: "pulse-ring" }), Object.assign(document.createElement("p"), { textContent: "巡检任务正在执行" }), Object.assign(document.createElement("small"), { textContent: "终态后展示已校验汇总结果" }));
    return;
  }
  badge.textContent = outcomeNames[result.outcome] || result.outcome;
  badge.className = `pill ${result.success ? "success" : "failure"}`;
  content.className = "result-metrics";
  const metrics = [
    ["巡检覆盖", `${result.checkpoints_visited} / ${result.checkpoints_total}`],
    ["物理碰撞", String(result.collision_count)],
    ["采样数据", String(result.sample_count)],
    ["异常触发", String(result.alarms_raised)],
    ["异常解除", String(result.alarms_cleared)],
    ["最高告警级别", severityNames[result.highest_event_severity] || result.highest_event_severity || "无"],
    ["运行时间", `${show(result.elapsed_simulation_time_s, 1)} s`],
    ["路径长度", `${show(result.actual_path_length_m, 2)} m`],
    ["最小净空", `${show(result.minimum_clearance_m, 3)} m`],
    ["失败原因", localizeFailureReason(result.failure_reason)],
  ];
  content.replaceChildren(...metrics.map(([label, value]) => { const card = document.createElement("div"); card.className = "result-metric"; const name = document.createElement("span"); name.textContent = label; const number = document.createElement("strong"); number.textContent = value; card.append(name, number); return card; }));
}

function render(data) {
  if (data.schema_version !== "patrol-dashboard/v1") throw new Error("不支持的看板数据版本");
  if (lastRunId !== null && data.run.run_id !== lastRunId) throw new Error("运行标识发生变化，已拒绝混合数据");
  if (data.run.sequence < lastSequence) return false;
  lastRunId = data.run.run_id;
  lastSequence = data.run.sequence;
  renderMap(data); renderTask(data); renderSensors(data); renderAlarms(data); renderResult(data);
  byId("map-note").textContent = data.source.map_boundary_note;
  byId("map-mode").textContent = data.source.mode === "terminal" ? "FINAL" : "LIVE";
  byId("updated-at").textContent = `run ${data.run.run_id} · ${new Date(data.generated_at_utc).toLocaleTimeString("zh-CN")}`;
  byId("connection-label").textContent = data.run.terminal ? "终态汇总已校验" : "实时数据已连接";
  byId("connection-dot").className = "connection-dot online";
  byId("error-banner").hidden = true;
  return data.run.terminal;
}

async function refresh() {
  try {
    const response = await fetch("/api/dashboard", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || `HTTP ${response.status}`);
    const terminal = render(data);
    failureCount = 0;
    if (terminal) pollingStopped = true;
  } catch (error) {
    failureCount += 1;
    byId("connection-dot").className = "connection-dot offline";
    byId("connection-label").textContent = "数据暂不可用";
    const banner = byId("error-banner");
    banner.textContent = `看板正在重试：${error.message}`; banner.hidden = failureCount < 2;
  } finally {
    if (!pollingStopped) window.setTimeout(refresh, pollIntervalMs);
  }
}

refresh();
