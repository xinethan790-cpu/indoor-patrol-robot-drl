'use strict';

const STATE_LABELS = {
  idle: '等待任务',
  navigating: '前往巡检点',
  dwelling: '到点停留',
  sampling: '采集环境数据',
  completed: '巡检完成',
  failed: '巡检失败',
};

const OUTCOME_LABELS = {
  running: '进行中',
  success: '成功',
  collision: '发生碰撞',
  safety_stop: '安全停止',
  timeout: '任务超时',
  error: '运行异常',
};

const SENSOR_LABELS = {
  temperature: '温度',
  smoke_obscuration: '烟雾遮蔽率',
  carbon_monoxide: '一氧化碳浓度',
};

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function formatNumber(value, digits, placeholder) {
  const precision = digits === undefined ? 1 : digits;
  if (!Number.isSafeInteger(precision) || precision < 0 || precision > 10) {
    throw new RangeError('digits 必须是 0 至 10 的整数');
  }
  if (!isFiniteNumber(value)) return placeholder === undefined ? '--' : placeholder;
  return value.toFixed(precision);
}

function formatReading(value, unit, digits) {
  const number = formatNumber(value, digits);
  return number === '--' ? number : `${number} ${unit}`;
}

function formatTemperature(value) {
  return formatReading(value, '°C', 1);
}

function formatSmoke(value) {
  return formatReading(value, '%/m', 2);
}

function formatCarbonMonoxide(value) {
  return formatReading(value, 'ppm', 1);
}

function formatDistance(value) {
  return formatReading(value, 'm', 2);
}

function formatDuration(value) {
  if (!isFiniteNumber(value) || value < 0) return '--';
  const totalSeconds = Math.floor(value);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}分${String(seconds).padStart(2, '0')}秒` : `${seconds}秒`;
}

function formatPercent(value, options) {
  if (!isFiniteNumber(value)) return '--';
  const input = options && options.input === 'percent' ? value : value * 100;
  return `${formatNumber(input, options && options.digits !== undefined ? options.digits : 0)}%`;
}

function formatState(state) {
  return STATE_LABELS[state] || '未知状态';
}

function formatOutcome(outcome) {
  return OUTCOME_LABELS[outcome] || '未知结果';
}

function formatAlarmEvent(event) {
  if (!event || typeof event !== 'object') return '未知告警';
  const sensor = SENSOR_LABELS[event.sensor_id] || event.sensor_id || '传感器';
  const action = event.event_type === 'cleared' ? '恢复正常' : '触发告警';
  const value = isFiniteNumber(event.observed_value)
    ? `：${formatNumber(event.observed_value, 2)} ${event.unit || ''}`.trimEnd()
    : '';
  return `${event.checkpoint_id || '未知点位'} · ${sensor}${action}${value}`;
}

module.exports = {
  formatAlarmEvent,
  formatCarbonMonoxide,
  formatDistance,
  formatDuration,
  formatNumber,
  formatOutcome,
  formatPercent,
  formatReading,
  formatSmoke,
  formatState,
  formatTemperature,
};
