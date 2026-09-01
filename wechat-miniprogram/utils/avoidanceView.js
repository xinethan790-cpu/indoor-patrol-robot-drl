'use strict';

const { formatNumber } = require('./format');

const GUARD_LABELS = {
  inactive: '未启用',
  reverse: '后退脱困',
  forward: '前进脱困',
  rotate: '旋转脱困',
  unrecoverable: '无法安全脱困',
};

function signed(value, digits) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${formatNumber(value, digits, '—')}`;
}

function actionText(action, kind) {
  if (!action || typeof action !== 'object') return '等待数据';
  if (kind === 'executed') {
    return `${signed(action.linear_mps, 2)} m/s · ${signed(action.angular_radps, 2)} rad/s`;
  }
  return `v ${signed(action.linear_normalized, 2)} · ω ${signed(action.angular_normalized, 2)}`;
}

function emptyAvoidanceView(dynamicObstacleCount) {
  return {
    avoidanceAvailable: false,
    avoidanceTone: 'idle',
    avoidanceStatus: '等待避障遥测',
    avoidanceSummary: '收到机器人控制数据后，将显示 PPO 决策与安全保护过程',
    minimumRangeText: '—',
    ttcText: '—',
    dynamicObstacleText: `${dynamicObstacleCount || 0} 个`,
    shieldText: '未介入',
    guardText: '未启用',
    policyActionText: '等待数据',
    filteredActionText: '等待数据',
    executedActionText: '等待数据',
  };
}

function buildAvoidanceView(avoidance, map) {
  const dynamicObstacleCount = map && Array.isArray(map.dynamic_obstacles)
    ? map.dynamic_obstacles.length
    : 0;
  if (!avoidance || avoidance.available !== true) {
    return emptyAvoidanceView(dynamicObstacleCount);
  }

  const shield = avoidance.safety_shield || {};
  const guard = avoidance.escape_guard || {};
  let tone = 'normal';
  let status = '自主避障正常';
  let summary = 'PPO 正在根据 LiDAR 与局部目标连续输出控制动作';
  if (guard.unrecoverable === true) {
    tone = 'critical';
    status = '安全停止';
    summary = '脱困保护判定当前状态无法安全恢复';
  } else if (guard.active === true) {
    tone = 'guard';
    status = GUARD_LABELS[guard.mode] || '脱困保护介入';
    summary = 'TTC 有界脱困正在覆盖常规策略动作';
  } else if (avoidance.critical_proximity === true) {
    tone = 'critical';
    status = '临界接近';
    summary = '机器人与障碍物距离进入临界区间';
  } else if (shield.active === true) {
    tone = 'warning';
    status = '安全盾介入';
    summary = '安全过滤器已修正 PPO 原始动作';
  } else if (!avoidance.actions || avoidance.actions.policy === null) {
    status = '驻留 / 停车';
    summary = '机器人正在巡检点驻留或已停车，当前帧没有新的 PPO 策略动作';
  }

  const actions = avoidance.actions || {};
  return {
    avoidanceAvailable: true,
    avoidanceTone: tone,
    avoidanceStatus: status,
    avoidanceSummary: summary,
    minimumRangeText: `${formatNumber(avoidance.minimum_lidar_range_m, 2, '—')} m`,
    ttcText: avoidance.robust_ttc_s === null
      ? '无碰撞趋势'
      : `${formatNumber(avoidance.robust_ttc_s, 2, '—')} s`,
    dynamicObstacleText: `${dynamicObstacleCount} 个`,
    shieldText: shield.active === true
      ? `已修正 ${formatNumber(shield.intervention, 3, '—')}`
      : '未介入',
    guardText: GUARD_LABELS[guard.mode] || guard.mode || '未启用',
    policyActionText: actionText(actions.policy, 'normalized'),
    filteredActionText: actionText(actions.filtered, 'normalized'),
    executedActionText: actionText(actions.executed, 'executed'),
  };
}

module.exports = {
  actionText,
  buildAvoidanceView,
  emptyAvoidanceView,
};
