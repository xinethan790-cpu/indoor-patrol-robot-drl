'use strict';

function finiteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} 必须是有限数字`);
  }
  return value;
}

function pointCoordinates(point, label) {
  if (!point || typeof point !== 'object') throw new TypeError(`${label} 必须是坐标对象`);
  const x = point.x_m ?? point.x;
  const y = point.y_m ?? point.y;
  return { x: finiteNumber(x, `${label}.x`), y: finiteNumber(y, `${label}.y`) };
}

function getMapBounds(map) {
  if (!map || typeof map !== 'object') throw new TypeError('map 必须是对象');
  const origin = pointCoordinates(map.origin, 'map.origin');
  const size = map.size || {};
  const widthM = finiteNumber(size.width_m ?? size.width, 'map.size.width');
  const heightM = finiteNumber(size.height_m ?? size.height, 'map.size.height');
  if (widthM <= 0 || heightM <= 0) throw new RangeError('地图宽高必须大于零');
  return {
    minX: origin.x,
    minY: origin.y,
    maxX: origin.x + widthM,
    maxY: origin.y + heightM,
    widthM,
    heightM,
  };
}

function normalizeOptions(options) {
  const source = options || {};
  const paddingPercent = source.paddingPercent ?? 0;
  finiteNumber(paddingPercent, 'paddingPercent');
  if (paddingPercent < 0 || paddingPercent >= 50) {
    throw new RangeError('paddingPercent 必须在 0（含）到 50（不含）之间');
  }
  return { clamp: source.clamp !== false, paddingPercent };
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function worldToPercent(point, map, options) {
  const value = pointCoordinates(point, 'point');
  const bounds = getMapBounds(map);
  const config = normalizeOptions(options);
  let normalizedX = (value.x - bounds.minX) / bounds.widthM;
  let normalizedY = (bounds.maxY - value.y) / bounds.heightM;
  if (config.clamp) {
    normalizedX = clamp(normalizedX, 0, 1);
    normalizedY = clamp(normalizedY, 0, 1);
  }
  const usable = 100 - (config.paddingPercent * 2);
  return {
    leftPercent: config.paddingPercent + (normalizedX * usable),
    topPercent: config.paddingPercent + (normalizedY * usable),
  };
}

function rectToPercent(rect, map, options) {
  if (!rect || typeof rect !== 'object') throw new TypeError('rect 必须是对象');
  const center = pointCoordinates(rect.center, 'rect.center');
  const size = rect.size || {};
  const widthM = finiteNumber(size.width_m ?? size.width, 'rect.size.width');
  const heightM = finiteNumber(size.height_m ?? size.height, 'rect.size.height');
  if (widthM < 0 || heightM < 0) throw new RangeError('矩形宽高不能为负数');
  const bounds = getMapBounds(map);
  const config = normalizeOptions(options);
  const topLeft = worldToPercent(
    { x: center.x - (widthM / 2), y: center.y + (heightM / 2) },
    map,
    config,
  );
  const usable = 100 - (config.paddingPercent * 2);
  return {
    leftPercent: topLeft.leftPercent,
    topPercent: topLeft.topPercent,
    widthPercent: (widthM / bounds.widthM) * usable,
    heightPercent: (heightM / bounds.heightM) * usable,
  };
}

function trajectoryToPercent(points, map, options) {
  if (!Array.isArray(points)) throw new TypeError('points 必须是数组');
  return points.map((point) => ({ ...point, ...worldToPercent(point, map, options) }));
}

module.exports = {
  getMapBounds,
  rectToPercent,
  trajectoryToPercent,
  worldToPercent,
};
