"use strict";

const mapTransform = require("../../utils/mapTransform");

const COLORS = Object.freeze({
  canvas: "#06131f",
  frame: "#123148",
  grid: "rgba(76, 159, 190, 0.16)",
  gridStrong: "rgba(78, 190, 224, 0.26)",
  obstacleFillTop: "#18384d",
  obstacleFillBottom: "#0d2537",
  obstacleStroke: "rgba(93, 199, 232, 0.56)",
  obstacleText: "#a9d4e4",
  routeShadow: "rgba(24, 211, 255, 0.16)",
  route: "#35d8ff",
  routeHead: "rgba(174, 244, 255, 0.92)",
  pending: "#597486",
  pendingFill: "#0c2331",
  completed: "#3ee6a8",
  completedFill: "#0d3d35",
  current: "#39dbff",
  currentFill: "#0d4056",
  checkpointText: "#e8f8ff",
  goal: "#ffd166",
  robot: "#59e4ff",
  robotCore: "#e9fcff",
  robotHeading: "#06202c",
  hint: "rgba(175, 210, 224, 0.58)",
});

const CHECKPOINT_STATUS = new Set(["completed", "current", "pending"]);
const MAX_GRID_LINES_PER_AXIS = 32;
const GRID_SPACING_M = 0.5;
const MIN_DRAW_SIZE_PX = 48;

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function positiveNumber(value) {
  return finiteNumber(value) && value > 0;
}

function isPoint(value) {
  if (!value || typeof value !== "object") return false;
  const x = finiteNumber(value.x_m) ? value.x_m : value.x;
  const y = finiteNumber(value.y_m) ? value.y_m : value.y;
  return finiteNumber(x) && finiteNumber(y);
}

function isRenderableMap(map) {
  return Boolean(
    map
      && typeof map === "object"
      && isPoint(map.origin)
      && map.size
      && positiveNumber(map.size.width_m)
      && positiveNumber(map.size.height_m),
  );
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function roundedRect(ctx, x, y, width, height, radius) {
  const safeRadius = Math.max(0, Math.min(radius, width / 2, height / 2));
  ctx.beginPath();
  ctx.moveTo(x + safeRadius, y);
  ctx.lineTo(x + width - safeRadius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + safeRadius);
  ctx.lineTo(x + width, y + height - safeRadius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - safeRadius, y + height);
  ctx.lineTo(x + safeRadius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - safeRadius);
  ctx.lineTo(x, y + safeRadius);
  ctx.quadraticCurveTo(x, y, x + safeRadius, y);
  ctx.closePath();
}

function checkpointPalette(status) {
  if (status === "completed") {
    return { stroke: COLORS.completed, fill: COLORS.completedFill };
  }
  if (status === "current") {
    return { stroke: COLORS.current, fill: COLORS.currentFill };
  }
  return { stroke: COLORS.pending, fill: COLORS.pendingFill };
}

Component({
  options: {
    pureDataPattern: /^_private/,
  },

  properties: {
    map: {
      type: Object,
      value: null,
      observer: "_onVisualInputChanged",
    },
    compact: {
      type: Boolean,
      value: false,
      observer: "_onVisualInputChanged",
    },
    ariaLabel: {
      type: String,
      value: "室内巡检机器人二维地图",
    },
  },

  data: {
    hasMap: false,
  },

  lifetimes: {
    attached() {
      this._privateAttached = true;
      this._privateDrawQueued = false;
      this._privateCanvas = null;
      this._privateContext = null;
      this._privateWidth = 0;
      this._privateHeight = 0;
      this._privatePixelRatio = 1;
      this._privateResizeHandler = () => this._queueDraw();
      if (typeof wx !== "undefined" && typeof wx.onWindowResize === "function") {
        wx.onWindowResize(this._privateResizeHandler);
      }
    },

    ready() {
      this._syncMapAvailability();
      this._queueDraw();
    },

    detached() {
      this._privateAttached = false;
      if (
        typeof wx !== "undefined"
        && typeof wx.offWindowResize === "function"
        && this._privateResizeHandler
      ) {
        wx.offWindowResize(this._privateResizeHandler);
      }
      this._privateResizeHandler = null;
      this._privateCanvas = null;
      this._privateContext = null;
    },
  },

  pageLifetimes: {
    show() {
      this._queueDraw();
    },
    resize() {
      this._queueDraw();
    },
  },

  methods: {
    _onVisualInputChanged() {
      if (!this._privateAttached) return;
      this._syncMapAvailability();
      this._queueDraw();
    },

    _syncMapAvailability() {
      const hasMap = isRenderableMap(this.properties.map);
      if (hasMap !== this.data.hasMap) this.setData({ hasMap });
    },

    _queueDraw() {
      if (!this._privateAttached || this._privateDrawQueued) return;
      this._privateDrawQueued = true;
      const run = () => {
        this._privateDrawQueued = false;
        if (this._privateAttached) this._measureAndDraw();
      };
      if (typeof wx !== "undefined" && typeof wx.nextTick === "function") {
        wx.nextTick(run);
      } else {
        setTimeout(run, 0);
      }
    },

    _measureAndDraw() {
      this.createSelectorQuery()
        .select("#patrol-map-canvas")
        .fields({ node: true, size: true })
        .exec((result) => {
          if (!this._privateAttached) return;
          const canvasInfo = result && result[0];
          if (
            !canvasInfo
            || !canvasInfo.node
            || !positiveNumber(canvasInfo.width)
            || !positiveNumber(canvasInfo.height)
          ) return;

          const pixelRatio = this._getPixelRatio();
          const dimensionsChanged = (
            this._privateCanvas !== canvasInfo.node
            || this._privateWidth !== canvasInfo.width
            || this._privateHeight !== canvasInfo.height
            || this._privatePixelRatio !== pixelRatio
          );
          this._privateCanvas = canvasInfo.node;
          this._privateWidth = canvasInfo.width;
          this._privateHeight = canvasInfo.height;
          this._privatePixelRatio = pixelRatio;
          if (!this._privateContext) {
            this._privateContext = canvasInfo.node.getContext("2d");
          }
          if (!this._privateContext) return;

          if (dimensionsChanged) {
            canvasInfo.node.width = Math.max(1, Math.round(canvasInfo.width * pixelRatio));
            canvasInfo.node.height = Math.max(1, Math.round(canvasInfo.height * pixelRatio));
            this._privateContext = canvasInfo.node.getContext("2d");
          }
          const ctx = this._privateContext;
          ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
          this._draw(ctx, canvasInfo.width, canvasInfo.height, this.properties.map);
        });
    },

    _getPixelRatio() {
      if (typeof wx === "undefined") return 1;
      try {
        const info = typeof wx.getWindowInfo === "function"
          ? wx.getWindowInfo()
          : wx.getSystemInfoSync();
        return clamp(Number(info.pixelRatio) || 1, 1, 3);
      } catch (error) {
        return 1;
      }
    },

    _draw(ctx, width, height, map) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = COLORS.canvas;
      ctx.fillRect(0, 0, width, height);
      if (width < MIN_DRAW_SIZE_PX || height < MIN_DRAW_SIZE_PX) return;
      if (!isRenderableMap(map)) {
        this._drawEmptyGrid(ctx, width, height);
        return;
      }

      let frame;
      try {
        frame = this._createFrame(map, width, height);
      } catch (error) {
        this._drawEmptyGrid(ctx, width, height);
        return;
      }

      this._drawGrid(ctx, map, frame);
      this._drawObstacles(ctx, map, frame);
      this._drawTrajectory(ctx, map, frame);
      this._drawCheckpoints(ctx, map, frame);
      this._drawLocalGoal(ctx, map, frame);
      this._drawRobot(ctx, map, frame);
      this._drawScaleHint(ctx, map, frame);
    },

    _createFrame(map, width, height) {
      const bounds = mapTransform.getMapBounds(map);
      const compact = this.properties.compact;
      const padding = compact ? 22 : 30;
      const footer = compact ? 8 : 13;
      const availableWidth = Math.max(1, width - padding * 2);
      const availableHeight = Math.max(1, height - padding * 2 - footer);
      const scale = Math.min(
        availableWidth / bounds.widthM,
        availableHeight / bounds.heightM,
      );
      const drawWidth = bounds.widthM * scale;
      const drawHeight = bounds.heightM * scale;
      return {
        x: (width - drawWidth) / 2,
        y: (height - footer - drawHeight) / 2,
        width: drawWidth,
        height: drawHeight,
        scale,
        bounds,
      };
    },

    _mapPoint(point, map, frame, clampToMap = true) {
      const percent = mapTransform.worldToPercent(point, map, { clamp: clampToMap });
      return {
        x: frame.x + (percent.leftPercent / 100) * frame.width,
        y: frame.y + (percent.topPercent / 100) * frame.height,
      };
    },

    _drawEmptyGrid(ctx, width, height) {
      const step = this.properties.compact ? 24 : 32;
      ctx.save();
      ctx.strokeStyle = "rgba(67, 150, 181, 0.10)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = step; x < width; x += step) {
        ctx.moveTo(x + 0.5, 0);
        ctx.lineTo(x + 0.5, height);
      }
      for (let y = step; y < height; y += step) {
        ctx.moveTo(0, y + 0.5);
        ctx.lineTo(width, y + 0.5);
      }
      ctx.stroke();
      ctx.restore();
    },

    _drawGrid(ctx, map, frame) {
      ctx.save();
      roundedRect(ctx, frame.x, frame.y, frame.width, frame.height, 8);
      ctx.fillStyle = "rgba(7, 28, 42, 0.88)";
      ctx.fill();
      ctx.clip();

      const xDivisions = Math.min(
        MAX_GRID_LINES_PER_AXIS,
        Math.max(1, Math.ceil(frame.bounds.widthM / GRID_SPACING_M)),
      );
      const yDivisions = Math.min(
        MAX_GRID_LINES_PER_AXIS,
        Math.max(1, Math.ceil(frame.bounds.heightM / GRID_SPACING_M)),
      );
      ctx.lineWidth = 1;
      for (let index = 0; index <= xDivisions; index += 1) {
        const x = frame.x + (index / xDivisions) * frame.width;
        ctx.beginPath();
        ctx.strokeStyle = index % 2 === 0 ? COLORS.gridStrong : COLORS.grid;
        ctx.moveTo(x, frame.y);
        ctx.lineTo(x, frame.y + frame.height);
        ctx.stroke();
      }
      for (let index = 0; index <= yDivisions; index += 1) {
        const y = frame.y + (index / yDivisions) * frame.height;
        ctx.beginPath();
        ctx.strokeStyle = index % 2 === 0 ? COLORS.gridStrong : COLORS.grid;
        ctx.moveTo(frame.x, y);
        ctx.lineTo(frame.x + frame.width, y);
        ctx.stroke();
      }
      ctx.restore();

      ctx.save();
      roundedRect(ctx, frame.x, frame.y, frame.width, frame.height, 8);
      ctx.strokeStyle = COLORS.frame;
      ctx.lineWidth = 1.5;
      ctx.shadowColor = "rgba(54, 211, 255, 0.18)";
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.restore();
    },

    _drawObstacles(ctx, map, frame) {
      safeArray(map.static_obstacles).forEach((obstacle) => {
        if (
          !obstacle
          || !isPoint(obstacle.center)
          || !obstacle.size
          || !positiveNumber(obstacle.size.width_m)
          || !positiveNumber(obstacle.size.height_m)
        ) return;
        try {
          const center = this._mapPoint(obstacle.center, map, frame);
          const obstacleWidth = obstacle.size.width_m * frame.scale;
          const obstacleHeight = obstacle.size.height_m * frame.scale;
          const x = center.x - obstacleWidth / 2;
          const y = center.y - obstacleHeight / 2;
          const gradient = ctx.createLinearGradient(x, y, x, y + obstacleHeight);
          gradient.addColorStop(0, COLORS.obstacleFillTop);
          gradient.addColorStop(1, COLORS.obstacleFillBottom);
          ctx.save();
          roundedRect(ctx, x, y, obstacleWidth, obstacleHeight, 5);
          ctx.fillStyle = gradient;
          ctx.fill();
          ctx.strokeStyle = COLORS.obstacleStroke;
          ctx.lineWidth = 1.2;
          ctx.stroke();
          if (!this.properties.compact && obstacleWidth >= 48 && obstacleHeight >= 22) {
            ctx.fillStyle = COLORS.obstacleText;
            ctx.font = "500 10px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(
              String(obstacle.label || obstacle.id || "障碍物").slice(0, 12),
              center.x,
              center.y,
              Math.max(20, obstacleWidth - 8),
            );
          }
          ctx.restore();
        } catch (error) {
          // Ignore one malformed obstacle without hiding the verified map frame.
        }
      });
    },

    _validTrajectory(map) {
      return safeArray(map.trajectory).filter(isPoint);
    },

    _drawTrajectory(ctx, map, frame) {
      const trajectory = this._validTrajectory(map);
      if (trajectory.length < 2) return;
      const points = [];
      trajectory.forEach((item) => {
        try {
          points.push(this._mapPoint(item, map, frame));
        } catch (error) {
          // Invalid points were already filtered; transformation failures remain isolated.
        }
      });
      if (points.length < 2) return;

      const trace = () => {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let index = 1; index < points.length; index += 1) {
          ctx.lineTo(points[index].x, points[index].y);
        }
      };
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      trace();
      ctx.strokeStyle = COLORS.routeShadow;
      ctx.lineWidth = this.properties.compact ? 7 : 10;
      ctx.stroke();
      trace();
      ctx.strokeStyle = COLORS.route;
      ctx.lineWidth = this.properties.compact ? 1.6 : 2.2;
      ctx.shadowColor = COLORS.route;
      ctx.shadowBlur = 7;
      ctx.stroke();
      const lastPoint = points[points.length - 1];
      ctx.beginPath();
      ctx.arc(lastPoint.x, lastPoint.y, 2.4, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.routeHead;
      ctx.fill();
      ctx.restore();
    },

    _drawCheckpoints(ctx, map, frame) {
      safeArray(map.checkpoints).forEach((checkpoint, index) => {
        if (!checkpoint || !isPoint(checkpoint.position)) return;
        try {
          const point = this._mapPoint(checkpoint.position, map, frame);
          const rawStatus = String(checkpoint.status || "pending");
          const status = CHECKPOINT_STATUS.has(rawStatus) ? rawStatus : "pending";
          const palette = checkpointPalette(status);
          const radius = this.properties.compact ? 7 : 9;
          ctx.save();
          if (status === "current") {
            ctx.beginPath();
            ctx.arc(point.x, point.y, radius + 5, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(57, 219, 255, 0.28)";
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.shadowColor = palette.stroke;
            ctx.shadowBlur = 12;
          }
          ctx.beginPath();
          ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = palette.fill;
          ctx.fill();
          ctx.strokeStyle = palette.stroke;
          ctx.lineWidth = 2;
          ctx.stroke();

          if (status === "completed") {
            ctx.beginPath();
            ctx.moveTo(point.x - radius * 0.45, point.y);
            ctx.lineTo(point.x - radius * 0.08, point.y + radius * 0.35);
            ctx.lineTo(point.x + radius * 0.5, point.y - radius * 0.36);
            ctx.strokeStyle = palette.stroke;
            ctx.lineWidth = 1.8;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.stroke();
          }

          ctx.shadowBlur = 0;
          ctx.fillStyle = COLORS.checkpointText;
          ctx.font = `${this.properties.compact ? 9 : 11}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          const id = String(checkpoint.id || `P${index + 1}`).slice(0, 8);
          ctx.fillText(id, point.x, point.y - radius - 5);
          ctx.restore();
        } catch (error) {
          // A malformed checkpoint must not break the remaining map rendering.
        }
      });
    },

    _drawLocalGoal(ctx, map, frame) {
      if (!isPoint(map.local_goal)) return;
      try {
        const point = this._mapPoint(map.local_goal, map, frame);
        const radius = this.properties.compact ? 5 : 7;
        ctx.save();
        ctx.strokeStyle = COLORS.goal;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(point.x - radius - 3, point.y);
        ctx.lineTo(point.x + radius + 3, point.y);
        ctx.moveTo(point.x, point.y - radius - 3);
        ctx.lineTo(point.x, point.y + radius + 3);
        ctx.stroke();
        ctx.restore();
      } catch (error) {
        // Local goal is optional live data.
      }
    },

    _robotHeading(map) {
      const robot = map.robot || {};
      const candidates = [robot.yaw_rad, robot.heading_rad, robot.theta_rad, map.robot_heading_rad];
      const heading = candidates.find(finiteNumber);
      return finiteNumber(heading) ? heading : 0;
    },

    _drawRobot(ctx, map, frame) {
      if (!isPoint(map.robot)) return;
      try {
        const point = this._mapPoint(map.robot, map, frame);
        const radius = this.properties.compact ? 8 : 11;
        const heading = this._robotHeading(map);
        ctx.save();
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius + 5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(89, 228, 255, 0.30)";
        ctx.lineWidth = 2;
        ctx.shadowColor = COLORS.robot;
        ctx.shadowBlur = 13;
        ctx.stroke();

        const gradient = ctx.createRadialGradient(
          point.x - radius * 0.3,
          point.y - radius * 0.35,
          radius * 0.12,
          point.x,
          point.y,
          radius,
        );
        gradient.addColorStop(0, COLORS.robotCore);
        gradient.addColorStop(1, COLORS.robot);
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        ctx.translate(point.x, point.y);
        ctx.rotate(-heading);
        ctx.beginPath();
        ctx.moveTo(radius * 0.72, 0);
        ctx.lineTo(-radius * 0.28, -radius * 0.36);
        ctx.lineTo(-radius * 0.28, radius * 0.36);
        ctx.closePath();
        ctx.fillStyle = COLORS.robotHeading;
        ctx.fill();
        ctx.restore();
      } catch (error) {
        // Robot data may be absent while the source reconnects.
      }
    },

    _drawScaleHint(ctx, map, frame) {
      if (this.properties.compact || frame.scale < 12) return;
      const oneMetre = Math.min(frame.scale, frame.width * 0.25);
      const y = Math.min(frame.y + frame.height + 14, this._privateHeight - 6);
      const x = frame.x;
      ctx.save();
      ctx.strokeStyle = COLORS.hint;
      ctx.fillStyle = COLORS.hint;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y - 3);
      ctx.lineTo(x, y);
      ctx.lineTo(x + oneMetre, y);
      ctx.lineTo(x + oneMetre, y - 3);
      ctx.stroke();
      ctx.font = "9px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      const distance = oneMetre === frame.scale ? "1 m" : `${(oneMetre / frame.scale).toFixed(1)} m`;
      ctx.fillText(distance, x + oneMetre / 2, y - 2);
      ctx.restore();
    },
  },
});
