# 开发记录

## 2026-08-28：首个可运行闭环

### 已完成

- 在 Mac mini M4 上安装并启动 Webots R2025a 原生应用；
- 建立 Python 3.11 虚拟环境、可安装包结构、测试和 GitHub Actions；
- 实现轻量二维动态避障环境，统一为 79 维观测和二维连续动作；
- 实现占据栅格、圆盘障碍膨胀、8 邻接 A*、禁止斜穿墙角与路径简化；
- 验证 TurtleBot3 Burger 的差速轮、LDS-01、GPS、IMU 和 Supervisor；
- 建立可视化横穿障碍演示世界；
- 实现 `WebotsPatrolEnv`，统一 LiDAR 方向、轮速限幅、回合重置和碰撞判定；
- 通过 Stable-Baselines3 环境检查，并完成 Webots 中 256 步 PPO 冒烟训练。

### 可复现结果

```text
ruff: passed
pytest: 15 passed
Webots PPO smoke: 256 steps, about 160 FPS, model saved successfully
```

生成的短训练模型位于 `artifacts/models/ppo_webots_smoke.zip`，按设计不提交 Git。该模型只证明“传感器 → Gymnasium → PPO → 模型保存”链路可用，不能作为避障效果结论。

### 下一里程碑

1. 完成 DWA 局部避障基线并与规则法统一接口；
2. 将 A* 路点管理器接入 Webots 巡检路线；
3. 增加横穿、迎面、路口交叉和随机游走场景；
4. 建立验证集/最终测试集隔离与 CSV 指标记录；
5. 开始 PPO 课程训练，并以成功率、碰撞率、路径效率和最小安全距离评测。

## 2026-08-28：A* + DWA 多点巡检闭环

### 已完成

- 实现确定性的 DWA 局部规划器、动态窗口、轨迹评分和应急制动；
- 在统一控制边界加入线速度/角速度加速度限制，公平约束规则法、DWA 和 PPO；
- 对 DWA 候选速度增加差速轮耦合约束，保证预测速度可由 TurtleBot3 执行；
- 将 A* 输出封装为世界坐标路径，并实现不可回退、不可隔墙跳点的航点状态机；
- 使用 YAML 已知地图栅格化中央静态柜体，并按机器人安全半径膨胀；
- 新增四巡检点 Webots 世界，后台完整跑通 A* 全局规划 + DWA 局部控制；
- 统一两个 Gymnasium 环境的逐步安全间距、累计最小间距、超时、仿真时间与实际动作字段；
- 建立版本化 episode 指标记录器和 CSV 输出接口，并隔离验证/最终测试种子。

### 可复现结果

```text
ruff: passed
pytest: 28 passed
Webots route demo: 4/4 checkpoints, 541 control steps, no collision
minimum LiDAR range: 0.225 m
```

注意：`minimum LiDAR range` 是传感器到表面的原始读数，不直接等同于机器人外轮廓净空；正式论文表格将使用统一校准后的 `minimum_clearance_m`。

## 2026-08-28：运动学课程训练与统一验证基准

> 本节保留首次运行的历史结果；当前可复现结果已由下方 2026-08-30 的干净重训更新。

### 已完成

- 完成 seed 11 四阶段 PPO 运动学课程，实际累计 811,008 steps；
- 在 `5000～5019` 验证 seeds 和 `G0～G3` 上比较 stage 1～3；
- stage 2 以 `0.6875` 成功率、`0.3125` 碰撞率优于 stage 1 和 stage 3，被选为 Webots 权重热启动；
- 用相同场景 seed 完成 Reactive、DWA、PPO 共 240 回合运动学基准；
- 加入正式 Webots 迁移/恢复训练 Controller、checkpoint、Monitor 日志和运行元数据；
- 加入 Webots 验证 Controller，并对 held-out test 增加显式确认门禁。

### 当前运动学验证结果

```text
Reactive: success 0.7250, collision 0.2750, timeout 0.0000
DWA:      success 0.5750, collision 0.3750, timeout 0.0500
PPO:      success 0.6875, collision 0.3125, timeout 0.0000
```

每种方法 80 回合，总计 240 回合。这里的 PPO 是 stage 2 运动学模型；该表不能作为 Webots 最终结论。详细边界见[当前实验进展](../reports/current-progress.md)，复现命令见[训练协议](training-protocol.md)。

### 下一里程碑

1. 完成 Webots seed 11 的 W0/W1/W2 课程微调与验证；
2. 复现 seed 22、33，并基于 Webots 验证集选择 checkpoint；
3. 将三种局部规划器接入同一 Webots `G0～G3` 批量协议；
4. 完成多航点巡检、泛化、消融和最终测试。

## 2026-08-30：运动学干净重训与正式协议加固

### 已完成

- 在 Windows Python 3.11.6、Stable-Baselines3 2.9.0、PyTorch 2.13.0 CPU 环境重新执行 seed 11 四阶段课程；
- 验证 stage 0～3 的实际累计步数分别为 `102400、253952、557056、811008`；
- 用完全相同的 `5000～5019` 验证 seeds 与 `G0～G3` 工况比较四个候选模型；
- 重新运行 Reactive、DWA、PPO 的 240 回合同场景运动学基准，并从原始 CSV 重建报告和 SVG；
- 将正式 Webots 测试设计固定为三训练 seed、验证证据锁、一次性 ledger 与 400 回合矩阵；
- 保持 `10000～10019` held-out test 未访问。

### 当前重训结果

```text
stage 0: success 0.6875, collision 0.3125
stage 1: success 0.7000, collision 0.3000
stage 2: success 0.7625, collision 0.2375  <- Webots 热启动
stage 3: success 0.7250, collision 0.2750

Reactive: success 0.7250, collision 0.2750, timeout 0.0000
DWA:      success 0.5750, collision 0.3750, timeout 0.0500
PPO:      success 0.7625, collision 0.2375, timeout 0.0000
```

stage 2 模型 SHA-256 为 `152060175d9f0db8e59e0a31d8e158e19395ff16636e6add6e9d2e6fe7372f8c`。这次结果覆盖的是运动学验证后端，只用于选择 Webots 初始化权重；即使 PPO 在该验证集上的总体指标最好，也不能据此宣称已经取得正式 Webots 或真机优势。
