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

## 2026-08-30：Webots 回合重置稳定性修复

### 根因与修复

首次 W0 长训练在多次回合重置后出现非有限观测。消融诊断确认，异常并非 PPO 更新或单独的 GPS 丢包：出错时 TurtleBot 的根节点位置、速度和接触状态已同时变为 `NaN`。旧逻辑只搬移根节点并清除惯性，没有恢复两个轮毂 `HingeJoint` 和脚轮 `BallJoint` 的完整内部状态，反复 teleport 后最终使物理约束失效。

修复后的每次 reset 会先用 `loadState("__init__")` 恢复 TurtleBot 及全部后代关节，再重新设置轮电机速度模式、根节点位姿和节点级物理状态。全局 `simulationResetPhysics()` 已移除。两次 settle step 均检查 Supervisor 状态，最终还会核对起点 XY、GPS/根节点偏差、IMU yaw、线速度、角速度和初始碰撞；GPS、IMU、动作、奖励或最终观测出现非有限值时一律失败，不使用 Supervisor 真值或旧传感器值替代策略输入。

训练与评测现在都会记录 reset health；失败时会额外保存异常、原始传感器值、Supervisor 状态、回合位置和 traceback。评测产物中的 reset health 属于运行完整性证据，不是策略可见输入。

### 修复后诊断结果

使用 seed 11、W0、正式默认 PPO 参数运行 6,144 steps：

```text
episodes: 50/50 success, 0 collision, 0 timeout
PPO updates: 110, all policy/optimizer/TensorBoard values finite
reset: 51 attempts, 51 successful, 102 finite settle checks, 51 readback checks
maximum start XY error: 0.000026 m
maximum GPS/root XY distance: 0.036032 m
maximum yaw error: 0.000003 rad
maximum reset linear/angular speed: 0.075327 m/s / 1.237338 rad/s
```

随后只在 validation seeds `5000、5001` 上完成 2 回合集成检查，结果为 2/2 success，且 `summary.json` 已包含相同的 reset-health provenance。该 2 回合运行只验证评测链路，不用于 checkpoint 选择，也不是正式实验结果。完整 Python 测试为 `248 passed`；`10000～10019` held-out test 仍未访问。

## 2026-08-30：首次 Webots 正式课程门禁结果

在 commit `4b85afef33da83410da7cc9f0d7f77ce9b0a9932` 上启动 seed 11 的预声明
`W0 → W1 → W2` 课程。W0 实际训练到 `50,176` steps，随后在 validation seeds
`5000～5019` 上取得 `20/20` success、`0/20` collision，正式通过门禁。W1 从该模型
继续训练到累计 `200,192` steps；训练和验证的 reset health 均全部通过，策略参数、优化器
与 TensorBoard 标量没有 NaN/Inf。

W1 固定验证结果为：

```text
success:   8/20 = 0.40
collision: 12/20 = 0.60
timeout:   0/20 = 0.00
gate: success >= 0.75 and collision <= 0.15  -> failed
```

runner 因门禁失败以预定状态停止，W2 没有启动，`test_split_was_run=false`。W1 最终模型
SHA-256 为 `84e0ebb459e0ef800b1e891f5d549237e246c04be33a3bed24f4811b18993cc8`；
manifest、训练 metadata、模型、validation summary 与 episodes CSV 已完成一致性审计。

对 1,673 个 W1 训练回合的离线诊断显示，总体成功率为 `45.91%`，最后 100 回合为
`61%`，但没有稳定泛化到 validation。水平长路线成功率为 `34.80%`，垂直路线为
`56.56%`；从低、中、高三个等宽障碍速度区间，成功率依次为 `56.92%、43.50%、
36.88%`。905 次 collision 中有 754 次（`83.31%`）是 LiDAR 阈值单独触发，表明
主要短板是中高速横穿障碍下的提前让行，而不是物理 reset、数值稳定性或 timeout。

该失败结果按原样保留，不通过放宽碰撞阈值修饰。下一轮只使用独立 diagnostic seeds
验证渐进速度课程、提前减速奖励和完整路线恢复；新配置冻结后才重新运行 validation，
最终 `10000～10019` held-out test 继续封存。

## 2026-08-30：W1 recovery-v1 诊断门禁结果

在 clean commit `9e8cb7d08a59b78f931704d6c3ea7083b3c36dab` 上，以首次 W1 失败模型
`84e0ebb...93cc8` 为精确策略迁移源执行 recovery-v1 的 E1。训练完成 `40,960`
steps，模型 SHA-256 为
`c5e7c3de42c7664477097b2734975660fec840ad5b7f45425098158c27cb7275`；482 次 reset
全部成功，训练过程未出现 NaN/Inf。

E1 只访问 diagnostic seeds `20000～20019`，结果为：

```text
success:   13/20 = 0.65
collision:  7/20 = 0.35
timeout:    0/20 = 0.00
gate: success >= 0.70 and collision <= 0.30  -> failed
```

runner 按门禁停止，E2、validation 和 held-out test 均未启动。离线训练诊断显示总体
成功率 `63.41%`，最后 100 回合提升到 `70%`；水平路线成功率 `55.88%`，垂直路线
`70.78%`。176 次 collision 中 156 次（`88.64%`）为 LiDAR-only，说明恢复方向已经
有效，但需要更多同难度巩固和更细的路线/速度桥接，不能通过事后降低门槛进入下一阶段。

下一轮 `webots_w1_recovery_v2.yaml` 从 E1 模型的精确 SHA 迁移到隔离 variant，依次使用
全新的 diagnostic 窗口 `20040～20059`、`20060～20079`、`20080～20099`。C1 保持
短路线进行巩固，C2 再扩大速度与水平路线，C3 才恢复完整速度；只有三段均通过后，F1
才允许访问一次 validation `5000～5019`。`10000～10019` 继续封存。
