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

> 本节保留 2026-08-28 本机可恢复的历史结果。下方 2026-08-30 的“干净重训”
> 声明已在 2026-09-02 因原模型与原始验证文件缺失而撤回；新 RB2A 只把本节的
> `e01bfc...` 模型作为冻结输入，不反推其原训练过程可端到端复现。

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

> 2026-09-02 完整性复核：本节当时记录的 `152060...` 模型、`61/19`
> 汇总及 `kinematic_validation_retrained_20260830` 原始文件均无法恢复，以下内容仅作
> 历史记录，不再作为实验结论。当前可复核输入为模型 `e01bfc...` 与
> `kinematic_validation_seed11` 的 `55/25` PPO 原始结果；新流程见
> `reports/webots_r2ab_targeted/rebaseline_protocol_2026-09-02.md`。

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

## 2026-09-01：避障可解释链路与小程序共享会话

### 已完成

- 将运行状态升级为向后兼容的 `patrol-live-status/v2`，从 Webots 直接记录动态障碍物位置、半径和指令速度；
- 将最近 LiDAR 距离、稳健 TTC、闭合速度、临界接近、PPO 原始动作、安全过滤动作、最终执行速度、安全盾介入和有界脱困状态贯通到看板 DTO；
- 从真实 `trajectory.csv` 重新导出 90 帧答辩回放，70 帧含精确控制步/仿真时间匹配的避障遥测，并强制保留最近接近和全部脱困关键帧；
- 在微信小程序地图绘制动态障碍物与实时速度方向，并新增“避障决策”卡片，直接呈现 `PPO → 安全过滤 → 执行`；
- 将数据源提升为 App 级共享会话，页面切换不再重启回放或云端订阅；隐藏页面停止高频绘图，返回后一次性刷新；
- 修复模式切换后旧地图、告警和终态残留，以及终态前后台切换后误显示暂停的问题；
- 修复 Cloud watch 旧异步错误可能解除新 observer 引用的竞态；
- 在 Python、Cloud 和小程序统一动态障碍上限、地图包络和半径约束；保留旧 `patrol-dashboard/v1` 快照与 outbox 的受限滚动升级兼容；
- 移除可部署配置中的密钥占位文本，增加不会输出密钥的预检，并统一拒绝空白填充、模板、示例和低多样性密钥；
- 增加真实 CloudBase 部署与微信真机验收清单，明确区分“代码/契约验证完成”和“云资源已上线”。

### 可复现结果

```text
ruff: passed
Python: 731 passed
微信小程序: 63 passed
Cloud Functions: 42 passed
CloudBase bundles: 3 built and verified
Webots operational smoke: 4/4 checkpoints, 0 collision, 651 control steps
minimum LiDAR range: 0.286139 m, escape-guard steps: 9
```

该 Webots 运行用于确认 R2025a 能读取障碍物真实几何并产出 v2 实时遥测，不新增算法效果结论。真实 CloudBase 资源、合法 HMAC canary 与微信真机仍按[真实链路验收记录](../reports/cloudbase-live-acceptance.md)标记为待验收；本轮代码没有上传为新的微信审核版本。

## 2026-09-02：一键云端联动与巡检历史闭环

### 已完成

- 增加 Webots 与 HTTPS/HMAC relay 的组合入口，统一运行编号、信号退出、relay 重启预算和终态 ACK 门禁；
- 将本地看板证据读取改为只解析追加且换行完整的 JSONL，文件截断或替换时安全重建；
- 将云端首次空库定义为健康等待，首帧到达后自动进入运行态，身份失败和运行中删除继续失败关闭；
- 在 `patrol_runs` 逐次保存不含秘密的轻量任务摘要，并增加登录鉴权的 `getPatrolHistory`；
- 增加小程序“巡检历史”页，覆盖分页、去重、运行编号、任务状态、自动时效刷新、空态和错误恢复；
- 只有成功且终态校验通过的任务显示“正常完成”；旧版终态不读取原始负载猜测结果，明确显示“旧版终态”；
- 分页游标使用域隔离 HMAC 防篡改，历史按不可变的首次接收时间排序，云端与客户端错误均采用安全短文案；
- 同步四个云函数、四个集合、三个索引的构建、权限、部署和真实链路验收清单。

### 可复现结果

```text
ruff: passed
Python: 776 passed
微信小程序: 89 passed
Cloud Functions: 55 passed
CloudBase bundles: 4 built and verified
```

本轮只更新源码、测试、构建产物和部署说明，没有部署 CloudBase、点击开发者工具“上传”或重新提交微信审核，因此当前审核版本不受影响。正式 Webots 补训仍受模型哈希与既有证据不一致的严格预检门禁保护，本轮没有启动或宣称新的训练结果。

## 2026-09-02：三端终态一致性加固

### 已完成

- 云端入口与微信小程序同步增加运行生命周期矩阵，拒绝运行态、终态标记和结果语义互相矛盾的数据；
- 完整终态同时绑定证据验证范围、结果类型和巡检点计数，运行时异常中止继续使用独立的受控最小证据分支；
- `progress_ratio` 与一位小数的 `progress_percent` 只能由完成巡检点数量推导，拒绝“4/4 但 0%”等伪造状态；
- 增加成功、各类正常失败、异常中止、提前成功、错误终态锁和矛盾进度的对称回归测试。

### 可复现结果

```text
微信小程序: 91 passed
Cloud Functions: 59 passed
```

本轮没有部署 CloudBase、上传微信代码或启动训练；当前微信审核版本不受影响。

## 2026-09-02：可追溯完整巡检报告

### 已完成

- 从通过 manifest 校验的 `inspection_samples.csv` 投影 P1～P4 点位采样，并把样本编号、传感器元数据、告警状态与终态巡检记录交叉绑定；
- 在 CloudBase 中只为普通已验证终态且告警历史完整的任务保存 `patrol-report/v1` 安全白名单，旧版、运行时中止和截断证据不会生成完整报告；
- 增加登录鉴权、固定设备和严格运行编号的 `getPatrolReport`，报告读取不透传 `terminal_payload`、请求哈希、nonce、签名或密钥；
- 小程序历史页按 cloud、demo、local 三种模式展示真实可用档案，只有 `report_available=true` 的记录可以进入详情；
- 新增完整报告页，展示终态核心指标、避障指标、最终轨迹、逐点温度/烟雾/一氧化碳读数、告警时间线、运行身份和仿真边界；
- 普通任务在到达 P1 前失败时保留诚实的零点位报告；旧记录缺失巡检证据时使用 `sample_count=null` 表示未知，不用 0 冒充已确认无采样；
- 三端补充终态矩阵、轨迹末点、地图状态、告警顺序与归属、样本范围、传感器身份、UTC 时间及严格字段白名单的对称回归测试。

### 可复现结果

```text
ruff: passed
Python: 789 passed
微信小程序: 106 passed
Cloud Functions: 78 passed
CloudBase bundles: 5 built and verified
```

本轮只更新源码、测试、构建配置与部署说明，没有部署 CloudBase、上传微信代码、重新提交微信审核或启动训练；当前审核版本不受影响。
