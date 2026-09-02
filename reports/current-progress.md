# 当前实验进展与结果边界

更新时间：2026-09-02。本页是开发阶段的可审计快照，不是毕业论文最终结果页。

## 1. 正式实验闭环实现状态

正式实验的软件协议已经补齐，但正式 Webots 数据尚未产生。当前已实现：

- 固定三次独立 PPO 训练 seed：`11、22、33`；
- 固定验证场景 seeds：`5000～5019`，只允许验证数据参与 checkpoint 选择；
- 固定 held-out test seeds：`10000～10019`；
- 在 W2 训练阶段为 immutable checkpoint/final model 生成 provenance sidecar，并复核三个 seed 的课程与算法配置一致性；
- 每个 seed 以 `SEED MODEL METADATA CURRICULUM SELECTION VALIDATION_EPISODES` 六个 `--model-evidence` 值提供模型证据；训练期 sidecar 由课程 manifest 登记，create-lock 只复核而不补写；
- 用模型、sidecar、训练元数据、课程/选择 manifest、验证 CSV、正式配置、world、evaluator、controller、完整 source bundle 和 clean Git revision 的 SHA-256 创建不可覆盖的 FinalLock v2；
- `--dry-run` 只读探测 Webots 版本与可执行文件哈希，读取并复核 world 文件哈希，预检恰好 20 个调用，但不启动 Webots 仿真、不会让模拟器打开 world，也不创建 ledger；
- 消费键只由受保护测试协议派生，同一协议的所有 campaign 共用 `artifacts/results/formal_ledgers/protected_test_<consumption-key>.json`；自定义 ledger 路径会被拒绝；
- runtime evidence 与 `attempt` 原子写入全局 ledger，任何已开始的 held-out test 只能进入 `success` 或 `invalidated` 终态，不能通过改 campaign、输出目录或 lock 重试；
- 严格汇总 `G0～G3 × 5 planners × 20 test seeds = 400 episodes`，从原始 CSV 复算指标并绑定模型哈希。

对应的公开入口是 [`configs/eval/webots_formal.yaml`](../configs/eval/webots_formal.yaml)、[`scripts/create_webots_final_lock.py`](../scripts/create_webots_final_lock.py)、[`scripts/run_webots_formal_test.py`](../scripts/run_webots_formal_test.py) 和 [`scripts/summarize_webots_formal.py`](../scripts/summarize_webots_formal.py)。正式执行不应绕过这些入口直接手工设置 test 环境变量。

五个 planner 为 `Reactive、DWA、PPO-seed11、PPO-seed22、PPO-seed33`。四个条件依次覆盖训练布局标称动态障碍、未见静态布局、高密度五动态障碍和更高速动态障碍。完整命令和失败语义见[训练复现协议](../docs/training-protocol.md)。

以下事项截至本页更新时间**没有完成**：三套 Webots 正式模型及其验证证据、三模型 final lock、正式 dry-run 记录，以及一次性 400 回合 held-out test。因此仓库当前没有“正式 Webots 成功率/碰撞率”，也没有可写入论文摘要的 PPO 三训练 seed 均值、标准差或置信区间。协议代码通过测试只说明实验管线的约束可执行，不代表机器人算法已取得预期效果。

## 2. 已归档的新基线输入

2026-09-02 完整性复核撤回了无法恢复的 `152060...` stage-2 模型及其
`61 成功 / 19 碰撞` 声明。当前归档的是更早、但模型字节和原始验证行均可复核的
冻结输入：`artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip`，
SHA-256 为 `e01bfcfa5dba22db4a8adf8b0ae8b325a6f7ef7eb50c2b1af1840e3d0c079c67`，
模型内部累计步数为 `557,056`。

model registry 中保留的候选记录如下；只有 stage 2 的模型与原始统一验证 CSV 被
归档为新基线输入，stage 1/3 仅作历史选择记录：

| 阶段 | 累计步数 | 成功/碰撞/超时 | 成功率 | 碰撞率 |
|---|---:|---:|---:|---:|
| stage 1 | 253,952 | 52 / 28 / 0 | 0.6500 | 0.3500 |
| stage 2 | 557,056 | **55 / 25 / 0** | **0.6875** | **0.3125** |
| stage 3 | 811,008 | 50 / 30 / 0 | 0.6250 | 0.3750 |

该冻结输入的原始训练过程没有 clean Git revision 与不可变训练 manifest，因此不能
反推为端到端可复现的旧训练。它的角色仅是新 RB2A 从当前 clean commit 启动时的
哈希锁定 warm-start；新 Webots 训练过程会重新生成 manifest、metadata、逐回合 CSV
和摘要。

## 3. 运动学统一基准

下表来自 `G0～G3 × 20 validation seeds × 3 planners = 240 episodes`。三种方法使用完全相同的运动学场景 seed；每个方法汇总 80 个 episode。

| 方法 | 成功/碰撞/超时 | 成功率 | 碰撞率 | 超时率 |
|---|---:|---:|---:|---:|
| Reactive | 58 / 22 / 0 | **0.7250** | **0.2750** | 0.0000 |
| DWA | 46 / 30 / 4 | 0.5750 | 0.3750 | 0.0500 |
| PPO stage 2 | 55 / 25 / 0 | 0.6875 | 0.3125 | 0.0000 |

当前这一次运动学验证中，Reactive 的总体成功率与碰撞率最好；PPO 的成功回合平均
完成时间和路径长度比更低。该结论只适用于本项目的当前运动学实现、固定验证 seeds
与参数，不能外推为某类算法普遍更优，也不能代替 Webots 物理仿真结果。

原始本地证据：

- `artifacts/results/kinematic_validation_seed11/episodes.csv`：240 行逐回合数据；
- `artifacts/results/kinematic_validation_seed11/summary.json`：按方法与条件聚合；
- `reports/kinematic_validation/`：只由上述两个输入重新生成的表格与 SVG。

模型、CSV 与 summary 已作为新基线输入归档并由 requirement manifest 绑定；论文图表
应继续由原始文件生成，不手工抄表。

## 4. 已验证但不能作为效果结论的内容

- Webots R2025a 在 Mac mini M4 原生 ARM64 环境可运行；
- Webots Controller、Supervisor、LDS-01、差速轮和 Gymnasium/PPO 链路已跑通；
- seed 99 的 512-step Webots 训练与两回合评测只属于冒烟测试；
- A* + DWA 演示世界完成过 4/4 巡检点、541 控制步、无接触碰撞；该单次演示不是批量统计实验。

任何 256/512-step 模型、单次成功演示或运动学权重都不得命名为“最终模型”。

### `curriculum_v2` 探索路线状态

`configs/train/webots_curriculum_v2.yaml` 定义了 R1/R2/R3 单障碍由慢到快的探索性课程。配置中的 `policy_transfer_model` 和 `preferred_resume_model` 只是本机训练恢复提示；对应 checkpoint 没有随仓库提供，路径本身也不能证明某个阶段已经完成。

本开发快照尚未提供 R1/R2/R3 全流程完成记录、固定正式门禁通过记录，或同时绑定评测模型路径、variant 与 SHA-256 的可加载 manifest，R3 的恢复模型也尚未锁定。R1/R2 配置允许的碰撞率 0.25、R3 的 0.30 都高于正式巡检上限 0.15，只能作为阶段探索门禁。因此 `curriculum_v2` 不计入已完成的 Webots 正式训练 seed，不支持任何正式效果结论。

### R2a/R2b 完整性复核与新基线

旧 R2a 的 `88/12` 和旧 R2b 的 `83/17` 下游模型、CSV 与摘要仍被保留，哈希也可
逐项校验；但它们继承的上游 warm-start 身份无法端到端恢复。因此这两组数字自
2026-09-02 起只作为历史验证记录和演示资产，不再作为当前论文实验结论。

新的 `r2a_rebaseline_v1` 已按预注册协议完成：从上述 `e01bfc...` 冻结输入出发，在
clean commit `3335fa3...` 上重新训练 `40,960` 步，PPO 与完整前向制动安全盾共同构成
受评系统。端点模型 SHA-256 为 `f52a4691...06c2c`，内部步数为 `40,960`。

从未用于旧 R2a/R2b 选择的 `6000～6099` 共 100 个固定 validation seeds 得到
`76 成功 / 24 碰撞 / 0 超时 / 0 错误`，成功率 `0.76`、碰撞率 `0.24`，通过预先锁定
的 `>=0.75` 与 `<=0.25` 门槛。验证种子逐行齐全，训练/验证 reset health 分别为
`370/370` 与 `100/100`，运行清单明确记录未执行 `10000～10019` held-out test。

该结果只比两个门槛各多 `0.01`，即一个 episode，因此应表述为“可追溯的单 seed
开发基线通过”，不能表述为稳健、正式或多 seed 结论。RB2A 已允许生成哈希绑定的
RB2B 续训配置。该配置现已锁定：它复制并绑定 RB2A 更新后端点模型的完整
`f52a4691...06c2c` SHA-256，保留优化器和已学习动作分布，只把障碍速度提高到
`0.18～0.25 m/s`，其余奖励、PPO、安全盾、路线和验证门槛保持不变。完整证据见
[`RB2A 端点报告`](webots_r2ab_targeted/r2a_rebaseline_v1_report.md)
和[`新基线协议`](webots_r2ab_targeted/rebaseline_protocol_2026-09-02.md)。

## 5. 应用展示链路状态

Webots 现在会发布版本化的动态障碍与避障遥测：障碍物位置、半径、速度，最近 LiDAR 距离、稳健 TTC、PPO 原始动作、安全过滤动作、最终执行速度、安全盾介入与有界脱困状态均可投影到本地看板和微信小程序。答辩离线夹具的 90 帧中有 70 帧与原始 `trajectory.csv` 的控制步和仿真时间精确匹配；其余帧明确标记为遥测不可用，不插值补造。地图和决策卡片提供应用层可解释展示，但仍不构成形式化安全证明。

一次新的 Webots R2025a `operational / seed 7000` 集成冒烟完成 4/4 点位、0 次接触碰撞、651 个控制步；最低 LiDAR 距离为 `0.286139 m`，记录 9 个有界脱困介入步。该运行只验证 Supervisor 几何读取、控制器与 v2 状态链路可以真实执行，不能替代已登记的批量应用验收，也不进入正式 PPO 三 seed 结论。

小程序、云函数和 Mac relay 已通过离线三端契约测试，旧 `patrol-dashboard/v1` 数据可在部署窗口受限兼容；App 级共享会话、隐藏页面降载、来源切换清空、云订阅竞态防护和密钥预检也已有回归测试。新普通终态还可生成独立的只读完整报告，把 P1～P4 点位读数、告警时间线、最终轨迹、避障指标与运行身份绑定到同一 `run_id`；旧版、运行中断和告警截断记录不会被误标为完整报告。真实 CloudBase 资源和微信真机尚未按[验收记录](cloudbase-live-acceptance.md)留下证据，因此不能宣称公网实时链路已经部署通过。

## 6. 尚缺少的正式证据

- [ ] Webots seed 11、22、33 的完整课程微调；
- [ ] 三个 seed 均在 Webots `5000～5019` 验证 seeds 上选择出通过固定门禁的正式 checkpoint；
- [ ] 三个 checkpoint 的模型、训练期 sidecar、元数据、课程/选择 manifest 和验证 CSV 均通过哈希复核；
- [ ] clean Git revision 上创建三模型 FinalLock v2，并归档 lock SHA-256；
- [ ] 正式 runner 的 dry-run 记录已复核 world 哈希并包含预期的 20 个调用，且未启动 Webots 仿真或创建 ledger；
- [ ] Webots 中 Reactive、DWA、三个独立 PPO seed 的 400 回合同场景批量比较；
- [ ] 未见布局、高密度、高速度条件的 Webots 泛化结果；
- [ ] 完整多航点巡检的 10 次以上批量结果；
- [ ] 安全奖励和场景随机化消融；
- [ ] safety shield 开启/关闭的同 seed 消融与介入率分析；
- [ ] 锁定配置后运行一次 held-out test，并保留 `success` 或 `invalidated` ledger 终态；
- [ ] 三训练 seed 的均值、标准差或置信区间；
- [ ] 失败轨迹、代表性视频和模型 SHA-256 发布记录。

完成以上项目后，才能把 Webots 结果写入论文摘要和最终结论。
