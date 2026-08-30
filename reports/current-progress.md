# 当前实验进展与结果边界

更新时间：2026-08-30。本页是开发阶段的可审计快照，不是毕业论文最终结果页。

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

## 2. 已完成的训练

seed 11 已完成四阶段运动学课程训练。YAML 请求总步数为 800,000，Stable-Baselines3 按完整 rollout 更新后的实际累计步数为 811,008。

验证集模型选择比较了 stage 0～3。stage 2 `nominal_dynamic` 在 `G0～G3` 共 80 个验证 episode 上取得最高成功率，因此被锁定为 Webots 的热启动来源。该文件的 SHA-256 为 `152060175d9f0db8e59e0a31d8e158e19395ff16636e6add6e9d2e6fe7372f8c`：

| 阶段 | 累计步数 | 成功/碰撞/超时 | 成功率 | 碰撞率 |
|---|---:|---:|---:|---:|
| stage 0 | 102,400 | 55 / 25 / 0 | 0.6875 | 0.3125 |
| stage 1 | 253,952 | 56 / 24 / 0 | 0.7000 | 0.3000 |
| stage 2 | 557,056 | 61 / 19 / 0 | **0.7625** | **0.2375** |
| stage 3 | 811,008 | 58 / 22 / 0 | 0.7250 | 0.2750 |

这说明从 stage 2 继续在更密、更快场景训练后，当前验证分布上的表现略有回落。stage 2 的选择属于正常的验证集早停/模型选择，不是最终测试结论。

## 3. 运动学统一基准

下表来自 `G0～G3 × 20 validation seeds × 3 planners = 240 episodes`。三种方法使用完全相同的运动学场景 seed；每个方法汇总 80 个 episode。

| 方法 | 成功/碰撞/超时 | 成功率 | 碰撞率 | 超时率 |
|---|---:|---:|---:|---:|
| Reactive | 58 / 22 / 0 | **0.7250** | **0.2750** | 0.0000 |
| DWA | 46 / 30 / 4 | 0.5750 | 0.3750 | 0.0500 |
| PPO stage 2 | 61 / 19 / 0 | **0.7625** | **0.2375** | 0.0000 |

当前这一次运动学验证中，PPO 的总体成功率和碰撞率最好，并在 `G0/G1` 上优于两个传统基线；在更困难的 `G2/G3` 上则与 Reactive 持平。该结论只适用于本项目的当前运动学实现、固定验证 seeds 与参数，不能外推为强化学习普遍优于传统方法，也不能代替 Webots 物理仿真结果。

原始本地证据：

- `artifacts/results/kinematic_validation_retrained_20260830/episodes.csv`：240 行逐回合数据；
- `artifacts/results/kinematic_validation_retrained_20260830/summary.json`：按方法与条件聚合；
- `artifacts/results/model_selection_retrained_stage{0,1,2,3}_*/`：四个阶段的选择数据。

这些文件默认被 Git 忽略，可通过 `docs/training-protocol.md` 中的命令重建。论文图表应由原始文件生成，不手工抄表。

## 4. 前向制动 SafetyShield 独立诊断确认

recovery-v1 风险诊断之后，项目只冻结了一个处理候选：模型权重和策略角速度保持不变，
当前方 `±60°` 扇区净空进入 `0.75 m` 减速区时限制线速度，并在 `0.24 m` 处限制为零。
原 v1 配对实验使用 `25728～25847`，主效应门槛全部通过，但报告检查器在 float32 执行
动作与 double 速度上限的 `1e-6` 边界上少计了一步，因此原报告继续保持
`invalid_manipulation`。离线复核从两组 float32 动作重新计算了全部介入，确认候选组
`3197/3197` 个 active step 都符合预设变换；该事后复核没有把 v1 改写为有效实验，只允许
开始一次新样本确认。

独立确认 v2 在 clean commit `680e6aceb15b3c655c8e2bf03438d877ca23a0a1` 上执行，使用
与 v1、validation、held-out test 均不重叠的 `54775～54894`。该连续窗口只按处理前协变量
选择：水平/垂直 × 正/反向四格各 30 回合，障碍初始方向正负各 60 回合；模型、参数、
样本数和所有通过门槛在运行前冻结，且禁止合并 v1/v2 结果补救 v2 失败。

| v2 处理臂 | 成功/碰撞/超时 | 成功率 | 碰撞率 |
|---|---:|---:|---:|
| recovery-v1 纯 PPO | 66 / 54 / 0 | 0.5500 | 0.4500 |
| PPO + 前向制动 | 95 / 25 / 0 | **0.7917** | **0.2083** |

120 对同场景转移为 `19 collision→collision`、`35 collision→success`、
`6 success→collision`、`60 success→success`。碰撞相对下降 `0.5370`，以
success/non-success 为端点的单侧精确 McNemar `p=2.436799150018487e-6`。两臂均无
timeout/error；共同成功为 60 对，逐 seed 候选/参考完成时间的中位比为 `1.0722`
（第 90 百分位为 `1.1897`），说明收益伴随可见但受控的通行时间代价。

候选组在 12,741 个控制步中介入 3,740 步（`29.35%`），覆盖 108/120 个回合。严格
float32 复核确认 3,740 个 active step 全部是速度上限绑定的线性制动，12,741 个步骤的
角动作均位级不变，记录的 intervention/active 与动作重算完全一致。runner 的主门槛和
操纵完整性检查均通过，自动状态为 `passed`。完整输入、图表、哈希与边界说明见
[前向制动独立诊断确认报告](webots_front_brake_confirmation/confirmation_report.md)。

该结果支持的结论严格限定为：**在 recovery-v1 单个训练 seed、DB1 单动态障碍和当前
Webots 实现中，冻结的前向制动闭环 profile 得到独立 diagnostic 确认。**它不等于
validation、held-out test、三训练 seed 均值、复杂 G0～G3 场景泛化或形式化安全保证。

## 5. 已验证但不能作为效果结论的内容

- Webots R2025a 在 Mac mini M4 原生 ARM64 环境可运行；
- Webots Controller、Supervisor、LDS-01、差速轮和 Gymnasium/PPO 链路已跑通；
- seed 99 的 512-step Webots 训练与两回合评测只属于冒烟测试；
- A* + DWA 演示世界完成过 4/4 巡检点、541 控制步、无接触碰撞；该单次演示不是批量统计实验。

任何 256/512-step 模型、单次成功演示或运动学权重都不得命名为“最终模型”。

### `curriculum_v2` 探索路线状态

`configs/train/webots_curriculum_v2.yaml` 定义了 R1/R2/R3 单障碍由慢到快的探索性课程。配置中的 `policy_transfer_model` 和 `preferred_resume_model` 只是本机训练恢复提示；对应 checkpoint 没有随仓库提供，路径本身也不能证明某个阶段已经完成。

本开发快照尚未提供 R1/R2/R3 全流程完成记录、固定正式门禁通过记录，或同时绑定评测模型路径、variant 与 SHA-256 的可加载 manifest，R3 的恢复模型也尚未锁定。R1/R2 配置允许的碰撞率 0.25、R3 的 0.30 都高于正式巡检上限 0.15，只能作为阶段探索门禁。因此 `curriculum_v2` 不计入已完成的 Webots 正式训练 seed，不支持任何正式效果结论。

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
- [ ] 在三训练 seed 和正式 G0～G3 条件中验证 safety shield，而不是用单模型 DB1 diagnostic 代替；
- [ ] 锁定配置后运行一次 held-out test，并保留 `success` 或 `invalidated` ledger 终态；
- [ ] 三训练 seed 的均值、标准差或置信区间；
- [ ] 失败轨迹、代表性视频和模型 SHA-256 发布记录。

完成以上项目后，才能把 Webots 结果写入论文摘要和最终结论。
