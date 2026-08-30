# 基于深度强化学习的室内巡检机器人动态避障仿真系统

本项目面向已知静态室内地图中的巡检任务，研究移动机器人面对行人、推车等动态障碍物时的安全导航问题。系统采用分层架构：A* 负责全局巡检路线，PPO 根据 LiDAR 和局部目标信息输出线速度与角速度，Webots 提供三维物理仿真。

项目是纯软件本科毕业设计，重点成果不是单条训练曲线，而是一个可复现的完整闭环：

> 场景生成 → 传感器观测 → 全局路径 → 局部避障 → 机器人执行 → 指标记录 → 批量评测

## 当前进度

- [x] 建立 Git 仓库和 Python 包结构
- [x] 固定 M4 原生 Python 3.11 环境方案
- [x] 实现符合 Gymnasium 接口的轻量二维动态避障原型
- [x] 定义与 Webots 共用的 79 维观测和二维连续动作接口
- [x] 加入规则避障冒烟基线、单元测试和 GitHub Actions
- [x] 通过 Webots Python Controller 与 Supervisor 的 M4 fast 模式冒烟测试
- [x] 验证 TurtleBot3 差速轮与 LDS-01 LiDAR，并建立动态障碍场景
- [x] 实现占据栅格 A* 与路径简化，并建立 Webots Gymnasium 环境
- [x] 实现 DWA 基线和 A* 多航点巡检演示
- [x] 完成 Webots 真实传感器到 PPO 模型保存的短链路验证
- [x] 完成 seed 11 的 811,008-step 运动学课程训练和阶段模型选择
- [x] 完成 Reactive、DWA、PPO 的 240 回合运动学统一验证基准
- [x] 实现三训练 seed、验证证据锁、一次性测试 ledger 和 400 回合汇总的正式实验协议
- [x] 完成 recovery-v1 的 120 回合 Webots 风险诊断与可审计图文报告
- [x] 完成前向制动 SafetyShield 的 120 对同场景独立诊断确认与逐步动作审计
- [ ] 完成 Webots 三 seed 正式训练、统一批量评测、泛化和奖励消融

轻量二维环境不是最终实验平台。它用于快速验证状态、动作、奖励、随机种子、训练和测试代码，并为 Webots 提供策略热启动。当前验证集选择的是 stage 2（557,056 steps），不是训练最久的 stage 3；完整选择依据见[当前实验进展](reports/current-progress.md)。正式论文效果结论仍以 Webots 批量实验为准。

最新的[Webots 动态避障风险诊断公开版](reports/webots_risk_diagnostic/diagnostic_report.md)提供按路线方向结果、机制证据对比和固定 seed 20500 风险时间线。该批次只使用 `diagnostic` seeds；正式 v1 机制结论为 `indeterminate`。LiDAR 原点敏感性是 post-hoc、nonformal 审计，不替代 v1，也没有访问 validation 或 held-out test 数据。

在此基础上，项目又用全新 `54775～54894` diagnostic seeds 对唯一冻结的前向制动候选做了独立同场景确认。纯 PPO 为 `66/120` success、`54/120` collision；启用候选后为 `95/120` success、`25/120` collision，碰撞相对下降 `53.70%`，配对转移为 `35` 次救回、`6` 次新增失败，单侧精确 McNemar `p=2.4368×10⁻⁶`。共同成功回合的完成时间中位比为 `1.0722`，没有 timeout 或 error。完整的[图文报告](reports/webots_front_brake_confirmation/confirmation_report.md)同时公开原始两臂结果、逐步 shield 证据、哈希和复现入口。这个 `passed` 只表示 recovery-v1 单模型、单障碍 DB1 场景中的独立诊断确认，不是 validation、held-out test、三训练 seed 或泛化结论。

正式 Webots 实验的**执行协议已经实现，但正式结果尚未产生**。固定流程为：分别用 `11、22、33` 三个训练 seed 训练和验证 PPO；在 clean Git revision 上将三个模型的训练期 sidecar、训练元数据、W0～W2 课程 manifest、验证选择 manifest 和逐回合验证 CSV 与正式配置、world、evaluator、controller 及完整 source bundle 一起绑定到不可覆盖的 FinalLock v2；先执行不消耗测试机会的 `--dry-run`；最后才对 `10000～10019` held-out seeds 运行一次 `G0～G3 ×（Reactive、DWA、PPO-seed11、PPO-seed22、PPO-seed33）= 20` 个 Webots 调用、共 400 回合。当前仓库没有可引用为论文结论的这 400 回合正式 Webots 输出，详见[训练复现协议](docs/training-protocol.md)和[当前实验进展](reports/current-progress.md)。

创建 final lock 时，每个 seed 的 `--model-evidence` 必须依次提供 `SEED MODEL METADATA CURRICULUM SELECTION VALIDATION_EPISODES` 六个值。训练期 `.formal-provenance.json` sidecar 不是第七个命令行参数；它必须已由 W2 训练生成并被课程 manifest 登记，create-lock 只做定位和复核，不会事后补写。

## 可审计模型选择与安全过滤

- Webots 评测汇总会绑定实际模型路径、variant 与 SHA-256；checkpoint 只能在 `validation` 数据上通过门禁后写入选择 manifest。正式巡检还固定要求至少 20 回合、成功率不低于 0.75、碰撞率不高于 0.15，并在加载时复核 manifest 指标和模型哈希。可变的 `latest`、运动学热启动和显式模型都不会被静默当成正式模型，探索运行必须设置 `WEBOTS_PATROL_ALLOW_UNVALIDATED_MODEL=1`。完整命令见[训练复现协议](docs/training-protocol.md)。
- Webots 环境已接入可选的 LiDAR safety shield，记录策略动作、过滤后动作、实际执行动作和介入指标。前向制动候选已在单模型、单障碍 `diagnostic` 配对实验中获得独立确认，但它仍是默认关闭的应用层启发式过滤，不是形式化安全保证；在三 seed、G0～G3 validation 与 held-out test 完成前，不能把该结果写成正式指标或跨场景泛化结论。

正式测试不能通过手工设置 `WEBOTS_EVAL_SPLIT=test` 绕过协议。推荐入口是 `scripts/run_webots_formal_test.py`：它校验 clean Git revision、配置/模型/验证证据哈希和 final lock，拒绝已有输出。消费键只由固定的受保护测试协议派生，不包含 campaign、模型、输出目录或时间戳；同一协议的所有 campaign 共用唯一的 `artifacts/results/formal_ledgers/protected_test_<consumption-key>.json`，自定义 ledger 路径会被拒绝。`--dry-run --confirm-test` 会只读探测 Webots 版本与可执行文件哈希，读取并复核 lock 绑定的 world 文件哈希，再打印并校验 20 个调用；它不启动 Webots 仿真、不会让模拟器打开 world，也不创建 ledger 或消耗测试机会。去掉 `--dry-run` 后，runner 会把 runtime evidence 与 `attempt` 原子写入全局 ledger；之后无论终态为 `success` 还是 `invalidated`，都不能通过改 campaign、输出目录或 lock 获得第二次机会。

## 系统边界

本项目包含：

- 单台差速轮巡检机器人；
- 已知地图和 4～6 个巡检航点；
- LiDAR、模拟定位与碰撞信息；
- 横穿、迎面、路口交叉和随机游走动态障碍；
- A* 全局规划；
- PPO、DWA 和规则避障三种局部方法；
- 自动批量评测、轨迹日志、泛化实验和奖励消融。

本项目不包含 SLAM、摄像头端到端控制、多机器人协作、ROS 2、Gazebo、实体机器人迁移或 VLA 大模型。

## 架构

```text
已知地图与巡检航点
        │
        ▼
    A* 全局规划
        │ 局部目标
        ▼
┌──────────────────────┐
│ PPO / DWA / 规则避障  │  ◀── LiDAR、目标方位、当前速度
└──────────────────────┘
        │ v, ω
        ▼
   差速轮速度控制
        │
        ▼
 Webots 机器人与动态障碍
        │
        └──▶ 日志、指标与轨迹
```

详见[系统架构](docs/architecture.md)、[强化学习设计](docs/rl-design.md)、[实验方案](docs/experiment-plan.md)和[训练复现协议](docs/training-protocol.md)。

## Mac mini M4 环境

固定使用原生 ARM64 Python 3.11，PPO 的 MLP 策略默认使用 CPU。训练阶段关闭 Webots 渲染并使用 fast 模式，演示阶段再开启三维窗口。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
python scripts/configure_webots_python.py
```

最后一条命令会为本机生成不提交到 Git 的 `runtime.ini`，确保 Webots
Controller 使用项目 `.venv`，而不是 macOS 的其他 Python。仓库移动或重新克隆后再运行一次即可。

安装 Webots 后应在 Finder 的“显示简介”中确认没有勾选“使用 Rosetta 打开”。本机安装与检查记录见 [M4 环境说明](docs/mac-m4-setup.md)。

## 快速验证

运行测试：

```bash
source .venv/bin/activate
ruff check .
pytest
```

运行规则避障冒烟测试：

```bash
python scripts/smoke_env.py --episodes 5 --seed 42
```

运行一次短 PPO 训练，验证训练链路：

```bash
python scripts/train_ppo.py --timesteps 5000 --n-envs 4 --n-steps 256
python scripts/evaluate_ppo.py artifacts/models/ppo_kinematic.zip --episodes 10
```

`artifacts/` 用于本地模型、日志、结果和视频，默认不提交大文件到 Git。最终模型和演示视频计划通过 GitHub Release 发布。

运行完整运动学课程与统一验证基准：

```bash
python scripts/train_curriculum.py --seed 11 \
  --output artifacts/models/ppo_curriculum_seed11
python scripts/benchmark_kinematic.py --split validation \
  --ppo-model artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip \
  --output-dir artifacts/results/kinematic_validation_seed11
```

当前 240 回合运动学验证的成功率/碰撞率为：Reactive `0.725/0.275`、DWA `0.575/0.375`、PPO stage 2 `0.7625/0.2375`。PPO 在当前固定验证集上的总体指标最好，但这组结果只用于开发期比较，不能包装成最终 Webots 结论。Webots 正式训练、恢复、验证和最终测试门禁命令见[训练复现协议](docs/training-protocol.md)。

打开 Webots 动态避障演示：

```bash
open -a Webots worlds/patrol_dynamic_demo.wbt
```

进入世界后点击工具栏的三角形运行按钮。红色圆柱是横穿走廊的动态障碍，绿色圆盘是目标点。当前里程碑及复现结果见[开发记录](docs/development-log.md)。

打开 A* + DWA 四点巡检演示：

```bash
open -a Webots worlds/patrol_route_demo.wbt
```

机器人会根据已知静态地图绕过中央柜体，依次访问四个彩色巡检点；黄色圆盘表示当前 A* 局部航点。

## 目录结构

```text
configs/       环境、训练和评测配置
controllers/   Webots 控制器、训练与评测入口
docs/          架构、强化学习和实验协议
protos/        Webots 自定义机器人与障碍模型
scripts/       训练、评测和冒烟脚本
src/           可测试的 Python 核心包
tests/         单元测试与接口测试
worlds/        Webots 训练、测试和演示世界
artifacts/     本地生成的模型、日志、结果和视频（默认 Git 忽略）
```

## 验收指标

主要指标包括任务成功率、碰撞率、超时率、路径效率、完成时间、最小安全距离、控制平滑度，以及 M4 CPU 上的单步决策延迟。训练奖励只用于诊断，不作为系统效果的主要结论。

## 参考文档

- [Webots 官方文档](https://cyberbotics.com/doc/guide/foreword)
- [Webots Gym 与 Stable-Baselines3 示例](https://github.com/cyberbotics/webots/blob/master/docs/guide/samples-howto.md)
- [Gymnasium 自定义环境教程](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Stable-Baselines3 强化学习实践建议](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

## 许可

项目许可证将在首次公开 GitHub 仓库前确定。
