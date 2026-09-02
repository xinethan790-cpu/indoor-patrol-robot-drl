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
- [x] 实现温度、烟雾遮蔽率、一氧化碳的确定性模拟采样、迟滞告警与哈希证据包
- [x] 将环境巡检采样与异常告警接入 Webots 四点任务
- [x] 实现 operational/stress 场景、TTC 有界脱困和可审计多种子巡检评测
- [x] 实现 Webots 逐帧采集与 90 秒中文 1080p 汇报视频生成
- [x] 完成 7100～7119 二十种子应用验收：完成率 95%，碰撞率 5%，门禁通过
- [x] 完成本地巡检看板：地图、轨迹、任务进度、环境数据、告警与终态结果
- [x] 完成原生微信小程序演示端：离线动态回放、本机实时、云端同步代码、告警与结果追溯
- [x] 完成 Webots → Mac HTTPS/HMAC 中转 → CloudBase → 微信小程序的链路实现与离线契约验证
- [x] 完成 Webots 与云端中继的一键联动、首帧等待状态及只读巡检历史页
- [x] 完成仿真终态证据、CloudBase 安全归档与小程序完整巡检报告闭环
- [ ] 完成真实 CloudBase 资源部署、合法 HMAC 探针和微信真机端到端验收
- [ ] 完成 Webots 三 seed 正式训练、统一批量评测、泛化和奖励消融

轻量二维环境不是最终实验平台。它用于快速验证状态、动作、奖励、随机种子、训练和测试代码，并为 Webots 提供策略热启动。当前验证集选择的是 stage 2（557,056 steps），不是训练最久的 stage 3；完整选择依据见[当前实验进展](reports/current-progress.md)。正式论文效果结论仍以 Webots 批量实验为准。

正式 Webots 实验的**执行协议已经实现，但正式结果尚未产生**。固定流程为：分别用 `11、22、33` 三个训练 seed 训练和验证 PPO；在 clean Git revision 上将三个模型的训练期 sidecar、训练元数据、W0～W2 课程 manifest、验证选择 manifest 和逐回合验证 CSV 与正式配置、world、evaluator、controller 及完整 source bundle 一起绑定到不可覆盖的 FinalLock v2；先执行不消耗测试机会的 `--dry-run`；最后才对 `10000～10019` held-out seeds 运行一次 `G0～G3 ×（Reactive、DWA、PPO-seed11、PPO-seed22、PPO-seed33）= 20` 个 Webots 调用、共 400 回合。当前仓库没有可引用为论文结论的这 400 回合正式 Webots 输出，详见[训练复现协议](docs/training-protocol.md)和[当前实验进展](reports/current-progress.md)。

创建 final lock 时，每个 seed 的 `--model-evidence` 必须依次提供 `SEED MODEL METADATA CURRICULUM SELECTION VALIDATION_EPISODES` 六个值。训练期 `.formal-provenance.json` sidecar 不是第七个命令行参数；它必须已由 W2 训练生成并被课程 manifest 登记，create-lock 只做定位和复核，不会事后补写。

## 可审计模型选择与安全过滤

- Webots 评测汇总会绑定实际模型路径、variant 与 SHA-256；checkpoint 只能在 `validation` 数据上通过门禁后写入选择 manifest。正式巡检还固定要求至少 20 回合、成功率不低于 0.75、碰撞率不高于 0.15，并在加载时复核 manifest 指标和模型哈希。可变的 `latest`、运动学热启动和显式模型都不会被静默当成正式模型，探索运行必须设置 `WEBOTS_PATROL_ALLOW_UNVALIDATED_MODEL=1`。完整命令见[训练复现协议](docs/training-protocol.md)。
- Webots 环境已接入可选的 LiDAR safety shield，记录策略动作、过滤后动作、实际执行动作和介入指标。它是默认关闭的应用层启发式过滤，不是形式化安全保证；当前也尚无开启/关闭的受控消融，因此不能宣称它已经改善正式指标。

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

## 环境巡检与异常告警

应用层使用纯软件模拟传感器，不额外增加硬件成本。机器人到达巡检点后可采集温度、烟雾遮蔽率和一氧化碳浓度；异常源按空间距离衰减，测量噪声由场景 seed 和样本编号确定，因此同一实验可以精确复放。告警同时要求连续超阈值样本，并使用独立恢复阈值抑制临界值附近的反复报警。

场景配置位于 `configs/inspection/environmental_patrol_v1.yaml`。每次巡检可生成读数 CSV、告警事件 JSONL、巡检点汇总、配置快照和 SHA-256 manifest；可用以下命令检查文件是否被修改：

```bash
python scripts/verify_inspection_bundle.py artifacts/results/<巡检结果目录>
```

该证据包提供完整性校验和确定性复放，不把普通文件哈希描述成数字签名或身份认证。环境巡检现已接入 Webots 四点控制器：机器人在每个巡检点驻留并采样，产生读数、告警、巡检点进度和异常证据；本地看板会将这些文件汇总成二维轨迹、任务进度、环境读数、活动告警、异常历史和终态指标。

四点任务另外提供 `operational` 与 `stress` 两套动态障碍配置，并记录每一步 LiDAR 距离、稳健 TTC、PPO 动作、安全过滤动作、最终轮速与有界脱困事件。TTC 脱困只属于当前巡检应用的探索性保护层，不能把已归档的 R2a 单障碍记录扩展成巡检系统资格，也不构成形式化安全保证。

## 本地巡检看板

看板只监听本机 `127.0.0.1`，并要求明确指定一个运行结果目录，不会自动读取其他项目文件。以下命令可直接展示毕业汇报用的成功运行：

```bash
python scripts/run_patrol_dashboard.py \
  --run-dir artifacts/results/graduation-demo-seed7000-v1 \
  --port 8765
```

随后访问 `http://127.0.0.1:8765`。页面每 0.75 秒刷新一次；任务运行中显示最新快照和快照差分告警，任务结束后改用 `summary.json`、`inspection_summary.json` 与 `alarm_events.jsonl` 的权威终态记录。二维地图来自当前仿真配置，仅用于毕设演示，不代表真实场地测绘或形式化安全边界。

## 微信小程序演示端

`wechat-miniprogram/` 是可直接导入微信开发者工具的原生小程序工程，包含巡检总览、Canvas 二维地图、动态障碍物与速度方向、PPO 原始动作到安全过滤再到最终执行的避障决策、P1～P4 任务进度、三类环境传感器、活动告警、异常时间线、终态指标、只读巡检历史、完整巡检报告与可审计运行身份。完整报告把每个巡检点的最终温度、烟雾和一氧化碳读数、告警状态、最终轨迹及关键避障指标关联到同一运行编号；旧版、运行中断或证据被截断的数据不会被标成“完整报告”。

小程序提供三种明确区分的数据模式：

- **云端实时同步（默认）**：Mac 只通过出站 HTTPS 上报经过现有证据仓库校验的仿真快照；CloudBase 验证 HMAC、时间窗、nonce、运行编号和单调序号，再把只读快照推送到微信真机。小程序不能写数据库，也不能控制机器人。
- **仿真演示回放**：将 `graduation-demo-seed7000-v1` 的 823 条状态压缩为 90 帧，在约 15 秒内复现四点巡检、动态障碍、避障决策、36 条采样以及 3 次告警触发和 3 次解除；不依赖网络，适合答辩和断网兜底。
- **电脑本地实时**：微信开发者工具通过 `http://127.0.0.1:8765/api/dashboard` 读取本机看板数据，带协议版本、运行编号、序号、超时、退避重试与终态停止保护。

导入前可运行小程序数据层测试：

```bash
cd wechat-miniprogram
npm test
```

详细导入、启动和验收步骤见[微信小程序说明](wechat-miniprogram/README.md)，云函数、数据库权限与 HTTPS 上报部署见[CloudBase 中转说明](cloudfunctions/README.md)。真机中的 `127.0.0.1` 指向手机自身，因此真机实时模式只走 CloudBase，不会暴露 Mac 端口。

实时链路固定绑定云开发环境 `cloud1-5gpkygv2be4de05a`。上传密钥只存在于 Mac 进程和云函数环境变量中，不进入小程序或 Git。当前仓库只准备并验证了部署代码和配置清单；创建云资源、设置密钥、开放 HTTP 路由以及真机验收必须在部署时显式执行，验收记录模板见[CloudBase 真实链路验收](reports/cloudbase-live-acceptance.md)。

云端资源就绪后，可用一条命令启动同一 `run_id` 的 Webots 巡检与只读上传中继：

```bash
PATROL_RELAY_UPLOAD_SECRET="$(security find-generic-password \
  -a indoor-patrol-relay -s PATROL_RELAY_UPLOAD_SECRET -w)" \
python scripts/run_patrol_with_relay.py \
  --endpoint https://YOUR_CLOUDBASE_DOMAIN/api/patrol/ingest \
  --scenario operational \
  --seed 7000
```

组合入口会在启动前检查配置与密钥，让 relay 有界等待 Webots 首帧，并在巡检结束后等待云端终态 ACK；上传链路故障不会改写机器人的控制动作，但会让组合命令以非零状态明确报告失败。加上 `--headless` 可用于无界面快速运行。

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
python -m pip install -e ".[dev,train,video]"
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

`artifacts/` 用于本地模型、日志、结果和视频，默认不提交大文件到 Git。最终模型和演示视频通过 GitHub Release 发布。

## 毕设汇报演示视频

[下载 90 秒中文演示视频（私有仓库，需登录 GitHub）](https://github.com/xinethan790-cpu/indoor-patrol-robot-drl/releases/tag/graduation-demo-v1)

演示素材来自 `graduation-demo-seed7000-v1` 的一次真实 Webots 运行：完成 `4/4` 个巡检点、碰撞 `0` 次、行驶 `11.74 m`，采集 `36` 条环境传感器样本并触发 `3` 次异常告警。视频为 1920×1080、H.264、无音频，包含题目页、系统架构、四点巡检、动态字幕、告警过程和结尾指标卡。该结果用于应用层功能展示，不替代正式多随机种子泛化结论。

重新录制并生成视频：

```bash
python scripts/run_webots_patrol.py \
  --seed 7000 \
  --run-id graduation-demo-seed7000-v1 \
  --output-dir artifacts/results/graduation-demo-seed7000-v1 \
  --capture-frames-dir artifacts/results/graduation-demo-seed7000-v1/frames \
  --capture-fps 10

python scripts/build_defense_demo_video.py \
  --frames-dir artifacts/results/graduation-demo-seed7000-v1/frames \
  --summary artifacts/results/graduation-demo-seed7000-v1/summary.json \
  --inspection-summary artifacts/results/graduation-demo-seed7000-v1/inspection_summary.json \
  --output artifacts/videos/indoor-patrol-robot-defense-demo-v1.mp4
```

生成器会同时写出 `.manifest.json`，记录视频参数、输入文件哈希和输出 SHA-256，便于确认演示画面与运行结果来自同一 `run_id`。

运行完整运动学课程与统一验证基准：

```bash
python scripts/train_curriculum.py --seed 11 \
  --output artifacts/models/ppo_curriculum_seed11
python scripts/benchmark_kinematic.py --split validation \
  --ppo-model artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip \
  --output-dir artifacts/results/kinematic_validation_seed11
```

当前可复核的 240 回合运动学验证成功率/碰撞率为：Reactive `0.725/0.275`、DWA `0.575/0.375`、PPO stage 2 `0.6875/0.3125`。Reactive 的总体成功率与碰撞率最好，PPO 的成功回合平均用时和路径长度比更低；这组结果只用于开发期比较，不能包装成最终 Webots 结论。2026-09-02 的证据复核与新 R2a/R2b 基线见[重建协议](reports/webots_r2ab_targeted/rebaseline_protocol_2026-09-02.md)，Webots 正式训练、恢复、验证和最终测试门禁命令见[训练复现协议](docs/training-protocol.md)。

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

运行一次带环境采样、PPO 避障和 TTC 有界脱困的四点巡检：

```bash
python scripts/run_webots_patrol.py --headless --scenario operational --seed 7000
```

运行 5 个种子的开发期试验：

```bash
python scripts/run_webots_patrol_campaign.py \
  --scenario operational \
  --seed-start 7000 \
  --episodes 5 \
  --output-dir artifacts/results/patrol-operational-pilot-seeds7000-7004
```

批量器会逐回合校验模型、场景、控制器、轨迹和事件文件的一致性。少于 20 回合只标记为 `insufficient_evidence`，用于调试而不能宣称通过应用验收；正式应用门禁至少需要 20 个新种子，完成率不低于 `0.75`、碰撞率不高于 `0.15` 且无执行错误。巡检应用门禁与受保护的论文正式测试协议相互独立。

Guard v2 已在 `7000～7004` 五个开发 seed 上取得 `5/5` 完成、`0` 碰撞、`0` safety stop 和 `0` 超时；该结果仍按协议标记为 `insufficient_evidence`。逐回合数据、固定哈希和边界说明见[四点巡检应用验证记录](reports/patrol-application-validation.md)。

随后冻结代码并一次性运行 `7100～7119` 二十个新 seed，取得 `19/20` 完成、`1/20` 碰撞、无 safety stop、无超时和无执行错误，应用门禁结果为 `pass`。该结论仅覆盖固定 operational 四点应用场景，不是形式化安全认证，也不替代受保护的论文正式测试。

[下载二十种子完整应用验收证据（私有仓库，需登录 GitHub）](https://github.com/xinethan790-cpu/indoor-patrol-robot-drl/releases/tag/patrol-application-acceptance-v1)

## 目录结构

```text
configs/       环境、训练和评测配置
controllers/   Webots 控制器、训练与评测入口
docs/          架构、强化学习和实验协议
protos/        Webots 自定义机器人与障碍模型
scripts/       训练、评测和冒烟脚本
src/           可测试的 Python 核心包
tests/         单元测试与接口测试
wechat-miniprogram/ 原生微信小程序演示端与数据层测试
cloudfunctions/ CloudBase 验签中转、只读快照函数、权限清单与测试
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
