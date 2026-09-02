# 训练、模型选择与复现实验协议

本文档规定从运动学预训练到 Webots 正式评测的唯一推荐流程。它的目的不是保证某个方法获胜，而是保证每次结果都能回答三个问题：使用了什么模型、在哪组场景上选择、是否提前看过最终测试集。

## 1. 两类后端不能混写

| 后端 | 用途 | 可以支持的结论 |
|---|---|---|
| 纯 Python 运动学环境 | 快速训练、接口调试、候选模型筛选、算法链路检查 | 运动学条件下的初步比较、Webots 权重热启动选择 |
| Webots R2025a | 差速轮动力学、真实 LiDAR 数据流、碰撞与最终系统评测 | 论文和答辩中的正式仿真结论 |

运动学模型与 Webots 模型虽然共享 79 维观测和二维动作语义，但两者的速度上限、控制响应、机器人尺寸和传感器误差不同。因此运动学结果不能改名为“Webots 结果”，运动学模型也只能作为 Webots PPO 的初始化权重。

## 2. 数据划分与使用规则

| 划分 | seed | 用途 | 使用次数 |
|---|---|---|---|
| 训练 | `11、22、33` | PPO 参数更新和独立重复训练 | 可重复 |
| 验证 | `5000～5019` | 阶段模型选择、超参数调整、失败分析 | 可重复，但必须记录改动 |
| 最终测试 | `10000～10019` | 配置、权重和阈值全部锁定后的最终报告 | 原则上只运行一次 |

运动学评测脚本默认使用验证集。访问运动学测试集必须同时传入 `--split test --confirm-test`。正式 Webots held-out test 不应直接调用评测 Controller；唯一推荐入口是 `scripts/run_webots_formal_test.py`，由它注入 `WEBOTS_EVAL_SPLIT=test` 和 `WEBOTS_CONFIRM_TEST=1`，并同时校验 final lock 与一次性 ledger。ledger 的消费键只由受保护测试协议与数据集定义，和 campaign 名称无关；改 campaign 不能获得第二次机会。不要根据最终测试结果重新选模型；若开始执行后发现实现错误，全局 ledger 必须保留为 `invalidated`，不能删除后重跑，也不能把失败尝试伪装成未访问测试集。

## 3. 环境准备与基础校验

在仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
python scripts/configure_webots_python.py
ruff check .
pytest
```

当前 Windows 正式实验环境使用 Python 3.11.6。为避免三次训练之间依赖版本漂移，应优先按仓库中的冻结清单重建环境，而不是在每个 seed 前重新解析宽松版本范围：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-formal-lock.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts\configure_webots_python.py
```

`requirements-formal-lock.txt` 锁定本次 Windows CPU 训练所用的 PyTorch、Stable-Baselines3、Gymnasium、NumPy、测试工具及其传递依赖；它不替代模型、配置、Git revision 和 Webots 版本的单独哈希记录。Apple Silicon 环境仍按 `docs/mac-m4-setup.md` 建立原生依赖，不能直接套用 Windows 的 `torch+cpu` wheel。

`configure_webots_python.py` 会生成各 Controller 的本机 `runtime.ini`。它包含绝对路径，不能提交到 Git；仓库移动、重新克隆或重建虚拟环境后必须再次生成。

每次正式训练前至少记录：Git commit、训练 seed、Python 和 Webots 版本、PPO 配置、课程配置、模型文件 SHA-256。Webots Controller 会在模型目录中生成 `run_metadata*.yaml`，其中包含热启动来源及其 SHA-256；正式论文数据还应保留 TensorBoard 日志和逐回合 CSV。

## 4. 阶段 A：运动学课程预训练

当前课程由 `configs/train/curriculum.yaml` 定义：

| 阶段 | 请求步数 | 动态障碍数量 | 障碍速度范围 | 目的 |
|---|---:|---:|---:|---|
| `goal_following` | 100,000 | 0 | 0 | 先学习目标跟随 |
| `sparse_dynamic` | 150,000 | 2 | 0.10～0.35 m/s | 学习基础避让 |
| `nominal_dynamic` | 300,000 | 5 | 0.15～0.55 m/s | 覆盖标准动态场景 |
| `dense_fast` | 250,000 | 7 | 0.20～0.65 m/s | 增加密度与速度压力 |

运行 seed 11：

```bash
source .venv/bin/activate
python scripts/train_curriculum.py \
  --seed 11 \
  --output artifacts/models/ppo_curriculum_seed11
```

Stable-Baselines3 以完整 rollout 更新，实际累计步数可能略大于 YAML 中的请求值。seed 11 的实际阶段终点为 `102400、253952、557056、811008` steps；论文应引用模型元数据中的实际步数，而不是简单写成 800k。

若在完成 stage 2 后中断，可从该权重继续 stage 3：

```bash
python scripts/train_curriculum.py \
  --seed 11 \
  --resume artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip \
  --start-stage 3 \
  --output artifacts/models/ppo_curriculum_seed11
```

`--stage-step-limit` 只用于流水线冒烟测试，不得把其输出当作正式候选模型。

## 5. 阶段 B：只在验证集选择热启动模型

对每个候选阶段分别运行 PPO-only 验证，例如：

```bash
python scripts/benchmark_kinematic.py \
  --split validation \
  --planners ppo \
  --ppo-model artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip \
  --output-dir artifacts/results/model_selection_stage2
```

下表保留 2026-09-02 完整性复核后的 registry 候选记录；其中只有 stage 2 的模型
和原始验证 CSV 被归档为新开发基线输入，stage 1/3 不进入新基线。此前记录的
`61/19` stage-2 结果及其模型哈希无法从仓库、本机历史或远端恢复，已撤回：

| 候选模型 | 实际累计步数 | 成功率 | 碰撞率 | 处理决定 |
|---|---:|---:|---:|---|
| stage 1 `sparse_dynamic` | 253,952 | 0.6500 | 0.3500 | 不选 |
| stage 2 `nominal_dynamic` | 557,056 | **0.6875** | **0.3125** | 选为新开发基线的可复核热启动 |
| stage 3 `dense_fast` | 811,008 | 0.6250 | 0.3750 | 不选 |

选择 stage 2 是验证集上的模型选择，不代表它已经通过 Webots，也不代表 PPO
优于 Reactive。stage 3 虽然训练更久，但验证性能退化，因此不能仅凭“最终权重”
或“训练步数更多”自动选用。新基线另外绑定该模型、原始 80 条 PPO 记录和完整
240 条统一基准；详见 `reports/webots_r2ab_targeted/rebaseline_protocol_2026-09-02.md`。

## 6. 阶段 C：统一运动学基准

用锁定的 stage 2 权重，对 Reactive、DWA、PPO 运行相同的 `G0～G3` 场景和 seed：

```bash
python scripts/benchmark_kinematic.py \
  --split validation \
  --ppo-model artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip \
  --output-dir artifacts/results/kinematic_validation_seed11
```

默认配置为 `4 条件 × 20 seeds × 3 方法 = 240 episodes`。原始逐回合数据写入 `episodes.csv`，聚合结果写入 `summary.json`。当前结果见 `reports/current-progress.md`；它们仍属于运动学验证结果。

## 7. 阶段 D：Webots 课程微调

开始前关闭其他占用同一世界的 Webots 实例，并确保代码与配置已提交。正式推荐入口会按 W0→W1→W2 执行训练和验证门禁，并把运行 manifest 写入 `artifacts/results/webots_curriculum_seed<seed>/`：

```bash
python scripts/run_webots_curriculum.py --webots "PATH_TO_WEBOTS_EXECUTABLE" --seed 11
python scripts/run_webots_curriculum.py --webots "PATH_TO_WEBOTS_EXECUTABLE" --seed 22
python scripts/run_webots_curriculum.py --webots "PATH_TO_WEBOTS_EXECUTABLE" --seed 33
```

三个 seed 必须是三次相互独立的训练，不能复制同一个模型后改文件名。运行器只会自动访问 `5000～5019` 验证 seeds，不会访问 held-out test。下面的环境变量命令保留为单阶段排错说明；正式证据应以运行器生成的 manifest、模型元数据、逐回合验证 CSV 和选择 manifest 为准。

### W0：无障碍迁移检查，50k

```bash
WEBOTS_TRAIN_SEED=11 \
WEBOTS_TRAIN_STEPS=50000 \
WEBOTS_OBSTACLE_COUNT=0 \
WEBOTS_TRANSFER_MODEL=artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip \
/Applications/Webots.app/Contents/MacOS/webots \
  --batch --mode=fast --no-rendering --stdout --stderr \
  worlds/webots_ppo_train.wbt
```

### W1：一个动态障碍，追加 150k

```bash
WEBOTS_TRAIN_SEED=11 \
WEBOTS_TRAIN_STEPS=150000 \
WEBOTS_OBSTACLE_COUNT=1 \
WEBOTS_RESUME_MODEL=artifacts/models/webots/seed_11/ppo_webots_latest.zip \
/Applications/Webots.app/Contents/MacOS/webots \
  --batch --mode=fast --no-rendering --stdout --stderr \
  worlds/webots_ppo_train.wbt
```

### W2：三个动态障碍，追加 300k

```bash
WEBOTS_TRAIN_SEED=11 \
WEBOTS_TRAIN_STEPS=300000 \
WEBOTS_OBSTACLE_COUNT=3 \
WEBOTS_RESUME_MODEL=artifacts/models/webots/seed_11/ppo_webots_latest.zip \
/Applications/Webots.app/Contents/MacOS/webots \
  --batch --mode=fast --no-rendering --stdout --stderr \
  worlds/webots_ppo_train.wbt
```

W0 首次运行通过 `policy.state_dict` 迁移策略参数，并创建新的 Webots 优化器；W1/W2 使用 `PPO.load` 恢复模型和优化器，且不重置累计时间步。每段结束后检查：

- `artifacts/models/webots/seed_11/ppo_webots_s11_step*.zip`；
- `artifacts/models/webots/seed_11/ppo_webots_latest.zip`；
- `artifacts/models/webots/seed_11/run_metadata_step*.yaml`；
- `artifacts/logs/webots/seed_11/training.monitor.csv`；
- `artifacts/logs/tensorboard/`。

seed 22 和 33 使用相同、已锁定的 stage 2 热启动权重，把命令中的 `WEBOTS_TRAIN_SEED` 和输出 seed 路径对应替换。三次 Webots 微调尚未全部完成前，不报告“三种子均值±标准差”。

## 8. Webots 验证、final lock 与一次性最终测试

每个训练阶段后可以使用验证 seeds 检查退化：

```bash
WEBOTS_MODEL_SEED=11 \
WEBOTS_EVAL_SPLIT=validation \
WEBOTS_EVAL_SEED_START=5000 \
WEBOTS_EVAL_EPISODES=20 \
WEBOTS_OBSTACLE_COUNT=3 \
/Applications/Webots.app/Contents/MacOS/webots \
  --batch --mode=fast --no-rendering --stdout --stderr \
  worlds/webots_ppo_evaluate.wbt
```

这条手工命令只适合诊断单个验证运行；它的默认结果目录名不作为 checkpoint 选择契约。正式候选的目录名、模型路径/哈希和 condition 应由 `run_webots_curriculum.py` 自动注入。

### 8.1 带 variant 的 checkpoint 选择

正式 W0/W1/W2 路线应分别为三个训练 seed 生成 W2 验证选择 manifest。下面三个命令只扫描各自 seed 的 `validation` summary；任何一个命令以状态码 2 退出，都表示没有模型通过 `success >= 0.75` 且 `collision <= 0.15` 的固定门禁，此时不得创建 final lock：

```bash
python scripts/select_webots_checkpoint.py --seed 11 --condition W2 \
  --episodes 20 --min-success-rate 0.75 --max-collision-rate 0.15
python scripts/select_webots_checkpoint.py --seed 22 --condition W2 \
  --episodes 20 --min-success-rate 0.75 --max-collision-rate 0.15
python scripts/select_webots_checkpoint.py --seed 33 --condition W2 \
  --episodes 20 --min-success-rate 0.75 --max-collision-rate 0.15
```

每份 `selection.json` 必须绑定实际模型路径和 SHA-256、训练 seed、condition、planner、完整 `5000～5019` 场景 seed、恰好 20 回合，以及原始 `episodes.csv` 所在结果目录。后续 final lock 会重新读取原始 CSV 计算成功率和碰撞率，而不是盲信汇总 JSON。

带 `model_variant` 的路线必须在自己的模型目录中选择 checkpoint，不能和 legacy 路线混在一起。以尚未完成的 `curriculum_v2` R1 探索路线为例，先把每个候选模型的验证结果写成 `webots_validation_r1_checkpoint<steps>_seed11/summary.json`（最终阶段模型使用 `final_step<steps>`），再运行：

```bash
python scripts/select_webots_checkpoint.py \
  --seed 11 \
  --condition R1 \
  --model-variant curriculum_v2 \
  --episodes 20 \
  --min-success-rate 0.75 \
  --max-collision-rate 0.25
```

`--model-variant curriculum_v2` 会把候选模型限定到 `artifacts/models/webots/seed_11/curriculum_v2/`，并把 variant 写入验证契约和选择结果。Webots 评测的 `summary.json` 必须同时记录实际评测模型的仓库相对路径、variant、训练 seed 与 SHA-256；选择器会逐项绑定到候选 checkpoint，拒绝把 base 模型的指标套到同 step 的 variant 模型。输出目录默认为 `artifacts/results/webots_validation_r1_selection_seed11_curriculum_v2/`。

这里的 R1 门禁允许碰撞率 0.25，只用于探索路线的阶段选择。即使选择文件内部状态为 `formal_selected`，它也**不满足正式巡检加载门槛**。正式巡检固定要求：验证集不少于 20 回合、成功率不低于 0.75、碰撞率不高于 0.15。若同一批验证结果满足固定门槛，应另存一份正式巡检 manifest：

```bash
python scripts/select_webots_checkpoint.py \
  --seed 11 \
  --condition R1 \
  --model-variant curriculum_v2 \
  --episodes 20 \
  --min-success-rate 0.75 \
  --max-collision-rate 0.15 \
  --output-dir artifacts/results/webots_validation_r1_patrol_selection_seed11_curriculum_v2
```

排名第一但未通过固定门槛的模型不能作为正式模型。

### 8.2 巡检加载的 fail-closed 规则

正式 A* + PPO 多点巡检应显式指向上一步生成的 JSON manifest：

```bash
WEBOTS_PATROL_SELECTION_MANIFEST=artifacts/results/webots_validation_r1_patrol_selection_seed11_curriculum_v2/selection.json \
python scripts/run_webots_patrol.py --headless
```

加载器只接受 `validation` 契约且 `selection_status=formal_selected` 的 manifest，并复核回合数、固定成功率/碰撞率门槛、候选实际指标和 `gate_passed`。随后检查模型位于 `artifacts/models/webots/`、文件存在，并重新计算 SHA-256 与 manifest 比对。任一条件不满足都会停止运行，不会自动改用其他模型。巡检输出同时记录实际加载模型的路径、SHA-256、训练域和选择原因。

开发期若确实需要测试未正式选择的模型，无论是显式 `WEBOTS_PATROL_MODEL=<path>`，还是启用“可变 Webots latest → 运动学 stage-2 热启动”的 fallback 链，都必须同时显式设置：

```bash
WEBOTS_PATROL_ALLOW_UNVALIDATED_MODEL=1 \
WEBOTS_PATROL_MODEL=artifacts/models/webots/seed_11/manual_candidate.zip \
python scripts/run_webots_patrol.py --headless
```

若不指定 `WEBOTS_PATROL_MODEL`，同一个开关才会允许 fallback 链。两种路径都会被标记为 `UNVALIDATED`，只能用于调试或演示，不能进入论文正式结果。未设置上述开关时，manifest 缺失、显式模型或校验失败都会报错退出。

课程运行 manifest 自 schema 2 起保存完整语义配置快照和 `curriculum_config_sha256`。恢复训练时，训练模型、variant、port、每阶段训练步数、world、门禁、controller 环境变量或恢复 checkpoint 任一变化都会拒绝继续，避免把不同实验接到同一运行记录中。旧 schema 1 manifest 不会被自动升级；应保留原文件作为历史证据，并为新配置启动新的运行记录，而不是手工补哈希。

### 8.3 创建三模型 final lock

三个 seed 都通过正式验证门禁后，回到训练时记录的同一个 clean Git revision。对每个 seed 准备六项证据：训练 seed、被选模型、该模型的训练元数据、课程运行 manifest、`selection.json`、与被选 summary 同目录的验证 `episodes.csv`。正式 W2 课程完成时会为该阶段的 final model 与中间 checkpoint 写入不可变 provenance sidecar，并把 sidecar 路径与哈希列入课程 manifest；创建 final lock 只复核这些训练期证据，不会事后补写 sidecar。下面是命令模板；所有 `PATH_TO_...` 占位符必须替换为仓库内的真实文件：

```bash
python scripts/create_webots_final_lock.py \
  --config configs/eval/webots_formal.yaml \
  --model-evidence 11 PATH_TO_SEED11_MODEL PATH_TO_SEED11_METADATA \
    artifacts/results/webots_curriculum_seed11/run_manifest.yaml \
    artifacts/results/webots_validation_w2_selection_seed11/selection.json \
    PATH_TO_SEED11_SELECTED_VALIDATION_EPISODES.csv \
  --model-evidence 22 PATH_TO_SEED22_MODEL PATH_TO_SEED22_METADATA \
    artifacts/results/webots_curriculum_seed22/run_manifest.yaml \
    artifacts/results/webots_validation_w2_selection_seed22/selection.json \
    PATH_TO_SEED22_SELECTED_VALIDATION_EPISODES.csv \
  --model-evidence 33 PATH_TO_SEED33_MODEL PATH_TO_SEED33_METADATA \
    artifacts/results/webots_curriculum_seed33/run_manifest.yaml \
    artifacts/results/webots_validation_w2_selection_seed33/selection.json \
    PATH_TO_SEED33_SELECTED_VALIDATION_EPISODES.csv
```

默认 campaign id 来自 `configs/eval/webots_formal.yaml` 的 `experiment_id`，因此默认 lock 为 `artifacts/results/formal_locks/webots_formal_g0_g3_v1.json`。创建过程是 fail-closed 的：要求当前工作树完全干净；三个训练 seed 必须精确为 `11、22、33`；三份训练元数据都必须记录当前 clean HEAD；三条路线必须同为 W2、同一 model variant、同一归一化课程协议指纹和同一算法配置（只允许 seed、路径和时间字段变化）；模型、sidecar、课程 manifest、选择 manifest、验证 CSV、正式配置、world、正式 evaluator/controller 与完整 Python source bundle 都会绑定 SHA-256；lock 采用不可覆盖写入。任一验证模型没有通过固定门禁，或原始 CSV 与选择指标不一致，命令都会失败。

final lock 只锁定输入，并不访问测试集。创建后不要修改代码、配置、模型或证据文件；正式 runner 会在开始前重新计算并核对全部哈希。

### 8.4 固定 400 回合矩阵

`configs/eval/webots_formal.yaml` 固定以下 held-out 设计：

| 条件 | 布局与扰动 |
|---|---|
| `G0` | 训练布局，3 个标称速度动态障碍 |
| `G1` | 未见静态布局，3 个标称速度动态障碍 |
| `G2` | 训练布局，5 条确定性交叉通道的高密度动态障碍 |
| `G3` | 训练布局，3 个更高速动态障碍 |

每个条件运行 `Reactive、DWA、PPO-seed11、PPO-seed22、PPO-seed33` 五个 planner，每个 planner 使用完全相同的 `10000～10019` 二十个场景 seed。因此正式执行量是：

> `4 conditions × 5 planners × 20 scenario seeds = 400 episodes`

这里的三个 PPO 项是三个独立训练重复，不是同一个权重的三个别名。Reactive 和 DWA 不绑定模型，PPO 则必须分别绑定 final lock 中对应训练 seed 的模型路径与 SHA-256。控制周期、速度/加速度限制、世界文件、障碍语义和 safety shield 状态也由配置和 provenance 固定。

### 8.5 先 dry-run，不消耗 held-out test

创建 lock 后先做完整预检：

```bash
python scripts/run_webots_formal_test.py \
  --lock artifacts/results/formal_locks/webots_formal_g0_g3_v1.json \
  --webots "PATH_TO_WEBOTS_EXECUTABLE" \
  --confirm-test \
  --dry-run
```

`--dry-run` 会校验 lock、当前 clean HEAD、配置和全部模型证据，执行只读的 `webots --version` 探测并核对 R2025a、可执行文件 SHA-256 和 Windows 版本/签名元数据，然后构造并打印恰好 20 个 Webots 命令，并拒绝已有 ledger、正式输出或冲突的 staging 目录。它会读取并校验 lock 绑定的 world 文件哈希，但**不会启动 Webots 仿真、不会让模拟器打开 world，不会创建 ledger 或结果目录，也不会消耗 held-out test**。这里仍要求 `--confirm-test`，含义是允许解析受保护的测试计划，不是允许实际执行。

dry-run 通过后，应人工归档其终端输出并核对：四个 condition 是否齐全、五个 planner label 是否齐全、三个 PPO 的训练 seed/模型哈希是否对应、Webots 可执行文件和端口是否正确。不要为“看看结果”单独手工运行某一个 test seed。

### 8.6 一次性执行与 ledger 终态

只有在模型、代码、配置、路径和 dry-run 输出全部确认后，才去掉 `--dry-run`：

```bash
python scripts/run_webots_formal_test.py \
  --lock artifacts/results/formal_locks/webots_formal_g0_g3_v1.json \
  --webots "PATH_TO_WEBOTS_EXECUTABLE" \
  --confirm-test
```

正式 runner 会先完成 lock/source/world/模型证据和 Webots 可执行文件的不可变预检，再以排他方式创建 test ledger，并把这次可执行文件 runtime evidence 原子绑定到 `attempt`，随后依次执行 20 个 Webots 调用。成功时会生成：

- `artifacts/results/formal_runs/<campaign>/runs/<condition>/<planner>/episodes.csv` 与 `summary.json`；
- `artifacts/results/formal_runs/<campaign>/logs/` 下的每次调用参数、标准输出、标准错误和退出状态；
- `formal_report.json`：从 20 份原始 CSV 重新计算的严格 400 回合汇总；
- `webots_executable.json`：可执行文件路径、SHA-256、实际探测版本和 Authenticode 状态；
- `result_manifest.json`：结果文件路径、SHA-256、模型和 source revision 绑定；
- `artifacts/results/formal_ledgers/protected_test_<consumption-key>.json`：由 `attempt` 转为 `success` 的全局一次性账本。

汇总器要求恰好 20 对输入、400 个唯一 episode、每个 planner/condition 都是完整的二十个相同 test seeds，并复核 PPO 模型哈希。PPO 跨训练 seed 的汇总同时给出均值、样本标准差和基于三个训练重复的双侧 Student-t 95% 置信区间。正常情况下 runner 已自动生成报告；`scripts/summarize_webots_formal.py` 只用于从保留的 20 对 `episodes.csv + summary.json` 重建/复核报告，重建时必须用 `--lock` 传入原 final lock，使全部输入 provenance 与锁定的 world、controller、evaluator、source bundle 和模型证据重新比对；该命令不需要也不允许重新运行仿真。

一旦 ledger 建立，该受保护测试协议的数据访问机会就已全局消耗。如果任一 Webots 调用、provenance 校验或 400 回合汇总失败，runner 会保留日志并把 ledger 置为终态 `invalidated`；该 ledger 不能覆盖或转回 `attempt`。不得通过删除 ledger、改 campaign、改输出目录或重新生成 lock 来重跑相同 held-out 协议。后续只能在论文中如实报告该次 invalidated 结果；除非预先定义了不同且独立的新测试数据集，否则不能把修复后的运行视为新的正式测试机会。

截至本页更新时间，仓库只完成了上述协议、控制器和汇总器的实现，**尚未创建满足条件的三模型 final lock，也尚未运行正式 400 回合 Webots held-out test**。现有 240 回合表是运动学验证，不得替代正式 Webots 结果。

## 9. 正式结果发布检查表

- [ ] 配置文件和代码已提交，Git commit 已记录；
- [ ] 三个训练 seed 均完成，失败运行也有记录；
- [ ] seed 11、22、33 各有模型、训练元数据、验证选择 manifest 和原始验证 CSV；
- [ ] 三个模型均在 `5000～5019` 上通过成功率 0.75 / 碰撞率 0.15 固定门禁；
- [ ] 所有方法使用相同地图、障碍轨迹、控制频率和速度限制；
- [ ] 模型只根据验证集选择；
- [ ] clean HEAD 上已创建并归档 final lock 及其 SHA-256；
- [ ] `--dry-run --confirm-test` 恰好打印 20 个预期调用，且没有创建 ledger；
- [ ] 最终测试只在锁定后运行一次，ledger 终态与执行结果一致；
- [ ] 正式结果恰好包含 20 个运行、400 个唯一 episode，结果表由原始 CSV 自动生成；
- [ ] 成功率同时报告碰撞率和超时率；
- [ ] 运动学结果、Webots 结果、冒烟测试结果分别标注；
- [ ] 最终权重、配置、SHA-256 和演示视频通过 GitHub Release 发布。
