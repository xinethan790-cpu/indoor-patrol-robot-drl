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

运动学评测脚本默认使用验证集。访问最终测试集必须同时传入 `--split test --confirm-test`。Webots 最终测试同样要求 `WEBOTS_EVAL_SPLIT=test` 和 `WEBOTS_CONFIRM_TEST=1`。不要根据最终测试结果重新选模型；若测试后发现实现错误，应修复后把旧结果标为无效，并完整记录原因。

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

当前 seed 11 的验证结果为：

| 候选模型 | 实际累计步数 | 成功率 | 碰撞率 | 处理决定 |
|---|---:|---:|---:|---|
| stage 1 `sparse_dynamic` | 253,952 | 0.6500 | 0.3500 | 不选 |
| stage 2 `nominal_dynamic` | 557,056 | 0.6875 | 0.3125 | 选为 Webots 热启动 |
| stage 3 `dense_fast` | 811,008 | 0.6250 | 0.3750 | 不选 |

选择 stage 2 是验证集上的模型选择，不代表它已经通过 Webots。stage 3 虽然训练更久，但验证性能退化，因此不能仅凭“最终权重”或“训练步数更多”自动选用。

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

开始前关闭其他占用同一世界的 Webots 实例。下面命令均在仓库根目录执行，使用无渲染 fast 模式。

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

## 8. Webots 验证与最终测试

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

只有当模型、场景矩阵和统计代码全部锁定后才运行最终测试：

```bash
WEBOTS_MODEL_SEED=11 \
WEBOTS_EVAL_SPLIT=test \
WEBOTS_EVAL_SEED_START=10000 \
WEBOTS_EVAL_EPISODES=20 \
WEBOTS_OBSTACLE_COUNT=3 \
WEBOTS_CONFIRM_TEST=1 \
/Applications/Webots.app/Contents/MacOS/webots \
  --batch --mode=fast --no-rendering --stdout --stderr \
  worlds/webots_ppo_evaluate.wbt
```

当前 Webots 评测 Controller 记录的是 `G0` 点到点动态避障条件。最终论文的 `G0～G3`、Reactive/DWA/PPO 公平比较和完整多航点巡检需要使用同一 Webots 场景协议后再汇总；不能用现有运动学 240 回合表代替。

## 9. 正式结果发布检查表

- [ ] 配置文件和代码已提交，Git commit 已记录；
- [ ] 三个训练 seed 均完成，失败运行也有记录；
- [ ] 所有方法使用相同地图、障碍轨迹、控制频率和速度限制；
- [ ] 模型只根据验证集选择；
- [ ] 最终测试只在锁定后运行；
- [ ] 结果表由原始 CSV 自动生成；
- [ ] 成功率同时报告碰撞率和超时率；
- [ ] 运动学结果、Webots 结果、冒烟测试结果分别标注；
- [ ] 最终权重、配置、SHA-256 和演示视频通过 GitHub Release 发布。
