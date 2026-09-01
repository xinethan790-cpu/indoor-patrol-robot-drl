# 四点巡检应用验证记录

本文记录 Webots 四点巡检应用的真实运行结果。这里的应用场景包含 A* 路线、
连续多航点任务、动态障碍、巡检驻留和环境采样，因此不属于 R2a 单障碍组件的
既有验证域。文中的结果是应用开发证据，不是生产安全认证或正式模型晋级结论。

## 高速双障碍诊断基线

| 日期 | 场景 seed | 结果 | 完成巡检点 | 物理接触 | LiDAR 临界接近 |
| --- | ---: | --- | ---: | ---: | ---: |
| 2026-08-31 | 7000 | 安全失败 | 0/4 | 0 | 1 |

本次场景包含两个动态障碍，速度分别为 `0.31 m/s` 和 `0.27 m/s`。机器人在
第 66 个控制步、仿真时间 `8.448 s` 处因全局 LiDAR 最小距离
`0.127633 m` 触发保护终止，Webots 未报告物理接触。

复现后新增的现场审计字段确认：最近波束为 policy-order 第 107 束，即机器人
左后方约 `107°`；此时纵向动态障碍位于 `(-0.92, -0.1392)`，机器人位于
`(-0.950222, -0.456909)`。R2a front-brake 组件覆盖前方 `±60°` 和侧方至
`±105°`，且其证据绑定配置不执行侧向转向，因此最近威胁落在组件干预范围之外。
与此同时，组件正在针对前方约 `0.358 m` 的中央柜体减速，机器人实际线速度仅
约 `0.030 m/s`，无法摆脱以 `0.31 m/s` 从侧后方逼近的障碍。

该失败不通过放宽距离阈值隐藏。后续验证分为两条线路：

1. 默认运行场景与 R2a 的速度运行域对齐，并保留高速双障碍场景作为压力测试；
2. 为四点巡检单独实现全向临界接近分类、物理步级安全监测和侧后方避让，再做
   patrol-specific 多 seed 验证，且不沿用 R2a 组件的资格标签。

原始 `trajectory.csv`、`collisions.csv`、`summary.json` 和环境巡检证据包保存在
本机 `artifacts/results/webots-patrol-seed7000-*` 目录；该目录按仓库规则不提交
大体积运行产物。本记录保留能够复核根因的固定指标，后续通过/失败结果都会追加。

## Guard v2 五种子开发验收

2026-09-01 使用固定 operational 场景、固定 R2a 模型与安全盾，以及
`patrol-ttc-escape-guard-v2`，顺序运行 `7000～7004` 五个场景 seed：

```bash
python scripts/run_webots_patrol_campaign.py \
  --scenario operational \
  --seed-start 7000 \
  --episodes 5 \
  --output-dir artifacts/results/patrol-operational-pilot-guard-v2-seeds7000-7004
```

| seed | 结果 | 巡检点 | 碰撞 | 原始临界接近 | Guard 激活/恢复/超时 | 控制步 | 路径（m） |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7000 | 完成 | 4/4 | 0 | 0 | 4/4/0 | 651 | 11.743 |
| 7001 | 完成 | 4/4 | 0 | 0 | 5/5/0 | 587 | 11.097 |
| 7002 | 完成 | 4/4 | 0 | 0 | 14/14/0 | 716 | 12.866 |
| 7003 | 完成 | 4/4 | 0 | 1 | 1/1/0 | 642 | 11.754 |
| 7004 | 完成 | 4/4 | 0 | 0 | 1/1/0 | 603 | 11.067 |

汇总结果为 `5/5` 完成、`0` 碰撞、`0` safety stop、`0` 超时、`0` 执行错误；
平均仿真用时 `86.5856 s`，平均路径 `11.7055 m`。seed 7003 的一条
`lidar_threshold` 记录表示原始 LiDAR 临界区间被审计并成功恢复，不是物理接触；
因此该回合仍为完成结果，但全局最小净空按保守口径记为 `0`。

批量器正确给出 `acceptance_status=insufficient_evidence`：五回合只用于开发期
回归，未达到应用门禁规定的最少 `20` 回合。不得把本节写成正式泛化通过结论。
固定汇总证据为：

- `episodes.csv` SHA-256：`ddcbb43ef04c12cff304b9a9ac836ee648bffc82b4e8672accc3f601f8634441`
- `summary.json` SHA-256：`33f8b3b23ffff29724deb6f95833d86fcb0397bd9eb54ce116e4e3fd476abe87`
- 模型 SHA-256：`051d1e2f07bb147d247c91ded821e58af5b87987bedc2cee4e287fdbdfcdcbe0`
- 部署档案 SHA-256：`f0d5bf258057589ac5f2da9fcd7e211c4b7a5639d147b61588bd6572ee579383`

原始逐步结果保存在本机对应 `artifacts/results/` 目录；仓库提交本记录和复现入口，
避免把大体积、机器生成的逐帧与逐步文件直接纳入 Git 历史。

五个回合的环境巡检证据包均通过 `scripts/verify_inspection_bundle.py` 校验。

## Guard v2 二十种子应用验收

五种子回归通过后，在 Git revision
`6512236566052d957c8bae9197e6cf1689c49a5a` 上冻结代码，并一次性运行此前未使用的
`7100～7119` 二十个 operational 场景 seed：

```bash
python scripts/run_webots_patrol_campaign.py \
  --scenario operational \
  --seed-start 7100 \
  --episodes 20 \
  --output-dir artifacts/results/patrol-operational-application-acceptance-seeds7100-7119-v1 \
  --port 1262 \
  --episode-wall-timeout-s 600
```

| 指标 | 结果 | 门禁 |
| --- | ---: | ---: |
| 回合数 | 20 | 至少 20 |
| 完成 | 19（95%） | 至少 75% |
| 碰撞 | 1（5%） | 不高于 15% |
| safety stop | 0（0%） | 记录但不单独设阈值 |
| 超时 | 0（0%） | 记录但不单独设阈值 |
| 执行错误 | 0（0%） | 必须为 0% |
| 平均仿真用时 | 85.9664 s | 描述性指标 |
| 平均路径 | 11.5452 m | 描述性指标 |

批量器给出 `acceptance_status=pass`，四项门禁检查全部为真，
`acceptance_failures=[]`。20 个环境巡检证据包也全部通过独立完整性校验。

唯一失败为 seed 7116：机器人在前往 P1 的第 194 个控制步、仿真时间
`24.8 s`、位置 `(-0.782393, -0.727599)` 发生 Webots 物理接触；最近 LiDAR
读数为 `0.123389 m`，动态障碍 0 位于 `(-0.92, -0.50752)`。该回合以
`collision` 原样计入汇总，没有重跑或排除。碰撞前 Guard 已激活，最终执行
`rotate`，进一步根因以同一回合的 `trajectory.csv`、`collisions.csv` 和
`ttc_escape_guard_events.csv` 为准。

固定证据与归档标识：

- `episodes.csv` SHA-256：`7bebd141017ea2616136159cbe4c67393496c3b6de12c0dd6a81931381716e29`
- `summary.json` SHA-256：`ed4e7826331389b8de4e9582acacca2966949a71d3fb9863cea2627adf2c3f27`
- 完整证据压缩包 SHA-256：`5981668c4874372f7ccf5382ab5159b4efd667850f942882f9a4b7069e90d745`
- 模型 SHA-256：`051d1e2f07bb147d247c91ded821e58af5b87987bedc2cee4e287fdbdfcdcbe0`
- 部署档案 SHA-256：`f0d5bf258057589ac5f2da9fcd7e211c4b7a5639d147b61588bd6572ee579383`
- 场景配置 SHA-256：`1a110da80f26bf944d07362d201da591f2e86bbb9569bda4d1df6b7ea256ca57`

本节结论是固定 operational 四点巡检场景的**应用性能门禁通过**。运行摘要仍明确
记录 `patrol_scenario_qualified=false`；它不是形式化安全认证，也不替代三训练 seed、
多算法、多工况的受保护论文正式测试。
