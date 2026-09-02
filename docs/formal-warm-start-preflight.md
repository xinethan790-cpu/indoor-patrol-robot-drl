# Webots 正式补训 warm-start 预检

正式 W0/W1/W2 课程在解析 Webots 可执行文件、创建训练 manifest 或启动进程前，必须先通过
warm-start 证据预检。入口为：

```bash
.venv/bin/python scripts/preflight_webots_warm_start.py
```

预检同时核验：

- `configs/train/webots_curriculum.yaml` 指向的热启动模型路径；
- requirement manifest 声明的模型 SHA-256、累计步数和 validation 指标；
- `configs/models/registry.yaml` 中同一候选模型的路径、哈希、步数、指标和选中角色；
- 原始 `episodes.csv`、`summary.json` 的文件哈希，以及从 CSV 重新计算的回合数、成功、碰撞、
  超时、错误和条件/seed 网格；
- 所有证据文件均为仓库内普通文件，路径不越界且任何路径分量都不是符号链接。

当前 requirement 固定记录 2026-08-30 报告所引用的 stage-2 重训模型和 validation 原始
证据。若本机只有旧模型或缺少原始 CSV，预检应返回 `NOT READY`。不得通过修改报告数值、
降低期望哈希、跳过预检或启动 held-out test 来消除错误；应从原训练机或可信归档恢复精确
匹配的模型和验证证据。恢复时先保留当前旧文件，不要直接覆盖后再追溯。

该预检只检查训练输入完整性，不启动 Webots、不训练模型、不访问 `10000～10019` held-out
test，也不创建正式 test ledger。
