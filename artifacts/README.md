# 本地实验产物目录

`artifacts/` 保存可重新生成的大文件和运行结果，默认不提交到 Git。仓库只保留各子目录的 `.gitkeep` 和本说明。

```text
artifacts/
├── logs/
│   ├── tensorboard/       训练曲线
│   └── webots/seed_*/     Monitor 逐回合日志
├── models/
│   ├── *.zip              运动学 PPO 权重与 checkpoint
│   └── webots/seed_*/     Webots 权重、checkpoint、run_metadata
├── results/
│   └── <experiment>/      episodes.csv 与 summary.json
└── videos/                答辩演示和失败案例视频
```

正式模型不直接提交到普通 Git 历史。发布时使用 GitHub Release，并同时提供：

- 模型文件和 SHA-256；
- 对应 Git commit；
- 训练与评测配置；
- Python、依赖和 Webots 版本；
- 原始 CSV、聚合 JSON 和复现实验命令。

当前被选为 Webots 热启动的运动学模型登记在 `configs/models/registry.yaml`。该登记不代表已有 Webots 最终模型。
