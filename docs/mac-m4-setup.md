# Mac mini M4 环境说明

## 已检查环境

- 架构：`arm64`；
- 操作系统：macOS 26.3；
- 系统默认 Python：3.14，不作为项目环境；
- 项目 Python：原生 ARM64 Python 3.11；
- Webots：R2025a universal binary，包含原生 `arm64`；
- Git：已安装。

Webots 当前官方稳定文档未明确覆盖本机的 macOS 26.3，因此第一个里程碑必须包含真实世界、Python Controller、Supervisor 和 fast 模式的冒烟测试。不能只根据“安装成功”判断完全兼容。

## Python 环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
python scripts/configure_webots_python.py
python -c "import platform; print(platform.machine())"
```

输出应为 `arm64`。

PPO 使用低维状态和小型 MLP，默认 `device="cpu"`。PyTorch MPS 可以作为后期测速项，但不是项目依赖，也不能把 MPS 与 CPU 结果混在同一性能表中。

## Webots 检查

```bash
file /Applications/Webots.app/Contents/MacOS/webots
```

结果应包含 `arm64`。在 Finder 中打开 Webots 的“显示简介”，确认没有勾选“使用 Rosetta 打开”。

已完成冒烟测试清单：

1. 启动原生 Webots R2025a；
2. 启动项目 `.venv` 中的 Python 3.11 Controller；
3. Supervisor 完成机器人、目标和动态障碍的回合重置；
4. TurtleBot3 差速轮、GPS、IMU 与 LDS-01 正常采样；
5. `fast`、`no-rendering` 模式完成 256 步 PPO 训练并保存模型；
6. 退出后无残留控制器进程。

## 不采用的主线依赖

- Isaac Sim/Isaac Lab：依赖 NVIDIA GPU，不适合 Apple Silicon 本地环境；
- ROS 2：macOS 缺少与 Ubuntu 同等方便的官方二进制链路；
- Gazebo：macOS 图形与依赖风险高于 Webots；
- 视觉端到端训练：算力和采样成本高，偏离本项目的低维动态避障目标。
