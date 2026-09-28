# Windows 开发环境说明

本项目原本在 Mac mini M4(原生 ARM64 Python 3.11)上开发,自 2026-09-28 起主开发
机为 Windows 11 笔记本(RTX 4060 Laptop)。本文记录在 Windows 上从零复现完整环境
的步骤,以及两平台的差异边界。Mac 侧流程见 [mac-m4-setup.md](mac-m4-setup.md)。

## 1. 基础环境

- Python 3.11(项目要求 `>=3.11,<3.13`;本机使用 3.11.6);
- Webots R2025a(与 Mac 相同版本,保证 world/controller 兼容);
- Git(建议 `core.autocrlf` 不再依赖:仓库 `.gitattributes` 已强制文本文件 LF)。

```bash
git clone https://github.com/xinethan790-cpu/indoor-patrol-robot-drl.git
cd indoor-patrol-robot-drl
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev,train,video]"
.venv\Scripts\python scripts\configure_webots_python.py
setx WEBOTS_HOME "C:\path\to\Webots-R2025a"
```

`configure_webots_python.py` 会把每个控制器绑定到项目 `.venv`
(`controllers/**/runtime.ini`,不入 Git);`WEBOTS_HOME` 供
`scripts/run_webots_patrol.py` 等脚本自动发现 Webots 可执行文件
(Windows/macOS 均已支持,也可用 `--webots` 显式指定)。

## 2. Webots PROTO 内容与离线缓存(关键步骤)

R2023 起,Webots 发行版的 `projects/` 只含清单骨架,PROTO/网格/纹理在首次
打开世界时从 `raw.githubusercontent.com` 按需下载并缓存。该域名在本机网络
不可达时,世界会加载失败(表现为 `Skipped unknown 'XXX' node or PROTO`)。

在本机网络下的完整修复流程:

1. 从 gitee 镜像补全安装目录的 `projects/` 内容(与官方 R2025a 标签同源):

   ```bash
   git clone --depth 1 --branch R2025a https://gitee.com/mirrors/webots.git webots-src
   cp -rn webots-src/projects/. "$WEBOTS_HOME/projects/"
   ```

2. 把相关子树资产预置进 Webots 按需缓存(缓存键 = SHA-1(asset URL)):

   ```bash
   .venv\Scripts\python scripts\seed_webots_proto_cache.py
   ```

   脚本幂等;若之前失败过,可先删除
   `%LOCALAPPDATA%\Cyberbotics\Webots\cache\assets\*` 再重跑。
3. 验证:`open -a` 换成直接双击或命令行启动
   `%WEBOTS_HOME%\msys64\mingw64\bin\webotsw.exe`,打开
   `worlds/patrol_dynamic_demo.wbt` 应无任何下载错误。

本机已用该流程修复,缓存约 980 项,后续新增世界引用新子树时需扩充
`scripts/seed_webots_proto_cache.py` 的 `SUBTREES` 列表。

## 3. 快速验证

```bash
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\run_webots_patrol.py --headless --scenario operational --seed 7000 --run-id windows-migration-smoke-v1 --output-dir artifacts\results\windows-migration-smoke-v1
.venv\Scripts\python scripts\run_patrol_dashboard.py --run-dir artifacts\results\windows-migration-smoke-v1 --port 8765
```

迁移完成时的可复核结果(`windows-migration-smoke-v1`):4/4 巡检点、0 碰撞、
627 控制步、PPO 推理延迟 P50 0.34 ms(CPU),与 Mac 端同类运行一致。

## 4. 平台差异与已知边界

- **测试套件全绿**:820 passed / 9 skipped,skipped 为平台性跳过(符号链接
  特权、SIGKILL 升级语义、下述 CRLF 写入缺口)。
- **符号链接类测试**在无管理员/开发者模式的账户上自动跳过
  (`WinError 1314`);如需覆盖,开启 Windows 开发者模式即可。
- **已知缺口(待下次重基线修复)**:`write_fixed_safety_outputs` /
  `analyze_paired_safety` 的报告写入使用文本模式,Windows 会把 LF 写成
  CRLF,使 `.sha256` sidecar 与磁盘字节失配。相关两个源文件被归档 R2A
  部署 profile 哈希绑定(`runtime_inputs.fixed_validator` /
  `paired_validator`),巡检控制器加载该 profile 做组件门禁,因此**不能**
  在不动审计链的情况下修改;修复需按 rebaseline 协议随新基线一起进行。
  该缺口只影响在 Windows 上**生成新**的 fixed-safety/paired 报告,不影响
  巡检、训练、看板与已有证据的读取校验。
- **演示视频重录**需要 ffmpeg 在 PATH 中(构建脚本已支持 PATH 发现,
  中文字体自动使用微软雅黑)。
- **训练硬件**:项目约定 PPO MLP 在 CPU 训练(Mac M4 同款设定);
  RTX 4060 8GB 对后续感知类扩展可用,当前协议无需改动。
