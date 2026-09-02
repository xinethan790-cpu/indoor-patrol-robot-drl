# CloudBase 仿真实时中转后端

本目录把 Mac 上的 Webots 仿真状态安全地中转到云开发环境
`cloud1-5gpkygv2be4de05a`。Mac 只发起出站 HTTPS，不开放本机端口；小程序只能读取
云端快照，不能写入或控制机器人。

## 数据链路

```text
Webots → Mac relay → HTTPS/HMAC → ingestPatrol
       → patrol_live/main → 小程序 watch
       → patrol_runs      → getPatrolHistory → 小程序历史页
                          └→ getPatrolReport  → 完整巡检报告
timer → cleanupPatrolNonces → 分批删除过期 nonce
```

- `ingestPatrol`：HTTP 访问服务入口。验证原始请求体、HMAC、时间窗、nonce、协议和
  单调序号，然后在事务中写入快照。
- `getPatrolSnapshot`：小程序云函数调用的只读兜底接口；直接 `watch` 失败时可获取同一份
  快照，并返回服务端计算的陈旧状态。
- `getPatrolHistory`：登录用户调用的只读分页接口；固定按已配置设备读取轻量任务摘要，
  不返回原始请求体、nonce、请求哈希、上传密钥或终态负载。
- `getPatrolReport`：登录用户按严格 `run_id` 读取固定设备的完整巡检报告。仅普通已验证终态且
  具备完整点位档案的新记录可读；活动任务、异常中止和旧记录统一返回脱敏的不可用错误。
- `cleanupPatrolNonces`：仅由云端每分钟定时器触发的维护函数；每次最多删除 100 条过期 nonce，
  可安全重复执行，客户端不可调用。
- `shared/`：五个函数共用的协议、鉴权、状态机、数据库事务和清理实现。
- `.dist/`：`npm run build` 生成的 CloudBase CLI 自包含部署目录，不纳入 Git。
- `../wechat-miniprogram/cloudfunctions/`：同一次构建生成的微信开发者工具部署镜像，也不纳入 Git。

## 固定协议

Mac 必须以 `POST` 上传未经改写的 `patrol-dashboard/v1` JSON 文本，并携带：

```text
X-Patrol-Signature-Version: patrol-hmac-sha256-v1
X-Patrol-Device-Id: macmini-m4
X-Patrol-Timestamp: <Unix epoch seconds>
X-Patrol-Nonce: <16-128 character random token>
X-Patrol-Body-SHA256: <lowercase SHA-256 hex of the exact body bytes>
X-Patrol-Signature: <lowercase HMAC-SHA256 hex>
```

签名原文严格为：

```text
device_id + "\n" + timestamp + "\n" + nonce + "\n" + body_sha256
```

HMAC 密钥只允许从云函数和 Mac 的 `PATROL_RELAY_UPLOAD_SECRET` 环境变量读取。不得放入
小程序、配置文件、命令行参数、Git 提交或日志。密钥至少 32 字节；建议由系统安全随机源
生成 32 字节并保存到 macOS 钥匙串，再注入 relay 进程。运行时会拒绝空值、少于 32 字节
的值、首尾包含空白的值、`{{...}}`/`${...}` 模板表达式，以及包含 `change-me`、
`example`、`default`、`password` 等标记或字符多样性过低的示例密钥。云函数与 Python
relay 使用相同规则，并且错误和预检输出都不会包含密钥本身。

成功响应始终包含 sidecar 可核验的 ACK：

```json
{
  "accepted": true,
  "device_id": "macmini-m4",
  "run_id": "example-run",
  "sequence": 42,
  "terminal": false,
  "disposition": "advanced"
}
```

精确重试会返回 `accepted: true` 和 `disposition: "idempotent"`，但不会刷新快照时间。
同 nonce 重放、序号回退、同序号内容冲突、终态后追加、运行中切换 run ID、历史 run ID
复用均返回 `409`。

云端还会校验运行生命周期，而不是只相信 `terminal` 布尔值：非终态只能处于
`navigating/dwelling/sampling + running`，且不能携带终态结果；完整终态必须是
`completed + success` 或 `failed + 明确失败结果`，并同时具有匹配的验证范围、巡检点计数和
由计数推导出的进度比例。受控的运行时异常中止仍使用独立的最小证据分支。任何相互矛盾的
状态都会在写入终态锁之前失败关闭。

`patrol-dashboard/v1` 的可选 `inspection` 字段在云端会严格校验。旧 v1 缺少该字段时只会
归一化为 `available: false, sample_count: null, visit_count: 0, visits: []`，不会伪造点位读数，
也不会生成可点击的完整报告。新终态会把已验证负载投影为独立
`patrol-report/v1`：仅包含来源说明、安全运行身份、任务与结果公开指标、点位读数、
正序告警历史和最终地图/轨迹。接口不从 `terminal_payload` 临时透传数据。

## 云数据库

先按 [database/collections.json](database/collections.json) 创建四个集合并配置索引：

- `patrol_live`：固定文档 `main`，登录的小程序用户可读、客户端不可写。
- `patrol_relay_state`：服务端私有的设备单调状态。
- `patrol_relay_nonces`：服务端私有的防重放记录；按 `expires_at_epoch_ms` 建升序索引。
- `patrol_runs`：服务端私有的 run ID 注册表、逐次更新的轻量历史摘要和终态证据；历史查询按
  设备、不可变的首次接收时间与 run ID 建组合索引，运行中的进度更新不会打乱分页顺序。
  小程序不能直接读取该集合，只能调用字段白名单接口。

每个集合的规则文件位于 `database/rules/`。`patrol_live` 使用：

```json
{ "read": "auth != null && auth.openid != null", "write": false }
```

其他集合使用 `{ "read": false, "write": false }`。云函数以管理端身份访问，不受客户端
规则限制。nonce 保留 24 小时。`cleanupPatrolNonces` 每分钟由七段 cron
`0 * * * * * *` 触发，只选择 `expires_at_epoch_ms < Date.now()` 的记录；单次查询和删除
都硬限制为 100 条，积压时由后续调用继续清理。relay 默认每 0.75 秒最多产生一条记录，
约为每分钟 80 条，因此 100 条/分钟的清理能力可以覆盖默认持续上传速率。等于清理时刻或
尚未到期的记录不会删除。

每分钟定时器会产生约 1,440 次函数调用/天，这是可预估的持续运行成本；部署前应结合云
开发当前免费额度和计费标准核算，并设置函数调用量与账单告警。不得为了减少调用次数而把
计划改回每小时清理，因为 100 条/小时低于默认上传速率，会造成 nonce 持续积压。

`patrol_live/main` 的稳定结构为：

```text
_id = main          # CloudBase 在读取固定 doc 时附加
schema_version = patrol-cloud-snapshot/v1
device_id
run_id
sequence
terminal
body_sha256
received_at_utc
received_at_epoch_ms
payload             # 完整 patrol-dashboard/v1
```

## 构建与测试

需要 Node.js 18 或更高版本：

```bash
cd cloudfunctions
npm test
npm run build
npm run verify:bundles
```

测试不依赖云账号或真实密钥，覆盖 HMAC 篡改、时间窗、nonce、请求体上限、协议验证、
幂等、冲突、回退、终态锁、新运行切换、run ID 防复用、只读快照、陈旧状态，以及 nonce
清理的到期边界、100 条硬上限与重复执行。测试还覆盖空、短、首尾空白、模板、示例和随机
有效密钥，确保填充空格的短密钥与公开占位符都不能通过鉴权；动态障碍物则覆盖 32/33 项
边界、地图圆形包络与合理半径约束。

## 部署前构建

```bash
cd cloudfunctions
npm test
npm run build
npm run verify:bundles
```

这会同时生成 CLI 用的 `.dist/` 和微信开发者工具可见的
`../wechat-miniprogram/cloudfunctions/`。构建不读取也不写入真实密钥。重新构建后，在微信
开发者工具中关闭并重新打开项目，左侧才会刷新出带云朵标记的五个函数目录。五个函数的
`package.json` 和 `package-lock.json` 都把 `wx-server-sdk` 精确锁定为 `3.0.1`，
`npm run verify:bundles` 会同时核验这一点。

部署前还必须用准备写入云函数的同一份钥匙串密钥执行离线预检：

```bash
PATROL_RELAY_UPLOAD_SECRET="$(security find-generic-password \
  -a indoor-patrol-relay -s PATROL_RELAY_UPLOAD_SECRET -w)" \
npm run preflight:secret
```

预检只输出 `ready`/`not ready`、错误代码和不敏感原因，不输出密钥内容；返回码为 0 才可
继续。该检查只验证当前本机提供的值，不会连接云端，也不能代替部署后在控制台核对环境变量。

## 启动 Mac 中转程序

云端资源配置完成并取得 `ingestPatrol` 的专用 HTTPS 地址后，推荐用组合入口一次启动
Webots 与 relay：

```bash
cd /path/to/indoor-patrol-robot-drl
PATROL_RELAY_UPLOAD_SECRET="$(security find-generic-password \
  -a indoor-patrol-relay -s PATROL_RELAY_UPLOAD_SECRET -w)" \
python scripts/run_patrol_with_relay.py \
  --endpoint https://YOUR_CLOUDBASE_DOMAIN/api/patrol/ingest \
  --scenario operational \
  --seed 7000 \
  --run-id patrol-cloud-demo-001
```

组合入口只生成一次 run ID 和结果目录；relay 可先启动并有界等待完整首帧，Webots 进程不会
继承上传密钥。SIGINT/SIGTERM 会联动停止两个进程；Webots 形成终态后，入口会有界等待 relay
收到严格匹配的云端 ACK。relay 故障不会改变机器人动作，但重启预算耗尽或终态 ACK 超时会
让组合命令返回非零，避免把“本地完成、云端未确认”误报成全链路成功。加 `--headless` 可关闭
渲染并快速运行。

原来的双终端方式仍保留用于诊断。此时先运行 `run_webots_patrol.py`，等结果目录中的
`status.json` 和 `events.jsonl` 出现后，再运行独立 sidecar：

```bash
cd /path/to/indoor-patrol-robot-drl
PATROL_RELAY_UPLOAD_SECRET="$(security find-generic-password \
  -a indoor-patrol-relay -s PATROL_RELAY_UPLOAD_SECRET -w)" \
python scripts/publish_patrol_relay.py \
  --run-dir artifacts/results/patrol-cloud-demo-001 \
  --endpoint https://YOUR_CLOUDBASE_DOMAIN/api/patrol/ingest
```

请先在 macOS“钥匙串访问”中建立账户名为 `indoor-patrol-relay`、项目名为
`PATROL_RELAY_UPLOAD_SECRET` 的密码项，并让它与云函数环境变量使用同一个不少于 32 字节
的随机密钥。命令不会把密钥写入项目文件、参数或日志；进程退出后也不会保留 shell 环境
变量。

sidecar 会先把待发的原始 JSON 请求体原子写入
`<run-dir>/.patrol-relay/state.json`，收到与该请求严格匹配的云端 ACK 后才推进序号。断网、
进程退出或 Mac 重启后，用同一 `--run-dir`、设备 ID 和端点再次执行上述命令，会原样重发
尚未确认的数据；若终态已确认则直接结束，不再访问网络。该状态文件只保存公开设备 ID、
run ID、序号、请求体及其哈希，不保存 HMAC 密钥或签名。

## 使用微信开发者工具部署函数

1. 先完成上面的构建，再重新打开 `wechat-miniprogram` 项目并选择环境
   `cloud1-5gpkygv2be4de05a`。
2. 依次右键 `ingestPatrol`、`getPatrolSnapshot`、`getPatrolHistory`、`getPatrolReport`、
   `cleanupPatrolNonces`，选择“上传并部署：云端安装依赖”。
3. 开发者工具只上传函数代码，不会读取 [cloudbaserc.json](cloudbaserc.json)，因此函数
   内存、超时和环境变量仍需在云开发控制台核对。

也可以使用 CloudBase CLI：先在 `cloudfunctions/` 执行 `tcb login`，然后执行
`tcb fn deploy --all`。CLI 会读取 [cloudbaserc.json](cloudbaserc.json) 中的 `.dist`
路径。为避免未展开的公开占位符被误当成真实密钥，`cloudbaserc.json` 有意不包含
`PATROL_RELAY_UPLOAD_SECRET`；无论使用 CLI 还是开发者工具上传，都必须在上传后通过云开发
控制台的函数环境变量安全设置真实值。

### 旧版 `patrol-dashboard/v1` 的滚动部署顺序

协议版本号保持不变，但新遥测增加了 `run.control_step`、`map.dynamic_obstacles`、
`avoidance` 和 `inspection`。为避免三端短暂错配，按以下顺序滚动：

1. 先部署新 `ingestPatrol`、`getPatrolSnapshot`、`getPatrolHistory` 和 `getPatrolReport`；前两个会把旧
   relay/outbox 缺少的三个字段规范化为 `control_step: null`、空动态障碍数组和明确的
   “避障证据不可用”对象。历史接口会把尚无轻量摘要字段的既有终态记录标为“旧版终态”，
   不读取 `terminal_payload` 猜测类型，也不会把它误标为正常完成。旧记录没有完整点位档案，
   因此 `report_available` 为 `false`。
2. 再发布新版小程序；新版能读取尚未更新的旧云快照，并以相同不可用默认值展示。
3. 最后更新并启动 Python/Webots 生产端，使它开始上报完整的新字段。已有旧 outbox 可以继续
   原样重试，HMAC 与请求体哈希仍按原始字节验证，不能为补字段而改写待重试请求体。

默认值只表示旧版未提供证据，不应被解释为“已检测且没有障碍”或“避障功能正常”。回滚时
反向执行，并在回滚生产端前先停 relay，避免新格式待发数据与更旧的消费者同时运行。

## 必须在控制台完成的资源配置

无论使用哪种函数上传方式，以下项目都不会仅靠 `project.config.json` 自动创建：

1. 按 [database/collections.json](database/collections.json) 手工创建四个集合、三个索引，
   并逐一应用 `database/rules/` 中的规则。
2. 给 `ingestPatrol` 设置 `PATROL_CLOUDBASE_ENV_ID`、`PATROL_RELAY_DEVICE_ID`、
   `PATROL_RELAY_UPLOAD_SECRET`；给 `getPatrolHistory` 设置相同的三个变量，其中密钥只用于
   域隔离的 HMAC 分页游标签名；给 `getPatrolSnapshot` 和 `getPatrolReport` 设置前两个变量。真实密钥不得粘贴到
   源码、示例配置、命令参数或日志。给 `cleanupPatrolNonces` 只设置
   `PATROL_CLOUDBASE_ENV_ID`。设置前先通过 `npm run preflight:secret`，设置后确认云端变量
   既不是空值，也不是花括号模板文本；缺少密钥时两个需要密钥的函数都必须失败关闭。密钥
   轮换后，旧分页游标会失效；在历史页下拉刷新即可从第一页重新读取。
3. 在云函数“权限控制”应用 [function.rules.json](function.rules.json)：客户端不能直接调用
   `ingestPatrol` 和 `cleanupPatrolNonces`，`getPatrolSnapshot`、`getPatrolHistory` 与 `getPatrolReport` 只允许
   具备微信身份的调用。
4. 在“环境配置 → HTTP 访问服务”把专用 HTTPS 路径（建议
   `/api/patrol/ingest`）路由到普通事件函数 `ingestPatrol`。只允许 `POST`，单 IP 限频不高于
   3 QPS，并先设置函数总配额和账单告警。公网路由需要允许 Mac 匿名到达函数，但函数内部
   HMAC 鉴权绝不能关闭；限频、总配额和费用告警未实际核验前不得开放 URL。
5. 核对 `cleanupPatrolNonces` 的每分钟定时触发器为七段 cron `0 * * * * * *`。不要为它创建
   HTTP 路由，也不要授予小程序调用权限；管理员需要补清积压时可在云函数控制台手工触发。
   同时确认约 1,440 次调用/天的预计用量在预算内，并启用调用量与账单告警。
6. 不要为 `getPatrolSnapshot`、`getPatrolHistory` 或 `getPatrolReport` 创建公网 HTTP 路由；实时订阅断开时，
   小程序通过前者只读轮询，历史页通过后者读取固定设备的分页摘要。
7. 先上传一条非终态和一条终态测试快照，确认 `patrol_live/main`、`patrol_runs` 和小程序
   陈旧提示正确；确认历史页能区分正在巡检、失联中断、异常中止、正常完成、未验证的任务
   终态和旧版终态；再开始一次全新的 run，确认小程序会自动切换到新任务。

## 部署后验证清单

- 五个函数均使用 Node.js 18，云端安装依赖成功；构建前本地已通过 `npm test`、
  `npm run build`、`npm run verify:bundles` 和真实候选密钥的 `npm run preflight:secret`。
- `cloudbaserc.json` 不含上传密钥或密钥占位符；云函数控制台中的真实密钥通过本地预检，
  删除该变量后 `ingestPatrol` 会失败关闭，重新设置后合法 HMAC 请求才恢复成功。
- `cleanupPatrolNonces` 显示每分钟定时器，且没有 HTTP 路由；从小程序客户端调用它会被
  权限规则拒绝。
- 在测试环境准备 101 条已过期 nonce 和 1 条未过期 nonce 后，由管理员触发两次清理：
  首次 `deleted` 不超过 100，第二次清完剩余过期项，未过期项仍存在；第三次返回
  `deleted: 0`，证明重复执行安全。生产环境不需要为验证写入伪造数据。
- `patrol_relay_nonces.expires_at_epoch_ms` 升序索引状态为“可用”，定时函数日志中没有
  nonce、签名、密钥或请求体内容。
- Mac 上传一条非终态与一条终态数据后，ACK、`patrol_live/main`、`patrol_runs` 和小程序
  展示的 run ID、序号与终态一致。
- `getPatrolHistory` 返回的记录固定属于 `macmini-m4`，篡改分页游标会被拒绝，运行进度更新
  不会改变历史排序，新增任务造成的跨页重复由客户端按 `run_id` 去重；上报期间可下拉刷新
  重建最新顺序。响应中没有 `terminal_payload`、
  `body_sha256`、nonce、签名或密钥字段。
- `getPatrolReport` 只接受一个 `run_id`，且返回 `patrol-report-response/v1`；活动任务、旧记录、
  异常中止或篡改的存储报告均返回同一脱敏错误。响应中没有 `terminal_payload`、
  上传 body 哈希、nonce、签名或密钥。

`.env.example` 只展示变量名称。真实 `.env`、构建产物和 `node_modules` 已被忽略。

## 运维边界

- 请求体上限为 128 KiB；时间偏差上限为正负 300 秒。
- 终态快照写入后，同一 run 不可继续追加；只有新的、从未注册的 run ID 可以接替。小程序
  会保留终态结果并继续监听下一次合法 run。
- 当前架构只传输仿真监测数据，不提供远程控制命令。
- 公开页面必须继续标注“Webots 仿真实时”，不得表述为物理机器人实测。
