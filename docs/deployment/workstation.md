# 双 RTX 3090 工作站部署

GPU 1 承载 MiniCPM-o 的 Thinker、Talker、Code2Wav；GPU 0 承载后台长上下文 LLM 和 Ditto；Ava 2D 使用 CPU。网关统一暴露推理与 Avatar API，应用负责身份、历史、长期记忆、Agent 编排及工具执行。

## 主线组合

统一入口组合 `deploy/compose/base.yml`、`workstation.yml`、`avatar.yml`、`llm.yml` 和 `inference.yml`；`--portrait` 添加神经头像与 GPU 0 MPS，`--vpn` 添加私网监听。

```sh
python tools/compose.py --portrait --vpn -- config --quiet
python tools/compose.py -- build omni-engine llm-engine omni-gateway omni-avatar
python tools/compose.py --portrait -- build omni-portrait
python tools/compose.py -- --profile assets run --rm model-fetch
python tools/compose.py -- --profile assets run --rm llm-model-fetch
python tools/compose.py --portrait -- --profile assets run --rm portrait-assets
python tools/compose.py --portrait --vpn -- up -d --wait
```

完成标准头像镜像与资产准备后，按下方 TensorRT 配置启用实测帧后端。

首次配置与共享网络创建见[快速开始](../../README.zh-CN.md#快速开始)。模型使用精确快照和独立目录，服务离线加载。应用、管理和 Avatar 内部令牌在根目录 `.env` 中分别配置；凭据与模型资产由本地忽略目录持有。引擎使用 Docker 私网，Avatar／Portrait 控制使用私有 Unix socket。宿主发布回环及指定 VPN 地址的网关 8000 端口。

### 独立 LLM 节点

在文本节点设置 `LLM_GPU_DEVICE`（设备 UUID 或宿主索引）和模型目录。独立项目默认将引擎发布到回环地址 8100；跨机接入通过受控内网或隧道，Gateway 的 `LLM_ENGINE_URL` 指向该私有入口。引擎直连绕开 Gateway 的应用鉴权与配额，限制入口仅供受信网关使用。工作站同卡配置下 Images、MPS、Ditto 跟随 LLM 的设备选择，Omni 使用独立的 `OMNI_GPU_DEVICE`。

```sh
python tools/compose.py --target llm -- build llm-engine
python tools/compose.py --target llm -- --profile assets run --rm llm-model-fetch
python tools/compose.py --target llm -- up -d --wait
python tools/compose.py --target llm -- stop llm-engine
```

## API 与上下文

| 入口 | 职责与配置 |
| --- | --- |
| `/v1`，`omni-default` | 实时语音／视觉助手与普通、流式 Chat；Chat 窗口 8192。 |
| `/v1/realtime?duplex=1` | 原生双工，默认 basic 历史水位 8000／6000。 |
| `/llm/v1`，`llm-default` | 长上下文 Responses HTTP／增量 SSE，兼容 Chat，支持结构化与 function 输出；单请求窗口 262144。 |
| `/healthz`、`/llm/healthz` | 各自后端的身份、根目录校验、就绪和准入状态。 |
| `/v1/avatar/*` | 参考图、准备任务、形象、音频绑定、WHEP、字幕与会话生命周期。 |
| `/v1/images/jobs` | 可选 Klein 生图任务，提交、查询、下载、取消。 |

Chat 调用方每次提交 messages 和所选记忆；Realtime 引擎维护活跃近期历史并按窗口策略滑动。两个模型由路径和别名显式选择。实时窗口沿用官方 `basic`；通过 `autostart=0` 和首条 `session.update.session.extra_body` 可配置新会话。可选 `context` 保留 24 个近期输入单元与最多 500 tokens 的较早模型生成文本，并重建窗口。长期历史由应用持有并按需选入提示。[窗口参数](../api.md#omni-实时历史窗口)。

Omni Chat 和后台 Chat／Responses 默认开启 CoT，思考预算沿用固定上游默认；Omni Chat 总输出默认 2048。调用方可覆盖模板开关、思考预算与总输出上限；Realtime 采用模型原生监听／说话策略。[预算字段](../api.md#cot-默认与调用方预算)。

## 四路并发与计算预算

`OMNI_WORKSTATION_CAP=2` 同时配置网关、Omni duplex 活跃窗口和三个阶段。Chat 与 Realtime 共用两个槽位。Ditto 用 `OMNI_PORTRAIT_CAP=2` 配置两个独立执行槽。下面的历史 TPS 与帧时来自单路 Ditto 和单路 Omni Chat 准入的基线部署。

双路 Omni 已实测两个 Chat 同时输出、两个 Realtime 的语音区间重叠，以及 Chat 和 Realtime 混跑。两个 Realtime 占用期间，第三个连接收到 `429`。

Ditto 双槽可打开两个独立会话，25 fps 的双路验收仍未通过。GPU 0 上的隔离测量中，Qwen 空闲且无 WHEP 订阅时，单会话为 24.8 fps，两个同时运行的会话分别为 11.5 和 16.7 fps。当前工作站以 `OMNI_PORTRAIT_CAP=1` 运行，双槽用于继续验收。

LLM 固定为 `axteam/Qwen3.5-9B-AWQ-4bit@5577b5b71e6f06b9f6a1f3743ef6ba6dc4ecd2df`，格式 compressed-tensors W4A16、非对称 group 32，使用 9 GiB BF16 KV 与 FP32 SSM。Omni 固定 OpenBMB AWQ 快照 `927faefa0bce96e557945e106f3b3d8b4d39f4b2`。每个引擎容器将分配的物理 GPU 映射为本地设备 0。

Qwen3.5 的启动空闲显存检查使用整卡的 75%，支持两个 Ditto 槽常驻时单独重启 LLM。KV 仍通过固定字节配置分配。

同一客户端空闲三轮实测的中位数如下。单路提速与四路总吞吐均在同一 256-token、CoT 开启的夹具下测量。

| LLM MPS 上限 | 单路端到端 TPS | 四路端到端合计 TPS |
| --- | ---: | ---: |
| 20% | 41.8 | 139.1 |
| 40% | 63.8 | 212.1 |
| 60% | 70.0 | 234.1 |
| 100% | 71.9 | 269.7 |

实机双路 Omni 和 Qwen 测量记录见 [容量验收](../../evaluations/results/workstation-dual-capacity.json)。

单 Ditto 运行配置可显式使用 `LLM_MPS_ACTIVE_THREAD_PERCENTAGE=100`。工作站用直接音频输入和 WHEP 连续跑了 90 秒，最大采样时间线偏差 64 ms。四轮四路 Qwen 实际生成重叠，最慢一路为 25.72 TPS。工作站当前使用该单 Ditto 配置。

`LLM_MAX_NUM_SEQS=4` 同时配置引擎与网关；`LLM_TOKEN_BUDGET=262144` 控制活跃请求的完整模板输入加最大输出预留。Chat、Responses、后台任务和续接共享预算，长短请求按 FIFO 等待可用序列与 token，完成或取消后释放。调用方显式设置短请求输出额度可释放并发空间。单请求 262144 与四路短请求分别验收；多路满窗口作为后续独立容量评估。[准入语义](../api.md#外部-swarmorchestrator-调用)。

GPU 0 使用项目内 `gpu-mps` 服务。覆盖层的基线 LLM 活跃线程上限为 20%，实机可见 16 个 SM。LLM 使用较低客户端优先级，Ditto 使用正常优先级。LLM 采用同步调度与 `FULL_DECODE_ONLY` CUDA Graph，捕获批次 1、2、4。MPS 分配计算资源，token 预算保护显存，两个机制分别生效。

当前工作站启用 TensorRT FP16 warp＋decoder，CoT 开启，思考预算采用上游默认；温度 0，每路总输出上限 256 tokens，预热 32 tokens。固定短提示按模板实报输入为 Omni 25、LLM 27 tokens，前缀缓存关闭。以下统计端到端思考＋正文总生成 TPS；通话场景列出两轮测量范围。

| 服务 | 生成请求数 | 其他负载 | 每路总生成 TPS | 合计 TPS |
| --- | --- | --- | --- | --- |
| Omni Chat | 1 | 空闲 | 65.97 | 65.97 |
| Omni Chat | 1 | 原生语音＋头像＋四路 LLM | 33.60–37.39 | 33.60–37.39 |
| LLM | 1 | 空闲 | 46.69 | 46.69 |
| LLM | 4 | 空闲 | 36.94–37.09 | 147.77 |
| LLM | 1 | 原生语音＋头像 | 25.55–26.46 | 25.55–26.46 |
| LLM | 4 | 原生语音＋头像 | 21.20–21.56 | 84.81–86.04 |
| LLM | 4 | 原生语音＋头像＋一路 Omni Chat | 19.47–21.47 | 77.89–85.33 |

单路 Chat 准入的基线中，同时提交的两个短请求每路为 65.34／32.74 TPS，合计 65.47 TPS，包含排队时间。后台四路以实际输出区间重叠验证同时生成。

每路 TPS = `usage.completion_tokens`（Omni Chat）或 `usage.output_tokens`（Responses）÷ 从本组请求发起到该路完成的秒数；合计 TPS = 本组全部生成 tokens ÷ 最晚完成时间。计量包含思考、预填充、传输和准入／GPU 等待。原始记录另存 `(tokens - 1) / (末增量 - 首增量)` 的生成区间参考值，精度受流式分块与控制 tokens 影响。

256-token 性能夹具覆盖总生成吞吐：Omni 本组额度用于思考阶段，报告分别记录思考与正文首增量。完整答案、预算与结构化正确性由独立功能夹具验证。[前台与全栈混跑](../../evaluations/results/omni-throughput-tensorrt-workstation.json) · [后台单路／四路](../../evaluations/results/throughput-tensorrt-workstation.json)。后台回归检查线为空闲单路至少 45 TPS、原生语音＋头像四路每路至少 15 TPS；额外 Omni Chat 的更重混跑单独报告。

Ditto 使用 FP16、十步采样、640×720 和 1200 ms 音频输入预留。全栈 90 秒生成 2,265 张真实神经帧，最大检查期采样时间线偏差 57 ms；后台共驻的 90 秒生成 2,267 帧、偏差 64 ms。两组均通过 25 fps／160 ms 检查、原生取消恢复与幂等释放。

### 渲染截止期反馈

每个头像槽在 MPS 控制卷原子发布单调时钟帧进度。LLM 在 GPU 批次派发前读取各槽有效反馈，按最紧迫的实际渲染余量分配算力。该私有信号仅包含帧时间信息。

| 反馈状态 | 派发行为 |
| --- | --- |
| 有效反馈，余量至少 240 ms | 正常派发。 |
| 有效反馈，余量低于 240 ms | 每次主动让出最多 18 ms，每 2 ms 检查进度，余量恢复后提前派发。 |
| 头像空闲、文件缺失、无效或更新间隔超过 500 ms | 正常派发。 |

会话释放时删除自己的时钟文件，异常退出后的记录按 500 ms 有效期处理。SSE 即时转发引擎增量。调整 MPS 或计算参数后，复验四路生成与头像时间线。

### MPS 生命周期与更新

`gpu-mps` 使用镜像内 NVIDIA 控制程序和多用户模式；GPU 0 客户端共享 PID／IPC 命名空间及控制 socket 卷，并依赖控制服务健康。GPU 1 专用于 Omni。参见 [NVIDIA MPS 文档](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html)。

重建控制服务时，先排空请求并停止客户端，再按顺序恢复：

```sh
python tools/compose.py --portrait --vpn -- stop llm-engine omni-portrait
python tools/compose.py --portrait --vpn -- up -d --wait --no-deps --force-recreate gpu-mps
python tools/compose.py --portrait --vpn -- up -d --wait llm-engine omni-portrait omni-gateway
```

已准备 avatar_id 持久化于卷中。Responses 续接保存在引擎进程内，调用方持有可跨重启恢复的业务历史。重建后的形象复用、四路／25 fps、音画与取消恢复通过。[MPS 生命周期](../../evaluations/results/mps-lifecycle-workstation.json) · [推理预算与空闲释放](../../evaluations/results/runtime-cot-deadline-workstation.json)。

常规更新以仓库根目录的统一入口构建并启动，项目名为 `tuber-omni-infra`；根目录 `.env`、`data/` 和 Docker named volumes 持有配置与资产。更新前排空请求，重建后由应用建立新会话。

## 会议式神经头像

`--portrait` 启用私有 `omni-portrait`，与 LLM 共驻 GPU 0；`omni-avatar` 持有会话、媒体与权限。参考图上传 → 异步准备 → avatar_id → 音频／自有 Realtime 绑定 → WHEP。模型只读挂载，参考图、特征和编译缓存分别持久化。已启用的神经 GPU 槽由通话和参考图准备共享，2D 会话由 CPU 渲染。

服务启动时完成编译和 Graph 捕获，会话就绪前预热实际线程并填充初始视频缓冲。输出 640×720、25 fps，输入预留 1200 ms。驱动扩展沿用同一音频、代次与媒体契约。[在线接口](../api.md#参考图与会议式头像) · [驱动与资产](../../services/portrait/README.md)。

## 可选生图

在 `.env` 设置独立的 `OMNI_IMAGE_INTERNAL_TOKEN` 和 `LLM_GPU_CONTROL_TOKEN`。先构建主线引擎，再构建私有 image worker。首次下载使用对宿主模型目录有写权限的 UID。

```sh
python tools/compose.py -- build omni-engine
python tools/compose.py --portrait --images -- build omni-gateway omni-portrait omni-images
python tools/compose.py --portrait --images -- --profile assets run --rm -u "$(id -u):$(id -g)" image-model-fetch
python tools/compose.py --portrait --images --vpn -- up -d --wait
```

Klein 4B 蒸馏快照使用原精度 BF16。加载和文本编码在 CPU 上进行，再申请 GPU 0。协调器等神经头像空闲并排空 LLM。当前空闲量加 KV 预算预计能满足 `OMNI_IMAGE_MIN_FREE_MIB` 时先仅释放 KV；预计空间不足时直接一级休眠，将 LLM 权重备份到 CPU。释放后以整卡空闲显存复核。阈值默认 13312 MiB。GPU 清理完成后先恢复 LLM，再开放生成。

双 Ditto 槽空闲驻留时，首轮只释放 KV 的样本中 GPU 0 峰值达到 24,044 MiB。提高余量并用 CPU 原权重恢复图像模块后，实测真人参考冷任务 15.89 秒、动漫参考热任务 9.28 秒、文生图热任务 7.78 秒。三项整卡峰值均为 21,206 MiB，LLM 恢复约 0.75 秒，已有 Responses 历史可续接。真人样本保留了可见脸部特征，徽章文字有失真。动漫样本保持发型和服装辨识度，眼睛和饰品细节有变化。 [记录与样图](../../evaluations/results/klein4b-workstation/)

停用生图前先取消或等待图片任务进入终态，并确认 `/llm/healthz` 为 `ready`，再停掉 `omni-images` 并在去掉 `--images` 后重建网关和 LLM。恢复失败时保留 GPU 保护标记，确认图像 worker 已退出 GPU 工作后恢复 LLM 引擎，协调器会在恢复确认后清除标记。模型和结果卷可复用。 [生图 API](../api.md#生图任务)

## Avatar 与访问

工作站为 `research-2@100.119.17.6`，仓库 `/home/research-2/code/omni-avatar-api`。API 客户端使用 `http://100.119.17.6:8000/v1` 或 `http://100.119.17.6:8000/llm/v1`，各应用使用独立 Bearer。`--vpn` 保留回环监听并发布 `OMNI_GATEWAY_VPN_BIND` 指定地址。

浏览器通过可信 localhost 上下文访问：

```sh
ssh -N -L 18080:127.0.0.1:8000 research-2@100.119.17.6
```

打开 `http://localhost:18080`。Avatar 使用 host 网络发布 ICE UDP，WHEP 媒体经 LAN／VPN 直连；SSH 隧道承载页面和控制 TCP。私网 HTTPS 由 tailnet 管理员启用 Serve 后配置。[私网访问验证](../../evaluations/results/vpn-api-workstation.json)。

## 实机准入

| 验证层 | 已覆盖范围 | 证据 |
| --- | --- | --- |
| 推理功能 | 分卡引擎、Responses SSE、JSON Schema、function 输出与夹具回填、归属续接、22,753-token 多轮历史、独立准入与取消；约 20 分钟 Ava 2D／双工共驻、40 次后台请求。 | [功能](../../evaluations/results/inference-functional-workstation.json)、[断流](../../evaluations/results/responses-stream-lifecycle.json) |
| Omni 上下文 | 显式关闭 CoT 的 20 条消息、7,928-token 早期事实检索；超限输入 400。 | [8K 边界](../../evaluations/results/omni-8k-multi-turn-workstation.json) |
| 并发预算 | 四路实际生成、约 28.5K 与三路短请求混跑、独立 worker 历史及并行续接、预算拒绝与释放。 | [并发](../../evaluations/results/llm-mps-concurrency-workstation.json)、[兼容性](../../evaluations/results/responses-mps-compatibility.json) |
| 默认 CoT | 默认完整答案、显式关闭、三个文本接口的 16-token 思考预算、非法值、JSON Schema、function 归属续接。工具结果由夹具回填。 | [默认与预算](../../evaluations/results/thinking-defaults-workstation.json)、[结构化与工具](../../evaluations/results/thinking-structured-workstation.json) |
| 神经媒体 | 当前 CoT 全栈 90 秒混跑、25 fps、时间线、取消恢复及释放；重启复用。 | [混跑](../../evaluations/results/omni-throughput-tensorrt-workstation.json)、[生命周期](../../evaluations/results/portrait-lifecycle-workstation.json) |

当前记录覆盖功能与短时性能。完整 262K 质量、默认 CoT 长历史、持续满载 QoS、实物播放及私网 HTTPS 按[后续计划](../plans/avatar-workstation.md)验收。资源读数以报告的采集位置、指标和采样周期为依据。

## TensorRT FP16 对照

工作站选择 `OMNI_PORTRAIT_FRAME_BACKEND=tensorrt`、`OMNI_PORTRAIT_IMAGE=omni-api-portrait:ditto-trt`、`OMNI_PORTRAIT_DOCKERFILE=Dockerfile.tensorrt`。标准部署默认 `cuda-graph`；外部部署通过这三个配置选择帧实现与独立镜像。[构建、预编译与启动](../../services/portrait/README.md#tensorrt-frame-backend)。

TensorRT CUDA 13 固定为 11.4.0.106，目标为 RTX 3090 SM 8.6；引擎在目标卡构建，缓存到头像卷 `/cache/tensorrt`。warp 和 decoder 合并为一个引擎，卷积 FP16、关键点与坐标 FP32，使用同一十步采样、分辨率、1200 ms 音频预留、MPS 20% 和渲染截止期策略。

| 同机同条件指标 | CUDA Graph | TensorRT |
| --- | --- | --- |
| warp＋decoder GPU 均值，100 次 | 26.81 ms | 21.10 ms |
| warp＋decoder GPU p95 | 27.16 ms | 21.20 ms |
| 通话＋头像期间 LLM 单路 | 21.37–24.65 TPS | 25.55–26.46 TPS |
| 通话＋头像期间 LLM 四路合计，两轮 | 68.06／75.87 TPS | 86.04／84.81 TPS |

独立帧均值减少 21.3%；两轮四路合计 TPS 的均值提高约 18.7%。空闲 LLM 在同轮对照中分别为 48.42／46.69 TPS，四路合计 153.26／147.77 TPS；头像收益体现于共享 GPU 的活动通话场景。端到端指标沿用上方公式与 CoT 256-token 夹具。

像素对照使用同一 Ava 准备特征、静止及两组固定种子关键点扰动；三组 MAE 为 0.000476–0.001069，PSNR 48.73–63.05 dB，静态对照帧已目检。内存字段分别记录 PyTorch 分配和设备空闲采样；运行余量结合共驻记录观察。主观动态口型由后续人工验收覆盖。

[独立帧与数值](../../evaluations/results/portrait-frame-tensorrt-workstation.json) · [CUDA Graph 同机对照](../../evaluations/results/throughput-cudagraph-reference-workstation.json) · [TensorRT 共驻](../../evaluations/results/throughput-tensorrt-workstation.json) · [TensorRT 全栈](../../evaluations/results/omni-throughput-tensorrt-workstation.json)。

TensorRT 重启后加载目标卡缓存，复用已准备形象的 30 秒四路／25 fps、时间线、取消恢复与幂等释放通过。[重启复用](../../evaluations/results/portrait-tensorrt-restart-workstation.json) · [运行配置与引擎身份](../../evaluations/results/portrait-tensorrt-runtime-workstation.json)。

## 其他部署目标

`--target workstation-omni` 提供双卡 Omni 配置，`--target workstation-bf16` 提供 BF16 对照；`--target orin` 和 `--target orin-cu129` 提供 AGX 配置与兼容性验证。`--portrait` 适用于分卡 `workstation` 目标。直接调用 Docker Compose 时，指定 `--project-directory <仓库根目录>` 和环境文件。AGX 细节见 [Orin](orin.md) 与 [CUDA 12.9](jetpack6-cu129.md)。
