# 双 RTX 3090 工作站部署

默认 `--target workstation` 采用 Personal-Agent-First 双卡并行布局：

- **LLM (`llm-engine`)**：`RedHatAI/Qwen3.8-27B-INT4`（SHA `71dface38d4cbfa34121f1d446d4a0fa1025f13c`）基于 `vllm/vllm-openai:v0.34.0@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1`（镜像 `omni-api-llm:0.34.0`），`TP=2` 横跨双 3090（`LLM_GPU_DEVICE_0=0`、`LLM_GPU_DEVICE_1=1`，`LLM_TP2_GPU_MEMORY_UTILIZATION=0.80`，`shm_size: 16gb`），默认 131072 上下文、MTP=3、`align` 前缀缓存、8 并发序列。
- **TTS (`omni-tts`)**：默认 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`（SHA `5d83992436eae1d760afd27aff78a71d676296fc`；内含 `speech_tokenizer`）运行于 GPU 1（`TTS_GPU_DEVICE=1`，`pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385`，镜像 `omni-api-tts:qwen3-0.6b`）。
- **全双工与 ASR (`omni-engine` + `omni-asr`)**：纯 CPU 运行，零显存占用。`omni-engine`（`python:3.12-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64`，镜像 `omni-api-duplex:1.0`）使用 `onnx-community/silero-vad`（SHA `e71cae966052b992a7eca6b17738916ce0eca4ec`）串联 `omni-asr`（`omni-api-asr:small-int8`）、`llm-engine` 与 `omni-tts`；`OMNI_DUPLEX_CAP=4`。
- **可选生图 (`omni-images`)**：默认放在 GPU 0（`IMAGE_GPU_DEVICE=0`），避开 GPU 1 的常驻 `omni-tts`；作业期间按 `OMNI_IMAGE_LLM_SLEEP_STRATEGY=auto` 自动释放 TP=2 KV 或休眠卸载权重，完成后恢复。

网关统一暴露推理 API，应用负责身份、历史、长期记忆、Agent 编排及工具执行。

## 主线组合

### 双卡放置、1.7B-Base 与 Genie-TTS 切换

双 3090 工作站可在 `.env` 中使用索引或 GPU UUID 绑定 `LLM_GPU_DEVICE_0`、`LLM_GPU_DEVICE_1`、`TTS_GPU_DEVICE`（默认 1）与 `IMAGE_GPU_DEVICE`（默认 0）。注意：TP=2 按双卡剩余显存较小的那一侧切分 KV，因此**不要**把 `omni-images` 放到与 `omni-tts` 相同的 GPU 1。

- **升级 `Qwen3-TTS-12Hz-1.7B-Base`**：将 `.env` 中的 `TTS_MODEL_ID` 设为 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`、`TTS_CHECKPOINT` 设为 `fd4b254389122332181a7c3db7f27e918eec64e3`，并把 `LLM_TP2_GPU_MEMORY_UTILIZATION` 从 `0.80` 下调至 `0.72`（为 GPU 1 留出 ~6.7 GiB 余量后即可常驻），再执行 `tts-model-fetch` 并重启 `omni-tts`。
- **切换 CPU `Genie-TTS`（0 MiB 显存）**：在 `.env` 设置 `TTS_BACKEND=genie`（及 `GENIE_DATA_DIR` / `GENIE_MODEL_DIR` / `GENIE_CHARACTER_NAME`），可完全腾空 GPU 1 上的 TTS 显存，为 `llm-engine` 释放更高 `LLM_TP2_GPU_MEMORY_UTILIZATION` 或 `262144` 上下文。
- **可控开关与回退**：思考默认上限由 `LLM_DEFAULT_THINKING_TOKEN_BUDGET=16384` 控制；`LLM_TP2_ENABLE_MTP`、`LLM_TP2_ASYNC_SCHEDULING`、`LLM_TP2_ENABLE_PREFIX_CACHING`、`LLM_TP2_ENABLE_LORA` 均可在 `.env` 单独切换，无需改镜像。

### 组合启动

统一入口默认组合 `base.yml`、`workstation.yml`、`llm.yml`、`inference.yml`、`asr.yml`、`llm-tp2.yml`、`tts.yml` 与 `omni-duplex.yml`；`--vpn` 添加私网监听，`--images` 添加 Klein 4B 生图。

```sh
python tools/compose.py --vpn -- config --quiet
python tools/compose.py -- build llm-engine omni-tts omni-asr omni-engine omni-gateway
python tools/compose.py -- --profile assets run --rm llm-model-fetch
python tools/compose.py -- --profile assets run --rm tts-model-fetch
python tools/compose.py -- --profile assets run --rm vad-model-fetch
python tools/compose.py --vpn -- up -d --wait
```

### 回退旧版单卡栈（`--target workstation-legacy`）

保留原 `Qwen3.5-9B-AWQ-4bit@5577b5b71e6f06b9f6a1f3743ef6ba6dc4ecd2df`（`omni-api-llm:0.32.0`）与 `MiniCPM-o-4_5-awq@927faefa0bce96e557945e106f3b3d8b4d39f4b2`（`omni-api-engine:0.32.0-workstation`）的 4 文件栈及各自镜像标签，模型快照与 `tuber-omni-infra` 命名卷互不覆盖。如需回退，一条命令切换回原部署（按需附加 `--images`、`--asr`、`--vpn`）：

```sh
python tools/compose.py --target workstation-legacy --images --asr --vpn -- up -d --wait
```

首次配置与共享网络创建见[快速开始](../../README.zh-CN.md#快速开始)。模型使用精确快照和独立目录，服务离线加载。应用、管理以及内部 worker 令牌（`OMNI_ASR_INTERNAL_TOKEN`、`OMNI_TTS_INTERNAL_TOKEN`）在根目录 `.env` 中分别配置；凭据与模型资产由本地忽略目录持有。引擎使用 Docker 私网。宿主发布回环及指定 VPN 地址的网关 8000 端口。

### 独立 LLM 节点

在文本节点设置 `LLM_GPU_DEVICE`（设备 UUID 或宿主索引）和模型目录。独立项目默认将引擎发布到回环地址 8100；跨机接入通过受控内网或隧道，Gateway 的 `LLM_ENGINE_URL` 指向该私有入口。引擎直连绕开 Gateway 的应用鉴权与配额，限制入口仅供受信网关使用。工作站同卡配置下 Images 跟随 LLM 的设备选择，Omni 使用独立的 `OMNI_GPU_DEVICE`。

```sh
python tools/compose.py --target llm -- build llm-engine
python tools/compose.py --target llm -- --profile assets run --rm llm-model-fetch
python tools/compose.py --target llm -- up -d --wait
python tools/compose.py --target llm -- stop llm-engine
```

### 独立 Omni 节点

`--target omni` 提供单独的 amd64 三阶段引擎与 model-fetch。选择 `OMNI_GPU_DEVICE`，依次执行 `build omni-engine`、`--profile assets run --rm model-fetch`、`up -d --wait`。引擎私有端口默认回环 8102；Gateway 的 `OMNI_ENGINE_URL` 指向客户端节点可达的私有地址。JetPack／ARM64 沿用各自的构建目标。

### 独立 Gateway

`--target gateway` 单独启动 CPU 网关，只需应用／管理凭据即可存活。配置 `LLM_ENGINE_URL`、`LLM_MODEL_ID`、`LLM_CHECKPOINT` 连接后台推理引擎，或加 `--remote-images` 连接专用图像池。启用远端 Omni 时设 `OMNI_GATEWAY_OMNI_ENABLED=1`，填写引擎 origin、模型 ID 和 revision。未启用 Omni 时 `/livez` 仍可用，`/healthz` 及 Omni 推理路径报告 503；LLM 使用自己的 `/llm/healthz`。

## API 与上下文

| 入口 | 职责与配置 |
| --- | --- |
| `/v1`，`omni-default` | CPU 全双工编排器（Silero VAD + `omni-asr` + `llm-engine` + `omni-tts`）；Chat 窗口跟随 `LLM_TP2_MAX_MODEL_LEN`（默认 131072），`--target workstation-legacy` 下为 8192。 |
| `/v1/realtime?duplex=1` | WebSocket 实时语音（16 kHz PCM16 输入 / 24 kHz PCM16 输出、打断 `epoch` 隔离与 `video_frames` 视觉注入），`OMNI_DUPLEX_CAP=4`（默认 4 路并发）。 |
| `/v1/audio/speech` | 零样本参考音色朗读；默认经 `omni-tts` 合成并由 CPU `omni-asr` 盲转写核对后返回 24 kHz WAV。 |
| `/v1/audio/transcriptions` | CPU `faster-whisper` 离线转写与分段时间戳；默认随 `workstation` 启用，旧目标可用 `--asr` 开启。 |
| `/llm/v1`，`llm-default` | 长上下文 Responses HTTP／增量 SSE，兼容 Chat，支持文本、内嵌图片、结构化与 function 输出；TP=2 默认窗口 131072（可调至 262144）、并发 8（可调至 16）。 |
| `/healthz`、`/llm/healthz` | 各自后端的身份、根目录校验、就绪和准入状态。 |
| `/v1/images/jobs` | 可选 Klein 生图任务，提交、查询、下载、取消（默认 GPU 0）。 |

现有 Qwen 权重包含视觉部分。升级时用原模型卷重建并重启 `llm-engine`，`LLM_MAX_IMAGES_PER_REQUEST` 默认 8，填写 0 可切回只加载语言部分。以 Responses 同一请求体先计量再生成，核对实际计量和图像内容。变更图片上限、推理并发或 KV 预留后按实测显存复核。

默认开启 CoT 时，`omni-gateway` 会在未显式指定 `thinking_token_budget` 时注入 `LLM_DEFAULT_THINKING_TOKEN_BUDGET=16384`（由 `vLLM 0.34.0` 原生预算采样控制，防止长上下文思考循环）；调用方可显式覆盖模板开关、思考预算与总输出上限；`omni-engine` 语音对话自动设置 `enable_thinking=false` 以压低首字延迟。[预算字段](../api.md#cot-默认与调用方预算)。

## 并发与计算预算（新默认 vs legacy 基线）

新默认 `workstation` 下：`OMNI_DUPLEX_CAP=4` 控制纯 CPU 全双工与 Omni Chat 共有 4 个实时槽位；`LLM_TP2_MAX_NUM_SEQS=8` 与 `LLM_TP2_TOKEN_BUDGET=262144` 控制 `llm-engine` 的活跃请求预留与 FIFO 排队（单请求默认上限 `LLM_TP2_MAX_MODEL_LEN=131072`）。

以下为历史单卡回滚目标（`--target workstation-legacy`，`OMNI_WORKSTATION_CAP=2`、`LLM_MAX_NUM_SEQS=4`）的实机基线记录，供回滚对照参考：

双路 Omni 已实测两个 Chat 同时输出、两个 Realtime 的语音区间重叠，以及 Chat 和 Realtime 混跑。两个 Realtime 占用期间，第三个连接收到 `429`。

### GPU 0 显存余量

分卡工作站默认启用 `OMNI_RELEASE_UNUSED_CUDA_CACHE=1`。Thinker 预热结束后归还空闲的 CUDA 分配器缓存。Code2Wav 在最后一个请求结束、无待处理请求和参考音频预取时，若分配器中的空闲预留达到 512 MiB，则归还可释放的缓存。模型权重、参考音频特征、Thinker 和 Talker 的 KV 预算、双路准入和 8192-token Chat 窗口沿用原配置。

双 RTX 3090 实测中，Thinker 预热结束时释放 **4,022 MiB**，Code2Wav 每次实际回收 **3,388–4,228 MiB**。每 250 毫秒采样整卡，候选负载期间最低空闲显存为 **2,821 MiB**，热机空闲约 **7,000 MiB**。这两项分别高于 2 GiB 和 3 GiB 的验收线，候选日志中未出现 CUDA OOM。

通过网关对照执行 10 轮，覆盖 8 组输出实际重叠的双路语音、8,056-token Chat 与语音混跑，以及图像输入与语音混跑。按固定音频输入结束到首段非静音 PCM 的时间计量，两侧各采集 18 条，P95 从 **12.63 秒**变为 **12.05 秒**。在工作站回环入口另测两路语音持续 90 秒、Qwen 单路和四路并行，以及长 Chat 输出期间继续接收语音块。私网访问的 90 秒试跑曾在约一分钟时断开，持续双路的通过记录来自回环入口。

如需恢复原分配策略，在工作站 `.env` 设置 `OMNI_RELEASE_UNUSED_CUDA_CACHE=0`，使用当前 Compose 选项执行 `up -d --no-deps omni-engine`。配置变更会重建 Omni 容器并结束当前 Omni 会话。

### 后台 Qwen 并发

LLM 固定为 `axteam/Qwen3.5-9B-AWQ-4bit@5577b5b71e6f06b9f6a1f3743ef6ba6dc4ecd2df`，格式 compressed-tensors W4A16、非对称 group 32，使用 9 GiB BF16 KV 与 FP32 SSM。Omni 固定 OpenBMB AWQ 快照 `927faefa0bce96e557945e106f3b3d8b4d39f4b2`。每个引擎容器将分配的物理 GPU 映射为本地设备 0。

`LLM_MAX_NUM_SEQS=4` 同时配置引擎与网关；`LLM_TOKEN_BUDGET=262144` 控制活跃请求的完整模板输入加最大输出预留。Chat、Responses、后台任务和续接共享预算，长短请求按 FIFO 等待可用序列与 token，完成或取消后释放。调用方显式设置短请求输出额度可释放并发空间。单请求 262144 与四路短请求分别验收；多路满窗口作为后续独立容量评估。[准入语义](../api.md#外部-swarmorchestrator-调用)。

每路 TPS = `usage.completion_tokens`（Omni Chat）或 `usage.output_tokens`（Responses）÷ 从本组请求发起到该路完成的秒数；合计 TPS = 本组全部生成 tokens ÷ 最晚完成时间。计量包含思考、预填充、传输和准入／GPU 等待。原始记录另存 `(tokens - 1) / (末增量 - 首增量)` 的生成区间参考值，精度受流式分块与控制 tokens 影响。

旧版 Ditto／MPS 部署的实测值随测量条件保存在[前台混跑](../../evaluations/results/omni-throughput-tensorrt-workstation.json)和[后台混跑](../../evaluations/results/throughput-tensorrt-workstation.json)。新版的吞吐可用[工作站工具](../../tools/README.md#tps-与混跑)测量。

### 当前纯推理部署

2026-11-19 在版本 `72755bd` 下，工作站运行 Gateway、MiniCPM-o、Qwen 和 Klein 四个服务。原 CPU 头像容器在停止前的空闲采样为 101.1 MiB 主机内存、65 个进程或线程、CPU 0.00%，该容器未分配 GPU。网关直接转发原生语音 WebSocket。

通过网关发送固定问题的 PCM，MiniCPM-o 返回非静音音频和正确答案的字幕。Klein 生成一张 1024×1024 PNG 用时 8.47 秒，作业仅让 Qwen 释放 KV 缓存；随后用原 Responses ID 成功取回生图前给出的数字。旧头像会话入口在鉴权后返回 404，旧事件路径不能签发浏览器 ticket。

使用下节相同的 256-token 短提示、温度和 CoT 配置，单路预热一次，再测一次单路和一次四路。Qwen 单路为 85.87 tokens/s，四路合计为 321.43 tokens/s。两次部署的结果均为短时采样。

### 移除 Ditto 后的工作站实测

2026-11-19 在同一台双 RTX 3090 工作站，从 `082f213` 的单 Ditto／MPS 部署切换到 `500ccc8`。Omni 使用 GPU0，Qwen 和按需运行的 Klein 使用 GPU1。以下显存来自模型就绪、未运行测试作业时的整卡采样。

| 指标 | 切换前 | 切换后 |
| --- | ---: | ---: |
| GPU1 空闲显存（MiB） | 4,546 | 6,322 |
| Qwen 单路端到端输出（tokens/s） | 81.74 | 86.35 |
| Qwen 四路合计端到端输出（tokens/s） | 301.81 | 323.19 |

两侧均通过同一工作站网关，使用 [Responses 短提示夹具](../../tools/qualification/concurrency.py) 的 `stream_requests`，温度为 0、CoT 开启、每路最多输出 256 tokens。先跑一次单路预热，再各测一次单路与四路。四路的实际输出区间重叠。吞吐值为本次短测的观测值。

在该历史版本中，Ava 和登记的 Inochi2D 原画 `q` 各运行四秒 WHEP，分别解码 101、100 帧视频，两路都收到非静音音频并成功关闭。Omni Chat 返回正确答案。Klein 完成一张 1024×1024 PNG，用时 8.94 秒，仅释放 LLM 的 KV 缓存；恢复后用生图前创建的 Responses ID 续接，成功取回原对话中的数字。

### 从旧头像部署升级

使用原版本和原有 Compose 选项停止并移除 `omni-avatar`，独立 Avatar 节点也需停止该服务。更新代码后，去掉 `--registered-avatars`，保留原有的图像与网络选项，重新构建并启动 `omni-gateway`。模型和数据卷可保留。用应用 Bearer 检查两个模型的健康接口、原生实时会话及生图任务。

如果从仍运行 Ditto 的版本升级，先停止 `omni-portrait` 和 `llm-engine`，再停止 `gpu-mps`。移除这三个已停止的容器，去掉 `--portrait` 后重建并启动 `llm-engine`。此步骤会中断当时的 LLM 请求。

## 可选生图

在 `.env` 设置独立的 `OMNI_IMAGE_INTERNAL_TOKEN` 和 `LLM_GPU_CONTROL_TOKEN`。各服务独立构建。首次下载使用对宿主模型目录有写权限的 UID。

设置 `OMNI_IMAGE_PRELOAD_CPU=1` 时，Klein 在启动期间加载 CPU 权重、预热文本编码器，完成后才监听 HTTP。首个实际作业仍按调用方的提示编码，并在取得 GPU 租约后上传权重。该选项也适用于独立图像节点。默认值 0 将 CPU 初始化推迟到首个作业。

```sh
python tools/compose.py -- build omni-engine llm-engine
python tools/compose.py --images -- build omni-gateway omni-images
python tools/compose.py --images -- --profile assets run --rm -u "$(id -u):$(id -g)" image-model-fetch
python tools/compose.py --images --vpn -- up -d --wait
```

Klein 4B 蒸馏快照使用原精度 BF16。加载和文本编码在 CPU 上进行，再申请共享 GPU。CPU 准备完成后，协调器暂停新的 LLM 准入并等待现有请求完成。当前空闲量加 KV 预算预计能满足 `OMNI_IMAGE_MIN_FREE_MIB` 时先仅释放 KV；预计空间不足时直接一级休眠，将 LLM 权重备份到 CPU。释放后以整卡空闲显存复核。阈值默认 13312 MiB。GPU 清理完成后先恢复 LLM，再开放生成。

在旧版双 Ditto 槽空闲驻留时，首轮只释放 KV 的样本中 GPU 0 峰值达到 24,044 MiB。提高余量并用 CPU 原权重恢复图像模块后，实测真人参考冷任务 15.89 秒、动漫参考热任务 9.28 秒、文生图热任务 7.78 秒。三项整卡峰值均为 21,206 MiB，LLM 恢复约 0.75 秒，已有 Responses 历史可续接。真人样本保留了可见脸部特征，徽章文字有失真。动漫样本保持发型和服装辨识度，眼睛和饰品细节有变化。 [记录与样图](../../evaluations/results/klein4b-workstation/)

停用生图前先取消或等待图片任务进入终态，并确认 `/llm/healthz` 为 `ready`，再停掉 `omni-images` 并在去掉 `--images` 后重建网关和 LLM。恢复失败时保留 GPU 保护标记，确认图像 worker 已退出 GPU 工作后恢复 LLM 引擎，协调器会在恢复确认后清除标记。模型和结果卷可复用。 [生图 API](../api.md#生图任务)

### 在另一节点运行 Klein

在图像节点设置 `IMAGE_GPU_DEVICE`、`OMNI_IMAGE_INTERNAL_TOKEN` 和模型目录，执行以下命令。默认监听回环 8101，可通过私网隧道接入；在专用网络上用 `IMAGE_PUBLISH_BIND` 选择受控监听地址。在 Gateway 节点设置可达的 `OMNI_IMAGE_WORKER_URL`、同一 worker token，选择 `--remote-images`。部署配置决定是否启用专用节点，按请求动态选择多个实例仍在设计阶段。

```sh
python tools/compose.py --target images -- build omni-images
python tools/compose.py --target images -- --profile assets run --rm image-model-fetch
python tools/compose.py --target images -- up -d --wait
python tools/compose.py --target gateway --remote-images -- up -d --wait
```

## 工作站访问

工作站为 `research-2@100.119.17.6`，仓库 `/home/research-2/code/omni-avatar-api`。API 客户端使用 `http://100.119.17.6:8000/v1` 或 `http://100.119.17.6:8000/llm/v1`，各应用使用独立 Bearer。`--vpn` 保留回环监听并发布 `OMNI_GATEWAY_VPN_BIND` 指定地址。

本地运维可在忽略提交的 `.env` 中填写 `WORKSTATION_HOST`、`WORKSTATION_USERNAME` 和可选的 `WORKSTATION_PASSWORD`。这些字段供 SSH 运维使用。调用 API 时使用应用 Bearer 凭据。

浏览器通过可信 localhost 上下文访问：

```sh
ssh -N -L 18080:127.0.0.1:8000 research-2@100.119.17.6
```

打开 `http://localhost:18080`。SSH 隧道承载页面与原生语音 WebSocket。私网 HTTPS 由 tailnet 管理员启用 Serve 后配置。[私网访问验证](../../evaluations/results/vpn-api-workstation.json)。

## 实机准入

| 验证层 | 已覆盖范围 | 证据 |
| --- | --- | --- |
| 推理功能 | 分卡引擎、Responses SSE、JSON Schema、function 输出与夹具回填、归属续接、22,753-token 多轮历史、独立准入与取消；约 20 分钟 Ava 2D／双工共驻、40 次后台请求。 | [功能](../../evaluations/results/inference-functional-workstation.json)、[断流](../../evaluations/results/responses-stream-lifecycle.json) |
| Omni 上下文 | 显式关闭 CoT 的 20 条消息、7,928-token 早期事实检索；超限输入 400。 | [8K 边界](../../evaluations/results/omni-8k-multi-turn-workstation.json) |
| 并发预算 | 四路实际生成、约 28.5K 与三路短请求混跑、独立 worker 历史及并行续接、预算拒绝与释放。 | [并发](../../evaluations/results/llm-mps-concurrency-workstation.json)、[兼容性](../../evaluations/results/responses-mps-compatibility.json) |
| 默认 CoT | 默认完整答案、显式关闭、三个文本接口的 16-token 思考预算、非法值、JSON Schema、function 归属续接。工具结果由夹具回填。 | [默认与预算](../../evaluations/results/thinking-defaults-workstation.json)、[结构化与工具](../../evaluations/results/thinking-structured-workstation.json) |

当前记录覆盖功能与短时性能。完整 262K 质量、默认 CoT 长历史、持续满载 QoS、实物播放及私网 HTTPS 按[后续计划](../plans/workstation-acceptance.md)验收。资源读数以报告的采集位置、指标和采样周期为依据。

## 其他部署目标

`--target workstation-legacy` 保留原有 `Qwen3.5-9B + MiniCPM-o-4_5` 分卡组合；`--target workstation-omni` 提供双卡 Omni 配置，`--target workstation-bf16` 提供 BF16 对照；`--target orin` 和 `--target orin-cu129` 提供 AGX 配置与兼容性验证。`workstation` 与 `workstation-legacy` 均支持 `--images` 启用共享 GPU 生图；非纯工作站目标可用 `--asr` 追加 CPU ASR worker（默认 `workstation` 已内置）。直接调用 Docker Compose 时，指定 `--project-directory <仓库根目录>` 和环境文件。AGX 细节见 [Orin](orin.md) 与 [CUDA 12.9](jetpack6-cu129.md)。


## 容量与发布后的清理

在工作站分别检查 `nvidia-smi`、`free -h`、`df -h` 和 `docker system df`。默认 `workstation` 限制为 TP=2 8 个生成序列（可调 16）与总量 `LLM_TP2_TOKEN_BUDGET=262144`、`OMNI_DUPLEX_CAP=4`；`omni-tts`（0.6B-Base）常驻 GPU 1。所有 LLM 请求按实际模板和输出额度准入，结束或取消后释放。Klein 默认在 GPU 0 上按作业取得显存，通过 `OMNI_IMAGE_LLM_SLEEP_STRATEGY=auto` 先尝试释放 TP=2 KV、不足时直接休眠卸载权重，完成后恢复后台引擎。调整任何常驻容量或上限后复核前台延迟、混跑与剩余显存。

在放大容量之前先确认实际瓶颈。发布完成后删除本次临时传输包和测试容器。核对候选镜像引用后，回收过时的 Docker 构建缓存与已退役镜像，保留当前镜像、最近回滚镜像、模型卷、业务卷和待核对应用数据。清理后重测剩余空间与已部署服务健康状态。
