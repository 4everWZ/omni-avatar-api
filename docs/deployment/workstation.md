# 双 RTX 3090 工作站部署

当前目标是推理服务分工：GPU 0 承载后台长上下文 LLM，GPU 1 承载实时 Omni 的 Thinker、Talker、Code2Wav；Avatar 也在这台工作站，2D 当前使用 CPU。模型与 Avatar 均暴露 API，业务方持有多轮历史、记忆、Agent 调度、工具授权和执行。工作站服务不替调用方实现这些业务逻辑。

## 主线组合

叠加 `deploy/compose/base.yml`、`deploy/compose/workstation.yml`、`deploy/compose/avatar.yml`、`deploy/compose/inference.yml`。前三文件组合保留为旧两卡 Omni 复现配置。基础 ARM64 与 BF16 候选文件不作为新的主线启动组合。

```sh
python tools/compose.py -- config --quiet
python tools/compose.py -- build omni-engine omni-gateway omni-avatar
python tools/compose.py -- --profile assets run --rm model-fetch
python tools/compose.py -- --profile assets run --rm llm-model-fetch
python tools/compose.py -- up -d
```

模型资产使用精确快照和独立目录，服务离线加载；两个引擎端口均在 Docker 私网。宿主默认只发布回环 8000。`.env` 中配置独立的应用、管理、Avatar 内部令牌，不打印或提交凭据。

## API 与上下文

| 入口 | 责任 |
| --- | --- |
| `/v1`，`omni-default` | 语音／视觉助手、普通和流式 Chat、原生 Realtime；Chat 窗口 8192，Realtime 默认 basic 水位 8000／6000。 |
| `/llm/v1`，`llm-default` | 以 Responses HTTP／增量 SSE 为主的长上下文推理，兼容 Chat；结构化回复与工具调用输出。固定 9B W4A16，配置窗口 262144，已验证 22,753-token 多轮续接。 |
| `/healthz`、`/llm/healthz` | 分别报告两个后端的身份、根目录校验、就绪和独立准入；一个后端异常不代替另一个的健康结论。 |
| `/v1/avatar/*` | 同宿主 Ava 渲染、音频源绑定、WHEP、字幕、代次和会话生命周期。 |

Chat 无跨请求业务记忆：应用每次提交希望使用的 messages；Realtime 由原生引擎保留活跃近期历史并按策略滑动。调用方可将外部记忆选入提示。服务不提供记忆库、自动摘要、Agent 循环或工具执行。两个模型通过不同前缀显式调用，网关不根据输入猜测或自动改用另一个模型。

GPU 设备通过容器设备请求隔离；每个容器只见自己的物理卡，因此阶段配置均使用本地设备 0。后台 LLM 保留单请求 262144 上限，默认最多 4 个生成序列，共享 262144 输入加最大输出的活跃 token 预算；具体排队与预留语义见 [API](../api.md)。KV 保留浮点，Gated DeltaNet 的 SSM 状态使用 FP32，不与权重一起量化。Omni 准入独立于 LLM；四路满窗口不在现有显存保证范围。

## 四路并发与计算预算

当前使用 `axteam/Qwen3.5-9B-AWQ-4bit@5577b5b71e6f06b9f6a1f3743ef6ba6dc4ecd2df`，实际格式为 compressed-tensors W4A16、非对称 group 32。保留 9 GiB BF16 KV、FP32 SSM 和原卡位：GPU 0 为 LLM 与 Ditto，GPU 1 为完整 Omni。`LLM_MAX_NUM_SEQS=4` 同时控制引擎和网关；活跃请求共享 `LLM_TOKEN_BUDGET=262144`，按完整输入加最大输出预留，见 [API](../api.md)。长短请求按 FIFO 等待，不能把异步排队视为四路生成。

神经头像覆盖通过独立 `gpu-mps` 服务启用 GPU 0 的 CUDA MPS。LLM 使用 `LLM_MPS_ACTIVE_THREAD_PERCENTAGE=20`、较低客户端优先级；Ditto 使用正常优先级。20% 是线程资源上限，并非独占 20% 执行时间；实机 LLM 可见 16 个 SM。LLM 启用 `FULL_DECODE_ONLY` CUDA Graph，捕获批次 1、2、4，并关闭异步调度。已移除 Core 固定等待补丁及 `LLM_GPU_STEP_PAUSE_MS`，SSE 即时转发实际生成。调整 MPS 上限需要重验头像时间线；token 预算保护显存，两者分别生效。

当前端到端输出 TPS 如下：29-token 短提示，每路 256 输出 tokens，温度 0、关闭 thinking、关闭前缀缓存；先预热 32 tokens，头像服务始终共驻。通话状态的两轮结果保留为范围。

| 并发请求数 | 头像状态 | 每路 TPS（输出 tokens/s） | 总 TPS（输出 tokens/s） |
| --- | --- | --- | --- |
| 1 | 服务共驻、空闲；无原生语音会话 | 47.18 | 47.18 |
| 4 | 服务共驻、空闲；无原生语音会话 | 37.12–37.27 | 148.47 |
| 1 | 原生语音与神经头像 25 fps 通话中 | 22.92–23.90 | 22.92–23.90 |
| 4 | 原生语音与神经头像 25 fps 通话中 | 18.68–19.42 | 74.73–77.34 |

口径：每路 TPS = `usage.output_tokens` ÷ 从本组请求发起到该路 `response.completed` 的秒数；总 TPS = 本组全部输出 tokens ÷ 最晚完成时间，包含预填充及传输等待，不用 SSE 事件数代替 token 数。生成区间另按 `(输出 tokens - 1) / (末增量时间 - 首增量时间)` 计算，空闲单路为 47.75 TPS。原始报告保留两种口径及首包时间；不将短输入吞吐外推到长上下文或持续满载。[吞吐与共驻](../../evaluations/results/throughput-mps-workstation.json)。工具默认的单路 45 TPS／通话四路每路 15 TPS 是防退化检查线，不代表用户确认了长期 SLA。

Ditto 运动预测维持有数值漂移检查的 FP16、十步采样、RNG、分辨率及原参考图接口。90 秒生成 2,269 张真实神经帧，接收 2,306 视频／4,610 音频帧及 36 个原生语音块；检查期最大采样时间线偏差 79 ms，原生取消和恢复通过。报告末尾取消恢复期间的样本独立于检查期偏差统计。这不保证四个满窗口同时推理，也不替代完整 262K 质量验收或长期满载 SLA。

同一配置的引擎指标采到 4 个同时运行序列；28,528-token 输入与三路短请求混跑、四个 worker 独立保存与并行续接通过。超出剩余 token 预算时，即使序列槽未满也排队超时 429；剩余预算可接短 Chat，取消后预留归零。[并发与预算](../../evaluations/results/llm-mps-concurrency-workstation.json)。更换 4bit 权重后，Responses 续接、JSON Schema、function 往返、增量 SSE、22,753-token 多轮检索与 Omni 8K 边界均复验通过。[兼容性](../../evaluations/results/responses-mps-compatibility.json)。旧 250 ms 等待方案的约 3.4 TPS 已被用户否决；原报告仅保留为撤回方案的历史基线，不是当前交付性能。[原单路基线](../../evaluations/results/single-request-tps-sleep-baseline.json)。

### MPS 生命周期

仅工作站神经头像覆盖启用 MPS；不启动宿主级守护进程，不修改宿主 compute mode。`gpu-mps` 使用镜像内的 NVIDIA 控制程序，以多用户模式运行、无网络；LLM 与 Ditto 共享该服务的 PID／IPC 命名空间及控制 socket 卷。两个客户端均依赖控制服务健康。GPU 1 的 Omni 不加入 MPS。参见 [NVIDIA MPS 文档](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html)。

控制服务需要重建时，先停客户端，再依次恢复，避免客户端持有已失效的命名空间：

```sh
python tools/compose.py --portrait --vpn -- stop llm-engine omni-portrait
python tools/compose.py --portrait --vpn -- up -d --wait --no-deps --force-recreate gpu-mps
python tools/compose.py --portrait --vpn -- up -d --wait llm-engine omni-portrait omni-gateway
```

凭据保持不变，准备好的 avatar_id 保存在卷中；引擎 RAM 内的 Responses 历史在重启后失效，调用方应保存业务历史。实机重建 MPS 与客户端后，复用已准备形象的 30 秒四路／25 fps、音画时间线及取消恢复检查通过。[重启复用](../../evaluations/results/mps-lifecycle-workstation.json) · [无固定等待及实际 MPS 资源检查](../../evaluations/results/mps-runtime-workstation.json)。

## Avatar 与访问

Avatar 使用 host 网络发布 ICE UDP；控制 API 走私有 Unix socket，共享卷只向网关开放。WHEP 媒体需 LAN／VPN 可达，单纯 SSH TCP 转发不转发媒体。浏览器验收页仍在网关根路径。

目标 `research-2@100.119.17.6`，远端仓库 `/home/research-2/code/omni-avatar-api`。可用 `ssh -N -L 18080:127.0.0.1:8000 research-2@100.119.17.6` 后访问 `http://localhost:18080`；媒体经 VPN 直连。Tailscale Serve 尚未配置，可信 HTTPS 入口待验收。

## 实机准入

分卡引擎、Responses 流式、JSON Schema、工具调用输出、归属续接、多轮事实保留、独立准入、取消及断流释放通过。原生双工及 Ava 2D 与后台 LLM 连续共驻约 20 分钟，完成 40 次后台请求。按用户要求以约半小时功能检查替代两小时负载。[功能报告](../../evaluations/results/inference-functional-workstation.json) · [断流生命周期](../../evaluations/results/responses-stream-lifecycle.json)。报告包含媒体解码与主机内存；本次运行容器未提供 nvidia-smi，因此未记录 GPU 峰值。LLM 的完整 262144 窗口与浏览器实物播放仍待验证。神经头像结果见下节。

旧两卡布局的短 Chat、结构化、视觉与 Avatar 120 秒轻量混合检查通过。[旧配置报告](../../evaluations/results/native-avatar-workstation.json)。独立 Ava 2D 的音画、权限与幂等释放通过。[Ava 报告](../../evaluations/results/avatar-2d-workstation.json)。新分工需重验受资源配置影响的部分；解码帧不等于真实扬声器／显示器播放。

旧近 40K Omni 检索在 AWQ、BF16、INT8／浮点 KV 和直接文本骨干参考中未通过。维护者建议当前版本多模态输入保持 8K。[官方边界](https://github.com/OpenBMB/MiniCPM-V/issues/1131#issuecomment-5238729878)。用户已明确采用近期约 8K 的多轮实时助手加外部记忆，长上下文交给独立 LLM；不再把一次性输入 40K 文本作为 Omni 准入要求。后台 LLM 的多轮历史和长上下文仍须独立验证。[当前计划](../plans/avatar-workstation.md)。
## 会议式神经头像

在上述四文件主线后叠加 `deploy/compose/portrait.yml`。`omni-avatar` 继续负责媒体与会话；新增 `omni-portrait` 在 GPU 0 与 LLM 共驻，通过私有 Unix socket 接收音频并回传帧。服务没有网络接口，模型只读挂载；参考图、准备特征和编译缓存使用独立持久卷。

```sh
python tools/compose.py --portrait -- build omni-portrait omni-avatar omni-gateway
python tools/compose.py --portrait -- --profile assets run --rm portrait-assets
python tools/compose.py --portrait -- up -d --wait
```

模型固定 Ditto PyTorch，启动时完成编译与 Graph 捕获；会话就绪前预热实际工作线程并填充初始视频缓冲。目标为 640×720、25 fps，音频输入预留 1200 ms；不改变 Omni／LLM 的模型、窗口或 KV 配置。参考图准备和通话共用一路神经 GPU 配额，其他 2D 会话保持独立。在线接口见 [API](../api.md#参考图与会议式头像)，驱动和生命周期见 [portrait](../../services/portrait/README.md)。

25 fps 会议式 Ava 已实测通过：120 秒接收 3,004 视频帧、6,004 音频帧；3,012 张实际神经生成帧，最大音画时间线采样偏差 58 ms；原生语音持续输出，完成 4 次后台 Responses 请求。GPU 显存采样最大值分别为 23,101／21,468 MiB；这是采样读数，不是完整高频峰值或持续满载保证。重启后复用、取消代次与恢复也通过。[神经头像](../../evaluations/results/portrait-functional-workstation.json) · [生命周期](../../evaluations/results/portrait-lifecycle-workstation.json)。

Tailscale Serve 当前未启用，tailnet 管理员需先在控制台开放该功能才能配置私网 HTTPS。浏览器可使用上述 localhost SSH 隧道的可信上下文；WHEP 媒体继续经 VPN 直连。

## 私网共享 API

需要其他应用直接访问时，再叠加 `deploy/compose/vpn.yml`，并在 `.env` 设置 `OMNI_GATEWAY_VPN_BIND` 为宿主的 VPN 地址。该覆盖保留回环监听，并增加指定私网地址的 8000 端口；模型引擎与神经工作器仍无宿主 TCP 入口。当前工作站地址为 `100.119.17.6`，API 基址分别是 `http://100.119.17.6:8000/v1` 和 `http://100.119.17.6:8000/llm/v1`，沿用各应用独立 Bearer。

```sh
python tools/compose.py --portrait --vpn -- up -d --wait omni-gateway
```

浏览器麦克风／屏幕仍使用 `http://localhost:18080` 的 SSH 隧道，或由 tailnet 管理员开启 Serve 后配置 HTTPS；直接私网 HTTP 地址供 API 客户端调用。

已从另一 VPN 客户端验证两个模型卡、Responses 增量 SSE、准备形象查询及匿名拒绝，回环入口继续保留。[访问证据](../../evaluations/results/vpn-api-workstation.json)。

Omni 的 8K 多轮边界已验证：20 条消息、7,928 个输入 tokens 能找回早期事实；9,975-token 输入返回 400。固定引擎适配修正了上游将 `VLLMValidationError` 包装成 500 的行为，其他服务错误保持原状态。[上下文边界](../../evaluations/results/omni-8k-multi-turn-workstation.json)。

## 从平铺目录迁移

原根目录的 `infra-*.yml` 已收拢到 `deploy/compose/`；服务目录位于 `services/`。使用仓库根目录的统一入口，避免 Compose 默认以配置文件所在目录解析绑定路径：

```sh
python tools/compose.py --portrait --vpn -- config --quiet
python tools/compose.py --portrait --vpn -- build omni-engine omni-gateway omni-avatar omni-portrait
python tools/compose.py --portrait --vpn -- up -d --wait
```

保留根目录 `.env`、`data/` 和现有 Docker named volumes，项目名继续使用 `tuber-omni-infra`。这些本地资产不属于源码目录移动。重建并更新容器会结束活跃会话，进程内 Responses 续接也会失效；更新前先让调用方排空请求。

其他可用配置：`--target workstation-omni` 复现原两卡 Omni；`--target workstation-bf16` 用于 BF16 对照；`--target orin` 和 `--target orin-cu129` 对应历史 Orin 路径。`--portrait` 仅适用于当前分卡 `workstation` 目标。直接调用 Docker Compose 时，务必使用 `--project-directory <仓库根目录>` 并显式指定环境文件。
