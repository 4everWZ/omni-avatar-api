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

## Omni 实时窗口选择

默认继续采用官方服务的 `basic`，水位 8000／6000，符合约 8K 近期助手＋外部记忆的分工。外部应用可在创建会话时通过现有 `session.update.session.extra_body` 选择模式及参数；先用 `autostart=0` 连接，不需要修改部署文件或重启模型。`context` 默认只保留 24 个近期输入单元与最多 500 tokens 的早期模型回复，需要重建窗口，不等同于保留更长的完整对话。[API 示例与限制](../api.md#omni-实时历史窗口)。

## 四路并发与计算预算

当前使用 `axteam/Qwen3.5-9B-AWQ-4bit@5577b5b71e6f06b9f6a1f3743ef6ba6dc4ecd2df`，实际格式为 compressed-tensors W4A16、非对称 group 32。保留 9 GiB BF16 KV、FP32 SSM 和原卡位：GPU 0 为 LLM 与 Ditto，GPU 1 为完整 Omni。`LLM_MAX_NUM_SEQS=4` 同时控制引擎和网关；活跃请求共享 `LLM_TOKEN_BUDGET=262144`，按完整输入加最大输出预留，见 [API](../api.md)。长短请求按 FIFO 等待，不能把异步排队视为四路生成。

神经头像覆盖通过独立 `gpu-mps` 服务启用 GPU 0 的 CUDA MPS。LLM 使用 `LLM_MPS_ACTIVE_THREAD_PERCENTAGE=20`、较低客户端优先级；Ditto 使用正常优先级。20% 是线程资源上限，并非独占 20% 执行时间；实机 LLM 可见 16 个 SM。LLM 启用 `FULL_DECODE_ONLY` CUDA Graph，捕获批次 1、2、4，并关闭异步调度。已移除固定 250 ms 等待及 `LLM_GPU_STEP_PAUSE_MS`。当前引擎在派发前读取真实渲染帧截止期；余量不足 240 ms 时单次主动让出不超过 18 ms，并在新帧补足余量后提前恢复。SSE 即时转发实际生成。调整 MPS 上限需要重验头像时间线；token 预算保护显存，两者分别生效。

当前 CoT 开启，思考预算未增加自定义上限；温度 0，每路总输出上限 256 tokens，先预热 32 tokens。固定同一整数短提示，按模板实报输入为 Omni 25、LLM 27 tokens，关闭前缀缓存。表内均为端到端思考＋正文总生成 TPS；通话状态保留两轮测量范围。

| 服务 | 生成请求数 | 其他负载 | 每路总生成 TPS | 合计 TPS |
| --- | --- | --- | --- | --- |
| Omni Chat | 1 | 空闲 | 65.54 | 65.54 |
| Omni Chat | 1 | 原生语音＋头像＋四路 LLM | 32.00–36.55 | 32.00–36.55 |
| LLM | 1 | 空闲 | 47.25 | 47.25 |
| LLM | 4 | 空闲 | 37.12–37.26 | 148.48 |
| LLM | 1 | 原生语音＋头像 | 18.33–23.53 | 18.33–23.53 |
| LLM | 4 | 原生语音＋头像 | 15.37–19.12 | 61.46–76.11 |
| LLM | 4 | 原生语音＋头像＋一路 Omni Chat | 13.49–16.66 | 53.97–66.32 |

前台普通 Chat 默认只有一个准入槽。两个同时提交的短请求实测每路 65.45／32.56 TPS（后者含排队等待），合计 65.13 TPS，输出区间未重叠；不宣称双路实际生成。后台四路要求输出区间真实重叠。

每路 TPS = `usage.completion_tokens`（Omni Chat）或 `usage.output_tokens`（Responses）÷ 从本组请求发起到该路完成的秒数；合计 TPS = 本组全部生成 tokens ÷ 最晚完成时间。均包含思考、预填充、传输和准入／GPU 等待，不以 SSE 事件数代替 token 数。原始记录另存 `(tokens - 1) / (末增量 - 首增量)` 的生成区间参考值；受分块与不可见控制 tokens 影响，不当作精确逐 token 延迟。

256-token 性能夹具中，部分请求会在思考阶段耗尽额度，Omni 本组未进入正文。因此表格不能当作正文吞吐或完整答复时延；思考与正文首增量时间分别记录。默认 CoT 完整答案由独立功能夹具验证。[前台与全栈混跑](../../evaluations/results/omni-throughput-final-workstation.json) · [后台单路／四路](../../evaluations/results/llm-throughput-cot-workstation.json)。后台验证器的空闲单路 45 TPS、原生语音＋头像四路每路 15 TPS 是该负载的防退化检查线；额外叠加 Omni Chat 的负载单独报告，不借用该通过结论。

Ditto 保留现有 FP16、十步采样、RNG、分辨率与 1200 ms 音频输入预留。全栈 90 秒生成 2,263 张真实神经帧，最大检查期采样时间线偏差 84 ms；后台单独共驻的 90 秒生成 2,267 帧、偏差 99 ms，均通过 25 fps／160 ms 检查、原生取消恢复及幂等释放。原关闭 CoT、没有截止期反馈的额外 Chat 混跑曾出现 203 ms 偏差，已作为失败基线保留。[原混跑基线](../../evaluations/results/omni-throughput-basic-workstation.json)。

CoT 默认开启、显式关闭、三个文本接口的 16-token 思考预算与非法预算拒绝通过；JSON Schema、function 调用及归属续接也在默认 CoT 下复验通过。[默认与预算](../../evaluations/results/thinking-defaults-workstation.json) · [结构化与工具续接](../../evaluations/results/thinking-structured-workstation.json)。这些工具仅回填字面量测试结果，不执行 Agent 或真实外部工具。Omni 既有 7,928-token 检索与后台 22,753-token 多轮历史报告使用显式关闭 CoT 的夹具；不将其误记为默认 CoT 的完整窗口质量验收。

旧固定 250 ms／约 3.4 TPS 方案及最初无反馈、关闭 CoT 的 MPS 吞吐报告只作为历史基线保留，不是当前配置性能。完整 262K 质量、四个满窗口、持续满载 SLA 与实物播放仍按各自范围另行验收。

### 渲染截止期反馈

MPS 仍允许四个序列真实同批生成。为覆盖额外一路 Omni Chat 的更重混跑，头像在已有 MPS 控制卷原子发布最近神经帧的单调时钟截止期；LLM 在真实 GPU 批次派发前据此短暂让出算力。反馈不含图像、文本或调用方身份，不增加公共 API 或跨服务 Python 导入。没有活跃头像、文件缺失／无效或超过 500 ms 未更新时不主动等待；会话释放删除自己的时钟文件，崩溃后旧记录自动失效。每次主动等待上限 18 ms，渲染进度追上则提前结束。

这与 token 显存准入分别生效；保留 262144 单请求上限、四序列、W4A16、9 GiB BF16 KV、FP32 SSM、十步运动采样和 1200 ms 音频输入预留。外部 CoT 默认及显式预算见 [API](../api.md#cot-默认与调用方预算)。旧关闭 CoT、没有截止期反馈的吞吐报告属于当时配置，不能当作开启 CoT 后的正文输出速度。

### MPS 生命周期

仅工作站神经头像覆盖启用 MPS；不启动宿主级守护进程，不修改宿主 compute mode。`gpu-mps` 使用镜像内的 NVIDIA 控制程序，以多用户模式运行、无网络；LLM 与 Ditto 共享该服务的 PID／IPC 命名空间及控制 socket 卷。两个客户端均依赖控制服务健康。GPU 1 的 Omni 不加入 MPS。参见 [NVIDIA MPS 文档](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html)。

控制服务需要重建时，先停客户端，再依次恢复，避免客户端持有已失效的命名空间：

```sh
python tools/compose.py --portrait --vpn -- stop llm-engine omni-portrait
python tools/compose.py --portrait --vpn -- up -d --wait --no-deps --force-recreate gpu-mps
python tools/compose.py --portrait --vpn -- up -d --wait llm-engine omni-portrait omni-gateway
```

凭据保持不变，准备好的 avatar_id 保存在卷中；引擎 RAM 内的 Responses 历史在重启后失效，调用方应保存业务历史。实机重建 MPS 与客户端后，复用已准备形象的 30 秒四路／25 fps、音画时间线及取消恢复检查通过。[重启复用](../../evaluations/results/mps-lifecycle-workstation.json) · [当前部署、官方默认预算和空闲恢复](../../evaluations/results/runtime-cot-deadline-workstation.json)。

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
