# 双 RTX 3090 工作站部署

工作站是主线目标。首个候选用官方 MiniCPM-o 4.5 AWQ，Chat 的配置上限由 checkpoint 读取，为 40,960 tokens，默认生成额度 4,096，Thinker 和 Talker 启用逐 token、逐 head INT4 KV。配置上限不等于经验证的有效长文范围，当前满窗验收失败。引擎使用官方固定版 AMD64 CUDA 镜像。RTX 3090 每卡 24 GB，分别核对峰值。 [NVIDIA 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/) · [vLLM-Omni 固定发行版](https://github.com/vllm-project/vllm-omni/releases/tag/v0.32.0)

## 首套运行配置

开始前用 `nvidia-smi` 和 `nvidia-smi topo -m` 核对两张卡、互联、当前占用及支持 CUDA 13 的驱动。核对 Docker、Compose 及 NVIDIA Container Toolkit 的容器 GPU 识别。配置让 GPU 0 运行 Thinker，GPU 1 运行 Talker 和 Code2Wav，阶段内并行度沿固定版单卡设置。Avatar 加入前测量第 2 张卡的实际余量。 [阶段配置](../engine/minicpmo-workstation.yaml) · [NVIDIA CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

将 `.env.example` 复制为私有 `.env`，填写独立的网关服务令牌、网关管理令牌和模型挂载目录。为不同调用方配置各自的令牌。网关默认监听宿主回环，可通过授权网络或私有反向代理提供服务。模型端口在内部网络。

```sh
docker network inspect tuber-omni >/dev/null 2>&1 || docker network create tuber-omni
docker compose -f infra-compose.yml -f infra-workstation.compose.yml config --quiet
docker compose -f infra-compose.yml -f infra-workstation.compose.yml build omni-engine omni-gateway
docker compose -f infra-compose.yml -f infra-workstation.compose.yml --profile assets run --rm model-fetch
docker compose -f infra-compose.yml -f infra-workstation.compose.yml up -d
docker compose -f infra-compose.yml -f infra-workstation.compose.yml ps
```

## 实机准入

### Ava 2D 覆盖配置

生成独立 `OMNI_AVATAR_INTERNAL_TOKEN`，在基础配置和工作站配置后叠加 `-f infra-avatar.compose.yml`，构建并启动 `omni-avatar` 和 `omni-gateway`。Avatar 采用 Linux host 网络发布动态 ICE UDP 端口，控制 API 通过私有共享卷中的 Unix socket 与网关相连，不监听宿主 TCP 端口。主机防火墙需允许授权 LAN/VPN 客户端与 Avatar 的 ICE 候选建立 UDP 通信。不能仅用 SSH 的 TCP 转发验证远端 WHEP。

```sh
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml config --quiet
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml build omni-avatar omni-gateway
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml up -d
```

浏览器验收页由网关根路径提供。可先用 SSH 转发网关到本地 `http://localhost:8000`，媒体仍通过 LAN/VPN 直接连接主机 ICE。局域网域名访问麦克风／屏幕时需可信 HTTPS 反向代理，并转发 WebSocket；浏览器使用应用凭据，不使用管理令牌。页面提供独立 WAV／测试音源，用于在模型就绪之前检查 Ava 音画。

本次目标机可通过 `research-2@100.119.17.6` 登录，仓库位于 `/home/research-2/code/omni-avatar-api`，私有凭据保存在该目录 `.env`。可运行 `ssh -N -L 18080:127.0.0.1:8000 research-2@100.119.17.6` 后访问 `http://localhost:18080`。Tailscale 节点域名为 `belal-pc.tailb05be8.ts.net`，Serve 当前未配置；可信 HTTPS 入口仍需完成节点及 tailnet 授权和配置。

浏览器发送原始 PCM，由原生模型决定听说；当前提供手动打断。自动语音活动插话需要服务端 VAD 配置与资产，不能仅凭 `supports_barge_in` 打开 `barge_in_on_speech`，该路径尚未验收。

停止时使用相同三文件组合执行 `stop`；保留 `.env` 和模型资产。原推理组合仍可独立启动。Avatar 不占 GPU；数字人驱动须在 2D 验收后另测共驻。

### 验收顺序

1. 先验引擎和网关。用至少两个独立的调用方验证普通／流式 Chat、结构化输出、视觉、长文正确性、排队和取消。按阶段核实设备索引、AWQ 与实际 INT4 KV，记录每卡显存、主机内存、交换、吞吐和首包。
2. 用独立 Realtime 客户端验证语音与视觉输入、重叠语音、打断、字幕、静默与重连。测一路原生双工加另一个客户端普通请求持续运行，确认会话释放和引擎可用。
3. 接 Avatar 会话 API，验同步音画、打断、权限隔离和资源回收。合成节目、WHEP 外部取流和 OBS 在核心服务之后单独验收。 [接口边界](api.md)

固定引擎镜像构建和容器两卡识别已通过。Ava 2D 独立运行 60 秒，收到 3,004 个音频帧、1,503 个视频帧与非静音音频，跨凭据隔离及幂等释放通过。[Ava 实机报告](../evaluations/results/avatar-2d-workstation.json)。

普通／流式 Chat、与提示一致的 JSON Schema、视觉颜色、排队超时及断开取消通过。一路原生双工及 Ava 的 120 秒短检查收到 6,004 个音频帧、3,004 个视频帧和 101 个原生音频块，语音算术问题回答正确；第二调用方每三十秒请求一次普通推理，四次均正确。结束后网关回到 batch，活动和排队计数归零，后续 Chat 可用。[短混合检查报告](../evaluations/results/native-avatar-workstation.json)。该报告验证轻量后台流量，不代表两小时持续负载或真实扬声器／显示器播放验收。

当前近 4 万 tokens 的长文检索失败：原配置空答或产生无关重复文本。提高生成预算、启用思考、关闭异步调度或整段预填充均未恢复答案。自动 KV 精度的早期对照在 CUDA 等待期间被手动终止，没有正确性结论；等待栈不能证明死锁，且双工折算的 token 容量不能直接当作普通长文的物理 KV 容量。未采用这些诊断覆盖作为主线。[原配置失败证据](../evaluations/results/omni-workstation-int4.json) · [诊断结果](../evaluations/results/omni-workstation-kv-probes.json)。

用户已确认继续评估官方非 AWQ BF16 权重。候选覆盖为 `infra-workstation-bf16.compose.yml`，应叠加在 `infra-workstation.compose.yml` 之后、`infra-avatar.compose.yml` 之前。它同时固定下载器、引擎和网关的模型身份为 `openbmb/MiniCPM-o-4_5@fb09628e98b1f55d8d5b863e0eccfcdd11bff499`，明确校验非量化 BF16 配置；沿用原窗口、阶段分工和 INT4 KV，只增加 GPU 0 的内存预算。使用相同文件组合执行 `config --quiet`、`build omni-engine`、`--profile assets run --rm model-fetch` 和 `up -d`。候选尚未通过长文和共驻验收，不作为已验收默认值。回退时去掉 BF16 覆盖并以原三文件组合执行 `up -d`；两个快照各自保留。

BF16 候选已通过普通／流式、结构化和视觉检查，但在相同 INT4 KV 下，39,932-token 检索仍为空答。[BF16 报告](../evaluations/results/omni-workstation-bf16.json)。INT8 KV 配 4 GiB 物理缓存，在 AWQ 和 BF16 权重下同样空答；AWQ 配 8 GiB 浮点 KV 也未恢复答案。[INT8 报告](../evaluations/results/omni-workstation-int8-kv.json) · [浮点 KV 报告](../evaluations/results/omni-workstation-float-kv.json)。直接以官方 BF16 文本骨干、Transformers SDPA 和 BF16 KV 绕过 Omni 服务，仍复现同一用例的立即结束符；这不支持继续仅靠量化调参交付满窗。[参考报告](../evaluations/results/omni-workstation-text-reference.json)。

维护者说明当前版本推荐的最大多模态输入为 8K，超过 12K 可能空答或重复；40,960 的配置上限不能作为该质量保证。[官方说明](https://github.com/OpenBMB/MiniCPM-V/issues/1131#issuecomment-5238729878)。用户要求的可靠满窗仍未满足，没有调低验收要求。原 AWQ 三文件组合已恢复，服务健康；长上下文与实时历史的方案选择及设备预算见[当前计划](plans/avatar-workstation.md#上下文调研与待选方案)，尚未更换模型或加入路由。

因此整体验收未通过；两小时负载、浏览器麦克风／屏幕的完整交互、插话与原生恢复、客户端渲染故障降级、可信 HTTPS 入口及数字人阶段仍待完成。后续推理基线选择见[当前计划](plans/avatar-workstation.md)。早期 Jetson 结果见[历史 AGX 证据](https://github.com/4everWZ/Tuber_Studio/blob/main/evaluations/results/omni-deployment-20261116.md)。
