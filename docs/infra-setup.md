# AGX 基础设施预备与运行

`infra-compose.yml` 启动 MiniCPM-o 4.5、共享推理网关与 MediaMTX。业务 Compose 可独立启动或重启。两者通过外部 Docker 网络 `tuber-omni` 相连。应用访问 `http://omni-gateway:8000`，无权直连 `omni-engine:8000`。网关公开 `omni-default`，首版提供 Chat Completions、Realtime 和模型列表，健康状态同时显示部署锁定的版本与上游探测结果。

## 固定基线

| 组件 | 首测来源 |
| --- | --- |
| AGX | NVIDIA 的 JetPack 7.2／R39.2 路径提供 SBSA 环境并展示了普通 vLLM 0.22 在 Orin SM 8.7 上运行。当前生产 R36.4 镜像是回退对照。用独立可恢复启动环境验证候选镜像。 [NVIDIA 设备示例](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-linux-r39-2-on-jetson-agx-orin-developer-kit-getting-started-and-feedback-thread/372156) |
| vLLM-Omni | `v0.32.0-aarch64` 镜像锁定 ARM64 摘要，包含 CUDA 13.0.2。项目镜像按上游三阶段解码器的实际导入补齐固定的 CosyVoice2 依赖。Docker 构建先检查 Python 导入。随后分别在 AGX 验证 CUDA、Thinker、Talker 与 Code2Wav。 [发布与架构](https://github.com/vllm-project/vllm-omni/releases/tag/v0.32.0) · [解码器源码](https://github.com/vllm-project/vllm-omni/blob/v0.32.0/vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_token2wav.py) |
| MiniCPM-o 4.5 AWQ | 固定 OpenBMB 官方版本 `927faefa0bce96e557945e106f3b3d8b4d39f4b2`。只读挂载完整模型、远端模型代码、声码器资产和参考音频。Thinker 上下文从 checkpoint 读取 40,960，网关透传实报 `max_model_len`。Thinker 和 Talker 分配逐 token、逐 head 的 4-bit KV。 [模型快照](https://huggingface.co/openbmb/MiniCPM-o-4_5-awq/tree/927faefa0bce96e557945e106f3b3d8b4d39f4b2) · [上游三阶段配置](https://github.com/vllm-project/vllm-omni/blob/v0.32.0/vllm_omni/deploy/minicpmo_4_5.yaml) |
| MediaMTX | 固定 1.21.4。首版用内部 RTSP/TCP 摄取，使用 WHEP 播放。发布和读取分别授权在 `program` 路径。 [MediaMTX 配置](https://github.com/bluenviron/mediamtx/blob/v1.21.4/mediamtx.yml) |

上游 Docker 标签和 checkpoint 已核对，AGX 上的 Omni 镜像构建、音视频推理、共驻容量和时延仍待实机验收。若目标机装载失败，先记录 JetPack／L4T、驱动、CUDA 版本、SM、内核报错和配置。随后用适配的 ARM64 CUDA 发行版根据实际缺失内核构建对应版本，或保留现有推理档。镜像为 ARM64 只说明 CPU 架构。

现有 R36.4.7 设备可试用单独的 [JetPack 6 CUDA 12.9 覆盖配置](jetpack6-cu129.md)。GPU 与音视频能力按该路径重新验收。

## 启动

AGX 需 Linux ARM64、Docker Engine、Compose 2.30 及以上与 NVIDIA Container Toolkit。先核对系统、Docker 与 GPU 运行时。

```sh
uname -m
cat /etc/nv_tegra_release
docker compose version
docker info --format '{{json .Runtimes}}'
```

1. 执行 `umask 077`，将 `.env.example` 复制为 `.env`。分别设置 `OMNI_GATEWAY_TOKEN`、`OMNI_GATEWAY_ADMIN_TOKEN`、`MEDIAMTX_PUBLISH_PASSWORD`、`MEDIAMTX_READ_PASSWORD`。用 `openssl rand -hex 32` 生成独立值。各额外应用可通过 `OMNI_GATEWAY_TOKENS` 配置独立令牌。
2. 依据真实 NVMe 挂载点设置 `OMNI_MODEL_DATA`，创建对应目录。默认 `./data/models`。首次下载前先核对挂载和剩余空间。
3. 创建共享网络并构建固定基础镜像，下载 checkpoint 后启动。

```sh
docker network inspect tuber-omni >/dev/null 2>&1 || docker network create tuber-omni
docker compose -f infra-compose.yml --profile assets config --quiet
docker compose -f infra-compose.yml build omni-engine omni-gateway
docker compose -f infra-compose.yml --profile assets run --rm model-fetch
docker compose -f infra-compose.yml up -d
docker compose -f infra-compose.yml ps
```

默认由 `model-fetch` 的下载网络获取模型。Docker 桥接无公网出口的 AGX 可让一次性的下载容器使用宿主网络，沿用同一模型版本、NVMe 挂载和快照校验。`omni-engine` 在内部网络读取已完成的快照。模型缓存留在宿主，业务数据由应用 Compose 持有。

网关默认在宿主回环绑定 8000。验证时可用 SSH 转发到其回环端口，或配置受控 LAN／VPN 的绑定地址与防火墙。内网应用容器始终使用 Docker 服务名。MediaMTX 用独立的播放凭据授权 WHEP。信令默认宿主回环 8889，ICE UDP 默认宿主回环 8189。远端浏览器或 OBS 使用时，将 `MEDIAMTX_ICE_BIND` 和 `MEDIAMTX_ICE_HOST` 指向实际可达的内网／VPN 接口，开放该接口的 8189/UDP。普通 SSH 端口转发只转 TCP 信令。网页麦克风与屏幕采集在远端使用 HTTPS，HTTPS 反向代理可后续把工作台与播放入口挂到同一可信域。

| 用途 | 内部地址与凭据 |
| --- | --- |
| 网关业务使用 | `http://omni-gateway:8000/v1`，Bearer 对应应用令牌，模型 `omni-default` |
| 网关模式控制 | `http://omni-gateway:8000/api/mode`，管理令牌应用在服务端，由直播会话切换 |
| 节目摄取 | `rtsp://mediamtx:8554/program`，RTSP/TCP，`tuber-publisher` 与 `MEDIAMTX_PUBLISH_PASSWORD` |
| 节目读取 | `http://mediamtx:8889/program/whep`，播放器使用 `tuber-viewer` 与 `MEDIAMTX_READ_PASSWORD` |

## 核验与回退

- 用应用凭据检查 `/healthz` 和 `/v1/models`，发一条 `model=omni-default` 的 Chat Completions，并测一次同一别名的 Realtime 会话。从 `session.created` 读取 `session.capabilities`。再测图像、PCM 音频实际回流、字幕、中断和已播回执，分别保存结果。 [双工接口](https://docs.vllm.ai/projects/vllm-omni/en/v0.32.0/serving/realtime_duplex_api/)
- Omni 0.32.0 接收 Chat Completions 的 `response_format`，部署镜像补上从已解析协议到 Thinker 采样参数的转交。升级时复验流式与普通响应的 JSON Schema，以及非法 schema 的错误返回。 [锁定版采样实现](https://github.com/vllm-project/vllm-omni/blob/5917730a4b02aedb44d374173cdc1b10484ccd52/vllm_omni/entrypoints/openai/serving_chat.py#L1423)
- `engine/minicpmo-orin.yaml` 继承镜像的三阶段配置，直接用官方 40,960 窗口与 4-bit KV 验收一条原生双工和普通 Chat 的共驻。整机目标为 Omni 30 GiB、Avatar 6 GiB、约 8 GiB 可用余量。按阶段记录实际权重、KV、工作区及整机指标。当前网关 Live 模式的普通并发为 1。提高到 2 前，验收缓存、设备余量和两小时直播混合负载。
- 验证 WHEP 播放客户端和 OBS 的实际音视频、ICE 与重连。独立确认只读用户无法发布、发布用户无法拉流。媒体确认与推理健康分别验收。
- 引擎配置、资产路径或依赖更新后先独立验证镜像与模型。停止候选时使用 `docker compose -f infra-compose.yml stop`，保留模型资产及业务 Compose 持久卷。回退切应用 Provider／媒体路由到实际已验的 9B 与 Genie 配置。
