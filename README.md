# Tuber Studio Omni Deploy

独立部署模型推理与共享 API 网关。Tuber Studio 的常驻 Agent、视频取材、运营和其他客户端通过同一网关复用模型。任务、记忆、平台账号和工具执行由相应业务服务持有。 [Tuber Studio](https://github.com/4everWZ/Tuber_Studio)

| 入口 | 用途 |
| --- | --- |
| `infra-compose.yml` | 模型获取、现有 ARM64 推理配置和鉴权网关。默认只发布网关，模型端口留在内部网络。 |
| `infra-cu129.compose.yml` | JetPack 6 的 CUDA 12.9 兼容配置。 [运行说明](docs/jetpack6-cu129.md) |
| `infra-media.compose.yml` | 可选的 MediaMTX 节目传输，在验收核心服务后再启用。 |

## 共享 API

应用使用 `/v1` 和稳定模型别名 `omni-default`。独立调用方使用各自的 Bearer 凭据。先用 `GET /healthz` 和 `GET /v1/models` 核对模型、版本、实际窗口及能力，再走 `POST /v1/chat/completions`。普通响应和 SSE 共用接口。原生双工由 `/v1/realtime?duplex=1` 提供，会话能力以握手回包为准。完成推理接口验收后再接业务。 [网关接口及配置](gateway/README.md)

## 部署状态

现有配置来自 AGX 的 ARM64 候选。官方 MiniCPM-o 4.5 AWQ、40,960 Chat 窗口及 Thinker／Talker 的 INT4 KV 有实机记录。长文答复和两小时混合负载仍有故障。 [AGX 运行说明](docs/infra-setup.md) · [AGX 实测证据](https://github.com/4everWZ/Tuber_Studio/blob/main/evaluations/results/omni-deployment-20261116.md)

下一阶段为计划中的 2× RTX 3090 工作站建立独立运行配置。工作站环境、GPU 分配和吞吐须到机后核验。验收用 Core 和一个独立调用方测试同一 API 的视觉、结构化输出、长文、并发、排队和取消，记录逐卡显存、整机内存和任务耗时。Realtime 先用独立客户端验收，OBS 和节目媒体排在后面。 [实施交接](https://github.com/4everWZ/Tuber_Studio/blob/main/docs/handoff.md)
