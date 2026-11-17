# Omni + Avatar API

面向多个应用的原生双工 Omni 和 Avatar 音画 API。现有共享网关覆盖普通推理、视觉和实时会话。目标接口允许应用为实时会话订阅同步的 Avatar 音画。各应用自行持有用户、记忆、任务和业务工具。

**主部署目标是双 RTX 3090 的 Linux x86-64 工作站。** 主线叠加 `infra-compose.yml`、`infra-workstation.compose.yml`、`infra-avatar.compose.yml` 和 `infra-inference.compose.yml`，将实时 Omni 与后台长上下文 LLM 分卡部署。 [工作站部署](docs/workstation.md) · [服务接口](docs/api.md)

| 入口 | 用途 |
| --- | --- |
| 上述四文件组合 | 工作站主线：GPU 1 实时 Omni，GPU 0 长上下文 LLM，同宿主 Ava 与鉴权网关。 |
| `infra-media.compose.yml` | 可选节目传输。仅在需要节目媒体时叠加。 |
| AGX 历史配置 | 基础文件的 ARM64 配置与 CUDA 12.9 覆盖用于历史复现。 [AGX 指南](docs/infra-setup.md) · [JetPack 6 指南](docs/jetpack6-cu129.md) |

## 共享 API

`/v1` 的 `omni-default` 提供约 8K 多轮近期窗口的 Chat 和 Realtime；`/llm/v1/responses` 的 `llm-default` 提供后台长上下文推理与增量 SSE。两者独立准入，应用持有外部记忆并执行工具调用。独立 Avatar 服务提供 Ava 2D 会话、WHEP 音画及浏览器页。数字人阶段采用参考图驱动的会议式说话头像，不强制 3D，沿用可插拔渲染接口。[网关接口及配置](gateway/README.md) · [Avatar 会话](docs/api.md#avatar-会话与音画) · [推进计划](docs/plans/avatar-workstation.md)

## 部署状态

分卡推理已部署。Responses 流式输出、归属隔离、续接、22,753-token 多轮历史、结构化回复、工具调用输出及取消通过；原生双工与 Ava 2D 连续运行约 20 分钟，期间完成 40 次后台 LLM 请求。按用户要求以约半小时功能检查替代两小时负载。LLM 配置窗口 262144，完整窗口的质量尚未验证；浏览器实物播放与会议式神经头像仍待验收。[工作站准入](docs/workstation.md#实机准入) · [功能证据](evaluations/results/inference-functional-workstation.json) · [断流生命周期](evaluations/results/responses-stream-lifecycle.json)
