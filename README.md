# Omni + Avatar API

面向多个应用的原生双工 Omni 和 Avatar 音画 API。现有共享网关覆盖普通推理、视觉和实时会话。目标接口允许应用为实时会话订阅同步的 Avatar 音画。各应用自行持有用户、记忆、任务和业务工具。

**主部署目标是双 RTX 3090 的 Linux x86-64 工作站。** 主线启动组合是 `infra-compose.yml` 加 `infra-workstation.compose.yml`。先验收共享推理和原生双工，再交付独立 Avatar 会话，节目和 OBS 随后接入。 [工作站部署](docs/workstation.md) · [服务接口](docs/api.md)

| 入口 | 用途 |
| --- | --- |
| `infra-compose.yml` ＋ `infra-workstation.compose.yml` | 工作站主线。固定 Linux amd64 引擎和工作站两卡配置，部署模型获取、推理引擎和鉴权网关。 |
| `infra-media.compose.yml` | 可选节目传输。仅在需要节目媒体时叠加。 |
| AGX 历史配置 | 基础文件的 ARM64 配置与 CUDA 12.9 覆盖用于历史复现。 [AGX 指南](docs/infra-setup.md) · [JetPack 6 指南](docs/jetpack6-cu129.md) |

## 共享 API

现有网关以 `omni-default` 提供 Chat Completions 和 Realtime，模型版本、实际窗口和能力由服务回报。独立调用方使用各自的 Bearer 凭据。独立 Avatar 服务提供 Ava 2D 会话、WHEP 音画及浏览器验收页，本地媒体与权限回归通过，工作站共驻待验收。2D 成功后，再以 Ava 为基础交付数字人驱动。[网关接口及配置](gateway/README.md) · [Avatar 会话](docs/api.md#avatar-会话与音画) · [推进计划](docs/plans/avatar-workstation.md)

## 部署状态

工作站固定引擎镜像已构建，容器可识别两张 3090。Ava 2D 独立音画、权限及释放检查通过；Omni 推理及两小时混合负载仍待验收。AGX 长文答复和持续负载的历史失败留在对应证据中。[工作站准入](docs/workstation.md#实机准入) · [Ava 实机报告](evaluations/results/avatar-2d-workstation.json) · [AGX 实测证据](https://github.com/4everWZ/Tuber_Studio/blob/main/evaluations/results/omni-deployment-20261116.md)
