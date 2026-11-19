# Omni Avatar API

[English](README.md) | [简体中文](README.zh-CN.md)

面向实时助手和长上下文应用的自托管推理与说话头像服务。通过一个鉴权网关，向多个应用提供多模态 Omni、独立的 Responses 推理接口和可插拔 Avatar 音画渲染。

调用应用负责用户、会话记忆、Agent 编排和工具执行；本仓库提供供这些应用调用的推理与媒体服务。

## 功能

- **实时多模态助手**：MiniCPM-o Chat Completions 与原生双工 Realtime，使用约 8K 的近期上下文窗口。
- **长上下文推理**：独立 Qwen3.5 后端，保留单请求 262K 窗口，以共享 token 预算控制最多 4 个生成序列，支持 Responses、增量 SSE、结构化回复和函数调用输出，并保留 Chat Completions 兼容入口。
- **CPU 头像**支持内置 Ava 2D 和已登记的 Inochi2D 原画，通过 WHEP 输出同步音画，支持打断和代次隔离。
- **共享基础服务**：应用独立 Bearer 凭据、后端独立准入、有界队列、取消以及按调用方归属隔离的响应续接。
- **固定部署版本**：锁定模型快照和推理镜像，将资产下载与离线服务分开。

## 架构

| 组件 | 职责 |
| --- | --- |
| Gateway | 对外鉴权、模型别名、流式代理和后端独立准入。 |
| Omni 引擎 | 低延迟语音／视觉交互和近期会话上下文。 |
| LLM 引擎 | 为应用和外部编排器提供长上下文文本推理。 |
| Avatar 服务 | 会话归属、音频时间线、渲染生命周期和 WHEP 音画。 |

已验证的工作站布局使用**双 RTX 3090**：Omni 使用一张卡，文本 LLM 使用另一张卡，也可按需让给 Klein 生图。头像使用宿主 CPU 渲染。AGX Orin 配置提供独立硬件评估路径。详见[架构](docs/architecture.md)与[部署配置](docs/deployment/workstation.md)。

## 快速开始

部署目标为 Linux NVIDIA 工作站，需要 Docker Engine、较新的 Docker Compose 插件、NVIDIA Container Toolkit 和 Python 3.12+。Python 用于统一部署入口，服务依赖在容器内安装。

在仓库根目录执行：

```sh
cp .env.example .env
```

为 `OMNI_GATEWAY_TOKEN`、`OMNI_GATEWAY_ADMIN_TOKEN` 和 `OMNI_AVATAR_INTERNAL_TOKEN` 设置**独立的随机值**。将 `OMNI_MODEL_DATA` 指向有足够空间存放模型快照的目录。可用配置见 [.env 示例](.env.example)。

首次部署创建共享应用网络，然后构建、下载资产并启动：

```sh
docker network create tuber-omni
python tools/compose.py -- build omni-engine llm-engine omni-gateway omni-avatar
python tools/compose.py -- --profile assets run --rm model-fetch
python tools/compose.py -- --profile assets run --rm llm-model-fetch
python tools/compose.py -- up -d --wait
```

独立节点可用 `--target` 选择 gateway、omni、llm、avatar 或 images。私有端点与设备选择见[工作站指南](docs/deployment/workstation.md)。

默认组合包含 Omni、文本 LLM、网关和 Ava 2D。API 默认监听 **`http://127.0.0.1:8000`**，浏览器示例位于 `/`。WHEP 媒体需要 LAN／VPN 直达；远程麦克风和屏幕采集需要可信的浏览器上下文。

统一入口以仓库根目录解析 `.env`、构建上下文和绑定目录。添加 `--vpn` 可发布到 `OMNI_GATEWAY_VPN_BIND`，添加 `--media` 可启用 MediaMTX。其他硬件／评估配置使用 `--target` 选择。配置与服务更新方式见[工作站指南](docs/deployment/workstation.md)。

## API

推理和 Avatar 接口均使用应用 Bearer 鉴权，管理凭据单独配置。

| 入口 | 模型 | 用途 |
| --- | --- | --- |
| `/v1/chat/completions` | `omni-default` | 多模态 Chat，支持流式文本。 |
| `/v1/realtime?duplex=1&model=omni-default` | `omni-default` | 原生双工 WebSocket。 |
| `/llm/v1/responses` | `llm-default` | 长上下文推理和 Responses 增量 SSE。 |
| `/llm/v1/chat/completions` | `llm-default` | 兼容 Chat 的客户端。 |
| `/v1/avatar/*` | — | 已登记原画、会话、音频与 WHEP。 |
| `/v1/images/jobs` | Klein 4B | 可选生图队列，支持参考图。 |
| `/healthz`、`/llm/healthz` | — | 需要鉴权的后端身份、就绪与准入查询。 |

使用 OpenAI Python SDK，在客户端环境设置应用令牌后：

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/llm/v1",
    api_key=os.environ["OMNI_GATEWAY_TOKEN"],
)
with client.responses.stream(
    model="llm-default",
    input="用一句话打个招呼。",
    max_output_tokens=512,
    extra_body={"thinking_token_budget": 128},
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

服务实现固定引擎支持的 HTTP／SSE Responses 子集。函数工具由调用方执行，响应续接保存在引擎进程内。支持范围、归属和取消语义见 [API 契约](docs/api.md)与[网关参考](services/gateway/README.md)。

## 仓库结构

```text
services/                 独立构建的服务边界
  gateway/                鉴权与推理／媒体代理
  llm/                    长上下文文本引擎及 vLLM 补丁
  inference/              启动器、模型配置、固定补丁与镜像
  avatar/                 会话、渲染与 WebRTC 音画
deploy/                   Compose 层与可选媒体配置
tools/                    部署、验收与诊断入口
tests/                    跨服务契约与部署工具测试
docs/                     架构、API、开发和部署指南
evaluations/              共享测试输入和保留的实机证据
```

`data/`、`runtime/`、`tmp/` 和 `.env` 属于本地忽略内容。各服务依赖由自己的镜像隔离；服务间通过 HTTP、WebSocket 和 Unix socket 契约通信。

## 开发与验证

[开发指南](docs/development.md)说明依赖隔离的测试方式与本地服务入口；[工具说明](tools/README.md)提供实机验收命令。

现有记录覆盖 Omni 8K 多轮检索、Responses 续接、约 22K tokens 的 LLM 历史，以及旧版 GPU 共享配置下的四路短请求同时生成。

当前 Omni Chat 和后台 Chat／Responses 默认开启 CoT，思考预算沿用上游默认；原生双工保留官方模型策略。实时窗口默认 `basic`（8000／6000），外部应用可在创建会话时调整窗口与文本推理预算，见 [API](docs/api.md#cot-默认与调用方预算)。

256-token 夹具覆盖思考＋正文总生成吞吐，额度可用于思考阶段；完整 CoT 答案、预算、JSON Schema 和工具续接由独立功能夹具验证。计算方式、原始数据和验证范围见[工作站指南](docs/deployment/workstation.md#四路并发与计算预算)。完整 262K 质量、持续负载与实物播放由[后续验收计划](docs/plans/avatar-workstation.md)跟进。

## 文档

- [API 契约](docs/api.md)
- [架构与模块边界](docs/architecture.md)
- [工作站部署与运维](docs/deployment/workstation.md)
- [AGX Orin 评估](docs/deployment/orin.md) · [JetPack 6／CUDA 12.9](docs/deployment/jetpack6-cu129.md)
- [开发指南](docs/development.md) · [部署与验收工具](tools/README.md)
