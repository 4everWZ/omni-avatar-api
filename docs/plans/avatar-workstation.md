# 工作站 Omni、LLM 与 Ava 交付计划

当前契约由 [API](../api.md)、[网关](../../gateway/README.md) 和 [工作站部署](../workstation.md) 维护。本文件协调尚未完成的部署与验收。

## 已确认分工

用户已确认解耦。Omni 是实时助手：约 8K 近期窗口用于多轮语音／视觉交互，长期信息由调用方的外部记忆补充。后台 LLM 独立承担长上下文推理。这里的上下文主要指多轮历史；不再将一次性向 Omni 输入近 40K 文本作为阶段一准入条件。

本仓库交付推理与 Avatar API，不实现 Agent 循环、任务调度、工具执行、记忆数据库、自动摘要或检索。Chat 调用方提交 messages 和所选记忆；Realtime 持有活跃会话。LLM 返回的工具调用属于推理输出，由调用方执行。历史滑动会丢弃原始输入，不能将外部记忆或摘要等同完整历史仍在 KV 中。

Omni、后台 LLM 和 Avatar 全部部署在当前双 3090 工作站；不将 Avatar 转移到 Orin。AGX 历史配置保留。

## 阶段一：推理服务与 2D Ava

1. GPU 0：固定 Qwen3.5-9B 8bit 权重候选，以 Responses HTTP／增量 SSE 为主接口，兼容 Chat，提供 JSON Schema、工具调用输出及 Chat token 计量。核实引擎兼容、实际窗口、多轮历史正确性、排队与取消。候选为 `axteam/Qwen3.5-9B-AWQ-8bit@b1c949034d7716055133dcc05c1f085beb2b83fb`，实际格式是 compressed-tensors W8A16，不根据仓库名称假定为 AWQ 内核。
2. GPU 1：MiniCPM-o 三阶段共驻，Chat 总窗口 8192，Realtime 原生 basic 历史水位 8000／6000，验证多轮保留、窗口滑动、语音、视觉、静默、插话和恢复。
3. 同一宿主的独立 Avatar 服务继续用 Ava 2D 素材、CPU 渲染和 WHEP 输出。验证原生音频绑定、字幕、同步、打断、隔离、释放和渲染故障降级。
4. 网关使用显式服务前缀与独立准入：`/v1` 的 `omni-default`；`/llm/v1` 的 `llm-default`。后台推理饱和不得阻塞实时助手，不做自动选模型或隐式 Agent 编排。
5. 多轮验收以客户端提交的历史和原生会话行为为依据；显式验证窗口边界及超限错误。后台 LLM 的原生配置上限为 262144，完整窗口的资源与正确性仍须实测，不能仅凭模型卡标记通过。
6. 按用户要求在约半小时内完成预检与约 20 分钟一路原生双工、Avatar、后台 LLM 混合功能检查；记录调用强度、首包、音画、可用的资源采样和释放，不要求两小时负载。验证浏览器麦克风／屏幕交互及可信访问入口。
7. 按逻辑提交 Git，凭据和运行资产不入库，不推送。

## 阶段二：Ava 数字人

2D API 功能检查通过后开展。用户已确认先做参考图驱动的会议式说话头像，不强制 3D。沿用 Ava、会话、音频源绑定和 WHEP 契约；参考图上传 → 异步准备任务 → avatar_id → 会话 → 流式音画。预留驱动扩展，以后可接真正的 3D 多视角形象。

当前评估固定版 Ditto PyTorch：已在 GPU 0 与后台 LLM 共驻加载并生成 Ava 画面，额外显存约 1.4 GiB，初测约 14 fps，尚未通过实时性能。下一步定位处理瓶颈，验证合适的单人参考图与延迟，再实现带归属隔离、任务状态、取消和资源释放的在线接口。不得用重复静态帧宣称神经头像实时生成成功。

## 当前进度

- SSH、Docker 和双卡已确认；主机约 61 GiB RAM，GPU 间为 PHB 连接。
- 原两卡 Omni、Ava 2D 独立 60 秒与原生双工／Avatar 120 秒轻量混合检查通过，但不替代新分工的共驻验收。[Ava](../../evaluations/results/avatar-2d-workstation.json) · [旧配置短混合](../../evaluations/results/native-avatar-workstation.json)。
- 分卡引擎已部署；Responses 流式、归属续接、结构化、工具调用输出、取消和独立准入通过。22,753-token 多轮续接检索正确，不能据此声称完整 262144 窗口质量通过。
- 约 20 分钟原生双工／Ava 2D 共驻通过：60,003 音频帧、30,003 视频帧、342 原生音频块和 40 次后台 Responses 请求。调用方隔离及幂等释放通过。[功能证据](../../evaluations/results/inference-functional-workstation.json) · [断流与取消](../../evaluations/results/responses-stream-lifecycle.json)。
- 浏览器实物麦克风／屏幕完整交互、可信 HTTPS 和会议式神经头像仍待验收。功能报告未采到 GPU 数据，不将 CLI 零散读数称为负载峰值。
- 旧 Omni 近 40K 的失败和精度对照保留为模型边界证据，不再是实时助手的新验收门槛。[INT4](../../evaluations/results/omni-workstation-int4.json) · [BF16](../../evaluations/results/omni-workstation-bf16.json) · [INT8 KV](../../evaluations/results/omni-workstation-int8-kv.json) · [直接文本骨干](../../evaluations/results/omni-workstation-text-reference.json)。