# 工作站 Omni、LLM 与 Ava 交付计划

当前契约由 [API](../api.md)、[网关](../../services/gateway/README.md) 和 [工作站部署](../deployment/workstation.md) 维护。本文件协调尚未完成的部署与验收。

## 已确认分工

用户已确认解耦。Omni 是实时助手：约 8K 近期窗口用于多轮语音／视觉交互，长期信息由调用方的外部记忆补充。后台 LLM 独立承担长上下文推理。这里的上下文主要指多轮历史；不再将一次性向 Omni 输入近 40K 文本作为阶段一准入条件。

本仓库交付推理与 Avatar API，不实现 Agent 循环、任务调度、工具执行、记忆数据库、自动摘要或检索。Chat 调用方提交 messages 和所选记忆；Realtime 持有活跃会话。LLM 返回的工具调用属于推理输出，由调用方执行。历史滑动会丢弃原始输入，不能将外部记忆或摘要等同完整历史仍在 KV 中。

Omni、后台 LLM 和 Avatar 全部部署在当前双 3090 工作站；不将 Avatar 转移到 Orin。AGX 历史配置保留。

## 阶段一：推理服务与 2D Ava

1. GPU 0：固定 Qwen3.5-9B W4A16，以 Responses HTTP／增量 SSE 为主接口，兼容 Chat，提供 JSON Schema、工具调用输出及 Chat token 计量。当前快照为 `axteam/Qwen3.5-9B-AWQ-4bit@5577b5b71e6f06b9f6a1f3743ef6ba6dc4ecd2df`，实际格式 compressed-tensors W4A16；原 8bit 候选保留为历史对照。
2. GPU 1：MiniCPM-o 三阶段共驻，Chat 总窗口 8192，Realtime 原生 basic 历史水位 8000／6000，验证多轮保留、窗口滑动、语音、视觉、静默、插话和恢复。
3. 同一宿主的独立 Avatar 服务继续用 Ava 2D 素材、CPU 渲染和 WHEP 输出。验证原生音频绑定、字幕、同步、打断、隔离、释放和渲染故障降级。
4. 网关使用显式服务前缀与独立准入：`/v1` 的 `omni-default`；`/llm/v1` 的 `llm-default`。后台推理饱和不得阻塞实时助手，不做自动选模型或隐式 Agent 编排。
5. 多轮验收以客户端提交的历史和原生会话行为为依据；显式验证窗口边界及超限错误。后台 LLM 的原生配置上限为 262144，完整窗口的资源与正确性仍须实测，不能仅凭模型卡标记通过。
6. 按用户要求在约半小时内完成预检与约 20 分钟一路原生双工、Avatar、后台 LLM 混合功能检查；记录调用强度、首包、音画、可用的资源采样和释放，不要求两小时负载。验证浏览器麦克风／屏幕交互及可信访问入口。
7. 按逻辑提交 Git，凭据和运行资产不入库，按用户授权合并并同步仓库与工作站。

## 阶段二：Ava 数字人

2D API 功能检查通过后开展。用户已确认先做参考图驱动的会议式说话头像，不强制 3D。沿用 Ava、会话、音频源绑定和 WHEP 契约；参考图上传 → 异步准备任务 → avatar_id → 会话 → 流式音画。预留驱动扩展，以后可接真正的 3D 多视角形象。

在线 API 与私有神经驱动已交付：上传 PNG／JPEG、异步准备与取消、归属隔离、持久化 avatar_id、音频／Realtime 绑定及 WHEP。准备与通话共用单个 GPU 配额，渲染失败释放工作线程并保留原生语音。

用户要求至少 25 fps。固定 Ditto 经编译、CUDA Graph、分块预热和初始视频缓冲优化后，120 秒实际生成 3,012 张神经帧，最大音画时间线采样偏差 58 ms；同时原生语音和 4 次后台 Responses 请求通过。维持 25 fps、1200 ms 音频输入预留，未改动两个推理服务的窗口或量化配置。准备／归属／取消测试 3 项、Avatar 回归 14 项及网关回归 15 项通过。重启复用、原生取消后的代次隔离与恢复通过。[功能证据](../../evaluations/results/portrait-functional-workstation.json) · [生命周期](../../evaluations/results/portrait-lifecycle-workstation.json)。

后续需人工确认真实麦克风／扬声器与主观口型观感；完整 262144 窗口质量、持续满载 QoS 未包含在当前功能通过结论内。私网 HTTPS 需要 tailnet 管理员启用 Serve；当前浏览器可使用 localhost SSH 隧道。

## 当前进度

- SSH、Docker 和双卡已确认；主机约 61 GiB RAM，GPU 间为 PHB 连接。
- 原两卡 Omni、Ava 2D 独立 60 秒与原生双工／Avatar 120 秒轻量混合检查通过，但不替代新分工的共驻验收。[Ava](../../evaluations/results/avatar-2d-workstation.json) · [旧配置短混合](../../evaluations/results/native-avatar-workstation.json)。
- 分卡引擎已部署；Responses 流式、归属续接、结构化、工具调用输出、取消和独立准入通过。22,753-token 多轮续接检索正确，不能据此声称完整 262144 窗口质量通过。
- 约 20 分钟原生双工／Ava 2D 共驻通过：60,003 音频帧、30,003 视频帧、342 原生音频块和 40 次后台 Responses 请求。调用方隔离及幂等释放通过。[功能证据](../../evaluations/results/inference-functional-workstation.json) · [断流与取消](../../evaluations/results/responses-stream-lifecycle.json)。
- 浏览器实物麦克风／屏幕完整交互与可信 HTTPS 仍待人工或管理员配置。功能报告未采到 GPU 数据，不将 CLI 零散读数称为负载峰值。
- 旧 Omni 近 40K 的失败和精度对照保留为模型边界证据，不再是实时助手的新验收门槛。[INT4](../../evaluations/results/omni-workstation-int4.json) · [BF16](../../evaluations/results/omni-workstation-bf16.json) · [INT8 KV](../../evaluations/results/omni-workstation-int8-kv.json) · [直接文本骨干](../../evaluations/results/omni-workstation-text-reference.json)。
私网 API 已通过 `deploy/compose/vpn.yml` 发布到 `100.119.17.6:8000`，同时保留回环入口；从另一 VPN 客户端验证两个模型卡、Responses 增量 SSE、形象归属与匿名拒绝。[私网验证](../../evaluations/results/vpn-api-workstation.json)。

Omni 7,928-token／20 消息的多轮早期事实检索及超限 400 已通过；相关 API 适配回归 3 项通过。后台 LLM 的官方 Python SDK 3.35.1 typed streaming、续接和查询也通过。[8K](../../evaluations/results/omni-8k-multi-turn-workstation.json) · [Responses SDK](../../evaluations/results/responses-sdk-workstation.json)。

外部任务按 Swarm／orchestrator 思路组织，当前优先交付推理 provider。保留单请求 262144、4 路较短请求真实同时生成、总 token 预算及 25 fps。用户已否决约 3.4 TPS 的固定等待方案，要求优化恢复原约 50 TPS 的单路速度；此前“不设最低速度”不再作为交付依据。

已移除固定等待，采用 W4A16、CUDA MPS 20% 活跃线程上限及解码 CUDA Graph，保留 9 GiB BF16 KV 与 FP32 SSM。空闲单路端到端 47.18 TPS（生成区间 47.75），空闲四路每路 37.12–37.27、合计 148.47；通话单路 22.92–23.90，通话四路每路 18.68–19.42、合计 74.73–77.34。90 秒真实神经帧 2,269、最大检查期采样偏差 79 ms、取消恢复通过。[吞吐与共驻](../../evaluations/results/throughput-mps-workstation.json)。旧 250 ms／3.4 TPS 报告仅作撤回方案的历史记录。

新配置复验引擎四序列、28.5K 与三路短请求混跑、四个 worker 独立保存及并行续接、预算拒绝与取消释放；Responses 工具、JSON Schema、SSE、22,753-token 多轮检索及 Omni 8K 边界通过。[并发预算](../../evaluations/results/llm-mps-concurrency-workstation.json) · [兼容性](../../evaluations/results/responses-mps-compatibility.json)。完整 262K 质量、长期满载与实物播放仍按前述边界另行验收。编排、工具执行、共享记忆与工作区由应用项目持有，具体框架集成尚未开展。

MPS 控制服务及客户端重建后，已准备形象复用、30 秒四路／25 fps、音画时间线与取消恢复通过。[重启证据](../../evaluations/results/mps-lifecycle-workstation.json)。
