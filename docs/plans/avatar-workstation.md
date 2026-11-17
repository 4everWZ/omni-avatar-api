# 工作站 Omni 与 Ava 交付计划

当前契约由 [API](../api.md)、[网关](../../gateway/README.md) 和 [工作站部署](../workstation.md) 维护。本文件只协调未完成工作。

## 阶段一：2D Ava

- 在双 3090 工作站部署固定版 Omni，验普通／流式、结构化输出、视觉、长文、排队和取消。
- 实现独立 Avatar 会话、来源归属、可插拔驱动、WHEP 同步音画、字幕、代次清除及故障降级。
- 以用户提供的 `tmp/ava.jpg` 生成正式半身、口型与眨眼素材，交付浏览器验收页。[正式素材与内置 imagegen 提示](../../avatar/assets/ava/README.md)。
- 实测一路双工及 Avatar，与第二调用方普通推理共驻两小时；单独验证权限隔离与释放。
- 按逻辑提交 Git，保持凭据与运行数据不入库，不推送。
- 上下文拉满是硬要求：原目标按 MiniCPM-o checkpoint 的 40,960-token 总窗口（输入加输出）验收；当前探测为 39,932 输入加 1,024 输出预算。不能以接口声明或短上下文成功替代满窗口正确性，也不能把另一个模型承担长文称为 Omni 原生满窗。用户要求先调研官方做法与设备适配，再决定双工历史策略。

## 阶段二：Ava 数字人

仅在阶段一验收成功后开展。沿用 Ava 角色、会话与媒体契约，新增数字人渲染驱动，复验同步、打断、降级及与 Omni 的资源共驻。

开始前与用户确定数字人的视觉形式、候选模型及可接受的延迟和显存预算；不得将“数字人”默认为真人写实风格，也不得为共驻擅自缩减 Omni 能力。

## 当前进度

- SSH 与 Docker 权限已验证；目标 `research-2@100.119.17.6`，两张 RTX 3090、约 61 GiB 主机内存。
- 固定引擎镜像构建成功，AWQ 模型快照下载及资产校验通过。短 Chat、结构化输出、流式和视觉通过；INT4 KV 的约 39,900-token 长文检索失败，同权重 KV／调度诊断未恢复正确答案。[失败证据](../../evaluations/results/omni-workstation-int4.json)。
- Avatar 服务、素材、浏览器及验证器已交付。23 项本地回归通过；浏览器已验独立音频连接、测试音、连续打断、快速关闭再连接及窄屏布局。
- Ava 2D 独立 60 秒实机音画、隔离和释放通过。[报告](../../evaluations/results/avatar-2d-workstation.json)。原生双工及 Ava 的 120 秒轻量混合检查、排队超时、取消和释放通过。[报告](../../evaluations/results/native-avatar-workstation.json)。普通请求为每三十秒一次，不作为持续满载证据。
- 两小时负载、首包与同步的完整测量、浏览器麦克风／屏幕交互、插话、原生恢复、客户端渲染故障降级及可信 HTTPS 入口仍待验收。完整负载强度须按实际调用方需求确定。
- 浏览器已确认麦克风测试源可上传 PCM 并收到原生回复／字幕与画面，但测试源未得到预期算术答案，麦克风语义验收不记通过。移除了未经服务端 VAD 配置的自动插话更新及客户端音量阈值判语音，保持原生决策与手动打断。
- 阶段二等待阶段一验收。
- 官方非 AWQ 候选 `openbmb/MiniCPM-o-4_5@fb09628e98b1f55d8d5b863e0eccfcdd11bff499` 已完成同窗口、同 INT4 KV 对照：普通／流式、结构化和视觉通过，39,932-token 检索仍空答。[报告](../../evaluations/results/omni-workstation-bf16.json)。仅提高权重精度未恢复该用例，不将候选作为主线；两套资产保留。
- 用户允许在性能余量足够时采用合适的 8bit 量化，不限定 4bit。若 BF16 权重正确但资源不足，评估 3090 及当前引擎支持的 8bit 权重方案；若证据指向 KV 精度，再单独评估更高精度 KV。分别记录权重与 KV 精度，不降低窗口或放宽正确性要求。
- INT8 KV 已对照 AWQ 和 BF16 权重；Thinker KV 提至 4 GiB 后，两者均能结束 39,932-token 请求，但仍空答。[INT8 报告](../../evaluations/results/omni-workstation-int8-kv.json)。早期 2 GiB 探测被手动终止，不作为死锁证据；后续同步采样发现预填充仍在推进，且双工窗口折算的 token 容量不能直接当作普通长文的物理 KV 容量。
- AWQ 配 8 GiB 浮点 KV／FlashAttention 仍空答。[报告](../../evaluations/results/omni-workstation-float-kv.json)。官方 BF16 文本骨干按层放到双卡，绕过 Omni 服务，以 Transformers SDPA／BF16 KV 复现相同空答；短问答正确，完整长文只生成结束符。[参考报告](../../evaluations/results/omni-workstation-text-reference.json) · [可复用定位工具](../../tools/reference_minicpmo_text.py)。这不证明所有长文都失败，但不足以交付可靠满窗；继续改量化参数缺乏依据。

## 上下文调研与待选方案

2026-11-17 核对官方资料：维护者明确建议当前 MiniCPM-o 4.5 的多模态输入保持在 8K，超过 12K 的空答／重复属于已知使用边界；提高截断上限或改变采样不能建立长上下文质量保证。[维护者说明](https://github.com/OpenBMB/MiniCPM-V/issues/1131#issuecomment-5238729878)。该说明针对多模态；本项目的纯文本参考失败另由上面的实测支持，不能混为同一证据。

官方实时示例采用滑动保留处理长会话，basic 示例水位为 8K／6K，context 模式保留近期单元与有限的已生成文字；这些策略不能保证所有旧音视频仍可直接访问。[固定版示例](https://github.com/vllm-project/vllm-omni/blob/v0.32.0/examples/online_serving/minicpmo/README.md)。`main` 已有更新的 MRv2 配置，不能将其视为当前固定镜像已提供或已验收。

推荐评估“共享网关、实时 Omni、独立长上下文服务”，尚未获准采用：

| 资源 | 建议职责与候选 | 需验证的边界 |
| --- | --- | --- |
| 3090 GPU 0 | 独立长上下文服务，优先评估 Qwen3.5-9B INT8 权重、浮点 KV | 原生上限为 262,144；40,960 只是原目标的回归门槛，不能称为该模型满窗。先验 INT8 模型兼容与质量，再测满窗正确性、首包和峰值。 |
| 3090 GPU 1 | MiniCPM-o 实时语音／视觉及 Talker、Code2Wav，原生滚动窗口 | 三阶段单卡共驻未测；按 8K 推荐范围设计近期历史，不承诺原生 40K 记忆。 |
| CPU／AGX Orin 64GB | 采集、会话、2D Ava；数字人仍在阶段一验收后选驱动 | 当前 2D 无 GPU 依赖；AGX 角色是建议，尚未修改部署。 |

Qwen3.5-9B 官方配置有 8 层全注意力、4 KV heads、head_dim=256：浮点 KV 在 40,960 tokens 约 1.25 GiB，262,144 约 8 GiB；另计线性注意力状态、权重、视觉编码器、激活和运行时。BF16 权重文件约 17.98 GiB，因此其原生满窗不适合承诺单张 3090 BF16 共驻；INT8 权重为待测候选。更保守候选是 Qwen3.5-4B BF16，权重文件约 8.68 GiB、同样 8 层全注意力，但质量须与 9B 对照。[官方 9B 模型](https://huggingface.co/Qwen/Qwen3.5-9B) · [官方 4B 模型](https://huggingface.co/Qwen/Qwen3.5-4B)。这些是配置估算，均不是目标机运行结果。

3090 属于 Ampere，vLLM 支持 INT8 W8A8；其硬件支持不等于具体 MiniCPM／Qwen 量化产物已合格。FP8 W8A8 不作为 Ampere 的原生加速方案。[固定版量化支持](https://docs.vllm.ai/en/v0.32.0/features/quantization/)。实测两卡为 PHB 连接，没有 NVLink；因此优先隔离两个服务，跨卡 TP 作为另行测量的备选。

待用户选择：接受独立长上下文模型与实时滚动历史，或坚持同一个 Omni 模型的可靠满窗并重新选型。前者改变原来“Omni 原生满窗”的语义，不能自动替换原验收要求。新模型、自动路由、摘要／检索记忆及历史截断均未部署；不得将摘要等同于保留完整原始上下文。
