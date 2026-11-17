# Repository tools

## 部署入口

`python tools/compose.py -- <Docker Compose 参数>` 默认选择当前分卡工作站。`--portrait`、`--vpn`、`--media` 开启可选层；`--target` 选择历史／评估硬件配置，`--env-file` 指定环境文件，`--dry-run` 查看命令。选项写在 `--` 之前，Docker 参数放在之后。入口固定仓库根目录解析模型目录和构建上下文。

## 工作站实机验证

`tools.qualification.workstation` 使用真实网关和 WHEP，不启动模拟引擎。依赖与 `services/avatar/requirements.txt` 相同。`.env` 需配置主应用及一个独立的额外应用令牌；凭据不写进报告。

```sh
python -m tools.qualification.workstation --env .env --avatar-only --seconds 60 --report tmp/avatar-check.json
python -m tools.qualification.inference --env .env --seconds 1200 --report tmp/inference-functional.json
python -m tools.qualification.concurrency --env .env --report tmp/llm-concurrency.json
```

历史单引擎 `tools.qualification.workstation` 完整模式检查模型身份、40,960 窗口、普通／流式推理、JSON Schema、视觉颜色和接近窗口上限的长文编号检索（重复记录、多样库存两种文档），再运行一路原生双工及 Ava，并由第二调用方每三十秒请求普通推理。该默认流量属于轻量后台请求，不是持续满载；完整负载强度须按实际调用方需求确定。原生输入、普通请求和媒体接收各自运行，普通请求耗时不阻塞音频输入。前九十秒没有原生语音输出即失败；结束时还需在字幕中找到语音问题的正确答案。

报告包括实际时长、解码音画帧及非静音帧、普通请求数、调用方隔离和幂等释放结果。GPU／主机指标来自运行验证器的机器，`metrics_collector_host` 标明采集位置；远端客户端运行的 GPU 数字不代表工作站。解码并消费媒体也不代表真实扬声器和显示器已经播放。

在工作站运行可以收集其 GPU 和内存指标。也可使用现有 Avatar 镜像，挂载代码及单独可写的报告目录，以读取 `.env` 的宿主 UID 运行，使用 host 网络和 `--gpus all`；不需要修改宿主 Python。

完整模式还用持续 SSE 占据普通准入，检查另一请求排队到期返回 429，再断开 SSE 并确认容量释放和后续推理可用。该工具不替代浏览器对麦克风、屏幕采集、重连、静默、重叠语音、字幕同步、打断和渲染故障降级的验收。当前按用户要求运行约半小时功能检查，不要求两小时；短报告不表示长时间压力测试通过。

`tools.diagnostics.minicpmo_text` 是长文定位工具：只读取同一官方 BF16 快照的 `llm.*` 权重，严格加载 Qwen3 文本骨干，用 Transformers SDPA、BF16 KV 和贪心解码运行短问答及相同的 39,932-token 检索。它按层分布到两张 GPU；使用现有引擎镜像中的 PyTorch、Transformers、Accelerate 与 safetensors，不需修改宿主依赖。运行前停止占 GPU 的引擎，挂载仓库与只读模型目录，用 `--gpus all`，结束后恢复原 Compose 服务。入口为 `python -m tools.diagnostics.minicpmo_text --checkpoint /models/<revision> --report tmp/text-reference.json`。它不经过 Omni 的多模态编码器、服务调度或输出解析；`state=completed` 只表示完成测试，必须检查各 case 的 `correct`。该结果不替代 Omni API 或原生双工验收。

`tools.qualification.inference` 是当前分卡主线验证器：先检查 Omni 8192、LLM 262144 的模型配置、Omni 约 7.9K-token 多轮早期事实与超限 400、Responses 归属续接、约 22K-token 多轮历史、JSON Schema、function 输出与测试夹具回填、增量 SSE、后台取消和独立准入，再进行 1200 秒原生双工／Ava 2D 与每三十秒后台 Responses 请求。它不实现 Agent。已通过的实机结果见 [功能报告](../evaluations/results/inference-functional-workstation.json)；完整 262K 窗口质量不在该通过结论内。

`tools.qualification.portrait --reference <单人PNG> --seconds 120 --report tmp/portrait-functional.json` 验证参考图上传、异步准备、跨调用方隔离、真实神经视频、原生语音绑定及后台 Responses 共驻。它按测试时长检查 25 fps 的实际生成总量、帧进度及共享音画时间线采样偏差、取消后的代次隔离与恢复和幂等释放。输出诊断帧保存在报告旁；`renderer.generated_frames` 来自神经帧接收计数，不以 WHEP 的重复画面凑数。短功能测试不证明持续满载 QoS 或物理扬声器播放。

共享语音和长文输入位于 `evaluations/fixtures/`，验收与诊断工具共同使用该层，彼此不导入流程实现。

`tools.qualification.concurrency` 验证四路 Responses 的输出区间实际重叠、约 28K 输入加三路短请求混跑、续接、token 压力导致的 429、剩余预算允许短 Chat 及取消释放。可用 `--engine-metrics http://<引擎私网地址>:8000/metrics` 同时采样 `vllm:num_requests_running` 并要求达到 4；不对外发布该端点。报告保留输入／输出计量、首增量与结束时间、准入和运行序列采样，不把异步提交数当成同时生成数。

`tools.qualification.portrait --reference <PNG> --seconds 90 --concurrent-llm 4` 在原生语音／神经头像播放期间执行四路各 256-token SSE，并要求完整输出区间重叠、真实神经帧达到 25 fps、采样时间线偏差小于 160 ms、取消后恢复。LLM 数字与输出夹具使用温度 0，保持判定内容不变；服务仍保留调用方的采样选择。当前通过 CUDA MPS 与解码 CUDA Graph 分配 GPU 算力，无固定批次等待；报告记录实际首增量和结束时间。

`tools.qualification.throughput --avatar-id <已有形象ID> --seconds 90 --report tmp/throughput.json` 在同一配置测量空闲单路／四路和原生语音、25 fps 头像通话中的单路／四路。默认开启 CoT，使用同一短提示并按模板实报输入 tokens，每路最多生成 256 tokens，温度 0；按引擎 usage 记录端到端 TPS、生成区间 TPS、总吞吐和四路输出重叠。预热 32 tokens 后检查单路端到端至少 45 TPS、通话四路每路至少 15 TPS，作为防退化检查线；这不是长上下文或长期 SLA。头像沿用真实神经帧、时间线及取消恢复判定。并发工具另验证四个 worker 的独立历史续接及总 token 预算，不提供 Agent 编排。

`tools.qualification.omni_throughput --avatar-id <已有形象ID> --seconds 90 --sliding-window-mode basic --report tmp/omni-throughput.json` 测量前台 `/v1/chat/completions` 的文本 SSE：预热后空闲单路、两个同时提交请求，以及原生双工、神经头像和四路后台 LLM 共驻时的额外一路 Omni Chat。也可用 `--reference <PNG>` 新建形象。按 `usage.completion_tokens` 计算端到端及生成区间 TPS，记录输出区间是否重叠；排队请求不算并行生成。原生语音以音频块、正确答复、媒体帧及时间线检查，不把音频块数或字幕增量数当作 token TPS。窗口参数通过 `autostart=0` 后首条会话更新提交；默认 `basic`，`context` 仅用于显式对照。整组通过仍要求 25 fps、最大检查期采样偏差小于 160 ms、原生取消恢复及幂等释放；吞吐数有效不代表其他判定也通过。

`tools.qualification.thinking --report tmp/thinking.json` 验证文本接口默认 CoT、显式关闭、调用方思考预算、非法预算、JSON Schema 与工具续接。完整答案夹具显式给出 4096 总输出额度；这不是修改服务默认预算。`thinking_token_budget=-1` 按上游定义为不限思考预算，非法负值用 -2 测试。工具回填固定的工具结果，不运行真实工具。

两个吞吐工具默认 `--thinking enabled`；可显式 `--thinking disabled` 复现旧对照。CoT 下计时从思考或正文的首个实际文本增量开始，token 数以引擎总生成 usage 为准，另存两个通道的增量数和首包时间。256-token 夹具可能在思考中截断，吞吐通过不能替代完整答案正确性验收。`portrait`／`throughput` 也支持 `--avatar-id` 复用已有形象，和 `--reference` 二选一。
