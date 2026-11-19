# Omni 推理服务接口

调用方从配置取得 API 地址和服务凭据。推理服务负责模型版本、原生双工会话和推理资源。业务身份、任务、记忆、外部工具授权及动作回执由调用应用持有。健康状态和会话握手报告可用能力，应用可以独立升级或替换服务地址。

当前采用显式模型分工：`/v1` 的 `omni-default` 负责约 8K 近期窗口的多轮实时助手；`/llm/v1` 的 `llm-default` 提供独立后台长上下文推理，两者有各自的健康状态、模型卡和准入队列。所有服务 均在当前双 3090 工作站。Chat 调用方通过 messages 提交历史和所选记忆，显式选择模型，并负责 Agent 编排与返回工具的执行。

## 已有推理接口

| 接口 | 用法 |
| --- | --- |
| `GET /livez`、`GET /healthz`、`GET /v1/models` | 分别确认网关存活、上游可用性和模型卡。客户端读取认证模型卡的实际 `max_model_len`，健康状态报告锁定版本和已验证能力。 |
| `POST /v1/chat/completions` | 普通与流式推理共用入口。使用服务配置的模型别名，按实际能力使用视觉、音频和结构化回复。 |
| `POST /v1/audio/speech` | 使用普通应用 Bearer，提交模型别名、台词和该角色的内联参考 WAV。返回通过转写核对的 WAV。 |
| `POST /v1/chat/prompt_tokens` | 文本按同一聊天模板计量，供需要精确上下文预算的客户端使用。 |
| `GET /v1/realtime?duplex=1` | Realtime WebSocket。以 `session.created` 报告的能力确定输入模态、切片、字幕、打断、恢复和播放确认。 |

Omni 的边界夹具显式关闭 CoT，已验证 20 条消息、7,928 个输入 tokens 的多轮历史检索；超限输入返回 400，调用方可先计量再选择保留的历史与记忆。计量接口允许计算超出窗口的文本，用于压缩预算。[8K 边界证据](../evaluations/results/omni-8k-multi-turn-workstation.json)。

Bearer 鉴权、排队、取消、超时和管理接口由[网关规范](../services/gateway/README.md)维护。原生双工能力由会话握手协商。

双卡工作站默认共享两个 Omni 槽位，Chat 和 Realtime 每次各使用一个。两路可任意组合。第三路按 FIFO 等待，超时或满队列返回 `429` 和 `Retry-After`。活动槽位由 `/api/mode` 报告，Chat 与 Realtime 分别计数。

## 参考音色朗读

MiniCPM-o 朗读适配器把同一份参考录音放入系统消息的音频上下文，并作为 Code2Wav 的参考波形。Thinker 接收带音频的上下文，Talker 接收 Thinker 的台词隐状态。准入按实际含音频的模板计量。客户端仍然只需上传一份参考。

需要 `voice` 参数的 Speech SDK 可传 `voice="reference"`，并通过 JSON 扩展字段 `reference_audio` 提供真实参考。每次请求均须带参考音频。

`input` 为指定台词（1 至 4096 字符），`model` 使用 Omni 模型别名，`reference_audio` 为含 `format: "wav"` 和 Base64 `data` 的对象。`response_format` 可省略或填 `wav`。参考材料接受单／双声道 PCM WAV，上限 8 MB、90 秒。参考须由调用方按角色授权选定并上传。URL、文件路径和调用方传入的引擎参数返回 400。

网关用同一 Omni 引擎合成，再单独提交生成音频作盲转写。两次调用分别获取并释放普通 Chat 槽位，共用该请求时限。收到完整输出且转写字词相符后，响应返回 `audio/wav`，`X-Omni-Transcript` 为实际转写的 UTF-8 Base64。核对允许句读和排版差异，保留拉丁词边界、数字和词序。音频无效、增漏词或转写未能核实原句返回 422。满队列返回 429 及 `Retry-After`，超过推理期限返回 504。

## CoT 默认与调用方预算

Omni Chat、后台 LLM Chat／Responses 默认 `chat_template_kwargs.enable_thinking=true`。网关在模板计量与推理前应用相同默认值，保留调用方显式覆盖。模型按问题选择直接回答或先生成思考段。

`thinking_token_budget` 默认采用固定上游的 `None`，思考长度、effort 和采样参数沿用模型／部署默认。Omni Chat 总生成默认值跟随固定官方部署配置，为 2048，LLM 使用其上游默认，均受各自上下文与准入预算约束。[固定版官方部署](https://github.com/vllm-project/vllm-omni/blob/v0.32.0/vllm_omni/deploy/minicpmo_4_5.yaml)。调用方应显式设置合适的总输出上限，以便短请求共享预算并发；思考与正文都占用这份输出额度。

| 字段 | 外部用法 |
| --- | --- |
| `chat_template_kwargs.enable_thinking` | 布尔值；显式 `false` 可关闭，省略默认开启。适用于 Chat 与 Responses。 |
| `thinking_token_budget` | 本地扩展，非负整数限制思考阶段，`-1` 采用上游语义，思考额度由总输出与上下文上限约束；省略为上游默认。Chat 和 Responses 均转交同一采样器，思考与正文共享总输出额度。 |
| Chat `max_tokens`；LLM Chat 也接受 `max_completion_tokens` | 限制思考＋正文总输出。Omni Chat 使用 `max_tokens`。 |
| Responses `max_output_tokens` | 限制思考＋正文总输出；同样参与共享 token 准入。 |
| Responses `reasoning.effort`／Chat `reasoning_effort` | 透传上游模板能力；`none` 在未显式指定模板开关时关闭思考。严格思考配额使用 `thinking_token_budget`。 |
| Responses `include_reasoning=false` | 隐藏返回的思考内容；生成的思考继续计入 tokens。 |

例如调用方可提交 `{"model":"llm-default","input":"计算 37×48","stream":true,"max_output_tokens":1024,"thinking_token_budget":256}`；该值作为单次请求的显式预算生效。SDK 通过 `extra_body` 发送本地扩展字段。思考预算按模型标记阶段计量；正文解释的总长度与格式分别由总输出额度和结构化约束控制。保留的 Responses 历史、工具结果与思考／正文共同使用模型上下文与准入预算。

原生 Realtime 双工使用模型官方监听／说话策略及语音分块预算。上述 CoT 开关与思考 token 预算适用于 Chat／Responses。

## 私聊中转认号主选定的一句话

固定版 vLLM-Omni 接受并回显 `input_audio_transcription`，此字段不会启动独立 ASR。原生最终用户转写事件取决于模型是否产出转写。服务端历史校准的 ASR 选择针对中断后已播的助手音频。[固定版双工协议](https://github.com/vllm-project/vllm-omni/blob/v0.32.0/docs/serving/realtime_duplex_api.md)

使用一次性票据的私有连接可在 WebSocket URL 加 `input_transcription=explicit`。客户端收到 `session.created.session.capabilities.gateway_input_transcription` 的 `mode=explicit_once` 和 `receipt=hmac-sha256-v1` 后可显示转认操作。连接协商本身不会保存音频或发起转写。`program=1` 不提供该能力。

| 方向 | 事件与含义 |
| --- | --- |
| 客户端 → 网关 | 用户选择开始后发 `gateway.input_transcription.start`。网关返回 `.started` 及本次 `capture_id`。使用原连接后续的单声道 16 kHz PCM16 `input_audio_buffer.append`，最长 15 秒。 |
| 客户端 → 网关 | `gateway.input_transcription.finish` 带相同 `capture_id`，仅提交本次选择的音频一次。客户端可用相同 ID 发 `.cancel` 丢弃录音；挂断会清理正在处理的请求。 |
| 网关 → 客户端 | `gateway.input_transcription.completed` 返回 `capture_id`、实际样本数 `samples`、`source=explicit_user_audio` 与 `transcript`。客户端应显示原文供核对，记忆与交办各需用户确认。 |
| 网关 → 客户端 | `gateway.input_transcription.failed` 返回本次 ID 与原因，如 `empty_audio`、`audio_bounds`、`capacity` 或 `budget`。 |

转认与原生 final 使用不同事件。网关将选中的录音封装为 WAV，走 Omni 的音频 Chat 和普通推理准入；不传会话提示、记忆、助手回复或预期句子。每个私聊连接分别持有在内存中的短时录音。

完成事件另含小写十六进制 `receipt`。业务服务使用签票的普通应用凭据作为 HMAC-SHA256 密钥，验证 UTF-8 JSON 数组 `["omni.mic.v1", sha256_hex(ticket), capture_id, samples, transcript]`（紧凑序列化，保留非 ASCII 字符）。业务端应同时核对票据签发时绑定的号主与角色。界面可以让号主修订文字；保存时保留经核实的机器原文，并单独记录号主最终确认的文字。密钥仅在服务端使用。

## Omni 实时历史窗口

默认沿用固定 vLLM-Omni 0.32 的 `basic`：高水位 8000、低水位 6000。省略窗口字段即采用这个官方服务默认值。Chat 总窗口为 8192 tokens，Realtime 滑动策略在该模型容量内管理历史保留。

外部调用方需要调整时，先连接 `ws://100.119.17.6:8000/v1/realtime?duplex=1&autostart=0`，再发送首条 `session.update`。`autostart=0` 让窗口参数在创建会话时生效，避免服务提前按默认值创建会话。沿用应用 Bearer；浏览器仍可通过已有 ticket 认证。示例：

```json
{
  "type": "session.update",
  "session": {
    "model": "omni-default",
    "modalities": ["audio", "text"],
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "playback_commit_policy": "ack_only",
    "extra_body": {
      "sliding_window_mode": "basic",
      "basic_window_high_tokens": 8000,
      "basic_window_low_tokens": 6000
    }
  }
}
```

| 模式 | 保留方式 | 使用边界 |
| --- | --- | --- |
| `basic`（默认） | 达到高水位后按完整输入单元移除最早历史，目标降至低水位；保留 system／参考音频前缀。 | 保留近期原始输入及回复，适合当前约 8K 实时助手＋外部记忆。 |
| `context`（调用方可选） | 默认保留 24 个近期输入单元，另带上至多 500 tokens 的较早模型生成文本，并重建保留窗口。 | 保留内容为近期输入和较早模型生成文本；历史重建增加计算开销，长期记忆由应用持有。 |
| `off`（调用方可选） | 持续累积会话历史。 | 历史与生成使用当前 8192-token 容量，调用方负责会话轮换。 |

如需试用 `context`，将 `extra_body` 替换为 `{"sliding_window_mode":"context","context_max_units":24,"context_previous_max_tokens":500}`。`unit` 按输入单元计数，token 预算按实际模板和模态独立计量。窗口配置在会话创建后固定，调整策略需新建会话。维持 `0 < low < high`，并按当前模型窗口留出输入及生成余量；参数由上游校验，配置应匹配实际显存。当前验收策略为 `basic`，`context` 作为调用方可选策略单独评估。[固定版官方说明](https://github.com/vllm-project/vllm-omni/blob/v0.32.0/docs/serving/realtime_duplex_api.md#open-a-session)。

## 后台 Responses

使用 `POST /llm/v1/responses`，模型 `llm-default`，设置 `stream: true` 接收原生增量 SSE。支持文本、多轮 `previous_response_id`、JSON Schema 及 function 工具调用输出；工具由应用执行。`GET /llm/v1/responses/{id}` 与 `POST /llm/v1/responses/{id}/cancel` 只允许原调用方访问，其他调用方得到 404。

当前实现 vLLM 0.32 的上述 HTTP／SSE 子集。响应与续接关系保存在引擎进程内；业务历史由调用方持久化，重启后可用保存的历史发起新请求。后台任务在客户端断开后保留准入直到结束或取消；前台流式请求断开会取消。详见[网关规范](../services/gateway/README.md)。

## 外部 Swarm／orchestrator 调用

外部编排器统一使用后台 `base_url=http://100.119.17.6:8000/llm/v1`、模型 `llm-default` 和应用 Bearer，优先调用 Responses；Chat 保留兼容。每个逻辑 worker 单独持有历史或 `previous_response_id` 分支；任务、交接、工具执行、共享记忆和工作区操作由应用项目负责。

LLM 单请求上限保留 262144 tokens，引擎和网关默认允许 4 个生成序列；四路短请求以实际输出重叠验收，多路满窗口按显存与总预算独立评估。所有活跃 Chat／Responses 共用 `LLM_TOKEN_BUDGET=262144`：每次准入预留完整模板输入（含 instructions、tools 和已保存续接历史）加最大输出，Chat 多候选分别计入序列和预算。计量使用同一引擎渲染器的实际 token 数。模型使用 W4A16 权重，保留 9 GiB BF16 KV 和 FP32 SSM。

调用方应显式设置 `max_output_tokens`（Chat 为 `max_completion_tokens`／`max_tokens`），以释放短请求并发空间；省略时保留引擎默认输出上限，可能按整个剩余窗口预留并串行运行。长短请求按 FIFO 等待可用序列与 token；达到队列上限或等待超时返回 429 和 Retry-After，单请求预算本身不可容纳返回 400。`/llm/api/mode` 报告 `reserved_tokens`、`available_tokens`、`token_budget`、活跃序列和排队数。后台及断流保留规则同上，结束或取消后释放预算。

`LLM_MAX_NUM_SEQS` 同时配置引擎和网关（默认 4）；`LLM_TOKEN_BUDGET` 控制全局活跃预留，模型窗口由 `max_model_len` 独立配置。`LLM_QUEUE_LIMIT`、`LLM_QUEUE_TIMEOUT`、`LLM_REQUEST_TIMEOUT` 可独立设置后台排队和总期限，默认继承现有 16／5 秒／1200 秒；Omni 的实时通道保持独立。提高并发或预算必须重新验证显存余量与混合推理吞吐。

当前验收范围是 provider API，具体 Antigravity／agent-orchestrator 客户端接入留在应用项目。[provider 约定](../services/gateway/README.md#external-orchestration-clients)。

## 浏览器实时语音

浏览器用应用 Bearer 调用 `POST /v1/browser-tickets`，提交 `{"path":"/v1/realtime"}`。返回的 ticket 有效期为 30 秒，仅可使用一次。使用该 ticket 连接 `/v1/realtime?duplex=1&ticket=…`，直接收发原生音频、字幕和播放回执。旧版 `/v1/avatar/*` 接口已退役。

## 生图任务

可选工作站生图服务沿用应用 Bearer。生图任务和 PNG 结果按提交凭据隔离。其他凭据访问同一 ID 返回 `404`。

| 接口 | 行为 |
| --- | --- |
| `POST /v1/images/jobs` | 提交 JSON，字段为 `prompt`（1–4096 字符）、可选 `reference_image_base64`（PNG/JPEG，最大 4 MiB 和 800 万像素）和可选整数 `seed`（0–4294967295）。返回 `202`、`id`、实际种子和 `Location`。一次生成一张 1024×1024 PNG。 |
| `GET /v1/images/jobs/{id}` | 返回 `queued`、`running`、`cancelling`、`completed`、`cancelled` 或 `failed`。成功时提供 `result_url`，失败时提供 `error`。 |
| `GET /v1/images/jobs/{id}/result` | 完成后获取 `image/png`，其他状态返回 `409`。 |
| `POST /v1/images/jobs/{id}/cancel` | 取消等待任务或请求运行任务协作停止，返回当前状态。安全释放 GPU 并恢复后进入终态。重复调用返回当前状态。 |

服务依次执行任务，最多额外排队八个。满队列返回 `429`，等待时限为 1200 秒，终态记录保留 24 小时，上限 256 条。协调进程重启时，未完成任务标为失败。

切换开始后，已准入的 LLM 生成正常完成，新的 Chat／Responses 生成按现有 FIFO 和等待时限准入，超时返回 `429` 和 `Retry-After`。已有 Responses 查询和取消可用，已存储的历史在正常切换后可用原 `previous_response_id` 续接。`/llm/healthz` 在切换时返回 `busy`，恢复状态不明时返回 `503`。

### 独立图像节点

Gateway 可用 `--remote-images` 将上述生图任务交给专用 worker。提交、归属校验、轮询、取消及下载沿用同一接口；专用节点独立执行准备和生成，端到端耗时包含排队与跨机传输。专用模式不需要文本服务、共享 GPU 卷或 LLM 控制凭据，结果不含同卡切换特有的 `gpu_release`。每个 worker 由一个 Gateway 作业队列驱动，作业元数据和结果保存在该 Gateway 的 image-jobs 卷。
