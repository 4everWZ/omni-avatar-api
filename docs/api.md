# Omni + Avatar 服务接口

调用方从配置取得 API 地址和服务凭据。推理服务负责模型版本、原生双工会话、推理资源和 Avatar 会话。业务身份、任务、记忆、外部工具授权及动作回执由调用应用持有。健康状态和会话握手报告可用能力，应用可以独立升级或替换服务地址。

当前采用显式模型分工：`/v1` 的 `omni-default` 负责约 8K 近期窗口的多轮实时助手；`/llm/v1` 的 `llm-default` 提供独立后台长上下文推理，两者有各自的健康状态、模型卡和准入队列。所有服务和 Avatar 均在当前双 3090 工作站。Chat 调用方通过 messages 提交历史和所选记忆，显式选择模型，并负责 Agent 编排与返回工具的执行。

## 已有推理接口

| 接口 | 用法 |
| --- | --- |
| `GET /livez`、`GET /healthz`、`GET /v1/models` | 分别确认网关存活、上游可用性和模型卡。客户端读取认证模型卡的实际 `max_model_len`，健康状态报告锁定版本和已验证能力。 |
| `POST /v1/chat/completions` | 普通与流式推理共用入口。使用服务配置的模型别名，按实际能力使用视觉、音频和结构化回复。 |
| `POST /v1/chat/prompt_tokens` | 文本按同一聊天模板计量，供需要精确上下文预算的客户端使用。 |
| `GET /v1/realtime?duplex=1` | Realtime WebSocket。以 `session.created` 报告的能力确定输入模态、切片、字幕、打断、恢复和播放确认。 |

Omni 的边界夹具显式关闭 CoT，已验证 20 条消息、7,928 个输入 tokens 的多轮历史检索；超限输入返回 400，调用方可先计量再选择保留的历史与记忆。计量接口允许计算超出窗口的文本，用于压缩预算。[8K 边界证据](../evaluations/results/omni-8k-multi-turn-workstation.json)。

Bearer 鉴权、排队、取消、超时和管理接口由[网关规范](../services/gateway/README.md)维护。原生模型服务的协议和 Avatar 能力独立发布，调用应用根据各自会话实际协商。

双卡工作站默认共享两个 Omni 槽位，Chat 和 Realtime 每次各使用一个。两路可任意组合。第三路按 FIFO 等待，超时或满队列返回 `429` 和 `Retry-After`。活动槽位由 `/api/mode` 报告，Chat 与 Realtime 分别计数。

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

`LLM_MAX_NUM_SEQS` 同时配置引擎和网关（默认 4）；`LLM_TOKEN_BUDGET` 控制全局活跃预留，模型窗口由 `max_model_len` 独立配置。`LLM_QUEUE_LIMIT`、`LLM_QUEUE_TIMEOUT`、`LLM_REQUEST_TIMEOUT` 可独立设置后台排队和总期限，默认继承现有 16／5 秒／1200 秒；Omni 的实时通道保持独立。提高并发或预算必须重新验证显存和头像共驻表现。

启用神经头像覆盖时，LLM 与 Ditto 通过 CUDA MPS 共驻站点选择的 GPU；LLM 活跃线程上限默认 20%，并启用批次 1／2／4 的解码 CUDA Graph、关闭异步调度。派发前读取实际帧进度，渲染余量低于 240 ms 时主动让出最多 18 ms，余量恢复后提前派发；反馈空闲或失效时正常派发，SSE 即时转发。计算资源上限与 token 显存预算分别生效；当前吞吐和重启流程见[工作站部署](deployment/workstation.md#四路并发与计算预算)。

当前验收范围是 provider API，具体 Antigravity／agent-orchestrator 客户端接入留在应用项目。[provider 约定](../services/gateway/README.md#external-orchestration-clients)。

## Avatar 会话与音画

Avatar 由独立服务持有，所有外部请求通过共享网关鉴权。用不同客户端创建独立会话，按会话隔离媒体和控制。渲染服务可替换驱动，统一消费带时间戳和代次的音频，输出同步的视频帧及媒体状态。渲染失败时，原生语音和字幕仍可交付。当前 Ava 2D 已通过分卡配置下约 20 分钟原生双工与后台 LLM 共驻检查；参考图驱动的会议式神经头像也已通过 25 fps、同步、归属隔离及取消恢复检查；浏览器实物交互列入后续人工验收。[当前验收结论](deployment/workstation.md#实机准入)。

| 会话操作 | 目标契约 |
| --- | --- |
| `POST /v1/avatar/sessions` | 从获准的 Realtime 会话或显式音频源创建 Avatar 会话。指定已登记的 `avatar_id`，服务校验来源归属、渲染能力与资源，返回会话 ID、实际格式和受限媒体订阅入口。 |
| `GET /v1/avatar/sessions/{id}` | 查询状态、关联语音会话、已协商的输出能力和失败原因。调用方只读自己或已获授权的会话。 |
| `DELETE /v1/avatar/sessions/{id}` | 幂等关闭渲染和媒体订阅，释放该会话资源。关联语音会话的关闭和恢复遵循 Realtime 生命周期。 |
| 音频、字幕与画面 | 首版用 WebRTC/WHEP 交付浏览器可消费的音画，并在会话事件中提供字幕、播放代次和时间戳。样式、编码、Alpha 或色键按协商报告。 |
| 打断和播放 | 语音取消或播放清除使旧代次的未播音频、字幕和帧失效。服务区分已生成、已传输和客户端确认已播。播放回执由实际消费者提交。 |

客户端直接验收启动、语音和画面同步、插话、失效代次、渲染故障时语音继续、授权隔离和正常释放。用两个独立调用方验证会话及凭据隔离。OBS 可作为后续订阅者，负责节目布局、合成和推流。

### 首版调用

所有请求沿用应用 Bearer，按调用方归属授权；会话查询失败与归属校验失败统一返回 404。媒体订阅同时校验凭据和会话归属。

```json
{"avatar_id":"ava","source":{"type":"audio"}}
```

创建返回 201，包含 `id`、`generation`、`state`、`format`、`media_url`、`events_url` 和 `audio_url`。显式音频通过 `POST audio_url` 提交 `{ "audio":"<base64>", "generation":0, "format":"pcm16", "sample_rate_hz":24000 }`，仅接受单声道 PCM16。每块最多五秒，未播缓冲最多三十秒。`{"type":"clear"}` 清除未播媒体并递增代次；旧代次输入返回 400。

Realtime 客户端仍使用原协议。客户端以 `?avatar=1` 选择 Avatar 后，网关额外发送 `gateway.session.created`，提供 `source_session_id` 和 `avatar_available`。将创建参数的 `source` 改为 `{"type":"realtime","session_id":"<source_session_id>"}`，只允许同一应用凭据绑定。内部复用同一路推理的音频和控制事件。`?avatar=1` 在连接期间自动进入 live 模式，工作站仍按共享总容量准入。原有 `program=1` 仍要求媒体专用凭据。未选择 Avatar 的 Realtime 连接保持原事件流。

### 音画与事件

- 向 `media_url` POST `application/sdp` 的完整 ICE offer，同时请求音频和视频；201 返回 SDP answer 和 `Location`。首版为非 trickle WHEP，DELETE `Location` 幂等释放订阅。每会话最多两个订阅者。媒体连接协商 Opus 和浏览器支持的视频编码；2D 输出为 640×720、25 FPS、RGB。
- `X-Avatar-Timeline-Origin-Ms` 将该订阅的媒体时间零点映射到 Avatar 会话时间。音频与视频由同一服务端时间线驱动；字幕事件携带该时间线上的 `start_ms`。
- `events_url` 是鉴权 WebSocket。事件包括 `avatar.snapshot`、`avatar.audio.queued`、`avatar.subtitle`、`avatar.cleared`、`avatar.failed`、`avatar.closed` 与 `avatar.resync_required`。`source_epoch` 提供上游打断栅栏（显式音频源为 -1），用于与原语音事件对齐。慢消费者收到 resync 后须重新建立会话；字幕与音频均按代次失效。
- `avatar.audio.queued` 提供音频在媒体时间线的 `start_ms/end_ms`、`response_id` 和原回复的 `source_start_ms`。该事件报告排队状态，实际播放由消费者回执报告。
- 消费者可在事件连接提交 `{"type":"playback.ack","generation":0,"played_ms":1000}`。服务保存为 `consumer_report`；生成、传输和消费报告分别计量。Realtime 历史回执仍由消费者按上游协议发送到原语音连接，使用回复相对位置。
- 打断会清除服务端旧代次缓冲；消费者同时停止本地播放并重建 WHEP 接收器，以丢弃已经进入客户端缓冲的旧帧。关闭源连接会关闭绑定的 Avatar；语音恢复后须用新的来源标识重建 Avatar。

浏览器通过鉴权 `POST /v1/browser-tickets` 提交 `{"path":"/v1/realtime"}` 或具体 `events_url`，获取有效三十秒、限定路径且仅用一次的 WebSocket ticket。浏览器把 ticket 放入对应升级请求，普通 HTTP/WHEP 请求继续使用 Bearer。应用长期凭据通过 Authorization 请求头传递。

默认最多四个 Avatar 会话，未活动三分钟回收。浏览器每二十秒查询一次活动会话保活。进程退出释放源连接、会话与媒体。

### 可插拔渲染与后续数字人

`services/avatar/avatars.json` 登记 `avatar_id`、驱动和素材目录；`OMNI_AVATAR_REGISTRY` 可指定替代登记文件。驱动接收带采样起点和代次的原始 PCM，按会话时间戳提供画面，并实现能力声明、代次清除和关闭；音频入队与取帧应保持非阻塞，神经驱动可在独立 worker 推理。媒体与权限不依赖具体驱动。`sprite2d` 使用 Ava 四状态素材和共享时间线的音量；`ditto` 提供参考图会议式说话头像。已登记驱动共用媒体和会话契约，未来 3D 多视角驱动沿用该接口。后续验收见[当前计划](plans/avatar-workstation.md)。

`inochi2d` 在媒体服务 CPU 上读取 Inochi2D 0.8 的 INP1 文件。在私有登记文件中加入例如 `{"my-avatar":{"driver":"inochi2d","root":"rigs/my-avatar/model.inp"}}`，并将文件及模型只读挂载到服务。相对路径以登记文件所在目录为准。通过原有 `POST /v1/avatar/sessions` 传入 `{"avatar_id":"my-avatar","source":{"type":"audio"}}`，向返回的 `audio_url` 提交 PCM，并在 `media_url` 订阅 WHEP。实时会话可沿用 `source.type=realtime`。返回的 `format.driver` 为 `inochi2d`。

该子集在 640×720 画布上支持 Node 与 Part、完整纹理的轴向矩形网格、Normal 混合和线性标量参数。局部平移、缩放与平面旋转依次沿节点树合成，按累计 zsort 递减绘制。平移与角度绑定采用加法，缩放和 Part 不透明度绑定采用乘法；关键点从参数声明的归一化 `axis_points` 读取。画面取同一播放位置的 20 ms PCM，以响度驱动 MouthOpen、频谱明亮度近似值驱动 MouthForm，眼睑、视线、头部和呼吸使用会话时钟。口型控制提供连续形变，音素辨识精度需独立验收。

## 参考图与会议式头像

准备形象时可向 `POST /v1/avatar/generations` 添加 `"infra":{"version":1,"portrait":{"crop_scale":2.3,"crop_vx_ratio":0,"crop_vy_ratio":-0.125,"crop_flag_do_rot":true}}`。前三项为有限 FP32 范围的数值，比例 crop_scale 为正，旋转开关为布尔值。示例取值即原有 registrar 默认值；准备任务绑定参考图及裁剪，修改需创建新形象。创建 Ditto 会话可声明 `"infra":{"version":1,"portrait":{"online":true}}`，返回 format.online=true。当前 Ditto 驱动仅提供在线渲染，显式 false 或未知选项返回 400。

参考图驱动的会议式说话头像采用可插拔 Ditto 驱动，生成参考图身份的会议式音画。启用 `deploy/compose/portrait.yml` 后，所有请求沿用应用 Bearer，通过网关进入私有准备／渲染服务；未启用时返回 503。参考图和生成结果仅原调用方可访问，其他调用方得到 404。

| 接口 | 行为 |
| --- | --- |
| `POST /v1/avatar/assets` | JSON `{"mime_type":"image/png","image_base64":"..."}`，也支持 JPEG。最多 4 MiB、800 万像素，解码后保留方向与比例并填充至 640×720。返回 201 和参考图 `id`。使用清楚的单人正面头像。 |
| `DELETE /v1/avatar/assets/{id}` | 删除未被任务引用的参考图；有依赖时返回 409。 |
| `POST /v1/avatar/generations` | `{"reference_asset_id":"...","driver":"ditto"}`，返回 202 和任务 `id`。准备人脸、外观和运动特征，输出参考图驱动的说话头像。 |
| `GET /v1/avatar/generations/{id}` | 状态 `queued → preparing → ready`；失败为 `failed`，取消为 `cancelled`。仅 `ready` 返回可用 `avatar_id`。 |
| `DELETE /v1/avatar/generations/{id}` | 活跃准备任务标为 cancelled，GPU 工作实际退出前继续占用配额；再次删除可清理记录。就绪任务删除形象及特征，若形象正在通话则返回 409。 |
| `GET /v1/avatar/avatars/{avatar_id}` | 查询已准备形象和驱动能力。已注册的 Inochi2D 原画使用部署方提供的 ID，容器及纹理可加载时返回同一 `id`、`driver: inochi2d` 和 `state: ready`。未知 ID 返回 404，损坏或缺失的注册文件返回 503。神经形象的 `/preview` 返回规范化参考图预览。 |
| `POST /v1/avatar/sessions` | 将就绪的 `portrait_...` 作为 `avatar_id`，仍按上述方式绑定 `audio` 或自有 `realtime` 源；返回同一套音频、事件及 WHEP 入口。 |

每次登记可在 `infra.portrait.calibration` 指定当前参考图的 `face_box`、`eyes` 或两者，坐标以规范化后的 640×720 资产为准。省略的阶段继续使用 Ditto 自动检测。GET 任务时可读回 `crop.calibration`；失败保留原错误码，并通过 `diagnostic.stage` 定位准备阶段。[按形象校准、两份原稿标注与隔离复测](ditto-calibration.md)

Ditto 支持两个渲染和参考图准备共享的 GPU 槽位，工作站默认启用一个，双路帧率待验收。准备任务等待空闲槽，忙时创建神经会话返回 `429`。CPU Avatar 的服务总会话上限是四路。每调用方最多保留 16 张图、16 个任务，总量各 64；客户端按需回收资产。准备形象持久化，重启后由客户端新建活跃会话。不同驱动返回各自实际能力，未来的 3D 驱动沿用同一接口。

神经会话创建时先完成实际渲染线程的预热再就绪；等待创建成功后再向绑定源发送需要形象播放的音频。输出 640×720、25 fps；音频播放预留 1200 ms 推理时间，`format.input_lead_ms` 明示。音画共用时间线，原生 Omni 音频仍独立输出。清空立即停播旧音频并丢弃旧代次视频，神经运动状态允许自然过渡。落后或停滞会发出 `avatar.failed`，调用方继续消费原生语音。会话 `renderer` 字段报告实际 `generated_frames`、最新生成及已显示帧时间，可用于检查是否真正达到实时生成速度。

单头像基线在 90 秒内生成 2,265 张真实神经帧，最大采样时间线偏差 57 ms。双头像的实机状态见工作站指南。 [混跑证据](../evaluations/results/omni-throughput-tensorrt-workstation.json) · [生命周期证据](../evaluations/results/portrait-lifecycle-workstation.json) · [工作站指南](deployment/workstation.md)。

## 生图任务

可选工作站生图服务沿用应用 Bearer。生图任务和 PNG 结果按提交凭据隔离。其他凭据访问同一 ID 返回 `404`。

| 接口 | 行为 |
| --- | --- |
| `POST /v1/images/jobs` | 提交 JSON，字段为 `prompt`（1–4096 字符）、可选 `reference_image_base64`（PNG/JPEG，最大 4 MiB 和 800 万像素）和可选整数 `seed`（0–4294967295）。返回 `202`、`id`、实际种子和 `Location`。一次生成一张 1024×1024 PNG。 |
| `GET /v1/images/jobs/{id}` | 返回 `queued`、`running`、`cancelling`、`completed`、`cancelled` 或 `failed`。成功时提供 `result_url`，失败时提供 `error`。 |
| `GET /v1/images/jobs/{id}/result` | 完成后获取 `image/png`，其他状态返回 `409`。 |
| `POST /v1/images/jobs/{id}/cancel` | 取消等待任务或请求运行任务协作停止，返回当前状态。安全释放 GPU 并恢复后进入终态。重复调用返回当前状态。 |

服务依次执行任务，最多额外排队八个。满队列返回 `429`，等待时限为 1200 秒，终态记录保留 24 小时，上限 256 条。协调进程重启时，未完成任务标为失败。运行中的神经头像和参考图准备优先于等待中的生图。

切换开始后，已准入的 LLM 生成正常完成，新的 Chat／Responses 生成按现有 FIFO 和等待时限准入，超时返回 `429` 和 `Retry-After`。已有 Responses 查询和取消可用，已存储的历史在正常切换后可用原 `previous_response_id` 续接。`/llm/healthz` 在切换时返回 `busy`，恢复状态不明时返回 `503`。

实现与实机验收进度见[当前计划](plans/avatar-workstation.md)；驱动、资产和资源生命周期见 [services/portrait/README.md](../services/portrait/README.md)。

### 独立图像节点

Gateway 可用 `--remote-images` 将上述生图任务交给专用 worker。提交、归属校验、轮询、取消及下载沿用同一接口；专用节点独立执行准备和生成，端到端耗时包含排队与跨机传输。专用模式不需要文本服务、共享 GPU 卷或 LLM 控制凭据，结果不含同卡切换特有的 `gpu_release`。每个 worker 由一个 Gateway 作业队列驱动，作业元数据和结果保存在该 Gateway 的 image-jobs 卷。
