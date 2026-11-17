# Omni + Avatar 服务接口

调用方从配置取得 API 地址和服务凭据。推理服务负责模型版本、原生双工会话、推理资源和 Avatar 会话。业务身份、任务、记忆、外部工具授权及动作回执由调用应用持有。健康状态和会话握手报告可用能力，应用可以独立升级或替换服务地址。

当前采用显式模型分工：`/v1` 的 `omni-default` 负责约 8K 近期窗口的多轮实时助手；`/llm/v1` 的 `llm-default` 提供独立后台长上下文推理，两者有各自的健康状态、模型卡和准入队列。所有服务和 Avatar 均在当前双 3090 工作站。Chat 调用方通过 messages 提交历史和所选记忆；服务不存储业务记忆，不运行 Agent、不执行返回的工具调用，也不自动切换模型。

## 已有推理接口

| 接口 | 用法 |
| --- | --- |
| `GET /livez`、`GET /healthz`、`GET /v1/models` | 分别确认网关存活、上游可用性和模型卡。客户端读取认证模型卡的实际 `max_model_len`，健康状态报告锁定版本和已验证能力。 |
| `POST /v1/chat/completions` | 普通与流式推理共用入口。使用服务配置的模型别名，按实际能力使用视觉、音频和结构化回复。 |
| `POST /v1/chat/prompt_tokens` | 文本按同一聊天模板计量，供需要精确上下文预算的客户端使用。 |
| `GET /v1/realtime?duplex=1` | Realtime WebSocket。以 `session.created` 报告的能力确定输入模态、切片、字幕、打断、恢复和播放确认。 |

Omni 已验证 20 条消息、7,928 个输入 tokens 的多轮历史检索；超限输入返回 400，调用方可先计量再选择保留的历史与记忆。计量接口允许计算超出窗口的文本，用于压缩预算。[8K 边界证据](../evaluations/results/omni-8k-multi-turn-workstation.json)。

Bearer 鉴权、排队、取消、超时和管理接口由[网关规范](../services/gateway/README.md)维护。原生模型服务的协议和 Avatar 能力独立发布，调用应用根据各自会话实际协商。

## 后台 Responses

使用 `POST /llm/v1/responses`，模型 `llm-default`，设置 `stream: true` 接收原生增量 SSE。支持文本、多轮 `previous_response_id`、JSON Schema 及 function 工具调用输出；工具由应用执行。`GET /llm/v1/responses/{id}` 与 `POST /llm/v1/responses/{id}/cancel` 只允许原调用方访问，其他调用方得到 404。

当前实现 vLLM 0.32 的 HTTP／SSE 子集，不承诺云端 Responses 全能力。保存的响应与续接关系位于引擎内存，重启即失效，不是业务记忆。后台任务在客户端断开后保留准入直到结束或取消；前台流式请求断开会取消。详见[网关规范](../services/gateway/README.md)。

## 外部 Swarm／orchestrator 调用

外部编排器统一使用后台 `base_url=http://100.119.17.6:8000/llm/v1`、模型 `llm-default` 和应用 Bearer，优先调用 Responses；Chat 保留兼容。每个逻辑 worker 单独持有历史或 `previous_response_id` 分支；任务、交接、工具执行、共享记忆和工作区操作由应用项目负责。

LLM 单请求上限保留 262144 tokens，引擎和网关默认允许 4 个生成序列；短请求四路并发是验收目标，不承诺四路满窗口。所有活跃 Chat／Responses 共用 `LLM_TOKEN_BUDGET=262144`：每次准入预留完整模板输入（含 instructions、tools 和已保存续接历史）加最大输出，Chat 多候选分别计入序列和预算。计量来自同一引擎渲染器，不用字符数估算。模型权重保持 8bit，KV 和 SSM 状态维持现有精度。

调用方应显式设置 `max_output_tokens`（Chat 为 `max_completion_tokens`／`max_tokens`），以释放短请求并发空间；省略时保留引擎默认输出上限，可能按整个剩余窗口预留并串行运行。长短请求按 FIFO 等待可用序列与 token；达到队列上限或等待超时返回 429 和 Retry-After，单请求预算本身不可容纳返回 400。`/llm/api/mode` 报告 `reserved_tokens`、`available_tokens`、`token_budget`、活跃序列和排队数。后台及断流保留规则同上，结束或取消后释放预算。

`LLM_MAX_NUM_SEQS` 同时配置引擎和网关（默认 4）；`LLM_TOKEN_BUDGET` 是全局活跃预算，不改变模型窗口。`LLM_QUEUE_LIMIT`、`LLM_QUEUE_TIMEOUT`、`LLM_REQUEST_TIMEOUT` 可独立设置后台排队和总期限，默认继承现有 16／5 秒／1200 秒；Omni 的实时通道保持独立。提高并发或预算必须重新验证显存和头像共驻表现。

启用神经头像覆盖时，`LLM_GPU_STEP_PAUSE_MS=250` 让每个真实 GPU 批次完成后再让出执行时间，并关闭 LLM 异步批次重叠；同一批次仍可生成四个序列，SSE 按实际生成即时转发。此配置优先保证 25 fps，实测四路各输出 256 tokens 时约 3.4 tokens/s/路。未启用神经头像覆盖时默认不让出该时间。该计算配额与 token 显存预算分别生效；当前保留 8bit 权重、浮点 KV 和 FP32 SSM。

当前验收范围是 provider API，具体 Antigravity／agent-orchestrator 客户端接入留在应用项目。[provider 约定](../services/gateway/README.md#external-orchestration-clients)。

## Avatar 会话与音画

Avatar 由独立服务持有，所有外部请求通过共享网关鉴权。用不同客户端创建独立会话，按会话隔离媒体和控制。渲染服务可替换驱动，统一消费带时间戳和代次的音频，输出同步的视频帧及媒体状态。渲染失败时，原生语音和字幕仍可交付。当前 Ava 2D 已通过分卡配置下约 20 分钟原生双工与后台 LLM 共驻检查；参考图驱动的会议式神经头像也已通过 25 fps、同步、归属隔离及取消恢复检查；浏览器实物交互仍需人工确认。[当前验收结论](deployment/workstation.md#实机准入)。

| 会话操作 | 目标契约 |
| --- | --- |
| `POST /v1/avatar/sessions` | 从获准的 Realtime 会话或显式音频源创建 Avatar 会话。指定已登记的 `avatar_id`，服务校验来源归属、渲染能力与资源，返回会话 ID、实际格式和受限媒体订阅入口。 |
| `GET /v1/avatar/sessions/{id}` | 查询状态、关联语音会话、已协商的输出能力和失败原因。调用方只读自己或已获授权的会话。 |
| `DELETE /v1/avatar/sessions/{id}` | 幂等关闭渲染和媒体订阅，释放该会话资源。关联语音会话的关闭和恢复遵循 Realtime 生命周期。 |
| 音频、字幕与画面 | 首版用 WebRTC/WHEP 交付浏览器可消费的音画，并在会话事件中提供字幕、播放代次和时间戳。样式、编码、Alpha 或色键按协商报告。 |
| 打断和播放 | 语音取消或播放清除使旧代次的未播音频、字幕和帧失效。服务区分已生成、已传输和客户端确认已播。播放回执由实际消费者提交。 |

客户端直接验收启动、语音和画面同步、插话、失效代次、渲染故障时语音继续、授权隔离和正常释放。用两个独立调用方验证会话及凭据隔离。OBS 可作为后续订阅者，负责节目布局、合成和推流。

### 首版调用

所有请求沿用应用 Bearer；拥有管理令牌不代表可以读取其他应用的会话。不存在与无权访问的会话均返回 404。会话 ID 不是媒体访问凭据。

```json
{"avatar_id":"ava","source":{"type":"audio"}}
```

创建返回 201，包含 `id`、`generation`、`state`、`format`、`media_url`、`events_url` 和 `audio_url`。显式音频通过 `POST audio_url` 提交 `{ "audio":"<base64>", "generation":0, "format":"pcm16", "sample_rate_hz":24000 }`，仅接受单声道 PCM16。每块最多五秒，未播缓冲最多三十秒。`{"type":"clear"}` 清除未播媒体并递增代次；旧代次输入返回 400。

Realtime 客户端仍使用原协议。客户端以 `?avatar=1` 选择 Avatar 后，网关额外发送 `gateway.session.created`，提供 `source_session_id` 和 `avatar_available`。将创建参数的 `source` 改为 `{"type":"realtime","session_id":"<source_session_id>"}`，只允许同一应用凭据绑定。内部复制音频和控制事件，不占第二路推理。`?avatar=1` 在连接期间自动进入 live 准入模式，以便另一个调用方同时发普通请求；原有 `program=1` 仍要求媒体专用凭据。未选择 Avatar 的 Realtime 连接保持原事件流。

### 音画与事件

- 向 `media_url` POST `application/sdp` 的完整 ICE offer，同时请求音频和视频；201 返回 SDP answer 和 `Location`。首版为非 trickle WHEP，DELETE `Location` 幂等释放订阅。每会话最多两个订阅者。媒体连接协商 Opus 和浏览器支持的视频编码；2D 输出为 640×720、25 FPS、不含 Alpha。
- `X-Avatar-Timeline-Origin-Ms` 将该订阅的媒体时间零点映射到 Avatar 会话时间。音频与视频由同一服务端时间线驱动；字幕事件携带该时间线上的 `start_ms`。
- `events_url` 是鉴权 WebSocket。事件包括 `avatar.snapshot`、`avatar.audio.queued`、`avatar.subtitle`、`avatar.cleared`、`avatar.failed`、`avatar.closed` 与 `avatar.resync_required`。`source_epoch` 提供上游打断栅栏（显式音频源为 -1），用于与原语音事件对齐。慢消费者收到 resync 后须重新建立会话；字幕与音频均按代次失效。
- `avatar.audio.queued` 提供音频在媒体时间线的 `start_ms/end_ms`、`response_id` 和原回复的 `source_start_ms`。它表示排入媒体时间线，不表示已经播放。
- 消费者可在事件连接提交 `{"type":"playback.ack","generation":0,"played_ms":1000}`。服务保存为 `consumer_report`，不将它等同生成或传输证明。Realtime 历史回执仍由消费者按上游协议发送到原语音连接，使用回复相对位置。
- 打断会清除服务端旧代次缓冲；消费者同时停止本地播放并重建 WHEP 接收器，以丢弃已经进入客户端缓冲的旧帧。关闭源连接会关闭绑定的 Avatar；语音恢复后须用新的来源标识重建 Avatar。

浏览器通过鉴权 `POST /v1/browser-tickets` 提交 `{"path":"/v1/realtime"}` 或具体 `events_url`，获取有效三十秒、限定路径且仅用一次的 WebSocket ticket。浏览器把 ticket 放入对应升级请求，普通 HTTP/WHEP 请求继续使用 Bearer。应用长期凭据不放入 URL。

默认最多四个 Avatar 会话，未活动三分钟回收。浏览器每二十秒查询一次活动会话保活。进程退出释放源连接、会话与媒体。

### 可插拔渲染与后续数字人

`services/avatar/avatars.json` 登记 `avatar_id`、驱动和素材目录；`OMNI_AVATAR_REGISTRY` 可指定替代登记文件。驱动接收带采样起点和代次的原始 PCM，按会话时间戳提供画面，并实现能力声明、代次清除和关闭；音频入队与取帧应保持非阻塞，神经驱动可在独立 worker 推理。媒体与权限不依赖具体驱动。首版 `sprite2d` 使用 Ava 的四状态素材和共享时间线的音量。2D 实机验收成功后，再以同一 Ava 为基础实现数字人驱动；数字人模型、视觉形式及性能目标在该阶段确定。推进顺序见[当前计划](plans/avatar-workstation.md)。

## 参考图与会议式头像

参考图驱动的会议式说话头像采用可插拔 Ditto 驱动，不强制 3D。启用 `deploy/compose/portrait.yml` 后，所有请求沿用应用 Bearer，通过网关进入私有准备／渲染服务；未启用时返回 503。参考图和生成结果仅原调用方可访问，其他调用方得到 404。

| 接口 | 行为 |
| --- | --- |
| `POST /v1/avatar/assets` | JSON `{"mime_type":"image/png","image_base64":"..."}`，也支持 JPEG。最多 4 MiB、800 万像素，解码后保留方向与比例并填充至 640×720。返回 201 和参考图 `id`。使用清楚的单人正面头像。 |
| `DELETE /v1/avatar/assets/{id}` | 删除未被任务引用的参考图；有依赖时返回 409。 |
| `POST /v1/avatar/generations` | `{"reference_asset_id":"...","driver":"ditto"}`，返回 202 和任务 `id`。准备人脸、外观和运动特征；不根据图片臆造一个可多视角旋转的 3D 模型。 |
| `GET /v1/avatar/generations/{id}` | 状态 `queued → preparing → ready`；失败为 `failed`，取消为 `cancelled`。仅 `ready` 返回可用 `avatar_id`。 |
| `DELETE /v1/avatar/generations/{id}` | 活跃准备任务标为 cancelled，GPU 工作实际退出前继续占用配额；再次删除可清理记录。就绪任务删除形象及特征，若形象正在通话则返回 409。 |
| `GET /v1/avatar/avatars/{avatar_id}` | 查询已准备形象和驱动能力。`/preview` 返回规范化参考图预览。 |
| `POST /v1/avatar/sessions` | 将就绪的 `portrait_...` 作为 `avatar_id`，仍按上述方式绑定 `audio` 或自有 `realtime` 源；返回同一套音频、事件及 WHEP 入口。 |

最多一路神经渲染与参考图准备共享 GPU 配额；忙时创建神经会话返回 429，准备任务可以排队。每调用方最多保留 16 张图、16 个任务，总量各 64；客户端删除不用的资产。准备形象持久化，活跃会话不跨重启恢复。不同驱动返回各自实际能力，未来的 3D 驱动沿用同一接口。

神经会话创建时先完成实际渲染线程的预热再就绪；等待创建成功后再向绑定源发送需要形象播放的音频。输出 640×720、25 fps；音频播放预留 1200 ms 推理时间，`format.input_lead_ms` 明示。音画共用时间线，原生 Omni 音频仍独立输出。清空立即停播旧音频并丢弃旧代次视频，神经运动状态允许自然过渡。落后或停滞会发出 `avatar.failed`，调用方继续消费原生语音。会话 `renderer` 字段报告实际 `generated_frames`、最新生成及已显示帧时间，可用于检查是否真正达到实时生成速度。

神经头像 120 秒共驻检查通过：3,012 张实际生成帧，最大音画时间线采样偏差 58 ms；重启后的准备、取消及恢复通过。[功能证据](../evaluations/results/portrait-functional-workstation.json) · [生命周期证据](../evaluations/results/portrait-lifecycle-workstation.json)。该结果覆盖轻量后台请求，不代表持续满载 QoS 或主观口型质量均已验收。

实现与实机验收进度见[当前计划](plans/avatar-workstation.md)；驱动、资产和资源生命周期见 [services/portrait/README.md](../services/portrait/README.md)。
