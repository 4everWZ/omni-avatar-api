# Omni + Avatar 服务接口

调用方从配置取得 API 地址和服务凭据。推理服务负责模型版本、原生双工会话、推理资源和 Avatar 会话。业务身份、任务、记忆、外部工具授权及动作回执由调用应用持有。健康状态和会话握手报告可用能力，应用可以独立升级或替换服务地址。

## 已有推理接口

| 接口 | 用法 |
| --- | --- |
| `GET /livez`、`GET /healthz`、`GET /v1/models` | 分别确认网关存活、上游可用性和模型卡。客户端读取认证模型卡的实际 `max_model_len`，健康状态报告锁定版本和已验证能力。 |
| `POST /v1/chat/completions` | 普通与流式推理共用入口。使用服务配置的模型别名，按实际能力使用视觉、音频和结构化回复。 |
| `POST /v1/chat/prompt_tokens` | 文本按同一聊天模板计量，供需要精确上下文预算的客户端使用。 |
| `GET /v1/realtime?duplex=1` | Realtime WebSocket。以 `session.created` 报告的能力确定输入模态、切片、字幕、打断、恢复和播放确认。 |

Bearer 鉴权、排队、取消、超时和管理接口由[网关规范](../gateway/README.md)维护。原生模型服务的协议和 Avatar 能力独立发布，调用应用根据各自会话实际协商。

## Avatar 会话与音画

Avatar 由独立服务持有，所有外部请求通过共享网关鉴权。用不同客户端创建独立会话，按会话隔离媒体和控制。渲染服务可替换驱动，统一消费带时间戳和代次的音频，输出同步的视频帧及媒体状态。渲染失败时，原生语音和字幕仍可交付。当前实现为 Ava 2D；本地媒体与权限回归通过，工作站共驻验收仍在进行。

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

Realtime 客户端仍使用原协议。启用 Avatar 覆盖配置后，网关额外发送 `gateway.session.created`，提供 `source_session_id` 和 `avatar_available`。将创建参数的 `source` 改为 `{"type":"realtime","session_id":"<source_session_id>"}`，只允许同一应用凭据绑定。内部复制音频和控制事件，不占第二路推理。`?avatar=1` 在连接期间自动进入 live 准入模式，以便另一个调用方同时发普通请求；原有 `program=1` 仍要求媒体专用凭据。

### 音画与事件

- 向 `media_url` POST `application/sdp` 的完整 ICE offer，同时请求音频和视频；201 返回 SDP answer 和 `Location`。首版为非 trickle WHEP，DELETE `Location` 幂等释放订阅。每会话最多两个订阅者。媒体连接协商 Opus 和浏览器支持的视频编码；2D 输出为 640×720、25 FPS、不含 Alpha。
- `X-Avatar-Timeline-Origin-Ms` 将该订阅的媒体时间零点映射到 Avatar 会话时间。音频与视频由同一服务端时间线驱动；字幕事件携带该时间线上的 `start_ms`。
- `events_url` 是鉴权 WebSocket。事件包括 `avatar.snapshot`、`avatar.audio.queued`、`avatar.subtitle`、`avatar.cleared`、`avatar.failed`、`avatar.closed` 与 `avatar.resync_required`。慢消费者收到 resync 后须重新建立会话；字幕与音频均按代次失效。
- `avatar.audio.queued` 提供音频在媒体时间线的 `start_ms/end_ms`、`response_id` 和原回复的 `source_start_ms`。它表示排入媒体时间线，不表示已经播放。
- 消费者可在事件连接提交 `{"type":"playback.ack","generation":0,"played_ms":1000}`。服务保存为 `consumer_report`，不将它等同生成或传输证明。Realtime 历史回执仍由消费者按上游协议发送到原语音连接，使用回复相对位置。
- 打断会清除服务端旧代次缓冲；消费者同时停止本地播放并重建 WHEP 接收器，以丢弃已经进入客户端缓冲的旧帧。关闭源连接会关闭绑定的 Avatar；语音恢复后须用新的来源标识重建 Avatar。

浏览器通过鉴权 `POST /v1/browser-tickets` 提交 `{"path":"/v1/realtime"}` 或具体 `events_url`，获取有效三十秒、限定路径且仅用一次的 WebSocket ticket。浏览器把 ticket 放入对应升级请求，普通 HTTP/WHEP 请求继续使用 Bearer。应用长期凭据不放入 URL。

默认最多四个 Avatar 会话，未活动三分钟回收。浏览器每二十秒查询一次活动会话保活。进程退出释放源连接、会话与媒体。

### 可插拔渲染与后续数字人

`avatar/avatars.json` 登记 `avatar_id`、驱动和素材目录；`OMNI_AVATAR_REGISTRY` 可指定替代登记文件。驱动边界为能力声明、带时间戳和音量的渲染、代次清除及关闭，媒体与权限不依赖具体驱动。首版 `sprite2d` 使用 Ava 的四状态素材。2D 实机验收成功后，再以同一 Ava 为基础实现数字人驱动；数字人模型、视觉形式及性能目标在该阶段确定。推进顺序见[当前计划](plans/avatar-workstation.md)。
