# 推理与媒体基础设施解耦设计

**设计提案，审计基线为 `6d8c074`。** 建议以能力和会话归属路由请求，以独立镜像交付 Omni、LLM、Avatar、Portrait 和可选图像推理服务。工作站上的同卡协作保留为部署选择。应用提交历史、参考素材和推理参数，消费模型输出并决定业务动作。Infra 持有应用级凭据、短期推理状态、媒体资产和算力租约；用户身份、长期记忆、工具授权与执行、任务拆解均由应用持有。现行接口和实机结论分别见 [API](../api.md) 和 [工作站部署](../deployment/workstation.md)。

## 一、现有耦合点审计

| 边界 | 实现证据 | 对部署或调用方的影响 |
| --- | --- | --- |
| Omni ↔ LLM 构建 | [推理 Dockerfile](../../services/inference/Dockerfile) 在同一镜像装 vocoder，复制两个启动器，依次修改上游 Chat、MiniCPM 和 LLM 准入等源码。[LLM 编排](../../deploy/compose/inference.yml) 的 fetch 和 serve 都直接引用工作站 Omni 镜像。 | 两个 HTTP 引擎已分进程，但升级纯文本 vLLM 也要处理 Omni 的依赖和补丁锚点。拆分时须保留同模板计量、结构化输出和思考预算的端到端语义。 |
| 神经渲染／生图 ↔ 推理镜像 | [Portrait Dockerfile](../../services/portrait/Dockerfile) 默认继承工作站引擎，venv 启用 system-site-packages。[Images Dockerfile](../../services/images/Dockerfile) 继承同一镜像，以 no-deps 安装增量包。 | 单独发布应用层仍依赖另一项服务的完整构建产物和 Python ABI。 |
| 设备与算力配额 | [inference.yml](../../deploy/compose/inference.yml) 把 Omni 放到物理卡 1，LLM 放到 0；[portrait.yml](../../deploy/compose/portrait.yml) 和 [images.yml](../../deploy/compose/images.yml) 也选择 0。[LLM 启动器](../../services/inference/llm.py) 固定多项 KV、图捕获和调度默认值；[Klein.free_memory](../../services/images/model.py) 查 NVML index 0。 | 设备隔离后的 CUDA 本地序号、采样设备和实际调度资源需明确映射。仅改 Compose 卡号不足以确保余量判断正确。 |
| LLM ↔ Portrait 帧时序 | Portrait overlay 让两者共享 MPS 控制卷、PID/IPC 命名空间，并依赖 MPS 健康。渲染端发布每槽 JSON 帧时钟，[LLM 调度钩子](../../services/inference/patches/gpu_deadline.py) 按同机单调时钟读余量，派发前最多让出 18 ms。 | 共享计算策略进入了文本引擎热路径；该时钟不能直接拿到另一主机比较。MPS 的启停还涉及客户端重建。 |
| Gateway ↔ Klein ↔ LLM／Portrait | [ImageJobs.run_job / restore](../../services/gateway/image_jobs.py) 准备 CPU 输入、探测 `portrait.lock`、暂停并排空文本准入、写 `image-active`，调用 LLM 的 `release_kv/offload/wake`。[Portrait](../../services/portrait/app.py) 取共享锁，[Images](../../services/images/app.py) 检查授权标记。恢复失败时文本准入保持暂停。 | 即便图像 worker 部署在有空卡的独立节点，现有入口也要求配置文本引擎并执行工作站恢复流程。网关重启还会把未终结的本地作业标为失败。 |
| Gateway ↔ Avatar ↔ Portrait | [Avatar overlay](../../deploy/compose/avatar.yml) 采用 host 网络与控制卷；Avatar 已有 TCP 监听分支，但 [Portrait 客户端](../../services/avatar/portrait_client.py) 经 UnixConnector 访问固定 origin，Portrait 默认仅监听 UDS。启用 Portrait 后 Avatar 的启动依赖其健康。 | 2D 与外部 PCM 本可单独运行。更换 DNS 不能替代两个 socket 卷。现有 WS 已携带起始采样／时间戳和 generation，可沿用消息语义。 |
| 媒体可达性 | [Avatar WHEP](../../services/avatar/app.py) 用 aiortc 和空 ICE server 列表，[浏览器示例](../../services/gateway/web/app.js) 同样配置；[MediaMTX](../../deploy/media/mediamtx.yml) 有单独的 program 发布／订阅路径。 | WHEP 信令可通而媒体 UDP 不可达。当前 Avatar 音画由 Avatar peer 交付，启用节目传输层不能自动解决其候选地址问题。 |
| 路由与副本 | [Config.text_backend](../../services/gateway/config.py) 从必填 Omni 配置派生单个 LLM，固定别名、版本和模型 root；[create_app](../../services/gateway/app.py) 递归挂载 /llm，只有该分支能挂 Images。健康检查匹配模型和路径，WS 中转只改写特定模型别名。 | 单 URL 可换主机，但没有同能力实例选择、独立注册或无 Omni 的启动配置。网关进程内 FIFO、token 预留、ticket 和能力缓存不能直接随副本数复制。 |
| 有状态对象 | [ResponsesProxy](../../services/gateway/responses_proxy.py) 签名校验调用方，以后台 monitor 持有准入；历史位于引擎进程。Avatar 的 sources、sessions、WHEP peer 在进程内；Portrait 的原图、作业 JSON、准备特征在本地卷，图像结果在 Gateway 卷。 | 任意轮询副本会丢续接、取消、归属或资源计数。模型名相同也不能恢复另一进程的 KV 或媒体连接。 |
| 底层参数 | [DittoBackend.setup / prepare](../../services/portrait/backend.py) 固定 online_mode=True、overlap_v2=75、10 步，使用固定裁剪参数准备底图；公开 API 已有 reference_asset_id。[Realtime relay](../../services/gateway/app.py) 可转发上游 session.update，已有窗口验收；仓库缺少逐后端 VAD 参数适配与验收。 | 有效默认值、可改字段、设置阶段及后端是否实际消费字段，均需通过能力契约表达。 |
| 编排与验收入口 | [tools/compose.py](../../tools/compose.py) 的 TARGETS 预组服务，Portrait／Images 限 workstation；[base.yml](../../deploy/compose/base.yml) 混合 ARM64 引擎、Gateway、共享网络与模型元数据。[资格验证工具](../../tools/README.md) 通过公共 API 加载，但 GPU 采样跟随采集主机。 | 单节点组合和验证仍有价值；跨节点需要分别定位服务和采样器，避免把客户端显存误记为服务端显存。 |

## 二、服务边界、通信与部署

| 拟议目录／发布单元 | 独立负责的资源 | 对端与协议 |
| --- | --- | --- |
| `services/gateway/` | 鉴权、路径兼容、应用归属、参数规范化、流转发；保留浏览器为协议示例 | HTTP 查询调度与申请租约；向选中实例代理 HTTP/SSE、WS。WHEP 信令在这里鉴权。 |
| `services/scheduler/` | 能力注册表、带时效的实例负载、原子资源预留、归属索引、队列与 drain；仅保存调度元数据 | 内网 HTTP 注册／心跳／续租／归还。首版单逻辑写入者，租约和绑定存事务存储；增加网关副本不增加配额。 |
| `services/omni/`、`services/llm/` | 各自拥有 fetch、锁定的镜像和权重 revision、启动器、配置、引擎适配和本服务测试。Omni 管多模态阶段、vocoder、实时 KV；LLM 管文本生成、Responses 状态与精确 token 计量 | Chat HTTP/SSE；Omni 原生双工 WS；LLM Responses HTTP/SSE。两边按同一份规范独立实现自己声明的能力。 |
| `services/avatar/` | 来源与媒体会话、2D renderer、时间线、字幕、消费者回执、WHEP peer 及 ICE | 接受显式 PCM 或 Gateway 的有界 source WS 分支；向 Portrait 发起 render WS。浏览器向 Gateway POST SDP，随后通过 ICE 直接或经 TURN 与媒体节点收发 WebRTC。 |
| `services/portrait/` | Ditto 参考图登记／预处理、形象版本、神经渲染槽和帧生成；自有 Torch/CUDA 依赖，TensorRT 为独立变体 | Gateway 可直接代理资产与准备任务的 HTTP；Avatar 按形象 ID 打开私有 render WS，预热完成再 ready。准备和渲染共用本实例槽池。 |
| `services/images/` | Klein 作业、CPU 编码、GPU 推理、取消、结果 TTL 和归属。将现有 ImageJobs 的作业生命周期从 Gateway 迁入 | HTTP 提交／查询／取消，队列只传任务 ID 和受控资产引用。独立 worker 自取本池租约；单实例可先用服务自有队列。 |

**独立构建与启动。** 每个服务目录有自己的 Dockerfile、依赖锁、版本化配置、健康检查及镜像摘要。通用 CUDA/Torch 基础层可按 ABI 单独发布并固定 digest。将 MiniCPM AWQ／滑窗／duplex 和 JetPack 构建归 Omni，将 Qwen parser、Responses 容量计算归 LLM；两个服务都依赖的思考语义以协议夹具分别验证。镜像构建从干净缓存只需自身目录和已发布基础层；模型下载与离线 serve 分开。

```text
deploy/
  services/{gateway,scheduler,omni,llm,avatar,portrait,images,program-media}.yml
  profiles/{workstation,orin,orin-cu129}/       # 已测运行参数与架构变体
  placement/                                  # 站点自有 endpoint、设备选择、ICE、卷
  policies/                                   # 别名、能力要求、资源池、队列与降级
tools/compose.py                              # 按服务和 profile 生成单节点命令
tools/qualification/                          # 按 endpoint 压测，按节点采样
```

每个服务清单可独立成 Compose project，节点运维选择镜像、平台、设备 UUID／运行时分配、模型缓存和监听地址。workstation 组合入口继续可复现旧部署；服务可各自执行 build、up、stop 和回滚。仅声明服务自身所需 secret。引擎启动后注册自己的协议版本、模型 revision、能力与 readiness；Gateway 在后端集为空时仍可存活，按能力报告可用性。显式注册或运维配置的 DNS/HTTPS endpoint 取代跨 project 容器名；服务间认证并校验对端身份，调用方提供的 owner、内部地址和算力控制头一律由网关剥离。ICE 发布地址、UDP 端口范围、STUN/TURN 和短期凭据归媒体站点配置。

**跨节点媒体与资产。** 保留现有 WS 数据面，协商协议版本、音频格式、时间单位、帧编码和最大消息长度。现有输入为网络序 `!QI` 的起始采样与 generation 加 PCM，输出为 `!dI` 的会话相对秒数与 generation 加 JPEG；清除、ready、失败走控制消息。用源 epoch、媒体代次和相对采样位置丢弃迟到包；慢消费者走有界缓冲与 resync，原生语音旁路继续。WHEP 的 Location、timeline origin 和 playback.ack 始终对应同一媒体实例。同机可选 UDS 或协商共享内存，跨节点以 HTTP/WS 为必测路径。队列用于异步准备和生图，逐帧链路无需消息代理。

参考图和就绪形象保留既有归属、依赖删除及保留语义。先按资源 ID 路由回资产所有者；跨节点迁移时以 owner 隔离的对象存储保存原图和任务结果，元数据记录输入版本、预处理参数和模型 revision。目标节点校验后从原图重建特征，按设备／精度／运行时版本本地重建 TensorRT 缓存；pickle 只信任服务生成的数据。临时响应 ID、一次性 ticket、会话绑定采用带期限的独立记录，浏览器 ticket 原子消费。日志记录请求 ID、耗时和资源量，避免记录提示词、音视频与凭据。

**升级与迁移顺序。** 先用现有接口夹具固定单后端行为，再拆镜像和服务清单；随后启用跨主机 TCP 与媒体测试，再切换注册、准入和作业所有权，最后接入可选 Offload。迁移已有卷时先停止资产写入，拷贝并抽查自有 ID 的查询、预览和删除依赖，成功后切换路由。滚动升级时新实例预热并通过能力探测，旧实例停止新租约，保留旧版本协议和绑定直到会话、后台任务结束或 drain 期限到达。活动 WS、WebRTC peer 与内存 KV 需要在原实例结束或显式重建；回滚让新请求重新进入旧版本，已建立的流继续归原实例。

## 三、Gateway 契约与参数透传

**入口兼容。** 保留 `/v1/chat/completions`、`/v1/chat/prompt_tokens`、`/v1/realtime`、`/llm/v1/{chat/completions,responses,models}`、Responses 查询／取消以及现有 Avatar／Images 资源路径。旧前缀分别解析为 Omni 和 LLM 默认能力池，显式别名继续约束目标。外部应用仅配置 Gateway 地址与 Bearer。新增鉴权的 `GET /v1/capabilities?model=<alias>&task=<task>`，按别名和任务列出可选 profile、schema 版本、字段类型／边界／默认值／可变阶段、支持的组合、实际 revision 与临时可用性。多个异构实例按 profile 分列，以免把字段并集误报为任一实例均支持。

**扩展格式。** Chat／Responses 保持标准顶层字段。SDK 的 `extra_body` 是合并到 HTTP JSON 顶层的调用参数；以下是合并后的原始请求。新增 `infra` 为网关保留命名空间，消费后移除；Realtime 的 `session.extra_body` 是协议中真实存在的对象。适配器按任务、引擎版本和能力白名单映射原生字段。显式值优先于 profile 默认，再采用该引擎默认；布尔 false、0、-1 和省略分别处理。重复表达同一参数且值冲突、未知扩展、越界或创建后修改只读字段均报具体字段路径。部署级显存、权重路径、设备、可执行代码和内部服务地址不在调用契约内。

```json
{
  "model": "llm-default",
  "input": "将输入整理成三项。",
  "stream": true,
  "max_output_tokens": 768,
  "thinking_token_budget": 128,
  "chat_template_kwargs": {"enable_thinking": true},
  "infra": {
    "version": 1,
    "routing": {"task": "text.responses", "offload": "allow", "fallback_models": []}
  }
}
```

| 字段／阶段 | 拟议约束与适配 |
| --- | --- |
| `infra.version`、`routing.task` | 使用扩展时版本为整数 1。task 省略则由入口及输入模态推导，显式指定必须与端点一致。例如 text.responses、omni.duplex、portrait.render、image.generate、image.edit。图生图要求参考输入，任务选择不触发提示词分类或业务子任务拆解。 |
| `routing.offload`、`fallback_models` | offload 为 disabled（默认）、allow、require。默认仍可在所选模型主池的等价副本间均衡；allow 可选合格的任务卸载池，require 仅选卸载池。fallback_models 为有序的显式允许别名列表，默认空；每个候选仍须满足全部模态、字段、输出协议、上下文及应用策略。 |
| LLM 每请求 | 保留 thinking_token_budget（整数 ≥0 或 -1）、模板开关、reasoning.effort／reasoning_effort、Chat 最大生成数及 Responses max_output_tokens。思考和正文共享输出上限；include_reasoning 只控制返回。按最终后端模板计量历史、tools、多候选及输出预留，切换候选必须重算。工具调用仅输出 function 描述，由应用执行。 |
| Omni 初始化 | basic 窗口要求整数 0 < low < high ≤ 此 profile 的有效窗口，并留生成余量；context 模式支持 context_max_units、context_previous_max_tokens；off 由调用方轮换。模型最大窗口由部署决定。VAD 默认 native 使用模型原生监听／说话策略；server_vad 及其 threshold（0–1）、silence_duration_ms（正整数）仅在适配器声明实际支持及范围后开放。 |
| Portrait 准备与运行 | generations 请求绑定必填 reference_asset_id（已上传、同 owner）、driver 和可选裁剪参数。底图方向规范化后执行裁剪，参数进入形象缓存键。会话创建可指定 online 布尔值，true 映射 Ditto online_mode=True；当前候选仅声明 true，false 需另有支持该模式的 profile。一次会话固定形象和准备版本；换底图须准备新形象并创建会话。 |

**实时初始化示例。** 对提出初始化要求的连接使用 `/v1/realtime?duplex=1&autostart=0`。网关在限定时间内接收首条更新，完成选路和校验后才创建上游会话；在 ready 前禁止音频入队。旧的自动启动路径继续使用已发布默认值。

```json
{
  "type": "session.update",
  "session": {
    "model": "omni-default",
    "modalities": ["audio", "text"],
    "playback_commit_policy": "ack_only",
    "extra_body": {
      "sliding_window_mode": "basic",
      "basic_window_high_tokens": 8000,
      "basic_window_low_tokens": 6000,
      "infra": {"version": 1, "vad": {"mode": "native"}}
    }
  }
}
```

**Portrait 调用示例。** 先 POST `/v1/avatar/assets` 上传 PNG/JPEG，保留当前 4 MiB、800 万像素限制。将返回 ID 用在下面的两个独立请求中；第二步的 avatar_id 取自第一步准备任务达到 ready 后的响应。

```json
{"driver":"ditto","reference_asset_id":"<asset-id>","infra":{"version":1,"portrait":{"crop_scale":2.3,"crop_vy_ratio":-0.125}}}
```

```json
{"avatar_id":"<ready-avatar-id>","source":{"type":"audio"},"infra":{"version":1,"portrait":{"online":true}}}
```

第一个请求发往 `/v1/avatar/generations`。crop_scale 为正有限数，crop_vy_ratio 为有限数，具体范围由 registrar profile 公布。第二个发往 `/v1/avatar/sessions`；若输入是原生语音则改用归属校验后的 realtime source ID。仅在渲染线程实际预热完成后报告 ready。

**调度流程与可观察结果。**

1. 后端实例登记 instance_id、启动代次、服务／协议版本、模型别名与 revision、支持任务和参数、最大上下文、资源池、专用或共享资源组，周期报告 ready/draining、空闲 slots、token 预算、队列等待和渲染余量。过期实例退出新请求候选。先按 owner／既有绑定、语义与资源约束过滤，再按预计排队加执行时间选择；同条件优先已预热实例。待机卸载节点单列冷启动成本。
2. 调度层按实例原子预留 slots／tokens／渲染槽，工作进程以 request_id、租约和单调递增的 fencing token 幂等接收。各能力池限长 FIFO、设等待期限与租户并发上限；超时取消还在排队的预留。后台 Responses 断流后继续占额，前台断流传播取消，直到工作结束或确认释放。租约超期先隔离实例并对账，确认前禁止再分配同一资源；网络分区时已建立媒体流可继续，新准入无法确认则失败。
3. 任务 Offload 在请求边界发生。例如 `POST /v1/images/jobs` 传 prompt、seed、可选 reference_image_base64 与 `infra.routing`：空闲时可路由到登记 `image.generate/edit` 的 FLUX.2-klein-4B；文本或 duplex 请求不能路由到该模型。独立图像池无需发现或唤醒 LLM。另一轻量文本模型须单独登记文本能力、参数与质量评测，且由调用方允许对应别名。
4. **同卡显存让渡**只在站点配置的共享资源组启用。由节点代理串行处理该组的 drain、按需释放 KV／权重、按设备 UUID 复查真实余量、派发、回收和唤醒；引擎仅在注册支持控制协议时参与。取消或崩溃后先确认 GPU 作业退出及 LLM ready，再释放栅栏；失败隔离该资源组。MPS、帧余量反馈及共享内存仅为同机优化；独占池和跨机池跳过整个切换过程。
5. 为 Responses ID（含 previous_response_id）、source、Avatar session、WHEP peer、准备任务和图片任务保存 owner → 实例／启动代次绑定。读取、取消、删除、续接按绑定执行，跨 owner 返回 404。新创建请求以应用作用域的幂等键去重；提交结果不明时先查原操作。SSE 已输出或 WS 已创建后不自动换模型重放。旧实例状态丢失时由客户端用自有历史新建会话；后台作业恢复按持久状态判定，无法重建的任务显式失败。
6. 支持扩展的 HTTP 响应头 `X-Infra-Request-Id`、`X-Infra-Model`、`X-Infra-Profile` 标明实际选择；协商扩展的客户端在 WS ready／作业状态中取得 effective_params、模型 revision、offload 与降级原因。保留原生 SSE 事件格式与结束语义，不把调度事件塞进模型文本。

| 故障／条件 | 调用方可见行为 |
| --- | --- |
| 参数非法／候选从未支持该能力 | 400，含 error.code、param、request_id；已升级的 WS 返回参数错误事件并终止这次创建。 |
| 合格池满／后端未启用或暂时不健康 | 队列到限或等待超时 429 + Retry-After；无健康合格后端 503。require 在无可用卸载节点时失败；allow 可返回主池。 |
| 超过上游期限／进行中断流 | 建流前 504；建流后关闭失败的连接，保留已发增量，不合成成功结束事件。 |
| Portrait、Avatar 或 source tap 故障 | 直接申请失败的能力返回错误；附加媒体失败报告 avatar_available=false 或 avatar.failed，原生音频／字幕继续。已选 Ditto 不静默替换形象；调用方可显式新建 2D 会话。 |
| ID 丢失／跨应用／旧媒体代次 | 跨应用与未知 ID 对外统一 404；已确认归属且实例代次失效的扩展会话报告 session_lost。clear 后丢弃旧代次帧，客户端清空本地缓冲并按协议重建 WHEP。 |

## 四、锁定资源配额前的验证

以相同 checkpoint、精度、输入夹具、随机种子、输出预算和媒体规格，依次测独占、同机分设备、同卡共驻、跨节点。记录服务镜像、设备 UUID、实际采集节点、驱动、并发到达模式和网络条件。冷启动与热态分组；逐级加负载至首次违约，再在拐点两侧重复并持续运行。报告样本数、分位数、失败和排队，不把并发提交数当成同时执行数。下表中已有数字仅作为对应工作站夹具的历史比较线。

| 试验与入口 | 变量、观测量与判定用途 |
| --- | --- |
| 独立交付与恢复 | 在干净节点单独构建／启动每个镜像；只启 LLM、只启 Images、只启 Avatar + PCM，再异机组合 Omni + Avatar + Portrait。滚动替换一个服务、撤销注册、重启 Gateway，验证其余能力可用、ticket 单用、归属隔离、ID 续接／失效及队列配额。 |
| LLM 窗口与并发 | 用 [inference](../../tools/qualification/inference.py)、[thinking](../../tools/qualification/thinking.py)、[concurrency](../../tools/qualification/concurrency.py) 测短／长／接近上限、续接、多个候选、思考开关与预算。记录计量与 usage、一／多路实际输出重叠、排队、TTFT、token 间隔、每路／总 TPS、KV 与主机峰值；验证 400、429、取消回收及结构化输出。用长度×并发曲线定 slots 与 token 池。 |
| Omni 实时性 | 用 [omni_throughput](../../tools/qualification/omni_throughput.py) 和双工夹具测 Chat + WS、双 WS、跨窗口长通话，比较 basic/context/off、不同音视频输入和已支持 VAD。记录首音频、静音误触发、漏切、打断到停播 p95/p99、队列、窗口回收后的检索正确率与峰值显存。选取单路延迟和并发均能满足的配置。 |
| Portrait 与媒体 | 用 [portrait](../../tools/qualification/portrait.py)、[portrait_frame](../../tools/qualification/portrait_frame.py) 比较不同底图、裁剪、在线准备、冷／热预热、CUDA Graph／TensorRT、单／双槽。分别测真实生成 FPS、WHEP 解码 FPS、重复帧、音画偏差、输入提前量、JPEG 带宽、画质与口型。现有夹具要求 25 fps、最大采样偏差 <160 ms；工作站双路隔离记录为 11.5／16.7 fps，不能据此开放双路配额。[已有结论](../plans/avatar-workstation.md) |
| 混跑与路由 | 用 [throughput](../../tools/qualification/throughput.py)、[workstation_capacity](../../tools/qualification/workstation_capacity.py) 比较各池独占、共驻、可选 MPS／帧反馈；在高预填充和突发图像请求下测媒体 deadline miss、每应用饥饿、路由等待及拒绝。重放旧夹具时保留空闲单路 45 TPS、通话四路每路 15 TPS 的对应检查线，另行商定新站点 SLO。 |
| Klein 与显存让渡 | 用 [images](../../tools/qualification/images.py) 对同种子文生图、真人／动漫参考图测独立 worker、远端卸载、同卡 KV-only／权重让渡；计 CPU 编码、上传、排空、生成、恢复 LLM 各阶段耗时、整卡显存与主机 RSS 峰值、传输量及图像质量。比较关／允／强制 Offload 的命中与总延迟；在取消、显存不足、唤醒失败后核验原 Responses 历史和其他池可用。 |
| 跨节点媒体与故障注入 | 加 RTT、抖动、限带宽、丢包，覆盖浏览器直连／TURN、非 trickle WHEP、清除与重订阅；记录 ICE 成功率、建联及首帧 p95、字幕／音画偏差、回执、CPU 编解码与网络吞吐。模拟租约过期、任务重复、节点失联、网关切换和慢消费者；要求零跨 owner 读取、零旧代次播放、零重复执行／超卖，恢复后无遗留租约。浏览器麦克风、实际扬声器与口型保留人工验收。 |

**资源评估产物。** 每个服务／模型 revision／精度／设备类别给出“可承载负载 → 尾延迟、质量、显存峰值、CPU/RAM、网络、失败率”实测表，单列同时满足的混跑组合、冷启动成本和故障恢复时间。按业务方给出的请求分布与延迟目标选择通过全部约束的最高负载，在观测到的波动和工作区峰值之外留余量，再发布池级 slots、token 预算、队列上限、超时与放置策略。新硬件、模型或驱动重新测量；本提案不预设卡号或最终配额。
