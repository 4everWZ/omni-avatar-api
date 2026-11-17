# 双 RTX 3090 工作站部署

当前目标是推理服务分工：GPU 0 承载后台长上下文 LLM，GPU 1 承载实时 Omni 的 Thinker、Talker、Code2Wav；Avatar 也在这台工作站，2D 当前使用 CPU。模型与 Avatar 均暴露 API，业务方持有多轮历史、记忆、Agent 调度、工具授权和执行。工作站服务不替调用方实现这些业务逻辑。

## 主线组合

叠加 `infra-compose.yml`、`infra-workstation.compose.yml`、`infra-avatar.compose.yml`、`infra-inference.compose.yml`。前三文件组合保留为旧两卡 Omni 复现配置。基础 ARM64 与 BF16 候选文件不作为新的主线启动组合。

```sh
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml -f infra-inference.compose.yml config --quiet
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml -f infra-inference.compose.yml build omni-engine omni-gateway omni-avatar
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml -f infra-inference.compose.yml --profile assets run --rm model-fetch
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml -f infra-inference.compose.yml --profile assets run --rm llm-model-fetch
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml -f infra-inference.compose.yml up -d
```

模型资产使用精确快照和独立目录，服务离线加载；两个引擎端口均在 Docker 私网。宿主默认只发布回环 8000。`.env` 中配置独立的应用、管理、Avatar 内部令牌，不打印或提交凭据。

## API 与上下文

| 入口 | 责任 |
| --- | --- |
| `/v1`，`omni-default` | 语音／视觉助手、普通和流式 Chat、原生 Realtime；Chat 窗口 8192，Realtime 默认 basic 水位 8000／6000。 |
| `/llm/v1`，`llm-default` | 以 Responses HTTP／增量 SSE 为主的长上下文推理，兼容 Chat；结构化回复与工具调用输出。固定 9B 8bit，配置窗口 262144，已验证 22,753-token 多轮续接。 |
| `/healthz`、`/llm/healthz` | 分别报告两个后端的身份、根目录校验、就绪和独立准入；一个后端异常不代替另一个的健康结论。 |
| `/v1/avatar/*` | 同宿主 Ava 渲染、音频源绑定、WHEP、字幕、代次和会话生命周期。 |

Chat 无跨请求业务记忆：应用每次提交希望使用的 messages；Realtime 由原生引擎保留活跃近期历史并按策略滑动。调用方可将外部记忆选入提示。服务不提供记忆库、自动摘要、Agent 循环或工具执行。两个模型通过不同前缀显式调用，网关不根据输入猜测或自动改用另一个模型。

GPU 设备通过容器设备请求隔离；每个容器只见自己的物理卡，因此阶段配置均使用本地设备 0。后台 LLM 单请求准入，KV 保留浮点，Gated DeltaNet 的 SSM 状态使用 FP32，不与权重一起量化。Omni 单路实时和普通推理的预算独立于 LLM。新分工的峰值与未来数字人余量须实测，不把旧跨卡结果当作新单卡共驻通过。

## Avatar 与访问

Avatar 使用 host 网络发布 ICE UDP；控制 API 走私有 Unix socket，共享卷只向网关开放。WHEP 媒体需 LAN／VPN 可达，单纯 SSH TCP 转发不转发媒体。浏览器验收页仍在网关根路径。

目标 `research-2@100.119.17.6`，远端仓库 `/home/research-2/code/omni-avatar-api`。可用 `ssh -N -L 18080:127.0.0.1:8000 research-2@100.119.17.6` 后访问 `http://localhost:18080`；媒体经 VPN 直连。Tailscale Serve 尚未配置，可信 HTTPS 入口待验收。

## 实机准入

分卡引擎、Responses 流式、JSON Schema、工具调用输出、归属续接、多轮事实保留、独立准入、取消及断流释放通过。原生双工及 Ava 2D 与后台 LLM 连续共驻约 20 分钟，完成 40 次后台请求。按用户要求以约半小时功能检查替代两小时负载。[功能报告](../evaluations/results/inference-functional-workstation.json) · [断流生命周期](../evaluations/results/responses-stream-lifecycle.json)。报告包含媒体解码与主机内存；本次运行容器未提供 nvidia-smi，因此未记录 GPU 峰值。LLM 的完整 262144 窗口与浏览器实物播放仍待验证。神经头像结果见下节。

旧两卡布局的短 Chat、结构化、视觉与 Avatar 120 秒轻量混合检查通过。[旧配置报告](../evaluations/results/native-avatar-workstation.json)。独立 Ava 2D 的音画、权限与幂等释放通过。[Ava 报告](../evaluations/results/avatar-2d-workstation.json)。新分工需重验受资源配置影响的部分；解码帧不等于真实扬声器／显示器播放。

旧近 40K Omni 检索在 AWQ、BF16、INT8／浮点 KV 和直接文本骨干参考中未通过。维护者建议当前版本多模态输入保持 8K。[官方边界](https://github.com/OpenBMB/MiniCPM-V/issues/1131#issuecomment-5238729878)。用户已明确采用近期约 8K 的多轮实时助手加外部记忆，长上下文交给独立 LLM；不再把一次性输入 40K 文本作为 Omni 准入要求。后台 LLM 的多轮历史和长上下文仍须独立验证。[当前计划](plans/avatar-workstation.md)。
## 会议式神经头像

在上述四文件主线后叠加 `infra-portrait.compose.yml`。`omni-avatar` 继续负责媒体与会话；新增 `omni-portrait` 在 GPU 0 与 LLM 共驻，通过私有 Unix socket 接收音频并回传帧。服务没有网络接口，模型只读挂载；参考图、准备特征和编译缓存使用独立持久卷。

```sh
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml -f infra-inference.compose.yml -f infra-portrait.compose.yml build omni-portrait omni-avatar omni-gateway
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml -f infra-inference.compose.yml -f infra-portrait.compose.yml --profile assets run --rm portrait-assets
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml -f infra-inference.compose.yml -f infra-portrait.compose.yml up -d --wait
```

模型固定 Ditto PyTorch，启动时完成编译与 Graph 捕获；会话就绪前预热实际工作线程并填充初始视频缓冲。目标为 640×720、25 fps，音频输入预留 1200 ms；不改变 Omni／LLM 的模型、窗口或 KV 配置。参考图准备和通话共用一路神经 GPU 配额，其他 2D 会话保持独立。在线接口见 [API](api.md#参考图与会议式头像)，驱动和生命周期见 [portrait](../portrait/README.md)。

25 fps 会议式 Ava 已实测通过：120 秒接收 3,004 视频帧、6,004 音频帧；3,012 张实际神经生成帧，最大音画时间线采样偏差 58 ms；原生语音持续输出，完成 4 次后台 Responses 请求。GPU 显存采样最大值分别为 23,101／21,468 MiB；这是采样读数，不是完整高频峰值或持续满载保证。重启后复用、取消代次与恢复也通过。[神经头像](../evaluations/results/portrait-functional-workstation.json) · [生命周期](../evaluations/results/portrait-lifecycle-workstation.json)。

Tailscale Serve 当前未启用，tailnet 管理员需先在控制台开放该功能才能配置私网 HTTPS。浏览器可使用上述 localhost SSH 隧道的可信上下文；WHEP 媒体继续经 VPN 直连。

## 私网共享 API

需要其他应用直接访问时，再叠加 `infra-vpn.compose.yml`，并在 `.env` 设置 `OMNI_GATEWAY_VPN_BIND` 为宿主的 VPN 地址。该覆盖保留回环监听，并增加指定私网地址的 8000 端口；模型引擎与神经工作器仍无宿主 TCP 入口。当前工作站地址为 `100.119.17.6`，API 基址分别是 `http://100.119.17.6:8000/v1` 和 `http://100.119.17.6:8000/llm/v1`，沿用各应用独立 Bearer。

```sh
docker compose -f infra-compose.yml -f infra-workstation.compose.yml -f infra-avatar.compose.yml -f infra-inference.compose.yml -f infra-portrait.compose.yml -f infra-vpn.compose.yml up -d --wait omni-gateway
```

浏览器麦克风／屏幕仍使用 `http://localhost:18080` 的 SSH 隧道，或由 tailnet 管理员开启 Serve 后配置 HTTPS；直接私网 HTTP 地址供 API 客户端调用。

已从另一 VPN 客户端验证两个模型卡、Responses 增量 SSE、准备形象查询及匿名拒绝，回环入口继续保留。[访问证据](../evaluations/results/vpn-api-workstation.json)。
