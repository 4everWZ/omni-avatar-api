# 双 RTX 3090 工作站部署

工作站是主线目标。首个候选用官方 MiniCPM-o 4.5 AWQ，Chat 窗口由 checkpoint 读取，基线为 40,960 tokens，默认生成额度 4,096，Thinker 和 Talker 启用逐 token、逐 head INT4 KV。引擎使用官方固定版 AMD64 CUDA 镜像。RTX 3090 每卡 24 GB，分别核对峰值。 [NVIDIA 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/) · [vLLM-Omni 固定发行版](https://github.com/vllm-project/vllm-omni/releases/tag/v0.32.0)

## 首套运行配置

开始前用 `nvidia-smi` 和 `nvidia-smi topo -m` 核对两张卡、互联、当前占用及支持 CUDA 13 的驱动。核对 Docker、Compose 及 NVIDIA Container Toolkit 的容器 GPU 识别。配置让 GPU 0 运行 Thinker，GPU 1 运行 Talker 和 Code2Wav，阶段内并行度沿固定版单卡设置。Avatar 加入前测量第 2 张卡的实际余量。 [阶段配置](../engine/minicpmo-workstation.yaml) · [NVIDIA CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

将 `.env.example` 复制为私有 `.env`，填写独立的网关服务令牌、网关管理令牌和模型挂载目录。为不同调用方配置各自的令牌。网关默认监听宿主回环，可通过授权网络或私有反向代理提供服务。模型端口在内部网络。

```sh
docker network inspect tuber-omni >/dev/null 2>&1 || docker network create tuber-omni
docker compose -f infra-compose.yml -f infra-workstation.compose.yml config --quiet
docker compose -f infra-compose.yml -f infra-workstation.compose.yml build omni-engine omni-gateway
docker compose -f infra-compose.yml -f infra-workstation.compose.yml --profile assets run --rm model-fetch
docker compose -f infra-compose.yml -f infra-workstation.compose.yml up -d
docker compose -f infra-compose.yml -f infra-workstation.compose.yml ps
```

## 实机准入

1. 先验引擎和网关。用至少两个独立的调用方验证普通／流式 Chat、结构化输出、视觉、长文正确性、排队和取消。按阶段核实设备索引、AWQ 与实际 INT4 KV，记录每卡显存、主机内存、交换、吞吐和首包。
2. 用独立 Realtime 客户端验证语音与视觉输入、重叠语音、打断、字幕、静默与重连。测一路原生双工加另一个客户端普通请求持续运行，确认会话释放和引擎可用。
3. 接 Avatar 会话 API，验同步音画、打断、权限隔离和资源回收。合成节目、WHEP 外部取流和 OBS 在核心服务之后单独验收。 [接口边界](api.md)

当前为工作站候选配置，GPU 构建和实测为待办。早期 Jetson 结果见[历史 AGX 证据](https://github.com/4everWZ/Tuber_Studio/blob/main/evaluations/results/omni-deployment-20261116.md)。
