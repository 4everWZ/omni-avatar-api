# JetPack 6 的历史 CUDA 12.9 验证路径

此页保留 AGX 兼容性实验。当前主线见[工作站部署](workstation.md)。

在 R36.4.7 上选择 `infra-cu129.compose.yml` 叠加到 `infra-compose.yml`。这条路径使用经验证可在目标 AGX 单进程加载的 NVIDIA Tegra CUDA 12.9 用户态库，加上 vLLM 官方 CUDA 12.9.1 ARM64 基础镜像和固定 vLLM-Omni 0.32.0 源码。容器构建与 MiniCPM-o 实机推理按下列步骤分别验收。基础部署继续使用 CUDA 13。 [JetPack 6 兼容矩阵](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-for-tegra-appnote/index.html#use-the-right-upgrade-package) · [vLLM ARM64 构建矩阵](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/.buildkite/release-pipeline.yaml)

## 准备 Tegra 兼容库和运算库

将下方 `compat_root` 设为实际 NVMe 的用户可写目录。运行容器的 Docker 数据目录也须有足够空间解包基础镜像和构建层。当前 CUDA 12.9.1 ARM64 基础镜像的压缩层约 11.2 GiB，MiniCPM-o 模型资产另需约 20 GB。

```sh
compat_root=/path/on/nvme/cuda-12.9-compat
mkdir -p "$compat_root"
curl -fL --retry 2 \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-compat-12-9_12.9.40580548-1_arm64.deb \
  -o "$compat_root/cuda-compat.deb"
printf '%s  %s\n' 36f3cd8f5b89b2c539a5e13e0eb5a3e5d860b9e2fe567a2ba0651b0e477e127c \
  "$compat_root/cuda-compat.deb" | sha256sum -c - && \
  dpkg-deb -x "$compat_root/cuda-compat.deb" "$compat_root"
export OMNI_CUDA129_COMPAT_DIR="$compat_root/usr/local/cuda-12.9/compat"
```

使用 Ubuntu 22.04 `arm64` 的 Tegra 包。解包过程只写指定目录。容器只读挂载兼容库。本覆盖文件关闭 SBSA 镜像启动前的 `NVIDIA_REQUIRE_CUDA` 检查，继续经 JetPack `nvidia` runtime 传递设备与驱动库。启动程序确认进程加载了指定目录中的库、`cuInit` 成功、驱动 API 报 12090、PyTorch 报 CUDA 12.9、设备为 SM 8.7，并验证 BF16 矩阵乘法。 [NVIDIA Tegra 包索引](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/) · [NVIDIA 容器运行时](https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/cmd/nvidia-container-runtime/README.md)

在同一 AGX 容器中，通用 cuBLAS 无法创建运算句柄，通用 cuDNN 卷积失败。Tegra cuBLAS 12.9.2.10 的 BF16 GEMM 和 Tegra cuDNN 9.20.0.48 的卷积均在 SM 8.7 上通过。解包并设置下面两个只读挂载。覆盖 wheel 路径可使 PyTorch 的绝对路径预加载也选用实际验过的库。 [PyTorch 库加载代码](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/__init__.py#L330)

```sh
repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64
(
  set -eu
  cd "$compat_root"
  blas=libcublas-12-9_12.9.2.10-1_arm64.deb
  dnn=libcudnn9-cuda-12_9.20.0.48-1_arm64.deb
  curl -fL --retry 2 "$repo/$blas" -o "$blas"
  curl -fL --retry 2 "$repo/$dnn" -o "$dnn"
  printf '%s  %s\n' \
    b03007e1839b8ddb815e9c6d87dad9c3648c05df199c2de45174cd30f386da71 "$blas" \
    54477521df7b05fb347a9d660559d37ed7d703edfcb98680fe5b12dcc5f176e3 "$dnn" | sha256sum -c -
  dpkg-deb -x "$blas" tegra-blas
  dpkg-deb -x "$dnn" tegra-dnn
)
export OMNI_TEGRA_CUBLAS_DIR="$compat_root/tegra-blas/usr/local/cuda-12.9/targets/aarch64-linux/lib"
export OMNI_TEGRA_CUDNN_DIR="$compat_root/tegra-dnn/usr/lib/aarch64-linux-gnu"
```

## 构建与实机准入

沿[基础设施指南](infra-setup.md)设置网关凭据、创建共享网络和模型目录。构建使用固定 ARM64 vLLM 镜像摘要与 Omni 源码归档校验值，保留基础镜像里的 PyTorch、torchvision、torchaudio、Triton、vLLM 和 CUDA ABI 相关包。Compose 两种配置共用相同模型 revision 和网关别名。节目传输另用 `infra-media.compose.yml`。

```sh
docker compose -f infra-compose.yml -f infra-cu129.compose.yml build omni-engine omni-gateway
docker compose -f infra-compose.yml -f infra-cu129.compose.yml run --rm --no-deps omni-engine \
  python3 /opt/tuber-omni/cu129_runtime.py probe
docker compose -f infra-compose.yml -f infra-cu129.compose.yml --profile assets run --rm model-fetch
docker compose -f infra-compose.yml -f infra-cu129.compose.yml up -d
```

BF16 探针通过后，从网关核验文字与静态图像，打开真实 Realtime 会话核验 `session.created` 能力、PCM 输入和实际返回音频，随后验收重连、中断、Avatar 和混合负载。若 CUDA API、主机设备节点、C 扩展和某阶段内核分别报错，保留原诊断并按该阶段排查。

## 构建 SM 8.7 PyTorch wheel

本机的 CUDA 12.9 用户态库能初始化 SM 8.7，固定 vLLM 基础镜像里的官方 PyTorch 2.13.0+cu129 在执行 `torch.ones(..., device="cuda")` 时出现 `no kernel image`。该 wheel 的 ARM64 构建矩阵为 SM 8.0、9.0、10.0、12.0。Tegra 与桌面平台有独立的 CUDA 二进制兼容边界。 [PyTorch wheel 构建矩阵](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/.ci/manywheel/build_env_setup.py) · [NVIDIA CUDA 二进制兼容](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/#binary-compatibility)

在原生 ARM64 构建机使用与 cu129 覆盖相同的镜像摘要，构建官方 PyTorch `v2.13.0` 对应的固定提交及子模块。构建过程沿用基础镜像的 cuDNN、NCCL、cuSPARSELt 和 C++ ABI，CUDA 目标设为 SM 8.7。初始构建限制为两个编译任务，每个 NVCC 进程使用一个内部线程。源码、CUDA 开发包和编译缓存需要 NVMe 容量。

```sh
docker build -f engine/Dockerfile.torch-sm87-cu129 --platform linux/arm64 \
  --target artifact --build-arg MAX_JOBS=2 \
  --output type=local,dest=/path/on/nvme/torch-sm87-dist engine
```

导出目录包含 `torch-2.13.0+cu129-*.whl` 和 `SHA256SUMS`。派生验收镜像安装 Ubuntu 的 OpenBLAS、校验 wheel 并核对该包的 SM 8.7 编译标记和 C++ ABI。使用同一个 Docker 数据目录，把派生镜像交给 Omni 构建。

```sh
docker build -f engine/Dockerfile.torch-runtime --platform linux/arm64 \
  --build-context sm87-wheels=/path/on/nvme/torch-sm87-dist \
  -t tuber-vllm-core:0.32.0-cu129-sm87 engine
export VLLM_CORE_IMAGE=tuber-vllm-core:0.32.0-cu129-sm87
docker compose -f infra-compose.yml -f infra-cu129.compose.yml build omni-engine
docker compose -f infra-compose.yml -f infra-cu129.compose.yml run --rm --no-deps omni-engine \
  python3 /opt/tuber-omni/cu129_runtime.py probe
```

随后依次验证音频与视觉所需的 CUDA 操作、vLLM 扩展、torchvision 和 torchaudio，再启动 MiniCPM-o。现场发现上游 vLLM 主扩展已有 SM 8.7，FlashAttention 2 只有 SM 8.0，所以两段模型注意力先使用 Triton。完整三阶段加载和接口继续按实机准入验收。

AGX 已导出 SM 8.7 wheel，并通过容器内张量、BF16 矩阵乘法、卷积和注意力预检。继承的 TorchCodec 0.17.0 图像扩展要求 CUDA 13 NVRTC。CUDA 12.9 镜像换用官方同版本 ARM64 CPU 解码 wheel，构建阶段核对 `AudioDecoder` 和 `VideoDecoder` 导入。模型计算继续使用 CUDA。 [TorchCodec 官方 wheel 索引](https://download.pytorch.org/whl/cpu/torchcodec/)

AGX 的早期设备级自动探测给 Thinker 留出 0 GiB KV，因此配置显式缓存分配。Thinker 的 36 层、8 个 KV 头和 128 维 head 使用逐 token、逐 head 的 4-bit Triton 打包，含缩放信息约 1.79 GiB 可容纳一个完整 Chat 窗口和一个原生双工滚动窗口，分配 2 GiB。Talker 分配 512 MiB。启动前核对 OpenBMB 官方 AWQ 快照，启动后核对实报窗口、两阶段实际 KV 格式及音画协议。窗口和推理并发的联合上限以实机混合负载为准。

Thinker 的每步输入预算为 2,048 token。两个请求都可调度时，`long_prefill_token_threshold: 1024` 给每路在该步留输入空间。单路可使用整批预算。此配置解决长预填占满单步预算的调度条件，Chat 总窗口继续使用 checkpoint 实报值。 [固定版 vLLM 调度器](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L640)

MiniCPM-o 原生会话的 Code2Wav 资源跟随会话保有。首段语音前和已结束的语音段之间可以静默，资源按会话租约和关闭清理。实际语音段中途的缺块、普通有限请求等继续按连接器原超时处理。 [固定版 Omni 异步连接器](https://github.com/vllm-project/vllm-omni/blob/5917730a4b02aedb44d374173cdc1b10484ccd52/vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L1496)
