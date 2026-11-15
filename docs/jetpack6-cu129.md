# JetPack 6 的 CUDA 12.9 验证路径

在 R36.4.7 上选择 `infra-cu129.compose.yml` 叠加到 `infra-compose.yml`。这条路径使用经验证可在目标 AGX 单进程加载的 NVIDIA Tegra CUDA 12.9 用户态库，加上 vLLM 官方 CUDA 12.9.1 ARM64 基础镜像和固定 vLLM-Omni 0.32.0 源码。容器构建与 MiniCPM-o 实机推理按下列步骤分别验收。基础部署继续使用 CUDA 13。 [JetPack 6 兼容矩阵](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-for-tegra-appnote/index.html#use-the-right-upgrade-package) · [vLLM ARM64 构建矩阵](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/.buildkite/release-pipeline.yaml)

## 准备 Tegra 兼容库

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

## 构建与实机准入

沿[基础设施指南](infra-setup.md)设置网关和媒体凭据、创建共享网络和模型目录。构建使用固定 ARM64 vLLM 镜像摘要与 Omni 源码归档校验值，保留基础镜像里的 PyTorch、torchvision、torchaudio、Triton、vLLM 和 CUDA ABI 相关包。Compose 两种配置共用相同模型 revision 和网关别名。

```sh
docker compose -f infra-compose.yml -f infra-cu129.compose.yml build omni-engine omni-gateway
docker compose -f infra-compose.yml -f infra-cu129.compose.yml run --rm --no-deps omni-engine \
  python3 /opt/tuber-omni/cu129_runtime.py probe
docker compose -f infra-compose.yml -f infra-cu129.compose.yml --profile assets run --rm model-fetch
docker compose -f infra-compose.yml -f infra-cu129.compose.yml up -d
```

BF16 探针通过后，从网关核验文字与静态图像，打开真实 Realtime 会话核验 `session.created` 能力、PCM 输入和实际返回音频，随后验收重连、中断、Avatar 和混合负载。若 CUDA API、主机设备节点、C 扩展和某阶段内核分别报错，保留原诊断并按该阶段排查。现行 9B／Genie 仍可由应用使用验证过的原端点。
