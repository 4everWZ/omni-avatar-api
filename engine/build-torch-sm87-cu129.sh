#!/bin/sh
set -eu

case "$1" in
    ''|*[!0-9]*) echo 'MAX_JOBS must be a positive integer.' >&2; exit 1 ;;
    *) MAX_JOBS=$1 ;;
esac
test "$MAX_JOBS" -gt 0

# Link to the exact cuDNN/NCCL shipped with the pinned Torch wheel. The
# runtime image already carries their versioned libraries and development headers.
nvidia_dir=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia")')
cudnn_dir="$nvidia_dir/cudnn"
nccl_dir="$nvidia_dir/nccl"
cusparselt_dir="$nvidia_dir/cusparselt"
test -s "$cudnn_dir/include/cudnn.h"
test -s "$nccl_dir/include/nccl.h"
test -s "$cudnn_dir/lib/libcudnn.so.9"
test -s "$nccl_dir/lib/libnccl.so.2"
test -s "$cusparselt_dir/include/cusparseLt.h"
test -s "$cusparselt_dir/lib/libcusparseLt.so.0"

mkdir -p /work/torch-cuda-links /artifact
ln -sf "$cudnn_dir/lib/libcudnn.so.9" /work/torch-cuda-links/libcudnn.so
ln -sf "$nccl_dir/lib/libnccl.so.2" /work/torch-cuda-links/libnccl.so
ln -sf "$cusparselt_dir/lib/libcusparseLt.so.0" /work/torch-cuda-links/libcusparseLt.so

abi=$(python3 -I -c 'import torch; assert torch.__version__ == "2.13.0+cu129"; print(int(torch.compiled_with_cxx11_abi()))')
export CXXFLAGS="${CXXFLAGS:-} -D_GLIBCXX_USE_CXX11_ABI=$abi"
export TORCH_NVCC_FLAGS="--threads=1 -D_GLIBCXX_USE_CXX11_ABI=$abi"
export TORCH_CUDA_ARCH_LIST=8.7
export MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export PYTORCH_BUILD_VERSION=2.13.0+cu129 PYTORCH_BUILD_NUMBER=1
export USE_CUDA=1 USE_CUDNN=1 USE_CUSPARSELT=1 USE_DISTRIBUTED=1 USE_NCCL=1 USE_SYSTEM_NCCL=1
export USE_STATIC_NCCL=0 USE_NVSHMEM=0 BUILD_TEST=0
export BLAS=OpenBLAS
export CUDNN_INCLUDE_DIR="$cudnn_dir/include" CUDNN_LIBRARY=/work/torch-cuda-links
export NCCL_INCLUDE_DIR="$nccl_dir/include" NCCL_LIB_DIR=/work/torch-cuda-links
export CUSPARSELT_INCLUDE_DIR="$cusparselt_dir/include" CUSPARSELT_LIBRARY=/work/torch-cuda-links
export LD_LIBRARY_PATH="$cudnn_dir/lib:$nccl_dir/lib:$cusparselt_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 -m build --wheel --no-isolation --outdir /artifact
cd /artifact
sha256sum torch-2.13.0+cu129-*.whl > SHA256SUMS
