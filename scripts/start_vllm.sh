#!/bin/bash

###############################################################################
# vLLM 推理服务器启动脚本
#
# 用于启动基于 Qwen 模型的 vLLM 推理服务
# 硬件要求: NVIDIA RTX 3060 (12GB VRAM)
#
# 使用方法:
#   ./scripts/start_vllm.sh
#
# 环境变量:
#   MODEL_NAME: 模型名称（默认: Qwen/Qwen2.5-7B-Instruct）
#   PORT: 服务端口（默认: 8000）
#   GPU_ID: 使用的 GPU ID（默认: 0）
#   MAX_MODEL_LEN: 最大序列长度（默认: 4096）
###############################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印信息函数
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置参数（可通过环境变量覆盖）
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${PORT:-8000}
GPU_ID=${GPU_ID:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

# 打印配置
info "vLLM 服务配置:"
echo "  模型: $MODEL_NAME"
echo "  端口: $PORT"
echo "  GPU ID: $GPU_ID"
echo "  最大序列长度: $MAX_MODEL_LEN"
echo "  GPU 内存利用率: $GPU_MEMORY_UTIL"
echo "  张量并行大小: $TENSOR_PARALLEL_SIZE"
echo ""

# 检查 GPU 可用性
info "检查 GPU 可用性..."
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi 未找到，请确保 NVIDIA 驱动已安装"
    exit 1
fi

# 显示 GPU 信息
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# 检查指定的 GPU 是否可用
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
if [ "$GPU_ID" -ge "$GPU_COUNT" ]; then
    error "GPU ID $GPU_ID 不可用，系统只有 $GPU_COUNT 个 GPU"
    exit 1
fi

info "使用 GPU $GPU_ID"

# 检查 vLLM 是否已安装
info "检查 vLLM 安装..."
if ! python -c "import vllm" 2>/dev/null; then
    error "vLLM 未安装，请先安装:"
    echo "  pip install vllm"
    exit 1
fi

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
info "vLLM 版本: $VLLM_VERSION"

# 检查模型是否存在（如果是本地路径）
if [[ "$MODEL_NAME" == /* ]] || [[ "$MODEL_NAME" == ./* ]]; then
    if [ ! -d "$MODEL_NAME" ]; then
        error "本地模型路径不存在: $MODEL_NAME"
        exit 1
    fi
    info "使用本地模型: $MODEL_NAME"
else
    info "使用 HuggingFace 模型: $MODEL_NAME"
    warn "首次运行会自动下载模型，可能需要较长时间"
fi

# 检查端口是否已被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    error "端口 $PORT 已被占用"
    info "可以使用以下命令查看占用情况:"
    echo "  lsof -i :$PORT"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 启动 vLLM 服务
info "启动 vLLM 服务..."
info "API 端点: http://localhost:$PORT/v1"
info "健康检查: http://localhost:$PORT/health"
info "API 文档: http://localhost:$PORT/docs"
echo ""
info "按 Ctrl+C 停止服务"
echo ""

# 启动命令
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --disable-log-requests

# 注意: 如果需要启用日志请求，移除 --disable-log-requests
# 注意: 如果需要使用量化模型，添加 --quantization awq 或 --quantization gptq
