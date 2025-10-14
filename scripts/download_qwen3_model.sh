#!/bin/bash

###############################################################################
# Qwen3-8B-AWQ 模型下载脚本
#
# 专为 RTX 3060 12GB 优化
# 模型: Qwen/Qwen3-8B-AWQ
# 大小: ~5GB (4-bit 量化)
# 显存: ~6GB
#
# 使用方法:
#   ./scripts/download_qwen3_model.sh
#   或后台运行:
#   nohup ./scripts/download_qwen3_model.sh > qwen3_download.log 2>&1 &
###############################################################################

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置
MODEL_NAME="Qwen/Qwen3-8B-AWQ"
MODEL_SIZE="~5GB"

echo ""
echo "======================================================================"
echo "                Qwen3-8B-AWQ 模型下载工具"
echo "======================================================================"
echo ""
echo "模型信息:"
echo "  名称: $MODEL_NAME"
echo "  大小: $MODEL_SIZE (4-bit AWQ 量化)"
echo "  显存: ~6GB"
echo "  上下文: 32K-131K tokens"
echo "  特性: Thinking Mode (思考模式)"
echo ""
echo "硬件要求:"
echo "  GPU: RTX 3060 12GB ✓"
echo "  vLLM: >=0.8.5"
echo "  磁盘: 至少 10GB 可用空间"
echo ""
echo "======================================================================"
echo ""

# 检查 conda 环境
if ! command -v conda &> /dev/null; then
    error "conda 未找到"
    exit 1
fi

# 激活环境
log "激活 conda 环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hppe || {
    error "无法激活 hppe 环境，请先运行 ./scripts/setup_vllm_env.sh"
    exit 1
}

info "✓ 环境已激活: hppe"

# 检查磁盘空间
log "检查磁盘空间..."
AVAILABLE_SPACE=$(df -BG /home/ivan | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED_SPACE=10

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    warn "磁盘空间不足（可用: ${AVAILABLE_SPACE}GB，需要: ${REQUIRED_SPACE}GB）"
    exit 1
fi

info "✓ 磁盘空间充足: ${AVAILABLE_SPACE}GB"

# 安装 huggingface-hub
log "检查 huggingface-hub..."
if ! python -c "import huggingface_hub" 2>/dev/null; then
    info "安装 huggingface-hub..."
    pip install -q -U huggingface-hub
fi
info "✓ huggingface-hub 已就绪"

echo ""
log "开始下载 $MODEL_NAME..."
echo ""

# 下载模型
python -c "
from huggingface_hub import snapshot_download
import sys

try:
    print('正在下载 $MODEL_NAME...')
    print('这可能需要 5-15 分钟（取决于网络速度）')
    print('')

    model_path = snapshot_download(
        repo_id='$MODEL_NAME',
        resume_download=True,
        local_files_only=False
    )

    print('')
    print('✓ 下载完成!')
    print(f'模型路径: {model_path}')

except KeyboardInterrupt:
    print('\\n下载被用户中断')
    sys.exit(130)
except Exception as e:
    print(f'✗ 下载失败: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    log "验证模型..."

    # 验证模型文件
    python -c "
from transformers import AutoConfig, AutoTokenizer

try:
    config = AutoConfig.from_pretrained('$MODEL_NAME')
    tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')

    print('✓ 模型验证通过')
    print(f'  总参数: {config.num_parameters:,}')
    print(f'  层数: {config.num_hidden_layers}')
    print(f'  词表大小: {len(tokenizer):,}')
    print(f'  上下文长度: {config.max_position_embeddings:,}')
except Exception as e:
    print(f'✗ 验证失败: {e}')
    exit(1)
"

    echo ""
    echo "======================================================================"
    log "Qwen3-8B-AWQ 下载完成！"
    echo "======================================================================"
    echo ""

    # 显示使用指南
    info "下一步操作:"
    echo ""
    echo "1. 启动 vLLM 服务:"
    echo "   MODEL_NAME=Qwen/Qwen3-8B-AWQ ./scripts/start_vllm.sh"
    echo ""
    echo "2. 或使用自定义配置:"
    echo "   conda activate hppe"
    echo "   python -m vllm.entrypoints.openai.api_server \\"
    echo "     --model Qwen/Qwen3-8B-AWQ \\"
    echo "     --quantization awq \\"
    echo "     --gpu-memory-utilization 0.85 \\"
    echo "     --max-model-len 32768 \\"
    echo "     --port 8000"
    echo ""
    echo "3. 测试 LLM 引擎:"
    echo "   PYTHONPATH=/home/ivan/HPPE/src python examples/llm_engine_example.py"
    echo ""
    echo "4. 使用 Thinking Mode (思考模式):"
    echo "   # 在 prompt 中添加 <|im_start|>thinking 触发"
    echo ""

    log "完成！🎉"
else
    error "下载失败"
    exit 1
fi
