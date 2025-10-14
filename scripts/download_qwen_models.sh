#!/bin/bash

###############################################################################
# Qwen2.5 模型批量下载脚本
#
# 根据 RTX 3060 12GB 显存，推荐下载顺序：
# 1. Qwen2.5-3B-Instruct-AWQ (开发测试)
# 2. Qwen2.5-7B-Instruct-AWQ (生产部署)
#
# 使用方法:
#   ./scripts/download_qwen_models.sh
#   或后台运行:
#   nohup ./scripts/download_qwen_models.sh > qwen_download.log 2>&1 &
###############################################################################

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# 日志函数
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

# 检查 conda 环境
if ! command -v conda &> /dev/null; then
    error "conda 未找到，请先安装 Anaconda"
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
REQUIRED_SPACE=15  # 需要约 15GB (3B: 2GB + 7B: 4.5GB + 缓存)

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    warn "磁盘空间可能不足（可用: ${AVAILABLE_SPACE}GB，需要: ${REQUIRED_SPACE}GB）"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    info "✓ 磁盘空间充足: ${AVAILABLE_SPACE}GB"
fi

# 安装 huggingface-hub（如果没有）
log "检查 huggingface-hub..."
if ! python -c "import huggingface_hub" 2>/dev/null; then
    info "安装 huggingface-hub..."
    pip install -U huggingface-hub
fi
info "✓ huggingface-hub 已就绪"

echo ""
echo "======================================================================"
echo "                    Qwen2.5 模型下载计划"
echo "======================================================================"
echo ""
echo "根据你的 RTX 3060 12GB 显存，推荐下载顺序："
echo ""
echo "1️⃣  Qwen2.5-3B-Instruct-AWQ  (~2GB)"
echo "    - 用途：开发测试、Prompt 调试"
echo "    - 显存：~3-4GB"
echo "    - 速度：快速加载，适合迭代"
echo ""
echo "2️⃣  Qwen2.5-7B-Instruct-AWQ  (~4.5GB)"
echo "    - 用途：生产部署、正式检测"
echo "    - 显存：~5-6GB"
echo "    - 性能：更好的理解和准确率"
echo ""
echo "======================================================================"
echo ""

read -p "开始下载？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    info "已取消下载"
    exit 0
fi

# 下载函数
download_model() {
    local MODEL_NAME=$1
    local MODEL_SIZE=$2

    log "开始下载: $MODEL_NAME ($MODEL_SIZE)"
    echo ""

    # 使用 Python 下载（支持断点续传）
    python -c "
from huggingface_hub import snapshot_download
import sys

try:
    print(f'正在下载 $MODEL_NAME...')
    model_path = snapshot_download(
        repo_id='$MODEL_NAME',
        resume_download=True,
        local_files_only=False
    )
    print(f'✓ 下载完成: {model_path}')
except KeyboardInterrupt:
    print('\\n下载被用户中断')
    sys.exit(130)
except Exception as e:
    print(f'✗ 下载失败: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        info "✓ $MODEL_NAME 下载成功"

        # 验证模型
        log "验证模型文件..."
        python -c "
from transformers import AutoConfig, AutoTokenizer

try:
    config = AutoConfig.from_pretrained('$MODEL_NAME')
    tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')
    print(f'✓ 模型验证通过')
    print(f'  参数数量: {config.num_parameters:,}')
    print(f'  词表大小: {len(tokenizer):,}')
except Exception as e:
    print(f'✗ 验证失败: {e}')
    exit(1)
"
    else
        error "✗ $MODEL_NAME 下载失败"
        return 1
    fi

    echo ""
}

# 下载模型
echo ""
log "========== 下载 1/2: Qwen2.5-3B-Instruct-AWQ =========="
download_model "Qwen/Qwen2.5-3B-Instruct-AWQ" "~2GB"

echo ""
log "========== 下载 2/2: Qwen2.5-7B-Instruct-AWQ =========="
download_model "Qwen/Qwen2.5-7B-Instruct-AWQ" "~4.5GB"

# 完成
echo ""
echo "======================================================================"
log "所有模型下载完成！"
echo "======================================================================"
echo ""

# 显示模型位置
log "模型存储位置:"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface/hub}"
echo "  $CACHE_DIR"
echo ""

# 显示使用指南
info "下一步操作:"
echo ""
echo "1. 启动 vLLM 服务（使用 3B 模型测试）:"
echo "   MODEL_NAME=Qwen/Qwen2.5-3B-Instruct-AWQ ./scripts/start_vllm.sh"
echo ""
echo "2. 或使用 7B 模型（生产部署）:"
echo "   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ ./scripts/start_vllm.sh"
echo ""
echo "3. 测试 LLM 引擎:"
echo "   PYTHONPATH=/home/ivan/HPPE/src python examples/llm_engine_example.py"
echo ""
echo "4. 运行 PII 检测测试:"
echo "   # (待实现 Story 2.3)"
echo ""

log "完成！🎉"
