#!/bin/bash

###############################################################################
# vLLM 环境配置脚本
#
# 创建 conda 环境并安装 vLLM 及其依赖
# 适用于 NVIDIA RTX 3060 (CUDA 12.x+)
#
# 使用方法:
#   ./scripts/setup_vllm_env.sh
#   或在后台运行:
#   nohup ./scripts/setup_vllm_env.sh > vllm_install.log 2>&1 &
###############################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# 配置
ENV_NAME="hppe"
PYTHON_VERSION="3.10"

log "开始配置 vLLM 环境..."
echo ""

# 1. 检查 conda 是否可用
info "检查 conda 安装..."
if ! command -v conda &> /dev/null; then
    error "conda 未找到，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

CONDA_VERSION=$(conda --version)
info "找到 $CONDA_VERSION"

# 2. 检查 CUDA 版本
info "检查 CUDA 版本..."
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi 未找到，请确保安装了 NVIDIA 驱动"
    exit 1
fi

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
info "CUDA 版本: $CUDA_VERSION"

if [[ $(echo "$CUDA_VERSION < 11.8" | bc -l) -eq 1 ]]; then
    warn "CUDA 版本较低 ($CUDA_VERSION)，建议使用 CUDA 11.8+"
fi

# 3. 检查环境是否已存在
log "检查 conda 环境..."
if conda env list | grep -q "^$ENV_NAME "; then
    warn "环境 '$ENV_NAME' 已存在"
    read -p "是否删除并重新创建？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "删除现有环境..."
        conda env remove -n $ENV_NAME -y
    else
        info "使用现有环境"
    fi
fi

# 4. 创建 conda 环境
if ! conda env list | grep -q "^$ENV_NAME "; then
    log "创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    info "✓ 环境创建成功"
else
    info "环境已存在，跳过创建"
fi

echo ""

# 5. 激活环境并安装依赖
log "安装 vLLM 及依赖包..."
info "这可能需要 5-15 分钟，请耐心等待..."
echo ""

# 在子shell中执行安装，避免影响当前shell
(
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $ENV_NAME

    # 升级 pip
    log "升级 pip..."
    pip install --upgrade pip

    # 安装 PyTorch (CUDA 12.1)
    log "安装 PyTorch (CUDA 12.1)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # 验证 PyTorch CUDA
    log "验证 PyTorch CUDA..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

    # 安装 vLLM
    log "安装 vLLM..."
    pip install vllm

    # 安装其他依赖
    log "安装其他依赖..."
    pip install openai requests pyyaml

    # 验证 vLLM 安装
    log "验证 vLLM 安装..."
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

    # 安装 HPPE 开发依赖（如果需要）
    if [ -f "requirements.txt" ]; then
        log "安装 HPPE 项目依赖..."
        pip install -r requirements.txt
    fi
)

echo ""
log "安装完成！"
echo ""

# 6. 显示激活命令
info "要使用 vLLM 环境，请运行:"
echo -e "  ${GREEN}conda activate $ENV_NAME${NC}"
echo ""

info "验证安装:"
echo "  conda activate $ENV_NAME"
echo "  python -c 'import vllm; print(vllm.__version__)'"
echo ""

info "启动 vLLM 服务:"
echo "  conda activate $ENV_NAME"
echo "  ./scripts/start_vllm.sh"
echo ""

# 7. 创建激活脚本
ACTIVATE_SCRIPT="activate_hppe.sh"
cat > $ACTIVATE_SCRIPT << 'EOF'
#!/bin/bash
# HPPE 环境快速激活脚本
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hppe
echo "✓ HPPE 环境已激活 (vLLM)"
echo ""
echo "快速启动 vLLM:"
echo "  ./scripts/start_vllm.sh"
EOF

chmod +x $ACTIVATE_SCRIPT
info "创建了快速激活脚本: ./$ACTIVATE_SCRIPT"

echo ""
log "所有步骤完成！"
info "环境名称: $ENV_NAME"
info "Python 版本: $PYTHON_VERSION"
info "安装位置: $(conda env list | grep $ENV_NAME | awk '{print $2}')"
