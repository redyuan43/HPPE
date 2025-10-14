#!/bin/bash
# HPPE 环境快速激活脚本
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hppe
echo "✓ HPPE 环境已激活 (vLLM)"
echo ""
echo "快速启动 vLLM:"
echo "  ./scripts/start_vllm.sh"
