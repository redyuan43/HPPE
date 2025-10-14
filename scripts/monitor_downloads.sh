#!/bin/bash

###############################################################################
# 下载进度实时监控脚本
###############################################################################

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

while true; do
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}        下载进度实时监控 (按 Ctrl+C 退出)        ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
    echo ""

    # vLLM 安装进度
    echo -e "${YELLOW}[1] vLLM 安装进度:${NC}"
    if ps aux | grep "setup_vllm_env.sh" | grep -v grep > /dev/null; then
        echo -e "${GREEN}✓ 进程运行中${NC}"
        echo "最新日志（最后 5 行）:"
        tail -5 /home/ivan/HPPE/vllm_install.log | sed 's/^/   /'
    else
        echo -e "${GREEN}✓ 已完成或未运行${NC}"
    fi

    echo ""
    echo -e "${YELLOW}[2] Qwen3 下载进度:${NC}"
    if ps aux | grep "download_qwen3_model.sh" | grep -v grep > /dev/null; then
        echo -e "${GREEN}✓ 进程运行中${NC}"
        echo "最新日志（最后 5 行）:"
        tail -5 /home/ivan/HPPE/qwen3_download.log | sed 's/^/   /'
    else
        echo -e "${GREEN}✓ 已完成或未运行${NC}"
    fi

    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════${NC}"
    echo "刷新时间: $(date '+%H:%M:%S')"
    echo "下次刷新: 10 秒后..."

    sleep 10
done
