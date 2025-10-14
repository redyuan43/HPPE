#!/bin/bash

###############################################################################
# vLLM 安装进度监控脚本
#
# 实时监控 vLLM 安装进度
###############################################################################

LOG_FILE="/home/ivan/HPPE/vllm_install.log"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

clear

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}        vLLM 安装进度监控              ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# 检查进程
echo -e "${YELLOW}[1] 检查安装进程...${NC}"
if ps aux | grep "setup_vllm_env.sh" | grep -v grep > /dev/null; then
    echo -e "${GREEN}✓ 安装进程正在运行${NC}"
    ps aux | grep "setup_vllm_env.sh" | grep -v grep | awk '{print "   PID:", $2, "启动时间:", $9}'
else
    echo -e "${GREEN}✓ 安装已完成或未启动${NC}"
fi

echo ""

# 检查日志文件
echo -e "${YELLOW}[2] 最新安装日志 (最后 20 行):${NC}"
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE" | sed 's/^/   /'
    echo ""
    echo -e "${BLUE}日志文件位置: $LOG_FILE${NC}"
    echo -e "${BLUE}日志大小: $(du -h $LOG_FILE | awk '{print $1}')${NC}"
else
    echo "   日志文件不存在"
fi

echo ""

# 检查环境
echo -e "${YELLOW}[3] 检查 conda 环境:${NC}"
if conda env list | grep -q "^hppe "; then
    echo -e "${GREEN}✓ hppe 环境已创建${NC}"

    # 检查 vLLM 是否安装
    if conda run -n hppe python -c "import vllm; print(f'   vLLM version: {vllm.__version__}')" 2>/dev/null; then
        echo -e "${GREEN}✓ vLLM 已安装${NC}"
    else
        echo -e "${YELLOW}⏳ vLLM 正在安装中...${NC}"
    fi
else
    echo -e "${YELLOW}⏳ 环境创建中...${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "监控命令:"
echo -e "  实时日志: ${GREEN}tail -f $LOG_FILE${NC}"
echo -e "  重新检查: ${GREEN}./scripts/check_install_progress.sh${NC}"
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
