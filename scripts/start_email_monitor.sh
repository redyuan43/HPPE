#!/bin/bash
################################################################################
# 训练进度邮件监控 - 一键启动脚本
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

echo "========================================================================"
echo "🔔 启动训练进度邮件监控"
echo "========================================================================"

# 检查Python脚本是否存在
if [ ! -f "scripts/training_email_monitor.py" ]; then
    echo "❌ 错误：找不到 scripts/training_email_monitor.py"
    exit 1
fi

# 检查配置
if grep -q "your_email@example.com" scripts/training_email_monitor.py; then
    echo "⚠️  警告：检测到默认配置，请先配置邮箱信息！"
    echo ""
    echo "请按照以下步骤配置："
    echo "  1. 编辑 scripts/training_email_monitor.py"
    echo "  2. 修改 RECEIVER_EMAIL（接收邮箱）"
    echo "  3. 修改 SENDER_EMAIL 和 SENDER_PASSWORD（发送邮箱和SMTP授权码）"
    echo ""
    echo "详细说明请查看: EMAIL_MONITOR_SETUP.md"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 检查是否已经在运行
if pgrep -f "training_email_monitor.py" > /dev/null; then
    echo "⚠️  监控程序已在运行中"
    echo ""
    ps aux | grep "training_email_monitor.py" | grep -v grep
    echo ""
    read -p "是否停止现有进程并重新启动？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "training_email_monitor.py"
        echo "✅ 已停止现有进程"
        sleep 2
    else
        echo "取消启动"
        exit 0
    fi
fi

# 启动监控程序
echo "🚀 启动监控程序..."
nohup python3 scripts/training_email_monitor.py > logs/email_monitor.log 2>&1 &
PID=$!

sleep 2

# 检查是否成功启动
if ps -p $PID > /dev/null; then
    echo ""
    echo "========================================================================"
    echo "✅ 监控程序启动成功！"
    echo "========================================================================"
    echo "进程ID: $PID"
    echo "日志文件: logs/email_monitor.log"
    echo ""
    echo "查看实时日志:"
    echo "  tail -f logs/email_monitor.log"
    echo ""
    echo "停止监控:"
    echo "  kill $PID"
    echo ""
    echo "监控程序将每2小时发送一次进度邮件"
    echo "========================================================================"
else
    echo ""
    echo "❌ 启动失败，请查看日志:"
    echo "  tail -f logs/email_monitor.log"
    exit 1
fi
