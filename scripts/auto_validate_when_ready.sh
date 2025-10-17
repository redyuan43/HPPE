#!/bin/bash
# 自动化验证脚本：等待17种模型训练完成后自动运行验证

MODEL_PATH="models/pii_qwen4b_17types/final"
CHECK_INTERVAL=300  # 检查间隔（秒）

echo "=================================================="
echo "🤖 自动化验证监控器"
echo "=================================================="
echo "目标模型: $MODEL_PATH"
echo "检查间隔: ${CHECK_INTERVAL}秒 ($((CHECK_INTERVAL / 60))分钟)"
echo ""

while true; do
    if [ -d "$MODEL_PATH" ]; then
        echo "✅ 检测到模型已生成！"
        echo ""

        # 等待模型文件完全写入
        echo "⏳ 等待60秒确保文件完整..."
        sleep 60

        # 运行对比验证
        echo ""
        echo "=================================================="
        echo "🚀 开始运行模型对比验证"
        echo "=================================================="

        python scripts/compare_6vs17_models.py \
            --model-6pii "models/pii_qwen4b_unsloth/final" \
            --model-17pii "$MODEL_PATH" \
            --test-data "data/test_datasets/17pii_test_cases.jsonl" \
            --output "comparison_6vs17_report_$(date +%Y%m%d_%H%M%S).json"

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo ""
            echo "✅ 验证完成！"
        else
            echo ""
            echo "❌ 验证失败（退出码: $EXIT_CODE）"
        fi

        break
    else
        CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$CURRENT_TIME] ⏳ 模型尚未生成，${CHECK_INTERVAL}秒后再检查..."
        sleep $CHECK_INTERVAL
    fi
done

echo ""
echo "🎉 自动化验证任务结束"
