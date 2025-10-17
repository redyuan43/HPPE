#!/bin/bash
# 监控双GPU训练并在1 epoch后自动验证

echo "=================================================="
echo "双GPU训练监控"
echo "=================================================="
echo ""

# 等待GPU1的激进配置完成1 epoch (879步,约2.7小时)
echo "[监控] GPU1 激进配置 (1 epoch)"
echo "目标: 879/1757 steps"
echo ""

while true; do
    # 检查GPU1训练进度
    if grep -q "100%\|1757/1757" logs/training_aggressive_20251015.log 2>/dev/null; then
        echo "[完成] GPU1 激进配置训练完成!"
        break
    fi

    # 显示进度
    PROGRESS=$(tail -1 logs/training_aggressive_20251015.log 2>/dev/null | grep -oP '\d+/1757' || echo "0/1757")
    PERCENT=$(tail -1 logs/training_aggressive_20251015.log 2>/dev/null | grep -oP '\s+\d+%' | tr -d ' ' || echo "0%")
    echo -ne "\r[GPU1] 进度: $PROGRESS ($PERCENT)"

    sleep 60  # 每分钟检查一次
done

echo -e "\n\n=================================================="
echo "[验证] 开始验证GPU1激进配置模型"
echo "=================================================="

python scripts/ultra_fast_validation.py \
    --model models/pii_detector_qwen3_06b_aggressive \
    --test-data data/merged_pii_dataset_test.jsonl \
    --sample-size 500 \
    --timeout 15 \
    2>&1 | tee logs/validation_aggressive.log

echo ""
echo "=================================================="
echo "[监控] GPU0 保守配置 (检查1 epoch进度)"
echo "目标: 879/1758 steps"
echo "=================================================="

# 检查GPU0是否也完成了1 epoch
while true; do
    CURRENT=$(tail -1 logs/continue_training_20251015_v2.log 2>/dev/null | grep -oP '\d+/1758' | cut -d'/' -f1)

    if [ "$CURRENT" -ge 879 ] 2>/dev/null; then
        echo "[完成] GPU0 已完成1 epoch (步数: $CURRENT/1758)"
        break
    fi

    PERCENT=$(tail -1 logs/continue_training_20251015_v2.log 2>/dev/null | grep -oP '\s+\d+%' | tr -d ' ' || echo "0%")
    echo -ne "\r[GPU0] 进度: $CURRENT/1758 ($PERCENT)"

    sleep 60
done

echo -e "\n\n=================================================="
echo "[决策] 对比两个配置的验证结果"
echo "=================================================="

# 提取结果
AGG_F1=$(grep "F1-Score:" logs/validation_aggressive.log | awk '{print $2}' | sed 's/%//')
AGG_RECALL=$(grep "Recall" logs/validation_aggressive.log | grep "召回率" | awk '{print $2}' | sed 's/%//')
AGG_PRECISION=$(grep "Precision" logs/validation_aggressive.log | grep "精确率" | awk '{print $2}' | sed 's/%//')

echo ""
echo "GPU1 激进配置 (LR=1.5e-4, r=12, 1 epoch):"
echo "  Precision: ${AGG_PRECISION}%"
echo "  Recall: ${AGG_RECALL}%"
echo "  F1-Score: ${AGG_F1}%"
echo ""

# 生成对比报告
cat > dual_training_comparison.md <<EOF
# 双GPU训练对比报告

**时间**: $(date '+%Y-%m-%d %H:%M')

## 配置对比

| 参数 | GPU0: 保守配置 | GPU1: 激进配置 |
|------|---------------|---------------|
| 学习率 | 1e-4 | **1.5e-4** |
| LoRA rank | 8 | **12** |
| LoRA alpha | 16 | **24** |
| Epochs | 2 (进行中) | 1 (已完成) |

## 1 Epoch后的验证结果

### GPU1: 激进配置

| 指标 | 结果 | 目标 | 状态 |
|------|------|------|------|
| Precision | ${AGG_PRECISION}% | ≥ 85% | $([ $(echo "$AGG_PRECISION >= 85" | bc) -eq 1 ] && echo "✅" || echo "❌") |
| Recall | ${AGG_RECALL}% | ≥ 90% | $([ $(echo "$AGG_RECALL >= 90" | bc) -eq 1 ] && echo "✅" || echo "❌") |
| F1-Score | ${AGG_F1}% | ≥ 87.5% | $([ $(echo "$AGG_F1 >= 87.5" | bc) -eq 1 ] && echo "✅" || echo "❌") |

## 下一步建议

EOF

# 基于结果给出建议
if (( $(echo "$AGG_F1 >= 87.5" | bc -l) )) && (( $(echo "$AGG_RECALL >= 90" | bc -l) )); then
    echo "### ✅ GPU1激进配置已达标!" >> dual_training_comparison.md
    echo "" >> dual_training_comparison.md
    echo "建议:" >> dual_training_comparison.md
    echo "1. 停止GPU0训练" >> dual_training_comparison.md
    echo "2. 使用GPU1模型作为最终模型" >> dual_training_comparison.md
    echo "3. 继续Story 2.3" >> dual_training_comparison.md
else
    echo "### ⚠️ GPU1激进配置尚未达标" >> dual_training_comparison.md
    echo "" >> dual_training_comparison.md
    echo "建议:" >> dual_training_comparison.md
    echo "1. 继续观察GPU0保守配置" >> dual_training_comparison.md
    echo "2. 可能需要更多训练轮次" >> dual_training_comparison.md
fi

echo ""
cat dual_training_comparison.md
echo ""
echo "=================================================="
echo "对比报告已保存: dual_training_comparison.md"
echo "=================================================="
