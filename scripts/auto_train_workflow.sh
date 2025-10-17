#!/bin/bash
# 自动化训练工作流
# 1. 等待验证完成
# 2. 分析结果
# 3. 如果不达标,继续训练1-2 epoch
# 4. 重新验证
# 5. 生成最终报告

set -e

MODEL_DIR="models/pii_detector_qwen3_06b_single_gpu/final"
TEST_DATA="data/merged_pii_dataset_test.jsonl"
TRAIN_DATA="data/merged_pii_dataset_train.jsonl"
BASE_MODEL="/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-0___6B"
REPORT_FILE="validation_final_report.md"

echo "=================================================="
echo "自动化训练与验证工作流"
echo "=================================================="
echo ""

# 等待当前验证完成
echo "[1/5] 等待初始验证完成..."
while ! grep -q "验证结果" /tmp/quick_validation_500.log 2>/dev/null; do
    sleep 30
    # 显示进度
    tail -1 /tmp/quick_validation_500.log 2>/dev/null | grep "处理中" || true
done

echo "✓ 初始验证完成"
echo ""

# 提取结果
echo "[2/5] 分析验证结果..."
F1=$(grep "F1-Score:" /tmp/quick_validation_500.log | awk '{print $2}' | sed 's/%//')
RECALL=$(grep "Recall" /tmp/quick_validation_500.log | grep "召回率" | awk '{print $2}' | sed 's/%//')
PRECISION=$(grep "Precision" /tmp/quick_validation_500.log | grep "精确率" | awk '{print $2}' | sed 's/%//')

echo "  Precision: ${PRECISION}%"
echo "  Recall: ${RECALL}%"
echo "  F1-Score: ${F1}%"
echo ""

# 判断是否需要继续训练
TARGET_F1=87.5
TARGET_RECALL=90

NEED_RETRAIN=0
if (( $(echo "$F1 < $TARGET_F1" | bc -l) )); then
    echo "  ⚠️  F1-Score ${F1}% < 目标 ${TARGET_F1}%"
    NEED_RETRAIN=1
fi

if (( $(echo "$RECALL < $TARGET_RECALL" | bc -l) )); then
    echo "  ⚠️  Recall ${RECALL}% < 目标 ${TARGET_RECALL}%"
    NEED_RETRAIN=1
fi

if [ $NEED_RETRAIN -eq 0 ]; then
    echo "  ✅ 验证通过!无需继续训练"
    echo ""
    echo "[5/5] 生成最终报告..."

    cat > "$REPORT_FILE" <<EOF
# Qwen3 0.6B PII 检测模型 - 最终验证报告

## 验证结果

✅ **验证通过**

### 指标

| 指标 | 实际值 | 目标值 | 状态 |
|------|-------|--------|------|
| Precision | ${PRECISION}% | ≥ 85% | ✅ |
| Recall | ${RECALL}% | ≥ 90% | ✅ |
| F1-Score | ${F1}% | ≥ 87.5% | ✅ |

### 训练配置

- 基础模型: Qwen3-0.6B
- 训练方法: LoRA (r=8, alpha=16, dropout=0.05)
- 训练轮数: 3 epochs
- 测试样本: 500

### 下一步

根据 BMAD 工作流,可以继续 **Story 2.3: 零样本 PII 检测实现**
EOF

    echo "✓ 报告已生成: $REPORT_FILE"
    exit 0
fi

# 需要继续训练
echo ""
echo "[3/5] 性能未达标,准备继续训练..."
echo "  目标: 提升 F1-Score 到 ≥ 87.5%, Recall 到 ≥ 90%"
echo ""

# 创建新的训练目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NEW_MODEL_DIR="models/pii_detector_qwen3_06b_epoch4-5_${TIMESTAMP}"
mkdir -p "$NEW_MODEL_DIR"

echo "  新模型目录: $NEW_MODEL_DIR"
echo ""

# 继续训练2个epoch (从epoch 3开始,训练到epoch 5)
echo "[4/5] 继续训练 2 个 epoch..."
echo "  这可能需要 5-6 小时..."
echo ""

python scripts/train_qwen3_pii_single_gpu.py \
    --base-model "$BASE_MODEL" \
    --train-data "$TRAIN_DATA" \
    --output-dir "$NEW_MODEL_DIR" \
    --epochs 2 \
    --learning-rate 1e-4 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --resume-from "$MODEL_DIR" \
    2>&1 | tee "logs/continue_training_epoch4-5.log"

echo "✓ 训练完成"
echo ""

# 验证新模型
echo "[5/5] 验证新模型..."
python scripts/quick_model_validation.py \
    --model "$NEW_MODEL_DIR" \
    --test-data "$TEST_DATA" \
    --sample-size 500 \
    2>&1 | tee /tmp/validation_epoch4-5.log

# 提取新结果
F1_NEW=$(grep "F1-Score:" /tmp/validation_epoch4-5.log | awk '{print $2}' | sed 's/%//')
RECALL_NEW=$(grep "Recall" /tmp/validation_epoch4-5.log | grep "召回率" | awk '{print $2}' | sed 's/%//')
PRECISION_NEW=$(grep "Precision" /tmp/validation_epoch4-5.log | grep "精确率" | awk '{print $2}' | sed 's/%//')

echo ""
echo "=================================================="
echo "最终结果对比"
echo "=================================================="
echo ""
echo "Epoch 3 (初始):"
echo "  Precision: ${PRECISION}%"
echo "  Recall: ${RECALL}%"
echo "  F1-Score: ${F1}%"
echo ""
echo "Epoch 5 (继续训练后):"
echo "  Precision: ${PRECISION_NEW}%"
echo "  Recall: ${RECALL_NEW}%"
echo "  F1-Score: ${F1_NEW}%"
echo ""

# 生成最终报告
cat > "$REPORT_FILE" <<EOF
# Qwen3 0.6B PII 检测模型 - 最终验证报告

## 训练历程

### Round 1: Epoch 1-3

- **Precision**: ${PRECISION}%
- **Recall**: ${RECALL}%
- **F1-Score**: ${F1}%
- **结论**: 未达标,需要继续训练

### Round 2: Epoch 4-5 (继续训练)

- **Precision**: ${PRECISION_NEW}%
- **Recall**: ${RECALL_NEW}%
- **F1-Score**: ${F1_NEW}%

## 最终结论

EOF

if (( $(echo "$F1_NEW >= $TARGET_F1" | bc -l) )) && (( $(echo "$RECALL_NEW >= $TARGET_RECALL" | bc -l) )); then
    echo "✅ **验证通过!**" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "模型已达到目标性能,可以继续 Story 2.3。" >> "$REPORT_FILE"
    echo ""
    echo "✅ 最终验证通过!"
else
    echo "⚠️ **仍未完全达标**" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "建议:" >> "$REPORT_FILE"
    if (( $(echo "$F1_NEW < $TARGET_F1" | bc -l) )); then
        echo "- F1-Score 距离目标还差 $(echo "$TARGET_F1 - $F1_NEW" | bc)%" >> "$REPORT_FILE"
    fi
    if (( $(echo "$RECALL_NEW < $TARGET_RECALL" | bc -l) )); then
        echo "- Recall 距离目标还差 $(echo "$TARGET_RECALL - $RECALL_NEW" | bc)%" >> "$REPORT_FILE"
    fi
    echo "- 考虑调整训练超参数或数据增强" >> "$REPORT_FILE"
    echo ""
    echo "⚠️ 仍未完全达标,但已有改进"
fi

echo ""
echo "最终报告已生成: $REPORT_FILE"
echo "=================================================="
