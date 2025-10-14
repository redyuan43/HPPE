#!/bin/bash
# 一键运行完整验证流程
#
# 用途：训练完成后，自动运行所有验证步骤，生成完整的验证报告
#
# 使用方式：
#   bash scripts/run_full_validation.sh [模型路径]
#
# 示例：
#   bash scripts/run_full_validation.sh models/pii_detector_qwen3_0.6b/final

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# 默认参数
MODEL_DIR="${1:-models/pii_detector_qwen3_0.6b/final}"
TEST_DATA="data/merged_pii_dataset_test.jsonl"
OUTPUT_DIR="evaluation_results/$(date +%Y%m%d_%H%M%S)"

# 打印配置
print_step "Qwen3 0.6B PII 模型完整验证流程"
print_info "模型路径: $MODEL_DIR"
print_info "测试数据: $TEST_DATA"
print_info "输出目录: $OUTPUT_DIR"
echo ""

# 检查模型是否存在
if [ ! -d "$MODEL_DIR" ]; then
    print_error "模型目录不存在: $MODEL_DIR"
    print_info "用法: bash scripts/run_full_validation.sh [模型路径]"
    exit 1
fi

# 检查测试数据是否存在
if [ ! -f "$TEST_DATA" ]; then
    print_error "测试数据不存在: $TEST_DATA"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 记录开始时间
START_TIME=$(date +%s)

# ============================================================
# Step 1: 模型加载与健康检查
# ============================================================
print_step "[1/4] 模型加载与健康检查"
print_info "验证模型文件完整性..."

# 检查必要文件
REQUIRED_FILES=(
    "adapter_model.safetensors"
    "adapter_config.json"
    "tokenizer_config.json"
)

ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$MODEL_DIR/$file" ]; then
        print_success "✓ $file"
    else
        print_error "✗ $file (缺失)"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    print_error "模型文件不完整，无法继续验证"
    exit 1
fi

print_success "健康检查完成"

# ============================================================
# Step 2: 测试集评估（核心）
# ============================================================
print_step "[2/4] 测试集评估（核心）"
print_info "在测试集上运行推理..."

python scripts/evaluate_trained_model.py \
    --model "$MODEL_DIR" \
    --test-data "$TEST_DATA" \
    --output "$OUTPUT_DIR/test_evaluation.json" \
    > "$OUTPUT_DIR/test_evaluation.log" 2>&1

if [ $? -eq 0 ]; then
    print_success "测试集评估完成"
else
    print_error "测试集评估失败，请查看日志: $OUTPUT_DIR/test_evaluation.log"
    exit 1
fi

# 提取关键指标
if [ -f "$OUTPUT_DIR/test_evaluation.json" ]; then
    # 使用 Python 提取指标
    python3 -c "
import json
with open('$OUTPUT_DIR/test_evaluation.json', 'r') as f:
    data = json.load(f)
    metrics = data['overall_metrics']['metrics']
    print(f\"  Precision: {metrics['precision']:.2%}\")
    print(f\"  Recall:    {metrics['recall']:.2%}\")
    print(f\"  F1-Score:  {metrics['f1_score']:.2%}\")
    print(f\"  F2-Score:  {metrics['f2_score']:.2%}\")
" 2>/dev/null || print_warning "无法提取指标（需要 Python 3）"
fi

# ============================================================
# Step 3: 性能基准测试
# ============================================================
print_step "[3/4] 性能基准测试"
print_info "测量推理延迟和吞吐量..."

# 注意：这里暂时跳过性能测试（需要实现 benchmark_model.py）
print_warning "性能基准测试脚本尚未实现，跳过此步骤"
print_info "预期实现: python scripts/benchmark_model.py --model $MODEL_DIR"

# ============================================================
# Step 4: 生成验证报告
# ============================================================
print_step "[4/4] 生成验证报告"
print_info "汇总评估结果..."

# 生成简单的文本报告
REPORT_FILE="$OUTPUT_DIR/validation_report.md"

cat > "$REPORT_FILE" <<EOF
# Qwen3 0.6B PII 模型验证报告

## 📋 基本信息

- **模型路径**: \`$MODEL_DIR\`
- **测试数据**: \`$TEST_DATA\`
- **验证时间**: $(date "+%Y-%m-%d %H:%M:%S")
- **输出目录**: \`$OUTPUT_DIR\`

---

## 📊 验证结果

### 1. 模型健康检查

✅ **通过**

- 模型文件完整
- 所有必要文件存在

### 2. 测试集评估

EOF

# 从 JSON 提取指标并写入报告
if [ -f "$OUTPUT_DIR/test_evaluation.json" ]; then
    python3 -c "
import json

with open('$OUTPUT_DIR/test_evaluation.json', 'r') as f:
    data = json.load(f)

overall = data['overall_metrics']
metrics = overall['metrics']
confusion = overall['confusion_matrix']
stats = overall['statistics']

# 判定结果
f1 = metrics['f1_score']
recall = metrics['recall']
passed = f1 >= 0.875 and recall >= 0.90

status = '✅ **通过验证**' if passed else '❌ **未通过验证**'

report = f'''
{status}

**准确性指标**:

| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| Precision | {metrics['precision']:.2%} | ≥ 85% | {'✅' if metrics['precision'] >= 0.85 else '❌'} |
| Recall | {metrics['recall']:.2%} | ≥ 90% | {'✅' if metrics['recall'] >= 0.90 else '❌'} |
| F1-Score | {metrics['f1_score']:.2%} | ≥ 87.5% | {'✅' if metrics['f1_score'] >= 0.875 else '❌'} |
| F2-Score | {metrics['f2_score']:.2%} | ≥ 88% | {'✅' if metrics['f2_score'] >= 0.88 else '❌'} |

**混淆矩阵**:

- True Positives (正确检测): {confusion['true_positives']:,}
- False Positives (误报): {confusion['false_positives']:,}
- False Negatives (漏报): {confusion['false_negatives']:,}

**样本统计**:

- 总样本数: {stats['total_samples']:,}
- 成功推理: {stats['success_count']:,}
- 推理失败: {stats['error_count']:,}

'''

with open('$REPORT_FILE', 'a') as f:
    f.write(report)
" 2>/dev/null || print_warning "无法生成详细报告（需要 Python 3）"
fi

# 追加结论
cat >> "$REPORT_FILE" <<EOF

### 3. 性能基准测试

⚠️ **跳过**（脚本未实现）

### 4. 鲁棒性测试

⚠️ **跳过**（脚本未实现）

---

## 🎯 结论

EOF

# 从 JSON 判定最终结果
if [ -f "$OUTPUT_DIR/test_evaluation.json" ]; then
    python3 -c "
import json

with open('$OUTPUT_DIR/test_evaluation.json', 'r') as f:
    data = json.load(f)

metrics = data['overall_metrics']['metrics']
f1 = metrics['f1_score']
recall = metrics['recall']
precision = metrics['precision']

passed = f1 >= 0.875 and recall >= 0.90

if passed:
    conclusion = '''
### ✅ 验证通过

模型已达到目标指标，可以部署到测试环境。

**通过项**:
- F1-Score 达标 ({:.2%} ≥ 87.5%)
- Recall 达标 ({:.2%} ≥ 90%)

**下一步行动**:
1. 部署到测试环境
2. 进行 A/B 测试
3. 收集真实场景反馈
'''.format(f1, recall)
else:
    issues = []
    if f1 < 0.875:
        issues.append(f'- F1-Score 未达标 ({f1:.2%} < 87.5%)')
    if recall < 0.90:
        issues.append(f'- Recall 未达标 ({recall:.2%} < 90%)')
    if precision < 0.85:
        issues.append(f'- Precision 偏低 ({precision:.2%} < 85%)')

    conclusion = '''
### ❌ 验证未通过

模型未达到目标指标，需要改进后重新验证。

**未通过项**:
{}

**改进建议**:
1. 增加训练轮次
2. 调整超参数
3. 检查训练数据质量
4. 考虑使用更大模型（Qwen3-1.7B）
'''.format('\n'.join(issues))

with open('$REPORT_FILE', 'a') as f:
    f.write(conclusion)
" 2>/dev/null
fi

cat >> "$REPORT_FILE" <<EOF

---

## 📁 详细结果文件

- 测试集评估: \`test_evaluation.json\`
- 评估日志: \`test_evaluation.log\`

查看完整结果:
\`\`\`bash
cat $OUTPUT_DIR/test_evaluation.json | python -m json.tool
\`\`\`

---

**报告生成时间**: $(date "+%Y-%m-%d %H:%M:%S")
EOF

print_success "验证报告生成完成"

# ============================================================
# 汇总结果
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

print_step "验证流程完成"
print_success "总耗时: ${MINUTES}分${SECONDS}秒"
print_info ""
print_info "查看验证报告:"
print_info "  cat $OUTPUT_DIR/validation_report.md"
print_info ""
print_info "查看详细结果:"
print_info "  cat $OUTPUT_DIR/test_evaluation.json"
print_info ""

# 显示报告摘要
if [ -f "$REPORT_FILE" ]; then
    echo ""
    print_step "验证报告摘要"
    # 提取关键结论
    grep -A 10 "^## 🎯 结论" "$REPORT_FILE" || true
fi
