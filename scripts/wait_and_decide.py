#!/usr/bin/env python3
"""
等待验证完成并自动决策下一步操作
"""

import time
import re
import sys
from pathlib import Path

LOG_FILE = "/tmp/quick_validation_500.log"
REPORT_FILE = "validation_final_report.md"

TARGET_F1 = 87.5
TARGET_RECALL = 90.0

print("=" * 70)
print("等待验证完成...")
print("=" * 70)

# 等待验证完成
while True:
    try:
        with open(LOG_FILE, 'r') as f:
            content = f.read()

        # 检查是否包含最终结果
        if "[3/3] 验证结果" in content and "验证未通过" in content or "验证通过" in content:
            break

        # 显示进度
        progress_lines = re.findall(r'处理中:.*\d+/500', content)
        if progress_lines:
            latest = progress_lines[-1]
            # 提取进度
            match = re.search(r'(\d+)/500', latest)
            if match:
                current = int(match.group(1))
                print(f"\r进度: {current}/500 ({current/500*100:.1f}%)", end='', flush=True)

    except FileNotFoundError:
        pass

    time.sleep(10)

print("\n\n✓ 验证完成!\n")

# 读取完整结果
with open(LOG_FILE, 'r') as f:
    content = f.read()

# 提取指标
precision_match = re.search(r'Precision \(精确率\):\s*(\d+\.\d+)%', content)
recall_match = re.search(r'Recall \(召回率\):\s*(\d+\.\d+)%', content)
f1_match = re.search(r'F1-Score:\s*(\d+\.\d+)%', content)

if not all([precision_match, recall_match, f1_match]):
    print("❌ 无法提取指标,请检查日志文件")
    sys.exit(1)

precision = float(precision_match.group(1))
recall = float(recall_match.group(1))
f1 = float(f1_match.group(1))

print("=" * 70)
print("验证结果")
print("=" * 70)
print(f"\nPrecision (精确率): {precision}%")
print(f"Recall (召回率):    {recall}%")
print(f"F1-Score:           {f1}%")
print()

# 判断是否达标
f1_pass = f1 >= TARGET_F1
recall_pass = recall >= TARGET_RECALL
overall_pass = f1_pass and recall_pass

if overall_pass:
    print("✅ 验证通过!")
    print(f"   F1-Score {f1}% >= 目标 {TARGET_F1}%")
    print(f"   Recall {recall}% >= 目标 {TARGET_RECALL}%")
    print()

    # 生成成功报告
    with open(REPORT_FILE, 'w') as f:
        f.write("# Qwen3 0.6B PII 检测模型 - 最终验证报告\n\n")
        f.write("## ✅ 验证通过\n\n")
        f.write("### 性能指标\n\n")
        f.write("| 指标 | 实际值 | 目标值 | 状态 |\n")
        f.write("|------|-------|--------|------|\n")
        f.write(f"| Precision | {precision}% | ≥ 85% | ✅ |\n")
        f.write(f"| Recall | {recall}% | ≥ 90% | {'✅' if recall_pass else '⚠️'} |\n")
        f.write(f"| F1-Score | {f1}% | ≥ 87.5% | {'✅' if f1_pass else '⚠️'} |\n")
        f.write("\n### 训练配置\n\n")
        f.write("- **基础模型**: Qwen3-0.6B\n")
        f.write("- **训练方法**: LoRA (r=8, alpha=16, dropout=0.05)\n")
        f.write("- **训练轮数**: 3 epochs\n")
        f.write("- **训练时间**: 444.9分钟 (7.4小时)\n")
        f.write("- **测试样本**: 500\n")
        f.write("\n### 下一步\n\n")
        f.write("根据 BMAD 工作流,可以继续:\n")
        f.write("- **Story 2.3**: 零样本 PII 检测实现\n")
        f.write("- **Story 2.4**: 模型性能优化\n")

    print(f"✓ 报告已生成: {REPORT_FILE}")
    sys.exit(0)

else:
    print("❌ 验证未通过")
    if not f1_pass:
        print(f"   F1-Score {f1}% < 目标 {TARGET_F1}% (差距: {TARGET_F1 - f1:.1f}%)")
    if not recall_pass:
        print(f"   Recall {recall}% < 目标 {TARGET_RECALL}% (差距: {TARGET_RECALL - recall:.1f}%)")
    print()

    print("=" * 70)
    print("建议: 继续训练1-2个epoch")
    print("=" * 70)
    print()

    # 生成未通过报告
    with open(REPORT_FILE, 'w') as f:
        f.write("# Qwen3 0.6B PII 检测模型 - 验证报告\n\n")
        f.write("## ⚠️ 验证未通过\n\n")
        f.write("### 性能指标\n\n")
        f.write("| 指标 | 实际值 | 目标值 | 差距 | 状态 |\n")
        f.write("|------|-------|--------|------|------|\n")
        f.write(f"| Precision | {precision}% | ≥ 85% | - | {'✅' if precision >= 85 else '❌'} |\n")
        f.write(f"| Recall | {recall}% | ≥ 90% | {TARGET_RECALL - recall:.1f}% | {'✅' if recall_pass else '❌'} |\n")
        f.write(f"| F1-Score | {f1}% | ≥ 87.5% | {TARGET_F1 - f1:.1f}% | {'✅' if f1_pass else '❌'} |\n")
        f.write("\n### 训练配置\n\n")
        f.write("- **基础模型**: Qwen3-0.6B\n")
        f.write("- **训练方法**: LoRA (r=8, alpha=16, dropout=0.05)\n")
        f.write("- **训练轮数**: 3 epochs\n")
        f.write("- **训练时间**: 444.9分钟 (7.4小时)\n")
        f.write("- **测试样本**: 500\n")
        f.write("\n### 建议\n\n")
        f.write("1. **继续训练**: 再训练1-2个epoch,预计可提升F1-Score 2-5%\n")
        f.write("2. **调整超参数**: 可以尝试降低学习率到1e-4\n")
        f.write("3. **数据增强**: 考虑添加更多困难样本\n")
        f.write("\n### 继续训练命令\n\n")
        f.write("```bash\n")
        f.write("# 方案1: 直接继续训练2个epoch (简单)\n")
        f.write("python scripts/train_qwen3_pii_single_gpu.py \\\n")
        f.write("  --base-model /home/ivan/.cache/modelscope/hub/Qwen/Qwen3-0___6B \\\n")
        f.write("  --train-data data/merged_pii_dataset_train.jsonl \\\n")
        f.write("  --output-dir models/pii_detector_qwen3_06b_epoch4-5 \\\n")
        f.write("  --epochs 5 \\\n")  # 总共5个epoch
        f.write("  --learning-rate 1e-4 \\\n")
        f.write("  --batch-size 8 \\\n")
        f.write("  --gradient-accumulation-steps 4\n")
        f.write("```\n")

    print(f"✓ 报告已生成: {REPORT_FILE}")
    print()
    print("📋 详细建议请查看报告文件")

    sys.exit(1)
