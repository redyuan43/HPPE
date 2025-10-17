# 激进配置模型验证报告 (1 Epoch)

**时间**: 2025-10-15 13:52
**模型**: models/pii_detector_qwen3_06b_aggressive/final

## 训练配置

```yaml
基础模型: Qwen3-0.6B
方法: LoRA
学习率: 1.5e-4  # 比保守1e-4更激进，比原始2e-4稍低
LoRA rank: 12   # 比基线8更大
LoRA alpha: 24
batch_size: 4
gradient_accumulation: 8
effective_batch_size: 32
epochs: 1       # 只训练1 epoch
总步数: 1757
训练时间: 112.2分钟 (1:52:11)
平均速度: 3.83秒/step
```

## 验证结果

**测试集**: 100个样本 (从7,028个测试样本中随机抽取)
**耗时**: 292.4秒 (2.92秒/样本)

### 混淆矩阵

|  | 实际为PII | 实际非PII |
|--|----------|----------|
| **预测为PII** | TP: 145 | FP: 9 |
| **预测非PII** | FN: 57  | TN: - |

### 准确性指标

| 指标 | 值 | 目标 | 状态 |
|------|-----|------|------|
| **Precision (精确率)** | 94.16% | ≥85% | ✅ 超标9.2% |
| **Recall (召回率)** | 71.78% | ≥90% | ❌ 差18.2% |
| **F1-Score** | 81.46% | ≥87.5% | ❌ 差6.0% |
| **F2-Score** | 75.36% | - | ℹ️ 参考 |

**验证结论**: ❌ 未通过

## 对比分析

### 与基线对比 (3 Epochs, LR=2e-4, LoRA r=8)

| 指标 | 基线 (3 epochs) | 激进 (1 epoch) | 变化 | 趋势 |
|------|----------------|---------------|------|------|
| **Precision** | 95.37% | 94.16% | -1.21% | ⬇️ 轻微下降 |
| **Recall** | 74.31% | 71.78% | -2.53% | ⬇️ 轻微下降 |
| **F1-Score** | 83.53% | 81.46% | -2.07% | ⬇️ 轻微下降 |

### 关键发现

1. **1 epoch训练不充分**
   - 所有指标都比3 epochs基线更差
   - 差距虽小(<3%),但方向一致下降
   - 说明模型还未充分学习

2. **Precision保持优秀**
   - 94.16%仍远超目标(85%)
   - 说明模型不会乱报
   - 激进配置本身没问题

3. **Recall仍是瓶颈**
   - 71.78% vs 目标90% (差18.2%)
   - 比基线(74.31%)还低2.53%
   - 漏检问题未改善,反而恶化

## 问题诊断

### 为什么激进配置表现更差？

1. **训练不足**
   - 1 epoch只是3 epochs的1/3
   - 模型还在学习初期
   - 更大的LoRA rank (12 vs 8)需要更多训练时间

2. **学习率影响**
   - LR=1.5e-4比基线1e-4高50%
   - 更快收敛,但可能过早陷入局部最优
   - 需要更多epoch来探索

3. **容量vs训练量trade-off**
   - LoRA r=12增加了模型容量
   - 但1 epoch的数据量不足以利用这个容量
   - 反而导致欠拟合

## 推荐行动

### 方案1: 继续训练当前激进配置 (推荐 ✅)

**操作**:
```bash
# 继续训练1-2个epoch
python scripts/train_pii_detector.py \
  --model models/pii_detector_qwen3_06b_aggressive/final \
  --data data/merged_pii_dataset_train.jsonl \
  --val-data data/merged_pii_dataset_validation.jsonl \
  --output models/pii_detector_qwen3_06b_aggressive_extended \
  --lora-r 12 \
  --lora-alpha 24 \
  --batch-size 4 \
  --gradient-accumulation 8 \
  --learning-rate 1.5e-4 \
  --epochs 2  # 继续训练2个epoch (总共3 epochs)
```

**优势**:
- 复用已训练的1 epoch
- 总训练时间 = 1.9小时 + 3.8小时 = 5.7小时
- 与基线的3 epochs总时间相当

**预期**:
- Recall提升至80-85%
- F1提升至85-88%
- 可能达标或接近达标

### 方案2: 使用基线配置继续训练

**操作**:
```bash
# 继续基线配置训练1-2个epoch
python scripts/continue_training.py \
  --checkpoint models/pii_detector_qwen3_06b_lora/final \
  --epochs 2  # 从3 epochs继续到5 epochs
```

**优势**:
- 基线已经接近目标 (F1=83.53%, 差4%)
- 更保守,成功概率更高

**劣势**:
- 可能仍然无法突破Recall瓶颈
- 没有利用激进配置的探索

### 方案3: 并行测试 (如果时间允许)

**操作**: 同时继续两个配置,最后选最好的

### 推荐方案: **方案1**

**理由**:
1. 激进配置虽然1 epoch表现差,但:
   - 方向正确 (Precision高)
   - 容量更大 (LoRA r=12)
   - 学习率更高 (可能更快突破)

2. 时间成本合理:
   - 再训2 epochs ≈ 3.8小时
   - 总时间与基线3 epochs相当

3. 探索价值:
   - 如果成功,得到更好的模型
   - 如果失败,也只是多用了2小时

## 下一步行动

**建议立即执行**:
1. 继续训练激进配置2个epoch (预计3.8小时)
2. 每1 epoch验证一次
3. 如果第2 epoch达标,停止训练
4. 如果第3 epoch仍未达标,分析原因

**预计完成时间**: ~17:30 (3.8小时后)

---

**报告生成时间**: 2025-10-15 13:52
**负责人**: Claude (自动化验证)
**状态**: ⚠️ 需要继续训练
