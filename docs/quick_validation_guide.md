# Qwen3 0.6B 模型训练后验证 - 快速指南

## 📋 概述

本指南提供 **5 分钟快速验证** 训练后模型的步骤。

---

## 🚀 一键验证（推荐）

训练完成后，运行以下命令即可完成完整验证：

```bash
# 默认验证路径
bash scripts/run_full_validation.sh

# 或指定模型路径
bash scripts/run_full_validation.sh models/pii_detector_qwen3_0.6b/final
```

**验证内容**：
1. ✅ 模型文件完整性检查
2. ✅ 测试集准确性评估
3. ✅ 生成验证报告

**预计耗时**: 10-20 分钟（取决于测试集大小）

---

## 📊 查看验证结果

### 方法 1: 查看验证报告（推荐）

```bash
# 查看最新的验证报告
cat evaluation_results/*/validation_report.md

# 或使用 Markdown 查看器
markdown-viewer evaluation_results/20251014_143000/validation_report.md
```

**报告包含**：
- ✅ 准确性指标（Precision、Recall、F1、F2）
- ✅ 混淆矩阵
- ✅ 通过/未通过判定
- ✅ 改进建议

### 方法 2: 查看 JSON 详细结果

```bash
# 查看详细评估结果
cat evaluation_results/20251014_143000/test_evaluation.json | python -m json.tool

# 提取关键指标
python3 -c "
import json
with open('evaluation_results/20251014_143000/test_evaluation.json', 'r') as f:
    data = json.load(f)
    metrics = data['overall_metrics']['metrics']
    print(f'Precision: {metrics[\"precision\"]:.2%}')
    print(f'Recall:    {metrics[\"recall\"]:.2%}')
    print(f'F1-Score:  {metrics[\"f1_score\"]:.2%}')
"
```

---

## ✅ 通过标准

模型需同时满足以下条件才能通过验证：

| 指标 | 通过标准 | 权重 |
|------|---------|------|
| **F1-Score** | ≥ 87.5% | 核心指标 |
| **Recall** | ≥ 90% | 核心指标 |
| **Precision** | ≥ 85% | 参考指标 |

**判定逻辑**：
- ✅ **通过**: F1 ≥ 87.5% AND Recall ≥ 90%
- ❌ **未通过**: 任一核心指标不达标

---

## 🎯 典型验证结果

### 示例 1: 通过验证

```
========================================
验证结果摘要
========================================

【总体指标】
  样本总数: 5,000
  成功推理: 4,998
  推理失败: 2

  混淆矩阵:
    TP (正确检测): 4,520
    FP (误报): 380
    FN (漏报): 420

  准确性指标:
    Precision: 92.25%
    Recall:    91.50%
    F1-Score:  91.87%
    F2-Score:  91.65%

【验证结果】
  ✅ 通过验证！
```

**下一步行动**：
1. 部署到测试环境
2. 进行 A/B 测试
3. 收集真实场景反馈

---

### 示例 2: 未通过验证

```
========================================
验证结果摘要
========================================

【总体指标】
  准确性指标:
    Precision: 85.20%
    Recall:    88.30%
    F1-Score:  86.72%
    F2-Score:  87.40%

【验证结果】
  ❌ 未通过验证
     - F1-Score 未达标 (86.72% < 87.5%)
     - Recall 未达标 (88.30% < 90%)
```

**改进建议**：

1. **增加训练轮次**（最快）
   ```bash
   # 在原有基础上继续训练 1-2 个 epoch
   python scripts/train_pii_detector.py \
       --model models/pii_detector_qwen3_0.6b/final \
       --data data/merged_pii_dataset_train.jsonl \
       --epochs 2 \
       --output models/pii_detector_qwen3_0.6b_continued
   ```

2. **调整超参数**
   - 降低学习率（2e-4 → 1e-4）
   - 增加 LoRA rank（8 → 16）

3. **分析 Bad Case**
   ```bash
   # 查看误报和漏报样本
   python scripts/analyze_errors.py \
       --results evaluation_results/20251014_143000/test_evaluation.json
   ```

4. **考虑更大模型**
   - 如果 0.6B 模型持续不达标，尝试 Qwen3-1.7B

---

## 🔧 高级验证选项

### 1. 单独运行测试集评估

```bash
python scripts/evaluate_trained_model.py \
    --model models/pii_detector_qwen3_0.6b/final \
    --test-data data/merged_pii_dataset_test.jsonl \
    --output evaluation_results/custom_evaluation.json
```

**参数说明**：
- `--model`: 模型路径（包含 adapter_model.safetensors）
- `--test-data`: 测试数据路径（.jsonl 格式）
- `--output`: 输出结果文件路径
- `--max-samples`: （可选）限制测试样本数，用于快速测试
- `--device`: （可选）指定设备（cuda/cpu/auto）

### 2. 快速测试（100 样本）

```bash
python scripts/evaluate_trained_model.py \
    --model models/pii_detector_qwen3_0.6b/final \
    --test-data data/merged_pii_dataset_test.jsonl \
    --output evaluation_results/quick_test.json \
    --max-samples 100
```

### 3. 对比多个模型

```bash
# 评估模型 A
bash scripts/run_full_validation.sh models/model_a

# 评估模型 B
bash scripts/run_full_validation.sh models/model_b

# 对比结果
python scripts/compare_models.py \
    evaluation_results/20251014_143000/test_evaluation.json \
    evaluation_results/20251014_150000/test_evaluation.json
```

---

## 🐛 故障排查

### 问题 1: 模型加载失败

**错误信息**：
```
FileNotFoundError: 模型文件不完整
```

**解决方案**：
1. 检查模型目录是否包含必要文件：
   ```bash
   ls -lh models/pii_detector_qwen3_0.6b/final/
   # 必须包含:
   # - adapter_model.safetensors
   # - adapter_config.json
   # - tokenizer_config.json
   ```

2. 如果文件不完整，重新训练或从备份恢复

---

### 问题 2: 推理速度过慢

**现象**：评估耗时 > 1 小时

**解决方案**：

1. **使用 GPU 加速**
   ```bash
   # 确认 GPU 可用
   nvidia-smi

   # 指定 GPU 运行
   CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_trained_model.py ...
   ```

2. **先测试小样本**
   ```bash
   # 使用 100 个样本快速测试
   python scripts/evaluate_trained_model.py \
       --max-samples 100 \
       ...
   ```

3. **检查 CPU/GPU 占用**
   ```bash
   # 实时监控
   watch -n 1 nvidia-smi
   htop
   ```

---

### 问题 3: 显存不足（OOM）

**错误信息**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**：

1. **减少 batch size**（修改 evaluate_trained_model.py）
2. **使用 CPU 推理**
   ```bash
   python scripts/evaluate_trained_model.py \
       --device cpu \
       ...
   ```
3. **使用模型量化**（需实现量化脚本）

---

### 问题 4: JSON 解析失败

**现象**：模型输出无法解析为 JSON

**解决方案**：

1. 查看模型原始输出：
   ```python
   # 手动测试
   python -c "
   from scripts.evaluate_trained_model import TrainedModelEvaluator
   evaluator = TrainedModelEvaluator('models/pii_detector_qwen3_0.6b/final')
   result = evaluator.detect_pii('我叫张三，手机号13800138000')
   print(result)
   "
   ```

2. 可能原因：
   - 模型训练不充分（输出格式不规范）
   - 训练数据格式问题
   - 需要继续训练

---

## 📚 相关文档

- 📘 [完整验证方案](model_validation_plan.md) - 详细的验证流程和指标说明
- 📗 [训练指南](train_model.md) - 如何训练 PII 检测模型
- 📕 [模型改进指南](model_improvement.md) - 如何提升模型性能

---

## 🎯 快速参考命令

```bash
# 训练后立即验证
bash scripts/run_full_validation.sh

# 查看验证报告
cat evaluation_results/*/validation_report.md

# 快速测试（100 样本）
python scripts/evaluate_trained_model.py \
    --model models/pii_detector_qwen3_0.6b/final \
    --test-data data/merged_pii_dataset_test.jsonl \
    --max-samples 100 \
    --output evaluation_results/quick_test.json

# 查看 JSON 结果
cat evaluation_results/*/test_evaluation.json | python -m json.tool
```

---

## ✨ 总结

### 验证流程（简化版）

```
训练完成 → 运行验证脚本 → 查看报告 → 判定是否通过
     ↓                                        ↓
     ↓                               ┌────────┴────────┐
     ↓                               ↓                 ↓
     ↓                            通过              未通过
     ↓                               ↓                 ↓
     └────────────────────────────→ 部署          改进重训
```

### 核心要点

1. ✅ **一键验证**: 使用 `run_full_validation.sh` 自动完成所有步骤
2. ✅ **快速反馈**: 10-20 分钟即可获得验证结果
3. ✅ **清晰标准**: F1 ≥ 87.5% AND Recall ≥ 90%
4. ✅ **详细报告**: 包含指标、混淆矩阵、改进建议

### 时间估算

| 任务 | 耗时 |
|------|------|
| 模型加载 | 1-2 分钟 |
| 测试集评估（5000样本） | 10-20 分钟 |
| 快速测试（100样本） | 1-2 分钟 |
| 报告生成 | < 1 分钟 |

---

**最后更新**: 2025-10-14
**维护者**: AI 工程师团队
