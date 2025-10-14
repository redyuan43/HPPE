# PII 检测模型选择指南

## 您的硬件配置
- 2x RTX 3060 (12GB each) = 24GB 总显存
- 当前运行：Qwen3-8B-AWQ (占用 12GB)

---

## 方案对比

### 方案 A：Qwen3-8B（继续使用当前模型）⭐⭐⭐

**配置：**
```yaml
模型：Qwen/Qwen3-8B-AWQ (4位量化)
用途：Zero-shot 推理（不微调）
显存：12GB
```

**优点：**
- ✅ 无需训练（直接使用）
- ✅ 强大的推理能力
- ✅ 支持 Thinking Mode

**缺点：**
- ❌ 推理较慢（8-12秒/样本）
- ❌ 输出格式不稳定
- ❌ 准确率一般（~85-90%）
- ❌ 量化版本**不能微调**

**适用场景：**
- 快速原型验证
- 低频使用
- 不要求高准确率

---

### 方案 B：Qwen2-1.5B SFT 微调 ⭐⭐⭐⭐⭐（推荐）

**配置：**
```yaml
基础模型：Qwen/Qwen2-1.5B (FP16)
训练方法：LoRA微调
训练数据：50k PII样本
显存：14GB（单卡训练）
```

**优点：**
- ✅ **推理快** (2-4秒/样本，快3-6倍)
- ✅ **准确率高** (95-99%，专门训练)
- ✅ **输出稳定** (固定JSON格式)
- ✅ **显存友好** (6-8GB推理)
- ✅ **训练时间短** (2-3小时)

**缺点：**
- ❌ 需要训练时间
- ❌ 需要标注数据（但我们已准备好脚本）
- ❌ 参数量小于8B（但对PII检测足够）

**适用场景：**
- **生产部署**（您的场景）
- 批处理为主（70%）
- 高准确率要求（99%）
- 云端脱敏场景

---

### 方案 C：Qwen3-8B 全精度微调 ⭐⭐

**配置：**
```yaml
基础模型：Qwen/Qwen3-8B (非量化版本)
训练方法：LoRA微调
显存：18-20GB
```

**优点：**
- ✅ 保留Qwen3优势
- ✅ 可以微调

**缺点：**
- ❌ **显存紧张**（20GB，需要优化）
- ❌ 训练时间长（6-8小时）
- ❌ 推理仍较慢
- ❌ 需要下载非量化版本（16GB下载）

**可行性分析：**
```
您的显存：2x 12GB = 24GB
需求：18-20GB

结论：勉强可行，但需要：
1. 使用梯度检查点
2. 小批次大小（batch_size=2）
3. 大梯度累积（gradient_accumulation=8-16）
4. 可能需要CPU offload
```

---

### 方案 D：混合架构（最优）⭐⭐⭐⭐⭐

**配置：**
```yaml
推理模型：Qwen2-1.5B SFT (专门训练的PII检测器)
备用模型：Qwen3-8B-AWQ (复杂场景fallback)
```

**架构：**
```
输入文本
   ↓
┌──────────────────────┐
│  快速预检查 (Regex)  │ ← 0.02ms
└──────────────────────┘
   ↓
包含结构化PII？
   ├─ 是 → 返回结果
   └─ 否 ↓
┌──────────────────────┐
│ Qwen2-1.5B SFT       │ ← 2-4秒 (主力)
│ (PII专用模型)        │
└──────────────────────┘
   ↓
检测到复杂实体？
   ├─ 否 → 返回结果
   └─ 是 ↓
┌──────────────────────┐
│ Qwen3-8B-AWQ         │ ← 8-12秒 (少数场景)
│ (复杂推理)           │
└──────────────────────┘
   ↓
返回最终结果
```

**优点：**
- ✅ **性能最优**（90%情况用快速模型）
- ✅ **准确率高**（专门训练+强大备用）
- ✅ **资源高效**（按需使用）
- ✅ **灵活切换**

**实现：**
```python
class HybridPIIDetector:
    def __init__(self):
        self.fast_model = Qwen2_1_5B_SFT()   # 主力
        self.powerful_model = Qwen3_8B()     # 备用

    def detect(self, text):
        # 1. Regex快速检测
        regex_result = self.regex_detector.detect(text)
        if regex_result.confidence > 0.95:
            return regex_result

        # 2. 微调模型检测（主力）
        sft_result = self.fast_model.detect(text)
        if sft_result.confidence > 0.90:
            return sft_result

        # 3. 复杂场景用强大模型
        if needs_deep_reasoning(text):
            return self.powerful_model.detect(text)

        return sft_result
```

---

## 推荐决策树

```
您的需求分析：
├─ 准确率要求：99% ✓
├─ 主要场景：批处理 70% ✓
├─ 数据脱敏 ✓
└─ 显存限制：24GB ✓

推荐方案：
1. 立即训练：方案 B (Qwen2-1.5B SFT)
2. 长期部署：方案 D (混合架构)
3. 保留：Qwen3-8B-AWQ (复杂场景backup)
```

---

## 实施建议

### 阶段 1：训练 Qwen2-1.5B（本周）

```bash
# 1. 准备数据
python scripts/download_datasets.py --all
python scripts/generate_synthetic_pii.py --num-samples 30000
python scripts/merge_datasets.py --all --total-samples 50000

# 2. 训练模型（2-3小时）
python scripts/train_pii_detector.py \
    --model Qwen/Qwen2-1.5B \
    --data data/merged_pii_dataset_train.jsonl \
    --val-data data/merged_pii_dataset_validation.jsonl \
    --lora-r 8 \
    --batch-size 4 \
    --epochs 3 \
    --output models/pii_detector_qwen2_1.5b
```

### 阶段 2：对比测试（本周）

```bash
# 测试两个模型
python examples/compare_models.py \
    --model1 "Qwen3-8B-AWQ" \
    --model2 "Qwen2-1.5B-SFT" \
    --test-data data/merged_pii_dataset_test.jsonl

# 对比指标：
# - 准确率
# - 推理速度
# - 显存占用
# - 输出稳定性
```

### 阶段 3：部署混合架构（下周）

```python
# 创建混合检测器
detector = HybridPIIDetector(
    fast_model="models/pii_detector_qwen2_1.5b",
    backup_model="Qwen3-8B-AWQ"
)

# 批处理优化
batch_results = detector.detect_batch(
    texts=large_dataset,
    use_fast_model_ratio=0.9  # 90%用快速模型
)
```

---

## 如果您坚持用 Qwen3-8B 微调

### 可行性：勉强可以

```bash
# 下载非量化版本（16GB）
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B

# 极限优化配置
python scripts/train_pii_detector.py \
    --model models/Qwen3-8B \
    --data data/merged_pii_dataset_train.jsonl \
    --lora-r 4 \                    # 减小rank
    --batch-size 1 \                # 最小批次
    --gradient-accumulation 16 \    # 大梯度累积
    --max-length 256 \              # 减小序列长度
    --epochs 2 \
    --output models/pii_detector_qwen3_8b
```

### 风险：
- ⚠️ 可能OOM（显存不足）
- ⚠️ 训练时间长（6-8小时）
- ⚠️ 推理仍然慢
- ⚠️ 性价比不高

---

## 总结

### 最佳实践路径

```
第1周：训练 Qwen2-1.5B
   ↓
第2周：评估效果，对比 Qwen3-8B
   ↓
第3周：部署混合架构
   ↓
长期：根据实际效果调整比例
```

### 预期效果

| 指标 | Qwen3-8B (现在) | Qwen2-1.5B SFT (训练后) | 提升 |
|------|----------------|------------------------|------|
| 准确率 | 85-90% | **95-99%** | +10% |
| 推理速度 | 8-12秒 | **2-4秒** | 3-6倍 |
| 显存占用 | 12GB | **6-8GB** | 节省50% |
| 输出稳定性 | 一般 | **优秀** | 显著提升 |
| 批处理10万样本 | ~20小时 | **~5小时** | 75%时间节省 |

---

**建议：先训练 Qwen2-1.5B，再决定是否需要保留 Qwen3-8B**

如果Qwen2-1.5B效果达到99%，您可以：
- 主要使用：Qwen2-1.5B SFT（快速+准确）
- 偶尔使用：Qwen3-8B（极复杂场景）

**不建议：直接微调Qwen3-8B**
- 投入高（时间+资源）
- 收益低（推理仍慢）
- 风险大（可能OOM）
