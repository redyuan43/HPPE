# Qwen3 0.6B PII 模型训练后验证方案

## 📋 目录

1. [验证目标与原则](#1-验证目标与原则)
2. [验证指标体系](#2-验证指标体系)
3. [验证流程](#3-验证流程)
4. [自动化验证工具](#4-自动化验证工具)
5. [验证基准与通过标准](#5-验证基准与通过标准)
6. [持续改进方案](#6-持续改进方案)

---

## 1. 验证目标与原则

### 1.1 核心目标

**目标**：确保训练后的 Qwen3 0.6B 模型能够可靠、准确地识别文本中的 PII 信息。

**关键问题**：
- ✅ 模型是否正确学习了 PII 检测任务？
- ✅ 模型在实际场景中的表现如何？
- ✅ 模型是否存在过拟合或欠拟合？
- ✅ 模型相比基线（正则引擎）有何优势？
- ✅ 模型的推理性能是否满足业务需求？

### 1.2 SOLID 验证原则

遵循 SOLID 原则设计验证流程：

- **S (Single Responsibility)**: 每个验证模块专注单一维度（准确性/性能/鲁棒性）
- **O (Open-Closed)**: 验证框架可扩展（新增 PII 类型、新指标）
- **L (Liskov Substitution)**: 不同评估器可互相替换
- **I (Interface Segregation)**: 验证接口专一，不强制实现无关功能
- **D (Dependency Inversion)**: 依赖抽象接口，而非具体实现

### 1.3 KISS & YAGNI 原则

- **KISS**: 优先使用简单直接的验证方法（混淆矩阵、F1-Score）
- **YAGNI**: 只实现当前明确需要的验证指标，避免过度设计
- **DRY**: 复用现有的 `evaluate_llm_vs_regex.py` 评估框架

---

## 2. 验证指标体系

### 2.1 准确性指标（核心）

#### 2.1.1 分类指标

基于混淆矩阵计算：

| 指标 | 计算公式 | 目标值 | 优先级 |
|------|---------|--------|--------|
| **Precision（精确率）** | TP / (TP + FP) | ≥ 85% | ⭐⭐⭐ |
| **Recall（召回率）** | TP / (TP + FN) | ≥ 90% | ⭐⭐⭐⭐⭐ |
| **F1-Score** | 2 × (P × R) / (P + R) | ≥ 87.5% | ⭐⭐⭐⭐ |
| **F2-Score** | 5 × (P × R) / (4P + R) | ≥ 88% | ⭐⭐⭐ |

**权重说明**：
- PII 检测场景更关注 **召回率**（不能漏检敏感信息）
- F2-Score 更侧重召回率，适合 PII 检测
- 精确率过低会导致大量误报，增加人工审核成本

#### 2.1.2 实体级指标

针对每种 PII 类型单独评估：

```
支持的 PII 类型（当前项目）:
1. PERSON_NAME（人名）
2. ADDRESS（地址）
3. ORGANIZATION（组织机构名）
4. CHINA_ID_CARD（中国身份证号）
5. CHINA_PHONE（中国手机号）
6. CHINA_BANK_CARD（中国银行卡号）
7. EMAIL（电子邮件）
8. IP_ADDRESS（IP地址）
9. CREDIT_CARD（信用卡号）
```

**验证方式**：
```python
# 每种 PII 类型单独计算 P/R/F1
metrics_by_type = {
    "PERSON_NAME": {"precision": 0.92, "recall": 0.88, "f1": 0.90},
    "ADDRESS": {"precision": 0.85, "recall": 0.91, "f1": 0.88},
    ...
}
```

#### 2.1.3 位置匹配精度

验证模型是否准确定位 PII 在文本中的位置：

- **Exact Match（严格匹配）**: 起始位置和结束位置完全一致
- **Overlap Match（重叠匹配）**: IoU (Intersection over Union) ≥ 0.5
- **Partial Match（部分匹配）**: 允许边界偏差 ±2 字符

**基准**：
- Exact Match Rate ≥ 80%
- Overlap Match Rate ≥ 95%

### 2.2 性能指标

#### 2.2.1 推理延迟

| 指标 | 目标值 | 测量方法 |
|------|--------|---------|
| **P50 延迟** | ≤ 300ms | 中位数 |
| **P95 延迟** | ≤ 800ms | 95分位数 |
| **P99 延迟** | ≤ 1500ms | 99分位数 |
| **平均延迟** | ≤ 400ms | 算术平均 |

**测试条件**：
- 文本长度: 100-500 字符（典型长度）
- 硬件: RTX 3060 12GB
- 量化: 4-bit / 8-bit（视具体训练配置）
- Batch Size: 1（单条推理）

#### 2.2.2 吞吐量

| 指标 | 目标值 | 适用场景 |
|------|--------|---------|
| **单 GPU 吞吐量** | ≥ 10 req/s | 实时检测 |
| **批处理吞吐量** | ≥ 100 doc/s | 批量脱敏 |

#### 2.2.3 显存占用

| 配置 | 预期显存 | 验证方法 |
|------|---------|---------|
| FP16 推理 | ≤ 3GB | `nvidia-smi` |
| 4-bit 量化 | ≤ 1.5GB | `nvidia-smi` |
| 8-bit 量化 | ≤ 2GB | `nvidia-smi` |

### 2.3 鲁棒性指标

#### 2.3.1 泛化能力

验证模型在不同场景下的表现：

| 场景类别 | 描述 | 评估方法 |
|---------|------|---------|
| **简单场景** | 单一 PII，标准格式 | F1 ≥ 95% |
| **复杂场景** | 多种 PII，非标准格式 | F1 ≥ 85% |
| **混合场景** | 中英文混合，专业术语 | F1 ≥ 80% |
| **边界场景** | 歧义、错别字、方言 | Recall ≥ 70% |

#### 2.3.2 鲁棒性测试用例

| 类型 | 示例 | 预期行为 |
|------|------|---------|
| **错别字** | "张彡"（三写错） | 仍应检测到人名 |
| **变体格式** | "138-0013-8000"（手机号带连字符） | 正确识别 |
| **上下文歧义** | "我在长城工作"（地点 vs 公司） | 需上下文判断 |
| **长文本** | 1000+ 字符 | 不应遗漏 PII |

### 2.4 对比基准指标

与正则引擎（Regex）对比：

| 维度 | 正则引擎 | Qwen3 0.6B 模型 | 要求 |
|------|---------|----------------|------|
| **结构化 PII 准确率** | 99.2% | - | ≥ 95%（允许略低） |
| **上下文 PII 准确率** | ~30% | - | ≥ 80%（大幅超越） |
| **推理延迟** | ~2.5ms | - | ≤ 400ms（允许慢） |
| **误报率** | 0.8% | - | ≤ 3%（允许略高） |

**通过标准**：
- ✅ 在上下文 PII 检测上显著优于正则（+50%）
- ✅ 在结构化 PII 检测上接近正则（-5% 以内）
- ✅ 推理延迟在可接受范围内（<1秒）

---

## 3. 验证流程

### 3.1 整体流程图

```
┌─────────────────────────────────────────────────────────┐
│               训练后模型验证流程                          │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ Step 1: 模型加载与健康检查      │
        │ - 加载微调后的模型              │
        │ - 验证模型文件完整性            │
        │ - 测试基础推理功能              │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ Step 2: 测试集评估（核心）      │
        │ - 在测试集上运行推理            │
        │ - 计算准确性指标                │
        │ - 生成混淆矩阵                  │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ Step 3: 性能基准测试           │
        │ - 测量推理延迟                  │
        │ - 测量吞吐量                    │
        │ - 监控显存占用                  │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ Step 4: 鲁棒性验证             │
        │ - 边界用例测试                  │
        │ - 长文本测试                    │
        │ - 异常格式测试                  │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ Step 5: 对比基准（与 Regex）   │
        │ - 并行运行 LLM 和 Regex        │
        │ - 对比准确性差异                │
        │ - 对比性能差异                  │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ Step 6: 生成验证报告           │
        │ - 汇总所有指标                  │
        │ - 判定是否通过验证              │
        │ - 输出改进建议                  │
        └───────────────────────────────┘
```

### 3.2 详细步骤

#### Step 1: 模型加载与健康检查

**目的**：确保模型文件完整，可以正常加载和推理。

**验证项**：
```python
# 1. 文件完整性检查
assert (model_dir / "adapter_model.safetensors").exists()  # LoRA 权重
assert (model_dir / "adapter_config.json").exists()        # LoRA 配置
assert (model_dir / "tokenizer_config.json").exists()      # 分词器

# 2. 模型加载测试
model = load_trained_model(model_dir)
assert model is not None

# 3. 基础推理测试
test_text = "我叫张三，手机号13800138000"
result = model.generate(test_text)
assert "entities" in result
```

**通过标准**：
- ✅ 所有必要文件存在
- ✅ 模型加载无报错
- ✅ 能生成有效的 JSON 输出

---

#### Step 2: 测试集评估（核心）

**目的**：在独立测试集上评估模型的准确性。

**数据集**：`data/merged_pii_dataset_test.jsonl`（~2.7MB，约 3,000-5,000 样本）

**评估脚本**：
```bash
python scripts/evaluate_trained_model.py \
    --model models/pii_detector_qwen3_0.6b/final \
    --test-data data/merged_pii_dataset_test.jsonl \
    --output evaluation_results/qwen3_0.6b_evaluation.json
```

**输出指标**：
```json
{
  "overall_metrics": {
    "precision": 0.88,
    "recall": 0.92,
    "f1_score": 0.90,
    "f2_score": 0.91,
    "accuracy": 0.89
  },
  "metrics_by_type": {
    "PERSON_NAME": {"precision": 0.92, "recall": 0.88, ...},
    "ADDRESS": {"precision": 0.85, "recall": 0.91, ...},
    ...
  },
  "confusion_matrix": {
    "true_positives": 2850,
    "false_positives": 320,
    "false_negatives": 180,
    "true_negatives": 4650
  }
}
```

**通过标准**：
- ✅ Overall F1-Score ≥ 87.5%
- ✅ Recall ≥ 90%
- ✅ 所有 PII 类型 F1 ≥ 80%

---

#### Step 3: 性能基准测试

**目的**：验证模型的推理性能是否满足业务需求。

**测试脚本**：
```bash
python scripts/benchmark_model.py \
    --model models/pii_detector_qwen3_0.6b/final \
    --num-samples 1000 \
    --batch-sizes 1,4,8 \
    --output evaluation_results/performance_benchmark.json
```

**测试维度**：

1. **延迟测试**（Batch Size = 1）
   ```python
   # 测试 1000 次推理
   latencies = []
   for text in test_texts:
       start = time.time()
       result = model.detect(text)
       latency = time.time() - start
       latencies.append(latency)

   print(f"P50: {np.percentile(latencies, 50) * 1000:.2f} ms")
   print(f"P95: {np.percentile(latencies, 95) * 1000:.2f} ms")
   ```

2. **吞吐量测试**（Batch Size = 8）
   ```python
   start = time.time()
   results = model.batch_detect(test_texts)  # 1000 个样本
   elapsed = time.time() - start
   throughput = len(test_texts) / elapsed
   print(f"吞吐量: {throughput:.2f} req/s")
   ```

3. **显存监控**
   ```bash
   # 推理前
   nvidia-smi --query-gpu=memory.used --format=csv
   # 推理中
   watch -n 1 nvidia-smi
   # 记录峰值显存
   ```

**通过标准**：
- ✅ P50 延迟 ≤ 300ms
- ✅ P95 延迟 ≤ 800ms
- ✅ 吞吐量 ≥ 10 req/s
- ✅ 显存占用 ≤ 3GB（FP16）

---

#### Step 4: 鲁棒性验证

**目的**：测试模型在边界场景和异常输入下的表现。

**测试用例集**：`data/robustness_test_cases.yaml`

```yaml
# 示例用例
test_cases:
  - id: robustness_001
    category: typo
    text: "我叫章三，手机号138-0013-8000"  # 错别字 + 格式变体
    expected_entities:
      - type: PERSON_NAME
        value: "章三"
      - type: CHINA_PHONE
        value: "138-0013-8000"

  - id: robustness_002
    category: ambiguity
    text: "我在长城工作，住在长城附近"  # 公司名 vs 地名
    expected_entities:
      - type: ORGANIZATION
        value: "长城"
        context: "工作"
      - type: ADDRESS
        value: "长城"
        context: "住在"

  - id: robustness_003
    category: long_text
    text: "..." # 1000+ 字符的长文本
    expected_entities: [...]
```

**评估方式**：
```python
# 按类别统计通过率
pass_rate_by_category = {
    "typo": 0.85,           # 错别字容忍度
    "ambiguity": 0.78,      # 歧义处理能力
    "long_text": 0.92,      # 长文本处理
    "format_variant": 0.88  # 格式变体识别
}
```

**通过标准**：
- ✅ 总体通过率 ≥ 80%
- ✅ 每个类别通过率 ≥ 70%

---

#### Step 5: 对比基准（与 Regex）

**目的**：验证 LLM 相比正则引擎的优势和劣势。

**使用现有工具**：`examples/evaluate_llm_vs_regex.py`

**评估维度**：

1. **结构化 PII 检测对比**
   ```
   PII 类型          Regex F1    LLM F1     差异
   ─────────────────────────────────────────
   CHINA_ID_CARD     99.0%       96.5%      -2.5%
   CHINA_PHONE       99.0%       97.2%      -1.8%
   EMAIL             100%        98.5%      -1.5%
   CHINA_BANK_CARD   99.0%       95.8%      -3.2%
   ```
   **结论**: LLM 在结构化 PII 上略低于 Regex，但仍在可接受范围（-5% 以内）

2. **上下文 PII 检测对比**
   ```
   PII 类型          Regex F1    LLM F1     差异
   ─────────────────────────────────────────
   PERSON_NAME       35.2%       88.5%      +53.3%
   ADDRESS           28.7%       82.3%      +53.6%
   ORGANIZATION      22.5%       79.8%      +57.3%
   ```
   **结论**: LLM 在上下文 PII 上大幅领先 Regex（+50% 以上）

3. **性能对比**
   ```
   维度              Regex       LLM        比值
   ─────────────────────────────────────────
   平均延迟          2.5ms       320ms      128x
   吞吐量            400 req/s   12 req/s   0.03x
   ```
   **结论**: LLM 性能显著慢于 Regex（预期中）

**通过标准**：
- ✅ 上下文 PII 检测 F1 ≥ 80%
- ✅ 相比 Regex 提升 ≥ 50%
- ✅ 结构化 PII 检测 F1 ≥ 95%

---

#### Step 6: 生成验证报告

**目的**：汇总所有验证结果，给出综合判定。

**报告格式**：`evaluation_results/validation_report.md`

```markdown
# Qwen3 0.6B PII 模型验证报告

## 📊 验证概要

| 维度 | 得分 | 状态 |
|------|------|------|
| 准确性 | 89.5/100 | ✅ 通过 |
| 性能 | 85.0/100 | ✅ 通过 |
| 鲁棒性 | 82.3/100 | ✅ 通过 |
| 对比基准 | 92.0/100 | ✅ 优秀 |

**总体评分**: 87.2/100 ✅ **通过验证**

## 📈 详细指标

### 1. 准确性指标
- Precision: 88.2%
- Recall: 91.5%
- F1-Score: 89.8%
- F2-Score: 90.5%

### 2. 性能指标
- P50 延迟: 285ms
- P95 延迟: 720ms
- 吞吐量: 12.3 req/s
- 显存占用: 2.8GB

### 3. 鲁棒性指标
- 总体通过率: 82.3%
- 错别字处理: 85.2%
- 歧义处理: 78.5%

### 4. 对比基准
- 上下文 PII F1: 85.2%（Regex: 28.5%, +56.7%）
- 结构化 PII F1: 96.8%（Regex: 99.0%, -2.2%）

## 🎯 结论与建议

### ✅ 通过项
1. 模型在测试集上达到目标 F1-Score (89.8% > 87.5%)
2. 上下文 PII 检测大幅超越 Regex（+56.7%）
3. 推理延迟在可接受范围内（P50 < 300ms）
4. 鲁棒性测试通过率达标（82.3% > 80%）

### ⚠️ 改进建议
1. **精确率偏低**（88.2%）: 建议增加后处理规则，过滤明显误报
2. **结构化 PII 检测略低于 Regex**: 考虑混合模式（Regex 预筛选）
3. **P95 延迟偏高**（720ms）: 建议使用 4-bit 量化降低延迟

### 🚀 下一步行动
1. ✅ **推荐部署**: 模型已通过验证，可部署到测试环境
2. 🔄 继续训练（可选）: 如需更高精确率，可增加训练轮次
3. 📊 A/B 测试: 与 Regex 并行运行，收集真实场景反馈
```

---

## 4. 自动化验证工具

### 4.1 工具架构

```
scripts/
├── evaluate_trained_model.py       # 核心评估脚本
├── benchmark_model.py              # 性能基准测试
├── test_robustness.py              # 鲁棒性测试
├── compare_with_baseline.py        # 对比基准（复用 evaluate_llm_vs_regex.py）
└── generate_validation_report.py   # 生成验证报告
```

### 4.2 一键验证脚本

**创建**: `scripts/run_full_validation.sh`

```bash
#!/bin/bash
# 一键运行完整验证流程

set -e  # 遇到错误立即退出

MODEL_DIR="models/pii_detector_qwen3_0.6b/final"
TEST_DATA="data/merged_pii_dataset_test.jsonl"
OUTPUT_DIR="evaluation_results/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Qwen3 0.6B PII 模型完整验证流程"
echo "=========================================="
echo ""
echo "模型路径: $MODEL_DIR"
echo "测试数据: $TEST_DATA"
echo "输出目录: $OUTPUT_DIR"
echo ""

# Step 1: 模型加载与健康检查
echo "[1/6] 模型加载与健康检查..."
python scripts/health_check.py \
    --model "$MODEL_DIR" \
    > "$OUTPUT_DIR/01_health_check.log" 2>&1
echo "✓ 健康检查完成"

# Step 2: 测试集评估
echo "[2/6] 测试集评估（核心）..."
python scripts/evaluate_trained_model.py \
    --model "$MODEL_DIR" \
    --test-data "$TEST_DATA" \
    --output "$OUTPUT_DIR/02_test_evaluation.json" \
    > "$OUTPUT_DIR/02_test_evaluation.log" 2>&1
echo "✓ 测试集评估完成"

# Step 3: 性能基准测试
echo "[3/6] 性能基准测试..."
python scripts/benchmark_model.py \
    --model "$MODEL_DIR" \
    --num-samples 1000 \
    --batch-sizes 1,4,8 \
    --output "$OUTPUT_DIR/03_performance_benchmark.json" \
    > "$OUTPUT_DIR/03_performance_benchmark.log" 2>&1
echo "✓ 性能基准测试完成"

# Step 4: 鲁棒性验证
echo "[4/6] 鲁棒性验证..."
python scripts/test_robustness.py \
    --model "$MODEL_DIR" \
    --test-cases data/robustness_test_cases.yaml \
    --output "$OUTPUT_DIR/04_robustness_test.json" \
    > "$OUTPUT_DIR/04_robustness_test.log" 2>&1
echo "✓ 鲁棒性验证完成"

# Step 5: 对比基准（与 Regex）
echo "[5/6] 对比基准测试..."
python examples/evaluate_llm_vs_regex.py \
    --llm-model "$MODEL_DIR" \
    --test-data "$TEST_DATA" \
    --output "$OUTPUT_DIR/05_llm_vs_regex.json" \
    > "$OUTPUT_DIR/05_llm_vs_regex.log" 2>&1
echo "✓ 对比基准测试完成"

# Step 6: 生成验证报告
echo "[6/6] 生成验证报告..."
python scripts/generate_validation_report.py \
    --input-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/validation_report.md" \
    > "$OUTPUT_DIR/06_report_generation.log" 2>&1
echo "✓ 验证报告生成完成"

echo ""
echo "=========================================="
echo "✅ 验证流程完成！"
echo "=========================================="
echo ""
echo "查看验证报告:"
echo "  cat $OUTPUT_DIR/validation_report.md"
echo ""
echo "或在浏览器中查看:"
echo "  markdown-viewer $OUTPUT_DIR/validation_report.md"
```

**使用方式**：
```bash
# 训练完成后立即运行
bash scripts/run_full_validation.sh
```

### 4.3 核心脚本实现

#### 4.3.1 `evaluate_trained_model.py`

见下文【自动化验证工具实现】部分

#### 4.3.2 `benchmark_model.py`

关键功能：
- 测量不同 batch size 下的延迟
- 计算 P50/P95/P99 分位数
- 监控 GPU 显存占用
- 生成性能曲线图

#### 4.3.3 `generate_validation_report.py`

关键功能：
- 汇总所有 JSON 结果文件
- 计算综合得分
- 判定是否通过验证
- 生成 Markdown 报告

---

## 5. 验证基准与通过标准

### 5.1 核心通过标准

模型**必须同时满足**以下条件才能通过验证：

| 维度 | 指标 | 通过标准 | 权重 |
|------|------|---------|------|
| **准确性** | F1-Score | ≥ 87.5% | 40% |
| **准确性** | Recall | ≥ 90% | 30% |
| **性能** | P50 延迟 | ≤ 300ms | 15% |
| **鲁棒性** | 通过率 | ≥ 80% | 10% |
| **对比基准** | 上下文 PII F1 | ≥ 80% | 5% |

**综合得分计算**：
```python
score = (
    accuracy_score * 0.40 +
    recall_score * 0.30 +
    performance_score * 0.15 +
    robustness_score * 0.10 +
    baseline_score * 0.05
)

if score >= 85:
    status = "✅ 优秀（推荐部署）"
elif score >= 75:
    status = "✅ 通过（可部署）"
elif score >= 65:
    status = "⚠️ 基本达标（需改进后部署）"
else:
    status = "❌ 未通过（需重新训练）"
```

### 5.2 分级通过标准

#### 🏅 优秀（Score ≥ 85）
- F1-Score ≥ 90%
- Recall ≥ 92%
- P50 延迟 ≤ 250ms
- 鲁棒性 ≥ 85%
- 上下文 PII F1 ≥ 85%

**行动**：直接部署到生产环境

#### ✅ 通过（75 ≤ Score < 85）
- F1-Score ≥ 87.5%
- Recall ≥ 90%
- P50 延迟 ≤ 300ms
- 鲁棒性 ≥ 80%
- 上下文 PII F1 ≥ 80%

**行动**：部署到测试环境，收集反馈

#### ⚠️ 基本达标（65 ≤ Score < 75）
- F1-Score ≥ 85%
- Recall ≥ 88%
- P50 延迟 ≤ 400ms
- 鲁棒性 ≥ 75%

**行动**：分析问题，针对性改进后重新验证

#### ❌ 未通过（Score < 65）
- 关键指标严重不达标

**行动**：
1. 检查训练数据质量
2. 调整训练超参数
3. 增加训练轮次
4. 考虑使用更大模型（Qwen3-1.7B）

---

## 6. 持续改进方案

### 6.1 迭代优化流程

```
训练 → 验证 → 分析 → 改进 → 重新训练
  ↑                                   ↓
  └───────────────────────────────────┘
```

### 6.2 常见问题与解决方案

#### 问题 1: F1-Score 不达标

**分析**：
- 检查测试集分布是否与训练集一致
- 分析混淆矩阵，找出主要错误类型
- 查看 Bad Case（误报和漏报样本）

**解决方案**：
1. **增加训练数据**：针对性补充困难样本
2. **调整超参数**：增加训练轮次、调整学习率
3. **改进 Prompt**：优化指令模板
4. **后处理规则**：增加规则过滤明显错误

#### 问题 2: Recall 低（漏报严重）

**解决方案**：
1. 降低检测阈值（如果使用置信度过滤）
2. 增加负样本训练数据
3. 调整训练目标，增加 Recall 权重

#### 问题 3: Precision 低（误报严重）

**解决方案**：
1. 提高检测阈值
2. 增加正样本训练数据
3. 使用正则引擎预筛选，LLM 精炼

#### 问题 4: 推理延迟过高

**解决方案**：
1. **模型量化**：4-bit / 8-bit 量化
2. **优化推理引擎**：使用 vLLM / TensorRT
3. **批处理**：批量推理提高吞吐
4. **模型蒸馏**：训练更小的模型

#### 问题 5: 鲁棒性差

**解决方案**：
1. 数据增强：添加错别字、格式变体样本
2. 多样化训练数据：覆盖更多边界场景
3. 对抗训练：使用困难样本进行训练

### 6.3 A/B 测试方案

部署后进行 A/B 测试，对比 LLM 和 Regex：

```python
# 流量分配
if random.random() < 0.5:
    # 50% 流量使用 LLM
    result = llm_detector.detect(text)
    log_result(method="LLM", result=result, text=text)
else:
    # 50% 流量使用 Regex
    result = regex_detector.detect(text)
    log_result(method="Regex", result=result, text=text)
```

**监控指标**：
- 准确性：人工抽样标注
- 性能：延迟、吞吐量
- 用户反馈：误报投诉率

### 6.4 持续监控

部署后建立监控体系：

| 指标 | 监控方式 | 告警阈值 |
|------|---------|---------|
| **准确性** | 定期抽样评估 | F1 < 85% |
| **推理延迟** | 实时监控 | P95 > 1s |
| **错误率** | 日志分析 | 异常率 > 5% |
| **显存占用** | GPU 监控 | 占用 > 80% |

---

## 7. 验证清单（Checklist）

训练完成后，逐项确认：

### 7.1 训练阶段检查

- [ ] 训练 Loss 正常收敛（最终 Loss < 0.3）
- [ ] 验证集 Loss 不显著高于训练集（无过拟合）
- [ ] 训练日志无异常错误
- [ ] 模型文件完整保存（adapter_model.safetensors 存在）

### 7.2 验证阶段检查

#### 准确性
- [ ] F1-Score ≥ 87.5%
- [ ] Recall ≥ 90%
- [ ] Precision ≥ 85%
- [ ] 所有 PII 类型 F1 ≥ 80%

#### 性能
- [ ] P50 延迟 ≤ 300ms
- [ ] P95 延迟 ≤ 800ms
- [ ] 吞吐量 ≥ 10 req/s
- [ ] 显存占用 ≤ 3GB

#### 鲁棒性
- [ ] 错别字测试通过率 ≥ 85%
- [ ] 歧义处理通过率 ≥ 75%
- [ ] 长文本测试通过率 ≥ 90%

#### 对比基准
- [ ] 上下文 PII F1 ≥ 80%
- [ ] 相比 Regex 提升 ≥ 50%
- [ ] 结构化 PII F1 ≥ 95%

### 7.3 部署前检查

- [ ] 验证报告已生成
- [ ] 综合得分 ≥ 75
- [ ] 所有核心指标通过
- [ ] Bad Case 已分析
- [ ] 改进建议已记录

---

## 8. 总结

### 8.1 关键要点

1. **验证≠测试**：验证是全面评估模型是否满足业务需求，不仅是准确性
2. **多维度评估**：准确性、性能、鲁棒性、对比基准缺一不可
3. **自动化优先**：使用脚本实现一键验证，减少人工操作
4. **持续改进**：验证不是终点，而是迭代优化的起点

### 8.2 验证时间估算

| 步骤 | 预计耗时 | 说明 |
|------|---------|------|
| 模型加载 | 1-2分钟 | 首次加载较慢 |
| 测试集评估 | 10-20分钟 | 取决于测试集大小（~5000样本） |
| 性能基准测试 | 5-10分钟 | 1000次推理 |
| 鲁棒性测试 | 5-10分钟 | ~500个边界用例 |
| 对比基准测试 | 15-30分钟 | 并行运行 LLM 和 Regex |
| 生成报告 | 1分钟 | 自动汇总 |
| **总计** | **35-70分钟** | 可后台运行 |

### 8.3 快速参考命令

```bash
# 一键完整验证
bash scripts/run_full_validation.sh

# 单独运行测试集评估
python scripts/evaluate_trained_model.py \
    --model models/pii_detector_qwen3_0.6b/final \
    --test-data data/merged_pii_dataset_test.jsonl \
    --output evaluation_results/test_evaluation.json

# 单独运行性能测试
python scripts/benchmark_model.py \
    --model models/pii_detector_qwen3_0.6b/final \
    --num-samples 1000 \
    --output evaluation_results/benchmark.json

# 查看验证报告
cat evaluation_results/latest/validation_report.md
```

---

**文档版本**: v1.0
**最后更新**: 2025-10-14
**负责人**: AI 工程师团队
**状态**: ✅ 已批准，可执行
