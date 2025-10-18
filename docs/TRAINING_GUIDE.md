# 17种PII模型训练指南

本文档用于在其他设备上进行模型训练。

---

## 📋 目录

1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [训练执行](#训练执行)
4. [验证测试](#验证测试)
5. [常见问题](#常见问题)

---

## 🛠️ 环境准备

### 硬件要求

- **GPU**: NVIDIA GPU（推荐RTX 3060及以上，12GB+ VRAM）
- **内存**: 32GB+ RAM
- **存储**: 50GB+ 可用空间

### 软件依赖

```bash
# 1. 创建Python虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. 安装核心依赖
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.56.2
pip install unsloth[cu128-ampere-torch280]
pip install datasets accelerate peft bitsandbytes

# 3. 安装项目依赖
cd HPPE
pip install -e .
```

### 验证环境

```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
python -c "import torch; print('GPU数量:', torch.cuda.device_count())"
python -c "import torch; print('GPU名称:', torch.cuda.get_device_name(0))"
```

---

## 📊 数据准备

### 标准PII类型

项目支持**17种标准PII类型**（定义在 `src/hppe/models/pii_types.py`）：

| **Phase 1 (6种)** | **Phase 2 (11种)** |
|-------------------|-------------------|
| PERSON_NAME      | BANK_CARD         |
| PHONE_NUMBER     | PASSPORT          |
| EMAIL            | DRIVER_LICENSE    |
| ADDRESS          | VEHICLE_PLATE     |
| ORGANIZATION     | IP_ADDRESS        |
| ID_CARD          | MAC_ADDRESS       |
|                  | POSTAL_CODE       |
|                  | IMEI              |
|                  | VIN               |
|                  | TAX_ID            |
|                  | SOCIAL_SECURITY   |

### 数据格式

训练数据使用JSONL格式（每行一个JSON对象）：

```json
{
  "text": "我是张三，电话13812345678",
  "entities": [
    {
      "type": "PERSON_NAME",
      "value": "张三",
      "start": 2,
      "end": 4,
      "confidence": 1.0
    },
    {
      "type": "PHONE_NUMBER",
      "value": "13812345678",
      "start": 7,
      "end": 18,
      "confidence": 1.0
    }
  ],
  "metadata": {
    "context": "self-introduction",
    "language": "zh"
  }
}
```

### 生成训练数据

#### 方法1：使用数据模板（快速开始）

```bash
# 生成标准训练数据模板（36个样本）
python scripts/generate_training_data_template.py

# 输出：data/training/17pii_training_template.jsonl
```

#### 方法2：扩展真实数据（生产级）

基于现有的训练数据生成脚本扩展：

```bash
# 参考现有脚本
scripts/generate_11pii_training_data.py

# 建议生成数量：
# - Phase 1 (6种): 500-1000 样本/类型
# - Phase 2 (11种): 300-500 样本/类型
# 总计：约8,000-15,000 样本
```

**扩展样本示例**：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from hppe.models.pii_types import PIIType, ALL_17_TYPES

# 为每种类型生成多样化样本
for pii_type in ALL_17_TYPES:
    samples = generate_samples_for_type(pii_type, count=500)
    # 使用不同的上下文、表达方式、格式变化
    # 添加噪声、混合样本、边缘案例
```

### 数据验证

```bash
# 验证数据一致性
python scripts/validate_data_consistency.py

# 应输出：
# ✅ 所有文件验证通过！
```

---

## 🚀 训练执行

### 训练脚本

使用 Unsloth 框架进行高效训练：

```bash
# 单GPU训练（推荐）
CUDA_VISIBLE_DEVICES=0 python scripts/train_pii_detector_unsloth_fixed.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --data_path data/training/17pii_training_data.jsonl \
    --output_dir models/pii_qwen4b_17types_final \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 2048 \
    --gradient_accumulation_steps 4
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model_name` | 基础模型 | `Qwen/Qwen2.5-3B-Instruct` |
| `--num_epochs` | 训练轮数 | 3-5 |
| `--batch_size` | 批大小 | 4 (12GB VRAM) |
| `--learning_rate` | 学习率 | 2e-5 |
| `--lora_r` | LoRA秩 | 16 |
| `--lora_alpha` | LoRA alpha | 32 |
| `--max_seq_length` | 最大序列长度 | 2048 |
| `--gradient_accumulation_steps` | 梯度累积 | 4 |

### 训练时长预估

**基于历史数据**（RTX 3060 12GB）：

- **数据量**: 8,000 样本
- **训练配置**: 3 epochs, batch_size=4, gradient_accumulation=4
- **预计时长**: **20-24 小时**

**优化建议**：

- 使用更强GPU（RTX 4090/A100）可减少至 8-12 小时
- 减少epochs至2可减少30%时间
- 增加batch_size（需更大VRAM）可加速训练

### 监控训练进度

```bash
# 查看训练日志
tail -f logs/train_17pii_full_*.log

# 关键指标：
# - loss: 逐步下降（目标 < 0.1）
# - learning_rate: 逐步衰减
# - epoch: 当前训练轮数
# - samples/sec: 训练速度

# 监控GPU使用率
nvidia-smi -l 5  # 每5秒刷新
```

---

## ✅ 验证测试

### 快速功能测试

```bash
# GPU0快速测试（10个样本）
python examples/quick_test_17pii_gpu0.py

# 预期输出：
# 📊 测试结果: 10/10 通过
# ✅ 17种PII模型基本功能正常！
```

### 模型对比验证

```bash
# 6种 vs 17种模型对比
python scripts/compare_6vs17_models.py \
    --model-6pii "models/pii_qwen4b_unsloth/final" \
    --model-17pii "models/pii_qwen4b_17types_final/final" \
    --test-data "data/test_datasets/17pii_test_cases.jsonl" \
    --output "comparison_report.json"

# 预期结果：
# - Precision: 60%+ (目标 70%+)
# - Recall: 60%+ (目标 70%+)
# - F1-Score: 60%+ (目标 70%+)
```

### 性能基准测试

```bash
# 延迟、吞吐量、显存测试
pytest tests/benchmark/test_llm_performance.py -v

# 测试内容：
# - P50/P95/P99延迟
# - RPS吞吐量
# - GPU显存占用
```

---

## ❓ 常见问题

### Q1: CUDA Out of Memory

**症状**：`RuntimeError: CUDA out of memory`

**解决方案**：

```python
# 1. 减小batch_size
--batch_size 2  # 从4降至2

# 2. 增加梯度累积
--gradient_accumulation_steps 8  # 从4增至8

# 3. 减少序列长度
--max_seq_length 1024  # 从2048降至1024

# 4. 启用梯度检查点（更慢但省内存）
--gradient_checkpointing True
```

### Q2: 训练loss不下降

**症状**：loss在1.0以上震荡

**原因**：

1. 学习率过高/过低
2. 数据质量问题
3. 样本数量不足

**解决方案**：

```bash
# 1. 调整学习率
--learning_rate 1e-5  # 降低学习率

# 2. 检查数据质量
python scripts/validate_data_consistency.py

# 3. 增加训练数据
# 目标：每种类型 ≥ 300 样本
```

### Q3: 模型输出格式错误

**症状**：模型输出不是有效JSON

**原因**：Prompt设计或训练数据问题

**解决方案**：

```python
# 检查训练数据的prompt格式
# 确保每个样本都有正确的instruction format:

{
  "messages": [
    {
      "role": "system",
      "content": "你是PII检测专家..."
    },
    {
      "role": "user",
      "content": "文本：我是张三"
    },
    {
      "role": "assistant",
      "content": "{\"entities\": [{\"type\": \"PERSON_NAME\", ...}]}"
    }
  ]
}
```

### Q4: 推理速度慢

**症状**：单次推理 > 2秒

**优化方案**：

```python
# 1. 使用4-bit量化
engine = QwenFineTunedEngine(
    model_path="models/pii_qwen4b_17types_final/final",
    device="cuda",
    load_in_4bit=True  # 启用4-bit量化
)

# 2. 减少max_new_tokens
generate(..., max_new_tokens=512)  # 从1024降至512

# 3. 批量推理
texts = ["文本1", "文本2", ...]
results = engine.batch_detect(texts)
```

### Q5: 类型名称不一致

**症状**：模型输出 `LICENSE_PLATE` 而非 `VEHICLE_PLATE`

**解决方案**：

```python
# 1. 使用类型标准化函数
from hppe.models.pii_types import normalize_pii_type

predicted_type = "LICENSE_PLATE"
standard_type = normalize_pii_type(predicted_type)
# 输出: "VEHICLE_PLATE"

# 2. 重新训练使用标准类型
# 确保训练数据全部使用标准类型名称
```

---

## 📚 参考资源

### 项目文档

- **标准PII类型定义**: `src/hppe/models/pii_types.py`
- **Epic 2完成报告**: `EPIC_2_COMPLETION_REPORT.md`
- **验证报告**: `comparison_6vs17_report_*.json`

### 相关脚本

- **数据标准化**: `scripts/normalize_test_dataset.py`
- **数据生成**: `scripts/generate_training_data_template.py`
- **数据验证**: `scripts/validate_data_consistency.py`
- **模型对比**: `scripts/compare_6vs17_models.py`

### 训练日志示例

```
Epoch 1/3:  25%|███▎      | 500/2000 [12:30<37:30, 0.66it/s, loss=0.847]
Epoch 1/3:  50%|██████    | 1000/2000 [25:00<25:00, 0.67it/s, loss=0.523]
Epoch 1/3:  75%|████████▊ | 1500/2000 [37:30<12:30, 0.67it/s, loss=0.312]
Epoch 1/3: 100%|██████████| 2000/2000 [50:00<00:00, 0.67it/s, loss=0.185]

✅ Epoch 1 完成，平均loss: 0.441
```

---

## 🎯 训练检查清单

**开始训练前**：

- [ ] GPU环境验证通过
- [ ] 训练数据已生成（≥5,000样本）
- [ ] 数据一致性验证通过
- [ ] 磁盘空间充足（≥50GB）
- [ ] 训练参数已配置

**训练过程中**：

- [ ] 监控loss下降趋势
- [ ] 监控GPU利用率（>80%）
- [ ] 定期保存checkpoint
- [ ] 记录训练时长和性能

**训练完成后**：

- [ ] 快速功能测试通过
- [ ] 模型对比验证（F1 ≥ 60%）
- [ ] 保存最终模型
- [ ] 记录训练配置和结果

---

## 📧 支持与反馈

如遇到问题，请检查：

1. 查看本文档的 **常见问题** 章节
2. 查看项目的 `EPIC_2_COMPLETION_REPORT.md`
3. 检查训练日志：`logs/train_17pii_full_*.log`

---

**文档版本**: 1.0
**更新日期**: 2025-10-18
**适用模型**: Qwen2.5-3B-Instruct + LoRA (17种PII)
