# PII 检测模型 SFT 训练指南

**目标：** 训练一个专门用于 PII 检测和脱敏的 LLM 模型

**场景：** 云端大模型数据脱敏 + 批处理优化（70% 批处理 + 30% 实时）

**准确率目标：** 99%（隐私保护关键）

---

## 📋 完整流程

### 阶段 1：数据准备

#### 1.1 下载现成数据集

```bash
# 安装依赖
pip install datasets huggingface_hub

# 下载推荐数据集（MSRA中文 + ai4privacy英文）
python scripts/download_datasets.py --all --output data/pii_datasets

# 或单独下载
python scripts/download_datasets.py --datasets msra ai4privacy --output data/pii_datasets

# 查看可用数据集
python scripts/download_datasets.py --list
```

**数据集详情：**
- **MSRA NER** (中文) - 50,000+ 样本
  - 人民日报新闻文本
  - 人名、地名、组织
- **ai4privacy** (英文为主) - 200,000 样本
  - 包含脱敏示例（非常适合脱敏场景）
  - 多种PII类型

#### 1.2 生成合成中文数据

```bash
# 安装依赖
pip install faker

# 生成 30,000 条中文合成数据
python scripts/generate_synthetic_pii.py \
    --num-samples 30000 \
    --language zh_CN \
    --output data/pii_datasets/synthetic_pii.jsonl

# 查看生成的数据
head -n 5 data/pii_datasets/synthetic_pii.jsonl | jq
```

**生成的数据类型：**
- 中文姓名（PERSON_NAME）
- 中国手机号（PHONE_NUMBER）
- 中文邮箱（EMAIL）
- 中国身份证（ID_CARD）
- 中文地址（ADDRESS）
- 中文组织（ORGANIZATION）

#### 1.3 合并数据集

```bash
# 使用推荐配置合并（30% MSRA + 30% ai4privacy + 40% 合成）
python scripts/merge_datasets.py \
    --all \
    --total-samples 50000 \
    --output data/merged_pii_dataset.jsonl

# 查看合并结果
ls -lh data/merged_pii_dataset_*.jsonl
```

**输出文件：**
- `merged_pii_dataset_train.jsonl` - 训练集 (40,000 样本, 80%)
- `merged_pii_dataset_validation.jsonl` - 验证集 (5,000 样本, 10%)
- `merged_pii_dataset_test.jsonl` - 测试集 (5,000 样本, 10%)

**数据格式（SFT格式）：**
```json
{
  "instruction": "检测以下文本中的 PII，并以 JSON 格式输出实体列表。",
  "input": "我叫张三，电话13800138000，在北京科技有限公司工作。",
  "output": {
    "entities": [
      {"type": "PERSON_NAME", "value": "张三", "start": 2, "end": 4},
      {"type": "PHONE_NUMBER", "value": "13800138000", "start": 6, "end": 17},
      {"type": "ORGANIZATION", "value": "北京科技有限公司", "start": 19, "end": 28}
    ]
  }
}
```

---

### 阶段 2：模型训练

#### 2.1 选择基础模型

**推荐模型（按显存要求）：**

| 模型 | 参数量 | 显存需求 | 推荐场景 |
|-----|--------|---------|---------|
| Qwen2-0.5B | 0.5B | ~4GB | 快速测试 |
| **Qwen2-1.5B** ⭐ | 1.5B | ~8GB | **推荐（平衡性能和速度）** |
| Qwen2-7B | 7B | ~20GB | 高精度需求 |
| Qwen3-8B-AWQ | 8B (4bit) | ~12GB | 当前使用的模型 |

**您的硬件：** 2x RTX 3060 (12GB each) = 24GB 总显存

#### 2.2 开始训练

**方案 A：快速测试（使用小样本）**
```bash
# 快速验证流程（5分钟完成）
python scripts/train_pii_detector.py \
    --model Qwen/Qwen2-1.5B \
    --data data/merged_pii_dataset_train.jsonl \
    --val-data data/merged_pii_dataset_validation.jsonl \
    --max-samples 1000 \
    --batch-size 8 \
    --epochs 1 \
    --output models/pii_detector_test
```

**方案 B：生产训练（推荐配置）**
```bash
# 完整训练（2-3小时）
python scripts/train_pii_detector.py \
    --model Qwen/Qwen2-1.5B \
    --data data/merged_pii_dataset_train.jsonl \
    --val-data data/merged_pii_dataset_validation.jsonl \
    --lora-r 8 \
    --lora-alpha 16 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --output models/pii_detector_qwen2_1.5b
```

**方案 C：高精度训练（更大LoRA rank）**
```bash
# 使用更大的LoRA配置（4-5小时）
python scripts/train_pii_detector.py \
    --model Qwen/Qwen2-1.5B \
    --data data/merged_pii_dataset_train.jsonl \
    --val-data data/merged_pii_dataset_validation.jsonl \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 4 \
    --gradient-accumulation 8 \
    --learning-rate 1e-4 \
    --epochs 5 \
    --output models/pii_detector_high_precision
```

#### 2.3 训练参数说明

**LoRA 参数：**
- `--lora-r`: LoRA rank（越大越精确，但训练越慢）
  - 推荐：8-16
  - 高精度：16-32
- `--lora-alpha`: LoRA alpha（通常是 r 的 2 倍）
  - 推荐：16-32
  - 高精度：32-64

**训练超参数：**
- `--batch-size`: 每个GPU的批次大小
  - RTX 3060 12GB: 推荐 4-8
- `--gradient-accumulation`: 梯度累积步数
  - 有效批次大小 = batch-size × gradient-accumulation × GPU数量
  - 推荐：4-8
- `--learning-rate`: 学习率
  - LoRA微调：1e-4 ~ 5e-4
  - 推荐：2e-4
- `--epochs`: 训练轮数
  - 小数据集（<10k）：5-10
  - 大数据集（50k+）：3-5

---

### 阶段 3：模型评估

#### 3.1 测试模型性能

```bash
# 创建测试脚本
cat > examples/test_trained_model.py << 'EOF'
import sys
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def test_pii_detection(model_path, test_cases):
    """测试训练好的模型"""
    print(f"加载模型: {model_path}")

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("\n测试样例：\n")

    for i, test_text in enumerate(test_cases, 1):
        print(f"[{i}] 输入: {test_text}")

        # 构建提示
        prompt = (
            f"<|im_start|>system\n"
            f"你是 PII 检测专家。检测文本中的 PII 并输出 JSON。<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{test_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.05,
            top_p=0.9,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"    输出: {response}\n")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/pii_detector_qwen2_1.5b/final"

    test_cases = [
        "我叫张三，电话13800138000。",
        "联系人：李四，邮箱lisi@example.com，在北京科技有限公司工作。",
        "身份证号：110101199003077578，住址：北京市朝阳区建国路1号。"
    ]

    test_pii_detection(model_path, test_cases)
EOF

# 运行测试
python examples/test_trained_model.py models/pii_detector_qwen2_1.5b/final
```

#### 3.2 在测试集上评估

```bash
# 创建评估脚本
python examples/evaluate_on_testset.py \
    --model models/pii_detector_qwen2_1.5b/final \
    --test-data data/merged_pii_dataset_test.jsonl \
    --output evaluation_results/sft_model_eval.json
```

**评估指标：**
- **精确率 (Precision)**: 检测到的PII中有多少是正确的
- **召回率 (Recall)**: 所有PII中有多少被检测到
- **F1 分数**: 精确率和召回率的调和平均
- **实体级准确率**: 每种PII类型的准确率

**目标：**
- 总体 F1 > 0.99
- 关键PII类型（姓名、电话、身份证）F1 > 0.995

---

### 阶段 4：模型部署

#### 4.1 集成到 vLLM 服务

```bash
# 停止当前 vLLM 服务
pkill -f vllm

# 启动使用训练模型的 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model models/pii_detector_qwen2_1.5b/final \
    --dtype auto \
    --api-key token-abc123 \
    --served-model-name pii-detector \
    --max-model-len 2048 \
    --enable-lora \
    --gpu-memory-utilization 0.9
```

#### 4.2 更新 HPPE 配置

```python
# 修改 configs/llm_config.yaml
llm_engine:
  model_name: "pii-detector"  # 使用训练好的模型
  base_url: "http://localhost:8000/v1"
  api_key: "token-abc123"
  timeout: 30
```

#### 4.3 测试集成

```bash
# 运行批量检测测试
python examples/test_batch_detector.py

# 测试脱敏流程
python examples/test_anonymization.py
```

---

## 🎯 优化建议

### 针对批处理场景（70%）

**1. 增加批处理样本**
```bash
# 生成批处理特定的训练数据
python scripts/generate_synthetic_pii.py \
    --num-samples 20000 \
    --output data/batch_processing_samples.jsonl

# 每个样本包含多个PII（模拟批处理场景）
```

**2. 数据增强**
```python
# 添加更多上下文变化
templates = [
    "批量处理：{name1}、{name2}、{name3}",
    "联系人列表：\n1. {name1} {phone1}\n2. {name2} {phone2}",
    "导出数据：{name} | {phone} | {email} | {address}"
]
```

### 针对脱敏场景

**1. 添加脱敏样本**
```json
{
  "instruction": "将文本中的 PII 替换为占位符",
  "input": "我叫张三，电话13800138000。",
  "output": {
    "anonymized": "我叫[PERSON_1]，电话[PHONE_1]。",
    "mapping": {
      "PERSON_1": "张三",
      "PHONE_1": "13800138000"
    }
  }
}
```

**2. 训练脱敏模型**
```bash
# 使用包含脱敏样本的数据集训练
python scripts/train_pii_detector.py \
    --model Qwen/Qwen2-1.5B \
    --data data/anonymization_dataset.jsonl \
    --output models/pii_anonymizer
```

---

## 📊 预期效果

### 性能指标

**检测准确率：**
- 训练前（Qwen3-8B zero-shot）：~85-90%
- 训练后（Qwen2-1.5B SFT）：**95-99%**

**推理速度：**
- 训练前（8B模型）：8-12秒/样本
- 训练后（1.5B模型）：**2-4秒/样本** (3-6倍提速)

**显存占用：**
- 训练前：12GB (8B AWQ)
- 训练后：**6-8GB** (1.5B FP16)

### 成本效益

**训练成本：**
- 数据准备：1-2小时
- 模型训练：2-3小时
- 总计：**3-5小时**

**推理成本（批处理10万样本）：**
- 训练前：~20小时
- 训练后：**~5小时** (75%时间节省)

---

## 🚀 快速开始命令

完整流程一键执行：

```bash
#!/bin/bash
# 完整 SFT 训练流程

# 1. 安装依赖
pip install datasets huggingface_hub faker transformers peft accelerate

# 2. 准备数据
python scripts/download_datasets.py --all --output data/pii_datasets
python scripts/generate_synthetic_pii.py --num-samples 30000 --output data/pii_datasets/synthetic_pii.jsonl
python scripts/merge_datasets.py --all --total-samples 50000 --output data/merged_pii_dataset.jsonl

# 3. 训练模型
python scripts/train_pii_detector.py \
    --model Qwen/Qwen2-1.5B \
    --data data/merged_pii_dataset_train.jsonl \
    --val-data data/merged_pii_dataset_validation.jsonl \
    --lora-r 8 \
    --batch-size 4 \
    --epochs 3 \
    --output models/pii_detector_qwen2_1.5b

# 4. 测试模型
python examples/test_trained_model.py models/pii_detector_qwen2_1.5b/final

echo "✓ SFT 训练完成！"
```

保存为 `scripts/run_full_sft_pipeline.sh` 并执行：
```bash
chmod +x scripts/run_full_sft_pipeline.sh
./scripts/run_full_sft_pipeline.sh
```

---

## 📚 相关文档

- **数据集资源**: `docs/pii_datasets_resources.md`
- **架构设计**: `docs/privacy_gateway_architecture.md`
- **LLM集成总结**: `docs/llm_integration_summary.md`
- **性能评估**: `evaluation_results/llm_vs_regex_summary.md`

---

## ❓ 常见问题

### Q1: 训练显存不足怎么办？

**解决方案：**
1. 减小 batch size: `--batch-size 2`
2. 增加梯度累积: `--gradient-accumulation 8`
3. 使用更小的模型: `Qwen/Qwen2-0.5B`
4. 减小 LoRA rank: `--lora-r 4`

### Q2: 训练太慢怎么办？

**解决方案：**
1. 使用更少的样本: `--max-samples 10000`
2. 减少训练轮数: `--epochs 2`
3. 增大批次大小: `--batch-size 8`
4. 使用多GPU训练

### Q3: 如何提高准确率？

**解决方案：**
1. 增加训练数据量（50k → 100k）
2. 增加训练轮数（3 → 5）
3. 使用更大的 LoRA rank（8 → 16）
4. 添加更多领域数据
5. 使用更大的基础模型（1.5B → 7B）

### Q4: 如何支持新的PII类型？

**解决方案：**
1. 在合成数据生成器中添加新类型
2. 收集或标注包含新类型的真实数据
3. 重新训练模型
4. 在 `ENTITY_TYPE_MAPPING` 中添加映射

---

**更新日期：** 2025-10-14
**作者：** HPPE 开发团队
