# Unsloth 迁移方案

**目标**: 使用Unsloth重新训练,提升速度和效果

## 为什么选择Unsloth?

### 优势对比

| 特性 | 当前方案 | Unsloth | 提升 |
|-----|---------|---------|------|
| **训练速度** | 2.5s/step | 0.8-1.2s/step | 2-3倍 ⚡ |
| **显存占用** | 9.8GB | 5-6GB | 节省40% |
| **Batch Size** | 4 | 8-12 | 2-3倍 |
| **总训练时间** | 3.8小时 | 1.5-2小时 | 节省50% |
| **支持量化** | FP16 | 4bit/8bit | 更省资源 |
| **Qwen支持** | ✅ | ✅ 专门优化 | 更稳定 |

### 核心优化技术

1. **Flash Attention 2** - 自动启用,加速注意力计算
2. **Gradient Checkpointing** - 自动优化,不需要手动配置
3. **Fused Kernels** - CUDA级别优化
4. **4bit LoRA** - 可选,进一步减少显存

## 安装步骤

```bash
# 安装Unsloth (支持CUDA 12.1+)
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# 或使用conda安装 (推荐)
conda install -c conda-forge unsloth
```

## 迁移代码

### 方案A: 最小修改 (推荐,30分钟)

创建新的训练脚本 `scripts/train_pii_detector_unsloth.py`:

```python
#!/usr/bin/env python3
"""使用Unsloth训练PII检测模型"""

import json
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# 1. 加载模型 (Unsloth自动优化)
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-0.6B",  # 或从checkpoint继续
    max_seq_length = max_seq_length,
    dtype = None,  # 自动选择FP16/BF16
    load_in_4bit = False,  # 设为True可进一步省显存
)

# 2. 配置LoRA (Unsloth优化版)
model = FastLanguageModel.get_peft_model(
    model,
    r = 12,  # 激进配置
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # Unsloth建议加上MLP层
    ],
    lora_alpha = 24,
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Unsloth优化版!
    random_state = 42,
)

# 3. 加载数据
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            # 转换为text格式
            text = sample["input"]
            entities = sample.get("output", {}).get("entities", [])

            prompt = (
                f"<|im_start|>system\n"
                f"你是 PII 检测专家。检测以下文本中的 PII，并以 JSON 格式输出实体列表。<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f'{{"entities": {json.dumps(entities, ensure_ascii=False)}}}<|im_end|>'
            )
            data.append({"text": prompt})
    return Dataset.from_list(data)

train_dataset = load_jsonl("data/merged_pii_dataset_train.jsonl")
eval_dataset = load_jsonl("data/merged_pii_dataset_validation.jsonl")

# 4. 训练配置
training_args = TrainingArguments(
    output_dir = "models/pii_qwen_unsloth",
    per_device_train_batch_size = 8,  # Unsloth可以用更大batch!
    gradient_accumulation_steps = 4,  # effective_batch_size=32
    warmup_steps = 100,
    num_train_epochs = 3,
    learning_rate = 1.5e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 10,
    optim = "adamw_8bit",  # Unsloth推荐
    weight_decay = 0.01,
    lr_scheduler_type = "cosine",  # 比linear更稳定
    seed = 42,
    save_strategy = "epoch",
    eval_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
)

# 5. 创建Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_args,
)

# 6. 训练
trainer.train()

# 7. 保存
model.save_pretrained("models/pii_qwen_unsloth/final")
tokenizer.save_pretrained("models/pii_qwen_unsloth/final")

print("✅ 训练完成!")
```

### 方案B: 从已有checkpoint继续 (复用1 epoch)

修改上述代码第1步:

```python
# 从已训练的模型继续
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "models/pii_detector_qwen3_06b_aggressive/final",  # 复用!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = False,
)
```

## 预期效果对比

### 训练时间

| 配置 | 当前方案 | Unsloth | 节省 |
|-----|---------|---------|------|
| **1 epoch** | 1.9小时 | 0.6-0.8小时 | 60% |
| **3 epochs** | 5.7小时 | 2-2.5小时 | 60% |

### 性能提升

**Unsloth的额外优势**:

1. **更大batch size (8 vs 4)**:
   - 更稳定的梯度 → 更好收敛
   - **预期Recall提升: +2-3%**

2. **Cosine学习率调度**:
   - 避免后期震荡
   - **预期F1提升: +1-2%**

3. **8bit优化器**:
   - 节省显存同时保持精度
   - 可以尝试更大的LoRA rank

**总体预期**:
- **当前**: Precision 94%, Recall 74%, F1 83.5%
- **Unsloth**: Precision 93-95%, Recall **77-80%**, F1 **85-87%**

**是否达标**: 可能仍差3-5%,但显著改善

## 进一步优化选项

如果Unsloth单独使用仍不够,可以组合:

### 选项1: Unsloth + 数据增强

```python
# 在load_jsonl中添加过采样
if sample.get("output", {}).get("entities"):
    data.append({"text": prompt})
    data.append({"text": prompt})  # PII样本重复2次
    data.append({"text": prompt})  # 重复3次
else:
    data.append({"text": prompt})
```

**预期**: Recall 77% → **82-85%**

### 选项2: Unsloth + 更大模型 (Qwen3-1.5B)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-1.5B",  # 2.5倍参数
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,  # 4bit量化以节省显存
)
```

**Unsloth优势**: 即使用4bit量化,精度损失也很小
**预期**: Recall 77% → **85-90%**, **可能达标**!

### 选项3: Unsloth + 更大batch + 数据增强 + Qwen3-1.5B (全力方案)

**配置**:
```python
model_name = "Qwen/Qwen3-1.5B"
load_in_4bit = True
per_device_train_batch_size = 4  # 4bit后可以用
lora_r = 16  # 更大rank
+ 数据过采样3倍
```

**预期**: Recall **88-92%**, F1 **90-93%**, **大概率达标**!
**时间**: 约3-4小时 (比当前方案快)

## 推荐行动

### 立即执行

1. **停止当前训练** (已跑1.5小时,节省剩余2.3小时)
2. **安装Unsloth** (5分钟)
3. **选择方案**:

**保守方案** (推荐新手):
- Unsloth + Qwen3-0.6B + 当前数据
- 时间: 2小时
- 预期: Recall 77-80%
- 风险: 可能仍不达标

**激进方案** (推荐你):
- Unsloth + Qwen3-1.5B (4bit) + 数据过采样
- 时间: 3-4小时
- 预期: Recall 85-90%
- 风险: 低,大概率达标

**你的情况**: 12GB显存,完全够用!

## 实施清单

- [ ] 停止当前训练 (kill PID 59373)
- [ ] 安装Unsloth
- [ ] 创建训练脚本 `scripts/train_pii_detector_unsloth.py`
- [ ] 决定是否数据过采样
- [ ] 决定使用0.6B还是1.5B模型
- [ ] 启动训练
- [ ] 每1 epoch验证一次

要不要我帮你立即开始?
