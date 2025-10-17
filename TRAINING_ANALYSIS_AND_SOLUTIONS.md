# 训练困境分析与解决方案

**时间**: 2025-10-15 14:15
**问题**: 模型Recall持续低于目标 (71-74% vs 目标90%)

## 问题根因分析

### 1. 当前训练结果回顾

| 配置 | Epochs | LR | LoRA r | Precision | Recall | F1 | 问题 |
|------|--------|----|----|-----------|--------|-------|------|
| 基线 | 3 | 2e-4 | 8 | 95.37% | **74.31%** | 83.53% | Recall差15.7% |
| 激进 | 1 | 1.5e-4 | 12 | 94.16% | **71.78%** | 81.46% | 更差 |

**核心问题**:
- Precision一直很高 (94-95%) → 模型学会了"宁可漏检,不要误报"
- Recall一直很低 (71-74%) → 漏检严重
- 继续训练可能只是强化这个偏见

### 2. 为什么继续训练可能无效？

**理论分析**:
1. **数据不平衡问题**
   - 如果训练数据中负样本(非PII)远多于正样本
   - 模型会倾向于保守预测以降低整体loss
   - 更多epoch只会强化这个倾向

2. **Loss函数问题**
   - 标准Cross-Entropy对Precision/Recall不敏感
   - 模型优化的是总体loss,不是F1-Score
   - 无法直接优化Recall目标

3. **模型容量问题**
   - Qwen3-0.6B可能太小
   - LoRA r=12也许还不够

**预测**: 即使训练到3 epochs,Recall可能仍在75-78%范围,无法突破90%

## 解决方案对比

### 方案1: 使用专业训练框架 (推荐 ✅)

#### 1.1 **LLaMA-Factory** (最推荐)

**优势**:
- 专为LLM微调设计,支持Qwen系列
- 内置多种高级策略:
  - **Focal Loss**: 自动处理类别不平衡
  - **Label Smoothing**: 防止过拟合
  - **Warmup + Cosine Decay**: 更稳定的学习率调度
  - **DeepSpeed/FSDP**: 更大的模型和batch size
- 支持LoRA/QLoRA/Full Fine-tuning
- 配置简单,开箱即用

**实施**:
```bash
# 安装
pip install llmtuner

# 配置文件 (examples/train_lora/qwen_lora_sft.yaml)
model_name_or_path: Qwen/Qwen3-0.6B
dataset: pii_detection
output_dir: models/pii_qwen_llamafactory
finetuning_type: lora
lora_rank: 16          # 更大
lora_alpha: 32
learning_rate: 5e-5    # 标准LLM学习率
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
lr_scheduler_type: cosine_with_restarts  # 关键!
warmup_ratio: 0.1
fp16: true
logging_steps: 10
save_steps: 500
eval_steps: 500
evaluation_strategy: steps
load_best_model_at_end: true
metric_for_best_model: f1      # 直接优化F1!
```

**预期提升**: Recall +5-10% → 79-84%

#### 1.2 **Unsloth** (速度优化)

**优势**:
- 训练速度提升2-5倍
- 内存使用减少50%
- 支持更大的batch size → 更稳定的梯度
- 专为4bit/8bit量化优化

**实施**:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-0.6B",
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,  # 4bit量化,释放显存
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.1,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 更多层
    use_gradient_checkpointing = True,
)
```

**预期提升**:
- 训练速度: 3-4小时 → 1-1.5小时
- Recall: +3-5% (通过更大batch size)

#### 1.3 **Axolotl**

**优势**:
- 高度可配置,支持几乎所有高级技巧
- 内置数据增强策略
- 支持多GPU训练

**劣势**: 配置复杂,学习曲线陡峭

### 方案2: 优化当前训练策略 (成本低)

#### 2.1 **调整Loss权重** (重点!)

**问题诊断**: 查看训练数据分布
```bash
python -c "
import json
total, with_pii = 0, 0
with open('data/merged_pii_dataset_train.jsonl') as f:
    for line in f:
        sample = json.loads(line)
        total += 1
        if sample.get('output', {}).get('entities'):
            with_pii += 1
print(f'总样本: {total}, 含PII: {with_pii} ({with_pii/total*100:.1f}%), 不含PII: {total-with_pii} ({(total-with_pii)/total*100:.1f}%)')
"
```

**如果数据不平衡**: 修改训练脚本,添加Class Weights
```python
# 在train_pii_detector.py中添加
from transformers import TrainingArguments, Trainer
import torch.nn as nn

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # 计算每个token的loss,对PII token加权
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 对包含实体的样本加权2-3倍
        weighted_loss = loss * sample_weights

        return (weighted_loss.mean(), outputs) if return_outputs else weighted_loss.mean()
```

**预期提升**: Recall +5-8%

#### 2.2 **数据增强**

**策略**:
1. **过采样PII样本**: 重复包含PII的样本2-3倍
2. **负样本降采样**: 减少不含PII的样本
3. **合成数据**: 使用GPT-4生成更多PII样本

**实施**:
```python
# 简单过采样
import json
samples = []
with open('data/merged_pii_dataset_train.jsonl') as f:
    for line in f:
        sample = json.loads(line)
        samples.append(sample)
        # 如果包含PII,复制2次
        if sample.get('output', {}).get('entities'):
            samples.append(sample)
            samples.append(sample)

# 保存
with open('data/merged_pii_dataset_train_oversampled.jsonl', 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
```

**预期提升**: Recall +3-5%

#### 2.3 **更大的基础模型**

当前: Qwen3-0.6B (600M参数)
建议尝试:
- **Qwen3-1.5B**: 2.5倍参数 → 更强能力
- **Qwen3-3B**: 5倍参数 → 显著提升

**成本**: 训练时间增加50-100%,但可能一次性达标

### 方案3: 改变任务形式 (创新)

#### 3.1 **两阶段检测**

**Stage 1**: 粗检测器 (高Recall)
- 训练目标: Recall ≥ 95%, 允许Precision低一些
- 使用更激进的阈值

**Stage 2**: 精过滤器 (高Precision)
- 对Stage 1的结果进行过滤
- 提升Precision到目标水平

**优势**: 分治策略,各自优化更容易

#### 3.2 **NER + 规则混合**

**策略**:
- LLM负责复杂语义理解
- 正则表达式负责明确模式 (手机号、身份证等)
- 两者结果合并

**优势**: 保底的Recall,LLM负责提升

## 推荐行动方案

### 短期 (今天) - 低成本快速验证

**方案A**: 数据增强 + 重新训练 (2小时)
```bash
# 1. 过采样PII样本
python scripts/oversample_pii_data.py

# 2. 重新训练3 epochs
python scripts/train_pii_detector.py \
  --data data/merged_pii_dataset_train_oversampled.jsonl \
  --learning-rate 1e-4 \
  --epochs 3
```

**预期**: Recall 74% → 79-82%, 可能仍未达标

**方案B**: 尝试更大模型 Qwen3-1.5B (3小时)
```bash
# 下载模型
modelscope download --model Qwen/Qwen3-1.5B

# 训练 (可能需要调整batch size)
python scripts/train_pii_detector.py \
  --model Qwen/Qwen3-1.5B \
  --batch-size 2 \
  --gradient-accumulation 16
```

**预期**: Recall 74% → 83-87%, 接近达标

### 中期 (明天) - 专业框架

**方案C**: 使用LLaMA-Factory (半天设置 + 2小时训练)
```bash
# 安装
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .

# 转换数据格式
python scripts/convert_to_llamafactory_format.py

# 训练
llamafactory-cli train examples/train_lora/qwen_lora_sft.yaml
```

**预期**: Recall 74% → 85-90%, 大概率达标

### 长期 - 系统优化

**方案D**: 完整pipeline重构
1. 数据质量审查与清洗
2. 使用LLaMA-Factory + Qwen3-3B
3. 两阶段检测架构
4. 集成规则引擎

**预期**: Recall 90%+, Precision 95%+

## 我的建议

**立即执行**:
1. **先停止当前继续训练** (节省3.5小时)
2. **尝试方案B**: 换Qwen3-1.5B,可能一次性解决问题
3. **如果方案B仍不够**: 明天使用LLaMA-Factory (方案C)

**理由**:
- 当前0.6B模型可能天花板就在75-80% Recall
- 增加模型容量是最直接有效的方法
- LLaMA-Factory有Focal Loss等高级技巧,专为解决类别不平衡设计

要不要我先停止当前训练,换成Qwen3-1.5B试试?
