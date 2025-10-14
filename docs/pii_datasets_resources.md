# PII 检测数据集资源清单

**用途：** SFT（监督微调）LLM 进行 PII 检测
**更新日期：** 2025-10-14

---

## 🌟 推荐数据集（优先使用）

### 1. ai4privacy/pii-masking-200k ⭐⭐⭐⭐⭐

**来源：** Hugging Face
**链接：** https://huggingface.co/datasets/ai4privacy/pii-masking-200k

**规模：**
- 43k 版本：43,000 条
- 200k 版本：200,000 条
- 300k 版本：300,000 条

**格式：**
```json
{
  "source_text": "My name is [NAME] and I live in [CITY]",
  "target_text": "My name is John Smith and I live in New York",
  "privacy_mask": {
    "[NAME]": "John Smith",
    "[CITY]": "New York"
  }
}
```

**PII 类型：**
- 姓名（NAME）
- 地址（ADDRESS）
- 电话（PHONE）
- 邮箱（EMAIL）
- 等

**优点：**
- ✅ 数据量大（200k-300k）
- ✅ 多语言支持（包含英文）
- ✅ 格式标准（适合 SFT）
- ✅ 质量高（人工验证）

**缺点：**
- ❌ 主要是英文，中文样本较少

**推荐用法：**
```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("ai4privacy/pii-masking-200k")

# 转换为 SFT 格式
def convert_to_sft_format(example):
    return {
        "instruction": f"检测以下文本中的 PII 并标注：{example['source_text']}",
        "output": example['privacy_mask']
    }

sft_dataset = dataset.map(convert_to_sft_format)
```

---

### 2. BigCode PII Dataset ⭐⭐⭐⭐

**来源：** Hugging Face
**链接：** https://huggingface.co/datasets/bigcode/bigcode-pii-dataset

**规模：** 12,099 样本

**特点：**
- 专注于代码中的 PII
- 支持 31 种编程语言
- 每个样本约 50 行代码

**PII 类型：**
- 用户名（Username）
- 邮箱（Email）
- IP 地址（IP Address）
- API 密钥（Keys）
- 密码（Passwords）
- ID

**标注格式：**
```python
{
  "code": "import requests\nAPI_KEY = 'sk-1234...'",
  "entities": [
    {
      "type": "KEY",
      "value": "sk-1234...",
      "start": 28,
      "end": 40
    }
  ]
}
```

**优点：**
- ✅ 专注代码场景
- ✅ 多语言代码
- ✅ 适合技术文档脱敏

**推荐用法：**
- 如果你的数据包含代码片段，这个数据集很有用
- 可以用于训练代码中的 PII 检测

---

### 3. MSRA NER Dataset ⭐⭐⭐⭐⭐（中文）

**来源：** Microsoft Research Asia
**链接：**
- Hugging Face: https://huggingface.co/datasets/levow/msra_ner
- 官方下载: https://www.microsoft.com/download/details.aspx?id=52531

**规模：** 50,000+ 标注样本

**来源：** 人民日报新闻文本

**PII 类型：**
- 人名（PER / nr）
- 地名（LOC / ns）
- 组织（ORG / nt）

**标注方案：** BIO 标注

**格式示例：**
```
我    O
叫    O
张    B-PER
三    I-PER
，    O
在    O
北    B-ORG
京    I-ORG
科    I-ORG
技    I-ORG
公    I-ORG
司    I-ORG
工    O
作    O
```

**优点：**
- ✅ 中文数据
- ✅ 新闻领域（通用性强）
- ✅ 权威标注（微软）
- ✅ 广泛使用（基准数据集）

**缺点：**
- ❌ 缺少电话、邮箱等结构化 PII
- ❌ 数据较老（2006年）

**推荐用法：**
```python
from datasets import load_dataset

# 加载 MSRA NER 数据集
dataset = load_dataset("levow/msra_ner")

# 转换为 SFT 格式
def convert_msra_to_sft(example):
    tokens = example['tokens']
    labels = example['ner_tags']

    # 提取实体
    entities = extract_entities(tokens, labels)

    return {
        "text": "".join(tokens),
        "entities": entities
    }
```

---

## 📚 补充数据集

### 4. People's Daily NER ⭐⭐⭐⭐（中文）

**来源：** 人民日报语料库
**规模：** 大规模

**特点：**
- 新闻领域
- 中文标注
- 实体类型：人名、地名、组织

**获取方式：**
- 通常与 MSRA 数据集一起使用
- 可在中文 NLP 社区获取

---

### 5. Weibo NER ⭐⭐⭐（中文，非正式文本）

**来源：** 微博文本
**特点：**
- 社交媒体文本
- 口语化、简短
- 适合非正式场景

**优点：**
- ✅ 口语化文本
- ✅ 包含昵称、缩写
- ✅ 更贴近实际应用

**缺点：**
- ❌ 噪声较多
- ❌ 标注质量参差不齐

---

### 6. CA4P-483（中文隐私政策）⭐⭐⭐

**来源：** 研究论文
**论文：** "A Fine-grained Chinese Software Privacy Policy Dataset"
**链接：** https://arxiv.org/abs/2212.04357

**规模：**
- 483 个中文安卓应用隐私政策
- 11,000+ 句子
- 52,000+ 细粒度标注

**PII 类型：**
- 隐私政策相关实体
- 符合法规的标注

**优点：**
- ✅ 专注隐私领域
- ✅ 中文数据
- ✅ 法规相关

**缺点：**
- ❌ 专注隐私政策文本（领域特定）
- ❌ 需要从论文获取

---

### 7. EduNER（中文教育领域）⭐⭐⭐

**来源：** 教育研究数据集
**论文：** "EduNER: a Chinese named entity recognition dataset for education research"
**链接：** https://link.springer.com/article/10.1007/s00521-023-08635-5

**特点：**
- 教育领域文本
- 中文标注
- 包含学生、教师、学校等实体

**用途：**
- 教育场景的 PII 检测
- 学生隐私保护

---

## 🔧 数据生成工具

### 8. Presidio（微软开源）⭐⭐⭐⭐⭐

**来源：** Microsoft
**GitHub：** https://github.com/microsoft/presidio

**功能：**
- PII 检测
- PII 脱敏
- 支持多语言

**数据生成：**
```python
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

# 检测 PII
results = analyzer.analyze(
    text="My phone is 13800138000",
    language='zh'
)

# 可以用于生成训练数据
```

**用途：**
- 基于规则生成初始标注
- 辅助人工标注
- 数据增强

---

### 9. Faker（合成数据生成）⭐⭐⭐⭐

**GitHub：** https://github.com/joke2k/faker

**功能：**
生成各类合成 PII 数据

**示例：**
```python
from faker import Faker

fake = Faker('zh_CN')  # 中文

# 生成训练数据
for _ in range(10000):
    text = f"我叫{fake.name()}，住在{fake.address()}，电话是{fake.phone_number()}"
    entities = [
        {"type": "PERSON_NAME", "value": fake.name()},
        {"type": "ADDRESS", "value": fake.address()},
        {"type": "PHONE_NUMBER", "value": fake.phone_number()}
    ]
```

**优点：**
- ✅ 快速生成大量数据
- ✅ 支持中文
- ✅ 多种 PII 类型

**缺点：**
- ❌ 合成数据，真实性不足
- ❌ 需要与真实数据混合使用

---

## 🎯 推荐训练策略

### 策略 1：多数据集混合（推荐）⭐⭐⭐⭐⭐

```python
# 数据配比
training_data = {
    "ai4privacy": 0.3,        # 30% - 英文通用 PII
    "MSRA": 0.3,              # 30% - 中文人名/地名/组织
    "BigCode": 0.1,           # 10% - 代码场景
    "Faker_synthetic": 0.2,   # 20% - 合成结构化 PII
    "Custom": 0.1             # 10% - 自己标注的领域数据
}

# 总量建议：10-50万样本
```

**原因：**
- 覆盖多种场景
- 平衡结构化和非结构化 PII
- 中英文混合

---

### 策略 2：渐进式训练

**阶段 1：基础训练（Qwen3-8B）**
```python
# 使用大模型生成高质量标注
teacher_model = Qwen3_8B()

# 标注 10-50 万真实数据
for text in your_corpus:
    entities = teacher_model.detect(text)
    training_data.append((text, entities))
```

**阶段 2：蒸馏训练（Qwen2-1.5B）**
```python
# 蒸馏到小模型
student_model = Qwen2_1_5B()

train(
    student_model,
    training_data,
    teacher_model=teacher_model,
    epochs=3
)
```

**阶段 3：领域适配**
```python
# 使用自己的数据 fine-tune
domain_data = your_specific_data

fine_tune(
    student_model,
    domain_data,
    learning_rate=1e-5
)
```

---

## 📊 数据增强技巧

### 1. 实体替换

```python
# 原始样本
text = "我叫张三，在北京工作"

# 增强样本
augmented = [
    "我叫李四，在北京工作",
    "我叫王五，在上海工作",
    "我叫赵六，在深圳工作"
]
```

### 2. 模板生成

```python
templates = [
    "我叫{name}，在{city}工作",
    "{name}是{org}的员工",
    "联系{name}：{phone}"
]

# 批量生成
for template in templates:
    for _ in range(1000):
        text = template.format(
            name=fake.name(),
            city=fake.city(),
            org=fake.company(),
            phone=fake.phone_number()
        )
```

### 3. 上下文变换

```python
# 同一个 PII，不同上下文
pii = "13800138000"

contexts = [
    f"我的电话是{pii}",
    f"请拨打{pii}联系我",
    f"手机号：{pii}",
    f"{pii}是我的联系方式"
]
```

---

## 🔨 实际使用建议

### 针对你的场景（云端大模型脱敏）

**推荐数据集组合：**

1. **MSRA NER**（30%）
   - 中文姓名、组织、地点
   - 新闻文本，通用性强

2. **ai4privacy/pii-masking-200k**（30%）
   - 多类型 PII
   - 脱敏/还原样本

3. **Faker 合成数据**（30%）
   - 中文电话、邮箱、身份证
   - 补充结构化 PII

4. **自己标注**（10%）
   - 你的实际业务数据
   - 最重要！

**总数据量：** 30-50 万样本

### SFT 数据格式

**格式 A：对话式**
```json
{
  "instruction": "检测以下文本中的 PII，并以 JSON 格式输出",
  "input": "我叫张三，电话13800138000",
  "output": {
    "entities": [
      {"type": "PERSON_NAME", "value": "张三", "start": 2, "end": 4},
      {"type": "PHONE_NUMBER", "value": "13800138000", "start": 6, "end": 17}
    ]
  }
}
```

**格式 B：单轮 Prompt**
```json
{
  "prompt": "/no_think\n你是 PII 检测专家。检测文本中的 PII：我叫张三，电话13800138000\n\n直接输出 JSON：",
  "completion": "{\"entities\": [{\"type\": \"PERSON_NAME\", \"value\": \"张三\"}, {\"type\": \"PHONE_NUMBER\", \"value\": \"13800138000\"}]}"
}
```

---

## 🚀 快速开始

### Step 1：下载数据集

```bash
# 安装依赖
pip install datasets huggingface_hub

# 下载 ai4privacy 数据集
python -c "
from datasets import load_dataset
dataset = load_dataset('ai4privacy/pii-masking-200k')
dataset.save_to_disk('data/ai4privacy')
"

# 下载 MSRA 数据集
python -c "
from datasets import load_dataset
dataset = load_dataset('levow/msra_ner')
dataset.save_to_disk('data/msra_ner')
"
```

### Step 2：生成合成数据

```bash
python scripts/generate_synthetic_pii.py \
  --output data/synthetic_pii.json \
  --num-samples 50000 \
  --language zh_CN
```

### Step 3：合并数据集

```bash
python scripts/merge_datasets.py \
  --inputs data/ai4privacy data/msra_ner data/synthetic_pii.json \
  --output data/merged_pii_dataset.json \
  --format sft
```

### Step 4：开始训练

```bash
# 使用 LoRA 微调
python scripts/train_pii_detector.py \
  --model Qwen/Qwen2-1.5B \
  --data data/merged_pii_dataset.json \
  --lora-r 8 \
  --epochs 3 \
  --output models/pii_detector_qwen2_1.5b
```

---

## 📌 总结

**最佳实践：**

1. ✅ **使用多数据集混合**（中英文 + 通用/领域）
2. ✅ **真实数据 + 合成数据**（70% 真实 + 30% 合成）
3. ✅ **包含脱敏样本**（ai4privacy 数据集）
4. ✅ **持续更新**（添加自己的业务数据）

**数据量建议：**
- 小规模尝试：5-10 万
- 生产部署：30-50 万
- 高精度要求：50-100 万

**99% 准确率路径：**
1. 使用 30 万+ 高质量数据
2. 包含多样化场景
3. 严格验证集测试
4. 持续迭代优化

---

**下一步：** 我可以帮你创建数据下载和预处理脚本，你需要吗？
