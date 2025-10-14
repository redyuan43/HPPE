# PII 隐私网关架构设计

**场景：** 本地数据 → 脱敏 → 云端 LLM → 还原 → 返回用户
**目标：** 嵌入式设备部署，99% 准确率，隐私优先
**工作负载：** 70% 批处理，30% 实时

---

## 核心需求分析

### 1. 数据流

```
┌──────────┐    ①检测    ┌──────────┐    ②脱敏    ┌──────────┐
│ 用户数据 │ ────────→   │ HPPE引擎 │ ────────→   │ 脱敏数据 │
│          │             │          │             │          │
│ "我叫张三│             │ 检测到:  │             │ "我叫[P1]│
│  在清华  │             │ - 张三   │             │  在[O1]  │
│  大学"   │             │ - 清华大学│            │  [O1]"   │
└──────────┘             └──────────┘             └──────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   映射表（加密） │
                    │  P1 → 张三      │
                    │  O1 → 清华大学   │
                    └─────────────────┘

┌──────────┐              ┌──────────┐    ④还原    ┌──────────┐
│云端LLM   │ ────────→    │ HPPE引擎 │ ────────→   │ 原始数据 │
│处理结果  │    ③返回     │          │             │          │
│"[P1]是   │              │ 查映射表: │             │"张三是   │
│ [O1]的   │              │ P1→张三  │             │ 清华大学 │
│ 学生"    │              │ O1→清华大学│            │ 的学生"  │
└──────────┘              └──────────┘             └──────────┘
```

### 2. 核心挑战

**准确率要求（99%）：**
- ❌ 漏检 1% → 敏感信息泄露到云端
- ❌ 误报 1% → 正常文本被替换，语义损坏

**嵌入式约束：**
- 有限的 CPU/内存/存储
- 可能没有 GPU
- 电池供电（功耗敏感）

**隐私保护：**
- 映射表必须加密存储
- 不能将映射表发送到云端
- 本地处理优先

---

## 架构设计

### 阶段 1：当前服务器架构（已实现）

**技术栈：**
- **检测：** Regex（结构化）+ Qwen3-8B（非结构化）
- **部署：** 服务器 + RTX 3060
- **准确率：** 100%（测试数据）

**局限：**
- 依赖 8B 模型（16GB 内存）
- 不适合嵌入式设备

### 阶段 2：嵌入式过渡架构（推荐）

#### 2.1 混合处理模式

```python
class EmbeddedPIIGateway:
    """嵌入式 PII 隐私网关"""

    def __init__(self):
        # 本地：快速 Regex 引擎
        self.local_detector = RegexDetector()

        # 可选：轻量 LLM（1-3B）
        self.light_llm = Optional[Qwen2_1_5B]()

        # 映射管理器（加密）
        self.mapper = EncryptedMapper()

    def anonymize(self, text: str, mode="safe"):
        """脱敏"""

        # 第一层：本地 Regex（必选，极快）
        regex_entities = self.local_detector.detect(text)

        # 第二层：轻量 LLM（可选，较慢）
        if mode == "safe" and self.light_llm:
            llm_entities = self.light_llm.detect(text)
            entities = merge(regex_entities, llm_entities)
        else:
            entities = regex_entities

        # 生成替换标记 + 保存映射
        anonymized, mapping = self._replace_entities(text, entities)
        self.mapper.store(mapping, encrypted=True)

        return anonymized

    def deanonymize(self, text: str, mapping_id: str):
        """还原"""
        mapping = self.mapper.load(mapping_id, decrypt=True)
        return self._restore_entities(text, mapping)
```

#### 2.2 分级处理策略

```
输入文本
  │
  ├─→ 快速模式（实时 30%）
  │     └─→ 仅 Regex（< 1ms）
  │         └─→ 覆盖率 60%，但速度极快
  │
  └─→ 安全模式（批处理 70%）
        └─→ Regex + 轻量 LLM（1-3s）
            └─→ 覆盖率 95%，准确率 99%
```

**权衡：**
- 实时场景：速度优先，接受 60% 覆盖率
- 批处理：准确率优先，99% 覆盖率

---

## 技术路线图

### Phase 1：脱敏/还原功能（立即实施）

**目标：** 在现有系统上添加脱敏和还原功能

**实现：**
```python
# 1. 创建脱敏器
from hppe.privacy import PIIAnonymizer

anonymizer = PIIAnonymizer(
    detector=HybridPIIDetector(mode="deep"),
    strategy="pseudonymization"  # 假名化
)

# 2. 脱敏
text = "我叫张三，在清华大学工作，电话13800138000"
result = anonymizer.anonymize(text)

print(result.anonymized_text)
# "我叫[PERSON_1]，在[ORG_1]工作，电话[PHONE_1]"

print(result.mapping)
# {
#   "PERSON_1": "张三",
#   "ORG_1": "清华大学",
#   "PHONE_1": "13800138000"
# }

# 3. 发送到云端 LLM
cloud_response = cloud_llm.generate(result.anonymized_text)
# "[PERSON_1]是[ORG_1]的教授"

# 4. 还原
original = anonymizer.deanonymize(
    cloud_response,
    result.mapping
)
print(original)
# "张三是清华大学的教授"
```

**优先级：** ⭐⭐⭐⭐⭐ 最高
**时间：** 1-2 周
**难度：** 🟢 简单

---

### Phase 2：模型蒸馏（3-6 个月）

**目标：** Qwen3-8B → Qwen2-1.5B，适配嵌入式

#### 2.1 数据准备

```python
# 使用 Qwen3-8B 生成训练数据
teacher_model = Qwen3_8B()

training_data = []
for text in diverse_corpus:  # 10-50万样本
    entities = teacher_model.detect(text)
    training_data.append({
        "text": text,
        "labels": entities  # 位置 + 类型 + 置信度
    })
```

**数据多样性：**
- 社交媒体文本（简短、口语化）
- 正式文档（完整、规范）
- 对话记录（多轮、上下文）
- 边界案例（模糊、困难）

#### 2.2 蒸馏训练

```python
# LoRA 微调（参数高效）
from peft import LoraConfig, get_peft_model

student_model = Qwen2_1_5B()

lora_config = LoraConfig(
    r=8,  # LoRA 秩
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

# 知识蒸馏损失
loss = (
    0.5 * task_loss +        # 任务损失
    0.3 * kl_divergence +    # 与教师模型的 KL 散度
    0.2 * feature_matching  # 中间特征匹配
)
```

**预期结果：**
- **模型大小：** 8B → 1.5B（减少 81%）
- **内存占用：** 16GB → 3GB
- **延迟：** 8.9s → 2-3s
- **准确率：** 100% → 97-99%

**适配嵌入式：**
```
设备类型           CPU            内存      模型选择
-------------------------------------------------
高端 (树莓派5)     Cortex-A76    8GB      Qwen2-1.5B (INT8)
中端 (Jetson Nano) Maxwell GPU   4GB      Qwen2-0.5B (INT4)
低端 (ESP32)       Xtensa        512KB    仅 Regex
```

---

### Phase 3：隐私保护机制（并行实施）

#### 3.1 映射表加密

```python
from cryptography.fernet import Fernet
import hashlib

class EncryptedMapper:
    def __init__(self, master_key: bytes):
        self.cipher = Fernet(master_key)

    def store(self, mapping: dict, session_id: str):
        """加密存储映射表"""

        # 序列化
        data = json.dumps(mapping).encode()

        # 加密
        encrypted = self.cipher.encrypt(data)

        # 存储（本地 SQLite / Redis）
        db.set(
            f"mapping:{session_id}",
            encrypted,
            expire=3600  # 1小时后自动删除
        )

    def load(self, session_id: str) -> dict:
        """解密读取映射表"""
        encrypted = db.get(f"mapping:{session_id}")
        data = self.cipher.decrypt(encrypted)
        return json.loads(data)
```

#### 3.2 安全传输

```python
# 方案 A：客户端加密（推荐）
class ClientSideEncryption:
    """客户端完全控制密钥"""

    def __init__(self):
        # 密钥永不离开客户端
        self.key = generate_key()

    def process(self, text):
        # 1. 本地检测 + 脱敏
        anonymized, mapping = detect_and_anonymize(text)

        # 2. 加密映射表（本地）
        encrypted_mapping = encrypt(mapping, self.key)

        # 3. 发送到云端（仅脱敏文本）
        response = cloud_llm(anonymized)

        # 4. 本地还原
        original = deanonymize(response, mapping)

        return original

# 方案 B：零知识证明（高级）
# - 证明数据已脱敏，但不泄露映射关系
# - 适用于合规审计场景
```

#### 3.3 隐私保护等级

```python
class PrivacyLevel(Enum):
    """隐私保护等级"""

    # L1：基础（快速）
    BASIC = "regex_only"
    # - 仅检测结构化 PII
    # - 适用于：日志记录、监控

    # L2：标准（平衡）
    STANDARD = "hybrid"
    # - Regex + 轻量 LLM
    # - 适用于：一般用户数据

    # L3：严格（最安全）
    STRICT = "conservative"
    # - 保守策略：宁可误报
    # - 人工审核辅助
    # - 适用于：医疗、金融数据
```

---

## 嵌入式优化策略

### 1. 模型量化

```python
# INT8 量化（推荐）
from optimum.quanto import quantize, freeze

model = Qwen2_1_5B()
quantize(model, weights=qint8)
freeze(model)

# 效果：
# - 模型大小：3GB → 1.5GB
# - 速度提升：+30%
# - 准确率：99.5% → 99%

# INT4 量化（激进）
quantize(model, weights=qint4)
# - 模型大小：3GB → 0.75GB
# - 速度提升：+50%
# - 准确率：99.5% → 97%（需验证）
```

### 2. 模型剪枝

```python
# 移除不重要的神经元
from torch.nn.utils import prune

# 结构化剪枝
prune.ln_structured(
    model.layers,
    name="weight",
    amount=0.3,  # 剪枝 30%
    n=2,
    dim=0
)

# 效果：
# - 模型大小：-30%
# - 延迟：-25%
# - 准确率：-1-2%
```

### 3. 知识蒸馏（推荐）

```python
# 方案：8B 教师 → 1.5B 学生

# 数据生成
teacher = Qwen3_8B()
student = Qwen2_1_5B()

for text in corpus:
    # 教师模型的输出
    teacher_logits = teacher(text)
    teacher_entities = extract_entities(teacher_logits)

    # 学生模型训练
    student_logits = student(text)

    # 蒸馏损失
    loss = (
        alpha * hard_loss(student_entities, true_entities) +
        (1-alpha) * soft_loss(student_logits, teacher_logits)
    )

    optimize(loss)
```

---

## 批处理优化（70% 场景）

### 1. 批量脱敏

```python
class BatchAnonymizer:
    def anonymize_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[AnonymizationResult]:
        """批量脱敏"""

        results = []

        # Regex 批处理（并行）
        with ThreadPoolExecutor(max_workers=8) as executor:
            regex_futures = [
                executor.submit(self.regex_detect, text)
                for text in texts
            ]
            regex_results = [f.result() for f in regex_futures]

        # LLM 批处理（分批）
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # 轻量 LLM 可能支持批量推理
            if self.light_llm.supports_batch:
                llm_results = self.light_llm.detect_batch(batch)
            else:
                llm_results = [
                    self.light_llm.detect(t) for t in batch
                ]

            # 合并结果
            for j, text in enumerate(batch):
                entities = merge(
                    regex_results[i+j],
                    llm_results[j]
                )
                result = self._anonymize_single(text, entities)
                results.append(result)

        return results
```

### 2. 流式处理

```python
def process_large_dataset(
    input_file: str,
    output_file: str,
    chunk_size: int = 1000
):
    """流式处理大文件"""

    anonymizer = BatchAnonymizer()

    with open(input_file, 'r') as fin, \
         open(output_file, 'w') as fout, \
         open('mappings.db', 'w') as fmap:

        while True:
            # 读取一批
            chunk = [fin.readline() for _ in range(chunk_size)]
            if not chunk[0]:
                break

            # 批量处理
            results = anonymizer.anonymize_batch(chunk)

            # 写入结果
            for result in results:
                fout.write(result.anonymized_text + '\n')
                fmap.write(json.dumps(result.mapping) + '\n')
```

---

## 性能预测

### 当前架构（服务器）

| 指标 | Regex | LLM (8B) | 混合 |
|------|-------|----------|------|
| 延迟（单个） | 0.02ms | 8.9s | 8.9s |
| 吞吐量（批处理） | 18K/s | 0.06/s | 0.06/s |
| 覆盖率 | 60% | 95% | 95% |
| 准确率 | 100% | 100% | 100% |

### 目标架构（嵌入式）

| 指标 | Regex | 轻量LLM (1.5B) | 混合 |
|------|-------|----------------|------|
| 延迟（单个） | 0.05ms | 2-3s | 3s |
| 吞吐量（批处理） | 10K/s | 0.3-0.5/s | 0.5/s |
| 覆盖率 | 60% | 90% | 90% |
| 准确率 | 100% | 97-99% | 98-99% |
| 内存占用 | <10MB | 2-3GB | 3GB |
| 功耗 | <1W | 5-10W | 10W |

**适配设备：**
- ✅ 树莓派 5（8GB RAM）
- ✅ Jetson Nano（4GB RAM，GPU）
- ❌ ESP32（内存太小，仅 Regex）

---

## 实施建议

### 优先级 1（立即，1-2周）：
1. **脱敏/还原功能** ⭐⭐⭐⭐⭐
   - 基于现有检测器
   - 添加映射管理
   - 加密存储

### 优先级 2（近期，1-2个月）：
2. **批处理优化** ⭐⭐⭐⭐⭐
   - 针对 70% 批处理场景
   - 流式处理大文件
   - 性能监控

3. **隐私保护机制** ⭐⭐⭐⭐⭐
   - 客户端加密
   - 安全传输
   - 合规审计日志

### 优先级 3（中期，3-6个月）：
4. **模型蒸馏** ⭐⭐⭐⭐⭐
   - 数据收集（10-50万）
   - 蒸馏训练
   - 准确率验证（目标 99%）

### 优先级 4（长期，6-12个月）：
5. **嵌入式适配** ⭐⭐⭐⭐
   - 量化优化
   - ARM 架构适配
   - 功耗优化

---

## 技术选型建议

### 轻量模型候选

1. **Qwen2-1.5B** ⭐⭐⭐⭐⭐（推荐）
   - 优势：中文优秀，官方支持
   - 内存：~3GB
   - 速度：2-3s/次
   - 适配：树莓派 5、Jetson

2. **Qwen2-0.5B** ⭐⭐⭐⭐
   - 优势：更小更快
   - 内存：~1GB
   - 速度：0.5-1s/次
   - 准确率：可能略低（需验证）

3. **仅 Regex** ⭐⭐⭐
   - 优势：极快，无依赖
   - 覆盖率：60%
   - 适用：低端设备、实时场景

### 部署方案

```python
# 配置文件驱动
config = {
    "deployment": {
        "device_type": "raspberry_pi_5",  # 或 jetson, server
        "model": "qwen2-1.5b-int8",
        "privacy_level": "standard",
        "batch_size": 16
    },
    "realtime": {
        "mode": "fast",  # 仅 Regex
        "timeout_ms": 100
    },
    "batch": {
        "mode": "safe",  # Regex + LLM
        "chunk_size": 1000
    }
}
```

---

## 总结

你的场景是一个 **隐私保护的智能网关**，核心要求是：

1. ✅ **99% 准确率** - 通过模型蒸馏 + 验证集测试保证
2. ✅ **隐私优先** - 客户端加密 + 本地处理
3. ✅ **批处理优化** - 流式处理 + 并行化（70% 场景）
4. ✅ **嵌入式就绪** - 模型蒸馏到 1.5B，量化到 INT8

**关键路径：**
```
当前（服务器 + 8B）
  ↓ 1-2周
添加脱敏/还原功能
  ↓ 1-2个月
批处理 + 隐私优化
  ↓ 3-6个月
模型蒸馏（8B → 1.5B）
  ↓ 6-12个月
嵌入式设备部署
```

**下一步：** 我建议先实现脱敏/还原功能，创建完整的数据流。你觉得如何？
