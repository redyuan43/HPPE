# Epic 4: 脱敏与屏蔽模块

**状态：** 🚀 进行中
**优先级：** P0
**开始日期：** 2025-10-18
**负责人：** AI Team
**Phase：** Phase 4

---

## 📋 目标概述

实现可配置的PII脱敏策略模块，提供多种脱敏方式，确保检测到的敏感信息被安全地脱敏处理，同时保持数据的可用性。

**关键目标**：
- 实现可扩展的脱敏策略框架
- 支持多种脱敏方式（编辑、屏蔽、哈希、合成替换）
- 保持文本的可读性和结构完整性
- 支持按PII类型配置不同策略

**成功指标**：
- **策略覆盖率**: 100%（所有PII类型可脱敏）
- **可读性保持**: 脱敏后文本语义可理解
- **可逆性**: 哈希策略不可逆
- **格式一致性**: 合成替换保持原格式

---

## 🎯 Story 列表

### Story 4.1: 脱敏策略框架

**优先级**: P0
**工作量**: 3-4天
**依赖**: Epic 3 完成

**需求描述**:

构建可扩展的脱敏策略框架，定义统一的策略接口，支持不同策略的注册和调用。

**核心组件**:

1. **AnonymizationStrategy（抽象基类）**
   ```python
   class AnonymizationStrategy(ABC):
       @abstractmethod
       def anonymize(self, entity: Entity, text: str) -> str:
           """对实体进行脱敏处理"""
           pass

       @abstractmethod
       def get_replacement(self, entity: Entity) -> str:
           """获取替换文本"""
           pass
   ```

2. **AnonymizationEngine（策略引擎）**
   ```python
   class AnonymizationEngine:
       def __init__(self):
           self.strategies = {}  # type -> strategy 映射

       def register_strategy(self, pii_type: str, strategy: AnonymizationStrategy):
           """注册策略"""
           pass

       def anonymize_entities(self, entities: List[Entity], text: str) -> str:
           """批量脱敏"""
           pass
   ```

3. **AnonymizationConfig（配置类）**
   ```python
   @dataclass
   class AnonymizationConfig:
       # PII类型到策略的映射
       type_strategy_map: Dict[str, str] = field(default_factory=dict)

       # 默认策略
       default_strategy: str = "redact"

       # 保留格式（如电话号码保留中间4位屏蔽）
       preserve_format: bool = True
   ```

**验收标准**:
- AC1: 策略接口定义完整，支持所有脱敏类型
- AC2: 策略引擎支持动态注册和配置
- AC3: 支持批量脱敏，处理实体位置偏移
- AC4: 包含完整的单元测试

**技术挑战**:
- 批量脱敏时维护文本位置偏移
- 策略的可配置性和可扩展性
- 处理重叠实体的脱敏

---

### Story 4.2: 编辑和屏蔽策略

**优先级**: P0
**工作量**: 2-3天
**依赖**: Story 4.1

**需求描述**:

实现两种最基础的脱敏策略：**编辑（Redaction）**和**屏蔽（Masking）**。

**1. 编辑策略（RedactionStrategy）**

完全删除PII，替换为固定占位符。

**示例**:
```
原文: "我的身份证号是110101199001011234，请核对"
脱敏: "我的身份证号是[REDACTED]，请核对"
```

**配置选项**:
```python
@dataclass
class RedactionConfig:
    # 占位符模板（可包含类型信息）
    placeholder_template: str = "[{entity_type}]"  # 如 "[ID_CARD]"

    # 是否保留部分信息（如前2后4）
    partial_reveal: Dict[str, Tuple[int, int]] = field(default_factory=dict)
```

**2. 屏蔽策略（MaskingStrategy）**

部分遮蔽PII，保留部分信息用于识别。

**示例**:
```
原文: "我的身份证号是110101199001011234"
脱敏: "我的身份证号是1101**********1234"  # 保留前4后4

原文: "手机号13812345678"
脱敏: "手机号138****5678"  # 保留前3后4
```

**配置选项**:
```python
@dataclass
class MaskingConfig:
    # 屏蔽字符
    mask_char: str = "*"

    # 按类型配置保留模式（前N后M）
    reveal_patterns: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "ID_CARD": (4, 4),        # 身份证保留前4后4
        "PHONE_NUMBER": (3, 4),   # 手机号保留前3后4
        "BANK_CARD": (6, 4),      # 银行卡保留前6后4
        "EMAIL": (2, 0),          # 邮箱保留前2字符
    })
```

**验收标准**:
- AC1: 编辑策略正确替换所有PII
- AC2: 屏蔽策略按配置保留指定位数
- AC3: 支持按PII类型定制屏蔽模式
- AC4: 脱敏后文本保持可读性
- AC5: 包含不同PII类型的测试用例

---

### Story 4.3: 哈希和加密策略

**优先级**: P0
**工作量**: 2-3天
**依赖**: Story 4.2

**需求描述**:

实现**哈希（Hashing）**和**加密（Encryption）**策略，提供不可逆和可逆的脱敏选项。

**1. 哈希策略（HashingStrategy）**

将PII转换为不可逆的哈希值，用于数据分析场景。

**示例**:
```
原文: "我的身份证号是110101199001011234"
脱敏: "我的身份证号是7a3f8c2d9e1b4a6f8c2d9e1b4a6f8c2d"
```

**配置选项**:
```python
@dataclass
class HashingConfig:
    # 哈希算法（sha256, sha512, md5）
    algorithm: str = "sha256"

    # 是否添加盐值
    use_salt: bool = True

    # 盐值（如果use_salt=True）
    salt: str = "hppe-default-salt"

    # 是否保留长度（用空格填充）
    preserve_length: bool = False

    # 是否添加前缀标识
    add_prefix: bool = True  # 如 "HASH:7a3f..."
```

**2. 加密策略（EncryptionStrategy）**

可逆加密，允许授权用户解密还原原始数据。

**示例**:
```
原文: "我的身份证号是110101199001011234"
脱敏: "我的身份证号是ENC:AES256:3f8c2d9e1b4a6f8c"
```

**配置选项**:
```python
@dataclass
class EncryptionConfig:
    # 加密算法（AES256, RSA）
    algorithm: str = "AES256"

    # 加密密钥（Base64编码）
    encryption_key: str = None  # 必需

    # 是否添加前缀标识
    add_prefix: bool = True  # 如 "ENC:AES256:..."

    # 输出格式（hex, base64）
    output_format: str = "hex"
```

**安全要求**:
- 密钥存储与代码分离
- 支持密钥轮换
- 加密结果可解密还原

**验收标准**:
- AC1: 哈希策略不可逆，相同输入产生相同哈希
- AC2: 支持多种哈希算法（SHA256, SHA512）
- AC3: 加密策略可逆，支持AES256加密
- AC4: 密钥管理安全（不硬编码）
- AC5: 包含安全性测试用例

---

### Story 4.4: 合成数据替换

**优先级**: P1
**工作量**: 3-4天
**依赖**: Story 4.3

**需求描述**:

实现**合成替换（Synthetic Replacement）**策略，用格式相同但虚假的数据替换PII，保持数据的统计特性。

**应用场景**:
- 数据分析：保留数据分布和模式
- 测试环境：生成逼真的测试数据
- 演示环境：展示功能但保护真实数据

**1. 合成策略（SyntheticStrategy）**

**示例**:
```
原文: "我的身份证号是110101199001011234"
脱敏: "我的身份证号是330206198705123456"  # 格式正确的虚假身份证

原文: "联系电话13812345678"
脱敏: "联系电话13956781234"  # 格式正确的虚假手机号

原文: "邮箱user@example.com"
脱敏: "邮箱john.doe@sample.org"  # 虚假邮箱

原文: "地址：北京市海淀区中关村大街1号"
脱敏: "地址：上海市浦东新区陆家嘴环路999号"  # 虚假地址
```

**配置选项**:
```python
@dataclass
class SyntheticConfig:
    # 是否保持一致性（同一PII生成相同合成值）
    consistent_mapping: bool = True

    # 合成数据池
    fake_data_providers: Dict[str, str] = field(default_factory=lambda: {
        "PERSON_NAME": "faker",      # 使用Faker库
        "ID_CARD": "generator",      # 自定义生成器
        "PHONE_NUMBER": "generator",
        "EMAIL": "faker",
        "ADDRESS": "faker",
    })

    # 地区偏好（如生成中国姓名）
    locale: str = "zh_CN"
```

**技术实现**:

1. **ID_CARD生成器**
   ```python
   def generate_fake_id_card() -> str:
       # 生成格式正确的虚假身份证
       # 1. 随机地区码
       # 2. 随机出生日期
       # 3. 随机序列号
       # 4. 计算正确的校验位
       return fake_id
   ```

2. **PHONE_NUMBER生成器**
   ```python
   def generate_fake_phone() -> str:
       # 生成格式正确的虚假手机号
       # 1. 选择运营商前缀（13x, 15x, 18x）
       # 2. 随机生成后8位
       return fake_phone
   ```

3. **使用Faker库**
   ```python
   from faker import Faker

   fake = Faker('zh_CN')
   fake_name = fake.name()
   fake_email = fake.email()
   fake_address = fake.address()
   ```

4. **一致性映射（可选）**
   ```python
   # 相同PII生成相同合成值
   mapping_cache = {}  # original_value -> synthetic_value

   def get_synthetic_value(original: str, entity_type: str) -> str:
       if original in mapping_cache:
           return mapping_cache[original]

       synthetic = generate_synthetic(entity_type)
       mapping_cache[original] = synthetic
       return synthetic
   ```

**验收标准**:
- AC1: 合成数据格式正确（通过格式验证）
- AC2: 合成数据与原数据不同（防止碰撞）
- AC3: 支持一致性映射（可选）
- AC4: 支持多种PII类型的合成
- AC5: 合成数据符合地区特征（如中文姓名）
- AC6: 包含多种PII类型的测试用例

**依赖库**:
- `Faker`: 生成虚假姓名、邮箱、地址
- 自定义生成器：身份证、银行卡、手机号

---

## 📦 交付物

### 代码组件
1. `src/hppe/anonymizer/` - 脱敏模块
   - `strategy.py` - 策略基类和接口
   - `engine.py` - 脱敏引擎
   - `strategies/` - 具体策略实现
     - `redaction.py` - 编辑策略
     - `masking.py` - 屏蔽策略
     - `hashing.py` - 哈希策略
     - `encryption.py` - 加密策略
     - `synthetic.py` - 合成替换策略
   - `config.py` - 配置类
   - `generators.py` - 合成数据生成器

2. `tests/test_anonymizer/` - 测试套件
   - `test_redaction.py`
   - `test_masking.py`
   - `test_hashing.py`
   - `test_encryption.py`
   - `test_synthetic.py`
   - `test_engine.py`

### 文档
1. `docs/anonymization_guide.md` - 脱敏使用指南
2. `docs/strategy_configuration.md` - 策略配置文档
3. `docs/security_considerations.md` - 安全注意事项

---

## 🔬 测试策略

### 单元测试
- 每个策略独立测试
- 覆盖正常场景和边界情况
- 测试覆盖率 > 90%

### 集成测试
- 脱敏引擎与检测引擎集成
- 多策略混合使用
- 批量脱敏性能测试

### 安全测试
- 哈希不可逆性验证
- 加密可逆性验证
- 密钥管理安全性测试

---

## 📊 成功指标

| 指标 | 目标 |
|------|------|
| **策略覆盖率** | 100%（所有PII类型） |
| **脱敏准确率** | 100%（无遗漏） |
| **可读性评分** | > 85%（人工评估） |
| **性能** | < 10ms/实体 |
| **测试覆盖率** | > 90% |

---

## 🚧 技术风险

### R1: 批量脱敏位置偏移
**风险**：批量替换时文本位置会变化，导致后续实体位置错误
**缓解**：从后向前处理实体，或维护偏移量映射

### R2: 合成数据碰撞
**风险**：合成数据可能与真实数据碰撞
**缓解**：生成后验证唯一性，使用大数据池

### R3: 加密密钥管理
**风险**：密钥泄露导致数据暴露
**缓解**：使用环境变量、密钥管理服务（如AWS KMS）

---

## 🎯 里程碑

| 里程碑 | 预计日期 | 状态 |
|--------|----------|------|
| M1: Story 4.1 完成 | Day 4 | ⏳ 计划中 |
| M2: Story 4.2 完成 | Day 7 | ⏳ 计划中 |
| M3: Story 4.3 完成 | Day 10 | ⏳ 计划中 |
| M4: Story 4.4 完成 | Day 14 | ⏳ 计划中 |
| M5: Epic 4 完成 | Day 14 | ⏳ 计划中 |

---

**文档版本：** 1.0
**最后更新：** 2025-10-18
**下一步：** 开始 Story 4.1 实现
