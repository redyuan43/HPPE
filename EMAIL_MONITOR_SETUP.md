# 训练进度邮件监控 - 配置指南

## 📧 功能说明

这个程序会每隔2小时自动发送训练进度到您的邮箱，包括：
- 当前训练进度（百分比、步数）
- 已用时间和预计剩余时间
- 训练速度
- 美观的HTML格式邮件

---

## ⚙️ 快速配置（3步完成）

### 第1步：获取SMTP授权码

**如果使用QQ邮箱**:
1. 登录QQ邮箱网页版
2. 点击"设置" → "账户"
3. 找到"POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV服务"
4. 开启"IMAP/SMTP服务"
5. 点击"生成授权码"
6. **保存这个授权码**（这不是你的QQ密码！）

**如果使用163邮箱**:
1. 登录163邮箱
2. 设置 → POP3/SMTP/IMAP
3. 开启IMAP/SMTP服务
4. 设置客户端授权密码
5. **保存这个授权码**

**如果使用Gmail**:
1. 开启两步验证
2. 生成应用专用密码
3. **保存这个密码**

---

### 第2步：修改配置

编辑文件：`scripts/training_email_monitor.py`

找到这几行并修改：

```python
# 接收邮件地址 (修改这里!)
RECEIVER_EMAIL = "your_email@example.com"  # 改成你的邮箱

# 发送邮件配置
SENDER_EMAIL = "your_qq_email@qq.com"      # 改成你的QQ邮箱
SENDER_PASSWORD = "your_smtp_password"      # 改成第1步获取的授权码

# SMTP服务器配置（QQ邮箱不用改，其他邮箱看下面）
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465
```

**其他邮箱配置**:
```python
# 163邮箱
SMTP_SERVER = "smtp.163.com"
SMTP_PORT = 465

# Gmail
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587  # Gmail使用587端口
```

---

### 第3步：启动监控

```bash
# 在后台运行监控程序
nohup python3 scripts/training_email_monitor.py > logs/email_monitor.log 2>&1 &

# 查看监控程序输出
tail -f logs/email_monitor.log
```

**搞定！** 🎉 现在每2小时会自动发送进度邮件到你的邮箱。

---

## 📊 邮件示例

你会收到这样的邮件：

```
标题: [训练监控] PII检测模型 Phase 2 训练 - 检查 #1

内容:
🚀 PII检测模型 Phase 2 训练 - 进度报告 #1

报告时间：2025-10-17 13:30:00
当前状态：训练中

[进度条] 35% ████████░░░░░░░░░░░░

训练进度    | 1,346 / 3,846 steps
完成百分比  | 35%
已用时间    | 8:12:34
预计剩余    | 15:23:45
训练速度    | 21.65 秒/step
```

---

## 🛠️ 常见问题

### Q1: 邮件发送失败？

**检查清单**:
1. ✅ 邮箱和密码是否正确？
2. ✅ 使用的是**SMTP授权码**，不是邮箱登录密码？
3. ✅ SMTP服务器地址和端口是否正确？
4. ✅ 网络是否畅通？

**查看详细错误**:
```bash
tail -f logs/email_monitor.log
```

### Q2: 如何修改检查间隔？

编辑脚本，找到这一行：
```python
CHECK_INTERVAL = 2 * 3600  # 2小时（秒）
```

修改为你想要的时间：
```python
CHECK_INTERVAL = 1 * 3600  # 1小时
CHECK_INTERVAL = 30 * 60   # 30分钟
CHECK_INTERVAL = 4 * 3600  # 4小时
```

### Q3: 如何停止监控？

```bash
# 找到进程ID
ps aux | grep training_email_monitor

# 停止进程
kill <进程ID>
```

### Q4: 收不到邮件？

1. 检查垃圾邮件箱
2. 查看监控日志：`tail -f logs/email_monitor.log`
3. 确认SMTP授权码正确
4. 尝试发送测试邮件（见下方测试命令）

---

## 🧪 测试邮件发送

配置完成后，先测试一下：

```bash
# 直接运行监控程序（不放后台）
python3 scripts/training_email_monitor.py

# 如果成功，应该看到:
# ✅ 邮件发送成功到: your_email@example.com
```

如果失败，根据错误信息调整配置。

---

## 📝 高级配置

### 自定义邮件标题

修改脚本中的：
```python
PROJECT_NAME = "PII检测模型 Phase 2 训练"  # 改成你喜欢的名字
```

### 自定义日志文件

如果训练日志文件名不同，修改：
```python
LOG_FILE = "logs/train_17pii_full_20251017_113333.log"
```

### 发送给多个邮箱

修改为逗号分隔：
```python
RECEIVER_EMAIL = "email1@example.com, email2@example.com"
```

---

## ⚡ 一键启动脚本

为了方便，我创建了一键启动脚本：

```bash
# 赋予执行权限
chmod +x scripts/start_email_monitor.sh

# 启动监控
./scripts/start_email_monitor.sh
```

---

## 📞 需要帮助？

如果遇到问题：
1. 查看日志：`tail -f logs/email_monitor.log`
2. 检查配置是否正确
3. 确认SMTP授权码是否有效

监控程序会持续运行直到：
- 训练完成
- 手动停止
- 发生错误

祝训练顺利！🚀
