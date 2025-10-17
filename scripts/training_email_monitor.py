#!/usr/bin/env python3
"""
训练进度邮件监控程序
每2小时发送一次训练进度到指定邮箱
"""
import os
import sys
import time
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

# ============================================================================
# 邮件配置 (请修改为您的配置)
# ============================================================================

# 接收邮件地址 (修改这里!)
RECEIVER_EMAIL = "your_email@example.com"

# 发送邮件配置 (使用QQ邮箱示例，也可以用163、Gmail等)
SENDER_EMAIL = "your_qq_email@qq.com"  # 发送者邮箱
SENDER_PASSWORD = "your_smtp_password"  # SMTP授权码(不是QQ密码!)

# SMTP服务器配置
SMTP_SERVER = "smtp.qq.com"  # QQ邮箱
SMTP_PORT = 465  # SSL端口

# 其他常用邮箱配置参考:
# 163邮箱: smtp.163.com, 端口465
# Gmail: smtp.gmail.com, 端口587 (需要TLS)
# 企业微信邮箱: smtp.exmail.qq.com, 端口465

# ============================================================================
# 监控配置
# ============================================================================

LOG_FILE = "logs/train_17pii_full_20251017_113333.log"  # 训练日志文件
CHECK_INTERVAL = 2 * 3600  # 检查间隔：2小时（秒）
PROJECT_NAME = "PII检测模型 Phase 2 训练"

# ============================================================================
# 提取训练进度
# ============================================================================

def parse_training_progress(log_file):
    """从日志文件提取训练进度"""
    try:
        if not os.path.exists(log_file):
            return {
                "status": "日志文件不存在",
                "error": f"找不到日志: {log_file}"
            }

        # 读取日志最后500行
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            recent_lines = lines[-500:] if len(lines) > 500 else lines

        # 查找最新的进度行
        # 格式: "  0%|          | 11/3846 [03:58<23:03:57, 21.65s/it]"
        progress_pattern = r'(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]+)<([^\]]+),\s*([\d.]+)s/it\]'

        latest_progress = None
        for line in reversed(recent_lines):
            match = re.search(progress_pattern, line)
            if match:
                percent, current, total, elapsed, remaining, speed = match.groups()
                latest_progress = {
                    "status": "训练中",
                    "percent": int(percent),
                    "current_step": int(current),
                    "total_steps": int(total),
                    "elapsed_time": elapsed,
                    "remaining_time": remaining,
                    "speed": float(speed)
                }
                break

        if latest_progress:
            return latest_progress

        # 检查是否已完成
        if any("训练完成" in line or "🎉" in line for line in recent_lines[-50:]):
            return {"status": "训练已完成", "message": "模型训练成功完成"}

        # 检查是否有错误
        error_keywords = ["Error", "Exception", "Traceback", "failed"]
        for line in reversed(recent_lines[-100:]):
            if any(keyword in line for keyword in error_keywords):
                return {
                    "status": "训练出错",
                    "error": line.strip()
                }

        # 默认状态
        return {
            "status": "等待中",
            "message": "训练尚未开始或日志未更新"
        }

    except Exception as e:
        return {
            "status": "解析错误",
            "error": str(e)
        }

# ============================================================================
# 生成邮件内容
# ============================================================================

def generate_email_content(progress, check_count):
    """生成HTML格式的邮件内容"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status = progress.get("status", "未知")

    # HTML邮件模板
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; }}
            .content {{ margin-top: 20px; }}
            .progress-bar {{
                width: 100%;
                height: 30px;
                background-color: #f0f0f0;
                border-radius: 15px;
                overflow: hidden;
                margin: 10px 0;
            }}
            .progress-fill {{
                height: 100%;
                background-color: #4CAF50;
                text-align: center;
                line-height: 30px;
                color: white;
                font-weight: bold;
            }}
            .info-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .info-table td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            .info-table td:first-child {{
                font-weight: bold;
                width: 150px;
                background-color: #f9f9f9;
            }}
            .footer {{
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🚀 {PROJECT_NAME} - 进度报告 #{check_count}</h2>
        </div>

        <div class="content">
            <p><strong>报告时间：</strong>{now}</p>
            <p><strong>当前状态：</strong><span style="color: {'green' if status == '训练中' else 'orange'}; font-weight: bold;">{status}</span></p>
    """

    if status == "训练中":
        percent = progress.get("percent", 0)
        current = progress.get("current_step", 0)
        total = progress.get("total_steps", 0)
        elapsed = progress.get("elapsed_time", "N/A")
        remaining = progress.get("remaining_time", "N/A")
        speed = progress.get("speed", 0)

        html += f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percent}%">{percent}%</div>
            </div>

            <table class="info-table">
                <tr>
                    <td>训练进度</td>
                    <td>{current:,} / {total:,} steps</td>
                </tr>
                <tr>
                    <td>完成百分比</td>
                    <td>{percent}%</td>
                </tr>
                <tr>
                    <td>已用时间</td>
                    <td>{elapsed}</td>
                </tr>
                <tr>
                    <td>预计剩余</td>
                    <td>{remaining}</td>
                </tr>
                <tr>
                    <td>训练速度</td>
                    <td>{speed:.2f} 秒/step</td>
                </tr>
            </table>
        """
    elif status == "训练已完成":
        html += f"""
            <div style="padding: 20px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; margin: 20px 0;">
                <h3 style="color: #155724; margin: 0;">✅ 训练成功完成！</h3>
                <p style="margin: 10px 0 0 0;">{progress.get('message', '')}</p>
            </div>
        """
    elif "错误" in status or "error" in status.lower():
        error_msg = progress.get("error", "未知错误")
        html += f"""
            <div style="padding: 20px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; margin: 20px 0;">
                <h3 style="color: #721c24; margin: 0;">❌ 训练出现错误</h3>
                <p style="margin: 10px 0 0 0; font-family: monospace; font-size: 12px;">{error_msg}</p>
            </div>
        """
    else:
        message = progress.get("message", progress.get("error", "无详细信息"))
        html += f"""
            <div style="padding: 20px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; margin: 20px 0;">
                <p style="margin: 0;">{message}</p>
            </div>
        """

    html += f"""
        </div>

        <div class="footer">
            <p>此邮件由训练监控系统自动发送</p>
            <p>日志文件: {LOG_FILE}</p>
            <p>检查间隔: {CHECK_INTERVAL // 3600} 小时</p>
        </div>
    </body>
    </html>
    """

    return html

# ============================================================================
# 发送邮件
# ============================================================================

def send_email(subject, html_content):
    """发送HTML格式邮件"""
    try:
        # 创建邮件
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL

        # 添加HTML内容
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)

        # 连接SMTP服务器并发送
        print(f"📧 连接SMTP服务器: {SMTP_SERVER}:{SMTP_PORT}")

        if SMTP_PORT == 465:
            # SSL连接
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        else:
            # TLS连接
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()

        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()

        print(f"✅ 邮件发送成功到: {RECEIVER_EMAIL}")
        return True

    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False

# ============================================================================
# 主监控循环
# ============================================================================

def main():
    print("="*70)
    print("🔔 训练进度邮件监控启动")
    print("="*70)
    print(f"项目: {PROJECT_NAME}")
    print(f"日志: {LOG_FILE}")
    print(f"接收邮箱: {RECEIVER_EMAIL}")
    print(f"检查间隔: {CHECK_INTERVAL // 3600} 小时")
    print("="*70 + "\n")

    # 检查配置
    if RECEIVER_EMAIL == "your_email@example.com" or SENDER_EMAIL == "your_qq_email@qq.com":
        print("❌ 错误：请先配置邮箱信息！")
        print("   1. 打开脚本文件")
        print("   2. 修改 RECEIVER_EMAIL（接收邮箱）")
        print("   3. 修改 SENDER_EMAIL 和 SENDER_PASSWORD（发送邮箱和授权码）")
        print("   4. 如需使用其他邮箱，请修改SMTP配置")
        sys.exit(1)

    check_count = 0

    try:
        while True:
            check_count += 1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n{'='*70}")
            print(f"🔍 第 {check_count} 次检查 ({now})")
            print(f"{'='*70}")

            # 解析进度
            progress = parse_training_progress(LOG_FILE)
            print(f"状态: {progress.get('status')}")

            if progress.get('status') == '训练中':
                print(f"进度: {progress.get('current_step')}/{progress.get('total_steps')} ({progress.get('percent')}%)")
                print(f"剩余时间: {progress.get('remaining_time')}")

            # 生成邮件内容
            subject = f"[训练监控] {PROJECT_NAME} - 检查 #{check_count}"
            html_content = generate_email_content(progress, check_count)

            # 发送邮件
            send_email(subject, html_content)

            # 如果训练完成，发送最后一封邮件后退出
            if progress.get('status') == '训练已完成':
                print("\n🎉 训练已完成，监控程序退出")
                break

            # 等待下次检查
            print(f"\n⏰ 下次检查时间: {CHECK_INTERVAL // 3600} 小时后")
            print(f"💤 等待中... (按Ctrl+C停止)")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n⚠️  监控程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
