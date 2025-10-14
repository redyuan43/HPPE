#!/usr/bin/env python3
"""从ModelScope下载Qwen3-4B模型"""

from modelscope import snapshot_download
import os

print("开始从 ModelScope 下载 Qwen3-4B 模型...")
print("预计大小: ~8GB")
print("预计时间: 5-10分钟（国内镜像更快）\n")

try:
    # ModelScope上的Qwen3-4B路径
    model_dir = snapshot_download(
        'Qwen/Qwen3-4B',
        cache_dir=os.path.expanduser('~/.cache/modelscope/hub'),
        revision='master'
    )

    print(f"\n✓ 模型下载完成！")
    print(f"路径: {model_dir}")

    # 验证文件
    import glob
    safetensors = glob.glob(f"{model_dir}/*.safetensors") + glob.glob(f"{model_dir}/**/*.safetensors", recursive=True)
    config_json = glob.glob(f"{model_dir}/config.json")

    print(f"\n文件验证:")
    print(f"  权重文件: {len(safetensors)} 个")
    print(f"  配置文件: {'✓' if config_json else '✗'}")

    if safetensors and config_json:
        print("\n🎉 模型文件完整，可以开始训练！")
        print(f"\n使用此路径训练: {model_dir}")
    else:
        print("\n⚠️  模型文件不完整，请重新下载")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n提示: 如果提示权限或网络问题，可能需要:")
    print("  1. pip install modelscope --upgrade")
    print("  2. 检查网络连接")
