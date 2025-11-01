#!/usr/bin/env python3
"""
训练监控脚本
实时监控训练状态，检测异常并自动处理
"""

import os
import time
import json
import argparse
from pathlib import Path

def monitor_training(log_dir="logs", check_interval=30):
    """
    监控训练状态
    
    Args:
        log_dir: 日志目录
        check_interval: 检查间隔（秒）
    """
    print(f"开始监控训练状态: {log_dir}")
    print(f"检查间隔: {check_interval}秒")
    
    log_file = os.path.join(log_dir, "train.log")
    if not os.path.exists(log_file):
        print(f"错误: 日志文件不存在 {log_file}")
        return
    
    last_size = 0
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    while True:
        try:
            # 检查日志文件大小
            current_size = os.path.getsize(log_file)
            
            if current_size > last_size:
                # 读取新的日志内容
                with open(log_file, 'r', encoding='utf-8') as f:
                    f.seek(last_size)
                    new_content = f.read()
                
                # 检查错误信息
                error_indicators = [
                    "nan loss",
                    "inf loss", 
                    "Loss too large",
                    "Invalid loss detected",
                    "Too many consecutive bad batches",
                    "Early stopping triggered"
                ]
                
                for indicator in error_indicators:
                    if indicator in new_content:
                        consecutive_errors += 1
                        print(f"⚠️  检测到错误: {indicator}")
                        
                        if consecutive_errors >= max_consecutive_errors:
                            print("🚨 检测到连续错误，建议检查训练状态")
                            consecutive_errors = 0
                
                # 检查训练进度
                if "epoch:" in new_content:
                    lines = new_content.split('\n')
                    for line in lines:
                        if "epoch:" in line and "loss:" in line:
                            print(f"📊 {line.strip()}")
                
                last_size = current_size
                consecutive_errors = 0  # 重置错误计数
            else:
                # 检查是否长时间没有更新
                if time.time() - os.path.getmtime(log_file) > 300:  # 5分钟
                    print("⚠️  日志文件长时间未更新，可能训练已停止")
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"监控错误: {e}")
            time.sleep(check_interval)

def check_training_health(log_dir="logs"):
    """
    检查训练健康状态
    """
    print("=== 训练健康检查 ===")
    
    # 检查日志文件
    log_file = os.path.join(log_dir, "train.log")
    if not os.path.exists(log_file):
        print("❌ 日志文件不存在")
        return False
    
    # 检查模型检查点
    model_files = list(Path(log_dir).glob("model_*.pt"))
    if not model_files:
        print("❌ 没有找到模型检查点")
        return False
    
    print(f"✅ 找到 {len(model_files)} 个模型检查点")
    
    # 检查最新的日志内容
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有错误
    error_count = sum(content.count(error) for error in [
        "nan loss", "inf loss", "Loss too large", "Invalid loss detected"
    ])
    
    if error_count > 0:
        print(f"⚠️  发现 {error_count} 个错误")
    else:
        print("✅ 没有发现错误")
    
    # 检查训练进度
    if "epoch:" in content:
        print("✅ 训练正在进行中")
    else:
        print("❌ 没有发现训练进度")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="训练监控脚本")
    parser.add_argument("--log_dir", default="logs", help="日志目录")
    parser.add_argument("--interval", type=int, default=30, help="检查间隔（秒）")
    parser.add_argument("--check", action="store_true", help="只检查健康状态，不持续监控")
    
    args = parser.parse_args()
    
    if args.check:
        check_training_health(args.log_dir)
    else:
        monitor_training(args.log_dir, args.interval)

if __name__ == "__main__":
    main()
