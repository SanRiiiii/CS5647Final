#!/usr/bin/env python3
"""
独立的F0统计信息提取器
基于现有的.f0.npy文件计算F0统计信息（最大值、最小值、均值、中位数）
完全独立于原本的特征提取脚本，不影响原有逻辑
"""

import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm

def extract_f0_stats(f0_path):
    """
    从F0文件提取统计信息
    
    Args:
        f0_path: .f0.npy文件路径
    
    Returns:
        dict: 包含F0统计信息的字典
    """
    try:
        # 加载F0数据
        f0_data = np.load(f0_path, allow_pickle=True)
        f0, uv = f0_data[0], f0_data[1]
        
        # 转换为numpy数组
        f0 = np.array(f0, dtype=float)
        uv = np.array(uv, dtype=float)
        
        # 过滤出非零F0值
        f0_nonzero = f0[f0 > 0]
        
        if len(f0_nonzero) > 0:
            f0_mean = np.mean(f0_nonzero)
            f0_median = np.median(f0_nonzero)
            f0_min = np.min(f0_nonzero)
            f0_max = np.max(f0_nonzero)
        else:
            f0_mean = 0.0
            f0_median = 0.0
            f0_min = 0.0
            f0_max = 0.0
        
        f0_stats = {
            'mean': float(f0_mean),
            'median': float(f0_median),
            'min': float(f0_min),
            'max': float(f0_max)
        }
        
        return f0_stats
        
    except Exception as e:
        print(f"Error processing {f0_path}: {e}")
        return None

def save_individual_f0_stats(f0_files, overwrite=False):
    """
    为每个F0文件保存单独的统计信息文件
    
    Args:
        f0_files: F0文件路径列表
        overwrite: 是否覆盖已存在的文件
    """
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for f0_path in tqdm(f0_files, desc="保存F0统计文件"):
        # 生成对应的统计文件路径
        # 例如: ./dataset/p278/p278_323.f0.npy -> ./dataset/p278/p278_323.f0_stats.npy
        stats_path = f0_path.replace('.f0.npy', '.f0_stats.npy')
        
        # 检查是否已存在
        if os.path.exists(stats_path) and not overwrite:
            skipped_count += 1
            continue
        
        # 提取统计信息
        f0_stats = extract_f0_stats(f0_path)
        
        if f0_stats is not None:
            # 保存统计信息
            np.save(stats_path, f0_stats)
            processed_count += 1
        else:
            error_count += 1
    
    return processed_count, skipped_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Extract F0 statistics from existing F0 files')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', 
                       help='Dataset directory path')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing stats files')
    
    args = parser.parse_args()
    
    # 获取所有F0文件
    f0_files = glob(os.path.join(args.dataset_dir, "**", "*.f0.npy"), recursive=True)
    print(f"Found {len(f0_files)} F0 files")
    
    if len(f0_files) == 0:
        print("No F0 files found. Please run preprocessing3_feature.py first to generate F0 files.")
        return
    
    # 保存单独的F0统计文件
    print("保存单独的F0统计文件...")
    processed_count, skipped_count, error_count = save_individual_f0_stats(
        f0_files, args.overwrite
    )
    
    print(f"\n处理完成:")
    print(f"  处理: {processed_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件")
    print(f"  错误: {error_count} 个文件")
    print(f"每个F0文件都有对应的 .f0_stats.npy 文件")

if __name__ == "__main__":
    main()
