#!/usr/bin/env python3
"""
独立的域标签提取器
基于音频文件路径为不同音频添加域标签标记
- 0: 语音 (p开头的数据)
- 1: 歌唱 (其他数据)
完全独立于原本的特征提取脚本，不影响原有逻辑
"""

import argparse
import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

def determine_domain_label(file_path):
    """
    根据文件路径确定域标签
    
    Args:
        file_path: 文件路径，如 "./dataset/p278/p278_323.wav"
    
    Returns:
        int: 0表示语音，1表示歌唱
    """
    # 提取文件名
    filename = os.path.basename(file_path)
    
    # 判断是否以p开头
    if filename.startswith('p'):
        return 0  # 语音
    else:
        return 1  # 歌唱

def extract_domain_labels(audio_files):
    """
    为音频文件列表提取域标签
    
    Args:
        audio_files: 音频文件路径列表
    
    Returns:
        dict: 文件路径到域标签的映射
    """
    domain_labels = {}
    
    for file_path in audio_files:
        domain_label = determine_domain_label(file_path)
        domain_labels[file_path] = domain_label
    
    return domain_labels


def analyze_domain_distribution(domain_labels):
    """
    分析域标签分布
    
    Args:
        domain_labels: 域标签字典
    """
    speech_count = 0
    singing_count = 0
    
    for label in domain_labels.values():
        if label == 0:
            speech_count += 1
        else:
            singing_count += 1
    
    print(f"\n域标签分布:")
    print(f"  语音文件 (p开头): {speech_count} 个")
    print(f"  歌唱文件 (其他): {singing_count} 个")
    print(f"  总计: {len(domain_labels)} 个文件")

def save_individual_domain_labels(audio_files, dataset_dir, overwrite=False):
    """
    为每个音频文件保存单独的域标签文件
    
    Args:
        audio_files: 音频文件路径列表
        dataset_dir: 数据集根目录
        overwrite: 是否覆盖已存在的文件
    """
    processed_count = 0
    skipped_count = 0
    
    for audio_file in tqdm(audio_files, desc="保存域标签文件"):
        # 确定域标签
        domain_label = determine_domain_label(audio_file)
        
        # 生成对应的域标签文件路径
        # 例如: ./dataset/p278/p278_323.wav -> ./dataset/p278/p278_323.domain.npy
        domain_file = audio_file.replace('.wav', '.domain.npy')
        
        # 检查是否已存在
        if os.path.exists(domain_file) and not overwrite:
            skipped_count += 1
            continue
        
        # 保存域标签
        try:
            np.save(domain_file, domain_label)
            processed_count += 1
        except Exception as e:
            print(f"Error saving {domain_file}: {e}")
    
    return processed_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description='Extract domain labels for audio files')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', 
                       help='Dataset directory path')
    parser.add_argument('--filelist', type=str, default=None,
                       help='Specific filelist to process (optional)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing domain label files')
    
    args = parser.parse_args()
    
    # 获取音频文件列表
    if args.filelist and os.path.exists(args.filelist):
        # 从文件列表读取
        with open(args.filelist, 'r', encoding='utf-8') as f:
            audio_files = [line.strip() for line in f if line.strip()]
        print(f"从文件列表读取: {len(audio_files)} 个文件")
    else:
        # 从数据集目录扫描
        audio_files = glob(os.path.join(args.dataset_dir, "**", "*.wav"), recursive=True)
        print(f"从数据集目录扫描: {len(audio_files)} 个文件")
    
    if len(audio_files) == 0:
        print("No audio files found.")
        return
    
    # 提取域标签
    print("提取域标签...")
    domain_labels = extract_domain_labels(audio_files)
    
    # 保存单独的域标签文件
    print("保存单独的域标签文件...")
    processed_count, skipped_count = save_individual_domain_labels(
        audio_files, args.dataset_dir, args.overwrite
    )
    print(f"处理完成: {processed_count} 个文件，跳过: {skipped_count} 个文件")
    
    # 分析分布
    analyze_domain_distribution(domain_labels)
    
    print(f"\n处理完成!")
    print(f"每个音频文件都有对应的 .domain.npy 文件")

if __name__ == "__main__":
    main()
