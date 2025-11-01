#!/usr/bin/env python3
"""
检查spk_embd文件的内容
"""

import os
import numpy as np
import torch
from pathlib import Path

def check_spk_embd_files(dataset_dir="./dataset", max_files=5):
    """
    检查spk_embd文件的内容
    
    Args:
        dataset_dir: 数据集目录
        max_files: 最多检查的文件数量
    """
    print("🔍 检查spk_embd文件...")
    print("=" * 60)
    
    spk_files = []
    
    # 查找所有.spk.npy文件
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.spk.npy'):
                spk_files.append(os.path.join(root, file))
    
    if not spk_files:
        print(f"❌ 在 {dataset_dir} 中没有找到任何 .spk.npy 文件")
        return
    
    print(f"📁 找到 {len(spk_files)} 个 .spk.npy 文件")
    print(f"🔢 检查前 {min(max_files, len(spk_files))} 个文件:")
    print()
    
    # 检查前几个文件
    for i, spk_file in enumerate(spk_files[:max_files]):
        try:
            print(f"[{i+1}] {spk_file}")
            
            # 加载文件
            spk_embd = np.load(spk_file)
            
            # 基本信息
            print(f"    📊 Shape: {spk_embd.shape}")
            print(f"    📊 Dtype: {spk_embd.dtype}")
            print(f"    📊 Size: {spk_embd.size} elements")
            
            # 数值统计
            print(f"    📈 Min: {spk_embd.min():.6f}")
            print(f"    📈 Max: {spk_embd.max():.6f}")
            print(f"    📈 Mean: {spk_embd.mean():.6f}")
            print(f"    📈 Std: {spk_embd.std():.6f}")
            
            # 检查是否有异常值
            has_nan = np.isnan(spk_embd).any()
            has_inf = np.isinf(spk_embd).any()
            print(f"    ⚠️  Has NaN: {has_nan}")
            print(f"    ⚠️  Has Inf: {has_inf}")
            
            # 显示前10个值
            print(f"    🔢 First 10 values: {spk_embd[:10]}")
            
            # 检查是否全为0
            is_all_zero = np.allclose(spk_embd, 0)
            print(f"    🔢 All zeros: {is_all_zero}")
            
            print()
            
        except Exception as e:
            print(f"    ❌ Error loading {spk_file}: {e}")
            print()

def check_spk_embd_in_filelist(filelist_path, max_files=5):
    """
    检查filelist中指定的spk_embd文件
    
    Args:
        filelist_path: filelist文件路径
        max_files: 最多检查的文件数量
    """
    print("🔍 检查filelist中的spk_embd文件...")
    print("=" * 60)
    
    if not os.path.exists(filelist_path):
        print(f"❌ Filelist文件不存在: {filelist_path}")
        return
    
    # 读取filelist
    with open(filelist_path, 'r') as f:
        lines = f.read().splitlines()
    
    print(f"📁 Filelist包含 {len(lines)} 个文件")
    print(f"🔢 检查前 {min(max_files, len(lines))} 个对应的spk_embd文件:")
    print()
    
    checked_count = 0
    for line in lines:
        if checked_count >= max_files:
            break
            
        # 构造spk_embd文件路径
        spk_file = line.replace('.wav', '.spk.npy')
        
        if os.path.exists(spk_file):
            try:
                print(f"[{checked_count+1}] {spk_file}")
                
                # 加载文件
                spk_embd = np.load(spk_file)
                
                # 基本信息
                print(f"    📊 Shape: {spk_embd.shape}")
                print(f"    📊 Dtype: {spk_embd.dtype}")
                print(f"    📊 Size: {spk_embd.size} elements")
                
                # 数值统计
                print(f"    📈 Min: {spk_embd.min():.6f}")
                print(f"    📈 Max: {spk_embd.max():.6f}")
                print(f"    📈 Mean: {spk_embd.mean():.6f}")
                print(f"    📈 Std: {spk_embd.std():.6f}")
                
                # 检查是否有异常值
                has_nan = np.isnan(spk_embd).any()
                has_inf = np.isinf(spk_embd).any()
                print(f"    ⚠️  Has NaN: {has_nan}")
                print(f"    ⚠️  Has Inf: {has_inf}")
                
                # 显示前10个值
                print(f"    🔢 First 10 values: {spk_embd[:10]}")
                
                # 检查是否全为0
                is_all_zero = np.allclose(spk_embd, 0)
                print(f"    🔢 All zeros: {is_all_zero}")
                
                print()
                checked_count += 1
                
            except Exception as e:
                print(f"    ❌ Error loading {spk_file}: {e}")
                print()
        else:
            print(f"[{checked_count+1}] ❌ Spk file not found: {spk_file}")
            print()
            checked_count += 1

def main():
    """主函数"""
    print("🎵 CoMoSVC Speaker Embedding Checker")
    print("=" * 60)
    
    # 检查dataset目录中的spk_embd文件
    dataset_dir = "./dataset"
    if os.path.exists(dataset_dir):
        check_spk_embd_files(dataset_dir, max_files=3)
    else:
        print(f"❌ Dataset目录不存在: {dataset_dir}")
    
    print("\n" + "=" * 60)
    
    # 检查filelist中的spk_embd文件
    filelist_paths = [
        "./filelists/train.txt",
        "./filelists/val.txt",
        "./filelists/train_with_domain.txt",
        "./filelists/val_with_domain.txt"
    ]
    
    for filelist_path in filelist_paths:
        if os.path.exists(filelist_path):
            print(f"\n📋 检查filelist: {filelist_path}")
            check_spk_embd_in_filelist(filelist_path, max_files=2)
            break
    else:
        print("❌ 没有找到任何filelist文件")

if __name__ == "__main__":
    main()
