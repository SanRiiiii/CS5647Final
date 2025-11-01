#!/usr/bin/env python3
"""
MFCC特征提取脚本
提取所有音频文件的MFCC_2_mean, MFCC_4_mean, MFCC_10_mean特征
"""

import os
import librosa
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial


def extract_mfcc_features(audio_path, target_sr=24000, n_mfcc=13):
    """
    提取音频的MFCC特征
    
    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率 (默认24000，与训练配置一致)
        n_mfcc: MFCC系数数量
    
    Returns:
        dict: 包含mfcc_2_mean, mfcc_4_mean, mfcc_10_mean的字典
    """
    try:
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        # 提取MFCC特征 - 使用与训练配置一致的参数
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=target_sr, 
            n_mfcc=n_mfcc,
            n_fft=512,        # 与训练配置的filter_length一致
            hop_length=128,   # 与训练配置的hop_length一致
            n_mels=80         # 与训练配置的mel维度一致
        )  # shape: (n_mfcc, T)
        
        # 计算指定MFCC系数的均值
        mfcc_2_mean = np.mean(mfcc[2, :])   # MFCC_2的均值
        mfcc_4_mean = np.mean(mfcc[4, :])   # MFCC_4的均值  
        mfcc_10_mean = np.mean(mfcc[10, :]) # MFCC_10的均值
        
        return {
            'mfcc_2_mean': mfcc_2_mean,
            'mfcc_4_mean': mfcc_4_mean,
            'mfcc_10_mean': mfcc_10_mean,
            'success': True
        }
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return {
            'mfcc_2_mean': 0.0,
            'mfcc_4_mean': 0.0,
            'mfcc_10_mean': 0.0,
            'success': False,
            'error': str(e)
        }


def process_single_file(args):
    """处理单个文件的包装函数，用于多进程"""
    audio_path, dataset_root = args
    
    # 构建输出路径：dataset/{singer}/{songname}.mfcc.npy
    # 例如: dataset_raw/p226/p226_006.wav -> dataset_raw/p226/p226_006.mfcc.npy
    audio_path_obj = Path(audio_path)
    output_path = audio_path_obj.with_suffix('.mfcc.npy')
    
    # 检查输出文件是否已存在
    if output_path.exists():
        return f"Skipped (exists): {audio_path}"
    
    # 提取MFCC特征
    result = extract_mfcc_features(audio_path)
    
    if result['success']:
        # 保存特征
        features = np.array([
            result['mfcc_2_mean'],
            result['mfcc_4_mean'], 
            result['mfcc_10_mean']
        ])
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, features)
        return f"Success: {audio_path} -> {output_path}"
    else:
        return f"Failed: {audio_path} - {result.get('error', 'Unknown error')}"


def find_audio_files(dataset_dir, extensions=['.wav', '.mp3', '.flac', '.m4a']):
    """递归查找所有音频文件"""
    audio_files = []
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files


def main():
    parser = argparse.ArgumentParser(description='Extract MFCC features from audio files')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Dataset directory containing audio files')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--target_sr', type=int, default=24000,
                       help='Target sample rate (default: 24000 to match training config)')
    parser.add_argument('--n_mfcc', type=int, default=13,
                       help='Number of MFCC coefficients')
    
    args = parser.parse_args()
    
    print(f"🔍 Searching for audio files in: {args.dataset_dir}")
    
    # 查找所有音频文件
    audio_files = find_audio_files(args.dataset_dir)
    print(f"📁 Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("❌ No audio files found!")
        return
    
    # 准备多进程参数
    process_args = [(audio_path, args.dataset_dir) for audio_path in audio_files]
    
    print(f"🚀 Starting MFCC extraction with {args.num_workers} workers...")
    
    # 使用多进程处理
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, process_args),
            total=len(process_args),
            desc="Extracting MFCC features"
        ))
    
    # 统计结果
    success_count = sum(1 for r in results if r.startswith("Success"))
    skip_count = sum(1 for r in results if r.startswith("Skipped"))
    fail_count = sum(1 for r in results if r.startswith("Failed"))
    
    print(f"\n📊 Extraction Results:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ⏭️  Skipped: {skip_count}")
    print(f"   ❌ Failed: {fail_count}")
    print(f"   📁 Total: {len(audio_files)}")
    
#     # 保存统计信息
#     stats = {
#         'total_files': len(audio_files),
#         'success_count': success_count,
#         'skip_count': skip_count,
#         'fail_count': fail_count,
#         'target_sr': args.target_sr,
#         'n_mfcc': args.n_mfcc,
#         'features': ['mfcc_2_mean', 'mfcc_4_mean', 'mfcc_10_mean']
#     }
    
#     stats_path = os.path.join(args.dataset_dir, 'mfcc_extraction_stats.npy')
#     np.save(stats_path, stats)
#     print(f"📈 Statistics saved to: {stats_path}")
    
    # 显示一些示例结果
    print(f"\n🔍 Sample extracted features:")
    sample_files = audio_files[:5]
    for audio_file in sample_files:
        output_file = Path(audio_file).with_suffix('.mfcc.npy')
        if output_file.exists():
            features = np.load(output_file)
            print(f"   {Path(audio_file).name}: mfcc_2={features[0]:.3f}, mfcc_4={features[1]:.3f}, mfcc_10={features[2]:.3f}")


if __name__ == "__main__":
    main()
