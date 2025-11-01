#!/usr/bin/env python3
"""
说话人嵌入提取脚本
使用Wespeaker模型提取说话人嵌入特征
支持多线程并行处理
"""

import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import librosa
import numpy as np
import torch
import tqdm
import soundfile as sf
import wespeaker
from glob import glob
from random import shuffle

def process_single_file(args):
    """
    处理单个音频文件
    
    Args:
        args: 包含文件路径和输出路径的元组
    """
    song_path, out_dir, model_path = args
    
    try:
        # 读取音频并重采样到16kHz
        y, sr = librosa.load(song_path, sr=16000)
        
        # 创建临时文件
        temp_path = f"./tmp/{Path(song_path).name}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        sf.write(temp_path, y, 16000)
        
        # 加载模型并提取嵌入
        model = wespeaker.load_model(model_path)
        emb = model.extract_embedding(temp_path)
        
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        
        # 保存嵌入
        output_path = out_dir / f"{Path(song_path).stem}.spk.npy"
        np.save(output_path, emb)
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return True
        
    except Exception as e:
        print(f"Error processing {song_path}: {e}")
        return False

def process_batch(file_args, model_path):
    """
    处理一批文件
    
    Args:
        file_args: 文件参数列表
        model_path: 模型路径
    """
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    
    print(f"Rank {rank} processing {len(file_args)} files")
    
    success_count = 0
    for args in tqdm.tqdm(file_args, position=rank, desc=f"Rank {rank}"):
        if process_single_file(args + (model_path,)):
            success_count += 1
    
    print(f"Rank {rank} completed: {success_count}/{len(file_args)} files processed successfully")
    return success_count

def parallel_extract_spk_embd(base_dir, result_dir, model_path, num_processes=8):
    """
    并行提取说话人嵌入
    
    Args:
        base_dir: 输入目录
        result_dir: 输出目录
        model_path: 模型路径
        num_processes: 进程数
    """
    base_dir = Path(base_dir)
    result_dir = Path(result_dir)
    
    # 收集所有音频文件
    all_files = []
    for spk in base_dir.iterdir():
        if spk.is_dir():
            song_dir = base_dir / spk.name
            out_dir = result_dir / spk.name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for song in song_dir.iterdir():
                if song.suffix.lower() == '.wav':
                    # 检查是否已存在嵌入文件
                    emb_path = out_dir / f"{song.stem}.spk.npy"
                    if not emb_path.exists():
                        all_files.append((song, out_dir))
    
    print(f"Found {len(all_files)} files to process")
    
    if len(all_files) == 0:
        print("No files to process. All speaker embeddings already exist.")
        return
    
    # 随机打乱文件列表
    shuffle(all_files)
    
    # 创建临时目录
    os.makedirs("./tmp", exist_ok=True)
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(all_files) / num_processes)
            end = int((i + 1) * len(all_files) / num_processes)
            file_chunk = all_files[start:end]
            if file_chunk:  # 只有当chunk不为空时才提交任务
                tasks.append(executor.submit(process_batch, file_chunk, model_path))
        
        # 等待所有任务完成
        total_success = 0
        for task in tqdm.tqdm(tasks, position=0, desc="Overall progress"):
            total_success += task.result()
    
    print(f"\n处理完成: {total_success}/{len(all_files)} 个文件成功处理")

def main():
    parser = argparse.ArgumentParser(description='Extract speaker embeddings using Wespeaker')
    parser.add_argument('--base_dir', type=str, default='./dataset_raw',
                       help='Base directory containing speaker folders')
    parser.add_argument('--result_dir', type=str, default='./dataset',
                       help='Result directory to save embeddings')
    parser.add_argument('--model_path', type=str, default='./voxblink2_samresnet34_ft',
                       help='Path to Wespeaker model')
    parser.add_argument('--num_processes', type=int, default=8,
                       help='Number of parallel processes')
    
    args = parser.parse_args()
    
    # 设置多进程启动方法
    mp.set_start_method("spawn", force=True)
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return
    
    # 检查输入目录
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory {args.base_dir} does not exist")
        return
    
    print(f"Using {args.num_processes} processes")
    print(f"Input directory: {args.base_dir}")
    print(f"Output directory: {args.result_dir}")
    print(f"Model path: {args.model_path}")
    
    # 开始处理
    parallel_extract_spk_embd(
        args.base_dir, 
        args.result_dir, 
        args.model_path, 
        args.num_processes
    )

if __name__ == "__main__":
    main()
