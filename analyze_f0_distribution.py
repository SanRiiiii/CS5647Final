#!/usr/bin/env python3
"""
F0分布分析脚本
从.f0.npy文件读取pitch序列，分析统计分布并可视化

使用方法：
    # 分析单个文件
    python analyze_f0_distribution.py --f0_file ./dataset/singer1/audio.f0.npy
    
    # 分析整个数据集
    python analyze_f0_distribution.py --dataset_dir ./dataset
    
    # 按说话人分析
    python analyze_f0_distribution.py --dataset_dir ./dataset --by_speaker
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from tqdm import tqdm

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_f0_file(f0_path):
    """
    加载.f0.npy文件
    
    Args:
        f0_path: F0文件路径
    
    Returns:
        f0: F0序列 (Hz)
        uv: voiced/unvoiced标记
    """
    try:
        f0_data = np.load(f0_path, allow_pickle=True)
        f0 = np.array(f0_data[0], dtype=float)
        uv = np.array(f0_data[1], dtype=float)
        return f0, uv
    except Exception as e:
        print(f"Error loading {f0_path}: {e}")
        return None, None


def load_f0_stats_file(stats_path):
    """
    加载.f0_stats.npy文件
    
    Args:
        stats_path: F0统计文件路径
    
    Returns:
        dict: {'mean': float, 'median': float, 'min': float, 'max': float}
    """
    try:
        stats_data = np.load(stats_path, allow_pickle=True).item()
        return stats_data
    except Exception as e:
        print(f"Error loading {stats_path}: {e}")
        return None


def analyze_f0_sequence(f0, uv, name=""):
    """
    分析单个F0序列的统计特征
    
    Args:
        f0: F0序列
        uv: UV标记
        name: 序列名称
    
    Returns:
        dict: 统计信息
    """
    # 提取有声段的F0
    f0_voiced = f0[f0 > 0]
    
    if len(f0_voiced) == 0:
        print(f"Warning: {name} has no voiced frames")
        return None
    
    # 计算统计量
    stats = {
        'name': name,
        'total_frames': len(f0),
        'voiced_frames': len(f0_voiced),
        'unvoiced_frames': len(f0) - len(f0_voiced),
        'voiced_ratio': len(f0_voiced) / len(f0),
        'f0_mean': np.mean(f0_voiced),
        'f0_median': np.median(f0_voiced),
        'f0_std': np.std(f0_voiced),
        'f0_min': np.min(f0_voiced),
        'f0_max': np.max(f0_voiced),
        'f0_range': np.max(f0_voiced) - np.min(f0_voiced),
        'f0_percentile_5': np.percentile(f0_voiced, 5),
        'f0_percentile_95': np.percentile(f0_voiced, 95),
    }
    
    # 计算log空间的统计量（与训练时对齐）
    f0_log = np.log(1 + f0_voiced / 700)
    stats['f0_log_mean'] = np.mean(f0_log)
    stats['f0_log_std'] = np.std(f0_log)
    
    return stats


def print_stats(stats):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"F0 Statistics: {stats['name']}")
    print(f"{'='*60}")
    print(f"Total frames:     {stats['total_frames']}")
    print(f"Voiced frames:    {stats['voiced_frames']} ({stats['voiced_ratio']*100:.1f}%)")
    print(f"Unvoiced frames:  {stats['unvoiced_frames']} ({(1-stats['voiced_ratio'])*100:.1f}%)")
    print(f"\n--- F0 Distribution (Hz) ---")
    print(f"Mean:             {stats['f0_mean']:.2f} Hz")
    print(f"Median:           {stats['f0_median']:.2f} Hz")
    print(f"Std:              {stats['f0_std']:.2f} Hz")
    print(f"Min:              {stats['f0_min']:.2f} Hz")
    print(f"Max:              {stats['f0_max']:.2f} Hz")
    print(f"Range:            {stats['f0_range']:.2f} Hz")
    print(f"5th percentile:   {stats['f0_percentile_5']:.2f} Hz")
    print(f"95th percentile:  {stats['f0_percentile_95']:.2f} Hz")
    print(f"\n--- Log Space (log(1+f0/700)) ---")
    print(f"Mean:             {stats['f0_log_mean']:.4f}")
    print(f"Std:              {stats['f0_log_std']:.4f}")
    print(f"{'='*60}\n")


def plot_f0_distribution(f0_list, labels=None, title="F0 Distribution", save_path=None):
    """
    绘制F0分布图
    
    Args:
        f0_list: F0序列列表
        labels: 标签列表
        title: 图表标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(f0_list)))
    
    # 1. 直方图
    ax1 = axes[0, 0]
    for i, f0 in enumerate(f0_list):
        f0_voiced = f0[f0 > 0]
        label = labels[i] if labels else f"Sequence {i+1}"
        ax1.hist(f0_voiced, bins=50, alpha=0.6, label=label, color=colors[i], edgecolor='black')
    ax1.set_xlabel('F0 (Hz)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('F0 Histogram', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱线图
    ax2 = axes[0, 1]
    f0_voiced_list = [f0[f0 > 0] for f0 in f0_list]
    box_labels = labels if labels else [f"Seq {i+1}" for i in range(len(f0_list))]
    bp = ax2.boxplot(f0_voiced_list, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('F0 (Hz)', fontsize=12)
    ax2.set_title('F0 Boxplot', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Log空间分布
    ax3 = axes[1, 0]
    for i, f0 in enumerate(f0_list):
        f0_voiced = f0[f0 > 0]
        f0_log = np.log(1 + f0_voiced / 700)
        label = labels[i] if labels else f"Sequence {i+1}"
        ax3.hist(f0_log, bins=50, alpha=0.6, label=label, color=colors[i], edgecolor='black')
    ax3.set_xlabel('log(1 + f0/700)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('F0 Log Space Distribution (Training Space)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积分布函数（CDF）
    ax4 = axes[1, 1]
    for i, f0 in enumerate(f0_list):
        f0_voiced = f0[f0 > 0]
        f0_sorted = np.sort(f0_voiced)
        cdf = np.arange(1, len(f0_sorted) + 1) / len(f0_sorted)
        label = labels[i] if labels else f"Sequence {i+1}"
        ax4.plot(f0_sorted, cdf, label=label, color=colors[i], linewidth=2)
    ax4.set_xlabel('F0 (Hz)', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('F0 Cumulative Distribution', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.show()


def analyze_f0_stats_quartet(dataset_dir, by_speaker=False):
    """
    分析F0统计四元组（mean, median, min, max）的分布
    
    Args:
        dataset_dir: 数据集目录
        by_speaker: 是否按说话人分组分析
    """
    # 查找所有F0统计文件
    stats_files = glob(os.path.join(dataset_dir, "**", "*.f0_stats.npy"), recursive=True)
    print(f"\n{'='*70}")
    print(f"F0 Statistics Quartet Analysis")
    print(f"{'='*70}")
    print(f"Found {len(stats_files)} F0 stats files\n")
    
    if len(stats_files) == 0:
        print("No .f0_stats.npy files found!")
        print("Please run f0_stats_extractor.py first.")
        return
    
    # 收集所有统计值
    all_means = []
    all_medians = []
    all_mins = []
    all_maxs = []
    all_ranges = []
    
    speaker_stats = defaultdict(lambda: {'means': [], 'medians': [], 'mins': [], 'maxs': []})
    
    for stats_path in tqdm(stats_files, desc="Loading F0 stats"):
        stats = load_f0_stats_file(stats_path)
        if stats and stats['mean'] > 0:  # 过滤无效数据
            all_means.append(stats['mean'])
            all_medians.append(stats['median'])
            all_mins.append(stats['min'])
            all_maxs.append(stats['max'])
            all_ranges.append(stats['max'] - stats['min'])
            
            if by_speaker:
                speaker = os.path.basename(os.path.dirname(stats_path))
                speaker_stats[speaker]['means'].append(stats['mean'])
                speaker_stats[speaker]['medians'].append(stats['median'])
                speaker_stats[speaker]['mins'].append(stats['min'])
                speaker_stats[speaker]['maxs'].append(stats['max'])
    
    # 转换为numpy数组
    all_means = np.array(all_means)
    all_medians = np.array(all_medians)
    all_mins = np.array(all_mins)
    all_maxs = np.array(all_maxs)
    all_ranges = np.array(all_ranges)
    
    # 打印统计摘要
    print(f"\n{'='*70}")
    print(f"F0 Statistics Quartet Summary (across {len(all_means)} files)")
    print(f"{'='*70}\n")
    
    stats_names = ['Mean', 'Median', 'Min', 'Max', 'Range']
    stats_arrays = [all_means, all_medians, all_mins, all_maxs, all_ranges]
    
    for name, arr in zip(stats_names, stats_arrays):
        print(f"--- {name} ---")
        print(f"  Mean:   {np.mean(arr):.2f} Hz")
        print(f"  Median: {np.median(arr):.2f} Hz")
        print(f"  Std:    {np.std(arr):.2f} Hz")
        print(f"  Min:    {np.min(arr):.2f} Hz")
        print(f"  Max:    {np.max(arr):.2f} Hz")
        print(f"  5th:    {np.percentile(arr, 5):.2f} Hz")
        print(f"  95th:   {np.percentile(arr, 95):.2f} Hz")
        print()
    
    # Log空间统计
    print(f"--- Log Space (log(1+f0/700)) ---")
    for name, arr in zip(['Mean', 'Median', 'Min', 'Max'], [all_means, all_medians, all_mins, all_maxs]):
        arr_log = np.log(1 + arr / 700)
        print(f"{name:8s}: mean={np.mean(arr_log):.4f}, std={np.std(arr_log):.4f}")
    print()
    
    # 可视化
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 四个统计量的分布（箱线图）
    ax1 = fig.add_subplot(gs[0, :])
    bp = ax1.boxplot([all_means, all_medians, all_mins, all_maxs],
                      labels=['Mean', 'Median', 'Min', 'Max'],
                      patch_artist=True,
                      showmeans=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel('F0 (Hz)', fontsize=12)
    ax1.set_title('F0 Statistics Quartet Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2-5. 各统计量的直方图
    positions = [(1, 0), (1, 1), (1, 2), (2, 0)]
    for idx, (name, arr, color, pos) in enumerate(zip(
        ['Mean', 'Median', 'Min', 'Max'],
        [all_means, all_medians, all_mins, all_maxs],
        colors,
        positions
    )):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.hist(arr, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(arr), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(arr):.1f}')
        ax.axvline(np.median(arr), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(arr):.1f}')
        ax.set_xlabel('F0 (Hz)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 6. Range分布
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(all_ranges, bins=30, color='#95E1D3', alpha=0.7, edgecolor='black')
    ax6.axvline(np.mean(all_ranges), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(all_ranges):.1f}')
    ax6.set_xlabel('F0 Range (Hz)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('F0 Range Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Mean vs Range 散点图
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.scatter(all_means, all_ranges, alpha=0.5, s=30, c=all_medians, cmap='viridis')
    ax7.set_xlabel('F0 Mean (Hz)', fontsize=11)
    ax7.set_ylabel('F0 Range (Hz)', fontsize=11)
    ax7.set_title('Mean vs Range', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax7.collections[0], ax=ax7)
    cbar.set_label('Median (Hz)', fontsize=10)
    
    plt.suptitle(f'F0 Statistics Analysis ({len(stats_files)} files)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    save_path = os.path.join(dataset_dir, "f0_stats_quartet_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ F0 stats quartet plot saved to: {save_path}")
    plt.show()
    
    # 按说话人分析
    if by_speaker and len(speaker_stats) > 0:
        print(f"\n{'='*70}")
        print(f"F0 Statistics by Speaker")
        print(f"{'='*70}\n")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('F0 Statistics by Speaker', fontsize=16, fontweight='bold')
        
        speakers = sorted(speaker_stats.keys())
        stat_names = ['Mean', 'Median', 'Min', 'Max']
        stat_keys = ['means', 'medians', 'mins', 'maxs']
        
        for idx, (ax, stat_name, stat_key) in enumerate(zip(axes.flat, stat_names, stat_keys)):
            data = [speaker_stats[spk][stat_key] for spk in speakers]
            bp = ax.boxplot(data, labels=speakers, patch_artist=True)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('F0 (Hz)', fontsize=11)
            ax.set_title(f'{stat_name} by Speaker', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 打印每个说话人的统计
            print(f"--- {stat_name} by Speaker ---")
            for spk in speakers:
                values = speaker_stats[spk][stat_key]
                print(f"  {spk:20s}: {np.mean(values):6.2f} Hz (±{np.std(values):5.2f})")
            print()
        
        plt.tight_layout()
        save_path = os.path.join(dataset_dir, "f0_stats_by_speaker.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Speaker comparison plot saved to: {save_path}")
        plt.show()


def analyze_dataset(dataset_dir, by_speaker=False):
    """
    分析整个数据集的F0分布
    
    Args:
        dataset_dir: 数据集目录
        by_speaker: 是否按说话人分组分析
    """
    # 查找所有F0文件
    f0_files = glob(os.path.join(dataset_dir, "**", "*.f0.npy"), recursive=True)
    print(f"Found {len(f0_files)} F0 files")
    
    if len(f0_files) == 0:
        print("No F0 files found!")
        return
    
    if by_speaker:
        # 按说话人分组
        speaker_f0 = defaultdict(list)
        
        for f0_path in tqdm(f0_files, desc="Loading F0 files"):
            speaker = os.path.basename(os.path.dirname(f0_path))
            f0, uv = load_f0_file(f0_path)
            if f0 is not None:
                speaker_f0[speaker].append(f0)
        
        # 分析每个说话人
        speaker_stats = {}
        all_f0_by_speaker = []
        speaker_names = []
        
        print(f"\n{'='*60}")
        print(f"Analysis by Speaker")
        print(f"{'='*60}\n")
        
        for speaker in sorted(speaker_f0.keys()):
            # 合并该说话人的所有F0
            all_f0 = np.concatenate(speaker_f0[speaker])
            all_f0_by_speaker.append(all_f0)
            speaker_names.append(speaker)
            
            # 分析
            stats = analyze_f0_sequence(all_f0, None, name=speaker)
            if stats:
                speaker_stats[speaker] = stats
                print_stats(stats)
        
        # 绘制对比图
        if len(all_f0_by_speaker) > 0:
            plot_f0_distribution(
                all_f0_by_speaker,
                labels=speaker_names,
                title=f"F0 Distribution by Speaker ({len(speaker_names)} speakers)",
                save_path=os.path.join(dataset_dir, "f0_distribution_by_speaker.png")
            )
    
    else:
        # 整体分析
        all_f0 = []
        all_stats = []
        
        for f0_path in tqdm(f0_files, desc="Loading F0 files"):
            f0, uv = load_f0_file(f0_path)
            if f0 is not None:
                all_f0.extend(f0[f0 > 0].tolist())
                
                # 单文件统计
                filename = os.path.basename(f0_path)
                stats = analyze_f0_sequence(f0, uv, name=filename)
                if stats:
                    all_stats.append(stats)
        
        all_f0 = np.array(all_f0)
        
        # 整体统计
        overall_stats = analyze_f0_sequence(all_f0, None, name="Overall Dataset")
        print_stats(overall_stats)
        
        # 各文件统计汇总
        print(f"\n{'='*60}")
        print(f"Per-File Statistics Summary")
        print(f"{'='*60}")
        print(f"Number of files: {len(all_stats)}")
        
        if len(all_stats) > 0:
            mean_f0s = [s['f0_mean'] for s in all_stats]
            print(f"Mean F0 across files:")
            print(f"  Average: {np.mean(mean_f0s):.2f} Hz")
            print(f"  Std:     {np.std(mean_f0s):.2f} Hz")
            print(f"  Range:   {np.min(mean_f0s):.2f} - {np.max(mean_f0s):.2f} Hz")
        
        # 绘图
        plot_f0_distribution(
            [all_f0],
            labels=["Overall"],
            title=f"F0 Distribution ({len(f0_files)} files)",
            save_path=os.path.join(dataset_dir, "f0_distribution_overall.png")
        )


def main():
    parser = argparse.ArgumentParser(description='Analyze F0 distribution from .npy files')
    parser.add_argument('--f0_file', type=str, help='Single F0 file to analyze')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', 
                       help='Dataset directory (analyze all F0 files)')
    parser.add_argument('--by_speaker', action='store_true',
                       help='Analyze by speaker (group by subdirectory)')
    parser.add_argument('--stats_only', action='store_true',
                       help='Only analyze F0 stats quartet (from .f0_stats.npy files)')
    
    args = parser.parse_args()
    
    if args.f0_file:
        # 分析单个文件
        print(f"Analyzing single file: {args.f0_file}")
        f0, uv = load_f0_file(args.f0_file)
        
        if f0 is not None:
            stats = analyze_f0_sequence(f0, uv, name=os.path.basename(args.f0_file))
            print_stats(stats)
            
            # 绘制F0轨迹
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
            
            # F0轨迹
            time = np.arange(len(f0)) * 512 / 44100  # 假设hop_length=512, sr=44100
            ax1.plot(time, f0, linewidth=1, alpha=0.7)
            ax1.set_ylabel('F0 (Hz)', fontsize=12)
            ax1.set_title(f'F0 Contour: {os.path.basename(args.f0_file)}', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, max(800, np.max(f0[f0 > 0]) * 1.1)])
            
            # UV标记
            ax2.plot(time, uv, linewidth=1, color='orange')
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Voiced/Unvoiced', fontsize=12)
            ax2.set_title('Voiced/Unvoiced Marker', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([-0.1, 1.1])
            
            plt.tight_layout()
            
            save_path = args.f0_file.replace('.f0.npy', '_analysis.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ F0 contour saved to: {save_path}")
            plt.show()
            
            # 分布图
            plot_f0_distribution(
                [f0],
                labels=[os.path.basename(args.f0_file)],
                title=f"F0 Distribution: {os.path.basename(args.f0_file)}",
                save_path=args.f0_file.replace('.f0.npy', '_distribution.png')
            )
    
    else:
        # 分析整个数据集
        if args.stats_only:
            # 只分析预计算的F0统计四元组
            analyze_f0_stats_quartet(args.dataset_dir, by_speaker=args.by_speaker)
        else:
            # 完整分析：F0序列 + 统计四元组
            analyze_dataset(args.dataset_dir, by_speaker=args.by_speaker)
            print("\n" + "="*70)
            print("Now analyzing F0 statistics quartet...")
            print("="*70)
            analyze_f0_stats_quartet(args.dataset_dir, by_speaker=args.by_speaker)


if __name__ == "__main__":
    main()

