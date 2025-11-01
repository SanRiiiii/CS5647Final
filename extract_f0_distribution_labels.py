#!/usr/bin/env python3
"""
F0分布离散化标签提取器

功能：
1. 从整个数据集计算log(1+f0/700)空间的分位数边界：p0, p20, p40, p60, p80, p100
2. 将log空间划分为5个类别（0-4）
3. 对每个音频样本：
   - 计算其F0的p10, p30, p50, p70, p90分位数
   - 判断这5个分位数各自属于哪个类别
   - 保存为5元组标签到 .wav.dist.npy 文件

使用方法：
    python extract_f0_distribution_labels.py --dataset_dir ./dataset
    
输出：
    - 每个音频对应一个 .wav.dist.npy 文件，包含5个类别标签（0-4）
    - 全局分位数边界保存在 f0_distribution_boundaries.npy
"""

import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_f0_file(f0_path):
    """加载F0文件"""
    try:
        f0_data = np.load(f0_path, allow_pickle=True)
        f0 = np.array(f0_data[0], dtype=float)
        return f0
    except Exception as e:
        print(f"Error loading {f0_path}: {e}")
        return None


def f0_to_log(f0):
    """F0转换到log空间：log(1+f0/700)"""
    # 只处理有声段
    f0_voiced = f0[f0 > 0]
    if len(f0_voiced) == 0:
        return np.array([])
    return np.log(1 + f0_voiced / 700)


def compute_global_boundaries(dataset_dir):
    """
    计算全局F0 log分布的边界
    
    Returns:
        boundaries: [p0, p20, p40, p60, p80, p100] 六个分位点
    """
    print("="*70)
    print("Step 1: Computing Global F0 Distribution Boundaries")
    print("="*70)
    
    # 查找所有F0文件
    f0_files = glob(os.path.join(dataset_dir, "**", "*.f0.npy"), recursive=True)
    print(f"Found {len(f0_files)} F0 files")
    
    if len(f0_files) == 0:
        raise FileNotFoundError("No F0 files found!")
    
    # 收集所有F0的log值
    all_f0_log = []
    
    for f0_path in tqdm(f0_files, desc="Loading F0 files"):
        f0 = load_f0_file(f0_path)
        if f0 is not None:
            f0_log = f0_to_log(f0)
            if len(f0_log) > 0:
                all_f0_log.extend(f0_log.tolist())
    
    all_f0_log = np.array(all_f0_log)
    print(f"Total voiced frames: {len(all_f0_log)}")
    
    # 计算分位数: p0, p20, p40, p60, p80, p100
    percentiles = [0, 20, 40, 60, 80, 100]
    boundaries = np.percentile(all_f0_log, percentiles)
    
    print(f"\nGlobal F0 Log Distribution Boundaries:")
    print(f"{'Percentile':>12s} | {'Log Value':>10s} | {'F0 (Hz)':>10s} | {'Category':>10s}")
    print("-" * 60)
    
    for i, (perc, bound) in enumerate(zip(percentiles, boundaries)):
        f0_hz = 700 * (np.exp(bound) - 1)
        if i < len(percentiles) - 1:
            category = f"Class {i}"
        else:
            category = "-"
        print(f"  p{perc:3d}       | {bound:10.4f} | {f0_hz:10.2f} | {category:>10s}")
    
    print(f"\nCategory Ranges (Log Space):")
    for i in range(len(boundaries) - 1):
        print(f"  Class {i}: [{boundaries[i]:.4f}, {boundaries[i+1]:.4f})")
        f0_low = 700 * (np.exp(boundaries[i]) - 1)
        f0_high = 700 * (np.exp(boundaries[i+1]) - 1)
        print(f"           [{f0_low:.2f} Hz, {f0_high:.2f} Hz)")
    
    return boundaries


def classify_value(value, boundaries):
    """
    将一个log值分类到对应的类别
    
    Args:
        value: log(1+f0/700)值
        boundaries: [p0, p20, p40, p60, p80, p100]
    
    Returns:
        category: 0-4
    """
    # 使用np.searchsorted找到value应该插入的位置
    # boundaries是升序的，searchsorted(side='right')返回第一个大于value的位置
    category = np.searchsorted(boundaries[1:], value, side='right')
    # category范围是0-4
    return min(category, 4)  # 确保不超过4


def process_single_file(f0_path, boundaries, overwrite=False):
    """
    处理单个F0文件，生成分布标签
    
    Args:
        f0_path: F0文件路径
        boundaries: 全局分位数边界
        overwrite: 是否覆盖已存在的文件
    
    Returns:
        dist_label: 5元组标签 [p10_class, p30_class, p50_class, p70_class, p90_class]
    """
    # 生成输出路径
    # 例如: ./dataset/singer/audio.wav.f0.npy -> ./dataset/singer/audio.wav.dist.npy
    dist_path = f0_path.replace('.f0.npy', '.dist.npy')
    
    # 检查是否已存在
    if os.path.exists(dist_path) and not overwrite:
        return None, "skipped"
    
    # 加载F0
    f0 = load_f0_file(f0_path)
    if f0 is None:
        return None, "error"
    
    # 转换到log空间
    f0_log = f0_to_log(f0)
    if len(f0_log) == 0:
        # 没有有声段，使用默认值（中间类别2）
        dist_label = np.array([2, 2, 2, 2, 2], dtype=np.int32)
        np.save(dist_path, dist_label)
        return dist_label, "empty"
    
    # 计算该音频的p10, p30, p50, p70, p90分位数
    sample_percentiles = [10, 30, 50, 70, 90]
    sample_values = np.percentile(f0_log, sample_percentiles)
    
    # 分类到对应的类别
    dist_label = np.array([classify_value(val, boundaries) for val in sample_values], dtype=np.int32)
    
    # 保存
    np.save(dist_path, dist_label)
    
    return dist_label, "success"


def visualize_distribution(boundaries, dataset_dir):
    """可视化分布边界和数据"""
    print("\n" + "="*70)
    print("Step 3: Visualizing Distribution")
    print("="*70)
    
    # 采样一些文件进行可视化
    f0_files = glob(os.path.join(dataset_dir, "**", "*.f0.npy"), recursive=True)
    sample_size = min(100, len(f0_files))
    sample_files = np.random.choice(f0_files, sample_size, replace=False)
    
    # 收集样本数据
    sample_f0_logs = []
    sample_labels = []
    
    for f0_path in tqdm(sample_files, desc="Sampling for visualization"):
        f0 = load_f0_file(f0_path)
        if f0 is not None:
            f0_log = f0_to_log(f0)
            if len(f0_log) > 0:
                sample_f0_logs.append(f0_log)
                
                # 读取对应的标签
                dist_path = f0_path.replace('.f0.npy', '.dist.npy')
                if os.path.exists(dist_path):
                    label = np.load(dist_path)
                    sample_labels.append(label)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('F0 Distribution Classification Visualization', fontsize=16, fontweight='bold')
    
    # 1. 全局分布 + 边界
    ax1 = axes[0, 0]
    all_f0_log = np.concatenate(sample_f0_logs)
    ax1.hist(all_f0_log, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 绘制边界线
    colors_boundary = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i, (bound, color) in enumerate(zip(boundaries, colors_boundary)):
        ax1.axvline(bound, color=color, linestyle='--', linewidth=2, 
                   label=f'p{[0,20,40,60,80,100][i]}={bound:.3f}')
    
    ax1.set_xlabel('log(1+f0/700)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Global F0 Log Distribution with Boundaries', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. 类别分布（5元组标签统计）
    ax2 = axes[0, 1]
    if len(sample_labels) > 0:
        sample_labels = np.array(sample_labels)
        positions = ['p10', 'p30', 'p50', 'p70', 'p90']
        
        # 计算每个位置的类别分布
        for pos_idx, pos_name in enumerate(positions):
            class_counts = np.bincount(sample_labels[:, pos_idx], minlength=5)
            ax2.bar(np.arange(5) + pos_idx*0.15, class_counts, width=0.15, 
                   label=pos_name, alpha=0.8)
        
        ax2.set_xlabel('Category', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution Label Statistics', fontsize=13, fontweight='bold')
        ax2.set_xticks([0, 1, 2, 3, 4])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 热力图：5元组标签模式
    ax3 = axes[1, 0]
    if len(sample_labels) > 0:
        # 统计最常见的5元组模式
        label_patterns = {}
        for label in sample_labels:
            pattern = tuple(label)
            label_patterns[pattern] = label_patterns.get(pattern, 0) + 1
        
        # 取前20个最常见的模式
        top_patterns = sorted(label_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if len(top_patterns) > 0:
            patterns_array = np.array([list(p[0]) for p in top_patterns])
            counts = [p[1] for p in top_patterns]
            
            im = ax3.imshow(patterns_array.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax3.set_yticks([0, 1, 2, 3, 4])
            ax3.set_yticklabels(['p10', 'p30', 'p50', 'p70', 'p90'])
            ax3.set_xlabel('Pattern Rank', fontsize=12)
            ax3.set_title('Top 20 Distribution Patterns', fontsize=13, fontweight='bold')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label('Category', fontsize=10)
            
            # 在每个格子上标注类别数字
            for i in range(patterns_array.shape[0]):
                for j in range(patterns_array.shape[1]):
                    text = ax3.text(i, j, int(patterns_array[i, j]),
                                   ha="center", va="center", color="black", fontsize=8)
    
    # 4. 类别转换矩阵（相邻分位数的类别变化）
    ax4 = axes[1, 1]
    if len(sample_labels) > 0:
        # 统计从p10->p30, p30->p50, p50->p70, p70->p90的转换
        transition_matrix = np.zeros((5, 5), dtype=int)
        
        for label in sample_labels:
            for i in range(len(label) - 1):
                from_class = label[i]
                to_class = label[i + 1]
                transition_matrix[from_class, to_class] += 1
        
        im = ax4.imshow(transition_matrix, cmap='Blues', interpolation='nearest')
        ax4.set_xlabel('To Category', fontsize=12)
        ax4.set_ylabel('From Category', fontsize=12)
        ax4.set_title('Category Transition Matrix', fontsize=13, fontweight='bold')
        ax4.set_xticks([0, 1, 2, 3, 4])
        ax4.set_yticks([0, 1, 2, 3, 4])
        
        # 添加数值标注
        for i in range(5):
            for j in range(5):
                text = ax4.text(j, i, transition_matrix[i, j],
                               ha="center", va="center", 
                               color="white" if transition_matrix[i, j] > transition_matrix.max()/2 else "black",
                               fontsize=10)
        
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    
    save_path = os.path.join(dataset_dir, "f0_distribution_classification.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Extract F0 distribution labels')
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                       help='Dataset directory')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing .dist.npy files')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Visualize distribution (default: True)')
    
    args = parser.parse_args()
    
    # Step 1: 计算全局边界
    boundaries = compute_global_boundaries(args.dataset_dir)
    
    # 保存边界
    boundary_path = os.path.join(args.dataset_dir, "f0_distribution_boundaries.npy")
    np.save(boundary_path, boundaries)
    print(f"\n✅ Boundaries saved to: {boundary_path}")
    
    # Step 2: 处理每个文件
    print("\n" + "="*70)
    print("Step 2: Processing Individual Files")
    print("="*70)
    
    f0_files = glob(os.path.join(args.dataset_dir, "**", "*.f0.npy"), recursive=True)
    
    success_count = 0
    skipped_count = 0
    error_count = 0
    empty_count = 0
    
    # 统计标签分布
    label_stats = {i: 0 for i in range(5)}
    
    for f0_path in tqdm(f0_files, desc="Generating distribution labels"):
        dist_label, status = process_single_file(f0_path, boundaries, args.overwrite)
        
        if status == "success":
            success_count += 1
            # 统计标签
            for label in dist_label:
                label_stats[label] += 1
        elif status == "skipped":
            skipped_count += 1
        elif status == "empty":
            empty_count += 1
            # 空文件使用默认标签2
            for _ in range(5):
                label_stats[2] += 1
        else:
            error_count += 1
    
    # 打印统计
    print(f"\n{'='*70}")
    print(f"Processing Complete")
    print(f"{'='*70}")
    print(f"  Processed:  {success_count} files")
    print(f"  Empty:      {empty_count} files (no voiced frames)")
    print(f"  Skipped:    {skipped_count} files")
    print(f"  Errors:     {error_count} files")
    print(f"  Total:      {len(f0_files)} files")
    
    print(f"\n{'='*70}")
    print(f"Label Distribution Statistics")
    print(f"{'='*70}")
    total_labels = sum(label_stats.values())
    for category in range(5):
        count = label_stats[category]
        percentage = count / total_labels * 100 if total_labels > 0 else 0
        print(f"  Category {category}: {count:6d} ({percentage:5.1f}%)")
    
    # Step 3: 可视化
    if args.visualize:
        visualize_distribution(boundaries, args.dataset_dir)
    
    print(f"\n✅ Done! Distribution labels saved to .dist.npy files")


if __name__ == "__main__":
    main()

