#!/usr/bin/env python3
"""
F0 Distribution Discretization Label Extractor

Functionality:
1. Compute quantile boundaries in log(1+f0/700) space from the entire dataset: p0, p20, p40, p60, p80, p100
2. Divide log space into 5 categories (0-4)
3. For each audio sample:
   - Calculate F0 quantiles at p10, p30, p50, p70, p90
   - Determine which category each of the 5 quantiles belongs to
   - Save as 5-tuple label to .wav.dist.npy file

Usage:
    python extract_f0_distribution_labels.py --dataset_dir ./dataset
    
Output:
    - Each audio file gets a .wav.dist.npy file containing 5 category labels (0-4)
    - Global quantile boundaries saved to f0_distribution_boundaries.npy
"""

import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm


def load_f0_file(f0_path):
    """Load F0 file"""
    try:
        f0_data = np.load(f0_path, allow_pickle=True)
        f0 = np.array(f0_data[0], dtype=float)
        return f0
    except Exception as e:
        print(f"Error loading {f0_path}: {e}")
        return None


def f0_to_log(f0):
    """Convert F0 to log space: log(1+f0/700)"""
    # Process only voiced segments
    f0_voiced = f0[f0 > 0]
    if len(f0_voiced) == 0:
        return np.array([])
    return np.log(1 + f0_voiced / 700)


def compute_global_boundaries(dataset_dir):
    """
    Compute global F0 log distribution boundaries
    
    Returns:
        boundaries: [p0, p20, p40, p60, p80, p100] six quantile points
    """
    print("="*70)
    print("Step 1: Computing Global F0 Distribution Boundaries")
    print("="*70)
    
    # Find all F0 files
    f0_files = glob(os.path.join(dataset_dir, "**", "*.f0.npy"), recursive=True)
    print(f"Found {len(f0_files)} F0 files")
    
    if len(f0_files) == 0:
        raise FileNotFoundError("No F0 files found!")
    
    # Collect all F0 log values
    all_f0_log = []
    
    for f0_path in tqdm(f0_files, desc="Loading F0 files"):
        f0 = load_f0_file(f0_path)
        if f0 is not None:
            f0_log = f0_to_log(f0)
            if len(f0_log) > 0:
                all_f0_log.extend(f0_log.tolist())
    
    all_f0_log = np.array(all_f0_log)
    print(f"Total voiced frames: {len(all_f0_log)}")
    
    # Compute quantiles: p0, p20, p40, p60, p80, p100
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
    Classify a log value into corresponding category
    
    Args:
        value: log(1+f0/700) value
        boundaries: [p0, p20, p40, p60, p80, p100]
    
    Returns:
        category: 0-4
    """
    # Use np.searchsorted to find the position where value should be inserted
    # boundaries is in ascending order, searchsorted(side='right') returns the first position greater than value
    category = np.searchsorted(boundaries[1:], value, side='right')
    # category range is 0-4
    return min(category, 4)  # Ensure not exceeding 4


def process_single_file(f0_path, boundaries, overwrite=False):
    """
    Process single F0 file to generate distribution labels
    
    Args:
        f0_path: F0 file path
        boundaries: Global quantile boundaries
        overwrite: Whether to overwrite existing files
    
    Returns:
        dist_label: 5-tuple label [p10_class, p30_class, p50_class, p70_class, p90_class]
    """
    # Generate output path
    # Example: ./dataset/singer/audio.wav.f0.npy -> ./dataset/singer/audio.wav.dist.npy
    dist_path = f0_path.replace('.f0.npy', '.dist.npy')
    
    # Check if already exists
    if os.path.exists(dist_path) and not overwrite:
        return None, "skipped"
    
    # Load F0
    f0 = load_f0_file(f0_path)
    if f0 is None:
        return None, "error"
    
    # Convert to log space
    f0_log = f0_to_log(f0)
    if len(f0_log) == 0:
        # No voiced segments, use default value (middle category 2)
        dist_label = np.array([2, 2, 2, 2, 2], dtype=np.int32)
        np.save(dist_path, dist_label)
        return dist_label, "empty"
    
    # Compute p10, p30, p50, p70, p90 quantiles for this audio
    sample_percentiles = [10, 30, 50, 70, 90]
    sample_values = np.percentile(f0_log, sample_percentiles)
    
    # Classify to corresponding categories
    dist_label = np.array([classify_value(val, boundaries) for val in sample_values], dtype=np.int32)
    
    # Save
    np.save(dist_path, dist_label)
    
    return dist_label, "success"


def main():
    parser = argparse.ArgumentParser(description='Extract F0 distribution labels')
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                       help='Dataset directory')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing .dist.npy files')
    
    args = parser.parse_args()
    
    # Step 1: Compute global boundaries
    boundaries = compute_global_boundaries(args.dataset_dir)
    
    # Save boundaries
    boundary_path = os.path.join(args.dataset_dir, "f0_distribution_boundaries.npy")
    np.save(boundary_path, boundaries)
    print(f"\nBoundaries saved to: {boundary_path}")
    
    # Step 2: Process individual files
    print("\n" + "="*70)
    print("Step 2: Processing Individual Files")
    print("="*70)
    
    f0_files = glob(os.path.join(args.dataset_dir, "**", "*.f0.npy"), recursive=True)
    
    for f0_path in tqdm(f0_files, desc="Generating distribution labels"):
        process_single_file(f0_path, boundaries, args.overwrite)
    
    print(f"\nDone! Distribution labels saved to .dist.npy files")


if __name__ == "__main__":
    main()

