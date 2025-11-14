#!/usr/bin/env python3
"""
Speaker Embedding Extraction Script
Extract speaker embedding features using Wespeaker model
Supports multi-threaded parallel processing
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
    Process a single audio file
    
    Args:
        args: Tuple containing file path and output path
    """
    song_path, out_dir, model_path = args
    
    try:
        # Load audio and resample to 16kHz
        y, sr = librosa.load(song_path, sr=16000)
        
        # Create temporary file
        temp_path = f"./tmp/{Path(song_path).name}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        sf.write(temp_path, y, 16000)
        
        # Load model and extract embedding
        model = wespeaker.load_model(model_path)
        emb = model.extract_embedding(temp_path)
        
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        
        # Save embedding
        output_path = out_dir / f"{Path(song_path).stem}.spk.npy"
        np.save(output_path, emb)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return True
        
    except Exception as e:
        print(f"Error processing {song_path}: {e}")
        return False

def process_batch(file_args, model_path):
    """
    Process a batch of files
    
    Args:
        file_args: List of file arguments
        model_path: Model path
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
    Extract speaker embeddings in parallel
    
    Args:
        base_dir: Input directory
        result_dir: Output directory
        model_path: Model path
        num_processes: Number of processes
    """
    base_dir = Path(base_dir)
    result_dir = Path(result_dir)
    
    # Collect all audio files
    all_files = []
    for spk in base_dir.iterdir():
        if spk.is_dir():
            song_dir = base_dir / spk.name
            out_dir = result_dir / spk.name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for song in song_dir.iterdir():
                if song.suffix.lower() == '.wav':
                    # Check if embedding file already exists
                    emb_path = out_dir / f"{song.stem}.spk.npy"
                    if not emb_path.exists():
                        all_files.append((song, out_dir))
    
    print(f"Found {len(all_files)} files to process")
    
    if len(all_files) == 0:
        print("No files to process. All speaker embeddings already exist.")
        return
    
    # Shuffle file list randomly
    shuffle(all_files)
    
    # Create temporary directory
    os.makedirs("./tmp", exist_ok=True)
    
    # Parallel processing
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(all_files) / num_processes)
            end = int((i + 1) * len(all_files) / num_processes)
            file_chunk = all_files[start:end]
            if file_chunk:  # Only submit task when chunk is not empty
                tasks.append(executor.submit(process_batch, file_chunk, model_path))
        
        # Wait for all tasks to complete
        total_success = 0
        for task in tqdm.tqdm(tasks, position=0, desc="Overall progress"):
            total_success += task.result()
    
    print(f"\nProcessing completed: {total_success}/{len(all_files)} files processed successfully")

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
    
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    
    # Check model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return
    
    # Check input directory
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory {args.base_dir} does not exist")
        return
    
    print(f"Using {args.num_processes} processes")
    print(f"Input directory: {args.base_dir}")
    print(f"Output directory: {args.result_dir}")
    print(f"Model path: {args.model_path}")
    
    # Start processing
    parallel_extract_spk_embd(
        args.base_dir, 
        args.result_dir, 
        args.model_path, 
        args.num_processes
    )

if __name__ == "__main__":
    main()
