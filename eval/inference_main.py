# -*- coding: utf-8 -*-
"""
Single-wav voice conversion inference for CoMoSVC:
- One source wav (--source_wav)
- One target wav (--target_wav)
- Extract speaker embedding on the fly using Wespeaker
- Save output to result/{model_name}/

Example:
python inference_single.py \
  --source_wav "/scratch/e1553951/CS5647Final/data/man_01.wav" \
  --target_wav "/scratch/e1553951/CS5647Final/data/SSB0273.wav" \
  --model_path "/scratch/e1553951/CS5647Final/logs/model_48000_baseline.pt" \
  --config_path "/scratch/e1553951/CS5647Final/logs/config_baseline.yaml" \
  --wespeaker_model "/scratch/e1553951/CS5647Final/voxblink2_samresnet34_ft" \
  --key 0
"""

import os
import logging
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
import infer_tool
from infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# ========== 提取 speaker embedding ==========
def extract_spk_emb(wav_path, model_path):
    """使用 Wespeaker 提取单个 wav 的 speaker embedding"""
    import wespeaker
    print(f"🎧 Extracting speaker embedding from: {wav_path}")
    # 初始化模型
    model = wespeaker.load_model(model_path)
    emb = model.extract_embedding(wav_path)
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    emb = emb / np.linalg.norm(emb)
    print(f"✅ Speaker embedding extracted, shape={emb.shape}")
    return emb


# ========== 主函数 ==========
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Single-wav Voice Conversion Inference")
    parser.add_argument("--source_wav", type=str, required=True, help="Path to source wav")
    parser.add_argument("--target_wav", type=str, required=True, help="Path to target wav")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--wespeaker_model", type=str, required=True, help="Path to Wespeaker pretrained model folder")
    parser.add_argument("--key", type=int, default=0, help="Key shift in semitones")
    parser.add_argument("--clip", type=float, default=0.0)
    parser.add_argument("--total_steps", type=int, default=100)
    args = parser.parse_args()

    # 固定参数
    slice_db = -40
    wav_format = "wav"
    pad_seconds = 0.5
    clip = args.clip
    key_shift = args.key

    # 输出路径
    model_name = Path(args.model_path).stem
    result_dir = Path("result_single") / model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎵 CoMoSVC Single-Wav Inference")
    print("=" * 80)
    print(f"🗣️ Source: {args.source_wav}")
    print(f"🎯 Target: {args.target_wav}")
    print(f"📁 Model: {args.model_path}")
    print(f"⚙️ Config: {args.config_path}")
    print(f"💾 Output folder: {result_dir}")
    print("=" * 80)

    # Step 1: 提取 target embedding
    spk_emb = extract_spk_emb(args.target_wav, args.wespeaker_model)

    # Step 2: 初始化模型
    svc_model = Svc(args.model_path, args.config_path, args.total_steps)

    # Step 3: 进行推理
    infer_tool.format_wav(args.source_wav)

    print(f"\n{'=' * 80}")
    print(f"🎙️ Converting: {args.source_wav}")
    print(f"Key shift: {key_shift:+d} semitones")
    print(f"{'=' * 80}\n")

    kwarg = {
        "raw_audio_path": args.source_wav,
        "tran": key_shift,
        "slice_db": slice_db,
        "pad_seconds": pad_seconds,
        "clip_seconds": clip,
        "spk_emb": spk_emb
    }

    audio = svc_model.slice_inference(**kwarg)

    # Step 4: 保存结果
    src_name = Path(args.source_wav).stem
    tgt_name = Path(args.target_wav).stem
    output_name = f"{src_name}_to_{tgt_name}_{model_name}.wav"
    res_path = result_dir / output_name

    sf.write(res_path, audio, svc_model.target_sample, format=wav_format)
    print(f"✅ Saved converted audio to: {res_path}")

    svc_model.clear_empty()
    print("=" * 80)
    print("🎉 Conversion finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
