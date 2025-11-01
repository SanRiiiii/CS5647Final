# -*- coding: utf-8 -*-
"""
验证 UTMOS 神经 MOS 预测器是否可用。
运行:
    python -m metrics.verify_utmos --wav your_audio.wav
或者直接运行：
    python metrics/verify_utmos.py --wav your_audio.wav
"""
import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
from huggingface_hub import hf_hub_download


def load_wav_16k_int16(path: str) -> np.ndarray:
    """加载音频，转换为16kHz单声道int16格式"""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype("int16")


def load_utmos_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    从 Hugging Face 加载 UTMOS 模型（VoiceMOS Challenge 2022 优胜系统）。
    如果下载失败，可尝试指定 proxy 或使用离线缓存。
    """
    print("[UTMOS] Loading pretrained model from Hugging Face ...")
    model_path = hf_hub_download(
        repo_id="balacoon/utmos",
        filename="utmos.jit",
        repo_type="model",
        local_dir="./.utmos"
    )
    model = torch.jit.load(model_path, map_location=device).eval()
    print("[UTMOS] Model loaded successfully.")
    return model


def predict_mos(model, wav_path: str, device="cuda" if torch.cuda.is_available() else "cpu") -> float:
    wav = load_wav_16k_int16(wav_path)
    x = torch.from_numpy(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(x).item()
    return float(score)


def main():
    parser = argparse.ArgumentParser(description="Verify UTMOS Neural MOS Predictor")
    parser.add_argument("--wav", type=str, required=True, help="Path to input wav file (16kHz preferred)")
    args = parser.parse_args()

    if not os.path.exists(args.wav):
        print(f"[Error] File not found: {args.wav}")
        return

    model = load_utmos_model()
    mos = predict_mos(model, args.wav)
    print(f"\n✅ [Result] MOS prediction for {os.path.basename(args.wav)}: {mos:.4f}")


if __name__ == "__main__":
    main()
