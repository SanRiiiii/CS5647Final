# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import soundfile as sf
import librosa
from huggingface_hub import hf_hub_download

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UTMOS:
    """
    使用 UTMOS（2022 优胜系统）JIT 模型进行 MOS 预测。
    模型来自 Hugging Face（balacoon/utmos 的演示权重示例）。:contentReference[oaicite:6]{index=6}
    """
    def __init__(self, repo_id: str = "balacoon/utmos", filename: str = "utmos.jit"):
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", local_dir="./.utmos")
        self.model = torch.jit.load(model_path, map_location=_DEVICE).eval()

    @staticmethod
    def _to_int16_16k_mono(path: str) -> np.ndarray:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767.0).astype("int16")

    @torch.no_grad()
    def predict(self, wav_path: str) -> float:
        x = self._to_int16_16k_mono(wav_path)
        x = torch.from_numpy(x).unsqueeze(0).to(_DEVICE)
        mos = self.model(x).item()
        return float(mos)

# （可选）SSL-MOS：建议直接调用官方仓库脚本，避免把 fairseq 训练逻辑耦合进来。:contentReference[oaicite:7]{index=7}
# 你也可以写一个 wrapper 用 subprocess 调 run_inference.py，对指定 wav 目录输出 answers.txt。
