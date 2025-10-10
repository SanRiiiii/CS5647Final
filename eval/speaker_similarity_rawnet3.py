# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa

from espnet2.bin.spk_inference import Speech2Embedding

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_RAWNET_TAG = "espnet/voxcelebs12_rawnet3"  # 官方推荐Tag :contentReference[oaicite:5]{index=5}

class RawNet3Embedder:
    def __init__(self, model_tag: str = _RAWNET_TAG, device: str = _DEVICE):
        # 下载并加载预训练 RawNet-3 嵌入器
        self.model = Speech2Embedding.from_pretrained(model_tag=model_tag, device=device)

    @staticmethod
    def _load_wav_16k(path: str) -> np.ndarray:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        return wav

    def embed(self, wav_path: str) -> torch.Tensor:
        wav = self._load_wav_16k(wav_path)
        with torch.no_grad():
            emb = self.model(np.asarray(wav))
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
            emb = emb.squeeze().float().cpu()
            emb = F.normalize(emb, dim=0)  # 归一化后做余弦
            return emb

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item())
