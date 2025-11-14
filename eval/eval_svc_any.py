# -*- coding: utf-8 -*-
"""
Evaluate SVC results:
✅ 支持模式：
  1️⃣ 单条 (gen.wav vs tgt.wav)
  2️⃣ 单模型文件夹 (自动匹配 man/woman)
  3️⃣ 多模型文件夹 (批量汇总)
✅ 输出：
  - 每条详细评分 CSV
  - 平均得分 summary.json
"""

import os, re, argparse, json, subprocess, sys
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from speechbrain.pretrained import EncoderClassifier


# ---------- Auto install utmosv2 ----------
try:
    import utmosv2
except ImportError:
    print("[AutoInstall] Installing utmosv2 from GitHub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/sarulab-speech/UTMOSv2.git"])
    import utmosv2


# ---------- ECAPA embedder ----------
class ECAPAEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Loading SpeechBrain ECAPA on {self.device} ...")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        self.model.eval()

    @torch.no_grad()
    def embed(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.to(self.device)
        emb = self.model.encode_batch(wav)
        return emb.squeeze(0).detach().cpu().numpy()


# ---------- UTMOSv2 scorer ----------
class UTMOSv2Scorer:
    def __init__(self):
        print("[Init] Loading UTMOSv2 model ...")
        self.model = utmosv2.create_model(pretrained=True)

    def mos(self, wav_path):
        try:
            return self.model.predict(input_path=wav_path)
        except Exception as e:
            print(f"[UTMOSv2 Error] {wav_path}: {e}")
            return None


# ---------- Helper ----------
def cosine(a, b):
    a = torch.tensor(a).unsqueeze(0)
    b = torch.tensor(b).unsqueeze(0)
    return float(cosine_similarity(a, b, dim=1).mean().item())


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """计算均值及置信区间"""
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.array(values)
    boots = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    boots = np.sort(boots)
    lower = boots[int((alpha/2)*n_boot)]
    upper = boots[int((1-alpha/2)*n_boot)]
    return (np.mean(vals), lower, upper)


def list_wavs(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav")]


def find_target_wav(base_root, spk_id):
    """
    递归搜索目标说话人文件夹；
    优先匹配 {spk_id}0001.wav，如果不存在则退回第一个文件。
    """
    for gender in ["man", "woman"]:
        spk_dir = os.path.join(base_root, gender, spk_id)
        if os.path.isdir(spk_dir):
            wavs = sorted(list_wavs(spk_dir))
            # ✅ 优先匹配 0001 文件
            for w in wavs:
                if re.search(rf"{spk_id}0001\.wav$", os.path.basename(w)):
                    return w
            # fallback：取第一个
            if wavs:
                return wavs[0]

    # fallback: 直接在根目录找
    direct_dir = os.path.join(base_root, spk_id)
    if os.path.isdir(direct_dir):
        wavs = sorted(list_wavs(direct_dir))
        for w in wavs:
            if re.search(rf"{spk_id}0001\.wav$", os.path.basename(w)):
                return w
        if wavs:
            return wavs[0]
    return None


# ---------- Core Evaluation ----------
def evaluate_model_folder(gen_dir, tgt_root, out_dir, embedder, mos_scorer):
    """评估单个模型文件夹"""
    model_name = os.path.basename(gen_dir.rstrip("/"))
    out_csv = os.path.join(out_dir, f"{model_name}_detailed.csv")
    out_summary = os.path.join(out_dir, f"{model_name}_summary.json")
    print(f"\n📂 Evaluating model folder: {model_name}")

    wavs = list_wavs(gen_dir)
    if not wavs:
        print("❌ No wav files found.")
        return

    records = []
    for wav in tqdm(wavs, desc=f"Evaluating {model_name}"):
        m = re.search(r"SSB\d{4}", os.path.basename(wav))
        if not m:
            continue
        spk_id = m.group(0)
        tgt_path = find_target_wav(tgt_root, spk_id)
        if not tgt_path or not os.path.exists(tgt_path):
            print(f"⚠️  Target not found for {spk_id}")
            continue

        g_emb = embedder.embed(wav)
        t_emb = embedder.embed(tgt_path)
        sim = cosine(g_emb, t_emb)
        mos = mos_scorer.mos(wav)
        records.append({
            "model": model_name,
            "speaker": spk_id,
            "gen_wav": wav,
            "tgt_wav": tgt_path,
            "cosine_sim": sim,
            "utmosv2": mos
        })

    if not records:
        print("❌ No valid pairs found.")
        return

    # Save per-item scores
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)

    # Compute average
    summary = {
        "model": model_name,
        "N_items": len(df),
        "cosine": bootstrap_ci(df["cosine_sim"]),
        "utmosv2": bootstrap_ci(df["utmosv2"]),
    }
    json.dump(summary, open(out_summary, "w"), indent=2)

    print(f"\n✅ Saved per-file scores → {out_csv}")
    print(f"📊 Summary → {out_summary}")
    print(f"   Avg Cosine: {summary['cosine'][0]:.4f}")
    print(f"   Avg UTMOSv2: {summary['utmosv2'][0]:.3f}")
    return df, summary


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", required=True, help="Generated wav file/folder/root")
    ap.add_argument("--tgt_root", required=True, help="Target wav base folder (contains man/woman)")
    ap.add_argument("--out_dir", default="./eval_result")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    embedder = ECAPAEmbedder()
    mos_scorer = UTMOSv2Scorer()

    # 单条模式
    if args.gen_root.lower().endswith(".wav") and args.tgt_root.lower().endswith(".wav"):
        print("\n🎧 Single-file evaluation mode")
        g_emb = embedder.embed(args.gen_root)
        t_emb = embedder.embed(args.tgt_root)
        sim = cosine(g_emb, t_emb)
        mos = mos_scorer.mos(args.gen_root)
        result = {"gen": args.gen_root, "tgt": args.tgt_root, "cosine": sim, "utmosv2": mos}
        out_path = os.path.join(args.out_dir, "single_eval.json")
        json.dump(result, open(out_path, "w"), indent=2)
        print(f"✅ Single eval done → {out_path}")
        return

    # 模型文件夹或多个模型根目录
    if os.path.isdir(args.gen_root):
        subdirs = [os.path.join(args.gen_root, d) for d in os.listdir(args.gen_root)
                   if os.path.isdir(os.path.join(args.gen_root, d)) and d.startswith("model_")]
        # 若传入的是单模型文件夹
        if not subdirs:
            evaluate_model_folder(args.gen_root, args.tgt_root, args.out_dir, embedder, mos_scorer)
        else:
            # 多模型批量模式
            all_summary = []
            for model_dir in subdirs:
                _, summary = evaluate_model_folder(model_dir, args.tgt_root, args.out_dir, embedder, mos_scorer)
                all_summary.append(summary)
            # 汇总输出
            summary_csv = os.path.join(args.out_dir, "all_models_summary.csv")
            pd.DataFrame(all_summary).to_csv(summary_csv, index=False)
            print(f"\n✅ Combined summary saved → {summary_csv}")
        return

    print("❌ Input not recognized (neither wav nor model folder).")


if __name__ == "__main__":
    main()
