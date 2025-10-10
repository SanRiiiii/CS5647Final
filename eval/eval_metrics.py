# -*- coding: utf-8 -*-
"""
用法：
python -m metrics.eval_metrics \
  --pairs pairs.csv \
  --out results_metrics.csv \
  --mos_backend utmos
CSV 格式：gen_wav,ref_wav （每行一对）

"""
import csv
import argparse
from pathlib import Path

from metrics.speaker_similarity_rawnet3 import RawNet3Embedder, cosine_similarity
from metrics.neural_mos import UTMOS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True, help="CSV with columns: gen_wav,ref_wav")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mos_backend", type=str, default="utmos", choices=["utmos", "none"])
    args = ap.parse_args()

    speaker = RawNet3Embedder()
    mos_model = UTMOS() if args.mos_backend == "utmos" else None

    out_rows = [("gen_wav","ref_wav","D_embed","MOS")]
    with open(args.pairs, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen = str(Path(row["gen_wav"]))
            ref = str(Path(row["ref_wav"]))
            # speaker similarity
            e_gen = speaker.embed(gen)
            e_ref = speaker.embed(ref)
            d_embed = cosine_similarity(e_gen, e_ref)
            # mos
            mos = mos_model.predict(gen) if mos_model else ""
            out_rows.append((gen, ref, f"{d_embed:.6f}", f"{mos:.4f}" if mos!="" else ""))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(out_rows)
    print(f"[OK] Wrote {args.out} with {len(out_rows)-1} rows.")

if __name__ == "__main__":
    main()
 