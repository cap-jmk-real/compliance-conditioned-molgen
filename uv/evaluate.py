"""Evaluate the two generator arms on the same sample count.

For each arm the script samples a fixed number of molecules and computes:
  - validity: fraction of samples that decode to a valid RDKit molecule
  - uniqueness: unique valid structures / valid samples
  - novelty: fraction of unique valid structures not in the training corpus
  - screen_pass_rate: fraction of unique valid structures that match none of the
    flagged structural families in smarts_rules.py
  - property score: mean predicted absorption maximum (nm) over unique valid
    structures, and the fraction whose predicted maximum falls in the target band

The steered arm is sampled with the band and screen control tokens set to the target
(BAND_IN, CLEAN). The plain arm is sampled unconditionally. Writes a results JSON and
a markdown table. All numbers come from the actual samples. Nothing is invented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import joblib
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uv.common import canonical_smiles, morgan_fingerprint, passes_screen, in_band  # noqa: E402
from uv import gen_model  # noqa: E402
import selfies as sf  # noqa: E402


def decode_valid(selfies_list):
    valid = []
    for s in selfies_list:
        if not s:
            continue
        try:
            smi = sf.decoder(s)
        except Exception:
            continue
        canon = canonical_smiles(smi)
        if canon is not None:
            valid.append(canon)
    return valid


def eval_arm(arm_path, vocab, device, n_samples, train_set, predictor, n_bits,
             band_lo, band_hi, seed):
    bundle = torch.load(arm_path, map_location=device, weights_only=False)
    model = gen_model.SelfiesLM(len(vocab), bundle["emb"], bundle["hidden"],
                                bundle["layers"]).to(device)
    model.load_state_dict(bundle["model"])
    steered = bundle["arm"] == "steered"
    prefix = [gen_model.BAND_IN, gen_model.CLEAN] if steered else None
    samples = gen_model.sample(model, vocab, device, n_samples, steered,
                               prefix_tokens=prefix, seed=seed)
    valid = decode_valid(samples)
    validity = len(valid) / max(len(samples), 1)
    unique = sorted(set(valid))
    uniqueness = len(unique) / max(len(valid), 1)
    novel = [s for s in unique if s not in train_set]
    novelty = len(novel) / max(len(unique), 1)

    screen_ok = 0
    preds = []
    for s in unique:
        if passes_screen(s):
            screen_ok += 1
        fp = morgan_fingerprint(s, n_bits=n_bits)
        if fp is not None:
            preds.append(float(predictor.predict(fp.reshape(1, -1))[0]))
    screen_rate = screen_ok / max(len(unique), 1)
    preds = np.array(preds) if preds else np.array([np.nan])
    in_band_frac = float(np.mean([in_band(p, band_lo, band_hi) for p in preds])) if len(preds) else 0.0

    return {
        "arm": bundle["arm"],
        "n_sampled": len(samples),
        "n_valid": len(valid),
        "n_unique": len(unique),
        "validity": round(validity, 4),
        "uniqueness": round(uniqueness, 4),
        "novelty": round(novelty, 4),
        "screen_pass_rate": round(screen_rate, 4),
        "mean_pred_abs_nm": round(float(np.nanmean(preds)), 2),
        "frac_in_band": round(in_band_frac, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="uv/data/corpus.csv")
    ap.add_argument("--vocab", default="uv/artifacts/vocab.json")
    ap.add_argument("--plain", default="uv/artifacts/generator_plain.pt")
    ap.add_argument("--steered", default="uv/artifacts/generator_steered.pt")
    ap.add_argument("--predictor", default="uv/artifacts/uvvis_predictor.joblib")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--band-lo", type=float, default=450.0)
    ap.add_argument("--band-hi", type=float, default=550.0)
    ap.add_argument("--out-json", default="uv/artifacts/eval_results.json")
    ap.add_argument("--out-md", default="uv/artifacts/eval_results.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = gen_model.Vocab.load(args.vocab)
    df = pd.read_csv(args.corpus)
    train_set = set(df["smiles"].tolist())
    pb = joblib.load(args.predictor)
    predictor, n_bits = pb["model"], pb["n_bits"]

    print(f"evaluating on {args.n_samples} samples per arm, band "
          f"{args.band_lo:.0f}-{args.band_hi:.0f} nm", flush=True)

    results = {}
    for name, path in [("plain", args.plain), ("steered", args.steered)]:
        print(f"sampling arm: {name}", flush=True)
        results[name] = eval_arm(path, vocab, device, args.n_samples, train_set,
                                 predictor, n_bits, args.band_lo, args.band_hi, args.seed)
        print(f"  {results[name]}", flush=True)

    meta = {"band_lo": args.band_lo, "band_hi": args.band_hi,
            "n_samples": args.n_samples, "arms": results}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(meta, fh, indent=2)

    cols = ["validity", "uniqueness", "novelty", "screen_pass_rate",
            "mean_pred_abs_nm", "frac_in_band"]
    lines = [f"# UV-Vis steered vs plain generation",
             "",
             f"Target band: {args.band_lo:.0f} to {args.band_hi:.0f} nm. "
             f"Samples per arm: {args.n_samples}.",
             "",
             "| metric | plain | steered |",
             "|---|---|---|"]
    for c in cols:
        lines.append(f"| {c} | {results['plain'][c]} | {results['steered'][c]} |")
    with open(args.out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"wrote {args.out_json} and {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
