"""Relabel the fixed ZINC250k corpus with the second dataset's surrogate.

The molecules, their order, the SELFIES strings and the structural-screen flags are
copied unchanged from the main run's corpus. Only the abs_pred column is replaced by the
prediction of the surrogate trained on the second dataset. This is what keeps the plain
arm of the main run valid for this experiment: the plain arm never reads abs_pred, so its
training data are byte-identical in both experiments. The script asserts the invariants.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from uv.common import morgan_fingerprint  # noqa: E402


def relabel(df: pd.DataFrame, model, n_bits: int, chunk: int = 20000) -> pd.DataFrame:
    preds = np.full(len(df), np.nan, dtype=float)
    smiles = df["smiles"].tolist()
    for start in range(0, len(smiles), chunk):
        idx, feats = [], []
        for i in range(start, min(start + chunk, len(smiles))):
            fp = morgan_fingerprint(smiles[i], n_bits=n_bits)
            if fp is not None:
                idx.append(i)
                feats.append(fp)
        if feats:
            preds[idx] = model.predict(np.vstack(feats))
        print(f"  relabelled {min(start + chunk, len(smiles))}/{len(smiles)}", flush=True)
    if np.isnan(preds).any():
        raise RuntimeError(f"{int(np.isnan(preds).sum())} molecules could not be featurised")
    out = df.copy()
    out["abs_pred"] = preds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="main run corpus.csv")
    ap.add_argument("--predictor", required=True, help="surrogate trained on dataset 2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.corpus)
    need = {"smiles", "selfies", "abs_pred", "passes_screen"}
    if not need <= set(df.columns):
        raise SystemExit(f"corpus needs columns {sorted(need)}, has {list(df.columns)}")
    bundle = joblib.load(args.predictor)
    out = relabel(df, bundle["model"], bundle["n_bits"])

    # Invariants the reuse of the plain arm depends on.
    assert len(out) == len(df)
    assert out["smiles"].equals(df["smiles"]) and out["selfies"].equals(df["selfies"])
    assert out["passes_screen"].equals(df["passes_screen"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    inb = ((out.abs_pred >= 400) & (out.abs_pred <= 450)).mean()
    print(f"wrote {args.out}: {len(out)} molecules; abs_pred median {out.abs_pred.median():.1f} nm; "
          f"share predicted in 400-450 nm: {inb:.4f}; "
          f"old-label share: {((df.abs_pred >= 400) & (df.abs_pred <= 450)).mean():.4f}", flush=True)
    print("corpus sha256 (smiles+selfies columns):",
          hashlib.sha256((out.smiles + "\t" + out.selfies).str.cat(sep="\n").encode()).hexdigest(),
          flush=True)


if __name__ == "__main__":
    main()
