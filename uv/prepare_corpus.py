"""Build the molecular corpus for the generative model and label it.

Corpus: ZINC250k (Gomez-Bombarelli / Aspuru-Guzik chemical_vae release), a standard
open set of drug-like molecules. Verified download (2026-11-16):
  https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/
  models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv  -> HTTP 200.

For each kept molecule the script stores:
  - canonical SMILES
  - SELFIES encoding
  - abs_pred: predicted first absorption maximum in nm from the step-2 predictor
  - passes_screen: True when the molecule matches none of the flagged structural
    families in smarts_rules.py (the screen is a metric, never a target)

Band membership is derived later at training time from abs_pred, so the band can be
chosen without re-running the predictor.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import joblib
import selfies as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uv.common import canonical_smiles, morgan_fingerprint, passes_screen  # noqa: E402

ZINC_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)


def _fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_smiles(local, n):
    if local and os.path.exists(local):
        print(f"reading local corpus: {local}", flush=True)
        df = pd.read_csv(local)
    else:
        print(f"downloading ZINC250k: {ZINC_URL}", flush=True)
        df = pd.read_csv(io.BytesIO(_fetch(ZINC_URL)))
        print(f"download ok: {len(df)} rows", flush=True)
    smiles = [s.strip() for s in df["smiles"].tolist()]
    if n and len(smiles) > n:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(smiles), size=n, replace=False)
        smiles = [smiles[i] for i in idx]
    return smiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="uv/data/corpus.csv")
    ap.add_argument("--predictor", default="uv/artifacts/uvvis_predictor.joblib")
    ap.add_argument("--local", default=None)
    ap.add_argument("--n", type=int, default=20000, help="corpus size (0 = all)")
    args = ap.parse_args()

    bundle = joblib.load(args.predictor)
    model, n_bits = bundle["model"], bundle["n_bits"]

    smiles = load_smiles(args.local, args.n)
    print(f"processing {len(smiles)} molecules", flush=True)

    rows = []
    bad_parse = bad_selfies = 0
    for i, smi in enumerate(smiles):
        canon = canonical_smiles(smi)
        if canon is None:
            bad_parse += 1
            continue
        try:
            enc = sf.encoder(canon)
        except Exception:
            bad_selfies += 1
            continue
        fp = morgan_fingerprint(canon, n_bits=n_bits)
        if fp is None:
            bad_parse += 1
            continue
        abs_pred = float(model.predict(fp.reshape(1, -1))[0])
        clean = passes_screen(canon)
        rows.append((canon, enc, abs_pred, bool(clean)))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(smiles)} done", flush=True)

    out = pd.DataFrame(rows, columns=["smiles", "selfies", "abs_pred", "passes_screen"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"kept {len(out)} molecules "
          f"(unparsable {bad_parse}, selfies-fail {bad_selfies})", flush=True)
    print(f"passes_screen True: {int(out.passes_screen.sum())} "
          f"({100 * out.passes_screen.mean():.1f}%)", flush=True)
    print(f"abs_pred median {out.abs_pred.median():.1f} nm", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
