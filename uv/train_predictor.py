"""Train a UV-Vis absorption-maximum predictor on the clean dataset.

Model choice: Morgan-fingerprint (ECFP radius 2, 2048 bits) features into a
HistGradientBoostingRegressor from scikit-learn. This is chosen over Chemprop for a
guaranteed clean install and a few-minute fit. If Chemprop is importable and
--try-chemprop is passed, the script says so and still uses the fingerprint model for
the committed result unless Chemprop training is wired in. No invented metrics.

Held-out test split (random, seed fixed). Reports MAE, RMSE and R2 on the test set.
Checkpoints the growing model to the run directory every few boosting stages and
resumes from the latest stage. Prints progress: stage, trees, elapsed, ETA, val MAE.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uv.common import morgan_fingerprint  # noqa: E402
from colab_checkpoint import CheckpointStore, Heartbeat  # noqa: E402


def featurize(smiles_list, n_bits):
    feats, keep = [], []
    for i, smi in enumerate(smiles_list):
        fp = morgan_fingerprint(smi, n_bits=n_bits)
        if fp is not None:
            feats.append(fp)
            keep.append(i)
    return np.vstack(feats), np.array(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="uv/data/uvvis_dataset.csv")
    ap.add_argument("--out-model", default="uv/artifacts/uvvis_predictor.joblib")
    ap.add_argument("--metrics", default="uv/artifacts/predictor_metrics.json")
    ap.add_argument("--run-dir", default="/content/drive/MyDrive/t25-checkpoints/uv-predictor")
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--total-trees", type=int, default=600)
    ap.add_argument("--stage-trees", type=int, default=100)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--try-chemprop", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    if args.try_chemprop:
        try:
            import chemprop  # noqa: F401
            print("chemprop import ok. This script still uses the fingerprint model "
                  "for the committed result unless Chemprop training is wired in.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"chemprop not available ({exc}); using the fingerprint model", flush=True)

    df = pd.read_csv(args.data)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=0).reset_index(drop=True)
    print(f"loaded {len(df)} structures", flush=True)

    X, keep = featurize(df["smiles"].tolist(), args.n_bits)
    y = df["abs_max_nm"].to_numpy()[keep]
    print(f"featurized {X.shape[0]} molecules into {X.shape[1]} bits", flush=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_frac, random_state=42
    )
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=7
    )
    print(f"train {len(y_tr2)} val {len(y_val)} test {len(y_te)}", flush=True)

    ck = CheckpointStore(args.run_dir, suffix=".joblib",
                         save_fn=lambda o, p: joblib.dump(o, p),
                         load_fn=lambda p: joblib.load(p))
    state = ck.latest()
    start_trees = 0
    model = None
    if state is not None:
        model = state["model"]
        start_trees = state["trees"]
        print(f"resumed from checkpoint at {start_trees} trees", flush=True)

    if model is None:
        model = HistGradientBoostingRegressor(
            max_iter=args.stage_trees, learning_rate=0.1, max_depth=None,
            l2_regularization=1.0, warm_start=True, random_state=0,
        )

    hb = Heartbeat(total_steps=args.total_trees, every=1)
    trees = start_trees
    while trees < args.total_trees:
        trees = min(trees + args.stage_trees, args.total_trees)
        model.set_params(max_iter=trees)
        model.fit(X_tr2, y_tr2)
        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        hb.tick(trees, val_mae=round(float(val_mae), 2))
        ck.save({"model": model, "trees": trees}, step=trees,
                metric=float(val_mae), lower_is_better=True)

    pred = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = float(r2_score(y_te, pred))
    print(f"TEST  MAE {mae:.2f} nm  RMSE {rmse:.2f} nm  R2 {r2:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump({"model": model, "n_bits": args.n_bits}, args.out_model)
    import json
    with open(args.metrics, "w") as fh:
        json.dump({"model": "morgan_ecfp2048_histgbm", "n_train": int(len(y_tr2)),
                   "n_test": int(len(y_te)), "mae_nm": mae, "rmse_nm": rmse, "r2": r2,
                   "total_trees": trees}, fh, indent=2)
    print(f"wrote {args.out_model} and {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
