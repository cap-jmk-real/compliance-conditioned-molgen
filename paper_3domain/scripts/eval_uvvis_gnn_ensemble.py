#!/usr/bin/env python3
"""Evaluation-only: compute 4-GNN ensemble test metrics when checkpoints are present (no retraining)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "uv/data/uvvis_dataset.csv"))
    ap.add_argument(
        "--ckpt-dir",
        default=str(ROOT / "uv/results/round3_gnn/checkpoints"),
        help="Directory with gine_uvvis.pt, gatv2_uvvis.pt, mpnn_uvvis.pt, carate_uvvis.pt",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "uv/results/round3_gnn/ensemble_uvvis_test_metrics.json"),
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from uv.gnn_predictor import build_gnn_from_ckpt, mol_to_graph, predict_smiles

    df = pd.read_csv(args.data)
    graphs, smiles_valid, y_vals = [], [], []
    for smi, y in zip(df.smiles, df.abs_max_nm):
        if mol_to_graph(str(smi)) is not None and np.isfinite(y):
            smiles_valid.append(str(smi))
            y_vals.append(float(y))
    y_arr = np.asarray(y_vals, dtype=np.float32)
    idx = np.arange(len(smiles_valid))
    tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=42)
    smiles_te = [smiles_valid[i] for i in te_idx]

    ckpt_dir = Path(args.ckpt_dir)
    archs = ["gine", "gatv2", "mpnn", "carate"]
    ckpts = [ckpt_dir / f"{a}_uvvis.pt" for a in archs]
    missing = [str(p) for p in ckpts if not p.is_file()]
    if missing:
        raise SystemExit(
            "Missing checkpoints (train with uv/gnn_predictor.py or copy GPU checkpoints):\n  "
            + "\n  ".join(missing)
        )

    dev = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    preds = []
    for p in ckpts:
        model, ym, ys = build_gnn_from_ckpt(p, dev)
        preds.append(predict_smiles(model, smiles_te, dev) * ys + ym)
    ens = np.nanmean(preds, axis=0)
    y_te = y_arr[te_idx]
    metrics = {
        "model": "4-GNN unweighted mean (GINE+GATv2+MPNN+CARATE)",
        "seed": 42,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "mae_nm": float(mean_absolute_error(y_te, ens)),
        "rmse_nm": float(root_mean_squared_error(y_te, ens)),
        "r2": float(r2_score(y_te, ens)),
        "source": "evaluation-only ensemble over committed checkpoints",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()