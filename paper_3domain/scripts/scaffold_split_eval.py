#!/usr/bin/env python3
"""Murcko-scaffold-held-out evaluation for UV-Vis GNN predictors (evaluation-only; requires checkpoints).

Example (after training or copying checkpoints into ckpt-dir):
  python3 paper_3domain/scripts/scaffold_split_eval.py \\
    --data uv/data/uvvis_dataset.csv \\
    --ckpt-dir uv/results/round3_gnn/checkpoints

Does not train models. See paper_3domain/generated/README.md.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]


def scaffold_key(smiles: str) -> str:
    mol = __import__("rdkit.Chem", fromlist=["Chem"]).Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    sc = MurckoScaffold.GetScaffoldForMol(mol)
    return __import__("rdkit.Chem", fromlist=["Chem"]).Chem.MolToSmiles(sc, isomericSmiles=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "uv/data/uvvis_dataset.csv"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "uv/results/round3_gnn/checkpoints"))
    ap.add_argument("--out", default=str(ROOT / "paper_3domain/generated/exports/scaffold_split_uvvis.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from uv.gnn_predictor import build_gnn_from_ckpt, mol_to_graph, predict_smiles

    df = pd.read_csv(args.data)
    rows = []
    groups = []
    for smi, y in zip(df.smiles, df.abs_max_nm):
        if mol_to_graph(str(smi)) is None or not np.isfinite(y):
            continue
        rows.append((str(smi), float(y)))
        groups.append(scaffold_key(str(smi)))
    if len(rows) < 50:
        raise SystemExit("Too few valid molecules for scaffold split")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
    idx = np.arange(len(rows))
    tr, te = next(gss.split(idx, groups=groups))
    smiles_te = [rows[i][0] for i in te]
    y_te = np.array([rows[i][1] for i in te], dtype=float)

    archs = ["gine", "gatv2", "mpnn", "carate"]
    ckpt_dir = Path(args.ckpt_dir)
    dev = torch.device(args.device)
    preds = []
    for a in archs:
        pth = ckpt_dir / f"{a}_uvvis.pt"
        if not pth.is_file():
            raise SystemExit(f"Missing checkpoint: {pth}")
        model, ym, ys = build_gnn_from_ckpt(pth, dev)
        preds.append(predict_smiles(model, smiles_te, dev) * ys + ym)
    ens = np.nanmean(preds, axis=0)
    out = {
        "split": "murcko_scaffold_group_shuffle",
        "seed": args.seed,
        "n_train_scaffolds": len(set(groups[i] for i in tr)),
        "n_test_scaffolds": len(set(groups[i] for i in te)),
        "n_test": int(len(te)),
        "ensemble_mae_nm": float(mean_absolute_error(y_te, ens)),
        "note": "Scaffold split is a stress test; main paper tables use random 90/10 split at seed 42.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()