"""A second, independent UV-Vis predictor used only to check the winners.

The meta-search optimises the in-band fraction under the fixed seed-0 surrogate
(Morgan bits into a gradient-boosted regressor). Any search against one surrogate can
learn to exploit that surrogate. This module trains a second predictor that differs in
features, learner, random seed and train/test split:

  features  Morgan COUNT fingerprint (radius 3, 1024 folded bins) + MACCS keys + a small
            set of RDKit descriptors that are physically tied to absorption (size of the
            largest conjugated system, aromatic ring count, heteroatom count, ...).
  learner   ExtraTreesRegressor (a different inductive bias from boosting).
  split     random split with its own seed.

What it is NOT: independent data. It is fitted on the same chromophore database as the
surrogate, so it shares that database's label noise and domain. It catches
surrogate-specific exploits (fingerprint-bit artefacts); it does not validate the
chemistry. Held-out MAE/R2 on its own split are reported next to every use.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors, rdFingerprintGenerator

N_BITS = 1024
RADIUS = 3
_GEN = None


def _generator():
    global _GEN
    if _GEN is None:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return _GEN


def largest_conjugated_system(mol) -> int:
    """Number of atoms in the largest connected set of conjugated bonds."""
    n = mol.GetNumAtoms()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    in_conj = set()
    for b in mol.GetBonds():
        if b.GetIsConjugated():
            a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            in_conj.update((a, c))
            ra, rc = find(a), find(c)
            if ra != rc:
                parent[ra] = rc
    if not in_conj:
        return 0
    counts: Dict[int, int] = {}
    for a in in_conj:
        r = find(a)
        counts[r] = counts.get(r, 0) + 1
    return max(counts.values())


DESCRIPTOR_NAMES = ["largest_conj", "n_conj_atoms_frac", "n_aromatic_rings", "n_rings",
                    "n_hetero", "heavy_atoms", "mol_wt", "frac_csp3", "tpsa", "logp",
                    "n_rot_bonds", "n_double_bonds", "n_donors", "n_acceptors"]


def descriptor_vector(mol) -> np.ndarray:
    heavy = mol.GetNumHeavyAtoms()
    conj_atoms = {i for b in mol.GetBonds() if b.GetIsConjugated()
                  for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}
    n_double = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    return np.array([
        largest_conjugated_system(mol),
        len(conj_atoms) / max(heavy, 1),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        heavy,
        Descriptors.MolWt(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        n_double,
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
    ], dtype=np.float32)


def featurize_one(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts = np.asarray(_generator().GetCountFingerprintAsNumPy(mol), dtype=np.float32)
    maccs = np.zeros(167, dtype=np.float32)
    bv = MACCSkeys.GenMACCSKeys(mol)
    for i in bv.GetOnBits():
        maccs[i] = 1.0
    return np.concatenate([counts, maccs, descriptor_vector(mol)])


def featurize(smiles: Sequence[Optional[str]]):
    feats, idx = [], []
    for i, s in enumerate(smiles):
        f = featurize_one(s) if s is not None else None
        if f is not None:
            feats.append(f)
            idx.append(i)
    if feats:
        return np.vstack(feats), np.array(idx)
    return np.zeros((0, N_BITS + 167 + len(DESCRIPTOR_NAMES)), dtype=np.float32), np.array([], int)


class HoldoutPredictor:
    def __init__(self, model, metrics: dict):
        self.model = model
        self.metrics = metrics

    def predict(self, smiles: Sequence[Optional[str]]) -> np.ndarray:
        out = np.full(len(smiles), np.nan, dtype=float)
        X, idx = featurize(smiles)
        if len(idx):
            out[idx] = self.model.predict(X)
        return out


def train_holdout(data_csv: str, seed: int = 1337, n_estimators: int = 300,
                  test_frac: float = 0.1, max_rows: int = 0, n_jobs: int = 4
                  ) -> HoldoutPredictor:
    import pandas as pd
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_csv)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    X, idx = featurize(df["smiles"].tolist())
    y = df["abs_max_nm"].to_numpy(dtype=float)[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=seed)
    model = ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=2,
                                max_features=0.5, random_state=seed, n_jobs=n_jobs)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "model": "morgan_count_r3_1024+maccs+conj_descriptors_extratrees",
        "seed": seed, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "mae_nm": float(mean_absolute_error(y_te, pred)),
        "rmse_nm": float(np.sqrt(mean_squared_error(y_te, pred))),
        "r2": float(r2_score(y_te, pred)),
        "note": "same chromophore database as the surrogate; different features, "
                "learner, seed and split",
    }
    return HoldoutPredictor(model, metrics)


def load_or_train(path: str, data_csv: str, **kw) -> HoldoutPredictor:
    import joblib
    if path and os.path.exists(path):
        blob = joblib.load(path)
        return HoldoutPredictor(blob["model"], blob["metrics"])
    hp = train_holdout(data_csv, **kw)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        joblib.dump({"model": hp.model, "metrics": hp.metrics}, tmp)
        os.replace(tmp, path)
        with open(os.path.splitext(path)[0] + "_metrics.json", "w") as fh:
            json.dump(hp.metrics, fh, indent=2)
    return hp
