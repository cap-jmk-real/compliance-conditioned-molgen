"""Metrics for one batch of generated molecules, shared by the meta-search.

Definitions follow evaluate.py (so numbers stay comparable with the paper) plus the
extra guards the meta-search needs against reward hacking of the surrogate:

  validity          valid decoded molecules / sampled
  uniqueness        unique valid / valid
  novelty           unique valid not in the training corpus / unique valid
  rule_pass_rate    unique valid matching no flagged structural family / unique valid
                    (evaluation constraint only, never a reward)
  in_band_fraction  unique valid whose surrogate prediction is in the band / unique valid
                    (the paper's headline metric)
  in_band_holdout   same, scored by the independent holdout predictor
  int_div           internal diversity, 1 - mean pairwise Tanimoto (Morgan r2, 2048 bits)
                    over up to `div_max` unique valid molecules (MOSES IntDiv1)
  yield_per_sample  unique valid novel in-band molecules / sampled
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from uv.common import passes_screen
from uv.steering import UVScorer, selfies_to_smiles

_FPGEN = None


def _fpgen():
    global _FPGEN
    if _FPGEN is None:
        _FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return _FPGEN


def internal_diversity(smiles: Sequence[str], max_n: int = 1000, seed: int = 0) -> float:
    """1 - mean pairwise Tanimoto similarity. NaN with fewer than two molecules."""
    smiles = list(smiles)
    if len(smiles) > max_n:
        rng = np.random.default_rng(seed)
        smiles = [smiles[i] for i in rng.choice(len(smiles), max_n, replace=False)]
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fps.append(_fpgen().GetFingerprint(mol))
    if len(fps) < 2:
        return float("nan")
    total, pairs = 0.0, 0
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        total += float(np.sum(sims))
        pairs += i
    return 1.0 - total / pairs


def compute_metrics(selfies_list: Sequence[str], train_set: Set[str], scorer: UVScorer,
                    band_lo: float, band_hi: float,
                    holdout: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
                    div_max: int = 1000, seed: int = 0) -> Dict[str, float]:
    n = len(selfies_list)
    smiles = [selfies_to_smiles(s) for s in selfies_list]
    valid = [s for s in smiles if s is not None]
    unique = sorted(set(valid))
    novel = [s for s in unique if s not in train_set]

    pred = scorer.predict(unique)
    ok = ~np.isnan(pred)
    inband = (pred >= band_lo) & (pred <= band_hi) & ok
    in_band_fraction = float(inband.sum() / max(int(ok.sum()), 1))
    rule_pass = sum(1 for s in unique if passes_screen(s)) / max(len(unique), 1)

    out = {
        "n_sampled": n,
        "n_valid": len(valid),
        "n_unique": len(unique),
        "validity": len(valid) / max(n, 1),
        "uniqueness": len(unique) / max(len(valid), 1),
        "novelty": len(novel) / max(len(unique), 1),
        "rule_pass_rate": rule_pass,
        "in_band_fraction": in_band_fraction,
        "mean_pred_abs_nm": float(np.nanmean(pred)) if ok.any() else float("nan"),
        "int_div": internal_diversity(unique, div_max, seed),
    }
    novel_set = set(novel)
    n_yield = sum(1 for s, b in zip(unique, inband) if b and s in novel_set)
    out["yield_per_sample"] = n_yield / max(n, 1)
    if holdout is not None and unique:
        hp = np.asarray(holdout(unique), dtype=float)
        hok = ~np.isnan(hp)
        hin = (hp >= band_lo) & (hp <= band_hi) & hok
        out["in_band_holdout"] = float(hin.sum() / max(int(hok.sum()), 1))
        out["mean_pred_holdout_nm"] = float(np.nanmean(hp)) if hok.any() else float("nan")
        # both predictors agree the molecule is in band
        both = inband & hin
        out["in_band_both"] = float(both.sum() / max(len(unique), 1))
        if ok.sum() > 2 and hok.sum() > 2:
            m = ok & hok
            out["pred_corr_surrogate_holdout"] = float(np.corrcoef(pred[m], hp[m])[0, 1]) \
                if np.std(pred[m]) > 0 and np.std(hp[m]) > 0 else float("nan")
    return {k: (round(v, 5) if isinstance(v, float) and not np.isnan(v) else v)
            for k, v in out.items()}
