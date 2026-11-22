"""Evaluate plain vs steered with the second dataset's surrogate, with every gate.

Same protocol as uv/evaluate.py: 20,000 samples per arm, the steered arm is prompted
with <BAND_IN><CLEAN>, validity / uniqueness / novelty / rule pass / in-band fraction are
computed with the same definitions, and the JSON uses the same schema, so
uv/aggregate_seeds.py reads it unchanged.

Additions (they do not change the headline numbers):
  - int_div: internal diversity, 1 - mean pairwise Tanimoto (Morgan r2, 2048 bits) over up
    to 1000 unique valid molecules (MOSES IntDiv1). Same definition as uv/steer_eval.py
    on main, used there for the diversity gate.
  - gates: validity >= 0.95, uniqueness >= 0.90, novelty >= 0.90, rule pass >= 0.95 and
    int_div >= 0.9 x plain, the pre-registered limits of the steering meta-search
    (uv/metasearch.py Limits).
  - cross_surrogate: the in-band fraction under the MAIN dataset's surrogate. The steered
    arm was never conditioned on that surrogate, so this is a check that the effect is
    not specific to the oracle that produced the labels. It is reported separately and
    never replaces the headline number.
The structural screen is a metric only. It is never a target or a reward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from uv import gen_model  # noqa: E402
from uv.common import in_band, morgan_fingerprint, passes_screen  # noqa: E402
from uv.evaluate import bootstrap_diff_ci, decode_valid  # noqa: E402

LIMITS = {"validity": 0.95, "uniqueness": 0.90, "novelty": 0.90,
          "rule_pass_rate": 0.95, "int_div_ratio": 0.90}


def internal_diversity(smiles, max_n=1000, seed=0):
    """1 - mean pairwise Tanimoto (Morgan r2, 2048 bits). Same as uv/steer_eval.py."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    smiles = list(smiles)
    if len(smiles) > max_n:
        rng = np.random.default_rng(seed)
        smiles = [smiles[i] for i in rng.choice(len(smiles), max_n, replace=False)]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fps.append(gen.GetFingerprint(mol))
    if len(fps) < 2:
        return float("nan")
    total, pairs = 0.0, 0
    for i in range(1, len(fps)):
        total += float(np.sum(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])))
        pairs += i
    return 1.0 - total / pairs


def _predict(predictor, n_bits, smiles):
    preds = []
    for s in smiles:
        fp = morgan_fingerprint(s, n_bits=n_bits)
        preds.append(float(predictor.predict(fp.reshape(1, -1))[0]) if fp is not None
                     else float("nan"))
    return np.array(preds, dtype=float)


def eval_arm_ext(arm_path, vocab, device, n_samples, train_set, predictor, n_bits,
                 band_lo, band_hi, seed, cross=None):
    """Same metrics and rounding as uv.evaluate.eval_arm, plus int_div and cross-surrogate."""
    bundle = torch.load(arm_path, map_location=device, weights_only=False)
    model = gen_model.SelfiesLM(len(vocab), bundle["emb"], bundle["hidden"],
                                bundle["layers"]).to(device)
    model.load_state_dict(bundle["model"])
    steered = bundle["arm"] == "steered"
    prefix = [gen_model.BAND_IN, gen_model.CLEAN] if steered else None
    samples = gen_model.sample(model, vocab, device, n_samples, steered,
                               prefix_tokens=prefix, seed=seed)
    valid = decode_valid(samples)
    unique = sorted(set(valid))
    novel = [s for s in unique if s not in train_set]
    screen_ok = sum(1 for s in unique if passes_screen(s))
    preds = _predict(predictor, n_bits, unique)
    scored = ~np.isnan(preds)
    flags = np.array([1.0 if in_band(p, band_lo, band_hi) else 0.0 for p in preds[scored]])
    summary = {
        "arm": bundle["arm"],
        "n_sampled": len(samples),
        "n_valid": len(valid),
        "n_unique": len(unique),
        "n_scored": int(scored.sum()),
        "validity": round(len(valid) / max(len(samples), 1), 4),
        "uniqueness": round(len(unique) / max(len(valid), 1), 4),
        "novelty": round(len(novel) / max(len(unique), 1), 4),
        "rule_pass_rate": round(screen_ok / max(len(unique), 1), 4),
        "mean_pred_abs_nm": round(float(np.nanmean(preds)) if scored.any() else float("nan"), 2),
        "in_band_fraction": round(float(flags.mean()) if len(flags) else 0.0, 4),
        "int_div": round(internal_diversity(unique, 1000, seed), 4),
    }
    if cross is not None:
        c_pred = _predict(cross[0], cross[1], unique)
        ok = ~np.isnan(c_pred)
        c_flags = np.array([1.0 if in_band(p, band_lo, band_hi) else 0.0 for p in c_pred[ok]])
        summary["cross_surrogate_in_band_fraction"] = round(
            float(c_flags.mean()) if len(c_flags) else 0.0, 4)
        summary["cross_surrogate_mean_pred_abs_nm"] = round(float(np.nanmean(c_pred)), 2)
        summary["cross_surrogate_flags"] = c_flags
    return summary, flags


def gate_report(arm: dict, plain_int_div: float) -> dict:
    """Pass/fail of one arm against the gates. Pure function of the numbers."""
    res = {}
    for key in ("validity", "uniqueness", "novelty", "rule_pass_rate"):
        res[key] = {"value": arm[key], "limit": LIMITS[key], "pass": bool(arm[key] >= LIMITS[key])}
    floor = LIMITS["int_div_ratio"] * plain_int_div
    res["int_div"] = {"value": arm["int_div"], "limit": round(floor, 4),
                      "pass": bool(arm["int_div"] >= floor)}
    res["passes_all"] = all(v["pass"] for v in res.values())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--steered", required=True)
    ap.add_argument("--predictor", required=True, help="surrogate trained on dataset 2")
    ap.add_argument("--predictor-metrics", default=None)
    ap.add_argument("--cross-predictor", default=None,
                    help="surrogate of the main dataset (secondary check only)")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--band-lo", type=float, default=400.0)
    ap.add_argument("--band-hi", type=float, default=450.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = gen_model.Vocab.load(args.vocab)
    train_set = set(pd.read_csv(args.corpus, usecols=["smiles"])["smiles"].tolist())
    pb = joblib.load(args.predictor)
    cross = None
    if args.cross_predictor:
        cb = joblib.load(args.cross_predictor)
        cross = (cb["model"], cb["n_bits"])
    pred_metrics = None
    if args.predictor_metrics and os.path.exists(args.predictor_metrics):
        with open(args.predictor_metrics) as fh:
            pred_metrics = json.load(fh)

    results, flags = {}, {}
    for name, path in (("plain", args.plain), ("steered", args.steered)):
        print(f"sampling arm: {name}", flush=True)
        summary, arm_flags = eval_arm_ext(path, vocab, device, args.n_samples, train_set,
                                          pb["model"], pb["n_bits"], args.band_lo,
                                          args.band_hi, args.seed, cross)
        c_flags = summary.pop("cross_surrogate_flags", None)
        results[name], flags[name] = summary, arm_flags
        if c_flags is not None:
            flags[name + "_cross"] = c_flags
        print(f"  {summary}", flush=True)

    boot = bootstrap_diff_ci(flags["plain"], flags["steered"], args.n_bootstrap, seed=args.seed)
    cross_boot = None
    if "plain_cross" in flags:
        cross_boot = bootstrap_diff_ci(flags["plain_cross"], flags["steered_cross"],
                                       args.n_bootstrap, seed=args.seed)
    plain_div = results["plain"]["int_div"]
    gates = {"plain": gate_report(results["plain"], plain_div),
             "steered": gate_report(results["steered"], plain_div)}
    meta = {"seed": args.seed, "dataset": "beard2019_text_mined_uvvis",
            "band_lo": args.band_lo, "band_hi": args.band_hi,
            "n_samples": args.n_samples, "device": device,
            "predictor_metrics": pred_metrics, "arms": results,
            "bootstrap_in_band_diff": boot, "bootstrap_in_band_diff_cross_surrogate": cross_boot,
            "limits": LIMITS, "gates": gates}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(meta, fh, indent=2)

    lines = [f"# Dataset 2 (Beard 2019), seed {args.seed}: steered vs plain", "",
             f"Band {args.band_lo:.0f} to {args.band_hi:.0f} nm, {args.n_samples} samples per arm, "
             f"device {device}.", "", "| metric | plain | steered |", "|---|---|---|"]
    for key in ("validity", "uniqueness", "novelty", "rule_pass_rate", "int_div",
                "in_band_fraction", "mean_pred_abs_nm", "n_valid", "n_unique", "n_scored",
                "cross_surrogate_in_band_fraction"):
        if key in results["plain"]:
            lines.append(f"| {key} | {results['plain'][key]} | {results['steered'][key]} |")
    if boot:
        lines += ["", f"In-band difference (steered - plain): {boot['point_estimate']}, bootstrap "
                      f"95% CI [{boot['ci_low']}, {boot['ci_high']}]."]
    lines += ["", f"Gates passed (steered): {gates['steered']['passes_all']}"]
    with open(args.out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
