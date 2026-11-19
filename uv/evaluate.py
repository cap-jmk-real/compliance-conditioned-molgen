"""Evaluate the two generator arms on the same sample count.

For each arm the script samples a fixed number of molecules and computes:
  - validity: fraction of samples that decode to a valid RDKit molecule
  - uniqueness: unique valid structures / valid samples
  - novelty: fraction of unique valid structures not in the training corpus
  - rule_pass_rate: fraction of unique valid structures that match none of the
    flagged structural families in smarts_rules.py (a screening metric, never a target)
  - mean predicted absorption maximum (nm) over unique valid structures
  - in_band_fraction: fraction whose predicted maximum falls in the target band

The steered arm is sampled with the band and screen control tokens set to the target
(BAND_IN, CLEAN). The plain arm is sampled unconditionally.

The script also reports a bootstrap 95 percent confidence interval on the difference
in in-band fraction between the two arms, so the comparison is honest about whether
any difference is real. If a predictor-metrics file is given, the regressor's held-out
MAE, RMSE and R2 are carried into the results. Writes a results JSON and a markdown
table. All numbers come from the actual samples. Nothing is invented.
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
    inband_flags = []
    for s in unique:
        if passes_screen(s):
            screen_ok += 1
        fp = morgan_fingerprint(s, n_bits=n_bits)
        if fp is not None:
            p = float(predictor.predict(fp.reshape(1, -1))[0])
            preds.append(p)
            inband_flags.append(1 if in_band(p, band_lo, band_hi) else 0)
    screen_rate = screen_ok / max(len(unique), 1)
    preds_arr = np.array(preds) if preds else np.array([np.nan])
    in_band_frac = float(np.mean(inband_flags)) if inband_flags else 0.0

    summary = {
        "arm": bundle["arm"],
        "n_sampled": len(samples),
        "n_valid": len(valid),
        "n_unique": len(unique),
        "n_scored": len(inband_flags),
        "validity": round(validity, 4),
        "uniqueness": round(uniqueness, 4),
        "novelty": round(novelty, 4),
        "rule_pass_rate": round(screen_rate, 4),
        "mean_pred_abs_nm": round(float(np.nanmean(preds_arr)), 2),
        "in_band_fraction": round(in_band_frac, 4),
    }
    return summary, np.array(inband_flags, dtype=float)


def bootstrap_diff_ci(flags_a, flags_b, n_boot, seed=0, alpha=0.05):
    """Bootstrap CI on mean(flags_b) - mean(flags_a) (steered minus plain).

    Each arm is resampled with replacement independently. Returns the point estimate,
    the CI bounds, and the fraction of resamples where the difference is positive.
    """
    if len(flags_a) == 0 or len(flags_b) == 0:
        return None
    rng = np.random.default_rng(seed)
    point = float(flags_b.mean() - flags_a.mean())
    diffs = np.empty(n_boot, dtype=float)
    na, nb = len(flags_a), len(flags_b)
    for i in range(n_boot):
        ra = flags_a[rng.integers(0, na, na)].mean()
        rb = flags_b[rng.integers(0, nb, nb)].mean()
        diffs[i] = rb - ra
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    frac_pos = float(np.mean(diffs > 0))
    return {
        "point_estimate": round(point, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "ci_level": round(1 - alpha, 3),
        "n_bootstrap": int(n_boot),
        "frac_resamples_positive": round(frac_pos, 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="uv/data/corpus.csv")
    ap.add_argument("--vocab", default="uv/artifacts/vocab.json")
    ap.add_argument("--plain", default="uv/artifacts/generator_plain.pt")
    ap.add_argument("--steered", default="uv/artifacts/generator_steered.pt")
    ap.add_argument("--predictor", default="uv/artifacts/uvvis_predictor.joblib")
    ap.add_argument("--predictor-metrics", default=None,
                    help="optional predictor metrics JSON (MAE/RMSE/R2) to carry through")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--band-lo", type=float, default=400.0)
    ap.add_argument("--band-hi", type=float, default=450.0)
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

    pred_metrics = None
    if args.predictor_metrics and os.path.exists(args.predictor_metrics):
        with open(args.predictor_metrics) as fh:
            pred_metrics = json.load(fh)

    print(f"evaluating on {args.n_samples} samples per arm, band "
          f"{args.band_lo:.0f}-{args.band_hi:.0f} nm", flush=True)

    results = {}
    flags = {}
    for name, path in [("plain", args.plain), ("steered", args.steered)]:
        print(f"sampling arm: {name}", flush=True)
        summary, arm_flags = eval_arm(path, vocab, device, args.n_samples, train_set,
                                      predictor, n_bits, args.band_lo, args.band_hi,
                                      args.seed)
        results[name] = summary
        flags[name] = arm_flags
        print(f"  {summary}", flush=True)

    boot = bootstrap_diff_ci(flags["plain"], flags["steered"], args.n_bootstrap,
                             seed=args.seed)
    if boot is not None:
        print(f"bootstrap diff (steered - plain) in-band fraction: "
              f"{boot['point_estimate']} CI95 [{boot['ci_low']}, {boot['ci_high']}] "
              f"excludes_zero={boot['excludes_zero']}", flush=True)

    meta = {
        "seed": args.seed,
        "band_lo": args.band_lo, "band_hi": args.band_hi,
        "n_samples": args.n_samples, "device": device,
        "predictor_metrics": pred_metrics,
        "arms": results,
        "bootstrap_in_band_diff": boot,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(meta, fh, indent=2)

    cols = [("validity", "validity"), ("uniqueness", "uniqueness"),
            ("novelty", "novelty"), ("rule_pass_rate", "rule pass rate"),
            ("in_band_fraction", "in-band fraction"),
            ("mean_pred_abs_nm", "mean predicted abs (nm)"),
            ("n_sampled", "samples"), ("n_valid", "valid"),
            ("n_unique", "unique"), ("n_scored", "scored")]
    lines = ["# UV-Vis steered vs plain generation",
             "",
             f"Target band: {args.band_lo:.0f} to {args.band_hi:.0f} nm. "
             f"Samples per arm: {args.n_samples}. Device: {device}.",
             "",
             "| metric | plain | steered |",
             "|---|---|---|"]
    for key, label in cols:
        lines.append(f"| {label} | {results['plain'][key]} | {results['steered'][key]} |")
    if pred_metrics:
        lines += ["",
                  "## Regressor held-out performance",
                  "",
                  "| metric | value |",
                  "|---|---|",
                  f"| MAE (nm) | {round(pred_metrics.get('mae_nm'), 2)} |",
                  f"| RMSE (nm) | {round(pred_metrics.get('rmse_nm'), 2)} |",
                  f"| R2 | {round(pred_metrics.get('r2'), 4)} |",
                  f"| train structures | {pred_metrics.get('n_train')} |",
                  f"| test structures | {pred_metrics.get('n_test')} |"]
    if boot:
        lines += ["",
                  "## Steering effect (bootstrap)",
                  "",
                  f"Difference in in-band fraction, steered minus plain: "
                  f"{boot['point_estimate']}.",
                  f"95 percent bootstrap CI: [{boot['ci_low']}, {boot['ci_high']}] "
                  f"from {boot['n_bootstrap']} resamples.",
                  f"Resamples with a positive difference: {boot['frac_resamples_positive']}.",
                  f"CI excludes zero: {boot['excludes_zero']}."]
    with open(args.out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"wrote {args.out_json} and {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
