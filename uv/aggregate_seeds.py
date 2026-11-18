"""Aggregate per-seed evaluation results for the #25 UV-Vis experiment.

Reads the combined evaluation JSON of each seed (as written by uv.evaluate: seed 0 is
the committed artifact in uv/results/eval_results.json, seeds 1+ come off Drive) and
reports, per metric per arm, the mean and sample standard deviation across seeds.

For the steering effect (steered minus plain in-band fraction) it reports the per-seed
differences, their mean and sd, and a paired check appropriate for a tiny sample: a
paired t statistic with df = n-1 and its two-sided p-value (Student-t, computed without
SciPy), plus the standardised effect size d_z = mean(diff)/sd(diff). With only n=3
seeds this test is badly underpowered, so it is reported as a direction-and-size check,
not a significance claim. Every number comes from the input JSONs; nothing is invented.
If fewer than two seeds are supplied the dispersion and the paired test are skipped and
the script says so.
"""
from __future__ import annotations

import argparse
import json
import math

ARMS = ("plain", "steered")
METRICS = [
    ("validity", "validity"),
    ("uniqueness", "uniqueness"),
    ("novelty", "novelty"),
    ("rule_pass_rate", "structural-screen pass rate"),
    ("in_band_fraction", "in-band fraction"),
    ("mean_pred_abs_nm", "mean predicted abs (nm)"),
]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    """Sample standard deviation (n-1). Returns 0.0 for a single value."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_quantile_two_sided(alpha, df):
    """Critical value c with P(|T| > c) = alpha for Student-t with df degrees of freedom.

    Solved by bisection on t_two_sided_p, so no SciPy is needed (df=2, alpha=0.05 gives
    4.3027)."""
    lo, hi = 0.0, 1.0e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_two_sided_p(t, df):
    """Two-sided p-value for a t statistic with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


import re


def _seed_from_name(path):
    m = re.search(r"eval_seed(\d+)", path)
    return int(m.group(1)) if m else None


def load_seed(path):
    with open(path) as fh:
        meta = json.load(fh)
    arms = meta.get("arms", {})
    seed = meta.get("seed")
    if seed is None:
        seed = _seed_from_name(path)
    if seed is None and "results/eval_results.json" in path:
        seed = 0
    return {
        "path": path,
        "seed": seed,
        "arms": {a: arms.get(a, {}) for a in ARMS},
        "bootstrap": meta.get("bootstrap_in_band_diff"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_json", nargs="+",
                    help="combined per-seed eval JSONs (seed 0 first by convention)")
    ap.add_argument("--out-json", default=None, help="optional path to write the summary JSON")
    ap.add_argument("--band", default="400-450 nm")
    args = ap.parse_args()

    seeds = [load_seed(p) for p in args.eval_json]
    n = len(seeds)

    lines = []
    lines.append(f"# Multi-seed aggregation ({n} seed{'s' if n != 1 else ''}, band {args.band})")
    lines.append("")
    lines.append(f"Sources: {', '.join(s['path'] for s in seeds)}")
    lines.append("")

    # Per-metric mean +/- sd per arm.
    summary = {"n_seeds": n, "per_metric": {}, "steering_effect": None}
    lines.append("## Per-arm metrics (mean +/- sd across seeds)")
    lines.append("")
    lines.append("| metric | plain | steered |")
    lines.append("|---|---|---|")
    for key, label in METRICS:
        row = {"label": label}
        cells = []
        for arm in ARMS:
            vals = [s["arms"][arm].get(key) for s in seeds]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if not vals:
                cells.append("n/a")
                row[arm] = None
                continue
            m, sdev = mean(vals), sd(vals)
            row[arm] = {"mean": m, "sd": sdev, "values": vals}
            if n >= 2:
                cells.append(f"{m:.4f} +/- {sdev:.4f}")
            else:
                cells.append(f"{m:.4f} (n=1)")
        summary["per_metric"][key] = row
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines.append("")

    # Steering effect: per-seed steered - plain in-band fraction.
    diffs = []
    pairs = []
    for s in seeds:
        p = s["arms"]["plain"].get("in_band_fraction")
        q = s["arms"]["steered"].get("in_band_fraction")
        if isinstance(p, (int, float)) and isinstance(q, (int, float)):
            diffs.append(q - p)
            pairs.append((s.get("seed"), p, q, q - p))

    lines.append("## Steering effect (steered minus plain in-band fraction)")
    lines.append("")
    lines.append("| seed | plain | steered | diff |")
    lines.append("|---|---|---|---|")
    for seed, p, q, d in pairs:
        lines.append(f"| {seed} | {p:.4f} | {q:.4f} | {d:+.4f} |")
    lines.append("")

    eff = {"per_seed_diffs": diffs, "n": len(diffs)}
    if len(diffs) >= 2:
        md, sdd = mean(diffs), sd(diffs)
        dz = md / sdd if sdd > 0 else float("inf")
        df = len(diffs) - 1
        t = md / (sdd / math.sqrt(len(diffs))) if sdd > 0 else float("inf")
        p = t_two_sided_p(t, df) if math.isfinite(t) else 0.0
        tcrit = t_quantile_two_sided(0.05, df)
        half = tcrit * sdd / math.sqrt(len(diffs))
        eff.update({"mean_diff": md, "sd_diff": sdd, "cohens_dz": dz,
                    "paired_t": t, "df": df, "two_sided_p": p,
                    "t_crit_95": tcrit, "ci95_mean_diff": [md - half, md + half]})
        lines.append(f"- mean diff across {len(diffs)} seeds: {md:+.4f} "
                     f"(sd {sdd:.4f})")
        lines.append(f"- standardised effect d_z = mean/sd = {dz:.2f}")
        lines.append(f"- paired t = {t:.3f}, df = {df}, two-sided p = {p:.4f}")
        lines.append(f"- seed-level 95% t-interval on the mean diff (t_crit {tcrit:.4f}): "
                     f"[{md - half:+.4f}, {md + half:+.4f}]")
        n_pos = sum(1 for d in diffs if d > 0)
        if n_pos == len(diffs):
            sign_note = "the effect is positive in every seed above"
        elif n_pos == 0:
            sign_note = "the effect is negative in every seed above"
        else:
            sign_note = (f"the effect is positive in {n_pos} of {len(diffs)} seeds, so "
                         f"its sign is not stable across seeds")
        lines.append(f"- CAUTION: n={len(diffs)} is very small; this paired test is "
                     f"badly underpowered. Read it as a direction-and-size check, not a "
                     f"significance claim. Here {sign_note}.")
    else:
        lines.append("- Only one seed supplied: no cross-seed dispersion or paired test. "
                     "Supply at least two seeds for mean +/- sd.")
    summary["steering_effect"] = eff
    lines.append("")

    # Carry each seed's within-run bootstrap CI for context.
    lines.append("## Per-seed within-run bootstrap CI (sampling at fixed weights)")
    lines.append("")
    lines.append("| seed | diff point | CI95 low | CI95 high |")
    lines.append("|---|---|---|---|")
    for s in seeds:
        b = s.get("bootstrap") or {}
        lines.append(f"| {s.get('seed')} | {b.get('point_estimate')} | "
                     f"{b.get('ci_low')} | {b.get('ci_high')} |")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
