"""Per-metric comparison of two evaluation JSONs (e.g. the same seed on two GPUs).

Used for the hardware-reproducibility check of the #25 UV-Vis experiment: seed 0 trained
and evaluated on an NVIDIA A100 (uv/results/eval_results.json) versus seed 0 on an NVIDIA
RTX 3060 (uv/results/seeds_rtx3060/eval_seed0.json). Prints a markdown table of
B - A per metric and arm, plus the bootstrap in-band difference of each run.
"""
from __future__ import annotations

import argparse
import json

METRICS = ["validity", "uniqueness", "novelty", "rule_pass_rate",
           "in_band_fraction", "mean_pred_abs_nm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    A, B = json.load(open(args.a)), json.load(open(args.b))
    rows, out = [], {"a": args.label_a, "b": args.label_b, "metrics": {}}
    print(f"| metric | arm | {args.label_a} | {args.label_b} | {args.label_b} - {args.label_a} |")
    print("|---|---|---|---|---|")
    for m in METRICS:
        for arm in ("plain", "steered"):
            x, y = A["arms"][arm][m], B["arms"][arm][m]
            out["metrics"][f"{arm}.{m}"] = {"a": x, "b": y, "diff": y - x}
            print(f"| {m} | {arm} | {x} | {y} | {y - x:+.4f} |")
    ba, bb = A["bootstrap_in_band_diff"], B["bootstrap_in_band_diff"]
    for lab, b in ((args.label_a, ba), (args.label_b, bb)):
        print(f"| in-band diff (bootstrap) | {lab} | {b['point_estimate']} "
              f"[{b['ci_low']}, {b['ci_high']}] | | |")
    out["bootstrap_diff"] = {"a": ba, "b": bb,
                             "diff": bb["point_estimate"] - ba["point_estimate"]}
    print(f"\nDifference of steering effects ({args.label_b} - {args.label_a}): "
          f"{out['bootstrap_diff']['diff']:+.4f}")
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
