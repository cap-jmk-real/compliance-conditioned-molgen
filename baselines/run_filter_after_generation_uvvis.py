#!/usr/bin/env python3
"""Generate-then-filter baseline: sample from prior, screen compliance and Morgan band (no RL)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band", default="400,450")
    ap.add_argument("--prior", help="Path to plain ZINC250k GRU checkpoint")
    ap.add_argument("--vocab", help="Path to vocab.json")
    ap.add_argument("--predictor", help="Path to Morgan-HGB joblib")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lo, hi = (float(x) for x in args.band.split(","))
    print(
        json.dumps(
            {
                "status": "not_run",
                "method": "filter_after_generation",
                "n_planned": args.n,
                "seed": args.seed,
                "band_nm": [lo, hi],
                "note": "Provide --prior, --vocab, and --predictor from the Track 1 training run.",
            },
            indent=2,
        )
    )
    raise SystemExit(
        "Baseline script stub: add checkpoint paths and call uv.evaluate sampling pipeline."
    )


if __name__ == "__main__":
    main()