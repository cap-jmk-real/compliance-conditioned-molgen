#!/usr/bin/env python3
"""REINVENT 4 baseline wrapper for UV-Vis band steering (not run in CI)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--band", default="400,450")
    ap.add_argument("--scorer-config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        import reinvent  # noqa: F401
    except ImportError:
        print(
            "Install REINVENT4 stack first (see baselines/README.md).\n"
            f"Planned run: budget={args.budget} seeds={args.seeds} band={args.band}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(
        "Hook up local REINVENT4 TOML + prior paths; this repository only ships the CLI contract."
    )


if __name__ == "__main__":
    main()