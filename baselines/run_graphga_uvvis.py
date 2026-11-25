#!/usr/bin/env python3
"""GraphGA baseline wrapper for UV-Vis band optimization (not run in CI)."""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--band", default="400,450")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        import guacamol  # noqa: F401
    except ImportError:
        print("pip install guacamol; wire GraphGA goal molecule in this script.", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit("Implement GraphGA oracle loop against uv oracles; CLI contract only in this commit.")


if __name__ == "__main__":
    main()