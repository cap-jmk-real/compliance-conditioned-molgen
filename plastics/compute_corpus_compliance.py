#!/usr/bin/env python3
"""Compute REACH+BPA failure rate on the PolyMetriX curated Tg corpus (evaluation-only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smarts_rules as sr  # noqa: E402

BPA_PATT = Chem.MolFromSmarts("c1ccc(cc1)C(C)(C)c2ccc(cc2)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(ROOT / "plastics/LAMALAB_CURATED_Tg_structured_polymerclass.csv"),
        help="Download from Zenodo 10.5281/zenodo.15210035",
    )
    ap.add_argument("--out", default=str(ROOT / "plastics/corpus_compliance_summary.json"))
    args = ap.parse_args()
    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(f"Missing corpus CSV at {path}")

    df = pd.read_csv(path)
    psmiles = [str(s) for s in df["PSMILES"]]
    n = len(psmiles)
    fail = 0
    tallies = {"pfas": 0, "bpa": 0, "organotin": 0, "ortho_phthalate": 0}
    for s in psmiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fail += 1
            continue
        bad = False
        for r in sr.RULES:
            if r.matches(mol):
                bad = True
                name = r.name.lower()
                if "pfas" in name:
                    tallies["pfas"] += 1
                elif "organotin" in name:
                    tallies["organotin"] += 1
                elif "phthal" in name:
                    tallies["ortho_phthalate"] += 1
        if mol.HasSubstructMatch(BPA_PATT):
            bad = True
            tallies["bpa"] += 1
        if bad:
            fail += 1
    summary = {
        "corpus_n": n,
        "fail_count": fail,
        "fail_fraction": float(fail / n),
        "tally_note": "Categories overlap; tallies are motif hits not mutually exclusive.",
        "tallies": tallies,
        "rules": "smarts_rules.py REACH families + BPA motif",
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()