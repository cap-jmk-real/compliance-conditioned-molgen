"""Build a clean UV-Vis dataset: SMILES plus first absorption maximum in nm.

Source: Joung et al. 2020, "Experimental database of optical properties of organic
compounds", Scientific Data 7, 295. Figshare item "DB for chromophore" (CC BY 4.0),
https://doi.org/10.6084/m9.figshare.12045567 . Paper reports 20,236 records over
7,016 chromophores and 365 solvents.

Download notes (verified 2026-11-16):
  - The figshare REST API and ndownloader hosts returned HTTP 403 from the build box
    network (Cloudflare). They are expected to work from Colab.
  - A byte-identical mirror of the published rev02 CSV is served from GitHub raw and
    is used as the fallback. Both URLs point at the same CC BY 4.0 data.

The script tries the figshare download first, then the GitHub mirror, then any local
copy passed with --local. It never invents data. It prints the actual row counts.

Column of interest: "Absorption max (nm)". The chromophore SMILES is column
"Chromophore". When a chromophore appears several times (different solvents) we keep
the median absorption maximum so each structure has one target value.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uv.common import canonical_smiles  # noqa: E402

# Primary: figshare rev02 file (the version whose record count matches the paper).
FIGSHARE_URL = "https://ndownloader.figshare.com/files/23637518"
# Fallback: GitHub raw mirror of the same published CSV (pinned commit).
GITHUB_MIRROR = (
    "https://raw.githubusercontent.com/howardwang1997/AI4PS/"
    "12113ccff61edaf1ac83294aef876a8782ddfd99/"
    "raw_data/DB%20for%20chromophore_Sci_Data_rev02.csv"
)

SMILES_COL = "Chromophore"
ABS_COL = "Absorption max (nm)"


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_raw(local: str | None) -> pd.DataFrame:
    if local and os.path.exists(local):
        print(f"reading local copy: {local}", flush=True)
        return pd.read_csv(local)
    last_err = None
    for name, url in [("figshare", FIGSHARE_URL), ("github-mirror", GITHUB_MIRROR)]:
        try:
            print(f"downloading from {name}: {url}", flush=True)
            raw = _fetch(url)
            df = pd.read_csv(io.BytesIO(raw))
            print(f"download ok from {name}: {len(df)} raw rows", flush=True)
            return df
        except Exception as exc:  # noqa: BLE001
            print(f"download from {name} failed: {exc}", flush=True)
            last_err = exc
    raise RuntimeError(f"could not obtain the UV-Vis source: {last_err}")


def build(df: pd.DataFrame) -> pd.DataFrame:
    n_raw = len(df)
    if SMILES_COL not in df.columns or ABS_COL not in df.columns:
        raise RuntimeError(f"unexpected columns: {list(df.columns)[:6]} ...")

    df = df[[SMILES_COL, ABS_COL]].copy()
    df[ABS_COL] = pd.to_numeric(df[ABS_COL], errors="coerce")
    n_abs = df[ABS_COL].notna().sum()
    df = df.dropna(subset=[SMILES_COL, ABS_COL])

    # Keep physically sensible UV-Vis maxima only.
    df = df[(df[ABS_COL] >= 150) & (df[ABS_COL] <= 1200)]
    n_range = len(df)

    rows = []
    bad = 0
    for smi, val in zip(df[SMILES_COL], df[ABS_COL]):
        canon = canonical_smiles(smi)
        if canon is None:
            bad += 1
            continue
        rows.append((canon, float(val)))
    clean = pd.DataFrame(rows, columns=["smiles", "abs_max_nm"])

    # One value per structure: median across solvents / duplicate measurements.
    grouped = clean.groupby("smiles", as_index=False)["abs_max_nm"].median()

    print(f"raw rows: {n_raw}", flush=True)
    print(f"rows with a numeric absorption max: {n_abs}", flush=True)
    print(f"rows in plausible range 150-1200 nm: {n_range}", flush=True)
    print(f"rows with unparsable SMILES dropped: {bad}", flush=True)
    print(f"unique chromophore structures: {len(grouped)}", flush=True)
    print(
        "abs_max_nm: min %.1f median %.1f max %.1f"
        % (grouped.abs_max_nm.min(), grouped.abs_max_nm.median(), grouped.abs_max_nm.max()),
        flush=True,
    )
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="uv/data/uvvis_dataset.csv")
    ap.add_argument("--local", default=None, help="path to a local copy of the source CSV")
    ap.add_argument("--max-rows", type=int, default=0, help="cap unique structures (smoke test)")
    args = ap.parse_args()

    df = load_raw(args.local)
    clean = build(df)
    if args.max_rows and len(clean) > args.max_rows:
        clean = clean.sample(n=args.max_rows, random_state=0).reset_index(drop=True)
        print(f"capped to {len(clean)} structures for a smoke run", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    clean.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(clean)} rows", flush=True)


if __name__ == "__main__":
    main()
