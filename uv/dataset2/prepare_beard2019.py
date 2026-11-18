"""Build the second UV-Vis dataset: SMILES plus experimental absorption maximum in nm.

Source: Beard, Sivaraman, Vazquez-Mayagoitia, Vishwanath and Cole (2019), "Comparative
dataset of experimental and computational attributes of UV/vis absorption spectra",
Scientific Data 6, 307, https://doi.org/10.1038/s41597-019-0306-0 .
Data record: figshare, https://doi.org/10.6084/m9.figshare.7619672.v2 , licence CC BY 4.0
(figshare record metadata and the open-access article). The records were text-mined with
ChemDataExtractor from 402,034 scientific documents, so the data are independent of the
manually curated literature compilation used for the main result (Joung et al. 2020,
10.6084/m9.figshare.12045567). The two sets may still share some structures; the overlap
is measured and reported by --compare, not assumed away.

The cleaning mirrors uv/prepare_uvvis.py exactly so the protocol is the same:
  1. keep records with a numeric experimental absorption maximum,
  2. keep 150 to 1200 nm,
  3. canonicalise the SMILES with RDKit and drop unparsable ones,
  4. one value per canonical structure: the median over records.

Input, in this order of preference:
  --local-csv  data/CSV/paper_allDB.csv from the figshare archive (sha256 checked)
  --local-zip  the figshare data.zip (md5 checked, the CSV member is read from it)
  otherwise    download data.zip (382 MB, md5 checked). The ndownloader.figshare.com host
               answers HTTP 403 from some networks; the same file is served from the
               public figshare S3 bucket, which is the second URL tried.

The CSV header contains commas inside column names (for example "lambda1 (sTDA, nm)"), so
the header is skipped and the eight columns are named explicitly. Column 6 is the
experimental absorption maximum "lambda_max (Exp, nm)".

The script never invents data and prints the real row counts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from uv.common import canonical_smiles  # noqa: E402

DOI_DATA = "10.6084/m9.figshare.7619672.v2"
DOI_PAPER = "10.1038/s41597-019-0306-0"
ZIP_MD5 = "304da0b0e5b33b29330bc6bf9516291b"        # figshare computed_md5 for data.zip
CSV_MEMBER = "data/CSV/paper_allDB.csv"
CSV_SHA256 = "fe407e73092b6aa56d426cf1cbf1d1de1f2cbc2444bf050b4ec698308ade1e16"
ZIP_URLS = [
    "https://ndownloader.figshare.com/files/14154623",
    "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/14154623/data.zip",
]
COLUMNS = ["SMI", "stda_nm", "stda_f", "tddft_nm", "tddft_f", "exp_nm", "eps", "solvent"]
LO_NM, HI_NM = 150.0, 1200.0


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Parse the figshare CSV. The header row is skipped (see module docstring)."""
    got = sha256_bytes(raw)
    if got != CSV_SHA256:
        raise RuntimeError(f"CSV sha256 mismatch: got {got}, expected {CSV_SHA256}")
    df = pd.read_csv(io.BytesIO(raw), skiprows=1, header=None, names=COLUMNS)
    if df.shape[1] != len(COLUMNS):
        raise RuntimeError(f"unexpected column count {df.shape[1]}")
    return df


def csv_from_zip(zip_path: str) -> bytes:
    got = md5_file(zip_path)
    if got != ZIP_MD5:
        raise RuntimeError(f"zip md5 mismatch: got {got}, expected {ZIP_MD5}")
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(CSV_MEMBER)


def download_zip(dest: str, timeout: int = 600) -> str:
    last = None
    for url in ZIP_URLS:
        try:
            print(f"downloading {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            if md5_file(dest) != ZIP_MD5:
                raise RuntimeError("md5 mismatch after download")
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"download failed: {exc}", flush=True)
            last = exc
    raise RuntimeError(f"could not obtain the dataset: {last}")


def build(df: pd.DataFrame):
    stats = {"raw_rows": int(len(df))}
    df = df[["SMI", "exp_nm"]].copy()
    df["exp_nm"] = pd.to_numeric(df["exp_nm"], errors="coerce")
    stats["rows_numeric_exp"] = int(df["exp_nm"].notna().sum())
    df = df.dropna(subset=["SMI", "exp_nm"])
    df = df[(df["exp_nm"] >= LO_NM) & (df["exp_nm"] <= HI_NM)]
    stats["rows_in_range"] = int(len(df))
    rows, bad = [], 0
    for smi, val in zip(df["SMI"], df["exp_nm"]):
        canon = canonical_smiles(smi)
        if canon is None:
            bad += 1
            continue
        rows.append((canon, float(val)))
    stats["rows_unparsable_dropped"] = bad
    clean = pd.DataFrame(rows, columns=["smiles", "abs_max_nm"])
    grouped = clean.groupby("smiles", as_index=False)["abs_max_nm"].median()
    stats["unique_structures"] = int(len(grouped))
    stats["abs_max_nm_min"] = float(grouped.abs_max_nm.min())
    stats["abs_max_nm_median"] = float(grouped.abs_max_nm.median())
    stats["abs_max_nm_max"] = float(grouped.abs_max_nm.max())
    return grouped, stats


def compare(grouped: pd.DataFrame, other_csv: str, band_lo: float, band_hi: float) -> dict:
    """Overlap and agreement with the main dataset (canonical SMILES join)."""
    other = pd.read_csv(other_csv)
    merged = grouped.merge(other, on="smiles", suffixes=("_ds2", "_ds1"))
    out = {"n_ds1": int(len(other)), "n_ds2": int(len(grouped)),
           "n_overlap": int(len(merged)),
           "frac_ds2_in_ds1": float(len(merged) / max(len(grouped), 1))}
    if len(merged) >= 3:
        a, b = merged["abs_max_nm_ds2"].to_numpy(), merged["abs_max_nm_ds1"].to_numpy()
        out["overlap_pearson_r"] = float(np.corrcoef(a, b)[0, 1])
        out["overlap_mae_nm"] = float(np.mean(np.abs(a - b)))
        out["overlap_median_abs_diff_nm"] = float(np.median(np.abs(a - b)))
    for name, d in (("ds1", other["abs_max_nm"]), ("ds2", grouped["abs_max_nm"])):
        out[f"in_band_frac_{name}"] = float(((d >= band_lo) & (d <= band_hi)).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="uv/data/uvvis_dataset2.csv")
    ap.add_argument("--stats-json", default=None)
    ap.add_argument("--local-csv", default=None)
    ap.add_argument("--local-zip", default=None)
    ap.add_argument("--compare", default=None,
                    help="path to the main dataset CSV (smiles, abs_max_nm) for the overlap report")
    ap.add_argument("--band-lo", type=float, default=400.0)
    ap.add_argument("--band-hi", type=float, default=450.0)
    args = ap.parse_args()

    if args.local_csv and os.path.exists(args.local_csv):
        print(f"reading local CSV: {args.local_csv}", flush=True)
        with open(args.local_csv, "rb") as fh:
            raw = fh.read()
    elif args.local_zip and os.path.exists(args.local_zip):
        print(f"reading local zip: {args.local_zip}", flush=True)
        raw = csv_from_zip(args.local_zip)
    else:
        tmp = os.path.join(os.path.dirname(os.path.abspath(args.out)), "beard2019_data.zip")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        raw = csv_from_zip(download_zip(tmp))

    grouped, stats = build(read_csv_bytes(raw))
    stats.update({"source_doi_data": DOI_DATA, "source_doi_paper": DOI_PAPER,
                  "licence": "CC BY 4.0", "source_csv_sha256": CSV_SHA256})
    for k, v in stats.items():
        print(f"{k}: {v}", flush=True)
    if args.compare:
        stats["compare_to_main_dataset"] = compare(grouped, args.compare,
                                                   args.band_lo, args.band_hi)
        print(json.dumps(stats["compare_to_main_dataset"], indent=2), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    grouped.to_csv(args.out, index=False)
    stats["out_sha256"] = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"wrote {args.out} with {len(grouped)} rows (sha256 {stats['out_sha256']})", flush=True)
    if args.stats_json:
        with open(args.stats_json, "w") as fh:
            json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
