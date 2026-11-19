#!/usr/bin/env python3
"""Fetch and deduplicate pChEMBL single-protein activities from ChEMBL REST API for CNS safety-gated track."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

TARGETS = {
    "5ht1a": "CHEMBL214",   # On-target: anxiety / depression (non-hallucinogenic)
    "5ht2b": "CHEMBL1833",  # Anti-target: cardiac valvulopathy
    "herg": "CHEMBL240",    # Anti-target: QT prolongation / cardiotoxicity
}


def fetch_target(name: str, tid: str, out_csv: Path, max_records: int = 12000) -> None:
    if out_csv.is_file() and out_csv.stat().st_size > 1000:
        df = pd.read_csv(out_csv)
        print(f"[{name}] cached {len(df)} unique molecules -> {out_csv}", flush=True)
        return
    base = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={tid}&pchembl_value__isnull=false&limit=1000"
    vals = defaultdict(list)
    url = base
    fetched = 0
    while url and fetched < max_records:
        req = urllib.request.Request(url, headers={"User-Agent": "compliance-molgen/1.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        acts = data.get("activities", [])
        if not acts:
            break
        for a in acts:
            smi = a.get("canonical_smiles")
            pv = a.get("pchembl_value")
            if not smi or pv is None:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            can = Chem.MolToSmiles(m, canonical=True)
            vals[can].append(float(pv))
        fetched += len(acts)
        nxt = data.get("page_meta", {}).get("next")
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
        if fetched % 3000 == 0 or not url:
            print(f"[{name}] fetched {fetched} raw -> {len(vals)} unique mols", flush=True)

    rows = [{"smiles": k, "pchembl": float(np.median(v))} for k, v in vals.items()]
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[{name}] saved {len(df)} unique molecules (median pChEMBL={df['pchembl'].median():.2f}) -> {out_csv}", flush=True)


def main() -> None:
    out_dir = Path("/home/s4il0r/t25-pharma/data")
    for name, tid in TARGETS.items():
        fetch_target(name, tid, out_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
