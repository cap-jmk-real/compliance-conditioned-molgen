#!/usr/bin/env python3
"""Run parallel ab-initio DF-B3LYP/6-31G TDA-DFT validation on 50 generated UV-Vis chromophores."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def run_one(args: tuple[int, str, float, float, float, int, int]) -> dict:
    idx, smi, pred_4gnn, pred_et, sa_score, brics_frags, heavy_atoms = args
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from pyscf import dft, gto
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    out = {
        "idx": idx,
        "SMILES": smi,
        "heavy_atoms": int(heavy_atoms),
        "sa_score": float(sa_score),
        "brics_frags": int(brics_frags),
        "pred_4gnn_nm": float(pred_4gnn),
        "pred_holdout_et_nm": float(pred_et),
        "converged": False,
        "homo_lumo_ev": None,
        "s1_nm": None,
        "s1_osc": None,
        "brightest_nm": None,
        "brightest_osc": None,
        "s1_ev": None,
        "elapsed_s": None,
        "error": None,
    }
    try:
        mol = Chem.MolFromSmiles(smi)
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 + idx
        if AllChem.EmbedMolecule(mol3d, params) != 0:
            AllChem.EmbedMolecule(mol3d, useRandomCoords=True, randomSeed=42 + idx)
        try:
            AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol3d, maxIters=500)
        conf = mol3d.GetConformer()
        atoms = [
            (mol3d.GetAtomWithIdx(i).GetSymbol(), tuple(conf.GetAtomPosition(i)))
            for i in range(mol3d.GetNumAtoms())
        ]

        pmol = gto.M(atom=atoms, basis="6-31g", verbose=0, max_memory=3000)
        mf = dft.RKS(pmol).density_fit()
        mf.xc = "b3lyp"
        mf.conv_tol = 1e-6
        mf.max_cycle = 80
        mf.grids.level = 2
        mf.kernel()
        if not mf.converged:
            mf = dft.newton(mf)
            mf.kernel()

        nocc = pmol.nelec[0]
        homo_lumo_ev = float((mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * 27.211386)

        td = mf.TDA()
        td.nstates = 3
        td.conv_tol = 1e-4
        td.max_cycle = 40
        e_ha, _ = td.kernel()
        osc = np.asarray(td.oscillator_strength(), dtype=float)
        e_ha = np.asarray(e_ha, dtype=float)
        lams = 45.56335 / np.maximum(e_ha, 1e-6)

        bright_idx = int(np.argmax(osc))
        out.update({
            "converged": bool(mf.converged),
            "homo_lumo_ev": round(homo_lumo_ev, 3),
            "s1_nm": round(float(lams[0]), 2),
            "s1_osc": round(float(osc[0]), 4),
            "brightest_nm": round(float(lams[bright_idx]), 2),
            "brightest_osc": round(float(osc[bright_idx]), 4),
            "s1_ev": round(float(e_ha[0] * 27.211386), 3),
            "elapsed_s": round(time.time() - t0, 1),
        })
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    root = Path("/home/s4il0r/t25-dft")
    df = pd.read_csv(root / "uvvis_50_candidates.csv")
    jsonl_path = root / "dft_50_results.jsonl"
    done_idx = set()
    results = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_idx.add(r["idx"])
                results.append(r)

    tasks = []
    for idx, row in df.iterrows():
        if idx in done_idx:
            continue
        tasks.append((
            int(idx),
            str(row["SMILES"]),
            float(row["pred_4gnn_nm"]),
            float(row["pred_holdout_et_nm"]),
            float(row["sa_score"]),
            int(row["brics_frags"]),
            int(row["heavy_atoms"]),
        ))

    print(f"Running DF-B3LYP/6-31G TDA-DFT on {len(tasks)} remaining molecules (8 workers x 4 threads)...", flush=True)
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
            print(
                f"[{len(results):02d}/50 | {time.time()-t_start:.0f}s] idx={r['idx']:02d} "
                f"conv={r['converged']} 4GNN={r['pred_4gnn_nm']:.1f}nm S1={r['s1_nm']}nm "
                f"bright={r['brightest_nm']}nm (f={r['brightest_osc']}) HL={r['homo_lumo_ev']}eV",
                flush=True,
            )

    results.sort(key=lambda r: r["idx"])
    res_df = pd.DataFrame(results)
    res_df.to_csv(root / "dft_50_results.csv", index=False)

    ok = res_df[res_df["converged"] & res_df["s1_nm"].notna()]
    s1 = ok["s1_nm"].to_numpy(dtype=float)
    br = ok["brightest_nm"].to_numpy(dtype=float)
    f_br = ok["brightest_osc"].to_numpy(dtype=float)
    gnn = ok["pred_4gnn_nm"].to_numpy(dtype=float)
    hl = ok["homo_lumo_ev"].to_numpy(dtype=float)

    summary = {
        "method": "DF-B3LYP/6-31G TDA-DFT (3 singlet states, ETKDGv3+MMFF94 3D geometry)",
        "n_total": int(len(res_df)),
        "n_converged": int(len(ok)),
        "scf_convergence_rate": round(float(len(ok) / max(len(res_df), 1)), 4),
        "s1_nm_mean": round(float(np.mean(s1)), 1),
        "s1_nm_median": round(float(np.median(s1)), 1),
        "s1_nm_sd": round(float(np.std(s1, ddof=1)), 1),
        "s1_in_400_450": int(np.sum((s1 >= 400.0) & (s1 <= 450.0))),
        "s1_in_380_470": int(np.sum((s1 >= 380.0) & (s1 <= 470.0))),
        "s1_ge_380_vis": int(np.sum(s1 >= 380.0)),
        "bright_nm_mean": round(float(np.mean(br)), 1),
        "bright_nm_median": round(float(np.median(br)), 1),
        "bright_osc_ge_0p05": int(np.sum(f_br >= 0.05)),
        "bright_osc_ge_0p10": int(np.sum(f_br >= 0.10)),
        "bright_in_380_470_and_f_ge_0p05": int(np.sum((br >= 380.0) & (br <= 470.0) & (f_br >= 0.05))),
        "mae_4gnn_vs_s1_nm": round(float(np.mean(np.abs(gnn - s1))), 1),
        "bias_4gnn_minus_s1_nm": round(float(np.mean(gnn - s1)), 1),
        "homo_lumo_ev_mean": round(float(np.mean(hl)), 2),
        "homo_lumo_ev_sd": round(float(np.std(hl, ddof=1)), 2),
        "total_wall_s": round(time.time() - t_start, 1),
    }
    (root / "dft_50_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DONE DFT SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
