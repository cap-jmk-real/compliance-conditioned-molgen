#!/usr/bin/env python3
"""Pharma Track 2 v2: 5-HT1A SFT warm-start + smooth multi-objective safety REINVENT (5-HT1A high, 5-HT2B & hERG low)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, MACCSkeys, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import selfies as sf

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, "/home/s4il0r/t25-metasearch/paper_worktree")
sys.path.insert(0, "/home/s4il0r/t25-metasearch/paper_worktree/uv")
sys.path.insert(0, "/home/s4il0r/t25-metasearch/repo")

import smarts_rules as sr  # noqa: E402
from gnn_predictor import build_gnn_from_ckpt, predict_smiles  # noqa: E402
import uv.metasearch2 as m2  # noqa: E402
from uv import steering  # noqa: E402

PAINS_PARAMS = FilterCatalog.FilterCatalogParams()
PAINS_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog.FilterCatalog(PAINS_PARAMS)


def check_compliance_and_cns(mol: Chem.Mol) -> Tuple[bool, bool, float]:
    for r in sr.RULES:
        if r.matches(mol):
            return False, False, 0.0
    if PAINS_CATALOG.HasMatch(mol):
        return False, False, 0.0
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    qed = float(QED.qed(mol))
    cns_ok = (mw <= 450.0) and (tpsa <= 90.0) and (hbd <= 3) and (qed >= 0.55)
    return True, cns_ok, qed


def featurize_holdout(smiles_list: List[str]) -> Tuple[np.ndarray, List[int]]:
    feats, idxs = [], []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp1 = np.zeros(1024, dtype=np.float32)
        bv = AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=1024)
        DataStructs.ConvertToNumpyArray(bv, fp1)
        fp2 = np.zeros(167, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), fp2)
        feats.append(np.concatenate([fp1, fp2]))
        idxs.append(i)
    return np.vstack(feats), idxs


def load_ensemble(model_dir: Path, prefix: str, archs: List[str], dev: torch.device):
    models = [build_gnn_from_ckpt(model_dir / f"{prefix}_{a}.pt", dev) for a in archs]

    def pred(smis: List[str]) -> np.ndarray:
        ps = [predict_smiles(m, smis, dev) * ys + ym for m, ym, ys in models]
        return np.nanmean(ps, axis=0)

    return pred


def internal_diversity(mols: List[Chem.Mol], n_sample: int = 800) -> float:
    if len(mols) < 2:
        return 0.0
    rng = np.random.default_rng(42)
    idx = rng.choice(len(mols), size=min(n_sample, len(mols)), replace=False)
    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, nBits=1024) for i in idx]
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
    return float(1.0 - np.mean(sims)) if sims else 0.0


def main() -> None:
    root = Path("/home/s4il0r/t25-pharma")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn_preds = {
        "5ht1a": load_ensemble(root / "models", "5ht1a", ["gine", "gatv2", "mpnn", "carate"], dev),
        "5ht2b": load_ensemble(root / "models", "5ht2b", ["gine", "gatv2", "carate"], dev),
        "herg": load_ensemble(root / "models", "herg", ["gine", "gatv2", "carate"], dev),
    }

    holdout_models = {}
    known_active_set = set()
    sft_smiles = []
    for t in ["5ht1a", "5ht2b", "herg"]:
        df = pd.read_csv(root / "data" / f"{t}.csv")
        smis = [str(s) for s in df["smiles"]]
        ys = df["pchembl"].to_numpy(dtype=np.float32)
        if t == "5ht1a":
            known_active_set.update(smis)
            # Collect compliant 5-HT1A ligands with pChEMBL >= 6.5 for gentle SFT warm-start
            for s, y in zip(smis, ys):
                if y >= 6.5:
                    m = Chem.MolFromSmiles(s)
                    if m is not None:
                        co, _, _ = check_compliance_and_cns(m)
                        if co:
                            sft_smiles.append(s)
        X, vi = featurize_holdout(smis)
        yv = ys[vi]
        htr, _ = train_test_split(np.arange(len(vi)), test_size=0.1, random_state=1337)
        et = ExtraTreesRegressor(n_estimators=200, random_state=1337, n_jobs=8)
        et.fit(X[htr], yv[htr])
        holdout_models[t] = et

    args = m2.make_parser().parse_args([
        "--mode", "search",
        "--root", str(root / "tmp_ctx"),
        "--plain", "/home/s4il0r/t25-multiseed/ckpt-seeds/generator_plain_seed1.pt",
        "--steered", "/home/s4il0r/t25-multiseed/ckpt-seeds/generator_steered_seed1.pt",
        "--vocab", "/home/s4il0r/t25-multiseed/ckpt-seeds/shared/vocab.json",
        "--corpus", "/home/s4il0r/t25-multiseed/ckpt-seeds/shared/corpus.csv",
        "--predictor", "/home/s4il0r/t25-multiseed/ckpt-seeds/shared/uvvis_predictor.joblib",
        "--uvvis-data", "/home/s4il0r/t25-multiseed/ckpt-seeds/shared/uvvis_dataset.csv",
    ])
    ctx = m2.build_context2(args)
    base_prior = ctx.frozen("plain")
    prefix = ctx.prefix("plain")

    # Step A: Gentle SFT warm-start on compliant 5-HT1A ChEMBL ligands (8 epochs)
    t2i = ctx.vocab.stoi
    bos, eos, pad = t2i["<bos>"], t2i["<eos>"], t2i["<pad>"]
    sft_seqs = []
    for s in sft_smiles:
        try:
            enc = sf.encoder(s)
            if not enc:
                continue
            toks = list(sf.split_selfies(enc))
            if all(t in t2i for t in toks) and len(toks) <= ctx.max_len - 2:
                sft_seqs.append([bos] + [t2i[t] for t in toks] + [eos])
        except Exception:
            continue
    print(f"[SFT warm-start] {len(sft_seqs)} SELFIES-encodable 5-HT1A ligands", flush=True)

    warm_prior = ctx.fresh("plain")
    opt_sft = torch.optim.AdamW(warm_prior.parameters(), lr=5e-4, weight_decay=1e-4)
    rng = np.random.default_rng(42)
    for ep in range(1, 9):
        warm_prior.train()
        perm = rng.permutation(len(sft_seqs))
        losses = []
        for start in range(0, len(perm), 64):
            chunk = [sft_seqs[i] for i in perm[start : start + 64]]
            L = max(len(c) for c in chunk)
            arr = torch.full((len(chunk), L), pad, dtype=torch.long, device=dev)
            for r, c in enumerate(chunk):
                arr[r, : len(c)] = torch.tensor(c, dtype=torch.long)
            logits, _ = warm_prior(arr[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), arr[:, 1:].reshape(-1), ignore_index=pad)
            opt_sft.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(warm_prior.parameters(), 2.0)
            opt_sft.step()
            losses.append(float(loss.item()))
        if ep in (4, 8):
            print(f"[SFT warm-start] epoch {ep}/8 | loss={np.mean(losses):.4f}", flush=True)

    warm_prior.eval()
    for p in warm_prior.parameters():
        p.requires_grad_(False)

    # Step B: Smooth multi-objective safety REINVENT with Bemis-Murcko scaffold diversity memory (500 steps)
    agent = ctx.fresh("plain")
    agent.load_state_dict(warm_prior.state_dict())
    opt = torch.optim.Adam(agent.parameters(), lr=1.2e-4)
    seen = Counter()
    scaffolds = Counter()
    scaffold_cap = 15
    sigma = 30.0

    for step in range(1, 501):
        agent.eval()
        experts = steering.blend_experts(agent, warm_prior, prefix, 0.0)
        tok, ln = steering.sample_tokens(experts, ctx.vocab, dev, 128, ctx.max_len, 1.0, 4040 + step, mask_special=False)
        sfs = steering.tokens_to_selfies(ctx.vocab, tok, ln)
        smis = [sf.decoder(s) for s in sfs]

        rewards = np.zeros(len(smis), dtype=np.float32)
        ok_idx, ok_smis, ok_scafs, ok_cns, ok_qeds = [], [], [], [], []
        batch_cans = []
        for i, s in enumerate(smis):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            can = Chem.MolToSmiles(m, canonical=True)
            batch_cans.append(can)
            if seen[can] >= 1:
                continue
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) or can
            except Exception:
                scaf = can
            if scaffolds[scaf] >= scaffold_cap:
                continue
            comp_ok, cns_ok, qed = check_compliance_and_cns(m)
            if comp_ok:
                ok_idx.append((i, can, scaf))
                ok_smis.append(can)
                ok_scafs.append(scaf)
                ok_cns.append(1.0 if cns_ok else 0.4)
                ok_qeds.append(qed)

        if ok_smis:
            p1a = gnn_preds["5ht1a"](ok_smis)
            p2b = gnn_preds["5ht2b"](ok_smis)
            pherg = gnn_preds["herg"](ok_smis)
            s_1a = np.clip((p1a - 5.5) / 2.5, 0.0, 1.0)
            s_2b = np.clip((7.0 - p2b) / 1.8, 0.0, 1.0)
            s_hg = np.clip((6.5 - pherg) / 1.5, 0.0, 1.0)
            r_tot = (0.45 * s_1a + 0.25 * s_2b + 0.20 * s_hg + 0.10 * np.asarray(ok_qeds)) * np.asarray(ok_cns)
            for (i, can, scaf), rv in zip(ok_idx, r_tot):
                if np.isfinite(rv):
                    rewards[i] = float(rv)
                    if rv >= 0.55:
                        seen[can] += 1
                        scaffolds[scaf] += 1

        agent.train()
        ag_lp = steering.sequence_logprob(agent, ctx.vocab, prefix, tok, ln)
        with torch.no_grad():
            pr_lp = steering.sequence_logprob(warm_prior, ctx.vocab, prefix, tok, ln)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=dev)
        loss = (((pr_lp + sigma * r_t) - ag_lp) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 2.0)
        opt.step()

        if step % 100 == 0:
            b_uniq = len(set(batch_cans)) / max(len(batch_cans), 1)
            print(
                f"[pharma-v3] step {step}/500 | mean_r={rewards.mean():.3f} | "
                f"high_r={(rewards >= 0.70).mean():.3f} | b_uniq={b_uniq:.3f} | scafs={len(scaffolds)}",
                flush=True,
            )

    torch.save(agent.state_dict(), root / "agent_5ht1a_safe_v3.pt")

    # Step C: Evaluate 3,000 fresh samples across ZINC prior, Warm-Start prior, and Safety-REINVENT
    def eval_mod(name: str, mod, temp: float = 0.95):
        mod.eval()
        experts = steering.blend_experts(mod, warm_prior, prefix, 0.0)
        sfs = []
        for i in range(6):
            tok, ln = steering.sample_tokens(experts, ctx.vocab, dev, 500, ctx.max_len, temp, 8888 + i, mask_special=False)
            sfs += steering.tokens_to_selfies(ctx.vocab, tok, ln)
        smis = [sf.decoder(s) for s in sfs]
        cans, mols, comp_f, cns_f, qeds = [], [], [], [], []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            cans.append(Chem.MolToSmiles(m, canonical=True))
            mols.append(m)
            co, cn, q = check_compliance_and_cns(m)
            comp_f.append(co)
            cns_f.append(cn)
            qeds.append(q)

        uniq = len(set(cans)) / max(len(cans), 1)
        novel = sum(1 for s in set(cans) if s not in ctx.train_set and s not in known_active_set) / max(len(set(cans)), 1)
        div = internal_diversity(mols, 800)

        p1a_g = gnn_preds["5ht1a"](cans)
        p2b_g = gnn_preds["5ht2b"](cans)
        phg_g = gnn_preds["herg"](cans)
        X, vi = featurize_holdout(cans)
        p1a_h = holdout_models["5ht1a"].predict(X)
        p2b_h = holdout_models["5ht2b"].predict(X)
        phg_h = holdout_models["herg"].predict(X)

        hit_gnn = (p1a_g >= 7.0) & (p2b_g <= 5.8) & (phg_g <= 5.5) & np.array(comp_f) & np.array(cns_f)
        hit_hold = (p1a_h >= 7.0) & (p2b_h <= 5.8) & (phg_h <= 5.5) & np.array([comp_f[i] and cns_f[i] for i in vi])
        hit_both = np.array([bool(hit_gnn[i] and hit_hold[j]) for j, i in enumerate(vi)])

        res = {
            "name": name,
            "n_samples": len(smis),
            "valid": len(cans) / len(smis),
            "uniqueness": uniq,
            "novelty": novel,
            "int_div": div,
            "reach_pains_pass": float(np.mean(comp_f)),
            "cns_bbb_pass": float(np.mean(cns_f)),
            "mean_qed": float(np.mean(qeds)),
            "5ht1a_ge_7_gnn": float((p1a_g >= 7.0).mean()),
            "5ht1a_ge_7_holdout": float((p1a_h >= 7.0).mean()),
            "5ht2b_le_5p8_gnn": float((p2b_g <= 5.8).mean()),
            "herg_le_5p5_gnn": float((phg_g <= 5.5).mean()),
            "all_criteria_gnn": float(hit_gnn.mean()),
            "all_criteria_holdout": float(hit_hold.mean()),
            "all_criteria_both": float(hit_both.mean()),
        }
        print(f"EVAL {name}:", json.dumps(res), flush=True)
        return res, cans, vi, p1a_g, p2b_g, phg_g, p1a_h, p2b_h, phg_h, qeds, hit_both

    res_base, *_ = eval_mod("zinc_prior", base_prior, temp=1.0)
    res_warm, *_ = eval_mod("sft_warmstart_prior", warm_prior, temp=1.0)
    res_agent, cans, vi, p1a_g, p2b_g, phg_g, p1a_h, p2b_h, phg_h, qeds, hit_both = eval_mod("cns_safe_reinvent_v3_scaffold", agent, temp=0.95)

    rows = []
    seen_out = set()
    for j, i in enumerate(vi):
        if not hit_both[j]:
            continue
        s = cans[i]
        if s in seen_out or s in ctx.train_set or s in known_active_set:
            continue
        seen_out.add(s)
        rows.append({
            "SMILES": s,
            "5HT1A_GNN": round(float(p1a_g[i]), 2),
            "5HT1A_HoldoutET": round(float(p1a_h[j]), 2),
            "5HT2B_GNN": round(float(p2b_g[i]), 2),
            "5HT2B_HoldoutET": round(float(p2b_h[j]), 2),
            "hERG_GNN": round(float(phg_g[i]), 2),
            "hERG_HoldoutET": round(float(phg_h[j]), 2),
            "QED": round(float(qeds[i]), 3),
        })
    rows.sort(key=lambda r: -(r["5HT1A_GNN"] + r["5HT1A_HoldoutET"]))
    pd.DataFrame(rows[:50]).to_csv(root / "top_cns_candidates_v3.csv", index=False)

    summary = {
        "targets": {"on_target": "5-HT1A (CHEMBL214)", "anti_targets": ["5-HT2B (CHEMBL1833)", "hERG (CHEMBL240)"]},
        "zinc_prior": res_base,
        "sft_warmstart_prior": res_warm,
        "cns_safe_reinvent_v3_scaffold": res_agent,
        "n_novel_safe_candidates_exported": len(rows[:50]),
    }
    (root / "pharma_summary_v3.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DONE:", root / "pharma_summary_v3.json", flush=True)


if __name__ == "__main__":
    main()
