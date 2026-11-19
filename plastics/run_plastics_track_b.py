#!/usr/bin/env python3
"""Plastics Track B: Compliance-conditioned PSMILES polymer generation (Tg 450-550 K, REACH + BPA-free)."""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, "/home/s4il0r/t25-metasearch/paper_worktree")
sys.path.insert(0, "/home/s4il0r/t25-metasearch/paper_worktree/uv")
import smarts_rules as sr  # noqa: E402
from gnn_predictor import build_gnn_from_ckpt, predict_smiles  # noqa: E402

TOKEN_RE = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Se|@@?|%\d\d|[A-Za-z0-9@+\-\(\)=#/\*\\\.])")
BPA_PATT = Chem.MolFromSmarts("c1ccc(cc1)C(C)(C)c2ccc(cc2)")


def tokenize_psmiles(s: str) -> List[str]:
    return TOKEN_RE.findall(s)


def is_valid_psmiles(s: str) -> Tuple[bool, Chem.Mol | None]:
    if s.count("*") != 2:
        return False, None
    mol = Chem.MolFromSmiles(s)
    if mol is None or mol.GetNumAtoms() < 4:
        return False, None
    # Verify exactly 2 dummy atoms (atomic num 0) each with degree 1
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2 or any(a.GetDegree() != 1 for a in dummies):
        return False, None
    return True, mol


def is_compliant(mol: Chem.Mol) -> bool:
    for r in sr.RULES:
        if r.matches(mol):
            return False
    if mol.HasSubstructMatch(BPA_PATT):
        return False
    return True


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


class PSmilesGRU(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 128, hidden: int = 384, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, h=None):
        out, h = self.gru(self.emb(x), h)
        return self.head(out), h


def sample_psmiles(model: PSmilesGRU, tok2id: dict, id2tok: list, n: int, device: torch.device, max_len: int = 110, temp: float = 0.95):
    model.eval()
    bos, eos = tok2id["<BOS>"], tok2id["<EOS>"]
    seqs, logps = [], []
    with torch.no_grad():
        for start in range(0, n, 256):
            bs = min(256, n - start)
            x = torch.full((bs, 1), bos, dtype=torch.long, device=device)
            h = None
            toks = [[] for _ in range(bs)]
            done = [False] * bs
            for _ in range(max_len):
                logits, h = model(x, h)
                probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
                nxt = torch.multinomial(probs, 1)
                x = nxt
                nxt_cpu = nxt.squeeze(1).tolist()
                for i, tid in enumerate(nxt_cpu):
                    if not done[i]:
                        if tid == eos:
                            done[i] = True
                        elif tid > 2:
                            toks[i].append(id2tok[tid])
                if all(done):
                    break
            seqs.extend("".join(t) for t in toks)
    return seqs


def seq_log_probs(model: PSmilesGRU, batch_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    logits, _ = model(batch_ids[:, :-1])
    lp = F.log_softmax(logits, dim=-1)
    tgt = batch_ids[:, 1:]
    tok_lp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    mask = (tgt != pad_id).float()
    return (tok_lp * mask).sum(dim=1)


def internal_diversity(mols: List[Chem.Mol], n_sample: int = 1000) -> float:
    if len(mols) < 2:
        return 0.0
    rng = np.random.default_rng(42)
    idx = rng.choice(len(mols), size=min(n_sample, len(mols)), replace=False)
    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, nBits=1024) for i in idx]
    sims = []
    for i in range(len(fps)):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        sims.extend(s)
    return float(1.0 - np.mean(sims)) if sims else 0.0


def main() -> None:
    out_dir = Path("/home/s4il0r/t25-plastics")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(out_dir / "LAMALAB_CURATED_Tg_structured_polymerclass.csv")
    psmiles_all = [str(s) for s in df["PSMILES"]]
    tg_all = df["labels.Exp_Tg(K)"].to_numpy(dtype=np.float32)
    train_set = set(psmiles_all)

    # 1. Load 4-GNN ensemble & evaluate on test split (seed 42)
    ckpts = [out_dir / f"gnn_{a}_tg.pt" for a in ["gine", "gatv2", "mpnn", "carate"]]
    gnns = [build_gnn_from_ckpt(p, dev) for p in ckpts]

    def pred_4gnn(smis: List[str]) -> np.ndarray:
        ps = [predict_smiles(m, smis, dev) * ys + ym for m, ym, ys in gnns]
        return np.nanmean(ps, axis=0)

    _, te_idx = train_test_split(np.arange(len(psmiles_all)), test_size=0.1, random_state=42)
    te_pred_gnn = pred_4gnn([psmiles_all[i] for i in te_idx])
    gnn_ens_metrics = {
        "mae_K": float(mean_absolute_error(tg_all[te_idx], te_pred_gnn)),
        "rmse_K": float(root_mean_squared_error(tg_all[te_idx], te_pred_gnn)),
        "r2": float(r2_score(tg_all[te_idx], te_pred_gnn)),
    }
    print("4-GNN Tg Ensemble (test):", gnn_ens_metrics, flush=True)

    # 2. Train independent Holdout ExtraTrees (seed 1337)
    X_all, valid_i = featurize_holdout(psmiles_all)
    y_all = tg_all[valid_i]
    h_tr, h_te = train_test_split(np.arange(len(valid_i)), test_size=0.1, random_state=1337)
    holdout_et = ExtraTreesRegressor(n_estimators=200, random_state=1337, n_jobs=8)
    holdout_et.fit(X_all[h_tr], y_all[h_tr])
    h_pred = holdout_et.predict(X_all[h_te])
    holdout_metrics = {
        "mae_K": float(mean_absolute_error(y_all[h_te], h_pred)),
        "rmse_K": float(root_mean_squared_error(y_all[h_te], h_pred)),
        "r2": float(r2_score(y_all[h_te], h_pred)),
    }
    print("Independent Holdout ExtraTrees (test):", holdout_metrics, flush=True)

    # 3. Build PSMILES vocabulary and train GRU prior on all 7367 polymers
    tok_lists = [tokenize_psmiles(s) for s in psmiles_all]
    vocab_tokens = ["<PAD>", "<BOS>", "<EOS>"] + sorted({t for tl in tok_lists for t in tl})
    tok2id = {t: i for i, t in enumerate(vocab_tokens)}
    encoded = [[tok2id["<BOS>"]] + [tok2id[t] for t in tl] + [tok2id["<EOS>"]] for tl in tok_lists]
    max_len = min(max(len(e) for e in encoded), 112)
    encoded = [e for e in encoded if len(e) <= max_len]

    prior = PSmilesGRU(len(vocab_tokens)).to(dev)
    opt = torch.optim.AdamW(prior.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(42)
    for ep in range(1, 31):
        prior.train()
        perm = rng.permutation(len(encoded))
        losses = []
        for start in range(0, len(perm), 128):
            chunk = [encoded[i] for i in perm[start : start + 128]]
            L = max(len(c) for c in chunk)
            arr = torch.zeros((len(chunk), L), dtype=torch.long, device=dev)
            for r, c in enumerate(chunk):
                arr[r, : len(c)] = torch.tensor(c, dtype=torch.long)
            logits, _ = prior(arr[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, len(vocab_tokens)), arr[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.item()))
        if ep in (10, 20, 30):
            print(f"[prior] epoch {ep}/30 | loss={np.mean(losses):.4f}", flush=True)

    torch.save({"state_dict": prior.state_dict(), "vocab": vocab_tokens}, out_dir / "psmiles_prior.pt")

    # 4. Compliance-conditioned REINVENT (target Tg 450-550 K, REACH + BPA-free)
    agent = PSmilesGRU(len(vocab_tokens)).to(dev)
    agent.load_state_dict(prior.state_dict())
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    opt_rl = torch.optim.Adam(agent.parameters(), lr=2e-4)
    seen_counts = Counter()
    sigma = 35.0
    band_lo, band_hi = 450.0, 550.0

    for step in range(1, 601):
        agent.eval()
        cand_smis = sample_psmiles(agent, tok2id, vocab_tokens, 128, dev, max_len=105, temp=1.0)
        # Score valid + compliant candidates with 4-GNN ensemble
        val_smis, val_idx = [], []
        rewards = np.zeros(len(cand_smis), dtype=np.float32)
        for i, s in enumerate(cand_smis):
            ok, mol = is_valid_psmiles(s)
            if ok and mol is not None and is_compliant(mol):
                if seen_counts[s] < 3:
                    val_smis.append(s)
                    val_idx.append(i)
        if val_smis:
            preds = pred_4gnn(val_smis)
            for idx_i, s, p in zip(val_idx, val_smis, preds):
                if not np.isfinite(p):
                    continue
                if band_lo <= p <= band_hi:
                    r = 1.0
                else:
                    dist = min(abs(p - band_lo), abs(p - band_hi))
                    r = float(math.exp(-((dist / 35.0) ** 2)))
                rewards[idx_i] = r
                if r >= 0.9:
                    seen_counts[s] += 1

        # Encode sampled sequences for REINVENT loss
        enc_batch = []
        for s in cand_smis:
            tl = [tok2id[t] for t in tokenize_psmiles(s) if t in tok2id]
            enc_batch.append([tok2id["<BOS>"]] + tl[:105] + [tok2id["<EOS>"]])
        L = max(len(c) for c in enc_batch)
        arr = torch.zeros((len(enc_batch), L), dtype=torch.long, device=dev)
        for r, c in enumerate(enc_batch):
            arr[r, : len(c)] = torch.tensor(c, dtype=torch.long)

        agent.train()
        ag_lp = seq_log_probs(agent, arr)
        with torch.no_grad():
            pr_lp = seq_log_probs(prior, arr)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=dev)
        aug_lp = pr_lp + sigma * r_t
        loss = ((aug_lp - ag_lp) ** 2).mean()
        opt_rl.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 2.0)
        opt_rl.step()

        if step % 150 == 0:
            print(f"[reinvent] step {step}/600 | mean_reward={rewards.mean():.3f} | high_r={(rewards >= 0.95).mean():.3f}", flush=True)

    torch.save({"state_dict": agent.state_dict(), "vocab": vocab_tokens}, out_dir / "psmiles_agent_450_550K.pt")

    # 5. Final evaluation on 3,000 fresh samples (Prior vs Compliance-REINVENT)
    def eval_model(name: str, mod: PSmilesGRU, temp: float = 0.95):
        smis = sample_psmiles(mod, tok2id, vocab_tokens, 3000, dev, max_len=105, temp=temp)
        valid_smis, valid_mols, comp_flags = [], [], []
        for s in smis:
            ok, m = is_valid_psmiles(s)
            if ok and m is not None:
                valid_smis.append(s)
                valid_mols.append(m)
                comp_flags.append(is_compliant(m))
        uniq = len(set(valid_smis)) / max(len(valid_smis), 1)
        novel = sum(1 for s in set(valid_smis) if s not in train_set) / max(len(set(valid_smis)), 1)
        comp_rate = float(np.mean(comp_flags)) if comp_flags else 0.0
        div = internal_diversity(valid_mols, n_sample=800)

        p_gnn = pred_4gnn(valid_smis)
        X_v, v_idx = featurize_holdout(valid_smis)
        p_hold = holdout_et.predict(X_v)
        in_gnn = float(((p_gnn >= band_lo) & (p_gnn <= band_hi)).mean())
        in_hold = float(((p_hold >= band_lo) & (p_hold <= band_hi)).mean())
        both_comp_in = float(
            np.mean([
                comp_flags[i] and (band_lo <= p_gnn[i] <= band_hi) and (band_lo <= p_hold[j] <= band_hi)
                for j, i in enumerate(v_idx)
            ])
        )
        res = {
            "name": name,
            "n_samples": len(smis),
            "psmiles_valid": len(valid_smis) / len(smis),
            "uniqueness": uniq,
            "novelty": novel,
            "compliance_pass": comp_rate,
            "int_div": div,
            "in_band_4gnn": in_gnn,
            "in_band_holdout_et": in_hold,
            "compliant_and_in_band_both": both_comp_in,
        }
        print(f"EVAL {name}:", json.dumps(res), flush=True)
        return res, valid_smis, comp_flags, p_gnn, p_hold, v_idx

    res_prior, *_ = eval_model("prior", prior, temp=1.0)
    res_agent, v_smis, c_flags, p_gnn, p_hold, v_idx = eval_model("compliance_reinvent_4gnn", agent, temp=0.95)

    # Export top novel compliant in-band polymers
    top_rows = []
    seen_out = set()
    for j, i in enumerate(v_idx):
        s = v_smis[i]
        if s in train_set or s in seen_out or not c_flags[i]:
            continue
        if band_lo <= p_gnn[i] <= band_hi and band_lo <= p_hold[j] <= band_hi:
            seen_out.add(s)
            top_rows.append({"PSMILES": s, "pred_Tg_4gnn_K": round(float(p_gnn[i]), 1), "pred_Tg_holdout_ET_K": round(float(p_hold[j]), 1)})
    pd.DataFrame(top_rows[:50]).to_csv(out_dir / "top_compliant_polymers.csv", index=False)

    summary = {
        "dataset": "LAMALAB_CURATED_Tg_structured_polymerclass.csv (CC-BY-4.0, Zenodo 10.5281/zenodo.15210035)",
        "target_band_K": [band_lo, band_hi],
        "compliance_rules": "REACH smarts_rules.py (PFAS, ortho-phthalates, organotin, etc.) + Bisphenol-A motif",
        "gnn_ensemble_test": gnn_ens_metrics,
        "holdout_et_test": holdout_metrics,
        "prior": res_prior,
        "compliance_reinvent_4gnn": res_agent,
        "n_novel_compliant_inband_exported": len(top_rows[:50]),
    }
    (out_dir / "track_b_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DONE:", out_dir / "track_b_summary.json", flush=True)


if __name__ == "__main__":
    main()
