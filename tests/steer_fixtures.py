"""Tiny synthetic fixtures for the steering / meta-search tests.

Everything is generated in-process: ordinary aromatic and aliphatic building blocks, a
toy label that grows with the size of the aromatic system (a stand-in so the pipeline has
something to learn, NOT a UV-Vis claim), a tiny surrogate, and two tiny generators
trained for a few steps. No network access, no data files.
"""
from __future__ import annotations

import itertools
import os
import zlib
import sys

import joblib
import numpy as np
import pandas as pd
import selfies as sf
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from uv import gen_model  # noqa: E402
from uv.common import canonical_smiles, morgan_fingerprint, passes_screen  # noqa: E402

CORES = ["c1ccccc1{}", "c1ccc2ccccc2c1{}", "c1ccncc1{}", "c1ccsc1{}", "c1ccc(cc1){}c1ccccc1",
         "C1CCCCC1{}", "c1ccc2c(c1){}cccc2"]
SUBS = ["", "C", "O", "N", "F", "Cl", "CC", "C=O", "C#N", "OC", "N(C)C", "CCO"]


def _smiles_pool():
    out = set()
    for core, sub in itertools.product(CORES, SUBS):
        smi = core.replace("{}", sub) if "{}" in core else core
        c = canonical_smiles(smi)
        if c:
            out.add(c)
    return sorted(out)


def toy_label(smi: str) -> float:
    """Toy band label: 250 nm plus 60 per aromatic ring (not a physical model)."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smi)
    return 250.0 + 60.0 * rdMolDescriptors.CalcNumAromaticRings(mol) + \
        5.0 * (zlib.crc32(smi.encode()) % 7)


def build(tmp_dir: str, band=(400.0, 450.0), gen_steps=30, seed=0):
    os.makedirs(tmp_dir, exist_ok=True)
    smiles = _smiles_pool()
    labels = np.array([toy_label(s) for s in smiles])
    uv = pd.DataFrame({"smiles": smiles, "abs_max_nm": labels})
    uv_path = os.path.join(tmp_dir, "uvvis.csv")
    uv.to_csv(uv_path, index=False)

    X = np.vstack([morgan_fingerprint(s, n_bits=512) for s in smiles])
    reg = HistGradientBoostingRegressor(max_iter=60, random_state=0).fit(X, labels)
    pred_path = os.path.join(tmp_dir, "surrogate.joblib")
    joblib.dump({"model": reg, "n_bits": 512}, pred_path)
    abs_pred = reg.predict(X)

    rows = []
    for s, a in zip(smiles, abs_pred):
        rows.append((s, sf.encoder(s), float(a), bool(passes_screen(s))))
    corpus = pd.DataFrame(rows, columns=["smiles", "selfies", "abs_pred", "passes_screen"])
    corpus_path = os.path.join(tmp_dir, "corpus.csv")
    corpus.to_csv(corpus_path, index=False)

    toks = []
    for s in corpus["selfies"]:
        toks += gen_model.tokenize(s)
    vocab = gen_model.Vocab(toks)
    vocab_path = os.path.join(tmp_dir, "vocab.json")
    vocab.save(vocab_path)

    paths = {}
    torch.manual_seed(seed)
    for arm in ("plain", "steered"):
        steered = arm == "steered"
        seqs = []
        for row in corpus.itertuples(index=False):
            ids = gen_model.build_sequence(vocab, row.selfies, steered, row.abs_pred,
                                           row.passes_screen, band[0], band[1])
            seqs.append(torch.tensor(ids))
        model = gen_model.SelfiesLM(len(vocab), 16, 32, 2)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        lossf = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
        for step in range(gen_steps):
            idx = np.random.default_rng(step).integers(0, len(seqs), 32)
            b = torch.nn.utils.rnn.pad_sequence([seqs[i] for i in idx], batch_first=True,
                                                padding_value=vocab.pad_id)
            logits, _ = model(b[:, :-1])
            loss = lossf(logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        path = os.path.join(tmp_dir, f"generator_{arm}.pt")
        torch.save({"model": model.state_dict(), "arm": arm, "vocab": vocab_path,
                    "emb": 16, "hidden": 32, "layers": 2,
                    "band_lo": band[0], "band_hi": band[1]}, path)
        paths[arm] = path
    return {"uv": uv_path, "predictor": pred_path, "corpus": corpus_path,
            "vocab": vocab_path, "plain": paths["plain"], "steered": paths["steered"],
            "band": band, "n_corpus": len(corpus)}
