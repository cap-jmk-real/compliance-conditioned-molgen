#!/usr/bin/env python3
"""Pure PyTorch + RDKit GNN architectures (GINE, GATv2, MPNN-GRU) for UV-Vis lambda_max."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ATOM_LIST = [0, 1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # 0 = '*' attachment point in PSMILES
HYB_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot(val, choices) -> List[float]:
    vec = [0.0] * (len(choices) + 1)
    idx = choices.index(val) if val in choices else len(choices)
    vec[idx] = 1.0
    return vec


def mol_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x_rows = []
    for a in mol.GetAtoms():
        feat = (
            one_hot(a.GetAtomicNum(), ATOM_LIST)
            + one_hot(a.GetHybridization(), HYB_LIST)
            + [
                min(a.GetTotalDegree(), 6) / 6.0,
                min(a.GetTotalNumHs(), 4) / 4.0,
                float(a.GetFormalCharge()),
                float(a.GetIsAromatic()),
                float(a.IsInRing()),
            ]
        )
        x_rows.append(feat)
    x = np.asarray(x_rows, dtype=np.float32)

    src, dst, e_rows = [], [], []
    btypes = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ef = one_hot(b.GetBondType(), btypes) + [
            float(b.GetIsConjugated()),
            float(b.IsInRing()),
        ]
        src.extend([u, v])
        dst.extend([v, u])
        e_rows.extend([ef, ef])
    if not src:
        src, dst = [0], [0]
        e_rows = [[0.0] * (len(btypes) + 3)]
    edge_index = np.asarray([src, dst], dtype=np.int64)
    edge_attr = np.asarray(e_rows, dtype=np.float32)
    return x, edge_index, edge_attr


def collate_graphs(graphs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], targets: np.ndarray | None = None):
    xs, eis, eas, batch = [], [], [], []
    offset = 0
    for gi, (x, ei, ea) in enumerate(graphs):
        n = x.shape[0]
        xs.append(torch.from_numpy(x))
        eis.append(torch.from_numpy(ei + offset))
        eas.append(torch.from_numpy(ea))
        batch.append(torch.full((n,), gi, dtype=torch.long))
        offset += n
    out = (
        torch.cat(xs, dim=0),
        torch.cat(eis, dim=1),
        torch.cat(eas, dim=0),
        torch.cat(batch, dim=0),
    )
    if targets is not None:
        return out + (torch.tensor(targets, dtype=torch.float32),)
    return out


def pool_mean_max(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    bs = int(batch.max().item()) + 1
    mean_pool = torch.zeros(bs, h.shape[1], device=h.device, dtype=h.dtype)
    counts = torch.zeros(bs, 1, device=h.device, dtype=h.dtype)
    mean_pool.index_add_(0, batch, h)
    counts.index_add_(0, batch, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype))
    mean_pool = mean_pool / counts.clamp(min=1.0)
    max_pool = torch.full((bs, h.shape[1]), -1e9, device=h.device, dtype=h.dtype)
    max_pool.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce="amax", include_self=True)
    return torch.cat([mean_pool, max_pool], dim=-1)


class GINELayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg = torch.relu(h[src] + self.edge_proj(edge_attr))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        out = self.mlp((1.0 + self.eps) * h + agg)
        return self.norm(h + out)


class UVVisGINE(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_in = nn.Sequential(nn.Linear(node_dim, hidden), nn.SiLU())
        self.layers = nn.ModuleList([GINELayer(hidden, edge_dim) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.node_in(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.head(pool_mean_max(h, batch)).squeeze(-1)


class GATv2Layer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, heads: int = 4):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.d_head = hidden // heads
        self.w_src = nn.Linear(hidden, hidden, bias=False)
        self.w_dst = nn.Linear(hidden, hidden, bias=False)
        self.w_edge = nn.Linear(edge_dim, hidden, bias=False)
        self.attn = nn.Parameter(torch.randn(1, heads, self.d_head) * 0.1)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        hs = self.w_src(h[src]).view(-1, self.heads, self.d_head)
        hd = self.w_dst(h[dst]).view(-1, self.heads, self.d_head)
        he = self.w_edge(edge_attr).view(-1, self.heads, self.d_head)
        e = (F.leaky_relu(hs + hd + he, 0.2) * self.attn).sum(dim=-1)  # [E, H]
        # Segment-wise softmax over dst
        e_max = torch.full((h.shape[0], self.heads), -1e9, device=h.device, dtype=h.dtype)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=True)
        alpha = torch.exp(e - e_max[dst])
        denom = torch.zeros((h.shape[0], self.heads), device=h.device, dtype=h.dtype)
        denom.index_add_(0, dst, alpha)
        alpha = alpha / denom[dst].clamp(min=1e-9)
        msg = (hs + he) * alpha.unsqueeze(-1)
        agg = torch.zeros((h.shape[0], self.heads, self.d_head), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, msg)
        out = self.out_proj(agg.reshape(h.shape[0], -1))
        return self.norm(h + F.silu(out))


class UVVisGATv2(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_in = nn.Sequential(nn.Linear(node_dim, hidden), nn.SiLU())
        self.layers = nn.ModuleList([GATv2Layer(hidden, edge_dim, heads=4) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.node_in(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.head(pool_mean_max(h, batch)).squeeze(-1)


class UVVisMPNN(nn.Module):
    """Tied-weight MPNN with GRU state update (Gilmer et al. style)."""
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.steps = layers
        self.node_in = nn.Sequential(nn.Linear(node_dim, hidden), nn.SiLU())
        self.msg_net = nn.Sequential(nn.Linear(hidden + edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.gru = nn.GRUCell(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.node_in(x)
        src, dst = edge_index[0], edge_index[1]
        for _ in range(self.steps):
            m = self.msg_net(torch.cat([h[src], edge_attr], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h = self.norm(self.gru(agg, h))
        return self.head(pool_mean_max(h, batch)).squeeze(-1)


class GCNLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype))
        inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        msg = (self.lin(h[src]) + self.edge_proj(edge_attr)) * inv_sqrt[src] * inv_sqrt[dst]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        return self.norm(h + F.silu(agg))


class UVVisCARATE(nn.Module):
    """CARATE-style molecular encoder combining GAT + GCN blocks with gated attention readout."""
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_in = nn.Sequential(nn.Linear(node_dim, hidden), nn.SiLU())
        blocks = []
        for i in range(layers):
            blocks.append(GATv2Layer(hidden, edge_dim, heads=4) if i % 2 == 0 else GCNLayer(hidden, edge_dim))
        self.layers = nn.ModuleList(blocks)
        self.att_gate = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.node_in(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        # Gated attention pooling + mean/max pooling
        bs = int(batch.max().item()) + 1
        scores = self.att_gate(h).squeeze(-1)
        s_max = torch.full((bs,), -1e9, device=h.device, dtype=h.dtype)
        s_max.scatter_reduce_(0, batch, scores, reduce="amax", include_self=True)
        w = torch.exp(scores - s_max[batch])
        w_sum = torch.zeros((bs,), device=h.device, dtype=h.dtype)
        w_sum.index_add_(0, batch, w)
        w = (w / w_sum[batch].clamp(min=1e-9)).unsqueeze(-1)
        att_pool = torch.zeros(bs, h.shape[1], device=h.device, dtype=h.dtype)
        att_pool.index_add_(0, batch, h * w)
        return self.head(torch.cat([att_pool, pool_mean_max(h, batch)], dim=-1)).squeeze(-1)


ARCHS = {"gine": UVVisGINE, "gatv2": UVVisGATv2, "mpnn": UVVisMPNN, "carate": UVVisCARATE}


def build_gnn_from_ckpt(ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "gine")
    cls = ARCHS[arch]
    model = cls(ckpt["node_dim"], ckpt["edge_dim"], ckpt["hidden"], ckpt["layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt["y_mean"]), float(ckpt["y_std"])


def predict_smiles(model: nn.Module, smiles_list: List[str], device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = np.full(len(smiles_list), np.nan, dtype=np.float32)
    valid_idx, valid_g = [], []
    for i, s in enumerate(smiles_list):
        g = mol_to_graph(s)
        if g is not None:
            valid_idx.append(i)
            valid_g.append(g)
    with torch.no_grad():
        for start in range(0, len(valid_g), batch_size):
            chunk = valid_g[start : start + batch_size]
            x, ei, ea, b = collate_graphs(chunk)
            out = model(x.to(device), ei.to(device), ea.to(device), b.to(device)).cpu().numpy()
            for k, val in enumerate(out):
                preds[valid_idx[start + k]] = val
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), default="gine")
    ap.add_argument("--data", default="/home/s4il0r/t25-multiseed/ckpt-seeds/shared/uvvis_dataset.csv")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--target-col", default="abs_max_nm")
    ap.add_argument("--out", default="")
    ap.add_argument("--metrics", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_path = args.out or f"/home/s4il0r/t25-metasearch/gnn_{args.arch}_uvvis.pt"
    metrics_path = args.metrics or f"/home/s4il0r/t25-metasearch/gnn_{args.arch}_uvvis_metrics.json"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))

    df = pd.read_csv(args.data)
    graphs, y_vals = [], []
    for smi, y in zip(df[args.smiles_col], df[args.target_col]):
        g = mol_to_graph(str(smi))
        if g is not None and np.isfinite(y):
            graphs.append(g)
            y_vals.append(float(y))
    y_arr = np.asarray(y_vals, dtype=np.float32)

    idx = np.arange(len(graphs))
    tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=args.seed)
    y_mean = float(y_arr[tr_idx].mean())
    y_std = float(y_arr[tr_idx].std())

    node_dim = graphs[0][0].shape[1]
    edge_dim = graphs[0][2].shape[1]
    model = ARCHS[args.arch](node_dim=node_dim, edge_dim=edge_dim, hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_mae = 1e9
    best_state = None
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(tr_idx)
        for start in range(0, len(perm), args.batch_size):
            bi = perm[start : start + args.batch_size]
            x, ei, ea, b, y = collate_graphs([graphs[i] for i in bi], (y_arr[bi] - y_mean) / y_std)
            opt.zero_grad()
            pred = model(x.to(device), ei.to(device), ea.to(device), b.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        te_preds = []
        with torch.no_grad():
            for start in range(0, len(te_idx), 256):
                bi = te_idx[start : start + 256]
                x, ei, ea, b, _ = collate_graphs([graphs[i] for i in bi], y_arr[bi])
                p = model(x.to(device), ei.to(device), ea.to(device), b.to(device)).cpu().numpy()
                te_preds.append(p * y_std + y_mean)
        te_pred = np.concatenate(te_preds)
        mae = float(mean_absolute_error(y_arr[te_idx], te_pred))
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 15 == 0 or ep == args.epochs:
            rmse = float(root_mean_squared_error(y_arr[te_idx], te_pred))
            r2 = float(r2_score(y_arr[te_idx], te_pred))
            print(f"[{args.arch}] epoch {ep:3d}/{args.epochs} | test MAE={mae:.2f} nm | RMSE={rmse:.2f} | R2={r2:.4f} (best={best_mae:.2f})", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    te_preds = []
    with torch.no_grad():
        for start in range(0, len(te_idx), 256):
            bi = te_idx[start : start + 256]
            x, ei, ea, b, _ = collate_graphs([graphs[i] for i in bi], y_arr[bi])
            p = model(x.to(device), ei.to(device), ea.to(device), b.to(device)).cpu().numpy()
            te_preds.append(p * y_std + y_mean)
    te_pred = np.concatenate(te_preds)
    metrics = {
        "arch": args.arch,
        "seed": args.seed,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "mae_nm": float(mean_absolute_error(y_arr[te_idx], te_pred)),
        "rmse_nm": float(root_mean_squared_error(y_arr[te_idx], te_pred)),
        "r2": float(r2_score(y_arr[te_idx], te_pred)),
        "y_mean": y_mean,
        "y_std": y_std,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": args.arch, "state_dict": best_state, "node_dim": node_dim, "edge_dim": edge_dim, "hidden": args.hidden, "layers": args.layers, "y_mean": y_mean, "y_std": y_std}, out_path)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2) + "\n")
    print("FINAL:", json.dumps(metrics))


if __name__ == "__main__":
    main()
