#!/usr/bin/env python3
"""Generate publication-quality vector PDF figures for the 3-domain compliance-conditioned molgen preprint."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
DARK = "#222222"
GREY = "#666666"
LIGHT_BG = "#F7F9FB"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = Path("/home/s4il0r/t25-metasearch/paper_worktree/paper_3domain")


def make_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, title_color=DARK):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=fc, edgecolor=ec, linewidth=1.3,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="top",
                fontsize=8.3, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h / 2 - 0.14, body, ha="center", va="center",
                fontsize=7.1, color=DARK, linespacing=1.28)

    def arrow(x1, y1, x2, y2, color=DARK, rad=0.0, ls="-"):
        a = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=5,head_width=3",
            connectionstyle=f"arc3,rad={rad}",
            color=color, linewidth=1.25, linestyle=ls,
        )
        ax.add_patch(a)

    # Column 1: Domain Priors
    box(0.15, 0.78, 2.15, 2.55, "1. Domain Priors",
        "UV-Vis: SELFIES GRU\n(ZINC250k, 249k)\n\n"
        "Plastics: P-SMILES GRU\n(PolyMetriX, 7.4k)\n\n"
        "CNS: SELFIES GRU +\n8-ep ChEMBL 5-HT1A SFT",
        "#EDF4FA", BLUE, BLUE)

    # Column 2: Hard Compliance Gates
    box(2.70, 0.78, 2.10, 2.55, "2. Compliance Gates",
        "Explicit SMARTS &\nPhysicochemical Filters\n\n"
        "REACH Annex XIV/XVII\n(PFAS, Phthalates, Sn)\n"
        "+ BPA (Plastics)\n+ PAINS & CNS-BBB\n\nFail → Zero Reward",
        "#FDF2EC", VERMILION, VERMILION)

    # Column 3: 4-GNN Ensemble + REINVENT
    box(5.20, 0.78, 2.20, 2.55, "3. 4-GNN Reward",
        "2D Molecular/Polymer\nGraph Ensemble:\n"
        "GINE + GATv2 +\nMPNN + CARATE\n\n"
        "Unweighted Mean →\nREINVENT Augmented\nLikelihood Update",
        "#EAF7F2", GREEN, GREEN)

    # Column 4: Independent Holdout Audit
    box(7.80, 0.78, 2.05, 2.55, "4. Holdout Audit",
        "Excluded from All\nTraining Rewards\n\n"
        "ExtraTrees Regressor\n(seed 1337, Morgan r3\n+ MACCS + Descriptors)\n\n"
        "Evaluates In-Band &\nDetects Reward Hacking",
        "#F5EFF8", PURPLE, PURPLE)

    # Forward arrows
    arrow(2.36, 2.06, 2.64, 2.06, DARK)
    arrow(4.86, 2.06, 5.14, 2.06, DARK)
    arrow(7.46, 2.06, 7.74, 2.06, PURPLE, ls="--")

    # Policy gradient feedback loop cleanly below boxes
    arrow(6.30, 0.72, 1.22, 0.72, GREEN, rad=-0.12)
    ax.text(3.76, 0.08, "REINVENT policy-gradient update  (prior-anchored KL + duplicate memory)",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=GREEN)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / "fig_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_pipeline.pdf")


def make_learning_curves() -> None:
    uv_data = json.loads((OUT_DIR / "uvvis_learning_curve.json").read_text())
    uv_steps = np.array([int(k) for k in uv_data.keys()])
    gnn_m = np.array([uv_data[str(s)]["gnn_mean"] for s in uv_steps])
    gnn_s = np.array([uv_data[str(s)]["gnn_std"] for s in uv_steps])
    et_m = np.array([uv_data[str(s)]["et_mean"] for s in uv_steps])
    et_s = np.array([uv_data[str(s)]["et_std"] for s in uv_steps])
    uniq_m = np.array([uv_data[str(s)]["uniq_mean"] for s in uv_steps])

    # Plastics real logged RL trajectory (run_track_b.log)
    pl_steps = np.array([0, 150, 300, 450, 600])
    pl_mean_r = np.array([0.131, 0.815, 0.771, 0.856, 0.832])
    pl_high_r = np.array([0.201, 0.766, 0.711, 0.828, 0.742])

    # Pharma v2 real logged RL trajectory (run_pharma_cns_v2.log)
    ph_steps = np.array([0, 125, 250, 375, 500])
    ph_mean_r = np.array([0.348, 0.526, 0.568, 0.632, 0.722])
    ph_high_r = np.array([0.005, 0.125, 0.195, 0.414, 0.711])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)

    # Panel A: UV-Vis 3-seed snapshot learning curve
    ax = axes[0]
    ax.plot(uv_steps, gnn_m, "-o", color=BLUE, markersize=3.8, linewidth=1.6, label="4-GNN in-band")
    ax.fill_between(uv_steps, gnn_m - gnn_s, gnn_m + gnn_s, color=BLUE, alpha=0.18)
    ax.plot(uv_steps, et_m, "-s", color=VERMILION, markersize=3.6, linewidth=1.6, label="Holdout ET in-band")
    ax.fill_between(uv_steps, et_m - et_s, et_m + et_s, color=VERMILION, alpha=0.16)
    ax.plot(uv_steps, uniq_m, "--", color=GREY, linewidth=1.2, label="Uniqueness")
    ax.set_title("A. UV-Vis (3 seeds, mean ± SD)")
    ax.set_xlabel("REINVENT step")
    ax.set_ylabel("Fraction / Score")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, borderpad=0.35)

    # Panel B: Plastics RL trajectory
    ax = axes[1]
    ax.plot(pl_steps, pl_mean_r, "-o", color=GREEN, markersize=4.0, linewidth=1.6, label="Mean RL reward")
    ax.plot(pl_steps, pl_high_r, "-^", color=ORANGE, markersize=4.0, linewidth=1.5, label="High-reward frac.")
    ax.axhline(0.9997, color=BLUE, linestyle=":", linewidth=1.1, label="Final compliance (0.9997)")
    ax.set_title("B. Plastics (Tg 450–550 K)")
    ax.set_xlabel("REINVENT step")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, borderpad=0.35)

    # Panel C: CNS Pharma RL trajectory
    ax = axes[2]
    ax.plot(ph_steps, ph_mean_r, "-o", color=PURPLE, markersize=4.0, linewidth=1.6, label="Mean safety reward")
    ax.plot(ph_steps, ph_high_r, "-^", color=VERMILION, markersize=4.0, linewidth=1.5, label="High-reward frac.")
    ax.axhline(0.959, color=GREEN, linestyle=":", linewidth=1.1, label="Final hERG safe (0.959)")
    ax.set_title("C. CNS Pharma (after SFT)")
    ax.set_xlabel("REINVENT step")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.90), frameon=True, framealpha=0.92, borderpad=0.35)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT_DIR / "fig_learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_learning_curves.pdf")


def make_cross_domain_summary() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Panel A: Held-out R^2 across all 5 endpoints
    endpoints = ["UV-Vis\nλmax", "Polymer\nTg", "CNS\n5-HT1A", "5-HT2B\n(anti)", "hERG\n(anti)"]
    r2_gnn = np.array([0.922, 0.922, 0.687, 0.460, 0.617])
    r2_et = np.array([0.928, 0.818, 0.350, 0.094, 0.286])
    x = np.arange(len(endpoints))
    w = 0.36
    b1 = ax1.bar(x - w / 2, r2_gnn, width=w, color=BLUE, label="GNN Ensemble")
    b2 = ax1.bar(x + w / 2, r2_et, width=w, color=ORANGE, label="Holdout ExtraTrees")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                     ha="center", va="bottom", fontsize=6.6, color=DARK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(endpoints, fontsize=7.4)
    ax1.set_ylabel("Held-out $R^2$")
    ax1.set_ylim(0, 1.16)
    ax1.set_title("A. Surrogate $R^2$ Across 5 Endpoints")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Panel B: CNS Selectivity Trap & Recovery (including Murcko scaffold memory)
    metrics = ["5-HT1A\n(GNN)", "5-HT1A\n(ET)", "5-HT2B\nsafe", "hERG\nsafe", "CNS-BBB", "REACH/\nPAINS", "Uniq."]
    zinc = np.array([0.001, 0.244, 0.565, 0.859, 0.667, 0.939, 1.000])
    sft = np.array([0.348, 0.490, 0.177, 0.610, 0.528, 0.978, 0.999])
    rl = np.array([0.421, 0.912, 0.551, 0.959, 0.982, 0.998, 0.266])
    rl_scaf = np.array([0.445, 0.507, 0.228, 0.750, 0.788, 0.987, 0.994])
    x2 = np.arange(len(metrics))
    w2 = 0.20
    ax2.bar(x2 - 1.5 * w2, zinc, width=w2, color=SKY, label="ZINC prior")
    ax2.bar(x2 - 0.5 * w2, sft, width=w2, color=VERMILION, label="SFT (8 ep.)")
    ax2.bar(x2 + 0.5 * w2, rl, width=w2, color=GREEN, label="Safety-RL")
    ax2.bar(x2 + 1.5 * w2, rl_scaf, width=w2, color=PURPLE, label="+ Murcko mem.")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics, fontsize=7.2)
    ax2.set_ylabel("Fraction (3,000 samples)")
    ax2.set_ylim(0, 1.28)
    ax2.set_title("B. CNS Selectivity Trap & Scaffold Memory")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax2.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.92, fontsize=6.6, columnspacing=0.9, handletextpad=0.4)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT_DIR / "fig_cross_domain_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_cross_domain_summary.pdf")


def make_dft_retrosynth_figure() -> None:
    import pandas as pd

    dft_df = pd.read_csv(OUT_DIR.parent / "dft_validation" / "dft_50_results.csv")
    ok = dft_df[dft_df["converged"] == True].copy()
    lam_br = ok["brightest_nm"].to_numpy(dtype=float)
    osc_br = ok["brightest_osc"].to_numpy(dtype=float)
    s1_nm = ok["s1_nm"].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.2, 1.0]})

    # Panel A: Brightest TD-DFT transition vs oscillator strength
    ax1.axvspan(400, 450, color=SKY, alpha=0.22, label="Target band (400–450 nm)")
    ax1.axvspan(380, 400, color=SKY, alpha=0.09)
    ax1.axvspan(450, 470, color=SKY, alpha=0.09)
    ax1.axhline(0.05, color=GREY, linestyle="--", linewidth=0.9, label="Bright cutoff (f = 0.05)")

    is_hit = (lam_br >= 380) & (lam_br <= 470) & (osc_br >= 0.05)
    ax1.scatter(lam_br[~is_hit], np.clip(osc_br[~is_hit], 0, 0.55), s=24, color=BLUE, alpha=0.65,
                edgecolors="none", label=f"Other converged (n={int((~is_hit).sum())})")
    ax1.scatter(lam_br[is_hit], osc_br[is_hit], s=42, color=GREEN, edgecolors=DARK, linewidth=0.6,
                zorder=4, label=f"Confirmed bright hits (n={int(is_hit.sum())})")
    ax1.axvline(float(np.median(s1_nm)), color=VERMILION, linestyle=":", linewidth=1.3,
                label=f"Median S1 ({np.median(s1_nm):.1f} nm)")
    ax1.set_xlim(250, 680)
    ax1.set_ylim(-0.02, 0.58)
    ax1.set_xlabel("DF-B3LYP/6-31G brightest excitation λ (nm)")
    ax1.set_ylabel("Oscillator strength f (clipped at 0.55)")
    ax1.set_title("A. Ab-Initio TD-DFT Audit (n = 46)")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=6.4, handletextpad=0.4)

    # Panel B: 3-Domain Retrosynthesis Audit (SA <= 4.0 and BRICS decomposable)
    domains = ["UV-Vis\n(n=592)\nSA 2.92", "Plastics\n(n=50)\nSA 2.48", "CNS Pharma\n(n=10)\nSA 3.71"]
    sa_ok = np.array([0.988, 1.000, 0.700])
    brics_ok = np.array([0.993, 0.980, 1.000])
    x = np.arange(len(domains))
    w = 0.34
    b1 = ax2.bar(x - w / 2, sa_ok, width=w, color=BLUE, label="SA-Score ≤ 4.0")
    b2 = ax2.bar(x + w / 2, brics_ok, width=w, color=GREEN, label="BRICS decomposable")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{100 * h:.0f}%",
                     ha="center", va="bottom", fontsize=7.0, color=DARK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, fontsize=7.5)
    ax2.set_ylabel("Pass fraction")
    ax2.set_ylim(0, 1.24)
    ax2.set_title("B. Retrosynthesis & SA-Score Audit")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax2.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.92, fontsize=6.8, columnspacing=0.8)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT_DIR / "fig_dft_retrosynth.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_dft_retrosynth.pdf")


if __name__ == "__main__":
    make_pipeline_diagram()
    make_learning_curves()
    make_cross_domain_summary()
    make_dft_retrosynth_figure()
