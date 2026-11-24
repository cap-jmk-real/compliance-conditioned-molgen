#!/usr/bin/env python3
"""Build a single-page TOC graphic PDF for JCIM submission."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "toc_graphic.pdf"


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.4, 1.2, "Prior\n(SELFIES/P-SMILES)", "#4C72B0"),
        (3.2, 1.2, "REACH / BPA /\nPAINS / BBB gates", "#C44E52"),
        (6.0, 1.2, "4-GNN ensemble\nreward", "#55A868"),
        (8.2, 1.2, "Holdout ExtraTrees\naudit", "#8172B3"),
    ]
    for x, y, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.0, 1.5, boxstyle="round,pad=0.08", fc=col, ec="white", lw=0.6, alpha=0.92))
        ax.text(x + 1.0, y + 0.75, txt, ha="center", va="center", fontsize=7.2, color="white", fontweight="bold")
    for x0, x1 in [(2.45, 3.15), (5.25, 5.95), (7.45, 8.15)]:
        ax.add_patch(FancyArrowPatch((x0, 1.95), (x1, 1.95), arrowstyle="-|>", mutation_scale=12, lw=1.4, color="#333333"))
    ax.text(5.0, 3.35, "Compliance-conditioned molecular generation", ha="center", fontsize=10, fontweight="bold")
    ax.text(5.0, 0.45, "UV-Vis  •  Engineering polymers  •  CNS ligands", ha="center", fontsize=8)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()